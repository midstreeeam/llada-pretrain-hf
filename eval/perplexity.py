from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def _auto_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        capability = torch.cuda.get_device_capability(device)
        if capability and capability[0] >= 8:
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    normalized = name.lower()
    if normalized == "auto":
        return _auto_dtype(device)
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype '{name}'. Choices: {list(mapping.keys()) + ['auto']}")
    return mapping[normalized]


def load_texts_from_jsonl(path: Path, field: str, limit: Optional[int], strip_newlines: bool) -> List[str]:
    if limit is not None and limit <= 0:
        return []
    if not path.exists():
        raise FileNotFoundError(f"JSONL file '{path}' not found.")

    texts: List[str] = []
    remaining = limit
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            value = record.get(field)
            if not isinstance(value, str):
                continue
            value = value.strip()
            if strip_newlines:
                value = " ".join(value.splitlines())
            if not value:
                continue
            texts.append(value)
            if remaining is not None:
                remaining -= 1
                if remaining <= 0:
                    break
    if limit is not None and len(texts) > limit:
        texts = texts[:limit]
    return texts


def load_texts_from_dataset(
    dataset_name: str,
    split: str,
    field: str,
    limit: Optional[int],
    strip_newlines: bool,
) -> List[str]:
    dataset = load_dataset(dataset_name, split=split)
    texts: List[str] = []
    for row in dataset:
        value = row.get(field)
        if not isinstance(value, str):
            continue
        text = value.strip()
        if strip_newlines:
            text = " ".join(text.splitlines())
        if not text:
            continue
        texts.append(text)
        if limit is not None and len(texts) >= limit:
            break
    return texts


@dataclass
class SequencePerplexity:
    text_index: int
    num_tokens: int
    neg_log_likelihood: float
    per_token_nll: float
    perplexity: float
    text_preview: Optional[str] = None


@dataclass
class PerplexitySummary:
    total_sequences: int
    total_tokens: int
    total_neg_log_likelihood: float
    mean_per_token_nll: float
    perplexity: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerplexityJobConfig:
    judge_model: str
    tokenizer_path: Optional[str]
    batch_size: int = 2
    max_context_length: int = 1024
    dtype: str = "auto"
    device: Optional[str] = None
    trust_remote_code: bool = False
    store_sequence_details: bool = True
    output_path: Optional[Path] = None
    output_details_path: Optional[Path] = None


def _score_batch(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Sequence[Tuple[float, int]]:
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits
    logits = logits[:, :-1, :]
    labels = input_ids[:, 1:]
    mask = attention_mask[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs * mask

    seq_log_probs = token_log_probs.sum(dim=1)
    token_counts = mask.sum(dim=1)
    results: List[Tuple[float, int]] = []
    for log_prob, count in zip(seq_log_probs, token_counts):
        results.append((float(log_prob.item()), int(count.item())))
    return results


def run_perplexity_job(texts: Sequence[str], config: PerplexityJobConfig) -> PerplexitySummary:
    if not texts:
        raise ValueError("No texts provided for perplexity computation.")

    device_str = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    dtype = _resolve_dtype(config.dtype, device)

    tokenizer_name = config.tokenizer_path or config.judge_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, trust_remote_code=config.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        config.judge_model,
        torch_dtype=dtype,
        trust_remote_code=config.trust_remote_code,
    )
    model.to(device)
    model.eval()

    total_neg_log_likelihood = 0.0
    total_tokens = 0
    processed_sequences = 0
    details: List[SequencePerplexity] = []

    texts_list = list(texts)
    total_batches = (len(texts_list) + config.batch_size - 1) // config.batch_size

    with torch.inference_mode():
        for batch_start in tqdm(
            range(0, len(texts_list), config.batch_size),
            total=total_batches,
            desc="Scoring",
            unit="batch",
        ):
            batch_texts = texts_list[batch_start : batch_start + config.batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=config.max_context_length,
                add_special_tokens=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            if input_ids.shape[1] < 2:
                continue

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            batch_scores = _score_batch(model, input_ids, attention_mask)

            for local_idx, (log_prob, token_count) in enumerate(batch_scores):
                if token_count == 0:
                    continue
                neg_log_likelihood = -log_prob
                per_token_nll = neg_log_likelihood / token_count
                perplexity = math.exp(per_token_nll)

                total_neg_log_likelihood += neg_log_likelihood
                total_tokens += token_count
                absolute_index = batch_start + local_idx
                processed_sequences += 1

                if config.store_sequence_details:
                    preview = batch_texts[local_idx]
                    if len(preview) > 160:
                        preview = preview[:157] + "..."
                    details.append(
                        SequencePerplexity(
                            text_index=absolute_index,
                            num_tokens=token_count,
                            neg_log_likelihood=neg_log_likelihood,
                            per_token_nll=per_token_nll,
                            perplexity=perplexity,
                            text_preview=preview,
                        )
                    )

    if total_tokens == 0:
        raise RuntimeError("Judge model produced zero valid tokens; check tokenizer settings.")

    mean_per_token_nll = total_neg_log_likelihood / total_tokens
    summary = PerplexitySummary(
        total_sequences=processed_sequences,
        total_tokens=total_tokens,
        total_neg_log_likelihood=total_neg_log_likelihood,
        mean_per_token_nll=mean_per_token_nll,
        perplexity=math.exp(mean_per_token_nll),
        metadata={
            "judge_model": config.judge_model,
            "dtype": str(dtype),
            "device": str(device),
            "max_context_length": config.max_context_length,
        },
    )

    if config.output_path:
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        with config.output_path.open("w", encoding="utf-8") as fh:
            json.dump(asdict(summary), fh, indent=2, ensure_ascii=False)

    if config.store_sequence_details and config.output_details_path:
        config.output_details_path.parent.mkdir(parents=True, exist_ok=True)
        with config.output_details_path.open("w", encoding="utf-8") as fh:
            for record in details:
                fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute perplexity of text corpora with a judge model.")
    parser.add_argument("--judge-model", required=True, help="Judge model (e.g., Qwen/Qwen3-1.7B).")
    parser.add_argument("--tokenizer-path", default=None, help="Optional tokenizer override.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom model implementations.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-context-length", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-details", action="store_true", help="Skip per-sample details to save space.")
    parser.add_argument("--output-json", type=str, required=True, help="Where to store the summary JSON.")
    parser.add_argument("--output-details", type=str, default=None, help="Optional JSONL for per-sample metrics.")
    parser.add_argument("--strip-newlines", action="store_true")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-jsonl", type=str, help="JSONL file containing text.")
    source.add_argument("--dataset-name", type=str, help="Hugging Face dataset identifier.")

    parser.add_argument("--text-field", type=str, default="generated_text", help="Field name containing the text.")
    parser.add_argument("--dataset-split", type=str, default="train[:1000]", help="Dataset split (if applicable).")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def _collect_texts(args: argparse.Namespace) -> List[str]:
    if args.input_jsonl:
        return load_texts_from_jsonl(
            path=Path(args.input_jsonl),
            field=args.text_field,
            limit=args.max_samples,
            strip_newlines=args.strip_newlines,
        )
    assert args.dataset_name
    return load_texts_from_dataset(
        dataset_name=args.dataset_name,
        split=args.dataset_split,
        field=args.text_field,
        limit=args.max_samples,
        strip_newlines=args.strip_newlines,
    )


def main() -> None:
    args = _parse_args()
    texts = _collect_texts(args)
    if not texts:
        raise ValueError("No texts found for perplexity computation.")
    config = PerplexityJobConfig(
        judge_model=args.judge_model,
        tokenizer_path=args.tokenizer_path,
        batch_size=args.batch_size,
        max_context_length=args.max_context_length,
        dtype=args.dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        store_sequence_details=not args.no_details,
        output_path=Path(args.output_json),
        output_details_path=Path(args.output_details) if args.output_details else None,
    )
    summary = run_perplexity_job(texts=texts, config=config)
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
