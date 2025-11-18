from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from datasets import load_dataset
from tqdm import tqdm

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoTokenizer

from generation import run_diffusion_generation
from llada.modeling_llada import LLaDAModelLM
from eval.perplexity import (
    PerplexityJobConfig,
    load_texts_from_dataset,
    load_texts_from_jsonl,
    run_perplexity_job,
)


def _auto_dtype_preference(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        capability = torch.cuda.get_device_capability(device)
        if capability and capability[0] >= 8:
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    normalized = name.lower()
    if normalized == "auto":
        return _auto_dtype_preference(device)
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype '{name}'. Choices: {list(mapping.keys()) + ['auto']}")
    return mapping[normalized]


def _normalize_prompt(text: str, strip_newlines: bool) -> str:
    text = text.strip()
    if strip_newlines:
        text = " ".join(text.splitlines())
    return text


def _smart_truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    # Cut to limit
    text = text[:max_chars]
    # Find last space to avoid splitting words
    last_space = text.rfind(" ")
    if last_space != -1:
        text = text[:last_space]
    return text.strip()


def load_tinystories_prompts(
    limit: int,
    split: str,
    min_chars: int,
    max_chars: Optional[int],
    strip_newlines: bool,
    seed: int,
    shuffle: bool,
) -> List[str]:
    dataset = load_dataset("roneneldan/TinyStories", split=split)
    if shuffle:
        dataset = dataset.shuffle(seed=seed)

    prompts: List[str] = []
    for row in dataset:
        text = row.get("text") or ""
        text = _normalize_prompt(text, strip_newlines=strip_newlines)
        if len(text) < min_chars:
            continue
        if max_chars is not None:
            text = _smart_truncate(text, max_chars)
        if text:
            prompts.append(text)
        if len(prompts) >= limit:
            break
    return prompts


def load_prompts_from_jsonl(path: Path, field: str, limit: Optional[int], strip_newlines: bool) -> List[str]:
    if limit is not None and limit <= 0:
        return []
    prompts: List[str] = []
    if not path.exists():
        raise FileNotFoundError(f"Prompt file '{path}' not found")
    remaining = limit if limit is not None else None
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            value = record.get(field)
            if not isinstance(value, str):
                continue
            value = _normalize_prompt(value, strip_newlines=strip_newlines)
            if value:
                prompts.append(value)
            if remaining is not None:
                remaining -= 1
                if remaining <= 0:
                    break
    if limit is not None and len(prompts) > limit:
        return prompts[:limit]
    return prompts


@dataclass
class GenerationJobConfig:
    model_path: str
    tokenizer_path: Optional[str]
    output_path: Path
    batch_size: int = 4
    prompt_max_tokens: int = 256
    append_eos_to_prompt: bool = False
    max_new_tokens: int = 128
    diffusion_steps: int = 32
    block_size: int = 8
    temperature: float = 1.0
    do_sample: bool = True
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    decode_top_k: int = 0
    remask_strategy: str = "low_confidence"
    device: Optional[str] = None
    dtype: str = "float16"
    metadata: Dict[str, Any] = field(default_factory=dict)


def generate_eval_corpus(prompts: Sequence[str], config: GenerationJobConfig) -> Dict[str, Any]:
    if not prompts:
        raise ValueError("At least one prompt is required to run generation.")

    if config.max_new_tokens % config.block_size != 0:
        raise ValueError("max_new_tokens must be divisible by block_size.")

    num_blocks = config.max_new_tokens // config.block_size
    if config.diffusion_steps % num_blocks != 0:
        raise ValueError("num_diffusion_steps must be divisible by (max_new_tokens / block_size).")

    device_str = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    torch_dtype = _resolve_dtype(config.dtype, device)

    tokenizer_path = config.tokenizer_path or config.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise ValueError("Tokenizer used for generation must define a mask_token_id.")
    model = LLaDAModelLM.from_pretrained(config.model_path, torch_dtype=torch_dtype)
    model.to(device)
    model.eval()

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_list = list(prompts)
    total_batches = (len(prompt_list) + config.batch_size - 1) // config.batch_size
    total_records = 0

    with config.output_path.open("w", encoding="utf-8") as writer, torch.inference_mode():
        for batch_index, start in enumerate(
            tqdm(range(0, len(prompt_list), config.batch_size), total=total_batches, desc="Generating", unit="batch")
        ):
            batch_prompts = prompt_list[start : start + config.batch_size]
            encoded = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=config.prompt_max_tokens,
                return_tensors="pt",
                add_special_tokens=False,
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)

            if config.append_eos_to_prompt:
                eos_id = tokenizer.eos_token_id
                if eos_id is not None:
                    eos_column = torch.full((input_ids.shape[0], 1), eos_id, dtype=input_ids.dtype)
                    input_ids = torch.cat([input_ids, eos_column], dim=1)
                    eos_mask = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype)
                    attention_mask = torch.cat([attention_mask, eos_mask], dim=1)

            prompt_tokens_cpu = input_ids.clone()
            prompt_attention_cpu = attention_mask.clone()

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            generated_sequences, fill_steps = run_diffusion_generation(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                mask_token_id=mask_token_id,
                max_new_tokens=config.max_new_tokens,
                steps=config.diffusion_steps,
                block_size=config.block_size,
                temperature=config.temperature,
                do_sample=config.do_sample,
                top_k=config.top_k,
                top_p=config.top_p,
                decode_top_k=config.decode_top_k,
                remask_strategy=config.remask_strategy,
                debug=False,
                tokenizer=None,
            )

            prompt_window = input_ids.shape[1]
            generated_sequences = generated_sequences.cpu()
            fill_steps = fill_steps.cpu()

            for idx, prompt_text in enumerate(batch_prompts):
                attention_len = int(prompt_attention_cpu[idx].sum().item())
                prompt_ids = prompt_tokens_cpu[idx, :attention_len].tolist()
                generated_ids = generated_sequences[idx, prompt_window : prompt_window + config.max_new_tokens].tolist()
                generated_step_vals = fill_steps[idx, prompt_window : prompt_window + config.max_new_tokens].tolist()

                decoded_output = tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                ).strip()

                record = {
                    "prompt_index": total_records,
                    "prompt": prompt_text,
                    "prompt_token_ids": prompt_ids,
                    "generated_token_ids": generated_ids,
                    "fill_steps": generated_step_vals,
                    "generated_text": decoded_output,
                    "metadata": {
                        "mask_token_id": mask_token_id,
                        "batch_index": batch_index,
                        "student_checkpoint": config.model_path,
                        **config.metadata,
                    },
                }
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_records += 1

    return {
        "num_prompts": len(prompts),
        "records_written": total_records,
        "output_path": str(config.output_path),
        "device": str(device),
        "dtype": str(torch_dtype),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="End-to-end perplexity evaluation pipeline for LLaDA checkpoints.")
    parser.add_argument("--student-checkpoint", required=True, help="Path to the trained LLaDA checkpoint.")
    parser.add_argument("--student-tokenizer", default=None, help="Optional tokenizer override for the student model.")

    parser.add_argument("--judge-model", required=True, help="Judge model for scoring (e.g., Qwen/Qwen3-1.7B).")
    parser.add_argument("--judge-tokenizer", default=None, help="Optional tokenizer override for the judge.")
    parser.add_argument("--judge-trust-remote-code", action="store_true", help="Allow custom judge model code.")

    parser.add_argument("--output-dir", required=True, help="Directory to store evaluation artifacts.")
    parser.add_argument("--generation-jsonl", default=None, help="Path for generation outputs (JSONL).")
    parser.add_argument("--skip-generation", action="store_true", help="Reuse an existing generation JSONL file.")

    parser.add_argument("--generation-num-prompts", type=int, default=128)
    parser.add_argument("--generation-prompt-source", choices=["tinystories"], default="tinystories")
    parser.add_argument("--generation-prompt-split", default="train[:10000]")
    parser.add_argument("--generation-prompt-min-chars", type=int, default=32)
    parser.add_argument("--generation-prompt-max-chars", type=int, default=256)
    parser.add_argument("--generation-strip-newlines", action="store_true")
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument("--generation-shuffle", action="store_true")
    parser.add_argument("--generation-extra-prompts-jsonl", default=None, help="Optional JSONL with custom prompts.")
    parser.add_argument("--generation-extra-prompts-field", default="prompt")
    parser.add_argument("--generation-extra-prompts-text", nargs="*", default=None)

    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--generation-prompt-max-tokens", type=int, default=256)
    parser.add_argument("--generation-append-eos", action="store_true")
    parser.add_argument("--generation-max-new-tokens", type=int, default=128)
    parser.add_argument("--generation-diffusion-steps", type=int, default=32)
    parser.add_argument("--generation-block-size", type=int, default=8)
    parser.add_argument("--generation-temperature", type=float, default=1.0)
    parser.add_argument("--generation-top-k", type=int, default=None)
    parser.add_argument("--generation-top-p", type=float, default=None)
    parser.add_argument("--generation-decode-top-k", type=int, default=0)
    parser.add_argument(
        "--generation-remask-strategy",
        default="low_confidence",
        choices=["low_confidence", "random"],
    )
    parser.add_argument("--generation-device", default=None)
    parser.add_argument("--generation-dtype", default="float16")

    parser.add_argument("--judge-batch-size", type=int, default=8)
    parser.add_argument("--judge-max-context", type=int, default=1024)
    parser.add_argument("--judge-device", default=None)
    parser.add_argument("--judge-dtype", default="auto")

    parser.add_argument("--reference-dataset", default="roneneldan/TinyStories")
    parser.add_argument("--reference-split", default="train[:2000]")
    parser.add_argument("--reference-text-field", default="text")
    parser.add_argument("--reference-max-samples", type=int, default=1000)
    parser.add_argument("--reference-strip-newlines", action="store_true")

    parser.add_argument("--generated-text-field", default="generated_text")
    parser.add_argument("--generated-max-samples", type=int, default=None)

    return parser

def _collect_prompts(args: argparse.Namespace) -> List[str]:
    prompts: List[str] = []
    if args.generation_prompt_source == "tinystories":
        prompts.extend(
            load_tinystories_prompts(
                limit=args.generation_num_prompts,
                split=args.generation_prompt_split,
                min_chars=args.generation_prompt_min_chars,
                max_chars=args.generation_prompt_max_chars,
                strip_newlines=args.generation_strip_newlines,
                seed=args.generation_seed,
                shuffle=args.generation_shuffle,
            )
        )

    if args.generation_extra_prompts_jsonl:
        remaining = None
        if args.generation_num_prompts:
            remaining = max(args.generation_num_prompts - len(prompts), 0)
        prompts.extend(
            load_prompts_from_jsonl(
                path=Path(args.generation_extra_prompts_jsonl),
                field=args.generation_extra_prompts_field,
                limit=remaining,
                strip_newlines=args.generation_strip_newlines,
            )
        )

    if args.generation_extra_prompts_text:
        prompts.extend(
            [
                _normalize_prompt(p, strip_newlines=args.generation_strip_newlines)
                for p in args.generation_extra_prompts_text
            ]
        )

    prompts = [p for p in prompts if p]
    if args.generation_num_prompts and len(prompts) > args.generation_num_prompts:
        prompts = prompts[: args.generation_num_prompts]
    if not prompts:
        raise ValueError("Prompt collection produced zero prompts.")
    return prompts


def _prepare_output_paths(args: argparse.Namespace, output_dir: Path) -> Dict[str, Path]:
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    generation_jsonl = (
        Path(args.generation_jsonl) if args.generation_jsonl else artifacts_dir / "model_generations.jsonl"
    )
    model_ppl_json = artifacts_dir / "model_generations_perplexity.json"
    model_ppl_details = artifacts_dir / "model_generations_perplexity.jsonl"
    reference_ppl_json = artifacts_dir / "reference_perplexity.json"
    reference_ppl_details = artifacts_dir / "reference_perplexity.jsonl"
    summary_json = output_dir / "perplexity_summary.json"

    return {
        "run_root": output_dir,
        "artifacts_dir": artifacts_dir,
        "generation_jsonl": generation_jsonl,
        "model_ppl_json": model_ppl_json,
        "model_ppl_details": model_ppl_details,
        "reference_ppl_json": reference_ppl_json,
        "reference_ppl_details": reference_ppl_details,
        "summary_json": summary_json,
    }


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    run_tag = f"gen_{args.generation_max_new_tokens}_{args.generation_diffusion_steps}_{args.generation_block_size}"
    run_root = Path(args.output_dir) / run_tag
    paths = _prepare_output_paths(args, run_root)

    generation_metadata = None
    if args.skip_generation:
        if not paths["generation_jsonl"].exists():
            raise FileNotFoundError(f"--skip-generation specified but {paths['generation_jsonl']} does not exist.")
    else:
        prompts = _collect_prompts(args)
        gen_config = GenerationJobConfig(
            model_path=args.student_checkpoint,
            tokenizer_path=args.student_tokenizer,
            output_path=paths["generation_jsonl"],
            batch_size=args.generation_batch_size,
            prompt_max_tokens=args.generation_prompt_max_tokens,
            append_eos_to_prompt=args.generation_append_eos,
            max_new_tokens=args.generation_max_new_tokens,
            diffusion_steps=args.generation_diffusion_steps,
            block_size=args.generation_block_size,
            temperature=args.generation_temperature,
            top_k=args.generation_top_k,
            top_p=args.generation_top_p,
            decode_top_k=args.generation_decode_top_k,
            remask_strategy=args.generation_remask_strategy,
            device=args.generation_device,
            dtype=args.generation_dtype,
            metadata={
                "prompt_source": args.generation_prompt_source,
                "prompt_split": args.generation_prompt_split,
                "num_prompts": len(prompts),
            },
        )
        generation_metadata = generate_eval_corpus(prompts, gen_config)

    generated_texts = load_texts_from_jsonl(
        path=paths["generation_jsonl"],
        field=args.generated_text_field,
        limit=args.generated_max_samples,
        strip_newlines=False,
    )
    if not generated_texts:
        raise RuntimeError("No generated texts found for perplexity scoring.")

    model_ppl_config = PerplexityJobConfig(
        judge_model=args.judge_model,
        tokenizer_path=args.judge_tokenizer,
        batch_size=args.judge_batch_size,
        max_context_length=args.judge_max_context,
        dtype=args.judge_dtype,
        device=args.judge_device,
        trust_remote_code=args.judge_trust_remote_code,
        store_sequence_details=True,
        output_path=paths["model_ppl_json"],
        output_details_path=paths["model_ppl_details"],
    )
    model_ppl_summary = run_perplexity_job(generated_texts, model_ppl_config)

    reference_texts = load_texts_from_dataset(
        dataset_name=args.reference_dataset,
        split=args.reference_split,
        field=args.reference_text_field,
        limit=args.reference_max_samples,
        strip_newlines=args.reference_strip_newlines,
    )
    reference_ppl_config = PerplexityJobConfig(
        judge_model=args.judge_model,
        tokenizer_path=args.judge_tokenizer,
        batch_size=args.judge_batch_size,
        max_context_length=args.judge_max_context,
        dtype=args.judge_dtype,
        device=args.judge_device,
        trust_remote_code=args.judge_trust_remote_code,
        store_sequence_details=True,
        output_path=paths["reference_ppl_json"],
        output_details_path=paths["reference_ppl_details"],
    )
    reference_ppl_summary = run_perplexity_job(reference_texts, reference_ppl_config)

    summary_payload = {
        "student_checkpoint": args.student_checkpoint,
        "student_tokenizer": args.student_tokenizer,
        "judge_model": args.judge_model,
        "generation_tag": run_tag,
        "timestamp": datetime.utcnow().isoformat(),
        "artifacts": {
            "generation_jsonl": str(paths["generation_jsonl"]),
            "model_ppl_summary": str(paths["model_ppl_json"]),
            "model_ppl_details": str(paths["model_ppl_details"]),
            "reference_ppl_summary": str(paths["reference_ppl_json"]),
            "reference_ppl_details": str(paths["reference_ppl_details"]),
        },
        "generation_metadata": generation_metadata,
        "model_perplexity": asdict(model_ppl_summary),
        "reference_perplexity": asdict(reference_ppl_summary),
    }

    paths["summary_json"].parent.mkdir(parents=True, exist_ok=True)
    with paths["summary_json"].open("w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, indent=2, ensure_ascii=False)

    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
