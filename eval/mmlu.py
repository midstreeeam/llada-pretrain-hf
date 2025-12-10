"""Lightweight MMLU evaluation for LLaDA checkpoints.

Usage:
    PYTHONPATH=$(pwd) python eval/mmlu.py \
        --model-path output/llada_135m_bb_p2/checkpoint-165000 \
        --tokenizer-path answerdotai/ModernBERT-base
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoConfig, AutoTokenizer

from llada.configuration_llada import LLaDAConfig
from llada.modeling_llada import LLaDAModelLM

AutoConfig.register("llada", LLaDAConfig)

CHOICE_LETTERS = ["A", "B", "C", "D"]


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    normalized = name.lower()
    if normalized == "auto":
        if device.type == "cuda":
            capability = torch.cuda.get_device_capability(device)
            if capability and capability[0] >= 8:
                return torch.bfloat16
            return torch.float16
        return torch.float32
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
        raise ValueError(f"Unsupported dtype '{name}'. Choices: {sorted(mapping)} + ['auto']")
    return mapping[normalized]


def _format_example(question: str, choices: Sequence[str], answer_letter: str | None = None) -> str:
    lines = [
        f"Question: {question}",
        *(f"{CHOICE_LETTERS[i]}) {choice}" for i, choice in enumerate(choices)),
    ]
    if answer_letter is None:
        lines.append("Answer: ")
    else:
        lines.append(f"Answer: {answer_letter}")
    return "\n".join(lines)


def _build_shot_prefix(dev_split, subjects: set[str] | None, num_shots: int) -> Dict[str, str]:
    shots: Dict[str, List[str]] = defaultdict(list)
    for row in dev_split:
        subject = row["subject"]
        if subjects and subject not in subjects:
            continue
        if len(shots[subject]) >= num_shots:
            continue
        answer_letter = CHOICE_LETTERS[int(row["answer"])]
        shots[subject].append(_format_example(row["question"], row["choices"], answer_letter))
    return {subj: "\n\n".join(examples) for subj, examples in shots.items()}


def _truncate_tokens(tokens: List[int], max_length: int) -> List[int]:
    if len(tokens) <= max_length:
        return tokens
    return tokens[-max_length:]


def _build_masked_input(
    base_prompt: str,
    num_masks: int,
    tokenizer: AutoTokenizer,
    max_length: int,
    bos_token_id: int | None,
    eos_token_id: int | None,
) -> Tuple[List[int], List[int], List[int]]:
    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer must define mask_token_id to run MMLU evaluation.")

    base_ids = tokenizer.encode(base_prompt, add_special_tokens=False)
    extra_tokens = num_masks
    if bos_token_id is not None:
        extra_tokens += 1
    if eos_token_id is not None:
        extra_tokens += 1

    if extra_tokens >= max_length:
        raise ValueError("max_length too small to fit BOS/EOS and mask tokens.")

    max_prompt_tokens = max_length - extra_tokens
    base_ids = _truncate_tokens(base_ids, max_prompt_tokens)

    input_ids: List[int] = []
    if bos_token_id is not None:
        input_ids.append(bos_token_id)
    input_ids.extend(base_ids)
    mask_positions = list(range(len(input_ids), len(input_ids) + num_masks))
    input_ids.extend([tokenizer.mask_token_id] * num_masks)
    if eos_token_id is not None:
        input_ids.append(eos_token_id)

    attention_mask = [1] * len(input_ids)
    return input_ids, attention_mask, mask_positions


def score_example(
    model: LLaDAModelLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    base_prompt: str,
    option_token_ids: List[List[int]],
    max_length: int,
) -> Tuple[int, List[float]]:
    """Return predicted answer index and per-option scores."""
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    option_inputs = []
    option_masks = []
    option_positions = []
    for option_tokens in option_token_ids:
        input_ids, attention_mask, mask_positions = _build_masked_input(
            base_prompt,
            num_masks=len(option_tokens),
            tokenizer=tokenizer,
            max_length=max_length,
            bos_token_id=bos_id,
            eos_token_id=eos_id,
        )
        option_inputs.append(torch.tensor(input_ids, dtype=torch.long))
        option_masks.append(torch.tensor(attention_mask, dtype=torch.long))
        option_positions.append(mask_positions)

    batch_input = torch.nn.utils.rnn.pad_sequence(option_inputs, batch_first=True, padding_value=tokenizer.pad_token_id)
    batch_mask = torch.nn.utils.rnn.pad_sequence(option_masks, batch_first=True, padding_value=0)

    with torch.inference_mode():
        outputs = model(input_ids=batch_input.to(device), attention_mask=batch_mask.to(device))
        log_probs = outputs.logits.log_softmax(-1).cpu()

    scores: List[float] = []
    for idx, target_tokens in enumerate(option_token_ids):
        positions = option_positions[idx]
        if len(positions) != len(target_tokens):
            raise ValueError("Mismatch between mask positions and target tokens.")
        token_scores = [
            float(log_probs[idx, pos, token_id].item()) for pos, token_id in zip(positions, target_tokens)
        ]
        scores.append(sum(token_scores) / max(1, len(token_scores)))

    pred_index = int(torch.tensor(scores).argmax().item())
    return pred_index, scores


def run_eval(args: argparse.Namespace) -> Dict[str, float]:
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    torch_dtype = _resolve_dtype(args.dtype, device)

    tokenizer_path = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = 50279
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = 50285
    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer missing mask_token_id; cannot score MMLU.")

    config = LLaDAConfig.from_pretrained(args.model_path)
    config.mask_token_id = tokenizer.mask_token_id
    model = LLaDAModelLM.from_pretrained(args.model_path, torch_dtype=torch_dtype, config=config)
    model.to(device)
    model.eval()

    subjects_filter = set(args.subjects) if args.subjects else None

    dev_split = load_dataset("cais/mmlu", "all", split="dev")
    shot_prefix = _build_shot_prefix(dev_split, subjects_filter, args.shots)

    eval_split = load_dataset("cais/mmlu", "all", split=args.split)

    total = 0
    overall_correct = 0
    subject_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})

    option_token_ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in CHOICE_LETTERS]
    max_examples = args.max_examples if args.max_examples and args.max_examples > 0 else None

    progress_total = len(eval_split) if max_examples is None else min(len(eval_split), max_examples)
    for row in tqdm(eval_split, total=progress_total, desc="Evaluating"):
        subject = row["subject"]
        if subjects_filter and subject not in subjects_filter:
            continue
        if max_examples is not None and total >= max_examples:
            break

        prefix = shot_prefix.get(subject, "").strip()
        base_prompt = _format_example(row["question"], row["choices"], None)
        if prefix:
            base_prompt = prefix + "\n\n" + base_prompt

        pred_idx, _ = score_example(
            model=model,
            tokenizer=tokenizer,
            device=device,
            base_prompt=base_prompt,
            option_token_ids=option_token_ids,
            max_length=args.max_length,
        )

        answer_idx = int(row["answer"])
        is_correct = int(pred_idx == answer_idx)

        total += 1
        overall_correct += is_correct
        subject_stats[subject]["correct"] += is_correct
        subject_stats[subject]["total"] += 1

    subject_acc = {
        subj: stats["correct"] / stats["total"] if stats["total"] else 0.0 for subj, stats in subject_stats.items()
    }
    macro_acc = sum(subject_acc.values()) / len(subject_acc) if subject_acc else 0.0
    overall_acc = overall_correct / total if total else 0.0

    results = {
        "overall_accuracy": overall_acc,
        "macro_accuracy": macro_acc,
        "total_questions": total,
        "shots": args.shots,
        "split": args.split,
        "model_path": args.model_path,
        "tokenizer_path": tokenizer_path,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "subject_accuracies": subject_acc,
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LLaDA checkpoints on MMLU.")
    parser.add_argument("--model-path", required=True, help="Checkpoint to evaluate.")
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Tokenizer to use (defaults to model path).",
    )
    parser.add_argument("--split", default="test", choices=["dev", "test"], help="MMLU split to score.")
    parser.add_argument("--shots", type=int, default=5, help="Number of few-shot exemplars per subject.")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit number of evaluation questions.")
    parser.add_argument("--dtype", default="auto", help="Model dtype: auto|float32|float16|bfloat16")
    parser.add_argument("--device", default=None, help="Device to run on (default: cuda if available).")
    parser.add_argument("--subjects", nargs="*", default=None, help="Optional subset of subjects to evaluate.")
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write results JSON (defaults under eval/runs/).",
    )

    args = parser.parse_args()
    results = run_eval(args)

    output_path = args.output_json
    if output_path is None:
        run_dir = REPO_ROOT / "eval" / "runs"
        run_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = run_dir / f"mmlu_{stamp}.json"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    overall = results["overall_accuracy"]
    macro = results["macro_accuracy"]
    print(f"Saved results to {output_path}")
    print(f"Overall accuracy: {overall:.4f}")
    print(f"Macro accuracy:   {macro:.4f}")


if __name__ == "__main__":
    main()
