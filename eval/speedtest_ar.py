#!/usr/bin/env python3
"""Autoregressive throughput benchmark driven by TinyLlama Stories prompts."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    normalized = name.lower()
    if normalized == "auto":
        if device.type == "cuda":
            major, _ = torch.cuda.get_device_capability(device)
            return torch.bfloat16 if major >= 8 else torch.float16
        return torch.float32
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype '{name}'. Choices: {list(mapping) + ['auto']}")
    return mapping[normalized]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark autoregressive CausalLM throughput using TinyLlama Stories prompts."
    )
    parser.add_argument(
        "--model-name",
        default="rzzhan/tiny-llama-stories-42m",
        help="HF repo ID or path for AutoModelForCausalLM (default: rzzhan/tiny-llama-stories-42m).",
    )
    parser.add_argument("--model-revision", default=None, help="Optional model revision/hash.")
    parser.add_argument("--tokenizer-name", default=None, help="Optional tokenizer override path.")
    parser.add_argument("--device", default=None, help="torch device (defaults to cuda if available).")
    parser.add_argument("--dtype", default="auto", help="Computation dtype (float16/bfloat16/float32/auto).")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom model code during load.")

    parser.add_argument("--num-prompts", type=int, default=256, help="Number of dataset prompts to benchmark.")
    parser.add_argument("--batch-size", type=int, default=8, help="Prompts per generation batch.")
    parser.add_argument("--prompt-max-tokens", type=int, default=512, help="Prompt truncation length.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Tokens to generate per prompt.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--do-sample", action="store_true", help="Enable multinomial sampling.")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k filtering (optional).")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p filtering (optional).")
    parser.add_argument("--no-kv-cache", action="store_true", help="Disable KV cache during generation.")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="Optional repetition penalty.")

    parser.add_argument(
        "--dataset-name",
        default="roneneldan/TinyStories",
        help="HF dataset providing prompts (default: roneneldan/TinyStories).",
    )
    parser.add_argument("--dataset-split", default="train", help="Dataset split to sample from.")
    parser.add_argument("--dataset-text-field", default="text", help="Name of the text field containing prompts.")
    parser.add_argument(
        "--dataset-shuffle",
        action="store_true",
        help="Shuffle the dataset before sampling prompts (uses --dataset-seed).",
    )
    parser.add_argument("--dataset-seed", type=int, default=0, help="Seed used when shuffling prompts.")

    return parser.parse_args()


def load_prompts(args: argparse.Namespace) -> List[str]:
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    if args.dataset_shuffle:
        dataset = dataset.shuffle(seed=args.dataset_seed)

    prompts: List[str] = []
    for row in dataset:
        value = row.get(args.dataset_text_field)
        if value is None:
            continue
        text = value if isinstance(value, str) else str(value)
        text = text.strip()
        if not text:
            continue
        prompts.append(text)
        if len(prompts) >= args.num_prompts:
            break

    if len(prompts) < args.num_prompts:
        raise ValueError(
            f"Only collected {len(prompts)} prompts from {args.dataset_name}. "
            "Consider reducing --num-prompts or verifying --dataset-text-field."
        )
    return prompts


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args)

    tokenizer_name = args.tokenizer_name or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, use_fast=True, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define or be assigned a pad_token for batching.")
    tokenizer.padding_side = "left"

    base_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(base_device)
    model_dtype = _resolve_dtype(args.dtype, device)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=model_dtype,
        trust_remote_code=args.trust_remote_code,
        revision=args.model_revision,
    )
    model.to(device).eval()

    total_tokens = args.num_prompts * args.max_new_tokens
    num_batches = (args.num_prompts + args.batch_size - 1) // args.batch_size

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": args.do_sample,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "use_cache": not args.no_kv_cache,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = args.repetition_penalty

    with torch.inference_mode():
        for start_idx in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[start_idx : start_idx + args.batch_size]
            encoded = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.prompt_max_tokens,
            )
            batch_ids = encoded["input_ids"].to(device, non_blocking=True)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device, non_blocking=True)
            model.generate(
                input_ids=batch_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

    elapsed = time.perf_counter() - start
    tokens_per_second = total_tokens / elapsed if elapsed > 0 else float("inf")
    peak_alloc = peak_reserved = None
    if device.type == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)

    print(
        f"Completed {total_tokens} tokens across {num_batches} batches from {args.dataset_name} in {elapsed:.2f}s."
    )
    print(f"Throughput: {tokens_per_second:.2f} tokens/s (max_new_tokens only).")
    if peak_alloc is not None:
        print(
            "Peak GPU memory (allocated/reserved): "
            f"{peak_alloc / (1024 ** 2):.2f} / {peak_reserved / (1024 ** 2):.2f} MiB; "
            f"use_cache={not args.no_kv_cache}."
        )


if __name__ == "__main__":
    main()
