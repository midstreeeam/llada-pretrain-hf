#!/usr/bin/env python3
"""Lightweight throughput benchmark for LLaDA diffusion generation."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generation import run_diffusion_generation  # noqa: E402
from llada.modeling_llada import LLaDAModelLM  # noqa: E402


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
        raise ValueError(f"Unsupported dtype '{name}'. Choices: {list(mapping)} + ['auto']")
    return mapping[normalized]


def _chunk(sequence: Sequence[int], size: int):
    for start in range(0, len(sequence), size):
        yield slice(start, start + size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark LLaDA diffusion generation throughput.")
    parser.add_argument("--model-path", required=True, help="Path to the LLaDA checkpoint.")
    parser.add_argument("--tokenizer-path", default=None, help="Optional tokenizer override.")
    parser.add_argument("--device", default=None, help="torch device (defaults to cuda if available).")
    parser.add_argument("--dtype", default="float16", help="Computation dtype (float16/bfloat16/float32/auto).")

    parser.add_argument("--num-prompts", type=int, default=256, help="Total prompts to benchmark.")
    parser.add_argument("--batch-size", type=int, default=4, help="Prompts per batch.")
    parser.add_argument("--prompt-text", default="Once upon a time, ", help="Seed text for prompts.")
    parser.add_argument("--prompt-max-tokens", type=int, default=256, help="Prompt truncation length.")
    parser.add_argument("--append-eos-to-prompt", action="store_true", help="Append EOS before sampling.")

    parser.add_argument("--max-new-tokens", type=int, default=128, help="Tokens to generate per prompt.")
    parser.add_argument("--diffusion-steps", type=int, default=32, help="Total diffusion iterations.")
    parser.add_argument("--block-size", type=int, default=8, help="Tokens filled per block.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--do-sample", action="store_true", help="Enable multinomial sampling.")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k filtering (optional).")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p filtering (optional).")
    parser.add_argument("--decode-top-k", type=int, default=0, help="Decode top-k restriction.")
    parser.add_argument(
        "--remask-strategy",
        choices=["low_confidence", "random"],
        default="low_confidence",
        help="Diffusion remask strategy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = [f"{args.prompt_text}{i}" for i in range(args.num_prompts)]
    tokenizer_name = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise ValueError("Tokenizer must define mask_token_id for diffusion generation.")

    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=args.prompt_max_tokens,
        return_tensors="pt",
        add_special_tokens=False,
    )

    base_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(base_device)
    model_dtype = _resolve_dtype(args.dtype, device)

    model = LLaDAModelLM.from_pretrained(args.model_path, torch_dtype=model_dtype)
    model.to(device).eval()

    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    total_tokens = args.num_prompts * args.max_new_tokens
    num_batches = (args.num_prompts + args.batch_size - 1) // args.batch_size

    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    start = time.perf_counter()

    with torch.inference_mode():
        for slc in _chunk(range(args.num_prompts), args.batch_size):
            batch_ids = input_ids[slc].to(device)
            batch_mask = attention_mask[slc].to(device)
            if args.append_eos_to_prompt:
                eos_id = tokenizer.eos_token_id
                if eos_id is not None:
                    eos_column = torch.full((batch_ids.shape[0], 1), eos_id, dtype=batch_ids.dtype, device=device)
                    batch_ids = torch.cat([batch_ids, eos_column], dim=1)
                    eos_mask = torch.ones((batch_mask.shape[0], 1), dtype=batch_mask.dtype, device=device)
                    batch_mask = torch.cat([batch_mask, eos_mask], dim=1)

            run_diffusion_generation(
                model=model,
                input_ids=batch_ids,
                attention_mask=batch_mask,
                mask_token_id=mask_token_id,
                max_new_tokens=args.max_new_tokens,
                steps=args.diffusion_steps,
                block_size=args.block_size,
                temperature=args.temperature,
                do_sample=args.do_sample,
                top_k=args.top_k,
                top_p=args.top_p,
                decode_top_k=args.decode_top_k,
                remask_strategy=args.remask_strategy,
                debug=False,
                tokenizer=None,
            )

    elapsed = time.perf_counter() - start
    tokens_per_second = total_tokens / elapsed if elapsed > 0 else float("inf")
    peak_mem = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None

    print(f"Completed {total_tokens} tokens across {num_batches} batches in {elapsed:.2f}s")
    print(f"Throughput: {tokens_per_second:.2f} tokens/s")
    if peak_mem is not None:
        print(f"Peak GPU memory: {peak_mem / (1024 ** 2):.2f} MiB")


if __name__ == "__main__":
    main()
