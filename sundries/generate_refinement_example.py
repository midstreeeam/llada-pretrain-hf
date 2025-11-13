#!/usr/bin/env python3
"""
Run one-block diffusion-style generation from a prompt and print refinement
snapshots at chosen steps (e.g., 4, 8, 16, 32, 64).

Example usage (uses trained 40M checkpoint):
  PYTHONPATH=$(pwd) python3 sundries/generate_refinement_example.py \
    --config model_config/llada_40m.json \
    --tokenizer answerdotai/ModernBERT-base \
    --checkpoint output/llada_40m_dl/checkpoint-463694 \
    --prompt "one day," \
    --max-new 64 \
    --steps 64 \
    --report-steps 4,8,16,32,64
"""

from __future__ import annotations

import argparse
import os
from typing import List

import torch
from transformers import AutoTokenizer

from llada.configuration_llada import LLaDAConfig
from llada.modeling_llada import LLaDAModelLM
from generation import run_diffusion_generation


def _decode_with_masks(tokenizer, ids: List[int]) -> str:
    # Keep special tokens like [MASK] visible
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
    return text.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=False, default="model_config/llada_40m.json")
    ap.add_argument("--tokenizer", type=str, required=False, default="answerdotai/ModernBERT-base")
    ap.add_argument("--checkpoint", type=str, required=False, default="output/llada_40m_dl/checkpoint-463694", help="Optional model checkpoint dir")
    ap.add_argument("--prompt", type=str, required=False, default="one day,")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--report-steps", type=str, default="4,8,16,32,64")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--sample", action="store_true", help="Use sampling instead of argmax")
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=0.0)
    ap.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["snapshot", "separate", "both"],
        help=(
            "Printing mode: \n"
            "  - snapshot: run once with --steps and show masked snapshots at --report-steps;\n"
            "  - separate: run separate full generations for each value in --report-steps;\n"
            "  - both: do both (default)."
        ),
    )
    args = ap.parse_args()

    report_steps = [int(s) for s in args.report_steps.split(",") if s.strip()]
    steps = args.steps
    if any(s > steps for s in report_steps):
        raise ValueError("All report steps must be <= --steps")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer must have a [MASK] token."
                         " Ensure your tokenizer_config.json defines 'mask_token'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if args.checkpoint and os.path.isdir(args.checkpoint):
        model = LLaDAModelLM.from_pretrained(args.checkpoint)
    else:
        cfg = LLaDAConfig.from_pretrained(args.config)
        model = LLaDAModelLM(cfg, init_params=True)
    model.to(device)
    model.eval()

    # Prepare prompt
    enc = tokenizer(
        [args.prompt],
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # One block generation: set block_size == max_new
    block_size = args.max_new
    do_sample = bool(args.sample)
    top_k = args.top_k if args.top_k and args.top_k > 0 else None
    top_p = args.top_p if args.top_p and args.top_p > 0 else None

    print("Prompt:", args.prompt)

    if args.mode in ("snapshot", "both"):
        with torch.inference_mode():
            full_seq, fill_steps = run_diffusion_generation(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                mask_token_id=tokenizer.mask_token_id,
                max_new_tokens=args.max_new,
                steps=steps,
                block_size=block_size,
                temperature=args.temperature,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                decode_top_k=0,
                remask_strategy="low_confidence",
                debug=False,
                tokenizer=None,
            )

        # Reconstruct intermediate sequences using fill_steps (positions filled at their step and never changed later)
        full_seq = full_seq.cpu()
        fill_steps = fill_steps.cpu()
        prompt_window = input_ids.size(1)

        gen_region_ids_final = full_seq[0, prompt_window : prompt_window + args.max_new].tolist()
        gen_region_steps = fill_steps[0, prompt_window : prompt_window + args.max_new].tolist()

        print("\nSnapshots reconstructed from a single run:")
        for s in report_steps:
            ids_at_s: List[int] = []
            for tok_id, step_tag in zip(gen_region_ids_final, gen_region_steps):
                if step_tag > 0 and step_tag <= s:
                    ids_at_s.append(tok_id)
                else:
                    ids_at_s.append(tokenizer.mask_token_id)
            text_at_s = _decode_with_masks(tokenizer, ids_at_s)
            print(f"[snapshot step {s:>3}] {text_at_s}")

    if args.mode in ("separate", "both"):
        print("\nFull generations from separate runs:")
        for s in report_steps:
            with torch.inference_mode():
                seq_s, _ = run_diffusion_generation(
                    model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    mask_token_id=tokenizer.mask_token_id,
                    max_new_tokens=args.max_new,
                    steps=s,
                    block_size=block_size,
                    temperature=args.temperature,
                    do_sample=do_sample,
                    top_k=top_k,
                    top_p=top_p,
                    decode_top_k=0,
                    remask_strategy="low_confidence",
                    debug=False,
                    tokenizer=None,
                )
            seq_s = seq_s.cpu()
            prompt_window = input_ids.size(1)
            gen_ids = seq_s[0, prompt_window : prompt_window + args.max_new].tolist()
            text_s = _decode_with_masks(tokenizer, gen_ids)
            print(f"[separate step {s:>3}] {text_s}")


if __name__ == "__main__":
    main()
