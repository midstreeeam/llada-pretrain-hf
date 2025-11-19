#!/usr/bin/env python3
"""
PYTHONPATH=$(pwd) python3 sundries/visualize_generation.py \
    --prompt "One day," \
    --output "generation_sar.gif" \
    --steps 64 \
    --max-new 64 \
    --block-size 32
"""

import argparse
import os
import numpy as np
from typing import List, Optional, Tuple
import torch
from transformers import AutoTokenizer
from PIL import Image, ImageDraw, ImageFont
import imageio

from llada.configuration_llada import LLaDAConfig
from llada.modeling_llada import LLaDAModelLM

# Import helper functions from generation.py to avoid duplication
from generation import _add_gumbel_noise, _get_num_transfer_tokens

def run_diffusion_generation_with_history(
    model,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor],
    *,
    mask_token_id: int,
    max_new_tokens: int,
    steps: int,
    block_size: int,
    temperature: float,
    do_sample: bool,
    top_k: Optional[int],
    top_p: Optional[float],
    decode_top_k: Optional[int],
    remask_strategy: str = "low_confidence",
    eos_token_id: Optional[int] = None,
) -> List[torch.LongTensor]:
    """
    Run Llada diffusion sampling and return the history of sequences at each step.
    """
    if remask_strategy not in {"low_confidence", "random", "refinement"}:
        raise ValueError(f"Unsupported remask strategy: {remask_strategy}")

    device = input_ids.device
    batch_size, prompt_len = input_ids.shape

    # Initialize sequence with prompt + masks
    x = torch.full(
        (batch_size, prompt_len + max_new_tokens),
        mask_token_id,
        dtype=torch.long,
        device=device,
    )
    x[:, :prompt_len] = input_ids.clone()

    # If eos_token_id is provided, set the last token to EOS
    if eos_token_id is not None:
        x[:, -1] = eos_token_id

    history = [x.clone().cpu()]

    if attention_mask is not None:
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((batch_size, max_new_tokens), dtype=attention_mask.dtype, device=device),
            ],
            dim=-1,
        )

    assert max_new_tokens % block_size == 0
    num_blocks = max_new_tokens // block_size
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    for block_id in range(num_blocks):
        block_start = prompt_len + block_id * block_size
        block_end = prompt_len + (block_id + 1) * block_size
        block_mask_index = (x[:, block_start:block_end] == mask_token_id)
        num_transfer_tokens = _get_num_transfer_tokens(block_mask_index, steps_per_block)

        for step_idx in range(steps_per_block):
            mask_index = (x == mask_token_id)
            outputs = model(x, attention_mask=attention_mask)
            logits = outputs.logits

            if temperature > 0:
                logits_with_noise = _add_gumbel_noise(logits, temperature)
            else:
                logits_with_noise = logits

            if do_sample:
                sample_logits = logits_with_noise if temperature > 0 else logits
                probs = torch.softmax(sample_logits, dim=-1)
                x0 = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.shape[0], probs.shape[1])
            else:
                x0 = torch.argmax(logits_with_noise, dim=-1)

            probs = torch.softmax(logits, dim=-1)
            gathered = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, gathered, torch.tensor(-float("inf"), device=device))

            confidence[:, :block_start] = -float("inf")
            confidence[:, block_end:] = -float("inf")

            transfer_index = torch.zeros_like(confidence, dtype=torch.bool)
            for b in range(batch_size):
                available = torch.nonzero(mask_index[b, block_start:block_end], as_tuple=False).squeeze(-1)
                if available.numel() == 0:
                    continue

                k = max(0, int(num_transfer_tokens[b, step_idx].item()))
                if decode_top_k is not None and decode_top_k > 0:
                    k = min(k, decode_top_k)
                k = min(k, available.numel())
                if k == 0:
                    continue

                if remask_strategy == "random":
                    chosen = available[torch.randperm(available.numel(), device=device)[:k]]
                else:
                    block_conf = confidence[b, block_start:block_end][available]
                    top_vals, top_idx = torch.topk(block_conf, k, dim=-1)
                    chosen = available[top_idx]

                transfer_index[b, block_start:block_end][chosen] = True

            updated_positions = transfer_index & mask_index
            x = torch.where(updated_positions, x0, x)
            
            # Record history after update
            history.append(x.clone().cpu())

    return history

def render_frame(tokenizer, token_ids: List[int], changed_indices: set, remasked_indices: set, width=800, height_per_token=30, tokens_per_row=10, font_size=16) -> Image.Image:
    """
    Render a sequence of token IDs into an image using a grid layout.
    """
    # Calculate required height
    num_tokens = len(token_ids)
    num_rows = (num_tokens + tokens_per_row - 1) // tokens_per_row
    # Add some padding and space for footer
    total_height = num_rows * height_per_token + 60 
    
    image = Image.new("RGB", (width, total_height), "white")
    draw = ImageDraw.Draw(image)
    
    # Try to load a monospaced font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
    except IOError:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
            
    cell_width = width // tokens_per_row
    
    for idx, token_id in enumerate(token_ids):
        row = idx // tokens_per_row
        col = idx % tokens_per_row
        
        x = col * cell_width
        y = row * height_per_token + 10
        
        # Determine content
        if token_id == tokenizer.mask_token_id:
            text = "[MASK]"
            color = "lightgray"
        else:
            # Decode with special tokens included to see what it really is
            raw_text = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=True)
            
            if not raw_text:
                text = f"[{token_id}]"
            elif raw_text.isspace():
                text = " " # Represent space visibly
            else:
                text = raw_text.strip() # Strip strictly for display if it has content
                
            # Determine color
            if idx in remasked_indices:
                color = "gold" # Use gold/dark yellow for visibility on white
            elif idx in changed_indices:
                color = "red"
            else:
                color = "black"

        # Draw cell background for changed tokens? Maybe just text color is enough.
        # Let's draw the text.
        
        # Center text in cell (horizontally)? Or left align?
        # Left align with small padding is usually cleaner for reading.
        
        # Check if text fits
        text_len = draw.textlength(text, font=font)
        if text_len > cell_width - 4:
            # Truncate if too long
            while text_len > cell_width - 10 and len(text) > 1:
                text = text[:-1]
                text_len = draw.textlength(text + "..", font=font)
            text += ".."
            
        draw.text((x + 5, y), text, font=font, fill=color)
        
    return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="model_config/llada_40m.json")
    parser.add_argument("--tokenizer", type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument("--checkpoint", type=str, default="output/llada_40m_dl/checkpoint-463694")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=0, help="Block size for semi-autoregressive generation. If 0, uses max-new.")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--output", type=str, default="generation.gif")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--remask-strategy", type=str, default="low_confidence", choices=["low_confidence", "random", "refinement"])
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.checkpoint and os.path.isdir(args.checkpoint):
        model = LLaDAModelLM.from_pretrained(args.checkpoint)
    else:
        cfg = LLaDAConfig.from_pretrained(args.config)
        model = LLaDAModelLM(cfg, init_params=True)
    
    model.to(device)
    model.eval()

    enc = tokenizer([args.prompt], padding=True, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    print(f"Generating for prompt: '{args.prompt}'...")
    
    # Use sep_token_id if eos_token_id is None (common for some BERT models)
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
    
    block_size = args.block_size if args.block_size > 0 else args.max_new

    with torch.inference_mode():
        history = run_diffusion_generation_with_history(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask_token_id=tokenizer.mask_token_id,
            max_new_tokens=args.max_new,
            steps=args.steps,
            block_size=block_size,
            temperature=args.temperature,
            do_sample=args.sample,
            top_k=None,
            top_p=None,
            decode_top_k=0,
            remask_strategy=args.remask_strategy,
            eos_token_id=eos_id,
        )

    print(f"captured {len(history)} frames. Rendering GIF...")
    
    frames = []
    
    # Calculate layout parameters based on total tokens
    total_tokens = history[0].size(1)
    # Aim for roughly square-ish aspect ratio or fitting within 800px width
    # If we have ~80 tokens, 8 per row gives 10 rows. 100px per cell.
    # 800px width is standard.
    tokens_per_row = 8
    width = 800
    
    prev_ids = None
    
    for i, seq in enumerate(history):
        ids = seq[0].tolist()
        
        changed_indices = set()
        remasked_indices = set()
        
        if prev_ids is not None:
            for idx, (old, new) in enumerate(zip(prev_ids, ids)):
                if old != new:
                    changed_indices.add(idx)
                    if new == tokenizer.mask_token_id:
                         remasked_indices.add(idx)
        else:
            # First frame (all masks except prompt), maybe highlight prompt?
            # Or just highlight nothing.
            pass
            
        img = render_frame(tokenizer, ids, changed_indices, remasked_indices, width=width, tokens_per_row=tokens_per_row)
        
        # Add step number
        draw = ImageDraw.Draw(img)
        try:
             font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 15)
        except:
             font = ImageFont.load_default()
        
        # Draw footer info
        h = img.height
        draw.text((10, h - 30), f"Step: {i}/{args.steps}", fill="blue", font=font)
        
        frames.append(np.array(img))
        prev_ids = ids

    # Pause on the last frame
    for _ in range(10):
        frames.append(frames[-1])

    imageio.mimsave(args.output, frames, fps=5, loop=0)
    print(f"GIF saved to {args.output}")

if __name__ == "__main__":
    main()

