"""
Utilities for Llada diffusion generation and preview.

Provides:
* run_diffusion_generation – diffusion-style sampler that fills masked tokens.
* GenerationPreviewCallback – Trainer callback that logs periodic previews.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import List, Optional, Tuple

import torch
from transformers.trainer_callback import TrainerCallback


def _add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Inject temperature-controlled noise before sampling."""
    if temperature <= 0:
        return logits
    logits64 = logits.to(torch.float64)
    noise = torch.rand_like(logits64, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise + 1e-12)) ** temperature
    return logits64.exp() / gumbel_noise


def _get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """Split masked tokens evenly across diffusion steps within a block."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(
        mask_num.size(0),
        steps,
        device=mask_index.device,
        dtype=torch.int64,
    ) + base
    for i in range(mask_num.size(0)):
        if remainder[i] > 0:
            num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


def run_diffusion_generation(
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
    debug: bool,
    tokenizer=None,
) -> Tuple[torch.LongTensor, torch.Tensor]:
    """Run Llada diffusion sampling given a masked prefix."""
    device = input_ids.device
    batch_size, prompt_len = input_ids.shape

    x = torch.full(
        (batch_size, prompt_len + max_new_tokens),
        mask_token_id,
        dtype=torch.long,
        device=device,
    )
    x[:, :prompt_len] = input_ids.clone()

    fill_steps = torch.zeros(
        (batch_size, prompt_len + max_new_tokens),
        dtype=torch.int16,
        device=device,
    )
    fill_steps[:, :prompt_len] = -1

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
                probs = torch.softmax(logits_with_noise, dim=-1)
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
                k = max(0, int(num_transfer_tokens[b, step_idx].item()))
                if decode_top_k is not None and decode_top_k > 0:
                    k = min(k, decode_top_k)
                if k == 0:
                    continue
                _, top_idx = torch.topk(confidence[b], k)
                transfer_index[b, top_idx] = True

            updated_positions = transfer_index & mask_index
            x = torch.where(updated_positions, x0, x)
            step_counter = block_id * steps_per_block + step_idx + 1
            if updated_positions.any():
                fill_steps[updated_positions] = step_counter

            if debug and tokenizer is not None:
                logger = logging.getLogger(__name__)
                changed = updated_positions.nonzero(as_tuple=False)
                if changed.numel() == 0:
                    logger.info(
                        "[debug] block %d step %d: no positions updated.",
                        block_id + 1,
                        step_idx + 1,
                    )
                else:
                    logger.info(
                        "[debug] block %d step %d: updated %d positions.",
                        block_id + 1,
                        step_idx + 1,
                        changed.size(0),
                    )
                    for b, pos in changed:
                        tok_id = x[b, pos].item()
                        tok_str = tokenizer.convert_ids_to_tokens(tok_id)
                        logger.info(
                            "    batch %d pos %d -> token %s (%d)",
                            int(b.item()),
                            int(pos.item()),
                            tok_str,
                            tok_id,
                        )

    return x, fill_steps


class GenerationPreviewCallback(TrainerCallback):
    """Periodically sample the current model to monitor qualitative progress."""

    def __init__(
        self,
        tokenizer,
        prompts: List[str],
        interval: int,
        max_new_tokens: int,
        num_diffusion_steps: int,
        temperature: float,
        do_sample: bool,
        top_k: Optional[int],
        top_p: Optional[float],
        block_size: int,
        decode_top_k: int,
        mask_token_id: Optional[int],
        debug_generation: bool,
        preview_log_path: Optional[str] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.interval = interval
        self.max_new_tokens = max_new_tokens
        self.num_diffusion_steps = num_diffusion_steps
        self.temperature = temperature
        self.do_sample = do_sample
        self.top_k = top_k
        self.top_p = top_p
        self.block_size = block_size
        self.decode_top_k = decode_top_k
        self.mask_token_id = mask_token_id
        self.trainer = None
        self.debug_generation = debug_generation
        self.preview_log_path = preview_log_path

        encoded = tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        self.input_ids = encoded["input_ids"]
        self.attention_mask = encoded.get("attention_mask")
        if self.attention_mask is None:
            self.attention_mask = torch.ones_like(self.input_ids)
        self.prompt_lengths = self.attention_mask.sum(dim=1)

    def _should_generate(self, state) -> bool:
        return (
            self.interval > 0
            and state.global_step != 0
            and state.global_step % self.interval == 0
        )

    def attach_trainer(self, trainer):
        self.trainer = trainer

    def _run_generation(self, trainer, model, state):
        logger = logging.getLogger(__name__)
        unwrapped = trainer.accelerator.unwrap_model(model)
        device = next(unwrapped.parameters()).device

        input_ids = self.input_ids.to(device)
        attention_mask = self.attention_mask.to(device) if self.attention_mask is not None else None

        was_training = unwrapped.training
        unwrapped.eval()
        with torch.inference_mode():
            generated_sequences, fill_steps = run_diffusion_generation(
                unwrapped,
                input_ids=input_ids,
                attention_mask=attention_mask,
                mask_token_id=self.mask_token_id,
                max_new_tokens=self.max_new_tokens,
                steps=self.num_diffusion_steps,
                block_size=self.block_size,
                temperature=self.temperature,
                do_sample=self.do_sample,
                top_k=self.top_k,
                top_p=self.top_p,
                decode_top_k=self.decode_top_k,
                debug=self.debug_generation,
                tokenizer=self.tokenizer if self.debug_generation else None,
            )
        if was_training:
            unwrapped.train()

        generated_sequences = generated_sequences.cpu()
        fill_steps = fill_steps.cpu()
        input_ids_cpu = self.input_ids
        prompt_window = input_ids_cpu.size(1)

        logger.info("=" * 60)
        logger.info("📝 Generation preview at step %s", state.global_step)
        preview_records = []
        for idx, prompt in enumerate(self.prompts):
            prompt_len = int(self.prompt_lengths[idx].item())
            full_prompt_ids = input_ids_cpu[idx, :prompt_len]
            generated_ids = generated_sequences[idx, prompt_window : prompt_window + self.max_new_tokens]
            generated_step_vals = fill_steps[idx, prompt_window : prompt_window + self.max_new_tokens]

            generated_list = []
            step_list = []
            for token_id, step_tag in zip(generated_ids.tolist(), generated_step_vals.tolist()):
                generated_list.append(token_id)
                step_list.append(step_tag)

            generated_text = self.tokenizer.decode(
                generated_list,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
            if not generated_text:
                generated_text = self.tokenizer.decode(
                    generated_list,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                ).strip()

            prompt_tokens = self.tokenizer.convert_ids_to_tokens(full_prompt_ids.tolist())
            gen_tokens = self.tokenizer.convert_ids_to_tokens(generated_list) if generated_list else []

            prompt_parts = []
            for tok, tid in zip(prompt_tokens, full_prompt_ids.tolist()):
                tok_display = tok if tok is not None else "<UNK>"
                prompt_parts.append(f"{tok_display}[P]")

            gen_parts = []
            for tok, tid, step_tag in zip(gen_tokens, generated_list, step_list):
                tok_display = tok if tok is not None else "<UNK>"
                tag = step_tag if step_tag > 0 else "?"
                gen_parts.append(f"{tok_display}[step{tag}]")

            token_trace = " ".join(prompt_parts + gen_parts)

            logger.info("[gen] prompt[%d] Prompt: %s", idx, prompt)
            logger.info("[gen] prompt[%d] Generated: %s", idx, generated_text if generated_text else "[empty]")
            logger.info("[gen] prompt[%d] tokens: %s", idx, token_trace if token_trace else "[empty]")
            logger.info("-" * 40)
            preview_records.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "type": "generation_preview",
                    "step": int(state.global_step),
                    "prompt_index": idx,
                    "prompt": prompt,
                    "generated_text": generated_text,
                    "token_trace": token_trace,
                }
            )
        logger.info("=" * 60)

        if self.preview_log_path and preview_records:
            with open(self.preview_log_path, "a", encoding="utf-8") as f:
                for record in preview_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def on_train_begin(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer")
        if trainer is not None:
            self.trainer = trainer
        return control

    def on_log(self, args, state, control, **kwargs):
        trainer = self.trainer
        if trainer is None:
            return control
        if not state.is_local_process_zero:
            return control
        if self.mask_token_id is None:
            return control
        if not self._should_generate(state):
            return control
        self._run_generation(trainer, trainer.model, state)
        return control


__all__ = [
    "GenerationPreviewCallback",
    "run_diffusion_generation",
]
