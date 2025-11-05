import argparse
import json
import logging
import os
import pathlib
import sys
from typing import List, Optional

import torch
import torch.multiprocessing as mp
from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlamaForCausalLM,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
import math

REPO_ROOT = pathlib.Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from collator import NTPCollator, LLaDACollator
from llada.modeling_llada import LLaDAModelLM
from llada.configuration_llada import LLaDAConfig
from trainer import MultipleLossTrainer
from utils.debug_func import analyze_weights, debug_data
from utils.load_dataset import get_dataset

# --- 设置日志 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
import ast

def is_main_process():
    """检查是否为主进程"""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
        return True
    except:
        return True



def check_for_checkpoints(output_dir):
    """
    检查指定的输出目录下是否存在类似 checkpoint- 的文件夹（更简练的版本）。
    """
    import re
    return os.path.exists(output_dir) and any(
        os.path.isdir(os.path.join(output_dir, item)) and re.match(r"^checkpoint-", item)
        for item in os.listdir(output_dir)
    )


def _add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        return logits
    logits64 = logits.to(torch.float64)
    noise = torch.rand_like(logits64, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise + 1e-12)) ** temperature
    return logits64.exp() / gumbel_noise


def _get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
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
) -> tuple[torch.LongTensor, torch.Tensor]:
    device = input_ids.device
    batch_size, prompt_len = input_ids.shape

    x = torch.full((batch_size, prompt_len + max_new_tokens), mask_token_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids.clone()

    fill_steps = torch.zeros((batch_size, prompt_len + max_new_tokens), dtype=torch.int16, device=device)
    fill_steps[:, :prompt_len] = -1

    if attention_mask is not None:
        attention_mask = torch.cat(
            [attention_mask, torch.ones((batch_size, max_new_tokens), dtype=attention_mask.dtype, device=device)],
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
                top_confidence, top_idx = torch.topk(confidence[b], k)
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

        logger.info("=" * 60)
        logger.info("📝 Generation preview at step %s", state.global_step)
        for idx, prompt in enumerate(self.prompts):
            prompt_len = int(self.prompt_lengths[idx].item())
            full_prompt_ids = input_ids_cpu[idx, :prompt_len]
            generated_ids = generated_sequences[idx, prompt_len : prompt_len + self.max_new_tokens]
            generated_step_vals = fill_steps[idx, prompt_len : prompt_len + self.max_new_tokens]

            generated_list = []
            step_list = []
            for token_id, step_tag in zip(generated_ids.tolist(), generated_step_vals.tolist()):
                tid = token_id
                if tid == self.mask_token_id:
                    continue
                if self.tokenizer.pad_token_id is not None and tid == self.tokenizer.pad_token_id:
                    break
                generated_list.append(tid)
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
                if step_tag <= 0:
                    gen_parts.append(f"{tok_display}[step?]")
                else:
                    gen_parts.append(f"{tok_display}[step{step_tag}]")

            token_trace = " ".join(prompt_parts + gen_parts)

            logger.info("[gen] prompt[%d] Prompt: %s", idx, prompt)
            logger.info("[gen] prompt[%d] Generated: %s", idx, generated_text if generated_text else "[empty]")
            logger.info("[gen] prompt[%d] tokens: %s", idx, token_trace if token_trace else "[empty]")
            logger.info("-" * 40)
        logger.info("=" * 60)

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


def main():
    # --- 1. 设置 ArgumentParser ---
    parser = argparse.ArgumentParser(description="使用可配置参数训练一个MLM模型")

    # 路径参数
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="预训练模型或本地模型/分词器的路径。")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="训练数据集的名称（如：finefineweb）。")

    parser.add_argument("--validation_dataset_name", type=str, default="finefineweb_validation",
                        help="验证数据集的名称（如：finefineweb_validation）。")
    parser.add_argument("--config_path", type=str, required=True,
                        help="模型配置文件的路径。")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="模型 checkpoints 和输出的保存路径。")
    parser.add_argument("--mode",default="llada")
    
    # MLM Schedule 参数
    parser.add_argument("--mlm_start_prob", type=float, default=0.25)
    parser.add_argument("--mlm_end_prob", type=float, default=0.15)
    parser.add_argument("--mlm_schedule_type", type=str, default='cosine')
    parser.add_argument("--tail_bias_factor", type=float, default=1.5)

    # 数据处理参数
    parser.add_argument("--max_length", type=int, default=512, help="输入序列的最大长度。")

    # TrainingArguments 参数
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)

    parser.add_argument("--per_device_eval_batch_size", type=int, default=2,
                        help="每个设备的评估批次大小。")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=500)

    parser.add_argument("--evaluation_strategy", type=str, default="steps",
                        help="评估策略 ('no', 'steps', 'epoch')。")
    parser.add_argument("--eval_steps", type=int, default=5000,
                        help="每隔多少步进行一次评估。")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action='store_true')
    parser.add_argument("--generation_interval", type=int, default=0,
                        help="每隔多少步执行一次示例文本生成，0 表示禁用。")
    parser.add_argument("--generation_prompts", nargs="+", default=None,
                        help="用于示例生成的提示语列表。")
    parser.add_argument("--generation_max_new_tokens", type=int, default=64,
                        help="示例生成时要生成的新 token 数量。")
    parser.add_argument("--generation_num_diffusion_steps", type=int, default=128,
                        help="扩散生成的迭代步数。")
    parser.add_argument("--generation_temperature", type=float, default=1.0,
                        help="扩散生成温度。")
    parser.add_argument("--generation_do_sample", action="store_true",
                        help="示例生成时是否启用采样。")
    parser.add_argument("--generation_top_k", type=int, default=None,
                        help="示例生成时的 top-k 采样阈值。")
    parser.add_argument("--generation_top_p", type=float, default=None,
                        help="示例生成时的 nucleus 采样阈值。")
    parser.add_argument("--generation_block_size", type=int, default=8,
                        help="扩散生成的块大小。")
    parser.add_argument("--generation_decode_top_k", type=int, default=0,
                        help="每步最多解码的 token 数量，0 表示不额外限制。")
    parser.add_argument("--generation_debug", action="store_true",
                        help="在示例生成时输出扩散每一步的详细变化。")

    args = parser.parse_args()

    # --- 2. 打印和保存参数配置 ---
    # 只在主进程打印和保存参数配置
    if is_main_process():
        logging.info("=" * 80)
        logging.info("训练参数配置:")
        logging.info("=" * 80)
        args_dict = vars(args)
        for key, value in args_dict.items():
            logging.info(f"{key:30}: {value}")
        logging.info("=" * 80)
        
        os.makedirs(args.output_dir, exist_ok=True)
        args_json_path = os.path.join(args.output_dir, "training_args.json")
        with open(args_json_path, "w", encoding="utf-8") as f:
            json.dump(args_dict, f, ensure_ascii=False, indent=4)
        logging.info(f"所有参数已保存至: {args_json_path}")



    # --- 4. 加载数据集和分词器 ---
    if is_main_process():
        logging.info(f"加载训练数据集 '{args.dataset_name}'...")
    train_dataset = get_dataset(args.dataset_name)

    eval_dataset = None
    if args.validation_dataset_name and args.validation_dataset_name.lower() not in ("", "none"):
        if is_main_process():
            logging.info(f"加载验证数据集 '{args.validation_dataset_name}'...")
        eval_dataset = get_dataset(args.validation_dataset_name)


    evaluation_strategy = args.evaluation_strategy
    eval_steps = args.eval_steps

    if eval_dataset is None:
        if evaluation_strategy not in ("no", "none"):
            logging.info("未提供验证数据集或加载失败，自动将 evaluation_strategy 设置为 'no'")
        evaluation_strategy = "no"
        eval_steps = None


    model_path = args.model_name_or_path
    if is_main_process():
        logging.info(f"从路径 '{model_path}' 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.eos_token is None:
        tokenizer.eos_token_id = 50279
    if tokenizer.bos_token is None:
        tokenizer.bos_token_id = 50285

    shared_step = mp.Value('i', 0)


    if args.mode == 'llada':
        config = LLaDAConfig.from_pretrained(args.config_path)
        model = LLaDAModelLM(config,init_params=True)
        config.register_for_auto_class()
        model.register_for_auto_class("AutoModel")
    elif args.mode == 'llama':
        config = AutoConfig.from_pretrained(args.config_path)
        model = LlamaForCausalLM(config)
    else:
        assert False
    # analyze_weights(model)


    if args.mode == 'llama':
        collator = NTPCollator(tokenizer, max_length=args.max_length)
    elif args.mode == 'llada':
        collator = LLaDACollator(tokenizer,max_length=args.max_length)
        

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={'min_lr_rate': 0.01},
        warmup_ratio=args.warmup_ratio,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        data_seed=args.seed,
        seed=args.seed,
        bf16=True,
        adam_beta2 = 0.95,
        weight_decay = 0.1,
        logging_steps=args.logging_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to='none',
        include_num_input_tokens_seen = True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        eval_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        # eval_on_start = True,
    )


    callbacks = []
    if args.generation_interval > 0:
        if tokenizer.mask_token_id is None:
            logging.warning("Tokenizer 缺少 mask_token，无法执行文本生成预览，将跳过此功能。")
        else:
            prompts = args.generation_prompts or [
                "Once upon a time, a curious child asked:",
                "In a quiet village by the sea,",
            ]
            logging.info(
                "启用文本生成预览，每 %s 步生成一次示例（debug=%s）。",
                args.generation_interval,
                args.generation_debug,
            )
            callbacks.append(
                GenerationPreviewCallback(
                    tokenizer=tokenizer,
                    prompts=prompts,
                    interval=args.generation_interval,
                    max_new_tokens=args.generation_max_new_tokens,
                    num_diffusion_steps=args.generation_num_diffusion_steps,
                    temperature=args.generation_temperature,
                    do_sample=args.generation_do_sample,
                    top_k=args.generation_top_k,
                    top_p=args.generation_top_p,
                    block_size=args.generation_block_size,
                    decode_top_k=args.generation_decode_top_k,
                    mask_token_id=tokenizer.mask_token_id,
                    debug_generation=args.generation_debug,
                )
            )

    # --- 7. 初始化并开始训练 ---
    trainer = MultipleLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset, # <-- 变量名更新
        eval_dataset=eval_dataset,   # <-- 新增，传递验证集
        data_collator=collator,
        # callbacks=None if args.mode == 'llama' or args.mode == 'llada' else [lazy_prob_scheduler_callback],
        keys_you_want_to_log = ['lm_loss','current_mlm_prob','masked_lm_loss','non_masked_lm_loss'],
        callbacks=callbacks if callbacks else None,
    )

    for cb in callbacks:
        attach = getattr(cb, "attach_trainer", None)
        if callable(attach):
            attach(trainer)

    # if is_main_process() and args.mode!='llama':
    #     debug_data(trainer, tokenizer, collator)

    if check_for_checkpoints(args.output_dir):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()



    

if __name__ == "__main__":
    main()
