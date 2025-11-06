import argparse
import json
import logging
import os
import pathlib
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import torch
import torch.multiprocessing as mp
from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlamaForCausalLM,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.utils import logging as hf_logging
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
from generation import GenerationPreviewCallback

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


class MetricsLoggerCallback(TrainerCallback):
    """Persist raw metrics emitted by Trainer into a jsonl file."""

    def __init__(self, log_path: Optional[str] = None) -> None:
        self.log_path = log_path
        self._start_time: Optional[float] = None
        self._initial_step: Optional[int] = None
        self._last_progress_line: Optional[str] = None

    @staticmethod
    def _format_timespan(seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02}:{minutes:02}:{secs:02}"

    def on_train_begin(self, args, state, control, **kwargs):
        self._start_time = time.time()
        if hasattr(state, "global_step") and state.global_step is not None:
            self._initial_step = int(state.global_step)
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not state.is_local_process_zero:
            return control

        if self._start_time is None:
            self._start_time = time.time()
        if self._initial_step is None and hasattr(state, "global_step") and state.global_step is not None:
            self._initial_step = int(state.global_step)

        now = time.time()
        elapsed = now - self._start_time if self._start_time is not None else 0.0
        global_step = int(state.global_step)
        initial_step = self._initial_step or 0
        steps_completed = max(0, global_step - initial_step)
        max_steps = getattr(state, "max_steps", None)
        progress_record = None
        progress_line = None

        if max_steps and max_steps > 0 and global_step >= 0:
            progress_ratio = min(1.0, global_step / max_steps)
            steps_per_sec = steps_completed / elapsed if elapsed > 0 and steps_completed > 0 else None
            remaining_steps = max_steps - global_step
            eta_seconds = (remaining_steps / steps_per_sec) if steps_per_sec and steps_per_sec > 0 else None

            progress_line_parts = [
                f"[progress] step {global_step}/{max_steps}",
                f"({progress_ratio * 100:.2f}%)",
                f"elapsed={self._format_timespan(elapsed)}",
            ]
            if eta_seconds is not None:
                progress_line_parts.append(f"eta={self._format_timespan(eta_seconds)}")
            if steps_per_sec is not None:
                progress_line_parts.append(f"step/s={steps_per_sec:.2f}")
            tokens_per_sec = logs.get("train_tokens_per_second")
            if isinstance(tokens_per_sec, (float, int)):
                progress_line_parts.append(f"tok/s={tokens_per_sec:.2f}")

            progress_line = " ".join(progress_line_parts)
            progress_record = {
                "step": global_step,
                "max_steps": max_steps,
                "percent": round(progress_ratio * 100, 4),
                "elapsed_seconds": round(elapsed, 4),
                "eta_seconds": round(eta_seconds, 4) if eta_seconds is not None else None,
                "steps_per_second": round(steps_per_sec, 6) if steps_per_sec is not None else None,
                "tokens_per_second": round(tokens_per_sec, 6) if isinstance(tokens_per_sec, (float, int)) else None,
                "initial_step": initial_step,
                "steps_completed": steps_completed,
            }

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "metrics",
            "step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "logs": logs,
        }
        if progress_record is not None:
            record["progress"] = progress_record

        scalar_items = [
            f"{key}={value:.4f}" if isinstance(value, (float, int)) else f"{key}={value}"
            for key, value in logs.items()
        ]
        if scalar_items:
            logging.info("[metrics] step %s %s", state.global_step, ", ".join(scalar_items))
        if progress_line and progress_line != self._last_progress_line:
            logging.info(progress_line)
            self._last_progress_line = progress_line

        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

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
    parser.add_argument(
        "--disable_tqdm",
        action="store_true",
        help="训练时禁用 tqdm 进度条。",
    )
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
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="从指定 checkpoint 继续训练。若未指定则在输出目录内自动搜索。",
    )

    args = parser.parse_args()

    hf_logging.enable_default_handler()
    hf_logging.enable_explicit_format()
    hf_logging.set_verbosity_info()
    hf_logging.disable_progress_bar()

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
        disable_tqdm=args.disable_tqdm,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        # eval_on_start = True,
    )
    callbacks = [MetricsLoggerCallback()]
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

    if args.resume_from_checkpoint:
        resume_path = os.path.expanduser(args.resume_from_checkpoint)
        logging.info("从 checkpoint %s 恢复训练。", resume_path)
        trainer.train(resume_from_checkpoint=resume_path)
    elif check_for_checkpoints(args.output_dir):
        logging.info("检测到现有 checkpoint，自动从最新的 checkpoint 恢复。")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()



    

if __name__ == "__main__":
    main()
