from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repository root is on sys.path so intra-project imports resolve.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llada.configuration_llada import LLaDAConfig
from llada.modeling_llada import LLaDAModelLM

# 格式化输出函数
def format_params(num_params):
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return str(num_params)

def calculate_model_params(model, model_name):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    print(f"\n{model_name} 参数统计:")
    print(f"总参数量: {format_params(total_params)} ({total_params:,})")
    print(f"可训练参数量: {format_params(trainable_params)} ({trainable_params:,})")
    print(f"不可训练参数量: {format_params(non_trainable_params)} ({non_trainable_params:,})")


def load_llada_model(config_path: Path, init_params: bool = True) -> LLaDAModelLM:
    config = LLaDAConfig.from_pretrained(str(config_path))
    return LLaDAModelLM(config, init_params=init_params)


def main() -> None:
    parser = argparse.ArgumentParser(description="统计 LLaDA 模型参数量")
    parser.add_argument(
        "configs",
        nargs="*",
        default=["model_config/llada_40m.json"]
    )
    parser.add_argument(
        "--no-init",
        action="store_true",
        help="不显式初始化参数，仅构建模型结构（适用于快速估算）。",
    )
    args = parser.parse_args()

    for config_arg in args.configs:
        config_path = Path(config_arg)
        if not config_path.is_absolute():
            config_path = (REPO_ROOT / config_path).resolve()
        if not config_path.exists():
            print(f"[warn] 配置文件不存在：{config_path}")
            continue

        try:
            model = load_llada_model(config_path, init_params=not args.no_init)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] 无法加载模型配置 {config_path}: {exc}")
            continue

        calculate_model_params(model, config_path.stem)


if __name__ == "__main__":
    main()
