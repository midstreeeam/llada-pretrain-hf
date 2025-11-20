#!/usr/bin/env python
"""
Download the full `m-a-p/FineFineWeb-sample` train split locally using Hugging Face
and configure this repo to read it from disk.

Usage (from repo root):

  python download_finefineweb_full.py \
      --out-dir ./data/finefineweb

After running, `utils/load_dataset.finefineweb` will load from this directory.
"""

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/finefineweb_full",
        help="Directory to save the full FineFineWeb-sample train split (for load_from_disk).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="m-a-p/FineFineWeb-sample",
        help="Hugging Face dataset name to download.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure we use an HF mirror if no endpoint is configured.
    if (
        "HF_ENDPOINT" not in os.environ
        and "HF_HUB_ENDPOINT" not in os.environ
        and "HF_HUB_BASE_URL" not in os.environ
    ):
        mirror = "https://hf-mirror.com"
        os.environ["HF_ENDPOINT"] = mirror
        os.environ["HF_HUB_ENDPOINT"] = mirror
        os.environ["HF_HUB_BASE_URL"] = mirror
        print(f"[env] HF endpoints not set; defaulting to {mirror}")

    print(f"[download] Loading full train split of {args.dataset}")
    from datasets import load_dataset

    ds = load_dataset(args.dataset, split="train")
    print(f"[download] Loaded {len(ds)} examples, saving to {out_dir}")
    ds.save_to_disk(str(out_dir))
    print(f"[build] Saved Arrow dataset to {out_dir}")

    # Update diffusion/dataset_config.json so `finefineweb` uses this path.
    cfg_dir = Path("diffusion")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "dataset_config.json"

    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = {}
    local_paths = cfg.get("local_paths", {})
    local_paths["finefineweb"] = str(out_dir)
    cfg["local_paths"] = local_paths

    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print(f"[config] Updated {cfg_path} with finefineweb -> {out_dir}")
    print("[done] You can now train with dataset_name=finefineweb and it will load from disk.")


if __name__ == "__main__":
    main()

