#!/usr/bin/env python
"""
Download a subset of the SmolLM corpus (specifically fineweb-edu-dedup) to a local JSONL file
and configure this repo to use it as the local source.

Usage (from repo root):
  python download_smollm.py \
    --out-dir ./data/fineweb_edu_dedup_20g \
    --max-bytes 20GiB
"""

import argparse
import json
import os
from pathlib import Path


def parse_size(arg: str) -> int:
    """Parse sizes like '10G', '10GiB', '10000000000' into bytes."""
    s = arg.strip().lower()
    multipliers = {
        "k": 10**3,
        "kb": 10**3,
        "kib": 2**10,
        "m": 10**6,
        "mb": 10**6,
        "mib": 2**20,
        "g": 10**9,
        "gb": 10**9,
        "gib": 2**30,
    }
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            return int(float(s[: -len(suffix)]) * mult)
    return int(float(s))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/smollm_10g",
        help="Directory to save the subset dataset (used with datasets.load_from_disk).",
    )
    parser.add_argument(
        "--max-bytes",
        type=str,
        default="10GiB",
        help="Approximate maximum raw text size to keep (e.g. '10GiB', '5G').",
    )
    parser.add_argument(
        "--percent",
        type=float,
        default=None,
        help="If set (e.g. 3.0), download this percentage of the dataset by row count instead of using --max-bytes.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceTB/smollm-corpus",
        help="Hugging Face dataset name to sample from.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="fineweb-edu-dedup",
        help="Dataset configuration/subset name.",
    )
    parser.add_argument(
        "--use-mirror",
        action="store_true",
        help="Use hf-mirror.com for downloading.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    # Ensure the target directory itself exists so we can write the JSONL file.
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / "smollm_subset.jsonl"

    # If percent mode is requested, ignore max_bytes and (re)write exactly that slice.
    if args.percent is not None:
        print(f"[mode] Using percent mode: {args.percent}% of {args.dataset} ({args.config})")

        # Ensure we use an HF mirror if no endpoint is configured.
        if args.use_mirror and "HF_ENDPOINT" not in os.environ and "HF_HUB_ENDPOINT" not in os.environ and "HF_HUB_BASE_URL" not in os.environ:
            mirror = "https://hf-mirror.com"
            os.environ["HF_ENDPOINT"] = mirror
            os.environ["HF_HUB_ENDPOINT"] = mirror
            os.environ["HF_HUB_BASE_URL"] = mirror
            print(f"[env] HF endpoints not set; defaulting to {mirror}")

        from datasets import load_dataset

        split_spec = f"train[:{args.percent}%]"
        print(f"[download] Loading split: {split_spec}")
        # Pass the config name (args.config) to load_dataset
        ds = load_dataset(args.dataset, args.config, split=split_spec)
        ds = ds.shuffle(seed=42)

        num_examples = 0
        total_bytes = 0
        with jsonl_path.open("w", encoding="utf-8") as writer:
            for row in ds:
                text = row.get("text") or row.get("content") or ""
                if not isinstance(text, str):
                    continue
                size = len(text.encode("utf-8", errors="ignore"))
                record = {"text": text}
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                num_examples += 1
                total_bytes += size
                if num_examples % 10000 == 0:
                    print(f"[download] Written {num_examples} examples, ~{total_bytes/2**30:.2f} GiB")

        print(f"[build] Final subset: {num_examples} examples, ~{total_bytes/2**30:.2f} GiB of text")
        print(f"[build] Saved JSONL to {jsonl_path}")

        # Update diffusion/dataset_config.json so `smollm` uses this path.
        cfg_dir = Path("diffusion")
        cfg_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = cfg_dir / "dataset_config.json"

        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
        else:
            cfg = {}
        local_paths = cfg.get("local_paths", {})
        local_paths["smollm"] = str(jsonl_path)
        cfg["local_paths"] = local_paths

        with cfg_path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

        print(f"[config] Updated {cfg_path} with smollm -> {jsonl_path}")
        print("[done] You can now train with dataset_name=smollm without remote download.")
        return

    max_bytes = parse_size(args.max_bytes)

    # Resume logic (byte-budget mode): if JSONL already exists, treat max_bytes as TOTAL target size
    existing_bytes = 0
    existing_examples = 0
    if jsonl_path.exists():
        existing_bytes = jsonl_path.stat().st_size
        # Count existing examples (lines)
        with jsonl_path.open("r", encoding="utf-8") as f_in:
            for existing_examples, _ in enumerate(f_in, start=1):
                pass
        print(
            f"[resume] Found existing subset: {existing_examples} examples, "
            f"~{existing_bytes/2**30:.2f} GiB"
        )
        if existing_bytes >= max_bytes:
            print("[resume] Existing file already meets or exceeds target size; nothing to do.")
            # Still ensure dataset_config points to this path.
            cfg_dir = Path("diffusion")
            cfg_dir.mkdir(parents=True, exist_ok=True)
            cfg_path = cfg_dir / "dataset_config.json"
            if cfg_path.exists():
                with cfg_path.open("r", encoding="utf-8") as f:
                    cfg = json.load(f)
            else:
                cfg = {}
            local_paths = cfg.get("local_paths", {})
            local_paths["smollm"] = str(jsonl_path)
            cfg["local_paths"] = local_paths
            with cfg_path.open("w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            print(f"[config] Updated {cfg_path} with smollm -> {jsonl_path}")
            return

    print(f"[download] Sampling from dataset: {args.dataset} ({args.config})")
    print(f"[download] Target directory     : {out_dir}")
    print(f"[download] Max raw text bytes   : {max_bytes}")

    print(f"[download] Max raw text bytes   : {max_bytes}")

    # Ensure we use an HF mirror if no endpoint is configured.
    if args.use_mirror and "HF_ENDPOINT" not in os.environ and "HF_HUB_ENDPOINT" not in os.environ and "HF_HUB_BASE_URL" not in os.environ:
        mirror = "https://hf-mirror.com"
        os.environ["HF_ENDPOINT"] = mirror
        os.environ["HF_HUB_ENDPOINT"] = mirror
        os.environ["HF_HUB_BASE_URL"] = mirror
        print(f"[env] HF endpoints not set; defaulting to {mirror}")

    # Import datasets *after* setting env so the hub client sees the mirror.
    from datasets import load_dataset

    # Use streaming so we don't need to download all shards up-front.
    # Pass args.config here as well
    stream = load_dataset(args.dataset, args.config, split="train", streaming=True)

    # Skip already-written examples in the stream (we still have to iterate them,
    # but we won't write them again).
    if existing_examples > 0:
        print(f"[resume] Skipping first {existing_examples} examples from remote stream...")
        it = iter(stream)
        skipped = 0
        while skipped < existing_examples:
            try:
                next(it)
            except StopIteration:
                break
            skipped += 1
        stream = it

    total_bytes = existing_bytes
    num_examples = existing_examples

    # Append to existing JSONL if present, otherwise create a new one.
    mode = "a" if jsonl_path.exists() else "w"
    with jsonl_path.open(mode, encoding="utf-8") as writer:
        for idx, row in enumerate(stream, start=existing_examples + 1):
            # Fall back to 'content' if 'text' is missing
            text = row.get("text") or row.get("content") or ""
            if not isinstance(text, str):
                continue
            size = len(text.encode("utf-8", errors="ignore"))
            # Allow at least one new example even if it slightly exceeds the budget.
            if total_bytes + size > max_bytes and num_examples > existing_examples:
                break

            record = {"text": text}
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")

            total_bytes += size
            num_examples += 1

            if num_examples % 10000 == 0:
                print(f"[download] Collected {num_examples} examples, ~{total_bytes/2**30:.2f} GiB of text")

    if num_examples == 0:
        raise SystemExit("No records collected; check network or dataset name.")

    print(f"[build] Final subset: {num_examples} examples, ~{total_bytes/2**30:.2f} GiB of text")
    print(f"[build] Saved/updated JSONL at {jsonl_path}")

    # Update diffusion/dataset_config.json so `smollm` uses this path.
    cfg_dir = Path("diffusion")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "dataset_config.json"

    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = {}
    local_paths = cfg.get("local_paths", {})
    local_paths["fineweb_edu_dedup"] = str(jsonl_path)
    cfg["local_paths"] = local_paths

    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print(f"[config] Updated {cfg_path} with fineweb_edu_dedup -> {out_dir}")
    print("[done] You can now train with dataset_name=fineweb_edu_dedup without remote download.")


if __name__ == "__main__":
    main()
