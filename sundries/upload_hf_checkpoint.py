#!/usr/bin/env python3
"""
Upload a local Transformers checkpoint folder to a Hugging Face Hub model repo.

Features
- Creates the repo if missing (optional).
- By default uploads only inference-needed files (config + weights + tokenizer),
  skipping large training states (optimizer/scheduler/scaler).
- Can upload the entire folder if requested.
- Optional tokenizer folder upload.

Prereqs
- pip install huggingface_hub
- huggingface-cli login   (or set env HUGGINGFACE_HUB_TOKEN)

Examples
  # Upload inference-only files from a checkpoint directory to a subfolder
  PYTHONPATH=$(pwd) python3 sundries/upload_hf_checkpoint.py \
    --repo-id Midstream/tiny_llada \
    --checkpoint output/llada_40m_dl/checkpoint-463694 \
    --path-in-repo llada_40m_dl/checkpoint-463694 \
    --create --only-inference

  # Upload full folder (includes optimizer/scheduler states)
  python3 sundries/upload_hf_checkpoint.py \
    --repo-id Midstream/tiny_llada \
    --checkpoint output/llada_40m_dl/checkpoint-463694 \
    --path-in-repo llada_40m_dl/checkpoint-463694 \
    --create

  # Also upload a tokenizer directory under tokenizer/
  python3 sundries/upload_hf_checkpoint.py \
    --repo-id Midstream/tiny_llada \
    --checkpoint output/llada_40m_dl/checkpoint-463694 \
    --path-in-repo llada_40m_dl/checkpoint-463694 \
    --tokenizer-dir tokenizers/tinystory \
    --tokenizer-path-in-repo tokenizer \
    --create --only-inference
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

from huggingface_hub import HfApi, create_repo


INFERENCE_ALLOW_PATTERNS: List[str] = [
    # Configs
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    # Weights (either format)
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "model.safetensors",
    "model.safetensors.index.json",
    # Tokenizer assets (various tokenizers)
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "spiece.model",
]

TRAINING_IGNORE_PATTERNS: List[str] = [
    # Typical trainer artifacts we often want to skip for inference
    "optimizer*",
    "scheduler*",
    "scaler*",
    "trainer_state.json",
    "training_args.bin",
    "rng_state*.pth",
    "*.lock",
    "events.out.tfevents*",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Upload a local checkpoint folder to Hugging Face Hub.")
    ap.add_argument("--repo-id", required=True, help="Target repo id, e.g. 'Midstream/tiny_llada'.")
    ap.add_argument("--checkpoint", required=True, help="Local checkpoint folder to upload.")
    ap.add_argument(
        "--path-in-repo", default=None, help="Optional path prefix inside repo, e.g. 'llada_40m_dl/checkpoint-1234'."
    )
    ap.add_argument(
        "--tokenizer-dir",
        default=None,
        help="Optional tokenizer directory to upload as well (e.g., 'tokenizers/tinystory').",
    )
    ap.add_argument(
        "--tokenizer-path-in-repo",
        default="tokenizer",
        help="Where to place the tokenizer folder in the repo (default: tokenizer).",
    )
    ap.add_argument("--private", action="store_true", help="Create repo as private (if --create).")
    ap.add_argument("--create", action="store_true", help="Create the repo if it does not exist.")
    ap.add_argument(
        "--only-inference",
        action="store_true",
        help="Upload only inference-related files (config/weights/tokenizer), skipping training states.",
    )
    ap.add_argument(
        "--commit-message",
        default=None,
        help="Optional commit message. Defaults to a descriptive message based on inputs.",
    )
    return ap.parse_args()


def ensure_repo(repo_id: str, private: bool) -> None:
    try:
        # Create if missing; if exists, this is a no-op when exist_ok=True
        create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        print(f"[ok] Repo ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        raise SystemExit(f"[error] Failed to ensure repo '{repo_id}': {e}")


def upload_folder(
    api: HfApi,
    repo_id: str,
    local_dir: Path,
    path_in_repo: Optional[str],
    only_inference: bool,
    commit_message: Optional[str],
):
    folder_path = str(local_dir)
    commit_message = commit_message or f"Upload {local_dir} ({'inference-only' if only_inference else 'full'})"
    kwargs = dict(repo_id=repo_id, repo_type="model", folder_path=folder_path, commit_message=commit_message)
    if path_in_repo:
        kwargs["path_in_repo"] = path_in_repo

    if only_inference:
        kwargs["allow_patterns"] = INFERENCE_ALLOW_PATTERNS
        kwargs["ignore_patterns"] = TRAINING_IGNORE_PATTERNS

    print(f"[upload] {folder_path} -> {repo_id}/{path_in_repo or ''} ...")
    res = api.upload_folder(**kwargs)  # returns a CommitInfo
    print(f"[done] Commit: {res.oid} @ {res.commit_url}")


def main() -> None:
    args = parse_args()

    checkpoint_dir = Path(args.checkpoint).resolve()
    if not checkpoint_dir.is_dir():
        raise SystemExit(f"[error] Checkpoint directory not found: {checkpoint_dir}")

    if args.create:
        ensure_repo(args.repo_id, private=bool(args.private))

    api = HfApi()

    upload_folder(
        api=api,
        repo_id=args.repo_id,
        local_dir=checkpoint_dir,
        path_in_repo=args.path_in_repo,
        only_inference=bool(args.only_inference),
        commit_message=args.commit_message,
    )

    if args.tokenizer_dir is not None:
        tok_dir = Path(args.tokenizer_dir).resolve()
        if not tok_dir.is_dir():
            raise SystemExit(f"[error] Tokenizer directory not found: {tok_dir}")
        upload_folder(
            api=api,
            repo_id=args.repo_id,
            local_dir=tok_dir,
            path_in_repo=args.tokenizer_path_in_repo,
            only_inference=True,  # always inference-only for tokenizer assets
            commit_message=args.commit_message or f"Upload tokenizer assets from {tok_dir}",
        )

    print("All uploads complete.")


if __name__ == "__main__":
    main()

