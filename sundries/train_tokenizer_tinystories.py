import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List

from datasets import load_dataset


def iter_text(ds) -> Iterable[str]:
    for ex in ds:
        t = ex.get("text")
        if isinstance(t, str) and t:
            yield t


def save_special_tokens_map(out_dir: Path, tokens: dict):
    (out_dir / "special_tokens_map.json").write_text(
        json.dumps(tokens, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def save_tokenizer_config(out_dir: Path, model_max_length: int):
    cfg = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": model_max_length,
        "clean_text": True,
        "strip_accents": True,
        "do_lower_case": True,
        "unk_token": "[UNK]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "mask_token": "[MASK]",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
    }
    (out_dir / "tokenizer_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_vocab_txt(out_dir: Path, vocab_tokens: List[str]):
    # For compatibility with BertTokenizerFast
    (out_dir / "vocab.txt").write_text("\n".join(vocab_tokens), encoding="utf-8")


def train_wordpiece(out_dir: Path, vocab_size: int, max_length: int):
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece
    from tokenizers.trainers import WordPieceTrainer
    from tokenizers.normalizers import Lowercase, NFD, StripAccents, Sequence
    from tokenizers.pre_tokenizers import Whitespace

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load TinyStories
    ds = load_dataset("roneneldan/TinyStories", split="train")

    # 2) Build tokenizer
    tok = Tokenizer(WordPiece(unk_token="[UNK]"))
    tok.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
    tok.pre_tokenizer = Whitespace()

    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<bos>", "<eos>"]
    trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=special_tokens, min_frequency=2)
    tok.train_from_iterator(iter_text(ds), trainer=trainer, length=len(ds))

    # 3) Save tokenizer.json
    tok.save(str(out_dir / "tokenizer.json"))

    # 4) Save special token metadata and config
    save_special_tokens_map(
        out_dir,
        {
            "unk_token": {"id": tok.token_to_id("[UNK]"), "content": "[UNK]"},
            "sep_token": {"id": tok.token_to_id("[SEP]"), "content": "[SEP]"},
            "pad_token": {"id": tok.token_to_id("[PAD]"), "content": "[PAD]"},
            "cls_token": {"id": tok.token_to_id("[CLS]"), "content": "[CLS]"},
            "mask_token": {"id": tok.token_to_id("[MASK]"), "content": "[MASK]"},
            "bos_token": {"id": tok.token_to_id("<bos>"), "content": "<bos>"},
            "eos_token": {"id": tok.token_to_id("<eos>"), "content": "<eos>"},
        },
    )
    save_tokenizer_config(out_dir, model_max_length=max_length)

    # 5) Save vocab.txt ordered by id (optional but useful)
    vocab = tok.get_vocab()
    id_to_token = [None] * len(vocab)
    for t, i in vocab.items():
        if 0 <= i < len(id_to_token):
            id_to_token[i] = t
    write_vocab_txt(out_dir, [t if t is not None else "[UNK]" for t in id_to_token])

    # 6) Emit a summary
    print("Saved tokenizer to:", out_dir)
    print("Vocab size:", len(vocab))
    for key in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<bos>", "<eos>"]:
        print(f"{key} ->", tok.token_to_id(key))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True, help="Directory to save the tokenizer")
    ap.add_argument("--vocab_size", type=int, default=4096)
    ap.add_argument("--max_length", type=int, default=1024)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    train_wordpiece(out_dir, vocab_size=args.vocab_size, max_length=args.max_length)


if __name__ == "__main__":
    main()

