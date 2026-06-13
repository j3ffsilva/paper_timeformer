#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from timeformers.pretrained_init import initialize_from_bert  # noqa: E402
from timeformers.real_models import RealStaticMLM  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize RealStaticMLM encoder layers and token embeddings from a BERT model."
    )
    parser.add_argument("--vocab", type=Path, required=True, help="JSON vocabulary produced by the training pipeline.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="prajjwal1/bert-tiny")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--keep-pos-suffix", action="store_true")
    args = parser.parse_args()

    from transformers import AutoModel, AutoTokenizer

    vocab = json.loads(args.vocab.read_text(encoding="utf-8"))
    if not isinstance(vocab, list) or not all(isinstance(token, str) for token in vocab):
        raise ValueError("--vocab must contain a JSON list of strings")
    try:
        pad_id = vocab.index("[PAD]")
    except ValueError as exc:
        raise ValueError("Vocabulary must contain [PAD]") from exc

    print(f"Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    bert_model = AutoModel.from_pretrained(args.model_name)
    model = RealStaticMLM(
        vocab_size=len(vocab),
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        pad_id=pad_id,
        norm_first=False,
        activation="gelu",
        layer_norm_eps=float(bert_model.config.layer_norm_eps),
        mask_padding=True,
    )
    report = initialize_from_bert(
        model,
        bert_model,
        tokenizer,
        vocab,
        model_name=args.model_name,
        strip_pos_suffix=not args.keep_pos_suffix,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    metadata = {
        "checkpoint": str(args.output),
        "model_config": {
            "vocab_size": len(vocab),
            "seq_len": args.seq_len,
            "d_model": args.d_model,
            "layers": args.layers,
            "heads": args.heads,
            "d_ff": args.d_ff,
            "dropout": args.dropout,
            "encoder_norm_order": "post",
            "activation": "gelu",
            "layer_norm_eps": float(bert_model.config.layer_norm_eps),
            "mask_padding": True,
        },
        "initialization": asdict(report),
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {metadata_path}")
    print(
        f"Copied {report.encoder_layers_copied} layers and "
        f"{report.token_embeddings_copied}/{len(vocab)} token embeddings"
    )


if __name__ == "__main__":
    main()
