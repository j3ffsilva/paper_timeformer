from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from torch import nn

from .real_models import RealStaticMLM


POS_SUFFIX = re.compile(r"_(nn|vb)$")


@dataclass(frozen=True)
class PretrainedInitReport:
    model_name: str
    encoder_layers_copied: int
    token_embeddings_copied: int
    token_embeddings_skipped: int
    strip_pos_suffix: bool


def _copy_parameter(destination: nn.Parameter, source: torch.Tensor, name: str) -> None:
    if destination.shape != source.shape:
        raise ValueError(
            f"Shape mismatch for {name}: destination={tuple(destination.shape)} "
            f"source={tuple(source.shape)}"
        )
    destination.copy_(source.to(device=destination.device, dtype=destination.dtype))


def _validate_compatibility(model: RealStaticMLM, bert_model: nn.Module) -> None:
    config = bert_model.config
    layers = list(bert_model.encoder.layer)
    target_layers = list(model.encoder.layers)
    expected = {
        "d_model": (model.d_model, config.hidden_size),
        "n_layers": (len(target_layers), config.num_hidden_layers),
        "n_heads": (target_layers[0].self_attn.num_heads, config.num_attention_heads),
        "d_ff": (target_layers[0].linear1.out_features, config.intermediate_size),
    }
    mismatches = [
        f"{name}: RealStaticMLM={actual}, BERT={wanted}"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if model.norm_first:
        mismatches.append("norm order: RealStaticMLM=pre, BERT=post")
    if model.activation_name != "gelu":
        mismatches.append(
            f"activation: RealStaticMLM={model.activation_name}, BERT=gelu"
        )
    if mismatches:
        raise ValueError("Incompatible BERT initialization:\n- " + "\n- ".join(mismatches))


def _copy_encoder_layer(target: nn.TransformerEncoderLayer, source: nn.Module) -> None:
    attention = source.attention
    with torch.no_grad():
        _copy_parameter(
            target.self_attn.in_proj_weight,
            torch.cat(
                [
                    attention.self.query.weight,
                    attention.self.key.weight,
                    attention.self.value.weight,
                ],
                dim=0,
            ),
            "self_attn.in_proj_weight",
        )
        _copy_parameter(
            target.self_attn.in_proj_bias,
            torch.cat(
                [
                    attention.self.query.bias,
                    attention.self.key.bias,
                    attention.self.value.bias,
                ],
                dim=0,
            ),
            "self_attn.in_proj_bias",
        )
        _copy_parameter(
            target.self_attn.out_proj.weight,
            attention.output.dense.weight,
            "self_attn.out_proj.weight",
        )
        _copy_parameter(
            target.self_attn.out_proj.bias,
            attention.output.dense.bias,
            "self_attn.out_proj.bias",
        )
        _copy_parameter(target.linear1.weight, source.intermediate.dense.weight, "linear1.weight")
        _copy_parameter(target.linear1.bias, source.intermediate.dense.bias, "linear1.bias")
        _copy_parameter(target.linear2.weight, source.output.dense.weight, "linear2.weight")
        _copy_parameter(target.linear2.bias, source.output.dense.bias, "linear2.bias")
        _copy_parameter(target.norm1.weight, attention.output.LayerNorm.weight, "norm1.weight")
        _copy_parameter(target.norm1.bias, attention.output.LayerNorm.bias, "norm1.bias")
        _copy_parameter(target.norm2.weight, source.output.LayerNorm.weight, "norm2.weight")
        _copy_parameter(target.norm2.bias, source.output.LayerNorm.bias, "norm2.bias")


def _wordpiece_ids(tokenizer, token: str) -> list[int]:
    if token == getattr(tokenizer, "unk_token", None):
        return [tokenizer.unk_token_id]
    encoded = tokenizer.encode(token, add_special_tokens=False)
    unk_id = tokenizer.unk_token_id
    if not encoded or (unk_id is not None and unk_id in encoded):
        return []
    return encoded


def initialize_from_bert(
    model: RealStaticMLM,
    bert_model: nn.Module,
    tokenizer,
    vocab: list[str],
    *,
    model_name: str,
    strip_pos_suffix: bool = True,
) -> PretrainedInitReport:
    """Copy compatible BERT encoder layers and averaged WordPiece embeddings."""
    if len(vocab) != model.vocab_size:
        raise ValueError(f"Vocabulary has {len(vocab)} entries, model expects {model.vocab_size}")
    _validate_compatibility(model, bert_model)

    for target_layer, source_layer in zip(model.encoder.layers, bert_model.encoder.layer):
        _copy_encoder_layer(target_layer, source_layer)

    source_embeddings = bert_model.embeddings.word_embeddings.weight
    copied = 0
    skipped = 0
    with torch.no_grad():
        for index, token in enumerate(vocab):
            lookup_token = POS_SUFFIX.sub("", token) if strip_pos_suffix else token
            piece_ids = _wordpiece_ids(tokenizer, lookup_token)
            if not piece_ids:
                skipped += 1
                continue
            vector = source_embeddings[piece_ids].mean(dim=0)
            model.token_emb.weight[index].copy_(
                vector.to(device=model.token_emb.weight.device, dtype=model.token_emb.weight.dtype)
            )
            copied += 1

    return PretrainedInitReport(
        model_name=model_name,
        encoder_layers_copied=len(model.encoder.layers),
        token_embeddings_copied=copied,
        token_embeddings_skipped=skipped,
        strip_pos_suffix=strip_pos_suffix,
    )
