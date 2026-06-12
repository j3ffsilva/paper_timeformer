#!/usr/bin/env python3
"""Tarefa 2 (adendo encoder-fixo) -- Passo 0' com BERT pre-treinado congelado.

Em palavras simples: ate aqui, toda medida de mudanca veio de um encoder
pequeno (d_model=128, 3 camadas) treinado do zero nos dados do SemEval. Este
script repete a medida mais simples (APD: distancia media entre as
ocorrencias de 1850 e as de 2000) usando o `bert-base-uncased` PRONTO
(congelado, sem nenhum treino nos nossos dados), nas MESMAS frases. Isso da'
um "teto de oraculo": se o BERT tambem nao discriminar bem o `truth.tsv`,
o problema nao e' o nosso encoder pequeno -- e' a tarefa/dados. Se o BERT
discriminar bem melhor, o gargalo e' a qualidade do encoder.

As frases sao lidas diretamente do corpus lematizado
(`data/processed/semeval2020_task1/eng_lemma/corpus/{1810-1860,1960-2010}.txt`),
onde a palavra-alvo ja aparece marcada com a POS (ex.: "plane_nn"). Para o
BERT, troca-se "plane_nn" -> "plane" e usa-se o mapeamento de subtokens do
tokenizer para extrair o vetor da palavra-alvo (media dos subtokens).

Por custo (CPU, ~58k ocorrencias no total), amostra-se no maximo
`--max-occurrences` por (palavra, periodo).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import (
    average_precision_score,
    normalized_mutual_info_score,
    roc_auc_score,
)
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

POS_SUFFIX = re.compile(r"_(nn|vb)$")


def strip_pos(token: str) -> str:
    return POS_SUFFIX.sub("", token)


def find_occurrences(corpus_path: Path, targets: set[str]) -> dict[str, list[tuple[list[str], int]]]:
    """For each target, list of (tokens_with_target_stripped, target_index)."""
    occurrences: dict[str, list[tuple[list[str], int]]] = {target: [] for target in targets}
    with corpus_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            tokens = line.split()
            for index, token in enumerate(tokens):
                if token in targets:
                    stripped = list(tokens)
                    stripped[index] = strip_pos(token)
                    occurrences[token].append((stripped, index))
    return occurrences


def evaluate_score(rows: list[dict], truth: dict[str, dict[str, float]], score_key: str) -> dict:
    rows = [row for row in rows if row[score_key] is not None]
    rows = sorted(rows, key=lambda row: row["target"])
    graded = np.array([truth[row["target"]]["graded"] for row in rows])
    binary = np.array([truth[row["target"]]["binary"] for row in rows])
    scores = np.array([row[score_key] for row in rows])
    rho, p_value = spearmanr(graded, scores)
    return {
        "n_targets": len(rows),
        "spearman": float(rho),
        "spearman_p": float(p_value),
        "roc_auc": float(roc_auc_score(binary, scores)),
        "average_precision": float(average_precision_score(binary, scores)),
    }


def average_pairwise_distance(vectors_a: Tensor, vectors_b: Tensor) -> float:
    a = F.normalize(vectors_a, dim=1)
    b = F.normalize(vectors_b, dim=1)
    return float((1.0 - (a @ b.T)).mean())


def cluster_period_nmi(vectors_a: Tensor, vectors_b: Tensor, *, seed: int) -> float | None:
    n_a, n_b = vectors_a.shape[0], vectors_b.shape[0]
    if n_a < 2 or n_b < 2:
        return None
    combined = F.normalize(torch.cat([vectors_a, vectors_b], dim=0), dim=1).numpy()
    labels = np.array([0] * n_a + [1] * n_b)
    clusters = KMeans(n_clusters=2, random_state=seed, n_init=10).fit_predict(combined)
    return float(normalized_mutual_info_score(labels, clusters))


def main() -> None:
    parser = argparse.ArgumentParser(description="Passo 0': APD/NMI com BERT pre-treinado congelado.")
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/processed/semeval2020_task1/eng_lemma/corpus"))
    parser.add_argument("--period0-file", default="1810-1860.txt")
    parser.add_argument("--period1-file", default="1960-2010.txt")
    parser.add_argument("--targets", type=Path, default=Path("data/processed/semeval2020_task1/eng_lemma/targets.txt"))
    parser.add_argument("--truth", type=Path, default=Path("data/processed/semeval2020_task1/eng_lemma/truth.tsv"))
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-occurrences", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--anchors", nargs="*", default=["plane_nn", "graft_nn", "chairman_nn", "tree_nn"])
    args = parser.parse_args()

    from transformers import AutoModel, AutoTokenizer

    targets = [line.strip() for line in args.targets.read_text(encoding="utf-8").splitlines() if line.strip()]
    target_set = set(targets)
    truth = {
        row["target"]: {"binary": float(row["binary"]), "graded": float(row["graded"])}
        for row in csv.DictReader(args.truth.open("r", encoding="utf-8", newline=""), delimiter="\t")
    }

    rng = random.Random(args.seed)
    print("Lendo corpora e localizando ocorrencias...")
    occ_d0 = find_occurrences(args.corpus_dir / args.period0_file, target_set)
    occ_d1 = find_occurrences(args.corpus_dir / args.period1_file, target_set)
    for target in targets:
        print(f"  {target}: d0={len(occ_d0[target])} d1={len(occ_d1[target])}")

    print("Carregando BERT...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name, output_hidden_states=True)
    model.eval()

    def encode_batch(examples: list[tuple[list[str], int]]) -> dict[str, Tensor]:
        token_lists = [tokens for tokens, _ in examples]
        encoding = tokenizer(
            token_lists,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**encoding)
        hidden_states = outputs.hidden_states  # tuple: (embeddings, layer1, ..., layer12)
        last = hidden_states[-1]
        mean_last_4 = torch.stack(hidden_states[-4:], dim=0).mean(dim=0)

        last_vecs, mean4_vecs = [], []
        for i, (_, target_index) in enumerate(examples):
            word_ids = encoding.word_ids(batch_index=i)
            subtoken_positions = [pos for pos, word_id in enumerate(word_ids) if word_id == target_index]
            if not subtoken_positions:
                subtoken_positions = [1]  # fallback: first non-special token
            positions = torch.tensor(subtoken_positions)
            last_vecs.append(last[i, positions].mean(dim=0))
            mean4_vecs.append(mean_last_4[i, positions].mean(dim=0))
        return {"last": torch.stack(last_vecs), "mean_last_4": torch.stack(mean4_vecs)}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    qualitative: dict = {}
    rows_by_layer: dict[str, list[dict]] = {"last": [], "mean_last_4": []}

    for target in targets:
        sample_d0 = occ_d0[target]
        sample_d1 = occ_d1[target]
        if len(sample_d0) > args.max_occurrences:
            sample_d0 = rng.sample(sample_d0, args.max_occurrences)
        if len(sample_d1) > args.max_occurrences:
            sample_d1 = rng.sample(sample_d1, args.max_occurrences)

        if not sample_d0 or not sample_d1:
            print(f"  AVISO: {target} sem ocorrencias suficientes (d0={len(sample_d0)}, d1={len(sample_d1)}) -- pulando")
            continue

        vectors = {"last": {"d0": [], "d1": []}, "mean_last_4": {"d0": [], "d1": []}}
        for side, sample in (("d0", sample_d0), ("d1", sample_d1)):
            for start in range(0, len(sample), args.batch_size):
                batch = sample[start : start + args.batch_size]
                encoded = encode_batch(batch)
                for layer in ("last", "mean_last_4"):
                    vectors[layer][side].append(encoded[layer])

        for layer in ("last", "mean_last_4"):
            v_d0 = torch.cat(vectors[layer]["d0"])
            v_d1 = torch.cat(vectors[layer]["d1"])
            apd = average_pairwise_distance(v_d0, v_d1)
            nmi = cluster_period_nmi(v_d0, v_d1, seed=args.seed)
            rows_by_layer[layer].append({
                "target": target,
                "apd": apd,
                "nmi": nmi,
                "n_d0": v_d0.shape[0],
                "n_d1": v_d1.shape[0],
            })

        print(f"  {target}: apd(last)={rows_by_layer['last'][-1]['apd']:.4f} apd(mean_last_4)={rows_by_layer['mean_last_4'][-1]['apd']:.4f}")

        if target in args.anchors:
            qualitative[target] = {
                "d0_sentences": [" ".join(tokens) for tokens, _ in sample_d0[:10]],
                "d1_sentences": [" ".join(tokens) for tokens, _ in sample_d1[:10]],
            }

    metrics = {}
    for layer, rows in rows_by_layer.items():
        metrics[layer] = {
            "apd": evaluate_score(rows, truth, "apd"),
            "nmi": evaluate_score(rows, truth, "nmi"),
        }
        path = args.output_dir / f"rows_{layer}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    result = {"model": args.model_name, "max_occurrences": args.max_occurrences, "metrics": metrics}
    (args.output_dir / "pretrained_oracle_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.output_dir / "qualitative_sentences.json").write_text(json.dumps(qualitative, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
