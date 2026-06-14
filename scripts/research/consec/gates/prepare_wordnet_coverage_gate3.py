#!/usr/bin/env python3
"""Prepare a WordNet coverage audit for all SemEval English targets.

The script is compatible with Python 3.7 so it can use the same NLTK WordNet
installation as the published ConSeC environment.
"""

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path


PERIODS = ("1810-1860", "1960-2010")
POS_MAP = {"nn": "n", "vb": "v"}
KNOWN_EVIDENCE = {
    "chairman_nn": "monosemous_control_gate2",
    "graft_nn": "partial_inventory_gap_gate2",
    "plane_nn": "validated_gate1",
    "tree_nn": "validated_gate2",
}
KNOWN_DECISIONS = {
    "chairman_nn": {
        "coverage_status": "monosemous_covered",
        "missing_or_mismatched_uses": "",
        "gate3_decision": "diagnostic_only",
        "confidence": "high",
        "notes": "Carried forward from Gate 2; one WordNet candidate.",
    },
    "graft_nn": {
        "coverage_status": "partial",
        "missing_or_mismatched_uses": (
            "botanical graft object/scion, distinct from the act of grafting"
        ),
        "gate3_decision": "eligible",
        "confidence": "high",
        "notes": (
            "Only preregistered corruption/medical subsets are eligible; "
            "botanical uses remain diagnostic-only."
        ),
    },
    "plane_nn": {
        "coverage_status": "sufficient",
        "missing_or_mismatched_uses": "",
        "gate3_decision": "eligible",
        "confidence": "high",
        "notes": "Carried forward from the blind-adjudicated ConSeC Gate 1.",
    },
    "tree_nn": {
        "coverage_status": "sufficient",
        "missing_or_mismatched_uses": "",
        "gate3_decision": "eligible",
        "confidence": "high",
        "notes": "Plant/diagram coverage validated in ConSeC Gate 2.",
    },
}

REVIEW_FIELDS = (
    "target",
    "pos",
    "n_wordnet_senses",
    "automatic_priority",
    "coverage_status",
    "missing_or_mismatched_uses",
    "gate3_decision",
    "confidence",
    "notes",
)

README = """# Auditoria de cobertura WordNet para a Porta 3

## Objetivo

Decidir, antes de executar novas previsões do ConSeC, se o inventário WordNet
3.0 contém sentidos adequados para representar os usos históricos de cada uma
das 37 palavras-alvo.

Esta não é uma tarefa de adivinhar se a palavra mudou semanticamente. Não
consulte o gold do SemEval nem previsões de modelos.

## Arquivos

Preencha somente:

```text
coverage_review.csv
```

Use como apoio:

```text
sense_inventory.csv
context_samples.csv
target_summary.csv
```

`sense_inventory.csv` contém todos os sensekeys e definições WordNet do alvo.
`context_samples.csv` contém quatro ocorrências determinísticas de cada período
por palavra, sempre que disponíveis.

Quatro linhas já vêm preenchidas porque foram decididas nas Portas 1 e 2:
`plane_nn`, `tree_nn`, `graft_nn` e `chairman_nn`. Elas permanecem visíveis
para deixar explícito que a decisão foi transportada de evidência anterior.
Revise as outras 33 linhas.

## Como revisar cada palavra

1. Leia todos os sentidos WordNet da palavra.
2. Leia os contextos de 1810-1860 e 1960-2010.
3. Pergunte se cada uso observado possui um sentido WordNet semanticamente
   adequado, sem forçá-lo ao candidato apenas "mais próximo".
4. Preencha os quatro campos de decisão.

Pode usar Google Translate ou dicionário para compreender o contexto. Não use
chatbot ou sistema WSD para decidir o sentido.

## `coverage_status`

Use exatamente um destes valores:

- `sufficient`: os usos amostrados possuem candidatos WordNet adequados;
- `partial`: pelo menos um uso observado não possui candidato exato, mas
  outros usos estão cobertos;
- `missing`: o uso principal observado não é representado adequadamente;
- `monosemous_covered`: há um único sentido e os contextos são compatíveis;
- `monosemous_mismatch`: há um único sentido, mas algum contexto usa outro
  significado;
- `unclear`: a amostra não permite decidir.

Uma palavra com muitos sentidos pode ser `sufficient`. Uma palavra com um
único sentido pode ser `monosemous_mismatch`. O número de sentidos mede carga
de revisão, não qualidade da cobertura.

## `missing_or_mismatched_uses`

Se a cobertura for `partial`, `missing` ou `monosemous_mismatch`, descreva
brevemente o uso ausente. Exemplo:

```text
objeto botânico enxertado (scion), distinto do ato de enxertar
```

Nos demais casos, pode deixar vazio.

## `gate3_decision`

Use:

- `eligible`: pode entrar na análise discriminativa da Porta 3;
- `diagnostic_only`: útil como controle ou para estudar lacuna, mas não deve
  entrar no score confirmatório;
- `exclude`: cobertura inadequada para esta linha experimental;
- `needs_more_context`: é necessário ampliar a amostra antes de decidir.

Palavras monossêmicas normalmente são `diagnostic_only`, pois não testam
desambiguação. Uma palavra com cobertura parcial pode ser `eligible` apenas se
os sentidos/subconjuntos cobertos puderem ser separados antes das previsões e
os usos ausentes forem mantidos fora do score.

## `confidence`

Use `high`, `medium` ou `low`.

## Critério de término

Ao final, todas as 37 linhas devem ter:

```text
coverage_status
gate3_decision
confidence
```

Não altere `target`, `pos`, `n_wordnet_senses` ou `automatic_priority`.
"""


def read_targets(path):
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def split_target(target):
    lemma, tag = target.rsplit("_", 1)
    if tag not in POS_MAP:
        raise ValueError("Unsupported target POS: {}".format(target))
    return lemma, POS_MAP[tag]


def strip_pos_suffix(token):
    if token.endswith("_nn") or token.endswith("_vb"):
        return token[:-3]
    return token


def centered_context(tokens, index, radius):
    start = max(0, index - radius)
    end = min(len(tokens), index + radius + 1)
    shown = [strip_pos_suffix(token) for token in tokens[start:end]]
    shown[index - start] = "[{}]".format(strip_pos_suffix(tokens[index]))
    return " ".join(shown)


def collect_occurrences(corpus_dir, targets, radius):
    target_set = set(targets)
    occurrences = {
        target: {period: [] for period in PERIODS} for target in targets
    }
    for period in PERIODS:
        path = corpus_dir / "{}.txt".format(period)
        with path.open(encoding="utf-8") as handle:
            for document_index, line in enumerate(handle):
                tokens = line.split()
                for token_index, token in enumerate(tokens):
                    if token not in target_set:
                        continue
                    occurrences[token][period].append(
                        {
                            "target": token,
                            "period": period,
                            "document_index": document_index,
                            "token_index": token_index,
                            "context": centered_context(
                                tokens, token_index, radius
                            ),
                        }
                    )
    return occurrences


def sample_occurrences(occurrences, per_period, seed):
    rows = []
    for target in sorted(occurrences):
        for period in PERIODS:
            candidates = list(occurrences[target][period])
            rng = random.Random(
                "{}:{}:{}:coverage".format(seed, target, period)
            )
            rng.shuffle(candidates)
            selected = sorted(
                candidates[:per_period],
                key=lambda row: (
                    row["document_index"],
                    row["token_index"],
                ),
            )
            for index, row in enumerate(selected, start=1):
                output = dict(row)
                output["sample_id"] = "{}-{}-{:02d}".format(
                    target, period, index
                )
                rows.append(output)
    return rows


def wordnet_inventory(target, wordnet):
    lemma, pos = split_target(target)
    rows = []
    for synset in wordnet.synsets(lemma, pos=pos):
        matching = [
            item
            for item in synset.lemmas()
            if item.name().lower() == lemma.lower()
        ]
        for item in matching:
            rows.append(
                {
                    "target": target,
                    "lemma": lemma,
                    "pos": pos,
                    "sensekey": item.key(),
                    "synset": synset.name(),
                    "definition": synset.definition(),
                    "examples": " | ".join(synset.examples()),
                }
            )
    return rows


def automatic_priority(n_senses):
    if n_senses == 0:
        return "no_inventory"
    if n_senses == 1:
        return "monosemous_control"
    if n_senses <= 3:
        return "low_review_burden"
    if n_senses <= 8:
        return "medium_review_burden"
    return "high_review_burden"


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare(args):
    from nltk.corpus import wordnet as wn

    targets = read_targets(args.targets)
    occurrences = collect_occurrences(
        args.corpus_dir, targets, args.context_radius
    )
    context_rows = sample_occurrences(
        occurrences, args.samples_per_period, args.seed
    )

    inventory_rows = []
    summary_rows = []
    review_rows = []
    for target in targets:
        target_inventory = wordnet_inventory(target, wn)
        inventory_rows.extend(target_inventory)
        lemma, pos = split_target(target)
        n_senses = len(target_inventory)
        counts = {
            period: len(occurrences[target][period]) for period in PERIODS
        }
        priority = automatic_priority(n_senses)
        summary_rows.append(
            {
                "target": target,
                "lemma": lemma,
                "pos": pos,
                "n_wordnet_senses": n_senses,
                "automatic_priority": priority,
                "d0_occurrences": counts["1810-1860"],
                "d1_occurrences": counts["1960-2010"],
                "prior_project_evidence": KNOWN_EVIDENCE.get(
                    target, "not_reviewed"
                ),
            }
        )
        review_row = {
            "target": target,
            "pos": pos,
            "n_wordnet_senses": n_senses,
            "automatic_priority": priority,
            "coverage_status": "",
            "missing_or_mismatched_uses": "",
            "gate3_decision": "",
            "confidence": "",
            "notes": "",
        }
        review_row.update(KNOWN_DECISIONS.get(target, {}))
        review_rows.append(review_row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.annotation_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "target_summary.csv",
        summary_rows,
        list(summary_rows[0]),
    )
    write_csv(
        args.output_dir / "sense_inventory.csv",
        inventory_rows,
        (
            "target",
            "lemma",
            "pos",
            "sensekey",
            "synset",
            "definition",
            "examples",
        ),
    )
    write_csv(
        args.output_dir / "context_samples.csv",
        context_rows,
        (
            "sample_id",
            "target",
            "period",
            "document_index",
            "token_index",
            "context",
        ),
    )
    write_csv(
        args.annotation_dir / "coverage_review.csv",
        review_rows,
        REVIEW_FIELDS,
    )
    for name in ("target_summary.csv", "sense_inventory.csv", "context_samples.csv"):
        (args.annotation_dir / name).write_bytes(
            (args.output_dir / name).read_bytes()
        )
    (args.annotation_dir / "README_REVIEWER.md").write_text(
        README, encoding="utf-8"
    )

    priorities = Counter(row["automatic_priority"] for row in summary_rows)
    summary = {
        "targets": len(targets),
        "wordnet_senses": len(inventory_rows),
        "context_samples": len(context_rows),
        "samples_per_target_period": args.samples_per_period,
        "seed": args.seed,
        "priority_counts": dict(sorted(priorities.items())),
        "known_evidence": KNOWN_EVIDENCE,
        "prefilled_decisions": len(KNOWN_DECISIONS),
        "pending_decisions": len(targets) - len(KNOWN_DECISIONS),
        "status": "awaiting_manual_coverage_review",
        "protocol_notes": [
            "No SemEval gold score is read or written.",
            "No ConSeC prediction is run or exposed.",
            "Automatic priority measures review burden, not semantic coverage.",
            "Gate 3 eligibility must be filled before model inference.",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--samples-per-period", type=int, default=4)
    parser.add_argument("--context-radius", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260613)
    return parser


if __name__ == "__main__":
    prepare(build_parser().parse_args())
