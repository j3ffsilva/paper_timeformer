#!/usr/bin/env python3
"""Prepare the preregistered ConSeC Gate 2 subsets without running a model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from timeformers.real_corpus import read_period_corpora  # noqa: E402


GRAFT_KEYWORDS = {
    "botanical_inventory_gap": {
        "apple",
        "bark",
        "branch",
        "bud",
        "cleft",
        "fruit",
        "grafting",
        "nursery",
        "orchard",
        "pear",
        "plant",
        "root",
        "sap",
        "scion",
        "shoot",
        "stock",
        "tree",
        "wax",
        "wedge",
    },
    "corruption": {
        "bribe",
        "bribery",
        "campaign",
        "corrupt",
        "corruption",
        "government",
        "illegal",
        "payoff",
        "payment",
        "police",
        "political",
        "politician",
        "racket",
        "scandal",
        "swindle",
    },
    "medical": {
        "aorta",
        "blood",
        "bone",
        "burn",
        "kidney",
        "leukemia",
        "marrow",
        "patient",
        "skin",
        "surgeon",
        "surgery",
        "tissue",
        "transplant",
        "wound",
    },
}

TREE_PLANT_KEYWORDS = {
    "bark",
    "branch",
    "forest",
    "fruit",
    "leaf",
    "leaves",
    "orchard",
    "pine",
    "plant",
    "root",
    "sap",
    "seed",
    "shade",
    "soil",
    "trunk",
    "wood",
    "woodland",
}

TREE_DIAGRAM_PRECEDERS = {
    "binary",
    "classification",
    "decision",
    "evolutionary",
    "family",
    "genealogical",
    "parse",
    "phylogenetic",
    "syntax",
    "taxonomic",
}

GRAFT_MEDICAL_NEAR_TARGET = {
    "aorta",
    "artery",
    "blood",
    "bone",
    "burn",
    "bypass",
    "heart",
    "kidney",
    "marrow",
    "organ",
    "patient",
    "skin",
    "surgery",
    "tissue",
    "transplant",
    "vascular",
    "wound",
}

WORDNET_INVENTORY = {
    "graft_nn": {
        "medical": "graft%1:08:00::",
        "corruption": "graft%1:04:01::",
        "grafting_act": "graft%1:04:00::",
    },
    "chairman_nn": {
        "presiding_officer": "chairman%1:18:01::",
    },
    "tree_nn": {
        "plant": "tree%1:20:00::",
        "diagram": "tree%1:25:00::",
        "person": "tree%1:18:00::",
    },
}

EXPECTED_MINIMUMS = {
    ("graft_nn", "corruption"): 15,
    ("graft_nn", "medical"): 15,
    ("tree_nn", "diagram"): 10,
    ("tree_nn", "plant"): 100,
    ("chairman_nn", "monosemous_control"): 100,
}

MAX_PER_PERIOD = {
    ("tree_nn", "plant"): 100,
    ("chairman_nn", "monosemous_control"): 50,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def centered_context(document: list[str], index: int, radius: int = 20) -> list[str]:
    return document[max(0, index - radius) : index + radius + 1]


def display_context(tokens: list[str], target: str) -> str:
    shown = list(tokens)
    target_positions = [index for index, token in enumerate(shown) if token == target]
    if not target_positions:
        raise ValueError(f"Target {target} is absent from context")
    center = min(target_positions, key=lambda index: abs(index - len(shown) // 2))
    shown[center] = f"[{target.removesuffix('_nn')}]"
    return " ".join(token.removesuffix("_nn") for token in shown)


def classify_graft(tokens: list[str]) -> tuple[str, int]:
    words = set(tokens)
    scores = {
        sense: len(words & keywords) for sense, keywords in GRAFT_KEYWORDS.items()
    }
    best = max(scores.values())
    winners = [sense for sense, score in scores.items() if score == best and score > 0]
    if len(winners) != 1:
        return "unlabeled", best
    winner = winners[0]
    if winner == "medical":
        target_index = tokens.index("graft_nn")
        near = set(tokens[max(0, target_index - 4) : target_index + 5])
        if not near & GRAFT_MEDICAL_NEAR_TARGET:
            return "unlabeled", best
    return winner, best


def classify_tree(document: list[str], index: int) -> tuple[str, int]:
    previous = document[index - 1] if index > 0 else ""
    following = document[index + 1] if index + 1 < len(document) else ""
    if previous in TREE_DIAGRAM_PRECEDERS or following == "diagram":
        return "diagram", 1
    tokens = centered_context(document, index, radius=12)
    if "family" in tokens:
        return "unlabeled", 0
    plant_score = len(set(tokens) & TREE_PLANT_KEYWORDS)
    return ("plant", plant_score) if plant_score >= 2 else ("unlabeled", plant_score)


def collect_candidates(corpus_dir: Path, seed: int) -> tuple[list[dict[str, str]], dict]:
    candidates = []
    raw_counts = Counter()
    selected_counts = Counter()

    for corpus in read_period_corpora(corpus_dir):
        period_rows = []
        for document_index, document in enumerate(corpus.documents):
            for token_index, target in enumerate(document):
                if target not in {"graft_nn", "chairman_nn", "tree_nn"}:
                    continue
                context = centered_context(document, token_index)
                if target == "graft_nn":
                    sense, evidence = classify_graft(context)
                elif target == "tree_nn":
                    sense, evidence = classify_tree(document, token_index)
                else:
                    sense, evidence = "monosemous_control", 1
                raw_counts[(corpus.period, target, sense)] += 1
                if sense == "unlabeled":
                    continue
                period_rows.append(
                    {
                        "sample_id": (
                            f"{corpus.period}-{target}-{document_index}-{token_index}"
                        ),
                        "corpus": corpus.period,
                        "target": target,
                        "heuristic_sense": sense,
                        "evidence_count": str(evidence),
                        "document_index": str(document_index),
                        "token_index": str(token_index),
                        "context": display_context(context, target),
                    }
                )

        grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
        for row in period_rows:
            grouped.setdefault((row["target"], row["heuristic_sense"]), []).append(row)
        for key, rows in grouped.items():
            limit = MAX_PER_PERIOD.get(key)
            if limit is not None and len(rows) > limit:
                random.Random(f"{seed}:{corpus.period}:{key}").shuffle(rows)
                rows = sorted(rows[:limit], key=lambda row: row["sample_id"])
            candidates.extend(rows)
            selected_counts[(corpus.period, *key)] += len(rows)

    candidates.sort(key=lambda row: row["sample_id"])
    aggregate = Counter(
        (row["target"], row["heuristic_sense"]) for row in candidates
    )
    minimum_checks = {
        f"{target}:{sense}": {
            "observed": aggregate[(target, sense)],
            "minimum": minimum,
            "passed": aggregate[(target, sense)] >= minimum,
        }
        for (target, sense), minimum in EXPECTED_MINIMUMS.items()
    }
    summary = {
        "seed": seed,
        "wordnet_inventory": WORDNET_INVENTORY,
        "raw_counts": [
            {
                "corpus": corpus,
                "target": target,
                "sense": sense,
                "count": count,
            }
            for (corpus, target, sense), count in sorted(raw_counts.items())
        ],
        "selected_counts": [
            {
                "corpus": corpus,
                "target": target,
                "sense": sense,
                "count": count,
            }
            for (corpus, target, sense), count in sorted(selected_counts.items())
        ],
        "minimum_checks": minimum_checks,
        "ready": all(check["passed"] for check in minimum_checks.values()),
        "design_notes": {
            "graft_botanical": (
                "Reported as an inventory gap, not scored as grafting_act."
            ),
            "chairman": (
                "Monosemous WordNet coverage control; excluded from WSD accuracy."
            ),
            "tree_person": (
                "Not evaluated because no reliable proper-name subset was found."
            ),
        },
    }
    return candidates, summary


AUDIT_FIELDS = ("item_id", "target", "context", "label", "confidence", "notes")
AUDIT_INSTRUCTIONS = """# Guia de anotação da Porta 2

## O que preencher

Preencha somente:

```text
annotator_a.csv
```

Em cada linha, complete `label`, `confidence` e, quando necessário, `notes`.
Não altere `item_id`, `target` ou `context`.

O alvo aparece entre colchetes: `[graft]` ou `[tree]`.

## Regras gerais

Classifique o significado que a palavra possui naquele contexto, não o tema
geral do texto. Use exatamente os rótulos indicados abaixo, em inglês e
minúsculas.

Pode usar Google Translate ou dicionário para compreender o inglês histórico.
Para reduzir a influência do tradutor, substitua `[graft]` ou `[tree]` por
`[PALAVRA]` antes de traduzir. Não pergunte ao tradutor, chatbot ou mecanismo
de busca qual é o sentido correto.

Não consulte previsões do ConSeC, rótulos heurísticos ou resultados anteriores
durante a anotação.

## Como classificar `[graft]`

### `corruption`

Use quando significa vantagem ilícita, suborno, propina ou dinheiro obtido por
abuso de poder.

Pistas comuns:

```text
bribe, corruption, police, government, payment, scandal, political
```

Exemplo inventado:

```text
the official receive money through [graft] and bribery
```

### `medical`

Use quando significa tecido, pele, osso, órgão, vaso ou material transplantado
ou implantado no corpo.

Pistas comuns:

```text
skin, bone, tissue, transplant, surgery, patient, artery, kidney
```

Exemplo inventado:

```text
the surgeon place a skin [graft] over the wound
```

### `botanical`

Use quando se refere a enxertia de plantas, à muda ou parte vegetal inserida
em outra planta, ao scion, stock, broto ou ramo enxertado.

Pistas comuns:

```text
tree, stock, scion, bud, branch, bark, orchard, fruit
```

Exemplo inventado:

```text
the gardener insert the [graft] into the stock
```

### `other`

Use quando o sentido está compreensível, mas não pertence às três categorias
anteriores. Por exemplo, uma junção ou acréscimo não médico e não botânico.

### `unclear`

Use quando o trecho não fornece informação suficiente ou permanece ambíguo
mesmo após tradução. Não use `unclear` apenas porque o inglês está estranho;
use-o quando realmente não for possível decidir o sentido.

## Como classificar `[tree]`

### `plant`

Use para uma árvore literal: planta lenhosa, tronco, galhos, folhas, raízes,
frutos, floresta ou partes físicas de uma árvore.

Exemplo inventado:

```text
the bird build its nest in the [tree]
```

Expressões como `tree stump`, `tree trunk` e `fruit tree` também são `plant`.

### `diagram`

Use para uma estrutura abstrata que se ramifica a partir de uma origem. Inclui
árvore genealógica, árvore familiar, árvore evolutiva, árvore sintática,
árvore de decisão e diagramas de diretórios.

Exemplo inventado:

```text
her ancestors appear in the family [tree]
```

Mesmo sem um desenho visível, `family tree` é `diagram`, não `plant`.

### `person`

Use somente quando `Tree` for sobrenome ou nome de uma pessoa, especialmente
o ator e produtor teatral Sir Herbert Beerbohm Tree. Não use `person` apenas
porque há uma pessoa perto de uma árvore literal.

### `other`

Use quando o sentido está claro, mas não corresponde a planta, estrutura
ramificada ou pessoa.

### `unclear`

Use quando não houver contexto suficiente para decidir ou quando duas
interpretações continuarem igualmente plausíveis.

## Confiança

Preencha `confidence` com:

- `high`: o contexto torna o sentido praticamente inequívoco;
- `medium`: há uma interpretação claramente mais provável, mas sobra dúvida;
- `low`: decisão frágil; outra interpretação ainda parece possível.

Se usar `unclear`, normalmente a confiança deve ser `medium` ou `high` na
avaliação de que o contexto é insuficiente. `low` significa que você escolheu
um rótulo, mas com pouca segurança.

## Notas

O campo `notes` é opcional. Use-o para registrar:

- dúvida entre dois rótulos;
- palavra ou construção difícil;
- necessidade de tradução;
- motivo para `other` ou `unclear`.

Não é necessário registrar notas nos casos evidentes.

## Exemplo de preenchimento

```csv
item_id,target,context,label,confidence,notes
G2-999,graft_nn,the surgeon use a skin [graft],medical,high,
G2-998,tree_nn,the names appear on the family [tree],diagram,high,
```

## Antes de terminar

Confira se:

1. todas as 87 linhas têm `label`;
2. todos os rótulos usam a grafia exata deste guia;
3. toda linha possui `confidence`;
4. nenhuma célula de contexto ou identificação foi modificada.
"""


def prepare_audit_package(
    candidates: list[dict[str, str]], annotation_dir: Path, seed: int
) -> int:
    audit_rows = [
        row
        for row in candidates
        if (
            row["target"] == "graft_nn"
            and row["heuristic_sense"] in {"corruption", "medical"}
        )
        or (
            row["target"] == "tree_nn"
            and row["heuristic_sense"] == "diagram"
        )
    ]
    for period in ("1810-1860", "1960-2010"):
        plants = [
            row
            for row in candidates
            if row["corpus"] == period
            and row["target"] == "tree_nn"
            and row["heuristic_sense"] == "plant"
        ]
        random.Random(f"{seed}:audit:{period}:tree:plant").shuffle(plants)
        audit_rows.extend(plants[:15])

    random.Random(f"{seed}:audit-order").shuffle(audit_rows)
    manifest = []
    form = []
    for index, row in enumerate(audit_rows, start=1):
        item_id = f"G2-{index:03d}"
        manifest.append(
            {
                "item_id": item_id,
                "sample_id": row["sample_id"],
                "corpus": row["corpus"],
                "target": row["target"],
                "context": row["context"],
            }
        )
        form.append(
            {
                "item_id": item_id,
                "target": row["target"],
                "context": row["context"],
                "label": "",
                "confidence": "",
                "notes": "",
            }
        )

    annotation_dir.mkdir(parents=True, exist_ok=True)
    (annotation_dir / "README_ANNOTATOR.md").write_text(
        AUDIT_INSTRUCTIONS, encoding="utf-8"
    )
    with (annotation_dir / "manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    with (annotation_dir / "annotator_a.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(form)
    return len(form)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--annotation-dir", type=Path)
    parser.add_argument("--seed", type=int, default=20260613)
    args = parser.parse_args()

    candidates, summary = collect_candidates(args.corpus_dir, args.seed)
    summary["corpus_sha256"] = {
        path.name: file_sha256(path)
        for path in sorted(args.corpus_dir.glob("*.txt"))
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "candidates.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    if args.annotation_dir:
        summary["audit_items"] = prepare_audit_package(
            candidates, args.annotation_dir, args.seed
        )
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
