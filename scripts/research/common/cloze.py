"""Keyword-based sense classification and cloze-context extraction helpers."""

from __future__ import annotations

GRAFT_SENSE_KEYWORDS = {
    "botanical": {
        "apple",
        "bark",
        "branch",
        "bud",
        "cleft",
        "fruit",
        "grafting",
        "orchard",
        "plant",
        "sap",
        "scion",
        "shoot",
        "stock",
        "tree",
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
        "marrow",
        "patient",
        "skin",
        "surgeon",
        "surgery",
        "transplant",
    },
}


SENSE_KEYWORDS = {
    "geometry": {
        "angle",
        "axis",
        "curve",
        "degree",
        "geometric",
        "geometry",
        "horizontal",
        "inclined",
        "line",
        "parallel",
        "perpendicular",
        "point",
        "projection",
        "surface",
        "vertical",
    },
    "aircraft": {
        "aircraft",
        "airline",
        "airport",
        "aboard",
        "bomber",
        "cockpit",
        "crew",
        "flight",
        "fly",
        "flying",
        "landing",
        "passenger",
        "pilot",
        "runway",
    },
    "tool": {
        "bead",
        "blade",
        "carpenter",
        "carpentry",
        "chisel",
        "gouge",
        "groove",
        "joiner",
        "mould",
        "moulding",
        "plough",
        "shave",
        "smooth",
        "timber",
        "tool",
        "wood",
    },
}


def high_confidence_sense(tokens: list[str]) -> str:
    scores = {
        sense: len(set(tokens) & keywords)
        for sense, keywords in SENSE_KEYWORDS.items()
    }
    best = max(scores.values())
    winners = [sense for sense, score in scores.items() if score == best and score > 0]
    return winners[0] if len(winners) == 1 else "unlabeled"


def classify_context(target: str, tokens: list[str]) -> str:
    if target != "graft_nn":
        return "unlabeled"
    scores = {
        sense: len(set(tokens) & keywords)
        for sense, keywords in GRAFT_SENSE_KEYWORDS.items()
    }
    best_score = max(scores.values())
    if best_score == 0:
        return "other"
    winners = [sense for sense, score in scores.items() if score == best_score]
    return winners[0] if len(winners) == 1 else "ambiguous"


def occurrence_contexts(corpus, target: str, seq_len: int) -> list[dict]:
    contexts = []
    content_len = seq_len - 2
    left_budget = content_len // 2
    for document in corpus.documents:
        for token_index, token in enumerate(document):
            if token != target:
                continue
            start = max(0, min(token_index - left_budget, len(document) - content_len))
            end = min(len(document), start + content_len)
            window = list(document[start:end])
            relative = token_index - start
            display = list(window)
            display[relative] = "[MASK]"
            contexts.append(
                {
                    "tokens": window,
                    "display": " ".join(display),
                    "sense": classify_context(target, window),
                }
            )
    return contexts
