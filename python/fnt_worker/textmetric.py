"""Text-similarity metrics for answers that have no exact target.

Mirrors `lib/textmetric.mjs` in the host — the two must agree, because the panel
shows a number the worker computed while the unit tests pin the JS version. The
shared behaviour is:

    coverage  — how much of the reference's content the output reaches, capped by
                how much the output could physically cover (a 400-char reply
                cannot reach every token of a 2000-char reference; demanding it
                measures answer length rather than relevance)
    precision — how much of the output is grounded in the reference
    shape     — whether the output has the structure the reference has
    f1        — 2·P·R/(P+R), the headline, because it punishes both failure modes

Calibration kept in both implementations (measured on real M0 outputs):
    identical answer -> 1.00, truncated reference -> 1.00,
    unrelated fluent text -> 0.00, empty -> 0.00

This module exists because the first attempt scored prose as "non-empty passes",
which reported 100% for both the base and the fine-tuned model and proved nothing.
"""

from __future__ import annotations

import re
from typing import Any

_CJK = re.compile(r"[\u4e00-\u9fff]")
_LATIN = re.compile(r"[a-z0-9_]{2,}")


def content_tokens(text: str) -> set[str]:
    """Content tokens: latin words plus CJK bigrams."""
    value = str(text or "").lower()
    tokens = set(_LATIN.findall(value))
    cjk = _CJK.findall(value)
    tokens.update(cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1))
    return tokens


def shape_of(text: str) -> dict[str, int | bool]:
    value = str(text or "")
    return {
        "code_fence": len(re.findall(r"```", value)),
        "bullets": len(re.findall(r"^\s*(?:[-*+]|\d+\.)\s+", value, re.M)),
        "headings": len(re.findall(r"^#{1,6}\s+", value, re.M)),
        "has_tool_json": bool(re.search(r'\{"tool"\s*:|"tool"\s*:', value)),
    }


def score_text_answer(output: str, reference: str) -> dict[str, Any]:
    out = str(output or "")
    ref = str(reference or "")
    reasons: list[str] = []

    if out.strip() == "":
        return {"coverage": 0.0, "precision": 0.0, "f1": 0.0, "shape": 0.0, "reasons": ["empty answer"]}

    out_tokens = content_tokens(out)
    ref_tokens = content_tokens(ref)

    reachable = 1.0 if not ref_tokens else min(1.0, len(out_tokens) / len(ref_tokens))
    hit = sum(1 for token in ref_tokens if token in out_tokens)
    raw_coverage = 1.0 if not ref_tokens else hit / len(ref_tokens)
    coverage = 0.0 if reachable == 0 else min(1.0, raw_coverage / reachable)
    if coverage < 0.25:
        reasons.append(f"reaches {raw_coverage:.0%} of the reference's content")

    grounded = sum(1 for token in out_tokens if token in ref_tokens)
    precision = 0.0 if not out_tokens else grounded / len(out_tokens)
    if precision < 0.3:
        reasons.append(f"only {precision:.0%} of its own content comes from the reference")

    want = shape_of(ref)
    got = shape_of(out)
    features: list[int] = []
    if want["code_fence"]:
        features.append(1 if got["code_fence"] else 0)
    if want["bullets"]:
        features.append(1 if got["bullets"] else 0)
    if want["headings"]:
        features.append(1 if got["headings"] else 0)
    if want["has_tool_json"]:
        features.append(1 if got["has_tool_json"] else 0)
    ratio = 1.0 if not ref else len(out) / len(ref)
    features.append(1 if 0.25 <= ratio <= 4 else 0)
    if len(features) > 1 and features[-1] == 0:
        reasons.append(f"answer length is {ratio:.2f}x the reference")
    shape = 1.0 if not features else sum(features) / len(features)

    f1 = 0.0 if coverage + precision == 0 else (2 * coverage * precision) / (coverage + precision)
    return {
        "coverage": round(coverage, 3),
        "precision": round(precision, 3),
        "f1": round(f1, 3),
        "shape": round(shape, 3),
        "reasons": reasons,
    }


def score_text_split(cases: list[dict[str, Any]], outputs: list[str]) -> dict[str, Any]:
    """Score a split, keeping the per-case detail.

    The floor is 0.15 and it is calibrated, not chosen: identical answers score 1.00
    and unrelated fluent text scores 0.00, while real 0.5B outputs land near 0.20.
    An earlier 0.4 floor rejected every real answer and reported 0% on-topic for
    both models, which reads as a broken metric rather than the honest "neither
    model answers these questions well" it actually was.
    """
    per_case = []
    for index, case in enumerate(cases):
        row = score_text_answer(outputs[index] if index < len(outputs) else "", case.get("expected", ""))
        per_case.append({"index": index, **row})

    def mean(key: str) -> float:
        if not per_case:
            return 0.0
        return round(sum(row[key] for row in per_case) / len(per_case), 3)

    on_topic = 0 if not per_case else sum(1 for row in per_case if row["f1"] >= 0.15) / len(per_case)
    return {
        "cases": len(per_case),
        "f1": mean("f1"),
        "coverage": mean("coverage"),
        "precision": mean("precision"),
        "shape": mean("shape"),
        "onTopicRate": round(on_topic, 3),
        "perCase": per_case,
    }
