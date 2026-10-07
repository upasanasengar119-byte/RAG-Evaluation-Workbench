"""Paired retrieval evaluation on a golden question set."""
from __future__ import annotations

import json
from pathlib import Path

from .search import Search


def evaluate(documents: list[dict], golden: list[dict], k: int) -> dict:
    search = Search(documents)
    results = []
    for case in golden:
        hits = [h for h in search.query(case["query"], k) if h["score"] >= 0.1]
        ids = [h["id"] for h in hits]
        relevant = set(case["relevant"])
        found = len(set(ids) & relevant)
        reciprocal = next(
            (1 / rank for rank, item in enumerate(ids, 1) if item in relevant),
            0,
        )
        results.append(
            {
                "query": case["query"],
                "retrieved": ids,
                "recall": (found / len(relevant)) if relevant else None,
                "precision": (found / len(ids)) if ids else 0.0,
                "rr": reciprocal if relevant else None,
                "abstained_correctly": (not ids) if not relevant else None,
            }
        )
    answerable = [row for row in results if row["recall"] is not None]
    return {
        "k": k,
        "mean_recall": sum(row["recall"] for row in answerable) / len(answerable),
        "mrr": sum(row["rr"] for row in answerable) / len(answerable),
        "cases": results,
    }


def run_experiment(
    documents_path: Path,
    golden_path: Path,
    k: int = 3,
    tenant: str = "A",
    output_path: Path | None = None,
) -> dict:
    if type(k) is not int or not 1 <= k <= 5:
        raise ValueError("k must be an integer between 1 and 5")
    documents = [
        doc
        for doc in json.loads(documents_path.read_text(encoding="utf-8"))
        if doc["tenant"] == tenant
    ]
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    available = {doc["id"] for doc in documents}
    if any(not set(case["relevant"]) <= available for case in golden):
        raise ValueError("golden evidence is not available to this tenant")
    baseline = evaluate(documents, golden, 1)
    candidate = evaluate(documents, golden, k)
    report = {
        "baseline": baseline,
        "candidate": candidate,
        "recall_delta": candidate["mean_recall"] - baseline["mean_recall"],
        "scope": "retrieval only; does not score generated-answer correctness",
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
