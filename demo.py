"""End-to-end demo for RAG Evaluation Workbench."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluate import run_experiment

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG retrieval evaluation demo")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--tenant", default="A")
    args = parser.parse_args()

    report = run_experiment(
        ROOT / "data" / "documents.json",
        ROOT / "data" / "golden.json",
        k=args.k,
        tenant=args.tenant,
        output_path=ROOT / "state" / "evaluation.json",
    )
    print(json.dumps(report, indent=2))
    print()
    print(
        f"Summary: baseline@1 recall={report['baseline']['mean_recall']:.2f} "
        f"MRR={report['baseline']['mrr']:.2f} | candidate@{args.k} "
        f"recall={report['candidate']['mean_recall']:.2f} "
        f"MRR={report['candidate']['mrr']:.2f} | delta={report['recall_delta']:.2f}"
    )
    last = report["candidate"]["cases"][-1]
    print(
        f"Unanswerable case '{last['query']}': retrieved={last['retrieved']} "
        f"abstained_correctly={last['abstained_correctly']}"
    )


if __name__ == "__main__":
    main()
