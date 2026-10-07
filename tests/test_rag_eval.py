"""Tests for RAG evaluation workbench."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import run_experiment


class RagEvalTests(unittest.TestCase):
    def test_candidate_quality(self):
        report = run_experiment(
            ROOT / "data" / "documents.json",
            ROOT / "data" / "golden.json",
            k=3,
        )
        self.assertGreaterEqual(report["candidate"]["mean_recall"], 0.8)
        self.assertTrue(report["candidate"]["cases"][-1]["abstained_correctly"])
        self.assertGreaterEqual(report["recall_delta"], 0)

    def test_k1_delta_zero(self):
        report = run_experiment(
            ROOT / "data" / "documents.json",
            ROOT / "data" / "golden.json",
            k=1,
        )
        self.assertEqual(report["recall_delta"], 0.0)

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            run_experiment(
                ROOT / "data" / "documents.json",
                ROOT / "data" / "golden.json",
                k=0,
            )


if __name__ == "__main__":
    unittest.main()
