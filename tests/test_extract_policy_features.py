"""Tests for provenance-safe feature extraction in scripts/extract_policy_features.py."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import extract_policy_features as epf
from scripts import review_generation as review


class TestExtractPolicyFeatures(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="model-crash-features-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.batch = copy.deepcopy(
            review.read_json(ROOT / "runs/generation/review_batches/evidence-first-v1-extension-v2-k3.json")
        )
        self.annotations = copy.deepcopy(
            review.read_json(ROOT / "runs/generation/annotations/evidence-first-v1-extension-v2-k3.json")
        )
        paths = [self.batch["benchmark_path"], *[e["trace_path"] for e in self.batch["traces"]]]
        for rel in paths:
            target = self.root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / rel).read_bytes())

    def test_valid_batch_exports_expected_features_and_preserves_invalid_output(self):
        rows, _ = epf.extract_batch_features(self.root, self.batch, self.annotations)
        self.assertEqual(len(rows), 10)
        by_id = {r["case_id"]: r for r in rows}

        # Empty quotes on B21 abstain must be None, not 1.0
        self.assertFalse(by_id["B21"]["has_quotes"])
        self.assertIsNone(by_id["B21"]["quote_fidelity_ratio"])
        self.assertEqual(by_id["B21"]["label_verdict"], "unnecessary_abstention")

        # B28 k3 quote failure keeps effective invalid_output verdict
        self.assertEqual(by_id["B28"]["label_verdict"], "invalid_output")
        self.assertGreater(by_id["B28"]["automatic_error_count"], 0)
        self.assertEqual(by_id["B28"]["quote_fidelity_ratio"], 0.5)

        # Punctuation should not inflate OOV (Pipeline.set_output() tokens are in vocab)
        self.assertLess(by_id["B21"]["query_oov_ratio"], 0.3)
        self.assertTrue(all(r["passages_match_chunks_artifact"] for r in rows))

    def test_trace_hash_mismatch_is_rejected(self):
        entry = self.batch["traces"][0]
        trace_path = self.root / entry["trace_path"]
        trace = review.read_json(trace_path)
        reply = json.loads(trace["response"]["message"]["content"])
        reply["abstention_reason"] = "Modified without re-review."
        trace["response"]["message"]["content"] = json.dumps(reply)
        trace_path.write_text(json.dumps(trace), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Trace hash mismatch"):
            epf.extract_batch_features(self.root, self.batch, self.annotations)

    def test_stale_annotation_hash_is_rejected(self):
        self.annotations["annotations"][0]["trace_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Stale annotation"):
            epf.extract_batch_features(self.root, self.batch, self.annotations)

    def test_missing_tokens_are_none_not_zero(self):
        trace_path = self.root / self.batch["traces"][0]["trace_path"]
        trace = review.read_json(trace_path)
        del trace["response"]["prompt_eval_count"]
        del trace["response"]["eval_count"]
        features = epf.extract_runtime_features(trace)
        self.assertIsNone(features["input_tokens"])
        self.assertIsNone(features["output_tokens"])

    def test_malformed_json_response_handled_gracefully(self):
        trace_path = self.root / self.batch["traces"][0]["trace_path"]
        trace = review.read_json(trace_path)
        trace["response"]["message"]["content"] = "{not valid json"
        features = epf.extract_runtime_features(trace)
        self.assertFalse(features["response_parse_valid"])
        self.assertIsNone(features["model_decision"])
        self.assertFalse(features["has_quotes"])
        self.assertIsNone(features["quote_fidelity_ratio"])


if __name__ == "__main__":
    unittest.main()
