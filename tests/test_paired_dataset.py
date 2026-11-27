"""Tests for question-level paired dataset construction and compatibility checks."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_paired_dataset as bpd
from scripts import review_generation as review


class TestPairedDataset(unittest.TestCase):
    def _copy_suite_to_temp(self, suite_cfg):
        temp = tempfile.TemporaryDirectory(prefix="model-crash-paired-")
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        for key in ("k3_batch", "k3_annotations", "k5_batch", "k5_annotations"):
            src = ROOT / suite_cfg[key]
            dst = root / suite_cfg[key]
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())

        for bkey in ("k3_batch", "k5_batch"):
            batch = review.read_json(ROOT / suite_cfg[bkey])
            for rel in [batch["benchmark_path"], *[e["trace_path"] for e in batch["traces"]]]:
                dst = root / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes((ROOT / rel).read_bytes())
        return root

    def test_real_dataset_has_30_questions_and_no_k5_leakage(self):
        dataset = bpd.build_paired_dataset(ROOT)
        self.assertEqual(dataset["total_questions"], 30)
        self.assertEqual(dataset["eligible_questions"], 30)
        self.assertEqual(dataset["excluded_questions"], 0)
        # Historical hashes were not recorded in traces, so cohort must be exploratory_unknown_provenance
        self.assertEqual(
            dataset["cohorts"],
            {"exploratory_unknown_provenance": 30},
        )
        for row in dataset["rows"]:
            f_keys = set(row["k3_features"].keys())
            self.assertNotIn("family_id", f_keys)
            self.assertFalse(any("k5" in k or "label" in k for k in f_keys))
            self.assertEqual(row["provenance"]["compatibility"]["top3_prefix"], "matched")
            self.assertEqual(row["provenance"]["compatibility"]["historical_model_digest"], "unknown")
            if row["suite"] == "pilot":
                self.assertEqual(row["provenance"]["compatibility"]["historical_retrieval_metadata"], "matched")
                self.assertEqual(row["provenance"]["compatibility"]["historical_benchmark_hash"], "matched")
            else:
                self.assertEqual(row["provenance"]["compatibility"]["historical_retrieval_metadata"], "unknown")
                self.assertEqual(row["provenance"]["compatibility"]["historical_benchmark_hash"], "unknown")

    def test_prefix_mismatch_excludes_question(self):
        cfg = bpd.DEFAULT_SUITES[2]  # extension-v2
        root = self._copy_suite_to_temp(cfg)
        k5_batch = review.read_json(root / cfg["k5_batch"])
        k5_annots = review.read_json(root / cfg["k5_annotations"])

        # Swap first two passages in B21 k5 and re-sign hashes so audit passes but prefix fails
        entry = k5_batch["traces"][0]
        t_path = root / entry["trace_path"]
        trace = review.read_json(t_path)
        payload = json.loads(trace["request"]["messages"][1]["content"])
        payload["passages"][0], payload["passages"][1] = payload["passages"][1], payload["passages"][0]
        trace["request"]["messages"][1]["content"] = json.dumps(payload)
        t_path.write_text(json.dumps(trace), encoding="utf-8")

        new_hash = review.file_hash(t_path)
        entry["trace_sha256"] = new_hash
        k5_annots["annotations"][0]["trace_sha256"] = new_hash
        (root / cfg["k5_batch"]).write_text(json.dumps(k5_batch), encoding="utf-8")
        (root / cfg["k5_annotations"]).write_text(json.dumps(k5_annots), encoding="utf-8")

        rows = bpd.build_suite_pairs(root, cfg)
        b21 = next(r for r in rows if r["case_id"] == "B21")
        self.assertFalse(b21["provenance"]["eligible"])
        self.assertEqual(b21["provenance"]["cohort"], "excluded")
        self.assertIn("compatibility_mismatch:top3_prefix", b21["provenance"]["exclusion_reasons"])

    def test_stale_annotation_excludes_question(self):
        cfg = bpd.DEFAULT_SUITES[2]
        root = self._copy_suite_to_temp(cfg)
        k3_annots = review.read_json(root / cfg["k3_annotations"])
        k3_annots["annotations"][0]["trace_sha256"] = "0" * 64
        (root / cfg["k3_annotations"]).write_text(json.dumps(k3_annots), encoding="utf-8")

        rows = bpd.build_suite_pairs(root, cfg)
        b21 = next(r for r in rows if r["case_id"] == "B21")
        self.assertFalse(b21["provenance"]["eligible"])
        self.assertIn("k3_annotation_stale", b21["provenance"]["exclusion_reasons"])

    def test_missing_settings_marked_unknown_not_matched(self):
        case = {"question": "Q", "corpus_id": "C"}
        k3_trace = {
            "corpus_id": "C",
            "prompt_version": "v1",
            "request": {
                "model": "m",
                "messages": [
                    {"role": "system", "content": "S"},
                    {"role": "user", "content": json.dumps({"question": "Q", "passages": []})},
                ],
                "format": {},
            },
        }
        k5_trace = copy.deepcopy(k3_trace)
        compat = bpd.check_pair_compatibility(case, k3_trace, k5_trace, {}, "C")
        self.assertEqual(compat["settings"], "unknown")
        self.assertEqual(compat["historical_retrieval_metadata"], "unknown")


if __name__ == "__main__":
    unittest.main()
