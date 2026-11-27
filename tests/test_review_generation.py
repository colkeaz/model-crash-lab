"""Regression checks for quote provenance and trace-bound manual reviews."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "review_generation", ROOT / "scripts" / "review_generation.py"
)
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


class QuoteTests(unittest.TestCase):
    def test_exact_and_normalized_matches_are_distinct(self):
        result = review.verify_quote("in", "alpha beta", {"in": "alpha\n beta"})
        self.assertFalse(result["exact_match"])
        self.assertTrue(result["normalized_match"])

    def test_empty_and_whitespace_quotes_are_rejected(self):
        for quote in ("", " \n\t"):
            with self.subTest(quote=quote):
                result = review.verify_quote("in", quote, {"in": "some text"})
                self.assertFalse(result["normalized_match"])
                self.assertFalse(result["exact_match"])

    def test_quote_from_unretrieved_source_is_rejected(self):
        result = review.verify_quote("elsewhere", "real quote", {"supplied": "real quote"})
        self.assertFalse(result["in_supplied_passages"])
        self.assertFalse(result["normalized_match"])

    def test_under_30_means_strictly_less_than_30(self):
        for count in (29, 30, 31):
            with self.subTest(count=count):
                quote = " ".join(["word"] * count)
                result = review.verify_quote("in", quote, {"in": quote})
                self.assertEqual(result["violates_under_30_words"], count >= 30)

    def test_wrong_quote_types_are_rejected(self):
        for quote in (None, 42, ["word"]):
            with self.subTest(quote=quote):
                self.assertFalse(review.verify_quote("in", quote, {"in": "word"})["normalized_match"])


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="model-crash-review-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.batch = copy.deepcopy(review.read_json(ROOT / review.DEFAULT_BATCH))
        self.annotations = copy.deepcopy(review.read_json(ROOT / review.DEFAULT_ANNOTATIONS))
        paths = [self.batch["benchmark_path"], *[e["trace_path"] for e in self.batch["traces"]]]
        for relative in paths:
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / relative).read_bytes())

    def run_audit(self):
        return review.audit_batch(self.root, self.batch, self.annotations)

    def first_trace(self):
        entry = self.batch["traces"][0]
        path = self.root / entry["trace_path"]
        return entry, path, review.read_json(path)

    def write_trace(self, path, trace):
        path.write_text(json.dumps(trace), encoding="utf-8")

    def test_real_batch_retains_results_and_reports_four_length_violations(self):
        report = self.run_audit()
        summary = report["summary"]
        self.assertEqual(summary["total_cases"], 10)
        self.assertEqual(report["expected_passages"], 3)
        self.assertEqual(summary["quotes_exact_matches"], 24)
        self.assertEqual(summary["quotes_violating_under_30_words"], 4)
        self.assertEqual(summary["verdicts"], {
            "pass": 4, "incomplete": 2, "appropriate_abstention": 3,
            "unsupported_extrapolation": 1,
        })
        b07 = next(c for c in report["cases"] if c["case_id"] == "B07")
        self.assertTrue(b07["completeness"]["is_complete"])
        self.assertFalse(b07["claim_grounding"])
        for case in report["cases"]:
            if case["decision"] == "abstain":
                self.assertIsNone(case["claim_grounding"])
                self.assertIsNone(case["completeness"])

    def test_changed_response_does_not_inherit_manual_pass(self):
        _, path, trace = self.first_trace()
        reply = json.loads(trace["response"]["message"]["content"])
        reply["answer"] = "A different answer"
        trace["response"]["message"]["content"] = json.dumps(reply)
        self.write_trace(path, trace)
        first = self.run_audit()["cases"][0]
        self.assertEqual(first["annotation_status"], "stale")
        self.assertEqual(first["verdict"], "unreviewed")
        self.assertIsNone(first["claim_grounding"])

    def test_new_trace_path_is_unreviewed_even_for_same_case(self):
        entry, old_path, _ = self.first_trace()
        new_relative = "runs/generation/new-run/B01.json"
        new_path = self.root / new_relative
        new_path.parent.mkdir(parents=True)
        new_path.write_bytes(old_path.read_bytes())
        entry["trace_path"] = new_relative
        first = self.run_audit()["cases"][0]
        self.assertEqual(first["annotation_status"], "unreviewed")
        self.assertEqual(first["verdict"], "unreviewed")

    def test_annotation_protocol_mismatch_is_stale(self):
        self.annotations["annotations"][0]["prompt_version"] = "other"
        self.assertEqual(self.run_audit()["cases"][0]["annotation_status"], "stale")

    def test_mixed_protocols_are_rejected(self):
        _, path, trace = self.first_trace()
        trace["prompt_version"] = "new-protocol"
        self.write_trace(path, trace)
        with self.assertRaisesRegex(ValueError, "mixes protocols"):
            self.run_audit()

    def test_changed_sampling_settings_are_rejected(self):
        _, path, trace = self.first_trace()
        trace["request"]["options"]["temperature"] = 1
        self.write_trace(path, trace)
        with self.assertRaisesRegex(ValueError, "mixes protocols"):
            self.run_audit()

    def test_changed_schema_is_rejected(self):
        _, path, trace = self.first_trace()
        ids = trace["request"]["format"]["properties"]["evidence"]["items"]["properties"]["chunk_id"]["enum"]
        del ids[1:]
        self.write_trace(path, trace)
        with self.assertRaisesRegex(ValueError, "mixes protocols"):
            self.run_audit()

    def test_benchmark_changes_cannot_reuse_annotations(self):
        path = self.root / self.batch["benchmark_path"]
        cases = review.read_json(path)
        cases[0]["required_claims"] = ["New grading requirement"]
        path.write_text(json.dumps(cases), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Benchmark changed"):
            self.run_audit()

    def test_no_quotes_is_not_reported_as_verbatim(self):
        _, _, trace = self.first_trace()
        trace["response"]["message"]["content"] = json.dumps({
            "evidence": [], "decision": "abstain", "answer": "",
            "abstention_reason": "Insufficient evidence",
        })
        reply, checks, errors = review.audit_trace(trace, review.request_passages(trace))
        self.assertEqual(reply["decision"], "abstain")
        self.assertEqual(checks, [])
        self.assertEqual(errors, [])
        entry, path, _ = self.first_trace()
        self.write_trace(path, trace)
        self.assertEqual(self.run_audit()["cases"][0]["quotation_fidelity"]["status"], "NO_QUOTES")

    def test_invalid_decision_is_not_counted_as_abstention(self):
        _, path, trace = self.first_trace()
        reply = json.loads(trace["response"]["message"]["content"])
        reply["decision"] = "maybe"
        trace["response"]["message"]["content"] = json.dumps(reply)
        self.write_trace(path, trace)
        report = self.run_audit()
        self.assertEqual(report["summary"]["abstentions"], 3)
        self.assertEqual(report["cases"][0]["verdict"], "invalid_output")

    def test_automatic_quote_errors_cannot_be_overridden_by_manual_pass(self):
        entry, path, trace = self.first_trace()
        reply = json.loads(trace["response"]["message"]["content"])
        reply["evidence"][0]["chunk_id"] = "unretrieved-source"
        trace["response"]["message"]["content"] = json.dumps(reply)
        self.write_trace(path, trace)
        digest = review.file_hash(path)
        entry["trace_sha256"] = digest
        self.annotations["annotations"][0]["trace_sha256"] = digest
        first = self.run_audit()["cases"][0]
        self.assertEqual(first["annotation_status"], "matched")
        self.assertEqual(first["manual_verdict"], "pass")
        self.assertEqual(first["verdict"], "invalid_output")

    def test_malformed_model_json_is_recorded_as_invalid(self):
        _, path, trace = self.first_trace()
        trace["response"]["message"]["content"] = '{"unfinished":'
        self.write_trace(path, trace)
        self.assertEqual(self.run_audit()["cases"][0]["verdict"], "invalid_output")

    def test_changed_model_finish_reason_is_invalid(self):
        _, _, trace = self.first_trace()
        trace["response"]["done_reason"] = "length"
        _, _, errors = review.audit_trace(trace, review.request_passages(trace))
        self.assertIn("Generation did not finish normally.", errors)

    def top_five_fixture(self):
        batch = review.read_json(
            ROOT / "runs/generation/review_batches/evidence-first-v1-top5-b06.json"
        )
        annotations = review.read_json(
            ROOT / "runs/generation/annotations/evidence-first-v1-top5-b06.json"
        )
        for entry in batch["traces"]:
            target = self.root / entry["trace_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / entry["trace_path"]).read_bytes())
        return batch, annotations

    def test_top_five_batch_uses_its_own_count_and_fresh_annotations(self):
        batch, annotations = self.top_five_fixture()
        report = review.audit_batch(self.root, batch, annotations)
        self.assertEqual(report["expected_passages"], 5)
        self.assertEqual(report["summary"]["total_cases"], 1)
        self.assertEqual(report["summary"]["invalid_outputs"], 0)
        self.assertEqual(report["cases"][0]["annotation_status"], "unreviewed")
        self.assertEqual(report["cases"][0]["verdict"], "unreviewed")

    def test_cli_count_cannot_override_batch_metadata(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            review.audit_batch(self.root, self.batch, self.annotations, expected_passages=5)

    def test_top_five_batch_rejects_missing_passage(self):
        batch, annotations = self.top_five_fixture()
        path = self.root / batch["traces"][0]["trace_path"]
        trace = review.read_json(path)
        payload = json.loads(trace["request"]["messages"][1]["content"])
        payload["passages"].pop()
        trace["request"]["messages"][1]["content"] = json.dumps(payload)
        self.write_trace(path, trace)
        with self.assertRaisesRegex(ValueError, "Expected 5 supplied passages, got 4"):
            review.audit_batch(self.root, batch, annotations)

    def test_full_top_five_audit_keeps_failed_attempt_and_remaining_cases(self):
        batch = review.read_json(
            ROOT / "runs/generation/review_batches/evidence-first-v1-top5-pilot.json"
        )
        annotations = review.read_json(
            ROOT / "runs/generation/annotations/evidence-first-v1-top5-pilot.json"
        )
        for entry in batch["traces"]:
            path = self.root / entry["trace_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / entry["trace_path"]).read_bytes())
        report = review.audit_batch(self.root, batch, annotations)
        self.assertEqual(report["summary"]["total_cases"], 10)
        self.assertEqual(report["summary"]["invalid_outputs"], 1)
        cases = {case["case_id"]: case for case in report["cases"]}
        self.assertEqual(cases["B05"]["verdict"], "invalid_output")
        self.assertEqual(cases["B05"]["quotation_fidelity"]["status"], "MISMATCH")
        self.assertEqual(cases["B09"]["decision"], "answer")
        self.assertEqual(cases["B10"]["decision"], "abstain")


if __name__ == "__main__":
    unittest.main()
