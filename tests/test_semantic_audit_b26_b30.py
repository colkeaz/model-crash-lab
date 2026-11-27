"""Verification test for model-crash-lab/semantic-audits/B26-B30.json."""

import hashlib
import json
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_FILE = PROJECT_ROOT / "semantic-audits" / "B26-B30.json"
BENCHMARK_FILE = PROJECT_ROOT / "benchmarks" / "extension-v2.json"
EXPECTED_PAIRS = {(f"B{i}", k) for i in range(26, 31) for k in (3, 5)}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


class TestSemanticAuditB26B30Integrity(unittest.TestCase):
    def test_semantic_audit_integrity(self):
        self.assertTrue(AUDIT_FILE.exists(), f"Audit file not found: {AUDIT_FILE}")
        self.assertTrue(BENCHMARK_FILE.exists(), f"Benchmark file not found: {BENCHMARK_FILE}")

        benchmark_hash = sha256_file(BENCHMARK_FILE)
        benchmark_data = json.loads(BENCHMARK_FILE.read_text(encoding="utf-8"))
        benchmark_cases = {case["id"]: case for case in benchmark_data}

        audit_entries = json.loads(AUDIT_FILE.read_text(encoding="utf-8"))
        self.assertIsInstance(audit_entries, list)
        self.assertEqual(len(audit_entries), 10)

        observed_pairs = set()

        for idx, entry in enumerate(audit_entries):
            case_id = entry.get("case_id")
            trace_path_str = entry.get("trace_path")
            self.assertIn(case_id, benchmark_cases, f"Entry {idx}: Unknown case_id {case_id}")
            self.assertTrue(trace_path_str, f"Entry {idx}: Missing trace_path")

            trace_path = PROJECT_ROOT / trace_path_str
            self.assertTrue(trace_path.exists(), f"Entry {idx}: Trace file not found: {trace_path}")

            self.assertEqual(entry.get("trace_sha256", "").upper(), sha256_file(trace_path))
            self.assertEqual(entry.get("benchmark_sha256", "").upper(), benchmark_hash)

            trace_data = json.loads(trace_path.read_text(encoding="utf-8"))
            k = trace_data.get("k")
            self.assertIn(k, (3, 5))
            pair_key = (case_id, k)
            self.assertNotIn(pair_key, observed_pairs)
            observed_pairs.add(pair_key)

            assessment = entry.get("assessment")
            self.assertIsInstance(assessment, dict)
            decision = assessment.get("decision")
            self.assertIn(decision, ("answer", "abstain"))
            saved_response = json.loads(trace_data["response"]["message"]["content"])
            self.assertEqual(decision, saved_response.get("decision"))

            benchmark_req_claims = benchmark_cases[case_id]["required_claims"]
            audited_claims = [rc["claim"] for rc in assessment.get("required_claims", [])]
            self.assertEqual(audited_claims, benchmark_req_claims)

            for rc in assessment.get("required_claims", []):
                self.assertIn(rc["status"], ("supported", "missing", "unsupported"))
                self.assertIsInstance(rc.get("note"), str)
                self.assertGreater(len(rc["note"].strip()), 0)

            self.assertIsNone(entry.get("historical_benchmark_hash"))

            claims_grounded = assessment.get("claims_grounded")
            abstention_supported = assessment.get("abstention_supported")
            protocol_valid = assessment.get("protocol_valid")
            response_schema_valid = assessment.get("response_schema_and_quote_occurrence_valid")
            quote_len_valid = assessment.get("quote_length_instruction_valid")

            self.assertIsInstance(protocol_valid, bool)
            self.assertIsInstance(response_schema_valid, bool)
            self.assertIsInstance(quote_len_valid, bool)

            if pair_key == ("B28", 3):
                self.assertFalse(response_schema_valid)
                self.assertFalse(quote_len_valid)
                self.assertFalse(protocol_valid)
            else:
                self.assertTrue(response_schema_valid)
                self.assertTrue(quote_len_valid)
                self.assertTrue(protocol_valid)

            if decision == "abstain":
                self.assertIsNone(claims_grounded)
                self.assertIsInstance(abstention_supported, bool)
            else:
                self.assertIsInstance(claims_grounded, bool)
                self.assertIsNone(abstention_supported)

            self.assertIsInstance(assessment.get("notes"), str)
            self.assertGreater(len(assessment["notes"].strip()), 0)

        self.assertEqual(observed_pairs, EXPECTED_PAIRS)


if __name__ == "__main__":
    unittest.main()
