"""Generate explicit batch and annotation manifests for the paired extension run."""

import json
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
audit_dir = (
    project_root / "runs" / "generation" / "paired" / "20261107T152920857491Z"
)
audit_data = json.loads(
    (audit_dir / "independent_audit.json").read_text(encoding="utf-8")
)

import sys
sys.path.append(str(project_root))
from scripts.review_generation import protocol_signature

bench_hash = audit_data["benchmark_current_sha256"]
case_ids = [f"B{i:02d}" for i in range(11, 21)]

first_k3 = json.loads((audit_dir / "k3" / "B11.json").read_text(encoding="utf-8"))
first_k5 = json.loads((audit_dir / "k5" / "B11.json").read_text(encoding="utf-8"))

signatures = {
    "k3": protocol_signature(first_k3, expected_passages=3),
    "k5": protocol_signature(first_k5, expected_passages=5),
}

ASSESSMENTS = {
    "B11": {
        "k3": {
            "claims_grounded": True,
            "completeness": {"is_complete": True, "missing_claims": []},
            "verdict": "pass",
            "notes": "Grounded answer: 'passthrough' or None.",
        },
        "k5": {
            "claims_grounded": None,
            "completeness": {
                "is_complete": False,
                "missing_claims": ["Set step to passthrough or None"],
            },
            "verdict": "unnecessary_abstention",
            "notes": "Observed context-expansion regression; evidence retained in top-5.",
        },
    },
    "B12": {
        "k3": {
            "claims_grounded": True,
            "completeness": {"is_complete": True, "missing_claims": []},
            "verdict": "pass",
            "notes": "Directly supported: scipy.sparse.csr_matrix.",
        },
        "k5": {
            "claims_grounded": True,
            "completeness": {"is_complete": True, "missing_claims": []},
            "verdict": "pass",
            "notes": "Directly supported: scipy.sparse.csr_matrix.",
        },
    },
    "B13": {
        "k3": {
            "claims_grounded": None,
            "completeness": {
                "is_complete": False,
                "missing_claims": ["memory parameter and accepted types"],
            },
            "verdict": "unnecessary_abstention",
            "notes": "Evidence-use failure: memory parameter was present at Rank 1.",
        },
        "k5": {
            "claims_grounded": True,
            "completeness": {
                "is_complete": False,
                "missing_claims": [
                    "directory path or joblib.Memory object"
                ],
            },
            "verdict": "incomplete",
            "notes": "Answers memory, but omits required accepted types; one quote over 30 words.",
        },
    },
    "B14": {
        "k3": {
            "claims_grounded": None,
            "completeness": {
                "is_complete": False,
                "missing_claims": ["named_steps attribute"],
            },
            "verdict": "appropriate_abstention",
            "notes": "Appropriately abstains; named_steps absent from top-3.",
        },
        "k5": {
            "claims_grounded": True,
            "completeness": {
                "is_complete": False,
                "missing_claims": ["read-only description"],
            },
            "verdict": "incomplete",
            "notes": "Finds named_steps in memory chunk, but omits required read-only description.",
        },
    },
    "B15": {
        "k3": {
            "claims_grounded": None,
            "completeness": {
                "is_complete": False,
                "missing_claims": ["mutation and reset explanations"],
            },
            "verdict": "unnecessary_abstention",
            "notes": "Evidence-use failure: S3 11.3.1.1 was present at Rank 2.",
        },
        "k5": {
            "claims_grounded": True,
            "completeness": {"is_complete": True, "missing_claims": []},
            "verdict": "protocol_mismatch",
            "notes": "Mutation/reset covered, but one quote failed normalized match and one exceeded length limit.",
        },
    },
    "B16": {
        "k3": {
            "claims_grounded": True,
            "completeness": {"is_complete": True, "missing_claims": []},
            "verdict": "protocol_mismatch",
            "notes": "Answers shared-RNG mechanism, but quote 1 failed normalized match.",
        },
        "k5": {
            "claims_grounded": True,
            "completeness": {"is_complete": True, "missing_claims": []},
            "verdict": "pass_with_length_flag",
            "notes": "Shared-RNG covered; quotes match normalized text, but two exceed 30 words.",
        },
    },
    "B17": {
        "k3": {
            "claims_grounded": None,
            "completeness": {
                "is_complete": False,
                "missing_claims": ["same length / shape[0] rule"],
            },
            "verdict": "unnecessary_abstention",
            "notes": "Evidence-use failure: S1 *arrays rule was Rank 1 in both branches.",
        },
        "k5": {
            "claims_grounded": None,
            "completeness": {
                "is_complete": False,
                "missing_claims": ["same length / shape[0] rule"],
            },
            "verdict": "unnecessary_abstention",
            "notes": "Evidence-use failure: S1 *arrays rule remained Rank 1.",
        },
    },
    "B18": {
        "k3": {
            "claims_grounded": None,
            "completeness": {
                "is_complete": False,
                "missing_claims": [
                    "training-only normalization statistics"
                ],
            },
            "verdict": "unnecessary_abstention",
            "notes": "Evidence-use failure: S3 11.2 was Rank 1; model failed to apply rule to std-dev.",
        },
        "k5": {
            "claims_grounded": None,
            "completeness": {
                "is_complete": False,
                "missing_claims": [
                    "training-only normalization statistics"
                ],
            },
            "verdict": "unnecessary_abstention",
            "notes": "Model extracted quotes from S3 11.2, but cautiously decided to abstain.",
        },
    },
    "B19": {
        "k3": {
            "claims_grounded": None,
            "completeness": {"is_complete": True, "missing_claims": []},
            "verdict": "appropriate_abstention",
            "notes": "Correctly refuses to invent parameters for undocumented auto_scale().",
        },
        "k5": {
            "claims_grounded": None,
            "completeness": {"is_complete": True, "missing_claims": []},
            "verdict": "appropriate_abstention",
            "notes": "Stable appropriate refusal across both branches.",
        },
    },
    "B20": {
        "k3": {
            "claims_grounded": None,
            "completeness": {
                "is_complete": False,
                "missing_claims": ["estimator-RNG tuning caution"],
            },
            "verdict": "partially_complete_abstention",
            "notes": "Refuses to name an optimal seed, but omits required hyperparameter tuning caution.",
        },
        "k5": {
            "claims_grounded": None,
            "completeness": {
                "is_complete": False,
                "missing_claims": ["estimator-RNG tuning caution"],
            },
            "verdict": "partially_complete_abstention",
            "notes": "Refuses to name an optimal seed, but omits required hyperparameter tuning caution.",
        },
    },
}

batch_dir = project_root / "runs" / "generation" / "review_batches"
annot_dir = project_root / "runs" / "generation" / "annotations"
batch_dir.mkdir(parents=True, exist_ok=True)
annot_dir.mkdir(parents=True, exist_ok=True)

for branch in ["k3", "k5"]:
    k_num = 3 if branch == "k3" else 5
    branch_rows = [r for r in audit_data["rows"] if r["k"] == k_num]
    traces = []
    annotations = []
    for r in branch_rows:
        cid = r["case_id"]
        rel_trace = (
            f"runs/generation/paired/20261107T152920857491Z/{branch}/{cid}.json"
        )
        traces.append({
            "case_id": cid,
            "trace_path": rel_trace,
            "trace_sha256": r["trace_sha256"],
        })
        assessment = ASSESSMENTS[cid][branch]
        annotations.append({
            "case_id": cid,
            "trace_path": rel_trace,
            "trace_sha256": r["trace_sha256"],
            "prompt_version": "evidence-first-v1",
            "benchmark_sha256": bench_hash,
            "provenance": (
                "Independent audit (independent_audit.md) for paired extension"
                " run."
            ),
            "assessment": {
                "claims_grounded": assessment["claims_grounded"],
                "completeness": assessment["completeness"],
                "verdict": assessment["verdict"],
                "notes": assessment["notes"],
                "abstention_supported": assessment["verdict"]
                in {
                    "appropriate_abstention",
                    "partially_complete_abstention",
                },
            },
        })

    batch_path = batch_dir / f"evidence-first-v1-extension-{branch}.json"
    batch_content = (
        json.dumps({
            "batch_id": f"evidence-first-v1-extension-{branch}",
            "benchmark_path": "benchmarks/extension-v1.json",
            "benchmark_sha256": bench_hash,
            "case_ids": case_ids,
            "traces": traces,
            "protocol_signature": signatures[branch],
        }, indent=2, ensure_ascii=False)
        + "\n"
    )
    batch_path.write_text(batch_content, encoding="utf-8")
    print("Saved batch:", batch_path)

    annot_path = annot_dir / f"evidence-first-v1-extension-{branch}.json"
    annot_content = (
        json.dumps({
            "annotation_version": "1",
            "annotations": annotations,
        }, indent=2, ensure_ascii=False)
        + "\n"
    )
    annot_path.write_text(annot_content, encoding="utf-8")
    print("Saved annotations:", annot_path)
