import argparse
import json
from pathlib import Path
from bs4 import BeautifulSoup

project_root = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="Validate a benchmark JSON file.")
parser.add_argument(
    "--benchmark",
    default="benchmarks/pilot.json",
    help="Path to benchmark file relative to project root (default: benchmarks/pilot.json)",
)
args = parser.parse_args()

benchmark_path = project_root / args.benchmark

with benchmark_path.open(encoding="utf-8") as f:
    cases = json.load(f)

if not isinstance(cases, list):
    raise ValueError("Benchmark must be a JSON list.")

if not cases:
    raise ValueError("Benchmark must contain at least one case.")

required_fields = {
    "id", "question", "category", "family_id", "corpus_id",
    "answerable", "reference_answer", "required_claims",
    "forbidden_claims", "evidence", "absence_reason",
    "review_status", "reviewer_notes",
}

snapshot_dir = (
    project_root / "corpus" / "snapshots"
    / "20261105T054527423001Z"
)

source_manifest = json.loads(
    (snapshot_dir / "manifest.json").read_text(encoding="utf-8")
)

source_anchors = {}

for source in source_manifest["sources"]:
    html = (snapshot_dir / source["file"]).read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    article = soup.select_one("article.bd-article")

    if article is None:
        raise ValueError(f"{source['source_id']}: article not found.")

    for gallery in article.select("section#gallery-examples"):
        gallery.decompose()

    source_anchors[source["source_id"]] = {
        element["id"] for element in article.select("[id]")
    }
seen_ids = set()

for index, case in enumerate(cases):
    if not isinstance(case, dict):
        raise ValueError(f"Case at index {index} must be a JSON object.")

    missing_fields = required_fields - case.keys()
    if missing_fields:
        raise ValueError(
            f"Case at index {index} is missing: {sorted(missing_fields)}"
        )

    case_id = case["id"]

    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError(f"Case at index {index} needs a nonempty string ID.")

    if case_id in seen_ids:
        raise ValueError(f"Duplicate case ID: {case_id}")

    seen_ids.add(case_id)
    allowed_categories = {"direct", "synthesis", "noisy", "unanswerable"}
    allowed_statuses = {"draft", "needs_revision", "approved"}

    category = case["category"]
    if not isinstance(category, str) or category not in allowed_categories:
        raise ValueError(f"{case_id}: invalid category.")

    status = case["review_status"]
    if not isinstance(status, str) or status not in allowed_statuses:
        raise ValueError(f"{case_id}: invalid review_status.")

    if (category == "unanswerable") != (not case["answerable"]):
        raise ValueError(
            f"{case_id}: category and answerable disagree."
        )


    if not isinstance(case["answerable"], bool):
        raise ValueError(f"{case_id}: answerable must be true or false.")

    for field in ("question", "family_id", "reference_answer"):
        value = case[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{case_id}: {field} must be a nonempty string.")
    for field in ("required_claims", "forbidden_claims"):
        value = case[field]

        if not isinstance(value, list):
            raise ValueError(f"{case_id}: {field} must be a list.")

        if any(
            not isinstance(item, str) or not item.strip()
            for item in value
        ):
            raise ValueError(
                f"{case_id}: {field} must contain nonempty strings."
            )

    if not case["required_claims"]:
        raise ValueError(f"{case_id}: required_claims must not be empty.")
    evidence = case["evidence"]

    if not isinstance(evidence, list):
        raise ValueError(f"{case_id}: evidence must be a list.")

    for entry in evidence:
        
        if not isinstance(entry, dict):
            raise ValueError(f"{case_id}: evidence entries must be objects.")

        for field in ("source_id", "section", "anchor"):
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{case_id}: evidence {field} must be a nonempty string."
                )

        if entry["source_id"] not in {"S1", "S2", "S3"}:
            raise ValueError(f"{case_id}: unknown evidence source.")
        source_id = entry["source_id"]

        if entry["anchor"] not in source_anchors[source_id]:
            raise ValueError(
                f"{case_id}: anchor {entry['anchor']!r} "
                f"not found in {source_id}."
            )

    if case["answerable"]:
        if not evidence:
            raise ValueError(f"{case_id}: answerable cases need evidence.")

        if case["absence_reason"] is not None:
            raise ValueError(
                f"{case_id}: answerable cases need absence_reason=null."
            )
    else:
        reason = case["absence_reason"]
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(
                f"{case_id}: unanswerable cases need an absence_reason."
            )
    corpus_id = case["corpus_id"]
    if corpus_id is not None:
        if not isinstance(corpus_id, str) or not corpus_id.strip():
            raise ValueError(
                f"{case_id}: corpus_id must be null or a nonempty string."
            )

    reviewer_notes = case["reviewer_notes"]
    if reviewer_notes is not None:
        if not isinstance(reviewer_notes, str):
            raise ValueError(
                f"{case_id}: reviewer_notes must be null or a string."
            )



print(f"Loaded benchmark has {len(cases)} cases. ")

