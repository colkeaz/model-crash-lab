import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import sklearn
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer

project_root = Path(__file__).resolve().parents[1]

chunks_path = (
    project_root / "corpus" / "chunks_v2"
    / "20261105T054527423001Z" / "chunks.json"
)
benchmark_path = project_root / "benchmarks" / "pilot.json"

chunks_bytes = chunks_path.read_bytes()
benchmark_bytes = benchmark_path.read_bytes()

artifact = json.loads(chunks_bytes)
cases = json.loads(benchmark_bytes)
chunks = artifact["chunks"]

if not chunks or not cases:
    raise ValueError("Chunks and benchmark cases must not be empty.")

for case in cases:
    if case["corpus_id"] != artifact["corpus_id"]:
        raise ValueError(f"{case['id']}: corpus ID mismatch.")

settings = {
    "word": {
        "ngram_range": (1, 2),
        "token_pattern": r"(?u)\b\w+\b",
        "norm": "l2",
    },
    "char_wb": {
        "analyzer": "char_wb",
        "ngram_range": (3, 5),
        "norm": "l2",
    },
}

vectorizer = FeatureUnion([
    ("word", TfidfVectorizer(**settings["word"])),
    ("char_wb", TfidfVectorizer(**settings["char_wb"])),
]
)

# Fit only on documentation—not questions or reference answers.
matrix = vectorizer.fit_transform(
    [chunk["text"] for chunk in chunks]
)

top_k = 5
results = []

for case in cases:
    query = vectorizer.transform([case["question"]])
    scores = (matrix @ query.T).toarray().ravel()

    ranked_indices = sorted(
        range(len(chunks)),
        key=lambda index: (-scores[index], chunks[index]["chunk_id"]),
    )

    positive_matches = [
        index for index in ranked_indices if scores[index] > 0
    ]

    top_results = [
        {
            "rank": rank,
            "score": float(scores[index]),
            **chunks[index],
        }
        for rank, index in enumerate(positive_matches[:top_k], start=1)
    ]

    results.append({
        "case_id": case["id"],
        "question": case["question"],
        "answerable": case["answerable"],
        "required_claims": case["required_claims"],
        "expected_evidence": case["evidence"],
        "top_results": top_results,
        "review": {
            "sufficient_in_top_k": None,
            "supporting_chunk_ids": [],
            "notes": "",
        } if case["answerable"] else None,
    })

    print(f"{case['id']}: retrieved {len(top_results)} passages.")

run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
output_dir = project_root / "runs" / "retrieval" / run_id
output_dir.mkdir(parents=True, exist_ok=False)

report = {
    "run_id": run_id,
    "retriever": "tfidf-feature-union-v1",
    "settings": settings,
    "top_k": top_k,
    "python_version": platform.python_version(),
    "sklearn_version": sklearn.__version__,
    "corpus_id": artifact["corpus_id"],
    "chunks_sha256": hashlib.sha256(chunks_bytes).hexdigest(),
    "benchmark_sha256": hashlib.sha256(benchmark_bytes).hexdigest(),
    "results": results,
}

output_path = output_dir / "results.json"
output_path.write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

print(f"Saved evaluation: {output_path}")
