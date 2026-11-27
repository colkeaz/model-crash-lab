
import argparse
import json
from pathlib import Path

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

parser = argparse.ArgumentParser()
parser.add_argument("question")
args = parser.parse_args()

if not args.question.strip():
    parser.error("Question must not be empty.")

project_root = Path(__file__).resolve().parents[1]
chunks_path = (
    project_root / "corpus" / "chunks"
    / "20261105T054527423001Z" / "chunks.json"
)

artifact = json.loads(chunks_path.read_text(encoding="utf-8"))
chunks = artifact["chunks"]

if not chunks:
    raise ValueError("No chunks available.")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    token_pattern=r"(?u)\b\w+\b",
    norm="l2",
)

# Learn vocabulary and document frequencies from documentation only.
matrix = vectorizer.fit_transform(
    [chunk["text"] for chunk in chunks]
)

# Apply that same vocabulary to the question.
query_vector = vectorizer.transform([args.question])

print(f"Indexed {len(chunks)} chunks.")
print(f"scikit-learn version: {sklearn.__version__}")

if query_vector.nnz == 0:
    print("No query terms occur in the index.")
else:
    scores = (matrix @ query_vector.T).toarray().ravel()

    ranked_indices = sorted(
        range(len(chunks)),
        key=lambda index: (-scores[index], chunks[index]["chunk_id"]),
    )

    positive_matches = [
        index for index in ranked_indices if scores[index] > 0
    ]

    for rank, index in enumerate(positive_matches[:5], start=1):
        chunk = chunks[index]

        print(f"\nRank {rank} | similarity={scores[index]:.4f}")
        print(f"Source: {chunk['source_id']}#{chunk['anchor']}")
        print(" > ".join(chunk["heading_path"]))
        print(chunk["text"])