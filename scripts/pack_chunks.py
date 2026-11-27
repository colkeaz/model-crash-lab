
import hashlib
import json
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
snapshot_id = "20261105T054527423001Z"
blocks_dir = project_root / "corpus" / "blocks" / snapshot_id
max_words = 300

chunks = []
buffer = []
current_key = None
corpus_id = None


def section_path(block):
    # Parameters and returns share their parent section.
    if block["block_type"] in {"return", "attribute"}:
        return block["heading_path"][:-1]
    return block["heading_path"] 

def render(items, path):
    body = "\n\n".join(block["text"] for _, block in items)
    return " > ".join(path) + "\n\n" + body


def flush():
    if not buffer:
        return

    source_id, anchor, path = current_key
    text = render(buffer, path)

    chunk = {
        "corpus_id": corpus_id,
        "source_id": source_id,
        "anchor": anchor,
        "heading_path": list(path),
        "block_indices": [index for index, _ in buffer],
        "text": text,
    }

    identity = json.dumps(
        chunk, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    chunk["chunk_id"] = "sha256:" + hashlib.sha256(identity).hexdigest()
    chunk["word_count"] = len(text.split())
    chunks.append(chunk)
    buffer.clear()


for source_id in ("S1", "S2", "S3"):
    artifact = json.loads(
        (blocks_dir / f"{source_id}.json").read_text(encoding="utf-8")
    )

    if corpus_id is None:
        corpus_id = artifact["corpus_id"]

    if artifact["corpus_id"] != corpus_id:
        raise ValueError(f"{source_id}: corpus ID mismatch.")

    for index, block in enumerate(artifact["blocks"]):
        path = tuple(section_path(block))
        key = (source_id, block["anchor"], path)

        if key != current_key:
            flush()
            current_key = key

        candidate = [*buffer, (index, block)]

        if len(render(candidate, path).split()) > max_words:
            flush()

        if len(render([(index, block)], path).split()) > max_words:
            raise ValueError(
                f"{source_id} block {index}: needs oversized-block handling."
            )

        buffer.append((index, block))

flush()

output_dir = project_root / "corpus" / "chunks_v2" / snapshot_id
output_dir.mkdir(parents=True, exist_ok=True)

output = {
    "chunker_version": "2",
    "corpus_id": corpus_id,
    "max_words": max_words,
    "overlap_words": 0,
    "chunks": chunks,
}

(output_dir / "chunks.json").write_text(
    json.dumps(output, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

print(f"Packed {len(chunks)} chunks.")
print(f"Largest chunk: {max(c['word_count'] for c in chunks)} words.")
print(f"Saved to: {output_dir / 'chunks.json'}")