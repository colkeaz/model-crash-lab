import hashlib
import json
from pathlib import Path

from bs4 import BeautifulSoup

Project = Path(__file__).resolve().parents[1]
snapshot_id = "20261105T054527423001Z"
snapshot_dir = Project / "corpus" / "snapshots" / snapshot_id

manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))

output_dir = Project / "corpus" / "extracted" / snapshot_id
# Guard: Fail-fast before creating directories or writing any files
if output_dir.resolve() == snapshot_dir.resolve():
    raise ValueError("Extraction output must differ from the raw snapshot directory.")

output_dir.mkdir(parents=True, exist_ok=True)

output_manifest_path = output_dir / "manifest.json"
existing_manifest = {}
if output_manifest_path.exists():
    try:
        existing_manifest = json.loads(output_manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        existing_manifest = {}

# Retain documentation_version and corpus_id across extractions
documentation_version = (
    manifest.get("documentation_version")
    or manifest.get("observed_version")
    or existing_manifest.get("documentation_version")
)

corpus_id = (
    manifest.get("corpus_id")
    or existing_manifest.get("corpus_id")
)

extraction_manifest = {
    "snapshot_id": snapshot_id,
}

if documentation_version is not None:
    extraction_manifest["documentation_version"] = documentation_version

if corpus_id is not None:
    extraction_manifest["corpus_id"] = corpus_id

extraction_manifest["extractor_version"] = "1"
extraction_manifest["sources"] = []
for source in manifest["sources"]:
    source_id = source["source_id"]
    raw_html = (snapshot_dir / source["file"]).read_bytes()

    if hashlib.sha256(raw_html).hexdigest() != source["sha256"]:
        raise ValueError(f"{source_id}: saved HTML does not match manifest hash.")
    soup = BeautifulSoup(raw_html.decode("utf-8"), "html.parser")
    article = soup.select_one("article.bd-article")

    if article is None:
        raise ValueError(f"{source_id}: article content was not found.")

    for element in article.select(
        "script, style, nav, a.headerlink, section#gallery-examples"
    ):
        element.decompose()

    for classifier in article.select("span.classifier"):
        classifier.insert_before(": ")

    code_blocks = {}

    for index, block in enumerate(article.find_all("pre")):
        marker = f"__MCL_CODE_BLOCK_{index}__"
        code_blocks[marker] = block.get_text()
        block.replace_with(f"\n{marker}\n")

    # Add boundaries around structural elements, rather than inline tokens.
    block_tags = [
        "h1", "h2", "h3", "h4", "h5", "h6",
        "p", "dt", "dd", "li", "div", "section",
        "tr", "blockquote",
    ]

    for element in article.find_all(block_tags):
        element.insert_before("\n")
        element.insert_after("\n")

    raw_text = article.get_text(separator="", strip=False)

    # Normalize prose while leaving the protected code blocks untouched.
    lines = [
        " ".join(line.split())
        for line in raw_text.splitlines()
    ]
    text = "\n".join(line for line in lines if line)

    for marker, code in code_blocks.items():
        text = text.replace(marker, code.rstrip())

    if not text:
        raise ValueError(f"{source_id}: extracted text is empty.")

    filename = f"{source_id}.txt"
    content = (text + "\n").encode("utf-8")
    (output_dir / filename).write_bytes(content)

    extraction_manifest["sources"].append({
        "source_id": source_id,
        "source_url": source["resolved_url"],
        "raw_sha256": source["sha256"],
        "file": filename,
        "sha256": hashlib.sha256(content).hexdigest(),
    })

    print(f"\n--- {source_id} ---")
    print(text[:500])
    print(f"Total extracted characters: {len(text)}")

output_manifest_path.write_text(
    json.dumps(extraction_manifest, indent=2) + "\n",
    encoding="utf-8",
)

print(f"\nExtracted files saved to: {output_dir}")

