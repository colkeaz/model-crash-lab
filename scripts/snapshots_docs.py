import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

project = Path(__file__).resolve().parents[1]

sources = {
    "S1": "https://scikit-learn.org/1.7/modules/generated/sklearn.model_selection.train_test_split.html",
    "S2": "https://scikit-learn.org/1.7/modules/generated/sklearn.pipeline.Pipeline.html",
    "S3": "https://scikit-learn.org/1.7/common_pitfalls.html",
}

snaphot_id =datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
output_dir = project/"corpus"/"snapshots"/snaphot_id
output_dir.mkdir(parents=True, exist_ok=False)

manifest = {
    "snapshot_id": snaphot_id,
    "documentation_branch": "1.7",
    "observed_version" : None,
    "sources": []
}

for source_id, url in sources.items():
    with urlopen(url, timeout=30) as response:
        content_type = response.headers.get_content_type()
        if content_type != "text/html":
            raise ValueError(f"{source_id}: expected HTML, got {content_type}")

        content = response.read()
        resolved_url = response.geturl()
    if not content:
        raise ValueError(f"{source_id}: received an empty page")
    filename = f"{source_id}.html"
    (output_dir / filename).write_bytes(content)

    manifest["sources"].append({
        "source_id": source_id,
        "requested_url": url,
        "resolved_url": resolved_url,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "file": filename,
        "sha256": hashlib.sha256(content).hexdigest(),
    })

    print(f"Saved {source_id} : {len(content)} bytes")

manifest_path = output_dir / "manifest.json"
manifest_path.write_text(
    json.dumps(manifest,indent =2) +"\n",
    encoding = "utf-8"
)


print(f"Snapshot {snaphot_id} saved to {output_dir}")