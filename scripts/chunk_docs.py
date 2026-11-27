import hashlib
import json
from pathlib import Path

from bs4 import BeautifulSoup


def parse_definition_list(
    definition_list, *, source_id, anchor, heading_path, block_type
):
    result = []

    for term in definition_list.find_all("dt", recursive=False):
        description = term.find_next_sibling("dd")
        name_element = term.find("strong") or term.find("code")

        if name_element is None:
            raise ValueError(
                f"{source_id}: definition name not found."
            )
        name = name_element.get_text(strip=True)
        classifier = term.select_one("span.classifier")
        type_text = (
            classifier.get_text(" ", strip=True)
            if classifier else ""
        )

        result.append({
            "source_id": source_id,
            "anchor": anchor,
            "heading_path": [*heading_path, name],
            "block_type": block_type,
            "text": (
                f"{name}: {type_text}\n"
                + " ".join(
                    description.get_text(" ", strip=True).split()
                )
            ),
        })

    return result
def walk_sections(node, heading_path, anchor):
    heading_tags = ["h1", "h2", "h3", "h4", "h5", "h6"]

    if node.name == "section":
        heading = node.find(heading_tags, recursive=False)

        if heading is None or not node.get("id"):
            raise ValueError("S3: section heading or anchor missing.")

        title = " ".join(heading.get_text(" ", strip=True).split())
        heading_path = [*heading_path, title]
        anchor = node["id"]

    if node.name in heading_tags:
        return

    if node.name in {"p", "pre"}:
        if anchor is None:
            raise ValueError("S3: content has no section anchor.")

        is_code = node.name == "pre"
        text = (
            node.get_text().rstrip()
            if is_code
            else " ".join(node.get_text(" ", strip=True).split())
        )

        if text:
            yield {
                "source_id": "S3",
                "anchor": anchor,
                "heading_path": heading_path.copy(),
                "block_type": "code" if is_code else "paragraph",
                "text": text,
            }

        return

    for child in node.children:
        if getattr(child, "name", None):
            yield from walk_sections(child, heading_path, anchor)


project_root = Path(__file__).resolve().parents[1]
snapshot_dir = (
    project_root / "corpus" / "snapshots"
    /"20261105T054527423001Z"
)

manifest = json.loads((snapshot_dir/ "manifest.json").read_text(encoding="utf-8"))

source = next(
    entry for entry in manifest["sources"]
    if entry["source_id"] == "S1"
)

raw_html = (snapshot_dir / source["file"]).read_bytes()
if hashlib.sha256(raw_html).hexdigest() != source["sha256"]:
    raise ValueError("S1: source hash mismatch.")

soup = BeautifulSoup(raw_html.decode("utf-8"), "html.parser")
signature = soup.find(id="sklearn.model_selection.train_test_split")

if signature is None:
    raise ValueError("S1: function definition not found.")

function_body = signature.find_next_sibling("dd")
fields = function_body.select_one("dl.field-list")

parameters_heading = next(
    (
        term for term in fields.find_all("dt", recursive=False)
        if term.get_text(strip=True).rstrip(":") == "Parameters"
    ),
    None,
)

if parameters_heading is None:
    raise ValueError("S1: Parameters heading not found.")

parameters_body = parameters_heading.find_next_sibling("dd")
parameter_list = parameters_body.find("dl")

blocks = []

overview_parts = []

for child in function_body.children:
    if child is fields:
        break

    if getattr(child, "name", None) == "p":
        overview_parts.append(
            " ".join(child.get_text(" ", strip=True).split())
        )

if not overview_parts:
    raise ValueError("S1: overview paragraphs not found.")

blocks.append({
    "source_id": "S1",
    "anchor": signature["id"],
    "heading_path": ["train_test_split", "Overview"],
    "block_type": "overview",
    "text": "\n\n".join(overview_parts),
})

blocks.extend(parse_definition_list(
    parameter_list,
    source_id="S1",
    anchor=signature["id"],
    heading_path=["train_test_split", "Parameters"],
    block_type="parameter",
))


returns_heading = next(
    (
        term for term in fields.find_all("dt", recursive=False)
        if term.get_text(strip=True).rstrip(":") == "Returns"
    ),
    None,
)

if returns_heading is None:
    raise ValueError("S1: Returns heading not found.")

returns_body = returns_heading.find_next_sibling("dd")
returns_list = returns_body.find("dl")

blocks.extend(parse_definition_list(
    returns_list,
    source_id="S1",
    anchor=signature["id"],
    heading_path=["train_test_split", "Returns"],
    block_type="return",
))

examples_heading = function_body.find(
    "p", class_="rubric", string="Examples"
)

if examples_heading is None:
    raise ValueError("S1: Examples heading not found.")

example_parts = []

for sibling in examples_heading.find_next_siblings():
    for code in sibling.find_all("pre"):
        example_parts.append(code.get_text().rstrip())

if not example_parts:
    raise ValueError("S1: example code not found.")

blocks.append({
    "source_id": "S1",
    "anchor": signature["id"],
    "heading_path": ["train_test_split", "Examples"],
    "block_type": "example",
    "text": "\n\n".join(example_parts),
})

# Check the expected structure for this fixed S1 snapshot.
assert len(blocks) == 9
assert blocks[0]["block_type"] == "overview"
assert blocks[-1]["block_type"] == "example"
assert sum(b["block_type"] == "parameter" for b in blocks) == 6
assert sum(b["block_type"] == "return" for b in blocks) == 1

extracted_manifest = json.loads(
    (
        project_root / "corpus" / "extracted"
        / snapshot_dir.name / "manifest.json"
    ).read_text(encoding="utf-8")
)

output_dir = (
    project_root / "corpus" / "blocks" / snapshot_dir.name
)
output_dir.mkdir(parents=True, exist_ok=True)

artifact = {
    "parser_version": "1",
    "corpus_id": extracted_manifest["corpus_id"],
    "source_id": source["source_id"],
    "raw_sha256": source["sha256"],
    "blocks": blocks,
}

output_path = output_dir / "S1.json"
previous = json.loads(output_path.read_text(encoding="utf-8"))

if previous["blocks"] != blocks:
    raise ValueError("Refactor changed the saved S1 blocks.")

print("Refactor verified: all S1 blocks are unchanged.")

output_path.write_text(
    json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

print(f"Saved structural blocks: {output_path}")
print(f"Parsed {len(blocks)} structural blocks.")

for block in blocks:
    print(json.dumps(block, indent=2, ensure_ascii=False))

s2_source = next(
    entry for entry in manifest["sources"]
    if entry["source_id"] == "S2"
)

s2_html = (snapshot_dir / s2_source["file"]).read_bytes()

if hashlib.sha256(s2_html).hexdigest() != s2_source["sha256"]:
    raise ValueError("S2: source hash mismatch.")

s2_soup = BeautifulSoup(s2_html.decode("utf-8"), "html.parser")
s2_signature = s2_soup.find(id="sklearn.pipeline.Pipeline")

if s2_signature is None:
    raise ValueError("S2: Pipeline definition not found.")

s2_body = s2_signature.find_next_sibling("dd")
s2_fields = s2_body.find("dl", class_="field-list", recursive=False)

if s2_fields is None:
    raise ValueError("S2: constructor fields not found.")

s2_parameters_heading = next(
    (
        term for term in s2_fields.find_all("dt", recursive=False)
        if term.get_text(strip=True).rstrip(":") == "Parameters"
    ),
    None,
)

if s2_parameters_heading is None:
    raise ValueError("S2: constructor Parameters heading not found.")

s2_parameters_body = s2_parameters_heading.find_next_sibling("dd")
s2_parameter_list = s2_parameters_body.find("dl")

s2_blocks = parse_definition_list(
    s2_parameter_list,
    source_id="S2",
    anchor=s2_signature["id"],
    heading_path=["Pipeline", "Parameters"],
    block_type="parameter",
)

assert [block["heading_path"][-1] for block in s2_blocks] == [
    "steps", "transform_input", "memory", "verbose"
]
s2_overview_parts = []

for child in s2_body.children:
    if child is s2_fields:
        break

    if getattr(child, "name", None) == "p":
        s2_overview_parts.append(
            " ".join(child.get_text(" ", strip=True).split())
        )

if not s2_overview_parts:
    raise ValueError("S2: overview not found.")

s2_blocks.insert(0, {
    "source_id": "S2",
    "anchor": s2_signature["id"],
    "heading_path": ["Pipeline", "Overview"],
    "block_type": "overview",
    "text": "\n\n".join(s2_overview_parts),
})

# Collect the class example, stopping before method definitions.
s2_examples_heading = s2_body.find(
    "p", class_="rubric", string="Examples"
)

if s2_examples_heading is None:
    raise ValueError("S2: class Examples heading not found.")

s2_example_parts = []

for sibling in s2_examples_heading.find_next_siblings():
    if sibling.name == "dl" and "py" in sibling.get("class", []):
        break

    for code in sibling.find_all("pre"):
        s2_example_parts.append(code.get_text().rstrip())

if not s2_example_parts:
    raise ValueError("S2: class example code not found.")

s2_blocks.append({
    "source_id": "S2",
    "anchor": s2_signature["id"],
    "heading_path": ["Pipeline", "Examples"],
    "block_type": "example",
    "text": "\n\n".join(s2_example_parts),
})

assert len(s2_blocks) == 6

method_count = 0

for method in s2_body.find_all("dl", class_="method", recursive=False):
    method_signature = method.find("dt", recursive=False)
    method_body = method.find("dd", recursive=False)

    if method_signature is None or method_body is None:
        raise ValueError("S2: incomplete method definition.")

    method_anchor = method_signature["id"]
    method_name = method_anchor.rsplit(".", 1)[-1]
    method_path = ["Pipeline", method_name]

    method_fields = method_body.find(
        "dl", class_="field-list", recursive=False
    )

    overview_parts = []

    for child in method_body.children:
        if child is method_fields:
            break

        if getattr(child, "name", None) == "p":
            overview_parts.append(
                " ".join(child.get_text(" ", strip=True).split())
            )

    if not overview_parts:
        raise ValueError(f"S2: {method_name} overview not found.")

    s2_blocks.append({
        "source_id": "S2",
        "anchor": method_anchor,
        "heading_path": [*method_path, "Overview"],
        "block_type": "overview",
        "text": "\n\n".join(overview_parts),
    })

    if method_fields is not None:
        for heading in method_fields.find_all("dt", recursive=False):
            label = heading.get_text(strip=True).rstrip(":")

            if label not in {"Parameters", "Returns"}:
                continue

            field_body = heading.find_next_sibling("dd")
            definition_list = field_body.find("dl")

            if definition_list is None:
                raise ValueError(
                    f"S2: {method_name} {label} definitions not found."
                )

            s2_blocks.extend(parse_definition_list(
                definition_list,
                source_id="S2",
                anchor=method_anchor,
                heading_path=[*method_path, label],
                block_type=(
                    "parameter" if label == "Parameters" else "return"
                ),
            ))

    method_count += 1

assert method_count == 17
attributes_heading = next(
    (
        term for term in s2_fields.find_all("dt", recursive=False)
        if term.get_text(strip=True).rstrip(":") == "Attributes"
    ),
    None,
)

if attributes_heading is None:
    raise ValueError("S2: Attributes heading not found.")

attributes_body = attributes_heading.find_next_sibling("dd")

attribute_blocks = parse_definition_list(
    attributes_body.find("dl"),
    source_id="S2",
    anchor=s2_signature["id"],
    heading_path=["Pipeline", "Attributes"],
    block_type="attribute",
)

assert [b["heading_path"][-1] for b in attribute_blocks] == [
    "named_steps", "classes_", "n_features_in_", "feature_names_in_"
]

# Preserve their position before the class example and methods.
example_index = next(
    index for index, block in enumerate(s2_blocks)
    if block["heading_path"] == ["Pipeline", "Examples"]
)
s2_blocks[example_index:example_index] = attribute_blocks

assert len(s2_blocks) == 75

s2_artifact = {
    "parser_version": "1",
    "corpus_id": extracted_manifest["corpus_id"],
    "source_id": "S2",
    "raw_sha256": s2_source["sha256"],
    "blocks": s2_blocks,
}

s2_output_path = output_dir / "S2.json"
s2_output_path.write_text(
    json.dumps(s2_artifact, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

print(f"Saved S2 structural blocks: {s2_output_path}")
print(f"S2: parsed {method_count} methods.")

print(f"\nS2: parsed {len(s2_blocks)} structural blocks total.")

for block in s2_blocks:
    print(json.dumps(block, indent=2, ensure_ascii=False))


s3_source = next(
    entry for entry in manifest["sources"]
    if entry["source_id"] == "S3"
)

s3_html = (snapshot_dir / s3_source["file"]).read_bytes()

if hashlib.sha256(s3_html).hexdigest() != s3_source["sha256"]:
    raise ValueError("S3: source hash mismatch.")

s3_soup = BeautifulSoup(s3_html.decode("utf-8"), "html.parser")
s3_article = s3_soup.select_one("article.bd-article")

if s3_article is None:
    raise ValueError("S3: article not found.")

for element in s3_article.select(
    "script, style, nav, a.headerlink, section#gallery-examples"
):
    element.decompose()

s3_blocks = list(walk_sections(s3_article, [], None))

paragraph_count = sum(
    block["block_type"] == "paragraph" for block in s3_blocks
)
code_count = sum(
    block["block_type"] == "code" for block in s3_blocks
)

assert paragraph_count == 74
assert code_count == 15

s3_artifact = {
    "parser_version": "1",
    "corpus_id": extracted_manifest["corpus_id"],
    "source_id": "S3",
    "raw_sha256": s3_source["sha256"],
    "blocks": s3_blocks,
}

s3_output_path = output_dir / "S3.json"
s3_output_path.write_text(
    json.dumps(s3_artifact, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

print(f"S3: {paragraph_count} paragraphs, {code_count} code blocks.")
print(f"Saved: {s3_output_path}")