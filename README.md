# Model Crash Lab

Model Crash Lab is an evaluation workbench for a documentation assistant backed by a local language model. It measures when retrieval supplies enough evidence, when the model uses that evidence correctly, and what it costs to expand retrieval or abstain.

The project uses a frozen scikit-learn 1.7.2 documentation corpus and 30 development questions. It records retrieved passages and model responses, checks quotations against the supplied text, and keeps semantic review separate from mechanical validation.

## What the experiments show

- The paired dataset compares three and five passages on the same questions. Its fixed baselines report coverage, selective error, and input and output tokens.
- On a 12-question development slice with local Qwen3 4B, the v2 prompt increased substantive answers from 3 to 11 and clean answers from 2 to 4. Combined selective error rose from 1/3 to 7/11, and v2 missed the advance gate.
- The benchmark and results are for development. The repository contains exploratory comparisons and no trained recovery policy.

## Run the offline checks

Use Python 3.14 and run these commands from the repository root. The documentation snapshot is version 1.7.2. Scikit-learn 1.9.0 is the Python dependency used by the retriever.

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/validate_benchmark.py --benchmark benchmarks/pilot.json
python scripts/validate_benchmark.py --benchmark benchmarks/extension-v1.json
python scripts/validate_benchmark.py --benchmark benchmarks/extension-v2.json
python scripts/evaluate_fixed_baselines.py --output runs/policy_data/fixed_baselines_replay.json
```

The baseline replay reads the saved paired dataset and writes a separate report. It requires no model server. Fresh generation experiments use Ollama with `qwen3:4b`. See the [experiment protocol](docs/evidence-first-v2-dev-experiment.md) for the pinned inputs, model settings, and advance criteria.

## Repository guide

| Path | Contents |
| --- | --- |
| [benchmarks/](benchmarks/) | Pilot and two benchmark extensions with required claims and evidence anchors |
| [corpus/](corpus/) | Frozen documentation snapshots, extracted text, structural blocks, and retrieval chunks |
| [scripts/](scripts/) | Benchmark, retrieval, generation, review, and reporting scripts |
| [semantic-audits/](semantic-audits/) | Trace-bound semantic judgments |
| [runs/](runs/) | Saved inputs, responses, evaluations, and provenance |
| [tests/](tests/) | Regression checks for review, pairing, and baseline metrics |

Read the [project note](docs/project-note.md), [paired evaluation design](docs/paired-policy-evaluation-design.md), and [v2 development report](docs/evidence-first-v2-dev-results.md). The frozen source links and hashes are recorded in the [corpus manifest](corpus/snapshots/20261105T054527423001Z/manifest.json).
