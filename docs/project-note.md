# Initial repository note

Model Crash Lab started with a benchmark for a documentation assistant. The first 30 questions cover API lookup, synthesis, noisy wording, and requests that lack enough evidence. The checked-in results are development records that help diagnose failure modes and compare recovery costs.

The current baseline uses the evidence-first v1 prompt. The v2 experiment increased answer coverage on its 12-case development slice and also increased combined selective error. Its recorded verdict is `advance = false`. The [experiment report](evidence-first-v2-dev-results.md) records the measurements and case transitions.

The next engineering milestone is to score the saved experiment from hash-bound trace data, fix the zero-control quote-length gate, and save revised results from the existing attempts. See the [evaluation design](paired-policy-evaluation-design.md) for how completeness, grounding, protocol validity, and cost are measured.
