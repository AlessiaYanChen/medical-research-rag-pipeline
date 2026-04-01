# Checkpoints

This file holds dated checkpoint history that used to live inline in `ROADMAP.md`.
It is a record of how the repo moved from small-corpus retrieval tuning into staged rollout work.

## March 23, 2026

- Stable, expanded, and OOD evaluation tracks were established on the small-corpus baseline.
- Retrieval quality on `medical_research_chunks_v1` was clean enough to treat benchmark-driven iteration as the default workflow.
- Header normalization, section-aware ranking, and table-linked context became part of the working retrieval baseline.

## March 24, 2026

- Rebuild, audit, and single-document repair tooling were hardened.
- Manifest-aware validation and duplicate-identity checks were added.
- Setup and onboarding documentation were tightened.

## March 25, 2026

- Parser bakeoff tooling was added under `experiments/`.
- `Docling` was adapted into the repo parser contract and evaluated on an isolated subset.
- Early parser-side cleanup focused on image-placeholder stripping, citation cleanup, and opening-section normalization.

## March 27, 2026

- Runtime-query coverage expanded with real UI-derived questions.
- Several broad retrieval experiments were inspected locally and either kept narrow or reverted when they did not preserve the benchmark line.
- Product direction shifted from open-ended retrieval redesign toward controlled staged rollouts with `Docling` as the active parser.

## March 31, 2026

- Stage 1 (`20 PDFs`) was first compiled as a failed rollout checkpoint on `medical_research_chunks_docling_v2_batch1`.
- Rebuild and audit passed, but stable, expanded, OOD, and runtime evidence did not justify promotion at that point.

## April 1, 2026

- Narrow retrieval fixes resolved the remaining rollout blocker families:
  - token-aware handling for `uti` so unrelated words like `utilization` no longer qualify as infectious-domain matches
  - stricter in-domain matching for study-design classification prompts
- Regression tests were added for both failure shapes.
- Stable, expanded, OOD, and runtime reruns on `medical_research_chunks_docling_v2_batch1` matched the small-corpus baselines on rollout-gated summary metrics.
- Manual spot checks were recorded.
- The regenerated stage-1 rollout report passed.

## Current Interpretation

- The formal stage-1 rollout gate is now passing.
- The main remaining risk is not the old blocker set; it is incomplete evaluation coverage of the `13` newly added papers in the `20`-PDF stage corpus.
- The clearest next checkpoint work is separate stage-coverage evaluation, especially same-topic ambiguity in the hepcidin cluster and validation of zero-table-chunk documents.
