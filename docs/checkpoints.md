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
- Separate stage-coverage evaluation was added at `data/eval/stage1_coverage_queries.json` and rerun on `medical_research_chunks_docling_v2_batch1`.
- Stage-coverage retrieval now reaches all `20` stage-1 papers at the document-selection level, with only a small remaining top-1 ambiguity in the hepcidin cluster.
- Zero-table audits on `1-s2.0-S0009912024000250-main` and `hepcidin diagnostic tool` indicate those PDFs do not currently show evidence of missing in-body tables in the ingested files.
- Additional manual UI spot checks are in progress, with the first newly-added-paper checks currently passing.

## April 1, 2026 (follow-up)

- Stage-1 and stage-2 manual UI spot checks are now complete.
- No confirmed retrieval failures were added to `known_gap_queries.json`; newly-added-paper checks passed.
- Stage-1 is fully de-risked. Stage-2 planning may begin.

## April 1, 2026 (answer quality hardening)

- Dependencies pinned in `requirements.txt`; `docling==2.81.0` added (was missing from the file).
- Research prompt hardened: instructions now cover study design identification, exact effect size reporting, CI/p-value inclusion, and key limitations.
- `ReasoningService.research()` return type changed from `str` to `ResearchAnswer` (insight, evidence_basis, citations, confidence).
- `ConfidenceLevel` enum added (HIGH/MEDIUM/LOW/INSUFFICIENT), derived from retrieval signals with no extra LLM call.
- UI updated: Evidence Basis and Citations are collapsible expanders; a coloured confidence banner appears above each insight.

## Current Interpretation

- The formal stage-1 rollout gate is passing and stage-1 is fully de-risked.
- The stage-1 coverage gap is closed.
- The remaining watch item is the hepcidin cluster top-1 ambiguity; it has not produced confirmed retrieval failures.
- Next work: answer-quality evaluation layer, then UI collection-selection and observability improvements.
