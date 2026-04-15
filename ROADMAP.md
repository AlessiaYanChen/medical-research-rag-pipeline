# Roadmap

## Goal

Make the medical research RAG pipeline reliable enough for a medium-scale corpus of roughly `100 PDFs`, then scale toward roughly `300 PDFs` without noisy retrieval, weak grounding, or unclear rollback paths.

## Current State

- Active parser: `Docling`
- Rollback parser: `Marker`
- Current small-corpus baseline collection: `medical_research_chunks_docling_v1`
- Current stage-1 rollout collection: `medical_research_chunks_docling_v2_batch1`
- Stage-1 formal gate status: `pass`
<<<<<<< HEAD
- Stage-1 substantive risk: manual UI spot-check completion is still in progress for the newly added papers, even though stage-coverage retrieval now exercises the full `20`-PDF corpus
=======
- Stage-1 synthesis validation: in progress — answer quality baseline exists, known-gap false-confidence gate is clear, and the stage-1 synthesis gate now passes; remaining manual spot checks still gate Stage 2
>>>>>>> a5f3f07 (Add synthesis gate and record stage-1 pass)

## Collection Roles

- `medical_research_chunks_docling_v1`: active small-corpus baseline used for regression comparison
- `medical_research_chunks_docling_v2_batch1`: passing stage-1 `20`-PDF rollout artifact
- `medical_research_chunks_v1`: rollback collection for the older `Marker` path

## Rollout Gate

Treat a rollout stage as acceptable only when all of these are true:

1. Rebuild completes with explicit failure reporting.
2. `scripts/audit_collection_state.py --fail-on-issues` passes.
3. Stable and expanded retrieval benchmarks stay within the baseline-relative rollout tolerance.
4. `runtime_queries.json` shows no material regression.
5. A manual spot-check report is written for real medical questions.
6. One rollout report is generated from the rebuild, audit, evaluation, and manual-check artifacts.

Current retrieval gate policy:

- Comparison is baseline-relative, not absolute-score based.
- The rollout report currently uses a `0.02` maximum allowed drop per gated summary metric when a baseline evaluation is provided.
- The gated summary metrics are:
  - `expected_doc_hit_rate`
  - `expected_header_hit_rate`
  - `top1_expected_doc_hit_rate`
  - `top1_expected_header_hit_rate`
  - `average_doc_precision`
  - `average_header_precision`
  - `cross_document_average_doc_precision`

## Active Risks

<<<<<<< HEAD
1. Same-topic ambiguity: the hepcidin cluster is still the clearest early multi-paper disambiguation risk; one residual top-1 ambiguity is under watch.
2. Answer reliability: the app measures retrieval quality but not final synthesis quality.
3. Operational drift: dependencies are unpinned, so parser or reranker behavior can shift silently.
=======
1. Same-topic ambiguity: the hepcidin cluster is still the clearest early multi-paper disambiguation risk; one residual top-1 ambiguity is under watch at the synthesis level.
2. Synthesis validation gap: the answer quality baseline exists, the known-gap false-confidence gate is clear, and the stage-1 synthesis gate passes; Stage 2 still must not begin until the remaining synthesis spot checks are completed.
>>>>>>> a5f3f07 (Add synthesis gate and record stage-1 pass)

## Phase Status

### Phase 1: Retrieval Quality Stabilization

Status: Complete

Outcome:
- diversity, section preference, and table-default behavior are in place

### Phase 2: Evaluation and Corpus Validation

Status: Complete

Outcome:
- stable, expanded, OOD, runtime, and known-gap evaluation tracks exist

### Phase 3: Document-Level Retrieval

Status: Deferred

Use only if benchmark evidence shows the current metadata-first retrieval path has stopped holding.

### Phase 3B: Recall Extensions

Status: Deferred

Hybrid retrieval, sparse retrieval, query expansion, and similar recall extensions stay deferred until benchmark evidence shows real lexical recall gaps.

### Phase 4: Metadata and Ingestion Hardening

Status: Complete

Outcome:
- rebuild, audit, manifest, duplicate detection, and repair tooling are in place

### Phase 4B: Parser Bakeoff

Status: Complete

Outcome:
- `Docling` is selected for new ingestion
- `Marker` is rollback only
- detailed parser-bakeoff notes now live in `docs/checkpoints.md`
- parser comparison artifacts remain under `data/parser_bakeoff/results/`

### Phase 5A: Medium-Scale Readiness

Status: In Progress

Objective:
- prove the `Docling` production line remains reliable through staged `20`, `50`, and `100` PDF rollouts

Current checkpoint:
- stage 1 (`20 PDFs`) passed its formal rollout gate on `medical_research_chunks_docling_v2_batch1`
- stage-coverage retrieval now hits all `20` papers at the document-selection level, and the roadmap-priority zero-table PDF audits are complete
- stage-1 and stage-2 manual UI spot checks are now complete; stage 1 is fully de-risked
- hepcidin cluster top-1 ambiguity remains a watch item but has not produced confirmed retrieval failures

Stage-1 coverage completion tasks:

1. Complete `data/eval/stage1_coverage_queries.json` and keep its results under `data/eval/results/retrieval_eval_stage1_coverage.json`. Status: complete.
2. Add at least `1-2` retrieval queries per newly added paper. Status: complete.
3. Add cross-document queries that mix baseline and newly added papers. Status: complete.
4. Add hepcidin-cluster disambiguation queries explicitly. Status: complete, with one residual top-1 ambiguity still worth watching in manual review.
5. Run `8-10` additional manual UI spot checks focused on newly added papers. Status: complete.
6. Spot-check selected zero-table-chunk papers against the source PDFs, starting with `1-s2.0-S0009912024000250-main` and `hepcidin diagnostic tool`. Status: complete in `data/eval/results/stage1_zero_table_spot_checks.json`.
7. Add confirmed new-paper misses to `known_gap_queries.json` rather than mixing them into the runtime baseline. Status: no confirmed new-paper misses currently require promotion to known-gap tracking.

Stage-2 readiness rule:

- keep `runtime_queries.json` limited to real user/runtime questions
- put synthetic corpus-coverage probes into a separate evaluation dataset
- stage-1 coverage gap is now closed; stage 2 may begin

### Phase 5: Corpus Rollout

Status: Planned

Objective:
- expand beyond `100 PDFs` only after Phase 5A is clean

Phase gate:
- do not begin broader corpus rollout until medium-scale readiness is proven

## Next Priorities

Immediate stabilization work:

1. ~~Pin dependencies to reduce parser and reranker drift risk.~~ Complete — all packages pinned in `requirements.txt`, including `docling` which was previously missing from the file.

Order of work (stage-1 coverage gap is now closed):

1. ~~Harden the medical research prompt for study design, effect sizes, uncertainty, and limitations.~~ Complete — prompt now instructs study design identification, exact effect size reporting, CI/p-value inclusion, and limitation noting.
2. ~~Move toward structured answer output with chunk-level citations.~~ Complete — `ReasoningService.research()` now returns `ResearchAnswer` (insight, evidence_basis, citations). Citations are the actual retrieved chunks, not LLM-generated text. UI renders them as collapsible expanders.
3. ~~Add a typed abstention or confidence signal.~~ Complete — `ConfidenceLevel` enum (HIGH/MEDIUM/LOW/INSUFFICIENT) derived from retrieval signals (chunk count, distinct docs, "Insufficient evidence" in insight). UI shows a coloured banner per level.
4. ~~Add a small answer-quality evaluation layer separate from retrieval evaluation.~~ Complete — `src/app/evaluation/answer_quality_eval.py` evaluates abstention accuracy, confidence thresholds, and doc-ID coverage in evidence basis. Runner at `scripts/evaluate_answer_quality.py`. Query format: `data/eval/answer_quality_queries.json`.
<<<<<<< HEAD
5. ~~Improve the UI collection-selection and rollback workflow.~~ Complete - collection roles now render in the UI, switching collections clears stale session state, and rollback ingestion is soft-blocked in `scripts/ui_app.py`.
6. ~~Add basic observability for latency and retrieved-chunk inspection.~~ Complete — retrieval now exposes `RetrievalResult` diagnostics (`latency_ms`, `initial_candidate_count`) and the Streamlit UI surfaces retrieval and synthesis latency alongside retrieved context and answer confidence.
=======
5. ~~Improve the UI collection-selection and rollback workflow.~~ Complete — `COLLECTION_ROLES` map and `get_collection_role()` helper added; switching collections resets stale session state; rollback collection (`medical_research_chunks_v1`) soft-blocks ingestion. `pytest.ini` added to fix Windows temp-dir permission errors in CI.
6. ~~Add basic observability for latency and retrieved-chunk inspection.~~ Complete — `RetrievalResult` dataclass exposes `latency_ms` and `initial_candidate_count` via `retrieve_with_diagnostics()`; UI shows retrieval and synthesis latency captions.

## Stage-1 Synthesis Validation (pre–Stage-2 gate)

Full plan in `docs/stage1_synthesis_validation.md`. Must complete before the 50-PDF rebuild.

1. Create `data/eval/answer_quality_queries.json` (19 queries: factual, abstention-expected, hepcidin disambiguation). Status: query structure drafted in plan; `expected_doc_ids` to be filled in by user.
2. Run answer quality baseline on `medical_research_chunks_docling_v2_batch1`. Status: complete, with baseline metrics recorded in `data/eval/results/answer_quality_eval_stage1_baseline.json`.
3. Run known-gap abstention check through synthesis. Status: complete, with no known-gap query returning `HIGH` confidence in `data/eval/results/answer_quality_eval_known_gaps_stage1_after_abstention_pass2.json`.
4. Cross-document synthesis UI spot checks (8 queries). Status: pending.
5. Build `scripts/run_synthesis_gate.py` with pass/fail thresholds. Status: complete, and the stage-1 gate passes in `data/eval/results/synthesis_gate_report_stage1.json`.
>>>>>>> a5f3f07 (Add synthesis gate and record stage-1 pass)

## Future Considerations

- Hybrid or sparse retrieval only if benchmark evidence shows lexical recall failures.
- Multi-turn conversation support only after grounding and answer reliability are stronger.
- Larger-scale corpus rollout only after the `100`-PDF line is operationally stable.

## Checkpoint History

Detailed dated checkpoint notes were moved out of this file.

See:
- `docs/checkpoints.md`
