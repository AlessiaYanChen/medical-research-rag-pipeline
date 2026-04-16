# Roadmap

## Goal

Make the medical research RAG pipeline reliable and benchmark-honest on the current `20`-PDF stage-1 corpus, then use that smaller corpus as the proving ground for any later expansion toward `100+` PDFs.

## Current State

- Active parser: `Docling`
- Rollback parser: `Marker`
- Current small-corpus baseline collection: `medical_research_chunks_docling_v1`
- Current stage-1 rollout collection: `medical_research_chunks_docling_v2_batch1`
- Stage-1 formal gate status: `pass`
- Stage-1 synthesis validation: complete — answer quality baseline exists, the known-gap false-confidence gate is clear, the stage-1 synthesis gate passes, and the synthesis spot checks passed `6/8` with no hepcidin mis-attribution

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

1. Same-topic ambiguity: the hepcidin cluster is still the clearest early multi-paper disambiguation risk; one residual top-1 ambiguity is under watch at the synthesis level.
2. Residual synthesis watch items: the stage-1 synthesis gate is clear. `AQ10` improved in the `2026-04-10` follow-up reruns, while `AQ13` still remains the main study-design classification watch item for Stage 2.
3. Ingestion metadata hardening is still the main operational risk before any corpus expansion beyond the current `20`-PDF stage.

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

Use only if benchmark evidence shows the current metadata-first retrieval path has stopped holding. Do not reopen this phase for the current `20`-PDF corpus unless a measured benchmark gap clearly requires it.

### Phase 3B: Recall Extensions

Status: Deferred

Hybrid retrieval, sparse retrieval, query expansion, and similar recall extensions stay deferred until benchmark evidence shows real lexical recall gaps.

### Phase 4: Metadata and Ingestion Hardening

Status: In Progress

Outcome so far:
- rebuild, audit, manifest, duplicate detection, and repair tooling are in place
- shared ingestion runtime helpers now live under `src/app/ingestion`, so rebuild, repair, UI ingest, and the end-to-end flow no longer depend on a test-named script for runtime behavior
- Qdrant write failures now fail loud instead of logging and continuing, which closes the silent partial-write hole during rebuild and single-document repair
- single-document repair now snapshots and restores the prior Qdrant points on replacement-write failure, reducing the chance of a failed repair leaving the collection without the original document
- collection rebuild now defaults to a fresh-target safety check and requires an explicit override before recreating an existing collection, while alias promotion is available as a separate step after staged validation
- rollout reporting can now carry the follow-up promotion command when a staged collection fully passes its gates, making the checklist handoff from validation to cutover explicit
- rebuild, UI ingest, and single-document repair now stamp `ingestion_version`, `chunker_version`, `source_sha256`, and `file_size_bytes` consistently into chunk metadata, rebuild manifests, and registry entries while keeping the legacy `chunking_version` field for compatibility
- `scripts/audit_collection_state.py` now also flags chunk-count sanity failures such as empty docs, broken text/table breakdowns, and extreme per-doc chunk-count outliers so parser pathologies surface before rollout promotion
- `scripts/audit_collection_state.py` now also reports per-doc metadata mismatches across Qdrant, manifest, and registry plus a small repair-plan payload for registry sync, metadata review, count review, and parser inspection follow-up

Next hardening work for the `20`-PDF corpus:
- keep human-readable `doc_id` values for UI and evaluation, while treating a content hash as the canonical dedup and file-identity signal

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
- prove the current `20`-PDF `Docling` artifact is stable enough to serve as the benchmark and ingestion reference point before any broader rollout work resumes

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
- stage-1 retrieval coverage gap is closed; the synthesis validation gate in `docs/stage1_synthesis_validation.md` is now passed, and broader stage-2 work should keep `AQ13` visible as the remaining synthesis watch item

### Phase 5: Corpus Rollout

Status: Planned

Objective:
- expand beyond `100 PDFs` only after Phase 5A is clean

Phase gate:
- do not begin broader corpus rollout until medium-scale readiness is proven

## Next Priorities

Immediate focus for the current `20`-PDF stage:

1. Harden ingestion metadata and reconciliation before any new PDF expansion work: version stamps, richer file identity, registry-vs-Qdrant diffing, and chunk-count sanity checks.
2. Inspect the remaining `AQ13` miss once more before any further retrieval tuning. Only change retrieval if a narrow cause is confirmed.
3. Inspect the remaining top-1 header misses before any further retrieval tuning. Treat `top1_expected_header_hit_rate` as a watch metric, not a cue for proactive heuristic changes.
4. Keep Phase 3 deferred and keep document locking as the narrow query-disambiguation tool for the `20`-PDF corpus.
5. Freeze embedding configuration until the benchmark is larger and can detect regressions honestly.

Detailed stage-1 query expansion matrix:
- `docs/stage1_benchmark_expansion_plan.md`

Current benchmark read after stage-1 coverage expansion:
- `data/eval/stage1_coverage_queries.json` now contains `88` retrieval queries across all `20` stage-1 papers.
- Current `2026-04-10` rerun on `medical_research_chunks_docling_v2_batch1`:
  - `expected_doc_hit_rate = 1.0`
  - `top1_expected_doc_hit_rate = 1.0`
  - `average_doc_precision = 1.0`
  - `cross_document_average_doc_precision = 1.0`
  - `expected_header_hit_rate = 1.0`
  - `top1_expected_header_hit_rate = 0.8182`
  - `average_header_precision = 0.793`
- On the expanded stage-1 set, document recall, cross-document precision, and at-least-one-hit header coverage are now clean. Remaining retrieval work is mostly top-1 section selection rather than coverage gaps.
- Operational next step on the `20`-PDF corpus: complete ingestion metadata hardening before any Stage-2 rebuild or corpus expansion work.

Previously completed stabilization work:

1. ~~Pin dependencies to reduce parser and reranker drift risk.~~ Complete — all packages pinned in `requirements.txt`, including `docling` which was previously missing from the file.

Order of work (stage-1 coverage gap is now closed):

1. ~~Harden the medical research prompt for study design, effect sizes, uncertainty, and limitations.~~ Complete — prompt now instructs study design identification, exact effect size reporting, CI/p-value inclusion, and limitation noting.
2. ~~Move toward structured answer output with chunk-level citations.~~ Complete — `ReasoningService.research()` now returns `ResearchAnswer` (insight, evidence_basis, citations). Citations are the actual retrieved chunks, not LLM-generated text. UI renders them as collapsible expanders.
3. ~~Add a typed abstention or confidence signal.~~ Complete — `ConfidenceLevel` enum (HIGH/MEDIUM/LOW/INSUFFICIENT) derived from retrieval signals (chunk count, distinct docs, "Insufficient evidence" in insight). UI shows a coloured banner per level.
4. ~~Add a small answer-quality evaluation layer separate from retrieval evaluation.~~ Complete — `src/app/evaluation/answer_quality_eval.py` evaluates abstention accuracy, confidence thresholds, and doc-ID coverage in evidence basis. Runner at `scripts/evaluate_answer_quality.py`. Query format: `data/eval/answer_quality_queries.json`.
5. ~~Improve the UI collection-selection and rollback workflow.~~ Complete — `COLLECTION_ROLES` map and `get_collection_role()` helper added; switching collections resets stale session state; rollback collection (`medical_research_chunks_v1`) soft-blocks ingestion. `pytest.ini` added to fix Windows temp-dir permission errors in CI.
6. ~~Add basic observability for latency and retrieved-chunk inspection.~~ Complete — `RetrievalResult` dataclass exposes `latency_ms` and `initial_candidate_count` via `retrieve_with_diagnostics()`; UI shows retrieval and synthesis latency captions.

## Stage-1 Synthesis Validation (pre–Stage-2 gate)

Full plan in `docs/stage1_synthesis_validation.md`. Must complete before the 50-PDF rebuild.

1. Create and expand `data/eval/answer_quality_queries.json` to cover factual answers, known-gap abstentions, hepcidin disambiguation, and plausible-but-absent false-positive traps. Status: complete for the initial expanded stage-1 set.
   - Current expanded run: `31` queries, `abstain_accuracy = 1.0`, `confidence_meets_minimum_rate = 0.8125`, `average_doc_id_coverage = 0.9821` in `data/eval/results/answer_quality_eval_stage1_expanded_absent_answer_v3.json`.
2. Run answer quality baseline on `medical_research_chunks_docling_v2_batch1`. Status: complete, with baseline metrics recorded in `data/eval/results/answer_quality_eval_stage1_baseline.json`.
3. Run known-gap abstention check through synthesis. Status: complete, with no known-gap query returning `HIGH` confidence in `data/eval/results/answer_quality_eval_known_gaps_stage1_after_abstention_pass2.json`.
4. Cross-document synthesis UI spot checks (8 queries). Status: complete, with `6/8` pass and no hepcidin mis-attribution recorded in `docs/manual_spot_checks_synthesis_2026-04-01.md`.
5. Build `scripts/run_synthesis_gate.py` with pass/fail thresholds. Status: complete, and the stage-1 gate passes in `data/eval/results/synthesis_gate_report_stage1.json`.

Latest follow-up read on `2026-04-10` after targeted retrieval adjustments:
- `data/eval/results/answer_quality_eval_stage1_watch_items_v8.json` restored `abstain_accuracy = 1.0`, kept `confidence_meets_minimum_rate = 0.8125`, and raised `average_doc_id_coverage` to `0.9911`.
- `AQ08`, `AQ10`, `AQ19`, and `AQ23` are all in a healthy state again.
- `AQ13` improved to `6/7` expected evidence docs with correct `RAPID` coverage, but it still misses the Nartey urine paper and remains the only active synthesis watch item from this follow-up pass.

## Future Considerations

- Hybrid or sparse retrieval only if benchmark evidence shows lexical recall failures.
- Multi-turn conversation support only after grounding and answer reliability are stronger.
- Larger-scale corpus rollout only after the `20`-PDF benchmark is harder, the ingestion metadata is better hardened, and the smaller-corpus signal remains clean.

## Checkpoint History

Detailed dated checkpoint notes were moved out of this file.

See:
- `docs/checkpoints.md`
