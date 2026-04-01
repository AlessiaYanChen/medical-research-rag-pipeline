# Roadmap

## Goal

Make the medical research RAG pipeline reliable enough for a medium-scale corpus of roughly `100 PDFs`, then scale toward roughly `300 PDFs` without noisy retrieval, weak grounding, or unclear rollback paths.

## Current State

- Active parser: `Docling`
- Rollback parser: `Marker`
- Current small-corpus baseline collection: `medical_research_chunks_docling_v1`
- Current stage-1 rollout collection: `medical_research_chunks_docling_v2_batch1`
- Stage-1 formal gate status: `pass`
- Stage-1 substantive risk: evaluation coverage still only exercises the original `7` baseline papers, not the full `20`-PDF corpus

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

1. Coverage gap: current eval datasets still cover only `7` of `20` stage-1 papers.
2. Same-topic ambiguity: the hepcidin cluster is the clearest early multi-paper disambiguation risk.
3. Parser blind spots: selected new papers show `0` table chunks and need PDF spot checks.
4. Answer reliability: the app measures retrieval quality but not final synthesis quality.
5. Operational drift: dependencies are unpinned, so parser or reranker behavior can shift silently.

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
- do not treat stage 1 as fully de-risked yet; close the coverage gap on the `13` newly added papers before stage 2

Stage-1 coverage completion tasks:

1. Create a separate synthetic coverage dataset, for example `data/eval/stage1_coverage_queries.json`.
2. Add at least `1-2` retrieval queries per newly added paper.
3. Add cross-document queries that mix baseline and newly added papers.
4. Add hepcidin-cluster disambiguation queries explicitly.
5. Run `8-10` additional manual UI spot checks focused on newly added papers.
6. Spot-check selected zero-table-chunk papers against the source PDFs, starting with `1-s2.0-S0009912024000250-main` and `hepcidin diagnostic tool`.
7. Add confirmed new-paper misses to `known_gap_queries.json` rather than mixing them into the runtime baseline.

Stage-2 readiness rule:

- keep `runtime_queries.json` limited to real user/runtime questions
- put synthetic corpus-coverage probes into a separate evaluation dataset
- do not begin stage 2 until the stage-1 coverage gap is closed or the risk is accepted explicitly

### Phase 5: Corpus Rollout

Status: Planned

Objective:
- expand beyond `100 PDFs` only after Phase 5A is clean

Phase gate:
- do not begin broader corpus rollout until medium-scale readiness is proven

## Next Priorities

Immediate stabilization work:

1. Pin dependencies to reduce parser and reranker drift risk.

Order of work after the stage-1 coverage gap is closed:

1. Harden the medical research prompt for study design, effect sizes, uncertainty, and limitations.
2. Move toward structured answer output with chunk-level citations.
3. Add a typed abstention or confidence signal.
4. Add a small answer-quality evaluation layer separate from retrieval evaluation.
5. Improve the UI collection-selection and rollback workflow.
6. Add basic observability for latency and retrieved-chunk inspection.

## Future Considerations

- Hybrid or sparse retrieval only if benchmark evidence shows lexical recall failures.
- Multi-turn conversation support only after grounding and answer reliability are stronger.
- Larger-scale corpus rollout only after the `100`-PDF line is operationally stable.

## Checkpoint History

Detailed dated checkpoint notes were moved out of this file.

See:
- `docs/checkpoints.md`
