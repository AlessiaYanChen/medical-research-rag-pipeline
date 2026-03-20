# Roadmap

## Goal

Make the medical research RAG pipeline reliable enough to support a corpus of roughly 300 PDFs without collapsing into noisy, redundant, citation-heavy retrieval.

## Current State

Implemented:

- Azure/OpenAI embedding adapter
- Parent-child retrieval
- Narrow parent return windows around matched child spans
- Structured retrieval results
- Text-first retrieval by default
- Low-value section filtering for references and front matter
- Optional table inclusion
- Basic section-aware ranking
- Stronger duplicate suppression and section diversity caps
- Retrieval evaluation harness with JSON/CSV export
- Starter benchmark query set for multi-paper validation
- Page-1/title-like opening header normalization to `Document Metadata/Abstract`
- Header normalization metadata including original vs normalized parent headers and header-role tagging

Current observed issues:

- Corpus management is still local-manifest based and not robust for large-scale ingestion
- The benchmark still needs continued expectation refinement as broader coverage surfaces edge cases

## Phase 1: Retrieval Quality Stabilization

Status: In progress

Objectives:

- Keep top-k evidence non-overlapping and diverse
- Prefer Results, Discussion, and Conclusion over Abstract when appropriate
- Keep tables out of default retrieval unless the query is table-oriented

Tasks:

1. Maintain the current section-ranking baseline and avoid regressions
2. Tune section diversity across Results, Discussion, and Conclusion
3. Validate top-k behavior across more than one paper
4. Confirm the current diversity caps hold up for multi-document retrieval

Exit criteria:

- Top 3-5 results are mostly non-overlapping
- Discussion/Results dominate for evidence-seeking clinical queries
- No false prose-as-table rendering in the UI

## Phase 2: Evaluation and Corpus Validation

Status: In progress

Objectives:

- Stop tuning retrieval blindly
- Validate retrieval quality before scaling to the full 300-PDF corpus

Tasks:

1. Expand the benchmark beyond the current starter set
2. Create 10-20 real research questions with expected evidence
3. Measure:
   - relevance
   - redundancy
   - section quality
   - table noise
   - citation noise
4. Track regressions after retrieval changes

Current checkpoint:

- `scripts/evaluate_retrieval.py` exists and writes JSON/CSV reports
- `data/eval/sample_queries.json` is in use as a starter evaluation set
- The stable retrieval baseline remains the 26-query `data/eval/sample_queries.json` dataset
- `data/eval/expanded_queries.json` now extends coverage to 43 queries across stewardship, review-style, title-query, and table-oriented retrieval
- Latest eval run on `medical_research_chunks_v1` showed:
  - expected doc hit rate: `1.0`
  - expected header hit rate: `1.0`
  - top-1 expected doc hit rate: `1.0`
  - top-1 expected header hit rate: `1.0`
  - average doc precision: `1.0`
  - average header precision: `0.8795`
  - cross-document average doc precision: `1.0`
  - citation noise queries: `1`
  - table-hit queries: `7`
  - non-structural header queries: `0`
- Benchmark metrics now explicitly include non-structural header hits so title-like or custom headers can be tracked as retrieval-quality debt
- A normalization pass now maps subsection/title/citation-like headers back to stable parent retrieval headers while preserving the original header in metadata
- Query-aware section weighting plus single-document metadata suppression materially improved section quality without reintroducing citation, table, or header-structure noise
- Preserving markdown table placement during parsing materially improved table retrieval and cross-document precision after re-ingestion
- The thematic-header chunker fix is now part of the ingestion baseline by normalizing markdown thematic headings back to stable retrieval sections while preserving the original header in metadata
- An experimental document-candidate retrieval stage was evaluated and removed because it underperformed the baseline on cross-document precision
- Cross-document precision on the current benchmark is now stabilized through singular-target document locking for title/trial/study queries plus explicit table-only and metric-table filtering for table-oriented retrieval
- Explicit `Table N` references are now preserved in chunk metadata so explicit table queries can still recover linked evidence when parser output leaves the table callout in narrative text
- `scripts/reingest_single_doc.py` now exists to repair one document in place without recreating the full collection

Exit criteria:

- Retrieval changes are judged against repeatable evidence
- Scaling decisions are based on metrics, not screenshots alone

## Phase 3: Document-Level Retrieval

Status: Deferred

Objectives:

- Scale retrieval beyond single-document testing
- Avoid collection-wide chunk competition across hundreds of PDFs

Tasks:

1. Add a document-level candidate retrieval stage
2. Retrieve chunks only within top candidate documents
3. Add optional document-level metadata filters
4. Add collection and corpus scoping in the retrieval service

Exit criteria:

- Retrieval remains relevant across the target corpus size
- Cross-document noise is materially reduced

## Phase 4: Metadata and Ingestion Hardening

Status: Planned

Objectives:

- Make ingestion reproducible and corpus-safe
- Store enough metadata to support filtering, auditing, and rebuilds

Tasks:

1. Standardize doc IDs and collection naming
2. Persist richer metadata:
   - title
   - file path
   - section role
   - chunk type
   - content role
   - page number
   - parent ID
3. Add ingestion versioning for chunking and embedding changes
4. Add dedup detection for re-ingested files

Exit criteria:

- Rebuilds are deterministic
- Metadata filters are available for large-corpus retrieval

## Phase 5: Corpus Rollout

Status: Planned

Objectives:

- Ingest and serve a few-hundred-document knowledge base

Tasks:

1. Run ingestion in batches
2. Add failure logging and retry handling
3. Add collection management and reconciliation with local registry
4. Validate Qdrant sizing and embedding cost assumptions

Exit criteria:

- Corpus ingestion is operationally manageable
- Retrieval remains usable at target corpus size

## Near-Term Next Moves

Recommended next implementation order:

1. Treat the current 26-query set as the stable retrieval baseline and the 43-query set as the active expansion track
2. Refine expectations where expanded benchmark cases still expose header-quality ambiguity before adding new ranking heuristics
3. Harden corpus metadata and rebuild workflows for medium-scale ingestion
4. Add ingestion/version metadata so single-doc repairs are auditable
5. Reconsider document-level retrieval only if cross-document precision stops holding at the expanded benchmark level
