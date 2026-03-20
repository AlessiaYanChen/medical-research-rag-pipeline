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
- Metadata-first retrieval filtering through Qdrant payload filters
- Deterministic collection rebuild tooling via `scripts/rebuild_collection.py`
- Table semantic metadata including metric/comparison flags and lightweight captions

Current observed issues:

- Corpus management is still local-manifest based and not robust for large-scale ingestion
- The benchmark still needs continued expectation refinement as broader coverage surfaces header-quality edge cases
- Header precision and table-hit behavior should be revalidated explicitly after the latest ingestion/retrieval changes before more retrieval logic is added
- Hybrid dense+sparse retrieval and ontology-backed query expansion remain unevaluated roadmap options rather than active work; they should only be prioritized if benchmark evidence exposes recall gaps that metadata-first retrieval cannot cover
- The current benchmark is still vulnerable to author-style bias; retrieval should also be checked against clinician-style and out-of-distribution phrasing before scaling to the full corpus
- Parser changes remain unevaluated against downstream retrieval; parser experimentation should be isolated and benchmark-driven rather than folded directly into the active ingestion path

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
3. Add a separate out-of-distribution phrasing track:
   - clinician-style journal-club questions
   - abbreviation-heavy questions
   - paraphrased variants that avoid repo-specific wording
   - adversarial wording reviewed manually before use
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
- `data/eval/ood_adversarial_queries.json` now defines a separate clinician-style and adversarial phrasing track for evaluation only
- Current recorded stable 26-query baseline on `medical_research_chunks_v1` shows:
  - expected doc hit rate: `1.0`
  - expected header hit rate: `1.0`
  - top-1 expected doc hit rate: `1.0`
  - top-1 expected header hit rate: `1.0`
  - average doc precision: `0.9923`
  - average header precision: `0.7724`
  - cross-document average doc precision: `0.975`
  - citation noise queries: `1`
  - table-hit queries: `4`
  - non-structural header queries: `0`
- Current recorded expanded 43-query benchmark on `medical_research_chunks_v1` shows:
  - expected doc hit rate: `1.0`
  - expected header hit rate: `1.0`
  - top-1 expected doc hit rate: `1.0`
  - top-1 expected header hit rate: `1.0`
  - average doc precision: `0.9953`
  - average header precision: `0.8578`
  - cross-document average doc precision: `0.99`
  - citation noise queries: `1`
  - table-hit queries: `6`
  - non-structural header queries: `0`
- Re-running the rebuilt collection on March 20, 2026 matched the same expanded-benchmark summary, so the current metadata-first ingestion baseline held after re-ingestion
- OOD reruns on March 20, 2026 now resolve the previously unresolved singular contrastive stewardship-review queries, so `O03` and `O10` both return the Fabre stewardship review in top-1 after a narrow document-level disambiguation step
- Benchmark metrics now explicitly include non-structural header hits so title-like or custom headers can be tracked as retrieval-quality debt
- A normalization pass now maps subsection/title/citation-like headers back to stable parent retrieval headers while preserving the original header in metadata
- Query-aware section weighting plus single-document metadata suppression materially improved section quality without reintroducing citation, table, or header-structure noise
- Preserving markdown table placement during parsing materially improved table retrieval and cross-document precision after re-ingestion
- The thematic-header chunker fix is now part of the ingestion baseline by normalizing markdown thematic headings back to stable retrieval sections while preserving the original header in metadata
- An experimental document-candidate retrieval stage was evaluated and removed because it underperformed the baseline on cross-document precision
- Cross-document precision on the current benchmark is now stabilized through singular-target document locking for title/trial/study queries plus explicit table-only and metric-table filtering for table-oriented retrieval
- Explicit `Table N` references are now preserved in chunk metadata so explicit table queries can still recover linked evidence when parser output leaves the table callout in narrative text
- `scripts/reingest_single_doc.py` now exists to repair one document in place without recreating the full collection
- Table semantic metadata is now part of the ingestion baseline so rebuilt collections can filter metric/comparison tables from payload metadata instead of re-deriving table type in ranking code
- Table-context improvement should proceed through explicit caption/prose linkage metadata rather than a positional "previous paragraph" heuristic
- A new diagnostic script, `scripts/inspect_retrieval_candidates.py`, now exists to inspect initial search hits, post-filter candidates, ranked candidates, and final returned chunks for one query before changing ranking logic
- Current OOD inspection established that the stewardship-review miss was not a candidate-recall failure: the Fabre paper was already present in early candidates, and a narrow document-level disambiguation step was sufficient to resolve `O03` and `O10`
- The next retrieval step should not add extra embedding stages, hybrid retrieval, or query expansion; header-precision and table-hit drift should be diagnosed first, and any new behavior should stay similarly narrow and benchmark-backed

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

## Phase 3B: Recall Extensions

Status: Deferred

Objectives:

- Add recall-oriented retrieval features only if benchmark evidence shows a real lexical or synonym gap
- Avoid introducing hidden query-policy complexity before the current metadata-first retrieval split is exhausted

Tasks:

1. Evaluate whether hybrid dense+sparse retrieval improves recall on measured failures rather than hypothetical scale concerns
2. Evaluate ontology-backed query expansion only on benchmark cases with confirmed abbreviation or synonym mismatch
3. Prefer explicit, auditable retrieval configuration over opaque expansion or branching behavior

Exit criteria:

- Recall extensions are justified by benchmark misses
- Added retrieval behavior stays observable and testable

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
   - canonical/original header variants
   - ingestion/chunking version
   - table semantics and captions
3. Add ingestion versioning for chunking and embedding changes
4. Add dedup detection for re-ingested files

Exit criteria:

- Rebuilds are deterministic
- Metadata filters are available for large-corpus retrieval
- Ingestion versioning is in place and enforced
- The local-manifest registry has been replaced or hardened enough that corpus rollout is not dependent on a drift-prone local-only record

## Phase 4B: Parser Bakeoff

Status: Deferred

Objectives:

- Compare candidate parsers inside this repo without destabilizing the active ingestion workflow
- Judge parser changes by downstream retrieval quality, not parsing aesthetics alone

Tasks:

1. Add isolated parser bakeoff tooling in this repo, preferably via a script or `experiments/` path
2. Run a fixed PDF subset through `Marker`, `Docling`, and any lightweight fallback parser candidates
3. Store parser outputs in separate artifact folders and ingest into separate Qdrant collections
4. Compare:
   - header quality
   - table extraction fidelity
   - caption or linked-prose recovery
   - downstream benchmark metrics
5. Prefer a clean parser replacement over a permanent blended parser pipeline unless a combined approach is deterministic and benchmark-backed

Exit criteria:

- Parser choice is justified by retrieval evidence
- Experimental parser work does not interfere with active ingestion or the production collection
- Parser comparison is completed before any large Phase 5 corpus rollout that would otherwise require expensive re-ingestion

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

Phase gate:

- Do not begin Phase 5 until Phase 4 exit criteria are met, specifically ingestion versioning plus registry hardening/replacement

## Near-Term Next Moves

Recommended next implementation order:

1. Diagnose any header-precision and table-hit drift on the stable 26-query baseline and the expanded 43-query benchmark before adding more retrieval logic
2. Keep the OOD/adversarial phrasing file as a separate evaluation-only track and review its expectations manually before it is used to justify retrieval changes
3. Use `scripts/inspect_retrieval_candidates.py` on any new OOD misses before changing ranking logic so candidate-recall problems are separated from document- or chunk-ranking problems
4. Keep any future retrieval changes narrow, metadata-first, and benchmark-backed; do not add extra embedding stages, hybrid retrieval, or query expansion unless measured recall gaps require them
5. Add setup hardening in parallel:
   - `requirements.txt` or equivalent install source
   - `.env.example`
   - clearer cross-platform setup docs
6. Add metadata-linked table caption/prose context so table hits can carry better reasoning context without positional heuristics
7. Harden corpus metadata and rebuild workflows for medium-scale ingestion
8. Run the isolated parser bakeoff in-repo before Phase 5 corpus rollout work grows expensive to redo
9. Reconsider document-level retrieval, hybrid retrieval, query expansion, or parser migration only if benchmark evidence shows the current metadata-first baseline has stopped holding
