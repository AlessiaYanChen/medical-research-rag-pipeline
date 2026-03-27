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
- Hybrid dense+sparse retrieval, including BM25- or Qdrant-sparse-style lexical retrieval, and query expansion options such as synonym expansion or HyDE remain unevaluated roadmap options rather than active work; they should only be prioritized if benchmark evidence exposes recall gaps that metadata-first retrieval cannot cover
- The current benchmark is still vulnerable to author-style bias; retrieval should also be checked against clinician-style and out-of-distribution phrasing before scaling to the full corpus
- Parser changes remain unevaluated against downstream retrieval; parser experimentation should be isolated and benchmark-driven rather than folded directly into the active ingestion path

## Phase 1: Retrieval Quality Stabilization

Status: Complete

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

Status: Complete

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
- Current March 23, 2026 stable 26-query rerun on `medical_research_chunks_v1` shows:
  - expected doc hit rate: `1.0`
  - expected header hit rate: `1.0`
  - top-1 expected doc hit rate: `1.0`
  - top-1 expected header hit rate: `1.0`
  - average doc precision: `1.0`
  - average header precision: `1.0`
  - cross-document average doc precision: `1.0`
  - citation noise queries: `1`
  - table-hit queries: `4`
  - non-structural header queries: `0`
- Current March 23, 2026 expanded 43-query rerun on `medical_research_chunks_v1` shows:
  - expected doc hit rate: `1.0`
  - expected header hit rate: `1.0`
  - top-1 expected doc hit rate: `1.0`
  - top-1 expected header hit rate: `1.0`
  - average doc precision: `1.0`
  - average header precision: `1.0`
  - cross-document average doc precision: `1.0`
  - citation noise queries: `1`
  - table-hit queries: `6`
  - non-structural header queries: `0`
- Re-running the rebuilt collection on March 23, 2026 preserved perfect expected doc/header hit rates; follow-up narrow final-selection suppressions removed the remaining doc-filtered `Methods`, `Introduction`, and conclusion-tail `Results` noise, so the stable and expanded benchmarks now both sit at `1.0` average header precision on the current collection
- Current March 23, 2026 OOD 12-query rerun on `medical_research_chunks_v1` shows:
  - expected doc hit rate: `1.0`
  - expected header hit rate: `1.0`
  - top-1 expected doc hit rate: `1.0`
  - top-1 expected header hit rate: `1.0`
  - average doc precision: `1.0`
  - average header precision: `1.0`
  - cross-document average doc precision: `1.0`
  - citation noise queries: `0`
  - table-hit queries: `2`
  - non-structural header queries: `0`
- OOD reruns now resolve both the earlier singular contrastive stewardship-review ambiguity and the remaining cross-document/doc-filtered precision tails, so `O03`, `O05`, `O10`, and `O11` now behave cleanly on the current collection
- Benchmark metrics now explicitly include non-structural header hits so title-like or custom headers can be tracked as retrieval-quality debt
- A normalization pass now maps subsection/title/citation-like headers back to stable parent retrieval headers while preserving the original header in metadata
- Query-aware section weighting plus single-document metadata suppression materially improved section quality without reintroducing citation, table, or header-structure noise
- Preserving markdown table placement during parsing materially improved table retrieval and cross-document precision after re-ingestion
- The thematic-header chunker fix is now part of the ingestion baseline by normalizing markdown thematic headings back to stable retrieval sections while preserving the original header in metadata
- An experimental document-candidate retrieval stage was evaluated and removed because it underperformed the baseline on cross-document precision
- Cross-document precision on the current benchmark is now stabilized through singular-target document locking for title/trial/study queries plus explicit table-only and metric-table filtering for table-oriented retrieval
- Explicit `Table N` references are now preserved in chunk metadata so explicit table queries can still recover linked evidence when parser output leaves the table callout in narrative text
- `scripts/reingest_single_doc.py` now exists to repair one document in place without recreating the full collection, and it can now update the rebuild manifest entry during the same operation so the local corpus record does not drift immediately after a repair
- Table semantic metadata is now part of the ingestion baseline so rebuilt collections can filter metric/comparison tables from payload metadata instead of re-deriving table type in ranking code
- Table-context improvement now proceeds through explicit caption/prose linkage metadata rather than a positional "previous paragraph" heuristic, and returned table chunks now surface that linked context when the metadata is present
- On March 24, 2026, table-context coverage was broadened within the same metadata-linked design: ingestion can now link same-section prose back to a table even without a literal `Table N` reference when caption/table terminology overlaps strongly enough to support a narrow semantic attachment
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
2. Evaluate sparse lexical retrieval options explicitly, including BM25-style retrieval or Qdrant sparse vectors, on terminology-heavy failures where exact token match may matter more than dense semantic similarity
3. Evaluate query expansion approaches, including ontology-backed synonym expansion and HyDE, only on benchmark cases with confirmed abbreviation or synonym mismatch
4. Prefer explicit, auditable retrieval configuration over opaque expansion or branching behavior

Exit criteria:

- Recall extensions are justified by benchmark misses
- Added retrieval behavior stays observable and testable

## Phase 4: Metadata and Ingestion Hardening

Status: Complete

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
5. Keep the UI/local registry synchronized with rebuild manifests so document-level repairs do not silently desynchronize the local corpus record from Qdrant-backed collection state
Current checkpoint:

- document ID derivation is now centralized in code across rebuild, UI ingestion, single-document reingest, and local test flows; the repo still preserves the current stem-based naming scheme, but future naming changes can now be made in one place instead of script by script
- a collection audit script now compares Qdrant, the rebuild manifest, and the local registry, and it can sync the registry from the manifest before reporting so drift becomes observable and repairable instead of implicit
- manifest-aware repair and audit paths now validate collection name plus ingestion/chunking versions against the active code baseline, so mismatched manifests are surfaced explicitly instead of being reused silently, and malformed manifest JSON now fails fast with a clear audit/repair error instead of a traceback
- rebuild, UI ingestion, and single-document repair now reject duplicate `doc_id`, `source_file`, or `local_file` identities before they write new corpus state, so re-ingested files cannot silently create parallel document entries for the same source PDF identity
- the collection audit now reports duplicate identity conflicts across Qdrant, the rebuild manifest, and the local registry, it can write a non-destructive cleanup plan that recommends safe keep/drop actions only when metadata establishes a clear canonical `doc_id`, and `--fail-on-issues` now lets the same audit act as an explicit rollout gate
- the March 23, 2026 audit on `medical_research_chunks_v1` returned zero missing-doc, count-mismatch, or duplicate-identity issues, and the generated cleanup plan was empty
- on March 24, 2026, `scripts/rebuild_collection.py` gained batch-oriented hardening for medium-scale ingest work:
  - `--continue-on-error` now allows later PDFs in the batch to continue after a per-document failure
  - `--failure-report-out` can now write a structured JSON report for per-file rebuild failures
  - when `--manifest-out` is omitted, rebuilds now write to the collection-specific default path `data/ingestion_manifests/<collection>_rebuild_manifest.json`, keeping rebuild, repair, and audit workflows aligned by default
  - partial-success rebuilds still write a manifest for successful documents but exit nonzero when any failures occurred, keeping automation and follow-up repair work explicit
- on March 24, 2026, `scripts/reingest_single_doc.py` gained structured failure reporting for single-document repair attempts:
  - `--failure-report-out` can now write a JSON failure record for a repair attempt
  - failure stages are now surfaced explicitly across malformed-manifest and other manifest-validation failures, embedding preflight, parse, chunk, delete, upsert, and manifest update paths
- on March 24, 2026, setup hardening/onboarding documentation was tightened without changing retrieval behavior:
  - the checked-in `requirements.txt` and `.env.example` remain the base setup surface
  - `README.md` now documents clearer setup/run instructions for both PowerShell/Windows and bash/macOS/Linux
  - the required env vars for embeddings, Qdrant, and optional answer synthesis are now documented explicitly
  - `.env.example` no longer relies on `${...}` interpolation for Azure embedding settings, avoiding a PowerShell/env-loader footgun that broke Azure embedding calls during onboarding verification
  - the documented onboarding flow was verified by rerunning unit tests and the stable retrieval eval against `medical_research_chunks_v1`, preserving the existing clean `1.0` retrieval baseline

Exit criteria:

- Rebuilds are deterministic
- Metadata filters are available for large-corpus retrieval
- Ingestion versioning is in place and enforced
- The local-manifest registry has been replaced or hardened enough that corpus rollout is not dependent on a drift-prone local-only record

## Phase 4B: Parser Bakeoff

Status: In Progress

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

Current checkpoint:

- `experiments/parser_bakeoff.py` now exists as an isolated in-repo bakeoff runner for `Marker`, `Docling`, or both
- parser artifacts now write under `data/parser_bakeoff/artifacts/<parser>/...`
- parser summaries and a combined comparison file now write under `data/parser_bakeoff/results/...`
- bakeoff ingestion uses parser-specific collection names and remains isolated from `medical_research_chunks_v1`
- `src/adapters/parsing/docling_parser.py` now adapts `Docling` output into the same `ParsedDocument` / `ParsedTable` contract used by the current ingestion path, keeping parser experimentation narrow
- the current chunker and retrieval logic remain unchanged; the bakeoff still needs fixed-subset runs and benchmark comparison before any parser migration decision
- the first March 25, 2026 8-PDF subset bakeoff run now completed end to end for both parser-specific collections:
  - `Docling` parsed and ingested the subset successfully after adapting `TableItem.export_to_dataframe()` into the repo's parser contract
  - a narrow `Docling` markdown cleanup pass now strips image placeholders, collapses repeated opening boilerplate, and reduces obvious spacing/OCR artifacts before chunking so parser-side regression diagnosis can continue without changing retrieval logic
  - `Docling` opening structured-abstract prose is now normalized into an explicit `Structured Abstract` section before the real article body so top-of-paper summary blocks are easier to diagnose and less likely to dominate body evidence during the bakeoff
  - follow-up regression diagnosis showed most remaining stable/expanded `citation_noise_hits` came from inline numeric citation clusters surviving inside otherwise-valid body chunks, so the isolated `Docling` parser now also strips those citation runs before chunking without changing retrieval behavior or the production `Marker` path
  - follow-up parser-side work recovered the pathological Culture-Free LOD table from page text and tightened linked table context so the remaining Culture-Free table chunks are no longer dropped as near-duplicates during final selection
- the current isolated `Docling` bakeoff summary is now stable at `2781` chunks total, `2751` text chunks, and `30` table chunks
- citation-noise regressions are now fixed and the stable/expanded regression set has narrowed to `Q19` only; `Q05` and `Q18` no longer regress against `Marker`
- the remaining `Q19` miss is no longer the Culture-Free table path; it is currently concentrated in cross-document selection where duplicate `smith-et-al-2023-comparison-of-three-rapid-diagnostic-tests-for-bloodstream-infections-using-benefit-risk-evaluation` evidence still outranks an additional expected table-bearing document
- March 25, 2026 handoff checkpoint after isolated parser-side diagnosis:
  - commit `a61e605` fixes the remaining isolated `Docling` `Q19` regression without touching production retrieval or `medical_research_chunks_v1`
  - the fix is isolated in `experiments/parser_bakeoff.py` and strips Smith 2023 bakeoff-only markdown noise that was creating duplicate metric-heavy evidence under `Time-to-results` and `Discussion`
  - the isolated `Docling` rerun now reports `2762` chunks total, `2732` text chunks, and `30` table chunks
  - regenerated `data/parser_bakeoff/results/comparisons/sample_regressions_docling_vs_marker.json` and `expanded_regressions_docling_vs_marker.json` now both show `0` regressions
  - current parser-side recommendation remains unchanged: keep `Marker` as production and keep `Docling` isolated unless a later benchmark-backed migration decision is made
  - the current next step is targeted parser-side diagnosis on that duplicated Smith evidence path with parser-specific artifact and chunk comparison, not a production parser switch
- March 26, 2026 production-readiness checkpoint for a controlled `Docling` migration:
  - commit `9219af1` adds a selectable ingestion parser with `Marker` still as the default, wiring `Docling` into rebuild, reingest, single-PDF debug, end-to-end test, and UI ingestion entry points without changing retrieval behavior
  - parser provenance is now persisted in collection manifests and the local registry so `Docling`-built collections are explicit operationally
  - a new collection `medical_research_chunks_docling_v1` was rebuilt locally with parser `docling` over the current 7-document uploaded set:
    - `doc_count: 7`
    - `chunk_count: 2512`
  - local retrieval evaluation against `medical_research_chunks_docling_v1` showed no benchmark blocker on the checked stable sets:
    - `sample_queries.json`: expected doc hit `1.0`, expected header hit `1.0`, top-1 doc/header `1.0`, average doc precision `0.9923`
    - `expanded_queries.json`: expected doc hit `1.0`, expected header hit `1.0`, top-1 doc/header `1.0`, average doc precision `0.9953`
  - manual spot-checking then exposed one real production-style regression on the query `What confirmation rate was achieved for Staphylococcus aureus by culture or PCR in the IRIDICA study?`
    - `Docling` did contain the right BAL SM evidence
    - the failure was retrieval-stage ranking and table eligibility, not missing parser content
  - commit `f5b3138` narrows retrieval heuristics for metric-style confirmation/rate questions so result/table evidence is surfaced correctly for that BAL/IRIDICA case without broad retrieval tuning
  - unit tests now pass after the retrieval fix with `162 passed`
  - current migration recommendation:
    - `Docling` is now viable for a controlled cutover to `medical_research_chunks_docling_v1`
    - keep `medical_research_chunks_v1` available as rollback
    - do not commit local generated state such as `data/eval/results/*`, `data/kb_registry.json`, or local collection manifests; document the checkpoint instead
  - follow-up manual UI testing on the active `Docling` collection has now been folded into `data/eval/runtime_queries.json`, which currently holds 28 real runtime queries
  - recent runtime-driven retrieval fixes stayed narrow and benchmark-backed:
    - contrastive document selection now resolves both `turnaround improvements, not stewardship policy` and `optimizing blood culture use rather than reporting rapid test outcomes`
    - BAL-specific single-study wording such as `the BAL IRIDICA study` is now treated as a single-document target even without an explicit doc filter, preventing unrelated FLAT/stewardship chunks from leaking into that query family
  - the current runtime benchmark on `medical_research_chunks_docling_v1` is locally clean at the document level:
    - `runtime_queries.json`: expected doc hit `1.0`, expected header hit `1.0`, top-1 expected doc hit `1.0`
  - latest unit-test checkpoint after those runtime fixes: `171 passed`
- March 27, 2026 runtime-driven retrieval checkpoint after a new manual UI batch on `medical_research_chunks_docling_v1`:
  - `data/eval/runtime_queries.json` now holds `34` real runtime queries after adding `R29` through `R34`
  - `data/eval/known_gap_queries.json` now holds `12` queries after adding `K12` for the Fabre low-diagnostic-value scenario-list question, which remains table-dependent and should stay out of the runtime regression set
  - commit `7303829` tunes retrieval ranking for explanatory and mechanistic prompts without changing retrieval architecture:
    - fixes the repeated `R31` miss on the Lin review `three main advantages of mNGS` query by lifting answer-bearing `Methods` evidence over generic introduction summaries
    - materially improves `R33` by surfacing BAL abstract mechanism evidence for the PCR/ESI-MS versus FLAT workflow comparison, while leaving the FLAT side intact
    - unit tests passed at `173 passed`
  - a follow-up narrow ranking change for `R32` improves the Banerjee mortality-comparison path by preferring outcome-bearing tables over RAPID demographics/characteristics tables:
    - the first returned chunk for `R32` is now the single-site RCT outcome table rather than a RAPID demographics table
    - the RAPID mortality table is still present in ranked candidates but does not yet make the final returned set, so this is a partial improvement rather than a full fix
    - unit tests pass locally at `174 passed`
  - the current runtime benchmark after these March 27 changes is:
    - `runtime_queries.json`: expected doc hit `1.0`, expected header hit `1.0`, top-1 expected doc hit `0.9706`, top-1 expected header hit `0.8529`
  - current diagnosis of the remaining March 27 runtime misses:
    - `R32` is a table-selection problem and should be handled separately from prose-ranking fixes
    - `R34` is not just ranking noise; it includes a Nartey-side candidate-recall weakness for the specific urinalysis-limitation prose about dipstick positive interferences and unnecessary downstream culture
  - current recommendation:
    - keep the explanatory-ranking fix
    - keep the query-set growth from real UI usage
    - do not attempt broad retrieval tuning
    - inspect `R34` as the next targeted retrieval problem because it is the first recent runtime case with a measured candidate-recall weakness rather than only final-ranking noise

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

1. Keep the stable and expanded benchmark records separate and treat the current `1.0` header-precision state as the regression baseline before adding more retrieval logic
2. Keep the OOD/adversarial phrasing file as a separate evaluation-only track and review its expectations manually before it is used to justify retrieval changes
3. Keep expanding the runtime regression set from real app usage, preferably `data/eval/runtime_queries.json`, before reopening retrieval architecture work:
   - the set now exists and should continue to grow from real UI usage rather than synthetic brainstorming
   - include both successes and failures
   - keep covering exact metric/rate questions, study-identification prompts, caveat queries, and abbreviation-heavy wording
4. Use `scripts/inspect_retrieval_candidates.py` on any new OOD or runtime-set misses before changing ranking logic so candidate-recall problems are separated from document- or chunk-ranking problems
5. Keep any future retrieval changes narrow, metadata-first, and benchmark-backed; do not add extra embedding stages, hybrid retrieval, or query expansion unless the runtime regression set shows measured recall gaps that require them
6. Keep setup hardening moving:
   - maintain the checked-in `requirements.txt`
   - maintain the checked-in `.env.example`
   - keep the clearer cross-platform setup docs aligned with the actual script/runtime behavior as setup changes land
7. Keep table-context coverage improvements narrow and metadata-linked so more returned table chunks carry caption/prose context without positional heuristics or new retrieval stages
8. Harden corpus metadata and rebuild workflows for medium-scale ingestion
9. Keep using `scripts/audit_collection_state.py --fail-on-issues` plus cleanup-plan output as the explicit pre-rollout corpus integrity check before Phase 5 work or any medium-scale ingest batch
10. Run the isolated parser bakeoff in-repo before Phase 5 corpus rollout work grows expensive to redo
11. Reconsider document-level retrieval, hybrid retrieval, query expansion, or parser migration only if benchmark evidence shows the current metadata-first baseline has stopped holding
