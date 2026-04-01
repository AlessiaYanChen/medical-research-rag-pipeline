# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Role

You are a Senior AI Systems Architect specialized in high-integrity, evidence-based RAG systems for medical research. Every summary must include a direct reference to the Source ID. If an answer cannot be found in retrieved context, state "Insufficient evidence."

## Commands

### Environment Setup
```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # then fill in API keys
docker run -p 6333:6333 qdrant/qdrant  # start Qdrant
```

### Testing
```bash
pytest tests/unit/                         # all unit tests
pytest tests/unit/test_chunker.py -v       # single test file
pytest tests/unit/ --cov=src               # with coverage
```

### Development Scripts
```bash
# Parse and ingest a single PDF end-to-end
python scripts/test_e2e_flow.py --pdf data/raw_pdfs/<file>.pdf --query "..." --parser docling --collection <collection>

# Validate a single PDF parses correctly
python scripts/test_single_pdf.py --pdf data/raw_pdfs/<file>.pdf --parser docling

# Batch rebuild a collection
python scripts/rebuild_collection.py --pdf-dir data/raw_pdfs/ --collection <collection> --parser docling --continue-on-error

# Audit for duplicates and missing metadata
python scripts/audit_collection_state.py --qdrant-url http://localhost:6333 --collection <collection> --fail-on-issues

# Debug retrieval candidates
python scripts/inspect_retrieval_candidates.py --query "..." --collection <collection> --initial-limit 20

# Re-ingest a single broken document
python scripts/reingest_single_doc.py --pdf data/raw_pdfs/<file>.pdf --collection <collection> --parser docling

# Run retrieval benchmark
python scripts/evaluate_retrieval.py --query-file data/eval/sample_queries.json --collection <collection> --output data/eval/results/retrieval_eval.json

# Build rollout gate report
python scripts/build_rollout_report.py --baseline-file data/eval/results/retrieval_eval_baseline.json --current-file data/eval/results/retrieval_eval_current.json --output data/eval/results/rollout_report.json

# Streamlit UI
streamlit run scripts/ui_app.py
```

## Architecture

The system follows **Hexagonal Architecture** (Ports & Adapters):

- **Domain** (`src/domain/`): Immutable `Chunk` and `ChunkMetadata` dataclasses. The `extra` dict on `ChunkMetadata` carries semantic tags (`parent_id`, table captions, metric flags, linked prose).
- **Ports** (`src/ports/`, `src/app/ports/`): Abstract contracts for `ParserPort`, `VectorRepository`, `LLMPort`, `ReRankerPort`.
- **Adapters** (`src/adapters/`, `src/app/adapters/`): Concrete implementations swappable without touching core logic.
- **Application Services** (`src/app/services/`, `src/app/tables/`, `src/app/ingestion/`): Orchestration only.

### End-to-End Data Flow
```
PDF → DoclingParser → Markdown + Tables
    → TableNormalizer (row-density heuristics)
    → UnifiedChunker (paragraph-aware text + atomic table chunks)
    → Chunk objects → QdrantRepository (upsert by chunk ID)

Query → OpenAIEmbeddingAdapter → Qdrant vector search
      → content/metadata filters + diversity selection
      → optional cross-encoder reranking
      → deduplication by parent_id
      → RetrievedChunk list → optional LLM synthesis via ReasoningService
```

### Key Components

| Component | File | Role |
|-----------|------|------|
| UnifiedChunker | `src/app/tables/table_chunker.py` | Paragraph-aware text chunks (900 chars, 1-paragraph overlap) + atomic table chunks |
| TableNormalizer | `src/app/tables/table_normalizer.py` | Remove metadata/title rows using row-density and variance heuristics |
| RetrievalService | `src/app/services/retrieval_service.py` | Semantic search with section preference, cross-document diversity, token-aware suppression |
| ReasoningService | `src/app/services/reasoning_service.py` | LLM synthesis; returns `ResearchAnswer` (insight, evidence_basis, citations, confidence) |
| QdrantRepository | `src/app/adapters/vectorstores/qdrant_repository.py` | Idempotent upsert, batch=100 |
| ParserFactory | `src/app/ingestion/parser_factory.py` | Selects Docling (active) or Marker (rollback) |
| DedupUtils | `src/app/ingestion/dedup_utils.py` | Prevents re-ingesting same PDF under different identities |

### Retrieval Filtering Logic (in `retrieval_service.py`)
- **Section priority**: Results/Methods/Discussion/Abstract > Introduction > Conclusion > Metadata
- **Cross-document diversity**: Max 3–4 distinct docs per query
- **Table defaults**: Table chunks ranked higher for data-oriented queries
- **Token-aware suppression**: Prevents false lexical matches (e.g., `uti` inside `utilization`)
- **Query-intent detection**: Conclusion queries suppress Results tails; study-design queries filter metadata-fallback chunks

## Active Collections

| Collection | Status |
|------------|--------|
| `medical_research_chunks_docling_v2_batch1` | Passing stage-1 artifact (20 PDFs); stage-1 and stage-2 spot checks complete; stage 2 may begin |
| `medical_research_chunks_docling_v1` | Baseline small corpus; active manual-testing collection |
| `medical_research_chunks_v1` | Marker-based rollback — do **not** touch |

## ReasoningService Output Types

`ReasoningService.research()` returns a `ResearchAnswer` dataclass:

```python
@dataclass(frozen=True)
class ResearchAnswer:
    insight: str           # parsed Research Insight section
    evidence_basis: str    # parsed Evidence Basis section
    citations: list[RetrievedChunk]   # actual retrieved chunks (authoritative)
    confidence: ConfidenceLevel       # HIGH / MEDIUM / LOW / INSUFFICIENT
```

`ConfidenceLevel` is derived from retrieval signals only (no extra LLM call):
- `INSUFFICIENT`: no citations or "insufficient evidence" in insight
- `HIGH`: ≥4 chunks AND ≥2 distinct doc_ids
- `MEDIUM`: ≥2 chunks
- `LOW`: 1 chunk

## Operational Rules

1. **Always update `ROADMAP.md`** before committing any changes.
2. **Do not add** hybrid retrieval, sparse retrieval, BM25, query expansion, synonym expansion, HyDE, or extra embedding stages unless medium-scale benchmark evidence clearly justifies it.
3. **Do not do proactive retrieval tuning.** Only make retrieval changes for a case that shows a measured candidate-recall weakness via `inspect_retrieval_candidates.py`.
4. **Do not commit**: `.env`, `.pytest_tmp_run/`, `data/eval/results/*`, `data/kb_registry.json`, `data/exports/`, or local rebuild manifests.
5. **Do not reopen** runtime cases `R31` or `R33` unless a regression appears. Do not broaden the `R32` patch without benchmark evidence.

## Rollout Gate (must pass before promoting a stage)

1. Rebuild completes with explicit error reporting
2. Audit passes: no duplicate identities, all metadata valid
3. Benchmark gated metrics within **0.02** of baseline on all of: `expected_doc_hit_rate`, `expected_header_hit_rate`, `top1_*`, `average_*_precision`, `cross_document_average_doc_precision`
4. Runtime queries show no material regression
5. Manual spot-check report generated (see `docs/manual_spot_checks_*.md`)
6. Rollout report artifacts exist

## When a UI Miss Appears

Use `scripts/inspect_retrieval_candidates.py` to separate:
1. Candidate recall failure (not surfaced at all)
2. Ranking/selection failure (present but ranked out)
3. Parser/content failure (chunk content malformed)

Only keep a code change if it is narrow and benchmark-safe on `runtime_queries.json`.

## Benchmark Query Files

| File | Queries | Purpose | Runner |
|------|---------|---------|--------|
| `data/eval/sample_queries.json` | 26 | Stable retrieval baseline (100% precision target) | `scripts/evaluate_retrieval.py` |
| `data/eval/expanded_queries.json` | 43 | Extended retrieval coverage | `scripts/evaluate_retrieval.py` |
| `data/eval/ood_adversarial_queries.json` | 32 | Out-of-distribution / adversarial phrasing | `scripts/evaluate_retrieval.py` |
| `data/eval/runtime_queries.json` | varies | Real user queries (regression tracking) | `scripts/evaluate_retrieval.py` |
| `data/eval/known_gap_queries.json` | varies | Documented expected misses | `scripts/evaluate_retrieval.py` |
| `data/eval/answer_quality_queries.json` | varies | Answer quality: abstention, confidence, doc-ID coverage | `scripts/evaluate_answer_quality.py` |

### Answer Quality Query Format (`answer_quality_queries.json`)

```json
[
  {
    "id": "AQ01",
    "query": "What is the organism detection rate in BAL samples?",
    "expected_abstain": false,
    "expected_confidence_min": "MEDIUM",
    "expected_doc_ids": ["iridica-bal"],
    "notes": "stable baseline — should always retrieve"
  }
]
```

Fields: `id` (string), `query` (required), `expected_abstain` (bool, default false), `expected_confidence_min` (HIGH/MEDIUM/LOW/INSUFFICIENT, optional), `expected_doc_ids` (array, checked against `evidence_basis` text), `notes` (string).
