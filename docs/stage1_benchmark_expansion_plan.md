# Stage-1 Benchmark Expansion Plan

Date: `2026-04-09`

Scope: keep the active corpus fixed at `medical_research_chunks_docling_v2_batch1` (`20` PDFs) and improve benchmark signal before any broader rollout work.

## Why this exists

The current stage-1 retrieval benchmark is too easy to keep serving as the main regression signal:

- `data/eval/stage1_coverage_queries.json` currently has `32` queries.
- Only `14` of the `20` stage-1 papers have direct single-document coverage in that file.
- Those `14` papers each have only `2` direct coverage queries.
- The remaining `6` papers currently appear only through older stable/expanded/runtime tracks or cross-document synthesis checks, not through dedicated stage-1 retrieval coverage.

The immediate goal is not to change retrieval architecture. The goal is to make regressions visible on the current `20`-PDF corpus.

## Recommended split

Keep retrieval and abstention evaluation separate:

- Retrieval benchmark expansion:
  - extend `data/eval/stage1_coverage_queries.json`
  - target roughly `88` retrieval queries
  - structure:
    - `20 x 4 = 80` direct per-paper queries
    - `8` cross-document or disambiguation queries
- Abstention / false-positive benchmark expansion:
  - extend `data/eval/answer_quality_queries.json` or add a separate answer-quality abstention dataset later
  - target roughly `10-12` plausible-but-absent queries
  - do not force absent-answer traps into `evaluate_retrieval.py`, because the current retrieval metrics do not score them meaningfully

## Target query families per paper

Each paper should ideally have `4` direct retrieval queries:

1. Factual lookup
2. Results or discussion synthesis
3. Table or metric query when the paper has table chunks; otherwise a metrics-in-prose query
4. Methods, limitations, title-navigation, or study-design query

Use `include_tables: true` only where the current corpus actually has table chunks and the query is metric-heavy.

## Stage-1 corpus coverage matrix

Legend:

- Current direct queries: current count in `data/eval/stage1_coverage_queries.json`
- Target direct queries: per-paper target for the expanded retrieval set
- Add: how many new direct queries are needed
- Table note: whether a true table-backed query is sensible from the current manifest

| Doc ID | Current direct queries | Target direct queries | Add | Table note | Recommended new slots |
|---|---:|---:|---:|---|---|
| `1-s2.0-S0009912024000250-main` | 2 | 4 | 2 | No table chunks in current manifest | add one results/clinical-implication query and one methods/assay-comparison query |
| `Aune-2020-Optimizing hepcidin measurement with` | 2 | 4 | 2 | Table-backed metric query is valid | add one methods/proficiency-testing workflow query and one limitations/reference-standard query |
| `BAL SM` | 0 | 4 | 4 | Table-backed metric query is valid | add factual detection query, results-vs-culture query, resistance/organism coverage metric query, and limitations/clinical-utility query |
| `Chen_Michael_IntJMolSci_2021` | 2 | 4 | 2 | Limited tables; prose-heavy method/results better | add one limitations/cohort-design query and one biomarker-interpretation query |
| `Culture-Free Lipidomics-Based Screening Test` | 0 | 4 | 4 | Table-backed metric query is valid | add optimization query, direct-detection workflow query, lipid biomarker results query, and limitations/special-population query |
| `fabre-et-al-blood-culture-utilization-in-the-hospital-setting-a-call-for-diagnostic-stewardship` | 0 | 4 | 4 | Table-backed query is possible but stewardship prose may be stronger | add guideline/stewardship argument query, inappropriate-culture-rate query, repeat-culture limitation query, and review-scope query |
| `hep anemia` | 2 | 4 | 2 | Mostly prose/review | add one ACD-focused query and one diagnostic/therapeutic implications query |
| `hep fer ratio liver` | 2 | 4 | 2 | Table-backed metric query is valid | add one cohort/biomarker framing query and one fibrosis/cirrhosis threshold or trend query |
| `hep in diagnosis of iron disorders 2016` | 2 | 4 | 2 | Mostly prose/review | add one IDA-vs-ACD disambiguation query and one assay-harmonization / decision-limit query |
| `hepcidin acute phase` | 2 | 4 | 2 | Light table support; prose results likely enough | add one pre/post infection response query and one viral-vs-bacterial interpretation query |
| `hepcidin ckd` | 2 | 4 | 2 | Results metrics exist; prose may still be sufficient | add one ferritin/regression query and one ESA-resistance or iron-homeostasis interpretation query |
| `hepcidin diagnostic tool` | 2 | 4 | 2 | No table chunks in current manifest | add one CKD mechanism/clearance query and one treatment/therapeutic-target query |
| `IgaN` | 2 | 4 | 2 | Large paper; likely benefits from method + result expansion | add one analytical workflow query and one patient-vs-control glycoform interpretation query |
| `IJGM-393329-blood-culture-negative-endocarditis--a-review-of-laboratory-` | 0 | 4 | 4 | Table-backed query is possible but review framing matters more | add one overview query, one diagnostic-methods comparison query, one review-scope or microbiology-methods query, and one limitations/query-disambiguation prompt |
| `jmsacl` | 2 | 4 | 2 | Table-backed metric query is valid | add one hepcidin:ferritin-index query and one clinical-utility / when-testing-is-useful query |
| `JOGC fibronectin` | 2 | 4 | 2 | Table-backed diagnostic-accuracy query is valid | add one concurrent-testing comparison query and one rule-out / workflow implication query |
| `nartey-et-al-2024-a-lipidomics-based-method-to-eliminate-negative-urine-culture-in-general-population` | 2 | 4 | 2 | Table-backed metric query is valid | add one urinalysis-limitation query and one workflow / negative-culture-elimination query |
| `RAPID` | 0 | 4 | 4 | Table-backed metrics valid but results prose is enough | add one organism-ID turnaround query, one AST turnaround query, one de-escalation or clinical-outcomes query, and one study-design/title-navigation query |
| `RCM publication` | 2 | 4 | 2 | Table-backed validation metrics are valid | add one sample-preparation/cost query and one reproducibility/recovery query |
| `Single site RCT` | 0 | 4 | 4 | Results metrics are strong; prose may suffice | add one clinical-outcomes query, one escalation/de-escalation query, one vancomycin-use query, and one study-design/title-navigation query |

## Cross-document retrieval slice

Keep `8` cross-document retrieval queries in the expanded stage-1 coverage set.

Recommended themes:

1. `RAPID` vs `Single site RCT` turnaround and stewardship comparison
2. `BAL SM` vs `Culture-Free Lipidomics-Based Screening Test` direct-detection modality comparison
3. `fabre...` vs `IJGM...` review-style diagnostic stewardship vs diagnostic-workup comparison
4. `hep in diagnosis of iron disorders 2016` vs `hepcidin ckd` disambiguation
5. `hep anemia` vs `hepcidin diagnostic tool` disambiguation
6. `Aune-2020-Optimizing hepcidin measurement with` vs `RCM publication` assay-standardization vs assay-implementation disambiguation
7. `nartey...` vs `Culture-Free Lipidomics-Based Screening Test` urine workflow vs broader FLAT method comparison
8. `jmsacl` vs `hepcidin ckd` interpretation-vs-clinical-cohort disambiguation

## Absent-answer slice

Do this in the answer-quality layer, not the retrieval layer.

Recommended absent-answer families:

1. Exact table-cell asks that are not present in prose
2. Figure-dependent interpretation asks
3. Specific subgroup or species-level counts absent from the current text
4. Plausible-but-not-indexed intervention comparisons
5. Cross-paper synthesis asks that require a paper not in the `20`-PDF corpus

Target `10-12` absent-answer queries total.

## Concrete next move

1. Expand `data/eval/stage1_coverage_queries.json` from `32` toward `~88` queries using the matrix above.
2. Start by filling the `6` papers that currently have zero direct stage-1 coverage:
   - `BAL SM`
   - `Culture-Free Lipidomics-Based Screening Test`
   - `fabre-et-al-blood-culture-utilization-in-the-hospital-setting-a-call-for-diagnostic-stewardship`
   - `IJGM-393329-blood-culture-negative-endocarditis--a-review-of-laboratory-`
   - `RAPID`
   - `Single site RCT`
3. Then bring the other `14` papers from `2` direct queries each up to `4`.
4. Only after that, add the absent-answer slice in the answer-quality benchmark.

Status after implementation on `2026-04-10`:

- `data/eval/stage1_coverage_queries.json` now contains `88` retrieval queries.
- All `20` stage-1 papers now have `4` direct retrieval queries each.
- The expanded benchmark now produces useful non-ceiling signal on `medical_research_chunks_docling_v2_batch1`:
  - `expected_doc_hit_rate = 1.0`
  - `top1_expected_doc_hit_rate = 1.0`
  - `average_doc_precision = 1.0`
  - `cross_document_average_doc_precision = 1.0`
  - `expected_header_hit_rate = 1.0`
  - `top1_expected_header_hit_rate = 0.8182`
  - `average_header_precision = 0.793`
- Interpretation: on the current `20`-PDF stage-1 corpus, document recall, cross-document precision, and at-least-one-hit header coverage are now clean on the expanded `88`-query retrieval benchmark.
- Remaining retrieval debt is narrower:
  - top-1 header selection still has room to improve
  - the benchmark is now better suited for future absent-answer and ingestion-hardening work than for large new retrieval architecture changes
- Recommended next step from this benchmark state:
  - expand the answer-quality layer with plausible-but-absent queries and hard negatives before doing more retrieval tuning
  - if top-1 header ordering is revisited later, inspect failing cases first rather than adding new heuristics proactively
- Stage-1 answer-quality follow-up on `2026-04-10`:
  - `data/eval/answer_quality_queries.json` has been expanded beyond the original factual + known-gap + hepcidin set to include plausible-but-absent queries
  - the new abstention slice targets:
    - out-of-corpus comparator papers
    - exact subgroup or threshold asks not stably present in prose
    - economic-analysis or mortality-benefit claims not supported by the current `20`-PDF corpus
    - figure-derived exact counts that should remain abstention targets
  - current expanded answer-quality run (`data/eval/results/answer_quality_eval_stage1_expanded_absent_answer_v3.json`) now reports:
    - `queries_total = 31`
    - `abstain_accuracy = 1.0`
    - `confidence_meets_minimum_rate = 0.8125`
    - `average_doc_id_coverage = 0.9821`
  - narrow reasoning guards were sufficient for the newly exposed false-positive families:
    - named out-of-corpus comparator queries like `AQ20`
    - exact subgroup-summary queries like `AQ22`

## Success criteria

- Every stage-1 paper has at least `4` direct retrieval queries.
- The retrieval benchmark reaches roughly `80-90` queries and no longer sits at an obvious ceiling because of undercoverage.
- Absent-answer and false-positive behavior is tracked separately in the answer-quality layer.
- No retrieval-architecture change is made unless the larger benchmark shows a real weakness.
