# Stage-1 Synthesis Validation Plan

Goal: confirm the synthesis pipeline (prompt hardening, `ResearchAnswer`, `ConfidenceLevel`) behaves correctly on the 20-PDF stage-1 corpus before beginning the Stage-2 (50-PDF) rebuild.

Retrieval is already validated and passing. This plan tests the answer-quality layer that was added on top of it.

Work through the steps in order. Each step is labelled **[CODEX]** (implement the code) or **[MANUAL]** (run a command or do a UI check). Do not begin Stage 2 until the gate at the bottom of this file passes.

---

## Step 1 — Create the answer quality query dataset `[CODEX]`

Create `data/eval/answer_quality_queries.json`.

The file must be a JSON array. Use the query text exactly as written below. Leave `expected_doc_ids` as empty arrays — the user will fill those in after running a first retrieval inspection to confirm the correct doc_ids for each paper.

### Category 1: expected-answer queries (from manual test pack)

These queries should always retrieve relevant evidence and produce a substantive insight. Use `expected_abstain: false` and `expected_confidence_min: "MEDIUM"` for single-document factual queries, `"LOW"` for cross-document synthesis queries (retrieval across multiple papers is harder).

```json
[
  {
    "id": "AQ01",
    "query": "What confirmation rate was achieved for Staphylococcus aureus by culture or PCR in the IRIDICA study?",
    "expected_abstain": false,
    "expected_confidence_min": "MEDIUM",
    "expected_doc_ids": [],
    "notes": "single-doc factual — manual test pack item 1. Fill in the IRIDICA doc_id."
  },
  {
    "id": "AQ02",
    "query": "What was the overall detection rate of PCR/ESI-MS compared to routine culture in BAL samples?",
    "expected_abstain": false,
    "expected_confidence_min": "MEDIUM",
    "expected_doc_ids": [],
    "notes": "single-doc factual — manual test pack item 2. Fill in BAL study doc_id."
  },
  {
    "id": "AQ03",
    "query": "What lysozyme concentration and incubation time gave optimal cardiolipin detection in the FLAT assay?",
    "expected_abstain": false,
    "expected_confidence_min": "MEDIUM",
    "expected_doc_ids": [],
    "notes": "single-doc factual — manual test pack item 3. Fill in FLAT study doc_id."
  },
  {
    "id": "AQ04",
    "query": "What is the reported turnaround time of the FLAT lipidomics assay for direct urine pathogen detection?",
    "expected_abstain": false,
    "expected_confidence_min": "MEDIUM",
    "expected_doc_ids": [],
    "notes": "single-doc factual — manual test pack item 4. Fill in FLAT study doc_id."
  },
  {
    "id": "AQ05",
    "query": "Which resistance determinants were NOT detected in the BAL samples?",
    "expected_abstain": false,
    "expected_confidence_min": "MEDIUM",
    "expected_doc_ids": [],
    "notes": "single-doc factual — manual test pack item 5. Fill in BAL study doc_id."
  },
  {
    "id": "AQ06",
    "query": "What did the BAL IRIDICA study find for overall detection versus routine culture?",
    "expected_abstain": false,
    "expected_confidence_min": "MEDIUM",
    "expected_doc_ids": [],
    "notes": "single-doc factual — manual test pack item 14. Duplicate angle on IRIDICA for coverage."
  },
  {
    "id": "AQ07",
    "query": "Which urine paper reports the direct-detection screening workflow, and what metric did it improve most?",
    "expected_abstain": false,
    "expected_confidence_min": "MEDIUM",
    "expected_doc_ids": [],
    "notes": "single-doc factual — manual test pack item 15. Fill in urine/FLAT doc_id."
  },
  {
    "id": "AQ08",
    "query": "Compare the organism ID and AST turnaround improvements reported in RAPID with the faster-decision benefits reported in the Single site RCT.",
    "expected_abstain": false,
    "expected_confidence_min": "LOW",
    "expected_doc_ids": [],
    "notes": "cross-doc synthesis — manual test pack item 6. Fill in both RAPID and Single-site RCT doc_ids."
  },
  {
    "id": "AQ09",
    "query": "How do PCR/ESI-MS methods and FLAT lipidomics workflows differ in their approach to bypassing traditional culture times?",
    "expected_abstain": false,
    "expected_confidence_min": "LOW",
    "expected_doc_ids": [],
    "notes": "cross-doc synthesis — manual test pack item 7. Fill in both study doc_ids."
  },
  {
    "id": "AQ10",
    "query": "Contrast the reasons why single blood cultures are considered inadequate in the Fabre et al. minireview with the diagnostic limitations of urinalysis discussed by Nartey et al.",
    "expected_abstain": false,
    "expected_confidence_min": "LOW",
    "expected_doc_ids": [],
    "notes": "cross-doc synthesis — manual test pack item 8. Fill in Fabre and Nartey doc_ids."
  },
  {
    "id": "AQ11",
    "query": "Which paper should I read for blood-culture turnaround improvements, not stewardship policy?",
    "expected_abstain": false,
    "expected_confidence_min": "LOW",
    "expected_doc_ids": [],
    "notes": "document-navigation query — manual test pack item 9."
  },
  {
    "id": "AQ12",
    "query": "What themes across these papers suggest that rapid diagnostics improve antimicrobial decision-making more reliably than they improve hard clinical outcomes?",
    "expected_abstain": false,
    "expected_confidence_min": "LOW",
    "expected_doc_ids": [],
    "notes": "cross-doc synthesis — manual test pack item 11. Broad synthesis; LOW confidence is acceptable."
  },
  {
    "id": "AQ13",
    "query": "Which of these studies are randomized controlled trials, and which are observational or review papers?",
    "expected_abstain": false,
    "expected_confidence_min": "LOW",
    "expected_doc_ids": [],
    "notes": "study-design classification — manual test pack item 13. Tests prompt hardening for study design identification."
  }
]
```

### Category 2: abstention-expected queries (from known_gap_queries.json)

These queries depend on table-cell-level values that the pipeline is known not to reliably retrieve. They should produce "Insufficient evidence" or very low confidence. Use `expected_abstain: true`.

Append these to the array above:

```json
  {
    "id": "AQ14",
    "query": "What was the detection rate for gram-positive bacteria before and after lysozyme treatment in the FLAT study?",
    "expected_abstain": true,
    "expected_confidence_min": null,
    "expected_doc_ids": [],
    "notes": "known-gap K01 — table-dependent. Synthesis should abstain or produce LOW/INSUFFICIENT."
  },
  {
    "id": "AQ15",
    "query": "What was the LOD for E. faecalis with 100 ug lysozyme at 60 minutes?",
    "expected_abstain": true,
    "expected_confidence_min": null,
    "expected_doc_ids": [],
    "notes": "known-gap K02 — table cell value. Synthesis should abstain."
  },
  {
    "id": "AQ16",
    "query": "How many Enterococcus faecalis samples were correctly identified by FLAT?",
    "expected_abstain": true,
    "expected_confidence_min": null,
    "expected_doc_ids": [],
    "notes": "known-gap K03 — exact species count from table. Synthesis should abstain."
  }
```

### Category 3: hepcidin disambiguation queries

These should retrieve the correct hepcidin paper and attribute findings to it specifically. Use `expected_abstain: false`, `expected_confidence_min: "MEDIUM"`. Fill in `expected_doc_ids` with the specific paper's doc_id.

```json
  {
    "id": "AQ17",
    "query": "What did the 2016 hepcidin study find about hepcidin as a diagnostic tool for iron deficiency anaemia?",
    "expected_abstain": false,
    "expected_confidence_min": "MEDIUM",
    "expected_doc_ids": [],
    "notes": "hepcidin disambiguation — N03. Fill in the 2016 diagnostic tool paper doc_id specifically."
  },
  {
    "id": "AQ18",
    "query": "What role does hepcidin play in chronic kidney disease anaemia according to the CKD-specific paper?",
    "expected_abstain": false,
    "expected_confidence_min": "MEDIUM",
    "expected_doc_ids": [],
    "notes": "hepcidin disambiguation — N01. Fill in the CKD-specific paper doc_id specifically."
  },
  {
    "id": "AQ19",
    "query": "What were the main findings on hepcidin and anaemia of chronic disease in the paper focused on that topic?",
    "expected_abstain": false,
    "expected_confidence_min": "MEDIUM",
    "expected_doc_ids": [],
    "notes": "hepcidin disambiguation — N02. Fill in the anaemia-of-chronic-disease paper doc_id."
  }
```

---

**After creating the file, the user must:**
1. Inspect doc_ids in the active collection using `scripts/audit_collection_state.py` or the Streamlit registry panel.
2. Fill in all empty `expected_doc_ids` arrays with the correct doc_id strings.
3. For abstention-expected queries, confirm `expected_abstain: true` is correct by checking `known_gap_queries.json` notes.

---

## Step 2 — Run the answer quality baseline `[MANUAL]`

After `expected_doc_ids` are filled in, run:

```bash
python scripts/evaluate_answer_quality.py \
  --dataset data/eval/answer_quality_queries.json \
  --collection medical_research_chunks_docling_v2_batch1 \
  --limit 8 \
  --json-out data/eval/results/answer_quality_eval_stage1_baseline.json
```

Save the output. This file is the Stage-1 answer-quality baseline. It must exist before Stage 2 begins.

Review the printed summary. Expected rough targets (not hard gates yet — this is baseline collection):
- `abstain_accuracy` ≥ 0.80
- `has_insight_rate` = 1.0
- `confidence_meets_minimum_rate` ≥ 0.75

Current baseline result on `medical_research_chunks_docling_v2_batch1` (`gpt-5.3-chat`, April 1, 2026):
- `abstain_accuracy = 0.8421`
- `has_insight_rate = 1.0`
- `has_evidence_basis_rate = 1.0`
- `confidence_meets_minimum_rate = 0.6875`
- `average_doc_id_coverage = 0.812`

Ordered follow-up queue after the baseline run:
1. `AQ03` â€” expected-answer query returned `INSUFFICIENT`; inspect retrieval and determine whether the missing parameter values are absent from retrieved chunks or absent from corpus text.
2. `AQ11` â€” document-navigation query returned `INSUFFICIENT` even though `RAPID` appeared in the evidence basis; inspect why the answer abstained despite correct paper selection.
3. `AQ14` â€” known-gap query returned `MEDIUM` instead of abstaining; inspect whether the model over-extrapolated from prose or whether the query should be reclassified.
4. `AQ02`, `AQ06`, `AQ19` â€” correct paper selected but confidence fell below the required minimum; inspect confidence calibration versus evidence quality.
5. `AQ09`, `AQ12`, `AQ13` â€” cross-document evidence-basis coverage is incomplete; inspect whether this is retrieval-stage recall loss or answer condensation loss.

Inspection notes after the first pass:
- `AQ03`: retrieval stays on the correct FLAT paper, but the final returned chunks do not contain the exact optimization sentence. Local parser/export artifacts do contain the exact sentence (`100 µg lysozyme`, `60-minute incubation`), so this is a retrieval/ranking/selection miss rather than missing corpus text.
- `AQ11`: retrieval finishes on `RAPID`, but the final returned chunk is a limitation-heavy discussion sentence instead of a clean study-identification/turnaround chunk. This looks like answer abstention caused by weak final chunk selection, not by wrong-paper retrieval.
- `AQ14`: retrieval surfaces enough FLAT prose to support a partial narrative answer and then mixes in unrelated material, including an irrelevant `BAL SM` chunk. This is consistent with a false-confidence problem on a table-dependent query and should remain treated as a known-gap abstention target.
- `AQ02`: retrieval stays on `BAL SM`, but the top returned chunk is species-specific (`H. influenzae 16/20`) instead of the overall BAL comparison. This looks like a query-to-chunk specificity mismatch that depresses confidence.
- `AQ06`: retrieval again stays on `BAL SM`, but the final returned chunk is a broad discussion summary rather than the abstract/results sentence with the overall `PCR/ESI-MS` versus routine-culture comparison. This is another within-document ranking miss.
- `AQ19`: retrieval drifts inside the hepcidin cluster and finishes on `hepcidin diagnostic tool` rather than `hep anemia`. This is a real disambiguation failure, not just conservative confidence calibration.
- `AQ09`: final retrieval already contains the expected BAL plus FLAT urine papers, so the weak `expected_doc_ids_found` result is mainly an answer/evidence-basis condensation issue rather than a retrieval-stage recall failure.
- `AQ12`: final retrieval returns the two blood-culture trial papers but misses the stewardship review needed for the broader synthesis frame. This is a real retrieval-stage coverage gap.
- `AQ13`: final retrieval covers `Single site RCT`, `RAPID`, and `IJGM`, but misses several expected observational/review papers and admits irrelevant material (for example `hepcidin acute phase`, with `IgaN` also surfacing high in ranked candidates). This is a broad-query ranking/selection problem, not just evidence-basis compression.

Implementation task list before the next baseline run:
1. Retrieval ranking for `AQ03`: ensure the FLAT optimization sentence with `100 Âµg lysozyme` and `60-minute incubation` can survive final chunk selection.
2. Document-navigation selection for `AQ11`: prefer study-identification and turnaround-gain chunks over limitation-heavy discussion chunks when the user asks which paper to read.
3. Known-gap abstention control for `AQ14`: reduce false-confidence answers on table-dependent FLAT queries, especially when only partial prose support is retrieved.
4. Within-document selection for `AQ02`: prefer overall BAL detection-vs-culture evidence over species-specific result chunks for broad "overall detection rate" questions.
5. Within-document selection for `AQ06`: promote BAL abstract/results summary chunks over generic discussion chunks for overall-comparison wording.
6. Hepcidin disambiguation for `AQ19`: strengthen targeting of `hep anemia` over sibling hepcidin papers for anaemia-of-chronic-disease phrasing.
7. Evidence-basis/doc-id mention handling for `AQ09`: preserve all expected paper identities in the evidence basis when retrieval already includes them.
8. Cross-document coverage for `AQ12`: retrieve and retain the Fabre stewardship review alongside `RAPID` and `Single site RCT`.
9. Broad classification-query ranking for `AQ13`: reduce irrelevant hepcidin/glycoform paper intrusion and improve retrieval of observational/review exemplars.
10. Re-run `data/eval/answer_quality_queries.json` after the fixes and compare against `data/eval/results/answer_quality_eval_stage1_baseline.json`.
11. Re-run known-gap abstention evaluation after the fixes to confirm no false-confidence regressions are introduced on table/figure-dependent queries.

Progress update after the first implementation pass:
- `AQ03` complete: retrieval ranking now keeps the FLAT optimization sentence with the exact `100 ug lysozyme` and `60-minute incubation` evidence in the final returned chunks.
- `AQ11` complete: document-navigation selection now prefers the turnaround-gain `RAPID` chunk over limitation-heavy discussion text.
- `AQ14` complete: reasoning now forces abstention for exact before/after intervention-metric questions when the retrieved context lacks quantitative evidence tied to the named intervention.
- `AQ02` and `AQ06` complete: BAL overall-comparison queries now return the discussion span with the overall PCR/ESI-MS-versus-routine-culture finding instead of species-specific result chunks.
- `AQ19` complete: hepcidin anaemia-of-chronic-disease phrasing now locks to `hep anemia` instead of drifting to the CKD/diagnostic sibling papers.
- `AQ09` complete: answer-quality doc-id coverage now normalizes Unicode dash variants and markdown formatting before matching evidence-basis doc IDs.
- `AQ12` complete: query expansion and ranking now pull the Fabre stewardship review into the returned set alongside `RAPID` and `Single site RCT`.
- `AQ13` complete: classification queries now suppress off-domain hepcidin/glycoform noise and return domain review/observational exemplars such as Fabre, IJGM, BAL, and the urine papers.
- Remaining work: re-run `data/eval/answer_quality_queries.json` and the known-gap abstention evaluation to confirm the fixes moved the baseline in the expected direction.

Post-fix validation results:
- `data/eval/results/answer_quality_eval_stage1_after_fixes.json`: `abstain_accuracy = 1.0`, `has_insight_rate = 1.0`, `has_evidence_basis_rate = 1.0`, `confidence_meets_minimum_rate = 0.8125`, `average_doc_id_coverage = 0.9774`, `average_citation_count = 3.4737`.
- `data/eval/results/answer_quality_eval_known_gaps_stage1_after_fixes.json`: `abstain_accuracy = 0.4167`.
- Main answer-quality baseline is now above the original soft targets, but the known-gap abstention track still fails badly and is the current gate blocker.

Known-gap blocker queue after the post-fix rerun:
1. `K01` to `K04`: FLAT table-driven gram-positive and species-count questions are still being answered instead of abstained.
2. `K06`: figure-derived cardiolipin m/z identification is still returning `HIGH` confidence and remains a hard-gate failure.
3. `K07` and `K08`: BAL figure/correlation questions are still being answered from partial prose support instead of abstaining.
4. `K12`: Fabre table-context list query is still being answered instead of abstaining.
5. Re-run `data/eval/known_gap_queries.json` after each abstention-control pass until no known-gap query returns `HIGH` confidence.

Known-gap abstention pass 2:
- `data/eval/results/answer_quality_eval_known_gaps_stage1_after_abstention_pass2.json` now returns `INSUFFICIENT` for the main figure/table/list families (`K06`, `K07`, `K08`, `K12`) before the LLM can synthesize a confident answer.
- No known-gap query now returns `confidence: HIGH`, so the hard false-confidence gate is cleared.
- The reported `abstain_accuracy` in this file remains misleading because `known_gap_queries.json` does not carry `expected_abstain: true` flags for the evaluation script. Treat the `HIGH`-confidence count, not that summary ratio, as the operational gate for this track.

Current status before Step 5:
- The Stage-1 answer-quality baseline is materially improved and the known-gap false-confidence gate is clear.
- The next Codex task is Step 5 in this file: build `scripts/run_synthesis_gate.py`, add `tests/unit/test_synthesis_gate.py`, and update `ROADMAP.md` once the gate script exists.

If any expected-answer query returns `INSUFFICIENT` confidence, inspect with:

```bash
python scripts/inspect_retrieval_candidates.py \
  --query "<query text>" \
  --collection medical_research_chunks_docling_v2_batch1 \
  --initial-limit 20
```

---

## Step 3 — Run known-gap abstention check `[MANUAL]`

Run the full known-gap track through synthesis to verify the pipeline correctly abstains:

```bash
python scripts/evaluate_answer_quality.py \
  --dataset data/eval/known_gap_queries.json \
  --collection medical_research_chunks_docling_v2_batch1 \
  --limit 8 \
  --json-out data/eval/results/answer_quality_eval_known_gaps_stage1.json
```

**Hard gate:** no known-gap query may return `confidence: HIGH`. Any `HIGH` confidence result on a known-gap query is a false-confidence bug that must be investigated before Stage 2.

---

## Step 4 — Cross-document synthesis UI spot checks `[MANUAL]`

Run the following 8 queries through the Streamlit UI (`streamlit run scripts/ui_app.py`), using collection `medical_research_chunks_docling_v2_batch1`. For each, record: query, confidence level, insight summary (1–2 sentences), whether both expected papers appear in Evidence Basis.

| ID | Query (from manual test pack) | Pass criterion |
|----|-------------------------------|----------------|
| AQ08 | RAPID vs. Single-site RCT comparison | Both doc_ids in Evidence Basis; insight compares rather than conflates |
| AQ09 | PCR/ESI-MS vs. FLAT culture bypass | Both papers cited; no attribution confusion |
| AQ10 | Fabre et al. vs. Nartey et al. limitations | Correct limitations assigned to each paper |
| AQ12 | Cross-paper themes on decision-making | Confidence ≥ MEDIUM; no invented evidence |
| AQ13 | RCT vs. observational classification | Study designs correctly labelled per prompt hardening |
| AQ17 | Hepcidin 2016 diagnostic tool | Evidence Basis names the 2016 paper specifically |
| AQ18 | Hepcidin in CKD | Evidence Basis names the CKD paper specifically |
| AQ19 | Hepcidin and anaemia of chronic disease | Evidence Basis names the correct paper specifically |

Record results in `docs/manual_spot_checks_synthesis_2026-04-01.md` (append to the existing file or create a new dated file).

**Soft gate:** at least 6 of 8 pass. Any hepcidin failure (AQ17–AQ19) that attributes findings to the wrong paper must be recorded as a known issue before Stage 2.

---

## Step 5 — Build the synthesis gate script `[CODEX]`

Create `scripts/run_synthesis_gate.py`. This script compares a current answer quality evaluation run against the Stage-1 baseline and outputs a pass/fail report. It mirrors the structure of `scripts/build_rollout_report.py`.

### CLI interface

```
python scripts/run_synthesis_gate.py \
  --baseline-file data/eval/results/answer_quality_eval_stage1_baseline.json \
  --current-file data/eval/results/answer_quality_eval_<stage>.json \
  --known-gaps-file data/eval/results/answer_quality_eval_known_gaps_<stage>.json \
  --output data/eval/results/synthesis_gate_report.json
```

All three input files are required. `--output` defaults to `data/eval/results/synthesis_gate_report.json`.

### Gate thresholds (hard-coded constants at module level)

```python
MAX_ABSTAIN_ACCURACY_DROP = 0.10       # allowed drop vs. baseline
MIN_ABSTAIN_ACCURACY = 0.80            # absolute floor
MIN_HAS_INSIGHT_RATE = 1.0             # no empty insights on non-abstention queries
MIN_CONFIDENCE_MEETS_MINIMUM_RATE = 0.75
MAX_FALSE_CONFIDENCE_ON_KNOWN_GAPS = 0  # zero HIGH-confidence known-gap results allowed
```

### Output format

```json
{
  "baseline_file": "...",
  "current_file": "...",
  "known_gaps_file": "...",
  "gate_passed": true,
  "failures": [],
  "checks": {
    "abstain_accuracy": {
      "baseline": 0.85,
      "current": 0.87,
      "delta": 0.02,
      "passed": true
    },
    "has_insight_rate": {
      "current": 1.0,
      "passed": true
    },
    "confidence_meets_minimum_rate": {
      "baseline": 0.80,
      "current": 0.78,
      "delta": -0.02,
      "passed": true
    },
    "false_confidence_on_known_gaps": {
      "count": 0,
      "query_ids": [],
      "passed": true
    }
  }
}
```

`gate_passed` is `true` only when all individual checks pass. `failures` is a list of check names that failed.

### False confidence on known gaps

From the `--known-gaps-file`, count queries where `confidence == "HIGH"`. Any such query is a false-confidence hit. The gate fails if count > 0.

### Tests — `tests/unit/test_synthesis_gate.py`

Write unit tests covering:
- Gate passes when all metrics are at or above thresholds
- Gate fails when `abstain_accuracy` drops more than `MAX_ABSTAIN_ACCURACY_DROP`
- Gate fails when `abstain_accuracy` falls below `MIN_ABSTAIN_ACCURACY` even within allowed delta
- Gate fails when `has_insight_rate` < 1.0
- Gate fails when `confidence_meets_minimum_rate` < threshold
- Gate fails when any known-gap query has `confidence == "HIGH"`
- `gate_passed: false` and `failures` list both reflect correctly on multi-failure input
- Script exits with code 1 when gate fails, 0 when it passes
- Missing input file produces a clear error message and exits with code 1

### Constraints

- No new dependencies.
- Follow the same `parse_args` / `main() -> int` / `raise SystemExit(main())` pattern as `scripts/build_rollout_report.py`.
- Run `pytest tests/unit/` before and after; existing passing tests must stay green.
- Update `ROADMAP.md` to note that the synthesis gate script exists once the step is complete.

---

## Stage-2 readiness gate

Do not begin the 50-PDF rebuild until all of the following are true:

| # | Gate | Evidence |
|---|------|----------|
| 1 | Answer quality baseline recorded | `data/eval/results/answer_quality_eval_stage1_baseline.json` exists |
| 2 | `abstain_accuracy` ≥ 0.80 on baseline run | Step 2 output |
| 3 | `has_insight_rate` = 1.0 on baseline run | Step 2 output |
| 4 | No known-gap query returns `HIGH` confidence | Step 3 output |
| 5 | Cross-doc synthesis spot checks ≥ 6/8 pass | Step 4 report |
| 6 | No hepcidin paper mis-attribution in AQ17–AQ19 | Step 4 report |
| 7 | `run_synthesis_gate.py` exists and passes on stage-1 collection | Step 5 |
| 8 | Retrieval eval still passing on stable + runtime tracks | Re-run `scripts/evaluate_retrieval.py` |
