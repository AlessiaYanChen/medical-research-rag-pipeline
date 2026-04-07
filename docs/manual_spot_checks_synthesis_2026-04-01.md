# Stage-1 Synthesis UI Spot Checks

Date recorded: `2026-04-07`

Collection: `medical_research_chunks_docling_v2_batch1`

Source artifact: `data/eval/results/answer_quality_eval_stage1_after_fixes.json`

Note: the environment did not have a live LLM key on `2026-04-07`, so this report was completed from the cached stage-1 synthesis outputs already generated from the same retrieval and reasoning stack used by the UI.

Live rerun note:
- A fresh Azure OpenAI rerun was completed on `2026-04-07` and written to `data/eval/results/manual_synthesis_spot_checks_stage1_2026-04-07_live.json`.
- The live rerun matched the recorded gate outcome: `6/8` pass, no hepcidin mis-attribution, with `AQ10` and `AQ13` remaining the watch items.

## Summary

- Result: `6/8` pass
- Soft gate: pass
- Hepcidin mis-attribution gate (`AQ17`-`AQ19`): pass
- Failures to keep on watch: `AQ10`, `AQ13`

## Results

| ID | Confidence | Pass | Insight summary | Evidence basis check |
|----|------------|------|-----------------|----------------------|
| AQ08 | `HIGH` | Pass | The answer cleanly compares RAPID's faster organism ID and AST turnaround with the single-site RCT's faster escalation, de-escalation, and reduced unnecessary vancomycin exposure. It compares the two papers rather than conflating them. | `RAPID` and `Single site RCT` both named in Evidence Basis. |
| AQ09 | `HIGH` | Pass | The answer distinguishes PCR/ESI-MS as direct nucleic-acid detection from FLAT as direct lipid-biomarker detection, and it contrasts their turnaround framing without mixing attribution. It correctly keeps BAL PCR/ESI-MS separate from the two FLAT urine papers. | Evidence Basis names `BAL SM`, `Culture-Free Lipidomics-Based Screening Test`, and `nartey-et-al-2024-a-lipidomics-based-method-to-eliminate-negative-urine-culture-in-general-population`. |
| AQ10 | `MEDIUM` | Fail | The answer correctly gives the Nartey urinalysis limitation, but it says the retrieved Fabre context is insufficient to state why single blood cultures are inadequate. That does not satisfy the spot-check requirement to assign the relevant limitation to each paper. | Both expected papers are named, but the Fabre-side limitation is not actually resolved. |
| AQ12 | `HIGH` | Pass | The answer synthesizes the expected theme: rapid diagnostics consistently improve identification and stewardship timing more clearly than mortality or length of stay. It grounds that claim in RAPID, the single-site RCT, and Fabre's stewardship framing without inventing evidence. | Evidence Basis names `RAPID`, `Single site RCT`, and `fabre-et-al-blood-culture-utilization-in-the-hospital-setting-a-call-for-diagnostic-stewardship`. |
| AQ13 | `HIGH` | Fail | The answer correctly labels `Single site RCT`, `BAL SM`, Fabre, and the IJGM review, but it does not correctly hard-label `RAPID` as randomized and it omits several expected papers from the classification set. That misses the prompt-hardening goal for study-design classification. | Evidence Basis includes `RAPID`, but the classification is not fully correct. |
| AQ17 | `HIGH` | Pass | The answer centers on the 2016 iron-disorders paper and describes hepcidin as a promising diagnostic biomarker for distinguishing iron deficiency anaemia from anaemia of chronic disease, with assay-harmonization caveats. The core attribution is correct. | Evidence Basis explicitly names `hep in diagnosis of iron disorders 2016`. |
| AQ18 | `HIGH` | Pass | The answer correctly anchors the CKD question in the CKD-specific paper and summarizes the ferritin correlation, lack of significant association with kidney-function markers in this cohort, and the iron-regulation interpretation. The core attribution is correct even though one supporting sentence also cites broader hepcidin literature. | Evidence Basis explicitly names `hepcidin ckd`. |
| AQ19 | `LOW` | Pass | The answer stays narrow and low-confidence, but it correctly anchors the response in the anaemia-focused hepcidin paper and does not misattribute the findings to the CKD or 2016 iron-disorders papers. That satisfies the hepcidin disambiguation requirement. | Evidence Basis explicitly names `hep anemia`. |

## Outcome

- Stage-1 synthesis Step 4 passes at `6/8`.
- No hepcidin paper mis-attribution was observed in `AQ17`-`AQ19`.
- `AQ10` and `AQ13` remain quality watch items, but they do not block the documented Step 4 gate.
