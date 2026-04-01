# Prioritized Retrieval Fix List

Date: 2026-04-01
Collection: `medical_research_chunks_docling_v2_batch1`

## Priority 1

- Reduce sibling-paper drift inside the hepcidin cluster.
  The strongest failures came from queries for `hep anemia`, `hep in diagnosis of iron disorders 2016`, `hepcidin ckd`, and `RCM publication`, where retrieval favored adjacent hepcidin papers with broadly similar terminology.

- Strengthen exact or near-exact `doc_focus` matching signals.
  Queries clearly aimed at one paper were often outranked by semantically related neighbors. Retrieval should reward rare title terms, canonical doc-id tokens, and distinctive phrase overlap more heavily.

- Audit re-ranking behavior on near-duplicate topical neighborhoods.
  Re-ranking often swapped the intended paper with a related sibling that shared domain vocabulary but not the exact answer target.

## Query Quality Caveat

- `N04` (`jmsacl`) and `N05` (`RCM publication`) were not strong query formulations.
- `N04` was too broad for a method-dense neighborhood.
- `N05` was highly generic and likely under-specified the intended paper.
- These should be rewritten before being used as high-confidence regression gates for retrieval tuning.

## Priority 2

- Improve handling of method-focused prompts.
  Queries asking for the main method, analytical platform, or principal finding underperformed when the corpus contained multiple assay or workflow papers in the same topic family.

- Add guardrails for "intended paper first" when answer-bearing evidence exists in multiple papers.
  Mixed results in the lipidomics pair showed that useful evidence can be found in both documents, but the wrong one often dominates.

- Review chunk selection strategy for conclusion versus methods versus table chunks.
  Some runs surfaced a relevant paper but not the most answer-bearing chunk. Better section preference could convert mixed results into passes.

## Priority 3

- Add targeted regression cases from the worst stage-2 failures.
  At minimum:
  `N02` anemia of chronic disease
  `N03` hepcidin diagnostic value for IDA vs ACD
  `N04` JMSACL method and clinical target
  `N05` RCM publication main method or finding
  For `N04` and `N05`, revise the query wording before promoting them to stable regression cases.

- Add paired-paper regression cases for the lipidomics documents.
  The `nartey-et-al-2024...` and `Culture-Free Lipidomics-Based Screening Test` pair should be tested together because they frequently displace each other.

- Preserve stage-1 positives as non-regression checks.
  Especially:
  `M04` hep fer ratio liver
  `M05` hepcidin acute phase
  `M06` hepcidin diagnostic tool
  `M07` IgaN

## Suggested Tuning Sequence

1. Fix hepcidin-cluster document discrimination first.
2. Re-test `N01`-`N05`.
3. Tune sibling-document handling for the two lipidomics papers.
4. Re-test `N06`-`N07`.
5. Confirm `N08`-`N09` remain stable.
