# Manual Spot Checks Summary

Date: 2026-04-01
Collection: `medical_research_chunks_docling_v2_batch1`

## Scope

This summary covers queue-backed manual UI spot checks from:

- `manual_spot_check_queue_stage1.json` (`M01`-`M08`)
- `manual_spot_check_queue_stage2.json` (`N01`-`N09`)

Stage 1 also contains 4 legacy entries in `manual_spot_checks_stage1.json` that are not part of the queue-backed counts below.

## Results

### Stage 1

- Pass: 5
- Mixed: 3
- Fail: 0

Statuses:

- `M01` mixed
- `M02` mixed
- `M03` pass
- `M04` pass
- `M05` pass
- `M06` pass
- `M07` pass
- `M08` mixed

### Stage 2

- Pass: 2
- Mixed: 3
- Fail: 4

Statuses:

- `N01` mixed
- `N02` fail
- `N03` fail
- `N04` fail
- `N05` fail
- `N06` mixed
- `N07` mixed
- `N08` pass
- `N09` pass

### Combined Queue-Backed Totals

- Pass: 7
- Mixed: 6
- Fail: 4

## Main Patterns

- Stage 1 was generally stable, with correct paper targeting usually preserved.
- Stage 2 exposed a harder cluster of retrieval blind spots, especially among hepcidin and method-focused papers with overlapping vocabulary.
- Several mixed results had answer-bearing evidence present, but the wrong sibling paper ranked first.
- `Enable re-ranking` sometimes improved evidence surfacing, but in multiple cases it demoted the intended paper behind a related document.

## Query Quality Note

- `N04` (`jmsacl`) and `N05` (`RCM publication`) should be treated with extra caution because the query formulations were weak and underspecified relative to the target papers.
- `N04` asked broadly for the analytical platform or method and its clinical target, which overlaps strongly with several other method papers in the collection.
- `N05` asked an especially generic question, "What is the main finding or method presented in this study?", which does not provide enough distinctive retrieval signal for the intended paper.
- These two results are still recorded as failures, but they are not as clean an indicator of retrieval weakness as the more specific stage-2 failures.

## Highest-Risk Retrieval Blind Spots

- `N02` `hep anemia`
- `N03` `hep in diagnosis of iron disorders 2016`
- `N04` `jmsacl` (interpret with caution due to weak query wording)
- `N05` `RCM publication` (interpret with caution due to weak query wording)

These were the clearest failures where the intended paper did not dominate retrieval and the answer path was not surfaced cleanly.
