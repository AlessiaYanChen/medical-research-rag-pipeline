# Manual Spot Checks 2026-04-01

This note records the completed manual UI spot-check batches and the immediate retrieval tuning change that followed.

## Coverage

- Stage 1 queue `M01`-`M08`: complete
- Stage 2 queue `N01`-`N09`: complete

Stage 1 queue-backed totals:

- Pass: 5
- Mixed: 3
- Fail: 0

Stage 2 queue-backed totals:

- Pass: 2
- Mixed: 3
- Fail: 4

Combined queue-backed totals:

- Pass: 7
- Mixed: 6
- Fail: 4

## Interpretation

- Stage 1 was broadly stable.
- Stage 2 exposed a stronger retrieval blind-spot cluster around hepcidin and assay-method papers with overlapping vocabulary.
- The clearest retrieval failures were `N02`, `N03`, `N04`, and `N05`.
- `N04` (`jmsacl`) and `N05` (`RCM publication`) should be interpreted cautiously because the queries were weak and under-specified.

## Retrieval Change Applied

The retrieval service was updated after the manual checks to improve singular-target query handling:

- queries phrased as `this paper` or `this study` are now treated as single-document-target queries
- singular-target queries now receive a stronger document-title and doc-id overlap boost
- `hepcidin` alone is no longer treated as a strong title-disambiguation token inside the hepcidin cluster

This change is intended to reduce sibling-paper drift for stage-2 style queries without disturbing existing contrastive query behavior.

## Validation

Focused retrieval regressions were added for the hepcidin single-document failure pattern, and the full retrieval-service unit file passed locally after the change.
