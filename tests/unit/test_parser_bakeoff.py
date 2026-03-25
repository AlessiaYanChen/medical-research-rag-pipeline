from __future__ import annotations

from experiments.parser_bakeoff import normalize_docling_bakeoff_markdown


def test_normalize_docling_bakeoff_markdown_leaves_non_target_docs_unchanged() -> None:
    markdown = "## RESULTS\n\nDiagnostic accuracy was reported in the main table."

    normalized = normalize_docling_bakeoff_markdown(
        doc_id="some-other-paper",
        markdown_text=markdown,
    )

    assert normalized == markdown


def test_normalize_docling_bakeoff_markdown_strips_smith_chart_and_survey_noise() -> None:
    markdown = """## Step 2: Plot the expected diagnostic difference in unidentified Gram-negative bacteria and accurately identify Gram-positive bacteria as a function of prevalence

To highlight trade-offs between two RDTs, the expected diagnostic differences were plotted.

## A.

## Expected difference: BioFire vs Sepsityper

Difference: BioFire-Sepsityper

100

80

Gram-positive prevalence (%)

FIG 3 The between platform differences in unidentified Gram-negative bacteria and accurately identified Gram-positive bacteria were plotted.

## Preferred diagnostic attributes based on survey

BioFire BCID2 was ranked first by every respondent. The most important diagnostic features were accurate pathogen identification (77.8%), identifying potential resistance (16.7%), and speed/time-to-results (5.6%).

## DISCUSSION

This study compared our in-house method FLAT MS to two FDA-cleared assays.

## BED-FRAME

FIG 5 The figure displays a comparison of weighted accuracy as a function of the relative importance and prevalence of Gram-positive bacteria.

monomicrobial PBCs was equivalent for all three assays.
"""

    normalized = normalize_docling_bakeoff_markdown(
        doc_id="smith-et-al-2023-comparison-of-three-rapid-diagnostic-tests-for-bloodstream-infections-using-benefit-risk-evaluation",
        markdown_text=markdown,
    )

    assert "## A." not in normalized
    assert "Expected difference: BioFire vs Sepsityper" not in normalized
    assert "Difference: BioFire-Sepsityper" not in normalized
    assert "Gram-positive prevalence (%)" not in normalized
    assert "FIG 3" not in normalized
    assert "## Preferred diagnostic attributes based on survey" not in normalized
    assert "ranked first by every respondent" not in normalized
    assert "FIG 5" not in normalized
    assert "## BED-FRAME\n\nFIG 5" not in normalized
    assert "## DISCUSSION" in normalized
    assert "monomicrobial PBCs was equivalent for all three assays." in normalized


def test_normalize_docling_bakeoff_markdown_strips_smith_watermark_lines() -> None:
    markdown = """## Time-to-results

Time-to-results for each RDT were similar.

Downloaded from https://journals.asm.org/journal/jcm on 22 August 2025 by 134.192.6.131.

## Hands-on time

Hands-on time was longer for Sepsityper.
"""

    normalized = normalize_docling_bakeoff_markdown(
        doc_id="smith-et-al-2023-comparison-of-three-rapid-diagnostic-tests-for-bloodstream-infections-using-benefit-risk-evaluation",
        markdown_text=markdown,
    )

    assert "Downloaded from https://journals.asm.org/journal/jcm" not in normalized
    assert "## Time-to-results" in normalized
    assert "## Hands-on time" in normalized
