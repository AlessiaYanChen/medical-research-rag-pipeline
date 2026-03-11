from __future__ import annotations

from io import StringIO
import logging

import pandas as pd

from src.app.tables.table_normalizer import TableNormalizer


def test_sanitize_table_trims_metadata_and_logs_warning(caplog) -> None:
    # RAPID.table_03.csv-like shape: metadata/title rows above header.
    csv_data = """RAPID Trial - Table 03,,,,
Lipidomics Screening Panel,,,,
Units: mg/dL (except p-value),,,,
Internal analysis set,,,,
Do not cite externally,,,,
Preprint appendix,,,,
Biomarker,Case,Control,P-value,Source ID
LDL-C,126,101,0.02,SRC-001
HDL-C,52,61,0.04,SRC-001
"""
    df = pd.read_csv(StringIO(csv_data), header=None)
    normalizer = TableNormalizer()

    with caplog.at_level(logging.WARNING):
        cleaned = normalizer.sanitize_table(df=df, file_name="culture_free.csv")

    assert cleaned.iloc[0].tolist() == ["Biomarker", "Case", "Control", "P-value", "Source ID"]
    assert cleaned.iloc[1].tolist() == ["LDL-C", "126", "101", "0.02", "SRC-001"]
    assert len(cleaned) == 3
    assert "High Metadata Density" in caplog.text
    assert "culture_free.csv" in caplog.text

    artifact = normalizer.get_last_metadata_artifact()
    assert artifact is not None
    assert artifact["trimmed_row_count"] == 6
    assert "Units: mg/dL (except p-value)" in artifact["rows"][2][0]


def test_sanitize_table_returns_full_df_when_no_header_found(caplog) -> None:
    csv_data = """Title Row,,,
Subtitle,,,
Appendix,,,
"""
    df = pd.read_csv(StringIO(csv_data), header=None)
    normalizer = TableNormalizer()

    with caplog.at_level(logging.WARNING):
        cleaned = normalizer.sanitize_table(df=df, file_name="no_header.csv")

    pd.testing.assert_frame_equal(cleaned, df)
    assert "Header not confidently identified" in caplog.text

    artifact = normalizer.get_last_metadata_artifact()
    assert artifact is not None
    assert artifact["trimmed_row_count"] == 0
