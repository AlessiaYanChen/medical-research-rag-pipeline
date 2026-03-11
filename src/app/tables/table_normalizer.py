from __future__ import annotations

import logging
import math

import pandas as pd


logger = logging.getLogger(__name__)


class TableNormalizer:
    """Normalize parsed tables by trimming metadata/title rows above the header."""

    def __init__(self) -> None:
        self.last_metadata_artifact: dict[str, object] | None = None

    def sanitize_table(self, df: pd.DataFrame, file_name: str) -> pd.DataFrame:
        if df.empty:
            self.last_metadata_artifact = {
                "file_name": file_name,
                "trimmed_row_count": 0,
                "rows": [],
            }
            return df

        window_size = min(10, len(df))
        top_slice = df.iloc[:window_size]
        non_null_counts = top_slice.notna().sum(axis=1).astype(int).tolist()
        if not non_null_counts:
            self.last_metadata_artifact = {
                "file_name": file_name,
                "trimmed_row_count": 0,
                "rows": [],
            }
            return df

        # Requirement: compare sparsity against the table's mode (most common row density).
        # Ignore 1-cell rows when possible so title rows do not dominate the mode.
        density_series = pd.Series(non_null_counts)
        meaningful_density = density_series[density_series > 1]
        mode_source = meaningful_density if not meaningful_density.empty else density_series
        mode_candidates = mode_source.mode()
        mode_non_null = int(mode_candidates.iloc[0]) if not mode_candidates.empty else 0
        # A mode of 1 is too weak to confidently infer a header row.
        if mode_non_null <= 1:
            logger.warning(
                "Header not confidently identified for file '%s'; returning original table.",
                file_name,
            )
            self.last_metadata_artifact = {
                "file_name": file_name,
                "trimmed_row_count": 0,
                "rows": [],
            }
            return df

        low_density_threshold = max(1, math.floor(mode_non_null * 0.7))
        first_header_pos: int | None = None

        for row_pos in range(window_size):
            row = top_slice.iloc[row_pos]
            non_null_count = int(non_null_counts[row_pos])

            values = [
                str(value).strip().lower()
                for value in row.tolist()
                if pd.notna(value) and str(value).strip() != ""
            ]
            unique_count = len(set(values))
            unique_ratio = (unique_count / len(values)) if values else 0.0
            low_variance = len(values) > 0 and unique_ratio <= 0.35

            is_sparse_vs_mode = non_null_count <= low_density_threshold and non_null_count < mode_non_null
            is_metadata = is_sparse_vs_mode or low_variance

            if is_metadata:
                continue

            # Header confidence: should be fairly dense and text-dominant.
            text_like_cells = sum(any(char.isalpha() for char in value) for value in values)
            text_ratio = (text_like_cells / len(values)) if values else 0.0
            header_confident = non_null_count >= mode_non_null and text_ratio >= 0.5

            if header_confident:
                first_header_pos = row_pos
                break

        if first_header_pos is None:
            logger.warning(
                "Header not confidently identified for file '%s'; returning original table.",
                file_name,
            )
            self.last_metadata_artifact = {
                "file_name": file_name,
                "trimmed_row_count": 0,
                "rows": [],
            }
            return df

        trimmed_rows = first_header_pos
        trimmed_df = df.iloc[:first_header_pos].copy()
        self.last_metadata_artifact = {
            "file_name": file_name,
            "trimmed_row_count": trimmed_rows,
            "rows": trimmed_df.fillna("").astype(str).values.tolist(),
        }

        if trimmed_rows > 5:
            logger.warning(
                "High Metadata Density detected while sanitizing table for file '%s': "
                "%d metadata rows trimmed.",
                file_name,
                trimmed_rows,
            )

        return df.iloc[first_header_pos:].reset_index(drop=True)

    def get_last_metadata_artifact(self) -> dict[str, object] | None:
        """Return JSON-serializable metadata rows trimmed in the last sanitize call."""
        return self.last_metadata_artifact
