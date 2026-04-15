from __future__ import annotations

from io import StringIO

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.app.tables.table_normalizer import TableNormalizer
from src.ports.parser_port import ParsedTable


def parsed_table_to_dataframe(table: ParsedTable) -> pd.DataFrame:
    if table.headers or table.rows:
        row_lists: list[list[str]] = []
        if table.headers:
            row_lists.append([str(value) for value in table.headers])
        for row in table.rows:
            row_lists.append([str(row.get(header, "")) for header in table.headers])
        return pd.DataFrame(row_lists)

    return pd.read_csv(StringIO(table.csv), header=None, engine="python", on_bad_lines="skip")


def dataframe_to_rows(df: pd.DataFrame) -> list[dict[str, str]]:
    if df.empty or len(df) < 2:
        return []

    headers = [str(value) for value in df.iloc[0].tolist()]
    rows: list[dict[str, str]] = []
    for row_idx in range(1, len(df)):
        values = ["" if pd.isna(value) else str(value) for value in df.iloc[row_idx].tolist()]
        if len(values) < len(headers):
            values.extend([""] * (len(headers) - len(values)))
        rows.append(dict(zip(headers, values)))
    return rows


def normalize_tables(parsed_tables: list[ParsedTable], file_name: str) -> list[dict[str, object]]:
    normalizer = TableNormalizer()
    normalized: list[dict[str, object]] = []

    for table in parsed_tables:
        cleaned_df = normalizer.sanitize_table(
            df=parsed_table_to_dataframe(table),
            file_name=file_name,
        )
        artifact: dict[str, object] = {
            "csv": cleaned_df.to_csv(index=False, header=False).strip(),
            "rows": dataframe_to_rows(cleaned_df),
        }

        metadata_artifact = normalizer.get_last_metadata_artifact()
        if metadata_artifact:
            artifact["normalization_metadata"] = metadata_artifact

        normalized.append(artifact)

    return normalized


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    *,
    recreate: bool = False,
) -> None:
    exists = client.collection_exists(collection_name)
    if exists and recreate:
        client.delete_collection(collection_name)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
