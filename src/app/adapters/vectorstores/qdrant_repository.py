from __future__ import annotations

import logging
from typing import Any, Callable
import uuid

from src.app.ports.repositories.vector_repository import VectorRepositoryPort
from src.domain.models.chunk import Chunk, ChunkMetadata


logger = logging.getLogger(__name__)


class QdrantRepository(VectorRepositoryPort):
    """
    Qdrant adapter for chunk persistence.

    Idempotency note:
    Qdrant upsert is idempotent by point ID. Reusing the same chunk ID updates
    the existing point instead of creating duplicates.
    """

    def __init__(
        self,
        qdrant_client: Any,
        collection_name: str,
        embedding_fn: Callable[[list[str]], list[list[float]]],
        batch_size: int = 100,
    ) -> None:
        self._client = qdrant_client
        self._collection_name = collection_name
        self._embedding_fn = embedding_fn
        self._batch_size = batch_size

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        success_count = 0
        failed_count = 0

        for start in range(0, len(chunks), self._batch_size):
            batch = chunks[start : start + self._batch_size]
            try:
                points = self._build_points(batch)
                self._client.upsert(collection_name=self._collection_name, points=points)
                success_count += len(batch)
            except Exception as exc:  # noqa: BLE001
                failed_count += len(batch)
                logger.exception(
                    "Qdrant upsert batch failed for collection '%s' at offset %d: %s",
                    self._collection_name,
                    start,
                    exc,
                )

        if failed_count > 0:
            logger.error(
                "Qdrant upsert completed with failures. stored=%d failed=%d total=%d",
                success_count,
                failed_count,
                len(chunks),
            )
        else:
            logger.info(
                "Qdrant upsert completed successfully. stored=%d total=%d",
                success_count,
                len(chunks),
            )

    def search(self, vector: list[float], doc_id: str | None = None, limit: int = 5) -> list[Chunk]:
        response = self._client.query_points(
            collection_name=self._collection_name,
            query=vector,
            query_filter=self._make_doc_filter(doc_id),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        results = getattr(response, "points", response)
        return [self._search_result_to_chunk(result) for result in results]

    def _build_points(self, chunks: list[Chunk]) -> list[Any]:
        vectors = self._embedding_fn([chunk.content for chunk in chunks])
        
        points: list[Any] = []
        for chunk, vector in zip(chunks, vectors):
            payload = {
                "content": chunk.content,
                "chunk_id": chunk.id,
                "doc_id": chunk.metadata.doc_id,
                "chunk_type": chunk.metadata.chunk_type,
                "parent_header": chunk.metadata.parent_header,
                "page_number": chunk.metadata.page_number,
                **chunk.metadata.extra,
            }
            points.append(self._make_point_struct(self._to_qdrant_point_id(chunk.id), vector, payload))
        return points

    @staticmethod
    def _make_point_struct(point_id: str, vector: list[float], payload: dict[str, Any]) -> Any:
        try:
            from qdrant_client.models import PointStruct  # type: ignore
        except Exception:  # noqa: BLE001
            # Fallback keeps this module importable/testable without qdrant-client.
            return {"id": point_id, "vector": vector, "payload": payload}
        return PointStruct(id=point_id, vector=vector, payload=payload)

    @staticmethod
    def _to_qdrant_point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    @staticmethod
    def _make_doc_filter(doc_id: str | None) -> Any:
        if not doc_id:
            return None
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue  # type: ignore
        except Exception:  # noqa: BLE001
            return {
                "must": [
                    {
                        "key": "doc_id",
                        "match": {"value": doc_id},
                    }
                ]
            }
        return Filter(
            must=[
                FieldCondition(
                    key="doc_id",
                    match=MatchValue(value=doc_id),
                )
            ]
        )

    @staticmethod
    def _search_result_to_chunk(result: Any) -> Chunk:
        payload = dict(getattr(result, "payload", {}) or {})
        chunk_id = str(payload.pop("chunk_id", getattr(result, "id")))
        content = str(payload.pop("content", ""))
        metadata = ChunkMetadata(
            doc_id=str(payload.pop("doc_id", "")),
            chunk_type=str(payload.pop("chunk_type", "")),
            parent_header=str(payload.pop("parent_header", "")),
            page_number=payload.pop("page_number", None),
            extra=payload,
        )
        return Chunk(id=chunk_id, content=content, metadata=metadata)
