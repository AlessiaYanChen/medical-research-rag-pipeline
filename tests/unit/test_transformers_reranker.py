from __future__ import annotations

from src.app.adapters.rerankers.transformers_reranker import TransformersReRanker
from src.domain.models.chunk import Chunk, ChunkMetadata


def test_transformers_reranker_orders_chunks_by_score_and_preserves_metadata() -> None:
    first = Chunk(
        id="DOC-1:00001",
        content="lower score chunk",
        metadata=ChunkMetadata(
            doc_id="DOC-1",
            chunk_type="text",
            parent_header="Background",
            page_number=1,
        ),
    )
    second = Chunk(
        id="DOC-1:00002",
        content="higher score chunk",
        metadata=ChunkMetadata(
            doc_id="DOC-1",
            chunk_type="table",
            parent_header="Results",
            page_number=2,
        ),
    )

    re_ranker = TransformersReRanker(
        scorer=lambda query, chunks: [0.1, 0.9],
    )

    ranked = re_ranker.rank("biomarker effect", [first, second], top_n=2)

    assert [chunk.id for chunk in ranked] == ["DOC-1:00002", "DOC-1:00001"]
    assert ranked[0].metadata.doc_id == "DOC-1"
    assert ranked[0].metadata.parent_header == "Results"
    assert ranked[0].metadata.chunk_type == "table"


def test_transformers_reranker_truncates_to_top_n() -> None:
    chunks = [
        Chunk(
            id=f"DOC-1:{idx:05d}",
            content=f"chunk-{idx}",
            metadata=ChunkMetadata(
                doc_id="DOC-1",
                chunk_type="text",
                parent_header="Results",
                page_number=idx,
            ),
        )
        for idx in range(3)
    ]
    re_ranker = TransformersReRanker(
        scorer=lambda query, items: [0.2, 0.9, 0.5],
    )

    ranked = re_ranker.rank("query", chunks, top_n=2)

    assert [chunk.id for chunk in ranked] == ["DOC-1:00001", "DOC-1:00002"]

