from scripts.ui_app import COLLECTION_ROLES, get_collection_role


def test_known_collections_are_in_roles_map():
    assert "medical_research_chunks_docling_v2_batch1" in COLLECTION_ROLES
    assert "medical_research_chunks_docling_v1" in COLLECTION_ROLES
    assert "medical_research_chunks_v1" in COLLECTION_ROLES


def test_known_collection_returns_correct_role():
    assert "Stage-1" in get_collection_role("medical_research_chunks_docling_v2_batch1")
    assert "Baseline" in get_collection_role("medical_research_chunks_docling_v1")
    assert "read-only" in get_collection_role("medical_research_chunks_v1")


def test_unknown_collection_returns_custom():
    assert get_collection_role("some_other_collection") == "Custom collection"


def test_rollback_collection_name_is_correct():
    # Guards against typos in the guard condition
    assert "read-only" in get_collection_role("medical_research_chunks_v1")
