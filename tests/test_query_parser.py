from __future__ import annotations

from convsearch.retrieval.query import build_fts_expressions, parse_query


def test_query_parser_preserves_phrases_exclusions_and_identifiers() -> None:
    parsed = parse_query('"local indexes" IndexFlatIP conversations.json -Pinecone')
    assert parsed.phrases == ("local indexes",)
    assert "IndexFlatIP" in parsed.identifiers
    assert "conversations.json" in parsed.identifiers
    assert parsed.excluded_terms == ("Pinecone",)


def test_query_parser_handles_model_and_column_identifiers() -> None:
    parsed = parse_query("BAAI/bge-small-en-v1.5 parent_source_node_id")
    assert parsed.identifiers == ("BAAI/bge-small-en-v1.5", "parent_source_node_id")
    expressions = build_fts_expressions(parsed)
    assert "BAAI/bge-small-en-v1.5" in expressions[0][1]
    assert "parent_source_node_id" in expressions[0][1]
