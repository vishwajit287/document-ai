from app.rag.vector_store import search_documents


def retrieve_context(
    query: str,
    k: int = 5,
    source_id: str | None = None
):
    if not query or not query.strip():
        raise ValueError("Question cannot be empty.")

    if k < 1 or k > 20:
        raise ValueError("k must be between 1 and 20.")

    results = search_documents(
        query=query,
        k=k,
        source_id=source_id
    )

    retrieved_chunks = []

    for result in results:
        metadata = result["metadata"]

        retrieved_chunks.append({
            "chunk_id": metadata.get("chunk_id"),
            "source_id": metadata.get("source_id"),
            "source_name": metadata.get("source_name"),
            "page": metadata.get("page"),
            "chunk_index": metadata.get("chunk_index"),
            "content": result["content"],
            "distance": result["distance"]
        })

    return retrieved_chunks