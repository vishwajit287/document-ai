import os
import hashlib

from pathlib import Path
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.llm.embeddings import get_embeddings

load_dotenv()


def get_vector_store():
    persist_directory = os.getenv(
        "CHROMA_PERSIST_DIR",
        "data/vector_store"
    )

    collection_name = os.getenv(
        "CHROMA_COLLECTION",
        "documind_chunks"
    )

    Path(persist_directory).mkdir(
        parents=True,
        exist_ok=True
    )

    embeddings = get_embeddings()

    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )


def generate_chunk_id(
    source_id: str,
    chunk_strategy: str,
    chunk_index: int,
    content: str
) -> str:

    unique_value = (
        f"{source_id}:"
        f"{chunk_strategy}:"
        f"{chunk_index}:"
        f"{content}"
    )

    return hashlib.sha256(
        unique_value.encode("utf-8")
    ).hexdigest()


def index_documents(
    documents: list[Document],
    source_id: str,
    source_name: str,
    chunk_strategy: str
):
    if not documents:
        raise ValueError("No documents provided for indexing.")

    vector_store = get_vector_store()

    ids = []
    enriched_documents = []

    for index, document in enumerate(documents):

        chunk_id = generate_chunk_id(
            source_id,
            chunk_strategy,
            index,
            document.page_content
        )

        metadata = {
            key: value
            for key, value in document.metadata.items()
            if isinstance(value, (str, int, float, bool))
        }

        metadata.update({
            "source_id": source_id,
            "source_name": source_name,
            "chunk_strategy": chunk_strategy,
            "chunk_index": index,
            "chunk_id": chunk_id
        })

        enriched_documents.append(
            Document(
                page_content=document.page_content,
                metadata=metadata
            )
        )

        ids.append(chunk_id)

    vector_store.add_documents(
        documents=enriched_documents,
        ids=ids
    )

    return {
        "source": source_name,
        "source_id": source_id,
        "chunk_strategy": chunk_strategy,
        "indexed_chunks": len(enriched_documents),
        "collection": os.getenv(
            "CHROMA_COLLECTION",
            "documind_chunks"
        ),
        "persist_directory": os.getenv(
            "CHROMA_PERSIST_DIR",
            "data/vector_store"
        )
    }


def search_documents(
    query: str,
    k: int = 5,
    source_id: str | None = None
):
    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    vector_store = get_vector_store()

    search_filter = (
        {"source_id": source_id}
        if source_id
        else None
    )

    results = vector_store.similarity_search_with_score(
        query=query,
        k=k,
        filter=search_filter
    )

    formatted_results = []

    for document, distance in results:
        formatted_results.append({
            "content": document.page_content,
            "metadata": document.metadata,
            "distance": round(float(distance), 6)
        })

    return formatted_results