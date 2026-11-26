import argparse
import json

from pathlib import Path
from langchain_core.documents import Document

from app.rag.vector_store import (
    index_documents,
    search_documents
)


def load_phase2_documents(input_path, chunk_strategy):

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Phase 2 file not found: {input_path}"
        )

    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    chunk_data = data.get("chunks", {}).get(chunk_strategy)

    if not chunk_data:
        raise ValueError(
            f"No chunks found for strategy: {chunk_strategy}"
        )

    documents = [
        Document(
            page_content=item["content"],
            metadata=item.get("metadata", {})
        )
        for item in chunk_data
    ]

    source_id = data.get("source_id")

    if not source_id:
        raise ValueError(
            "source_id is missing from Phase 2 JSON."
        )

    source_name = data.get(
        "source",
        input_path.name
    )

    return documents, source_id, source_name


def run_index(args):

    documents, source_id, source_name = (
        load_phase2_documents(
            args.input,
            args.chunk_strategy
        )
    )

    print("\nStarting document embedding and indexing...")
    print(f"Document: {source_name}")
    print(f"Chunks: {len(documents)}")
    print(f"Chunk strategy: {args.chunk_strategy}")

    result = index_documents(
        documents=documents,
        source_id=source_id,
        source_name=source_name,
        chunk_strategy=args.chunk_strategy
    )

    print("\n========== INDEXING RESULT ==========")
    print(json.dumps(result, indent=4))


def run_search(args):

    print("\nSearching vector database...")
    print(f"Query: {args.query}")

    results = search_documents(
        query=args.query,
        k=args.k,
        source_id=args.source_id
    )

    print("\n========== SEARCH RESULTS ==========")

    if not results:
        print("No matching chunks found.")
        return

    for index, result in enumerate(results, start=1):

        print(f"\nResult {index}")
        print("-" * 50)

        print("Distance:", result["distance"])
        print("Metadata:", result["metadata"])
        print("Content:")
        print(result["content"])


def main():

    parser = argparse.ArgumentParser(
        description="DocuMind AI - Phase 5 Embeddings and Vector Store"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True
    )

    index_parser = subparsers.add_parser(
        "index",
        help="Generate embeddings and index document chunks"
    )

    index_parser.add_argument(
        "--input",
        required=True
    )

    index_parser.add_argument(
        "--chunk-strategy",
        choices=["recursive", "structure_aware"],
        default="recursive"
    )

    index_parser.set_defaults(func=run_index)

    search_parser = subparsers.add_parser(
        "search",
        help="Search indexed document chunks"
    )

    search_parser.add_argument(
        "--query",
        required=True
    )

    search_parser.add_argument(
        "--k",
        type=int,
        default=5
    )

    search_parser.add_argument(
        "--source-id",
        default=None
    )

    search_parser.set_defaults(func=run_search)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()