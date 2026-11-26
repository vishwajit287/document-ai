
import argparse
import json

from pathlib import Path
from dotenv import load_dotenv

from langchain_core.documents import Document

from app.summarization.orchestrator import (
    run_summarization
)


load_dotenv()


def load_processed_chunks(
    json_path: str,
    strategy: str
):

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    raw_chunks = data["chunks"][strategy]

    documents = []

    for item in raw_chunks:

        documents.append(
            Document(
                page_content=item["content"],
                metadata=item["metadata"]
            )
        )

    return data, documents


def execute_phase3(
    json_path: str,
    chunk_strategy: str,
    summary_strategy: str
):

    print("\n========== DOCUMIND AI PHASE 3 ==========")

    source_data, chunks = load_processed_chunks(
        json_path,
        chunk_strategy
    )

    print(f"Source: {source_data['source']}")
    print(f"Chunks: {len(chunks)}")
    print(f"Chunking strategy: {chunk_strategy}")

    result = run_summarization(
        chunks,
        strategy=summary_strategy
    )

    output = {
        "source": source_data["source"],
        "chunking_strategy": chunk_strategy,
        "summarization_strategy": summary_strategy,
        "summary": result["summary"],
        "processing_details": {
            key: value
            for key, value in result.items()
            if key not in ["summary", "strategy"]
        },
        "usage": result["usage"]
    }

    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / (
        f"{Path(json_path).stem}_{summary_strategy}_summary.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n========== USAGE METRICS ==========")

    print(
        json.dumps(
            result["usage"],
            indent=4
        )
    )

    print("\n========== GENERATED SUMMARY ==========")

    print(
        json.dumps(
            result["summary"],
            indent=4,
            ensure_ascii=False
        )
    )

    print("\nOutput saved:", output_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="DocuMind AI Phase 3"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to Phase 2 processed JSON"
    )

    parser.add_argument(
        "--chunk-strategy",
        choices=[
            "recursive",
            "structure_aware"
        ],
        default="recursive"
    )

    parser.add_argument(
        "--strategy",
        choices=[
            "stuff",
            "map_reduce",
            "refine"
        ],
        default="map_reduce"
    )

    args = parser.parse_args()

    execute_phase3(
        args.input,
        args.chunk_strategy,
        args.strategy
    )
