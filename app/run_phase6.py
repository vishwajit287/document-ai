import argparse
import json

from pathlib import Path

from app.rag.qa_chain import DocumentQAEngine


def main():

    parser = argparse.ArgumentParser(
        description="DocuMind AI - RAG Question Answering"
    )

    parser.add_argument(
        "--question",
        required=True,
        help="Question to ask about indexed documents"
    )

    parser.add_argument(
        "--source-id",
        default=None,
        help="Restrict retrieval to a specific document"
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of chunks to retrieve"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path"
    )

    args = parser.parse_args()

    engine = DocumentQAEngine()

    print("\nStarting DocuMind AI RAG pipeline...")
    print(f"Question: {args.question}")
    print(f"Top K: {args.k}")

    result = engine.answer_question(
        question=args.question,
        k=args.k,
        source_id=args.source_id
    )

    print("\n========== ANSWER ==========")
    print(result["answer"])

    print("\n========== CITATIONS ==========")

    if result["citations"]:
        for citation in result["citations"]:
            print(
                f"Source: {citation['source']} | "
                f"Page: {citation['page']} | "
                f"Chunk: {citation['chunk_id']}"
            )
    else:
        print("No verified supporting citations.")

    print("\n========== STATUS ==========")
    print(
        "Insufficient information:",
        result["insufficient_information"]
    )

    print("\n========== USAGE ==========")
    print(json.dumps(result["usage"], indent=4))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                result,
                file,
                indent=4,
                ensure_ascii=False
            )

        print(f"\nResult saved to: {output_path}")


if __name__ == "__main__":
    main()