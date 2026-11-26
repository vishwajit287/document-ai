import argparse
import json

from pathlib import Path

from app.evaluation.rag_evaluator import (
    RAGEvaluator,
    generate_evaluation_report
)


def main():

    parser = argparse.ArgumentParser(
        description="DocuMind AI - RAG Evaluation"
    )

    parser.add_argument(
        "--dataset",
        default="data/processed/rag_evaluation_dataset.json"
    )

    parser.add_argument(
        "--output",
        default="artifacts/phase7_evaluation_report.json"
    )

    args = parser.parse_args()

    evaluator = RAGEvaluator()

    print("\nStarting DocuMind AI RAG evaluation...")

    results = evaluator.evaluate_dataset(args.dataset)

    report = generate_evaluation_report(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n========== RAG EVALUATION REPORT ==========")

    print(
        json.dumps(
            report["average_scores"],
            indent=4
        )
    )

    print("\nCitation validity:", report["citation_validity"])
    print("Hallucination count:", report["hallucination_count"])

    print("\nReport saved:", output_path)


if __name__ == "__main__":
    main()