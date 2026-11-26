import argparse

from app.evaluation.retrieval_evaluator import (
    collect_retrieval_results,
    score_retrieval_results
)


def main():
    parser = argparse.ArgumentParser(
        description="DocuMind AI Retrieval Quality Evaluation"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True
    )

    collect_parser = subparsers.add_parser("collect")

    collect_parser.add_argument(
        "--tests",
        default="data/processed/retrieval_test_cases.json"
    )

    collect_parser.add_argument(
        "--output",
        default="artifacts/retrieval_eval_results.json"
    )

    collect_parser.add_argument("--k", type=int, default=5)
    collect_parser.add_argument("--source-id", default=None)

    score_parser = subparsers.add_parser("score")

    score_parser.add_argument(
        "--input",
        default="artifacts/retrieval_eval_results.json"
    )

    score_parser.add_argument(
        "--output",
        default="artifacts/retrieval_eval_scored.json"
    )

    score_parser.add_argument("--k", type=int, default=5)

    args = parser.parse_args()

    if args.command == "collect":
        collect_retrieval_results(
            args.tests,
            args.output,
            args.k,
            args.source_id
        )

    elif args.command == "score":
        score_retrieval_results(
            args.input,
            args.output,
            args.k
        )


if __name__ == "__main__":
    main()