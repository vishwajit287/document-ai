import json
import math

from pathlib import Path

from app.rag.vector_store import search_documents


def collect_retrieval_results(
    test_file: str,
    output_file: str,
    k: int = 5,
    source_id: str | None = None
):
    with open(test_file, "r", encoding="utf-8") as file:
        test_cases = json.load(file)

    evaluations = []

    for test_case in test_cases:
        query = test_case["query"]

        print(f"\nEvaluating: {query}")

        results = search_documents(
            query=query,
            k=k,
            source_id=source_id
        )

        evaluated_results = []

        for rank, result in enumerate(results, start=1):
            evaluated_results.append({
                "rank": rank,
                "chunk_id": result["metadata"].get("chunk_id"),
                "page": result["metadata"].get("page"),
                "distance": result["distance"],
                "content": result["content"],
                "relevance": None
            })

        evaluations.append({
            "query_id": test_case["query_id"],
            "query": query,
            "expected_topics": test_case.get(
                "expected_topics", []
            ),
            "relevant_chunk_ids": test_case.get(
                "relevant_chunk_ids", []
            ),
            "results": evaluated_results
        })

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            evaluations,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"\nEvaluation results saved: {output_path}")


def calculate_metrics(evaluations: list, k: int = 5):
    precision_scores = []
    reciprocal_ranks = []
    ndcg_scores = []
    recall_scores = []

    recall_available = True

    for evaluation in evaluations:
        results = evaluation["results"][:k]

        relevance = [
            int(item["relevance"] or 0)
            for item in results
        ]

        relevant_count = sum(relevance)

        precision_scores.append(
            relevant_count / k
        )

        first_relevant_rank = next(
            (
                index + 1
                for index, score in enumerate(relevance)
                if score > 0
            ),
            None
        )

        reciprocal_ranks.append(
            1 / first_relevant_rank
            if first_relevant_rank
            else 0
        )

        dcg = sum(
            (2 ** score - 1) / math.log2(index + 2)
            for index, score in enumerate(relevance)
        )

        ideal = sorted(relevance, reverse=True)

        idcg = sum(
            (2 ** score - 1) / math.log2(index + 2)
            for index, score in enumerate(ideal)
        )

        ndcg_scores.append(
            dcg / idcg if idcg else 0
        )

        expected_ids = set(
            evaluation.get("relevant_chunk_ids", [])
        )

        if not expected_ids:
            recall_available = False
        else:
            retrieved_ids = {
                item.get("chunk_id")
                for item in results
            }

            recall_scores.append(
                len(expected_ids & retrieved_ids)
                / len(expected_ids)
            )

    metrics = {
        f"precision_at_{k}": round(
            sum(precision_scores) / len(precision_scores), 4
        ) if precision_scores else 0,

        "mean_reciprocal_rank": round(
            sum(reciprocal_ranks) / len(reciprocal_ranks), 4
        ) if reciprocal_ranks else 0,

        f"ndcg_at_{k}": round(
            sum(ndcg_scores) / len(ndcg_scores), 4
        ) if ndcg_scores else 0
    }

    if recall_available and recall_scores:
        metrics[f"recall_at_{k}"] = round(
            sum(recall_scores) / len(recall_scores), 4
        )
    else:
        metrics["recall"] = (
            "Not calculated: provide independently verified "
            "relevant_chunk_ids for every query."
        )

    return metrics


def score_retrieval_results(
    input_file: str,
    output_file: str,
    k: int = 5
):
    with open(input_file, "r", encoding="utf-8") as file:
        evaluations = json.load(file)

    for evaluation in evaluations:
        for result in evaluation["results"]:
            if result["relevance"] not in (0, 1, 2):
                raise ValueError(
                    "Every result must have relevance 0, 1, or 2."
                )

    metrics = calculate_metrics(evaluations, k)

    output = {
        "metrics": metrics,
        "query_count": len(evaluations),
        "judgments": evaluations
    }

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4, ensure_ascii=False)

    print("\n========== RETRIEVAL QUALITY ==========")
    print(json.dumps(metrics, indent=4))
    print(f"\nScored results saved: {output_file}")