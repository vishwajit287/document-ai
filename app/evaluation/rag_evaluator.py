from pydantic import BaseModel, Field
import json

from pathlib import Path

from streamlit import metric

from app.llm.llm_factory import get_llm
from app.llm.usage_tracker import UsageTracker

from app.rag.qa_chain import DocumentQAEngine

from app.prompts.evaluation_prompts import (
    build_evaluation_prompt
)


class RAGEvaluationScores(BaseModel):

    faithfulness: float = Field(
        ge=0,
        le=1,
        description=(
            "Proportion of answer claims supported by retrieved context."
        )
    )

    answer_relevance: float = Field(
        ge=0,
        le=1,
        description=(
            "How directly and completely the answer addresses the question."
        )
    )

    context_precision: float = Field(
        ge=0,
        le=1,
        description=(
            "How much of the retrieved context is relevant to the question."
        )
    )

    context_recall: float = Field(
        ge=0,
        le=1,
        description=(
            "How much of the reference answer is supported by retrieved context."
        )
    )

    hallucination_detected: bool = Field(
        description="Whether the answer contains unsupported factual claims."
    )

    unsupported_claims: list[str] = Field(
        description="Specific claims not supported by the retrieved context."
    )

    missing_information: list[str] = Field(
        description="Reference-answer information missing from retrieved context."
    )

    reasoning: str = Field(
        description="Brief explanation for the assigned scores."
    )
    
class RAGEvaluator:

    def __init__(self):
        self.llm = get_llm()

        self.structured_llm = self.llm.with_structured_output(
            RAGEvaluationScores,
            method="json_schema",
            include_raw=True
        )

        self.qa_engine = DocumentQAEngine()
        self.tracker = UsageTracker()

    def evaluate_single(self, test_case: dict):

        question = test_case["question"]
        reference_answer = test_case["reference_answer"]
        source_id = test_case.get("source_id")

        # Run the actual RAG pipeline
        rag_result = self.qa_engine.answer_question(
            question=question,
            k=5,
            source_id=source_id
        )

        contexts = rag_result.get("retrieved_contexts", [])
        answer = rag_result["answer"]

        # Run the LLM evaluator
        prompt = build_evaluation_prompt(
            question=question,
            reference_answer=reference_answer,
            generated_answer=answer,
            contexts=contexts
        )

        start = self.tracker.start_timer()

        response = self.structured_llm.invoke(prompt)

        self.tracker.stop_timer(start)

        raw = response.get("raw")

        if raw is not None:
            self.tracker.add_usage(raw)

        if response.get("parsing_error"):
            raise ValueError(
                f"Evaluation parsing failed: {response['parsing_error']}"
            )

        scores = response.get("parsed")

        if scores is None:
            raise ValueError(
                "Evaluator did not return valid structured output."
            )

        # Deterministic citation validation
        retrieved_ids = {
            item["chunk_id"]
            for item in contexts
        }

        invalid_ids = rag_result.get(
            "invalid_cited_chunk_ids", []
        )

        cited_ids = {
            item["chunk_id"]
            for item in rag_result.get("citations", [])
        }
        
        valid_citations = len(invalid_ids) == 0


        return {
            "id": test_case["id"],
            "question": question,
            "reference_answer": reference_answer,
            "generated_answer": answer,
            "scores": scores.model_dump(),
            "citation_validity": valid_citations,
            "citation_count": len(cited_ids),
            "retrieved_context_count": len(contexts),
            "retrieved_contexts": contexts,
            "citations": rag_result.get("citations", []),
            "rag_usage": rag_result.get("usage", {}),
            "evaluation_usage": self.tracker.get_metrics(),
            "model_cited_chunk_ids": rag_result.get(
                "model_cited_chunk_ids", []
            ),
            "invalid_cited_chunk_ids": invalid_ids,
        }

    def evaluate_dataset(self, dataset_path: str):

        path = Path(dataset_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Evaluation dataset not found: {path}"
            )

        with path.open("r", encoding="utf-8") as file:
            dataset = json.load(file)

        if not dataset:
            raise ValueError("Evaluation dataset is empty.")

        results = []

        for test_case in dataset:
            print(f"\nEvaluating {test_case['id']}...")
            result = self.evaluate_single(test_case)
            results.append(result)

        return results
    
    def generate_evaluation_report(results: list[dict]):

        metric_names = [
            "faithfulness",
            "answer_relevance",
            "context_precision",
            "context_recall"
        ]

        averages = {}

        for metric in metric_names:
            values = [
                item["scores"][metric]
                for item in results
            ]

            averages[metric] = round(
                sum(values) / len(values),
                4
            ) if values else 0

        citation_validity = sum(
            1 for item in results
            if item["citation_validity"]
        ) / len(results) if results else 0

        hallucination_count = sum(
            1 for item in results
            if item["scores"]["hallucination_detected"]
        )

        return {
            "total_questions": len(results),
            "average_scores": averages,
            "citation_validity": round(citation_validity, 4),
            "hallucination_count": hallucination_count,
            "evaluation_results": results
        }