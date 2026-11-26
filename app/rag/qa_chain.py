from app.llm.llm_factory import get_llm
from app.llm.usage_tracker import UsageTracker

from app.rag.retriever import retrieve_context
from app.rag.schemas import RAGResponse
from app.prompts.qa_prompts import build_qa_prompt


def build_context(chunks: list[dict]) -> str:
    context_parts = []

    for chunk in chunks:
        context_parts.append(
            f"""
<document_chunk>
Chunk ID: {chunk["chunk_id"]}
Source: {chunk["source_name"]}
Page: {chunk["page"]}
Content:
{chunk["content"]}
</document_chunk>
"""
        )

    return "\n".join(context_parts)


class DocumentQAEngine:

    def __init__(self):
        self.llm = get_llm()

        self.structured_llm = self.llm.with_structured_output(
            RAGResponse,
            method="json_schema",
            include_raw=True
        )

        self.tracker = UsageTracker()

    def answer_question(
        self,
        question: str,
        k: int = 5,
        source_id: str | None = None
    ):

        # Step 1: Retrieve relevant chunks
        retrieved_chunks = retrieve_context(
            query=question,
            k=k,
            source_id=source_id
        )

        if not retrieved_chunks:
            return {
                "question": question,
                "answer": (
                    "I could not find relevant information "
                    "in the indexed documents."
                ),
                "insufficient_information": True,
                "citations": [],
                "retrieved_chunks": [],
                "usage": self.tracker.get_metrics()
            }

        # Step 2: Build context
        context = build_context(retrieved_chunks)

        # Step 3: Generate answer
        prompt = build_qa_prompt(
            question=question,
            context=context
        )

        start = self.tracker.start_timer()

        response = self.structured_llm.invoke(prompt)

        self.tracker.stop_timer(start)

        raw_response = response.get("raw")

        if raw_response is not None:
            self.tracker.add_usage(raw_response)

        if response.get("parsing_error"):
            raise ValueError(
                f"RAG response parsing failed: "
                f"{response['parsing_error']}"
            )

        parsed_response = response.get("parsed")

        if parsed_response is None:
            raise ValueError(
                "LLM did not return a valid structured response."
            )

        # Step 4: Validate citations against retrieved chunks
        chunks_by_id = {
            chunk["chunk_id"]: chunk
            for chunk in retrieved_chunks
            if chunk.get("chunk_id")
        }

        validated_ids = list(dict.fromkeys(
            chunk_id
            for chunk_id in parsed_response.cited_chunk_ids
            if chunk_id in chunks_by_id
        ))

        citations = []

        for chunk_id in validated_ids:
            chunk = chunks_by_id[chunk_id]

            citations.append({
                "chunk_id": chunk_id,
                "source": chunk["source_name"],
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"],
                "supporting_content": chunk["content"]
            })

        insufficient = parsed_response.insufficient_information

        # Do not return a confident answer with no verified citations.
        if not citations:
            insufficient = True

        answer = parsed_response.answer

        if not citations and not insufficient:
            answer = (
                "I could not verify sufficient supporting evidence "
                "in the retrieved document sections."
            )

        return {
            "question": question,
            "answer": answer,
            "insufficient_information": insufficient,
            "citations": citations,
            "retrieved_chunks": [
                {
                    "chunk_id": chunk["chunk_id"],
                    "page": chunk["page"],
                    "distance": chunk["distance"]
                }
                for chunk in retrieved_chunks
            ],
            "usage": self.tracker.get_metrics()
        }