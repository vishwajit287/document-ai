
from app.summarization.llm_utils import SummarizationEngine

from app.prompts.summary_prompts import (
    SYSTEM_SUMMARY_PROMPT,
    REFINE_PROMPT,
    FINAL_SUMMARY_PROMPT
)


def summarize_refine(chunks):

    engine = SummarizationEngine()

    current_summary = ""

    print("\n[REFINE] Processing document chunks...")

    for index, chunk in enumerate(chunks, start=1):

        print(
            f"Refining chunk {index}/{len(chunks)}"
        )

        if not current_summary:

            current_summary = (
                "No existing summary. "
                "Generate an initial summary."
            )

        prompt = f"""
        {SYSTEM_SUMMARY_PROMPT}

        {REFINE_PROMPT.format(
            existing_summary=current_summary,
            context=chunk.page_content
        )}
        """

        current_summary = engine.invoke_text(
            prompt
        )

    print("\n[FINAL] Generating structured summary...")

    final_prompt = f"""
    {SYSTEM_SUMMARY_PROMPT}

    {FINAL_SUMMARY_PROMPT.format(
        context=current_summary
    )}
    """

    final_summary = engine.invoke_structured(
        final_prompt
    )

    return {
        "strategy": "refine",
        "summary": final_summary.model_dump(),
        "processed_chunks": len(chunks),
        "usage": engine.get_usage()
    }
