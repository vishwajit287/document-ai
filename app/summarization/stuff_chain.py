
from app.summarization.llm_utils import SummarizationEngine

from app.prompts.summary_prompts import (
    SYSTEM_SUMMARY_PROMPT,
    FINAL_SUMMARY_PROMPT
)

from app.preprocessing.token_counter import count_tokens


MAX_STUFF_INPUT_TOKENS = 10000


def summarize_stuff(chunks):

    engine = SummarizationEngine()

    context = "\n\n".join(
        chunk.page_content
        for chunk in chunks
    )

    token_count = count_tokens(context)

    if token_count > MAX_STUFF_INPUT_TOKENS:

        raise ValueError(
            f"Document contains {token_count} tokens. "
            f"Stuff strategy limit is "
            f"{MAX_STUFF_INPUT_TOKENS} tokens."
        )

    prompt = f"""
    {SYSTEM_SUMMARY_PROMPT}

    {FINAL_SUMMARY_PROMPT.format(context=context)}
    """

    summary = engine.invoke_structured(prompt)

    return {
        "strategy": "stuff",
        "summary": summary.model_dump(),
        "usage": engine.get_usage()
    }
