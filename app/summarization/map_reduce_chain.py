
from app.summarization.llm_utils import SummarizationEngine

from app.prompts.summary_prompts import (
    SYSTEM_SUMMARY_PROMPT,
    MAP_PROMPT,
    REDUCE_PROMPT,
    FINAL_SUMMARY_PROMPT
)

from app.preprocessing.token_counter import count_tokens


REDUCE_BATCH_TOKEN_LIMIT = 6000
MAX_REDUCTION_ROUNDS = 5


def create_batches(texts, token_limit):

    batches = []
    current_batch = []
    current_tokens = 0

    for text in texts:

        tokens = count_tokens(text)

        if current_batch and (
            current_tokens + tokens > token_limit
        ):

            batches.append(current_batch)

            current_batch = []
            current_tokens = 0

        current_batch.append(text)
        current_tokens += tokens

    if current_batch:
        batches.append(current_batch)

    return batches


def summarize_map_reduce(chunks):

    engine = SummarizationEngine()

    # MAP PHASE
    print("\n[MAP] Summarizing individual chunks...")

    mapped_summaries = []

    for index, chunk in enumerate(chunks, start=1):

        print(
            f"Processing chunk {index}/{len(chunks)}"
        )

        prompt = f"""
        {SYSTEM_SUMMARY_PROMPT}

        {MAP_PROMPT.format(
            context=chunk.page_content
        )}
        """

        result = engine.invoke_text(prompt)

        mapped_summaries.append(result)

    # REDUCE PHASE
    print("\n[REDUCE] Combining summaries...")

    current_summaries = mapped_summaries

    reduction_round = 0

    while (
        count_tokens("\n\n".join(current_summaries))
        > REDUCE_BATCH_TOKEN_LIMIT
    ):

        reduction_round += 1

        if reduction_round > MAX_REDUCTION_ROUNDS:
            raise ValueError(
                "Maximum reduction rounds exceeded."
            )

        print(
            f"Reduction round: {reduction_round}"
        )

        batches = create_batches(
            current_summaries,
            REDUCE_BATCH_TOKEN_LIMIT
        )

        reduced_summaries = []

        for batch in batches:

            context = "\n\n".join(batch)

            prompt = f"""
            {SYSTEM_SUMMARY_PROMPT}

            {REDUCE_PROMPT.format(
                context=context
            )}
            """

            result = engine.invoke_text(prompt)

            reduced_summaries.append(result)

        if (
            len(reduced_summaries)
            >= len(current_summaries)
        ):
            raise ValueError(
                "Reduction is not converging."
            )

        current_summaries = reduced_summaries

    # FINAL SUMMARY
    print("\n[FINAL] Generating structured summary...")

    final_context = "\n\n".join(current_summaries)

    final_prompt = f"""
    {SYSTEM_SUMMARY_PROMPT}

    {FINAL_SUMMARY_PROMPT.format(
        context=final_context
    )}
    """

    final_summary = engine.invoke_structured(
        final_prompt
    )

    return {
        "strategy": "map_reduce",
        "summary": final_summary.model_dump(),
        "map_chunks": len(mapped_summaries),
        "reduction_rounds": reduction_round,
        "usage": engine.get_usage()
    }
