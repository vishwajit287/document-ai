
from app.summarization.stuff_chain import summarize_stuff
from app.summarization.map_reduce_chain import summarize_map_reduce
from app.summarization.refine_chain import summarize_refine


STRATEGIES = {
    "stuff": summarize_stuff,
    "map_reduce": summarize_map_reduce,
    "refine": summarize_refine
}


def run_summarization(
    chunks,
    strategy="map_reduce"
):

    if strategy not in STRATEGIES:

        raise ValueError(
            f"Unsupported strategy: {strategy}. "
            f"Available: {list(STRATEGIES.keys())}"
        )

    if not chunks:
        raise ValueError(
            "No document chunks available."
        )

    print("\n========== AI SUMMARIZATION ==========")

    print(f"Strategy: {strategy}")
    print(f"Total chunks: {len(chunks)}")

    summarizer = STRATEGIES[strategy]

    result = summarizer(chunks)

    print("\n========== AI SUMMARIZATION COMPLETE ==========")

    return result
