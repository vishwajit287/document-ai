from typing import Any, TypedDict


class SummaryWorkflowState(TypedDict, total=False):
    # Input configuration
    input_path: str
    chunk_strategy: str
    strategy: str

    # Loaded document
    source: str
    chunks: list[Any]
    total_pages: int
    source_id: str

    # Summarization results
    summary: dict
    usage: dict
    processing_details: dict

    # Output
    output_path: str

    # Workflow monitoring
    execution_log: list[dict]
    status: str
    error: str