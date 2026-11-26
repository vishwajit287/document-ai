import json
from pathlib import Path

from langchain_core.documents import Document
from app.summarization.stuff_chain import summarize_stuff
from app.summarization.map_reduce_chain import summarize_map_reduce
from app.summarization.refine_chain import summarize_refine
from app.summarization.schemas import DocumentSummary

from app.utils.workflow_logger import (
    start_node,
    finish_node,
    append_log
)


def load_document_node(state):
    start = start_node()

    try:
        input_path = Path(state["input_path"])

        if not input_path.exists():
            raise FileNotFoundError(
                f"Phase 2 file not found: {input_path}"
            )

        with input_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        chunk_strategy = state.get(
            "chunk_strategy",
            "recursive"
        )

        chunk_data = data.get("chunks", {}).get(chunk_strategy)

        if not chunk_data:
            raise ValueError(
                f"No chunks found for strategy: {chunk_strategy}"
            )

        documents = [
            Document(
                page_content=item["content"],
                metadata=item.get("metadata", {})
            )
            for item in chunk_data
        ]

        if not documents:
            raise ValueError("The document contains no chunks.")

        log = finish_node(
            "load_document",
            start,
            "success",
            f"Loaded {len(documents)} chunks"
        )

        return {
            "source": data.get("source", input_path.name),
            "source_id": data.get("source_id", ""),
            "total_pages": data.get("total_pages", 0),
            "chunks": documents,
            "execution_log": append_log(state, log),
            "status": "running",
            "error": ""
        }

    except Exception as exc:
        log = finish_node(
            "load_document",
            start,
            "failed",
            str(exc)
        )

        return {
            "status": "failed",
            "error": str(exc),
            "execution_log": append_log(state, log)
        }


def route_strategy_node(state):
    start = start_node()

    try:
        strategy = state.get("strategy", "map_reduce")

        supported = ["stuff", "map_reduce", "refine"]

        if strategy not in supported:
            raise ValueError(
                f"Unsupported strategy: {strategy}. "
                f"Choose from {supported}"
            )

        log = finish_node(
            "route_strategy",
            start,
            "success",
            f"Selected strategy: {strategy}"
        )

        return {
            "strategy": strategy,
            "execution_log": append_log(state, log)
        }

    except Exception as exc:
        log = finish_node(
            "route_strategy",
            start,
            "failed",
            str(exc)
        )

        return {
            "status": "failed",
            "error": str(exc),
            "execution_log": append_log(state, log)
        }


def _run_summarizer(state, strategy, summarizer):
    start = start_node()

    try:
        result = summarizer(state["chunks"])

        log = finish_node(
            f"summarize_{strategy}",
            start,
            "success",
            f"Completed {strategy} summarization"
        )

        return {
            "summary": result["summary"],
            "usage": result.get("usage", {}),
            "processing_details": {
                key: value
                for key, value in result.items()
                if key not in ("summary", "usage", "strategy")
            },
            "execution_log": append_log(state, log)
        }

    except Exception as exc:
        log = finish_node(
            f"summarize_{strategy}",
            start,
            "failed",
            str(exc)
        )

        return {
            "status": "failed",
            "error": str(exc),
            "execution_log": append_log(state, log)
        }


def summarize_stuff_node(state):
    return _run_summarizer(
        state,
        "stuff",
        summarize_stuff
    )


def summarize_map_reduce_node(state):
    return _run_summarizer(
        state,
        "map_reduce",
        summarize_map_reduce
    )


def summarize_refine_node(state):
    return _run_summarizer(
        state,
        "refine",
        summarize_refine
    )


def validate_summary_node(state):
    start = start_node()

    try:
        summary = DocumentSummary.model_validate(
            state["summary"]
        )

        validated_summary = summary.model_dump()

        log = finish_node(
            "validate_summary",
            start,
            "success",
            "Pydantic validation successful"
        )

        return {
            "summary": validated_summary,
            "execution_log": append_log(state, log)
        }

    except Exception as exc:
        log = finish_node(
            "validate_summary",
            start,
            "failed",
            str(exc)
        )

        return {
            "status": "failed",
            "error": str(exc),
            "execution_log": append_log(state, log)
        }


def save_summary_node(state):
    start = start_node()

    try:
        input_path = Path(state["input_path"])

        output_path = (
            Path("artifacts")
            / f"{input_path.stem}_phase4_{state['strategy']}_summary.json"
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        output_data = {
            "source": state.get("source"),
            "source_id": state.get("source_id"),
            "total_pages": state.get("total_pages"),
            "chunk_strategy": state.get("chunk_strategy"),
            "summarization_strategy": state["strategy"],
            "summary": state["summary"],
            "processing_details": state.get(
                "processing_details", {}
            ),
            "usage": state.get("usage", {}),
            "workflow": state.get("execution_log", [])
        }

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                output_data,
                file,
                indent=4,
                ensure_ascii=False
            )

        log = finish_node(
            "save_summary",
            start,
            "success",
            f"Saved output to {output_path}"
        )

        return {
            "output_path": str(output_path),
            "execution_log": append_log(state, log),
            "status": "completed"
        }

    except Exception as exc:
        log = finish_node(
            "save_summary",
            start,
            "failed",
            str(exc)
        )

        return {
            "status": "failed",
            "error": str(exc),
            "execution_log": append_log(state, log)
        }