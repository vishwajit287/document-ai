import argparse
import json

from app.graphs.summary_graph import build_summary_graph


def main():

    parser = argparse.ArgumentParser(
        description="DocuMind AI - Phase 4 LangGraph Workflow"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to Phase 2 JSON file"
    )

    parser.add_argument(
        "--chunk-strategy",
        choices=["recursive", "structure_aware"],
        default="recursive"
    )

    parser.add_argument(
        "--strategy",
        choices=["stuff", "map_reduce", "refine"],
        default="map_reduce"
    )

    args = parser.parse_args()

    initial_state = {
        "input_path": args.input,
        "chunk_strategy": args.chunk_strategy,
        "strategy": args.strategy,
        "execution_log": [],
        "status": "running",
        "error": ""
    }

    graph = build_summary_graph()

    print("\nStarting DocuMind AI LangGraph workflow...")
    print(f"Input: {args.input}")
    print(f"Chunk strategy: {args.chunk_strategy}")
    print(f"Summarization strategy: {args.strategy}")

    result = graph.invoke(initial_state)

    print("\n========== WORKFLOW RESULT ==========")
    print(f"Status: {result.get('status')}")

    if result.get("error"):
        print(f"Error: {result['error']}")

    if result.get("output_path"):
        print(f"Output: {result['output_path']}")

    print("\n========== NODE EXECUTION ==========")

    for entry in result.get("execution_log", []):
        print(
            f"{entry['node']} | "
            f"{entry['status']} | "
            f"{entry['execution_time_seconds']} seconds"
        )

    if result.get("usage"):
        print("\n========== LLM USAGE ==========")
        print(
            json.dumps(
                result["usage"],
                indent=4
            )
        )

    if result.get("status") != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()