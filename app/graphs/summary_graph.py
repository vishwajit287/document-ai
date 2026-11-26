from langgraph.graph import StateGraph, END

from app.graphs.state import SummaryWorkflowState

from app.graphs.nodes import (
    load_document_node,
    route_strategy_node,
    summarize_stuff_node,
    summarize_map_reduce_node,
    summarize_refine_node,
    validate_summary_node,
    save_summary_node
)

from app.graphs.edges import (
    check_status,
    select_strategy
)


def build_summary_graph():

    workflow = StateGraph(SummaryWorkflowState)

    # Register nodes
    workflow.add_node(
        "load_document",
        load_document_node
    )

    workflow.add_node(
        "route_strategy",
        route_strategy_node
    )

    workflow.add_node(
        "summarize_stuff",
        summarize_stuff_node
    )

    workflow.add_node(
        "summarize_map_reduce",
        summarize_map_reduce_node
    )

    workflow.add_node(
        "summarize_refine",
        summarize_refine_node
    )

    workflow.add_node(
        "validate_summary",
        validate_summary_node
    )

    workflow.add_node(
        "save_summary",
        save_summary_node
    )

    # Entry point
    workflow.set_entry_point("load_document")

    # Load -> strategy selection
    workflow.add_conditional_edges(
        "load_document",
        check_status,
        {
            "continue": "route_strategy",
            "error": END
        }
    )

    # Strategy routing
    workflow.add_conditional_edges(
        "route_strategy",
        lambda state: (
            "error"
            if state.get("status") == "failed"
            else select_strategy(state)
        ),
        {
            "stuff": "summarize_stuff",
            "map_reduce": "summarize_map_reduce",
            "refine": "summarize_refine",
            "error": END
        }
    )

    # Each strategy -> validation
    for node_name in [
        "summarize_stuff",
        "summarize_map_reduce",
        "summarize_refine"
    ]:
        workflow.add_conditional_edges(
            node_name,
            check_status,
            {
                "continue": "validate_summary",
                "error": END
            }
        )

    # Validation -> save
    workflow.add_conditional_edges(
        "validate_summary",
        check_status,
        {
            "continue": "save_summary",
            "error": END
        }
    )

    # Final node
    workflow.add_edge("save_summary", END)

    return workflow.compile()