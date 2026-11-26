from langgraph.graph import END


def check_status(state):
    if state.get("status") == "failed":
        return "error"

    return "continue"


def select_strategy(state):
    strategy = state.get("strategy", "map_reduce")

    routes = {
        "stuff": "stuff",
        "map_reduce": "map_reduce",
        "refine": "refine"
    }

    return routes.get(strategy, "error")