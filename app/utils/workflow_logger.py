from datetime import datetime, timezone
from time import perf_counter


def start_node():
    return perf_counter()


def finish_node(
    node_name: str,
    start_time: float,
    status: str,
    details: str = ""
):
    elapsed = round(perf_counter() - start_time, 3)

    return {
        "node": node_name,
        "status": status,
        "execution_time_seconds": elapsed,
        "details": details,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


def append_log(state: dict, entry: dict):
    logs = list(state.get("execution_log", []))
    logs.append(entry)
    return logs