
import os

from dotenv import load_dotenv

load_dotenv()


def validate_langsmith_config() -> bool:
    tracing = os.getenv("LANGSMITH_TRACING", "false").lower()
    api_key = os.getenv("LANGSMITH_API_KEY")
    project = os.getenv("LANGSMITH_PROJECT")

    if tracing != "true":
        print("LangSmith tracing is disabled.")
        return False

    if not api_key:
        raise ValueError("LANGSMITH_API_KEY is missing from .env")

    if not project:
        raise ValueError("LANGSMITH_PROJECT is missing from .env")

    print("LangSmith configuration validated.")
    print(f"Project: {project}")
    print("Tracing: Enabled")

    return True
