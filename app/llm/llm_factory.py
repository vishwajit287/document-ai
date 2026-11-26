
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_llm():

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing from .env"
        )

    model_name = os.getenv(
        "OPENAI_MODEL",
        "gpt-4o-mini"
    )

    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        temperature=0,
        max_retries=2,
        timeout=120
    )

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
    )
