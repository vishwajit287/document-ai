from pydantic import BaseModel, Field


class RAGResponse(BaseModel):
    answer: str = Field(
        description="Answer based only on the supplied document context"
    )

    cited_chunk_ids: list[str] = Field(
        description=(
            "IDs of the supplied chunks that directly support the answer. "
            "Use only IDs provided in the context."
        )
    )

    insufficient_information: bool = Field(
        description=(
            "True if the supplied context does not contain enough "
            "information to answer the question reliably."
        )
    )