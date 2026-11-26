
from pydantic import BaseModel, Field


class KeyPoint(BaseModel):
    title: str = Field(
        description="Title of the important point"
    )

    description: str = Field(
        description="Explanation of the point"
    )


class ImportantFinding(BaseModel):
    finding: str = Field(
        description="Important finding from the document"
    )

    evidence: str = Field(
        description="Supporting evidence from the document"
    )


class DocumentSummary(BaseModel):

    document_title: str = Field(
        description="Title of the document"
    )

    executive_summary: str = Field(
        description="Concise overview of the entire document"
    )

    key_points: list[KeyPoint] = Field(
        description="Most important points"
    )

    important_findings: list[ImportantFinding] = Field(
        description="Important findings with supporting evidence"
    )

    conclusion: str = Field(
        description="Conclusion explicitly supported by the document"
    )

    action_items: list[str] = Field(
        description="Action items explicitly mentioned in the document"
    )

    topics: list[str] = Field(
        description="Major topics discussed in the document"
    )
