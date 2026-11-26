
import re

from langchain_core.documents import Document


def clean_text(text: str) -> str:
    """
    Normalize extracted PDF text.
    """

    # Normalize line endings
    text = text.replace("\r\n", "\n")

    # Remove excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove leading and trailing spaces
    text = text.strip()

    return text


def clean_documents(
    documents: list[Document]
) -> list[Document]:

    cleaned_documents = []

    for document in documents:

        cleaned_content = clean_text(
            document.page_content
        )

        if not cleaned_content:
            continue

        cleaned_documents.append(
            Document(
                page_content=cleaned_content,
                metadata=document.metadata.copy()
            )
        )

    return cleaned_documents
