
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.preprocessing.token_counter import (
    count_tokens,
    get_tokenizer
)


MODEL_NAME = "gpt-4o-mini"


def create_recursive_splitter(
    chunk_size=700,
    chunk_overlap=100
):

    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=MODEL_NAME,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ],
        add_start_index=True
    )


def split_documents_recursive(
    documents: list[Document],
    chunk_size=700,
    chunk_overlap=100
) -> list[Document]:

    splitter = create_recursive_splitter(
        chunk_size,
        chunk_overlap
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks, start=1):

        chunk.metadata["chunk_id"] = index

        chunk.metadata["chunk_method"] = "recursive"

        chunk.metadata["token_count"] = count_tokens(
            chunk.page_content,
            MODEL_NAME
        )

    return chunks


# Strategy B: Structure-aware chunking

def split_documents_structure_aware(
    documents: list[Document],
    chunk_size=700,
    chunk_overlap=100
) -> list[Document]:

    chunks = []
    tokenizer = get_tokenizer(MODEL_NAME)

    def token_count(text):
        return len(tokenizer.encode(text))

    def create_chunk(text, metadata):

        if not text.strip():
            return

        chunk = Document(
            page_content=text,
            metadata=metadata.copy()
        )

        chunk.metadata["chunk_method"] = "structure_aware"
        chunk.metadata["token_count"] = token_count(text)

        chunks.append(chunk)

    for document in documents:

        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(
                r"\n\s*\n",
                document.page_content
            )
            if paragraph.strip()
        ]

        current_paragraphs = []

        def flush():

            if current_paragraphs:

                text = "\n\n".join(current_paragraphs)

                create_chunk(
                    text,
                    document.metadata
                )

                current_paragraphs.clear()

        for paragraph in paragraphs:

            paragraph_tokens = token_count(paragraph)

            # Handle paragraphs exceeding the token budget
            if paragraph_tokens > chunk_size:

                flush()

                oversized_splitter = create_recursive_splitter(
                    chunk_size,
                    chunk_overlap
                )

                for part in oversized_splitter.split_text(
                    paragraph
                ):

                    create_chunk(
                        part,
                        document.metadata
                    )

                continue

            candidate = "\n\n".join(
                current_paragraphs + [paragraph]
            )

            if (
                current_paragraphs
                and token_count(candidate) > chunk_size
            ):

                previous_text = "\n\n".join(
                    current_paragraphs
                )

                flush()

                # Carry a small overlap from the previous chunk
                previous_tokens = tokenizer.encode(
                    previous_text
                )

                overlap_text = tokenizer.decode(
                    previous_tokens[-chunk_overlap:]
                )

                if (
                    overlap_text.strip()
                    and token_count(
                        overlap_text + "\n\n" + paragraph
                    ) <= chunk_size
                ):
                    current_paragraphs.append(overlap_text.strip())

            current_paragraphs.append(paragraph)

        flush()

    for index, chunk in enumerate(chunks, start=1):

        chunk.metadata["chunk_id"] = index

    return chunks
