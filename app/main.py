
import argparse
import json

from pathlib import Path
from dotenv import load_dotenv

from app.ingestion.document_validator import validate_pdf
from app.ingestion.pdf_loader import extract_pdf_pages

from app.preprocessing.text_cleaner import clean_documents

from app.preprocessing.structure_analyzer import (
    extract_document_structure,
    enrich_document_metadata
)

from app.preprocessing.chunker import (
    split_documents_recursive,
    split_documents_structure_aware
)

from app.preprocessing.token_counter import count_tokens


load_dotenv()


def serialize_chunks(chunks):

    return [
        {
            "chunk_id": chunk.metadata["chunk_id"],
            "content": chunk.page_content,
            "metadata": chunk.metadata
        }
        for chunk in chunks
    ]


def calculate_metrics(chunks):

    token_counts = [
        chunk.metadata["token_count"]
        for chunk in chunks
    ]

    return {
        "total_chunks": len(chunks),
        "total_tokens": sum(token_counts),
        "average_tokens": (
            round(sum(token_counts) / len(token_counts), 2)
            if token_counts else 0
        ),
        "maximum_tokens": max(token_counts, default=0),
        "minimum_tokens": min(token_counts, default=0)
    }


def process_pdf(file_path: str):

    print("\n========== DOCUMIND AI ==========")

    # 1. Validate document
    print("\n[1] Validating PDF...")

    pdf_path = validate_pdf(file_path)

    # 2. Extract text and OCR
    print("\n[2] Extracting PDF content...")

    documents = extract_pdf_pages(str(pdf_path))

    if not documents:
        raise ValueError("No readable PDF content found.")

    ocr_pages = sum(
        1 for doc in documents
        if doc.metadata["extraction_method"] == "ocr"
    )

    print(f"Pages extracted: {len(documents)}")
    print(f"OCR pages: {ocr_pages}")

    # 3. Clean extracted text
    print("\n[3] Cleaning text...")

    cleaned_documents = clean_documents(documents)

    total_characters = sum(
        len(doc.page_content)
        for doc in cleaned_documents
    )

    # 4. Document structure
    print("\n[4] Detecting headings and structure...")

    structure = extract_document_structure(
        str(pdf_path)
    )

    cleaned_documents = enrich_document_metadata(
        cleaned_documents,
        structure
    )

    total_headings = sum(
        len(headings)
        for headings in structure.values()
    )

    print(f"Headings detected: {total_headings}")

    # 5. Token analysis
    print("\n[5] Calculating document tokens...")

    total_tokens = sum(
        count_tokens(doc.page_content)
        for doc in cleaned_documents
    )

    print(f"Total characters: {total_characters}")
    print(f"Total tokens: {total_tokens}")

    # 6. Recursive chunking
    print("\n[6] Recursive token chunking...")

    recursive_chunks = split_documents_recursive(
        cleaned_documents,
        chunk_size=700,
        chunk_overlap=100
    )

    recursive_metrics = calculate_metrics(
        recursive_chunks
    )

    print(f"Chunks: {len(recursive_chunks)}")

    # 7. Structure-aware chunking
    print("\n[7] Structure-aware chunking...")

    structure_chunks = split_documents_structure_aware(
        cleaned_documents,
        chunk_size=700,
        chunk_overlap=100
    )

    structure_metrics = calculate_metrics(
        structure_chunks
    )

    print(f"Chunks: {len(structure_chunks)}")

    # 8. Comparison
    print("\n========== CHUNKING COMPARISON ==========")

    print(
        f"Recursive chunks: "
        f"{recursive_metrics['total_chunks']}"
    )

    print(
        f"Structure-aware chunks: "
        f"{structure_metrics['total_chunks']}"
    )

    print(
        f"Recursive average tokens: "
        f"{recursive_metrics['average_tokens']}"
    )

    print(
        f"Structure-aware average tokens: "
        f"{structure_metrics['average_tokens']}"
    )

    # 9. Save output
    output = {
        "source": pdf_path.name,
        "source_id": cleaned_documents[0].metadata["source_id"],
        "total_pages": cleaned_documents[0].metadata["total_pages"],
        "pages_extracted": len(cleaned_documents),
        "ocr_pages": ocr_pages,
        "total_characters": total_characters,
        "total_tokens": total_tokens,
        "total_headings": total_headings,

        "chunking_comparison": {
            "recursive": recursive_metrics,
            "structure_aware": structure_metrics
        },

        "chunks": {
            "recursive": serialize_chunks(recursive_chunks),
            "structure_aware": serialize_chunks(structure_chunks)
        }
    }

    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / (
        f"{pdf_path.stem}_phase2.json"
    )

    with open(output_file, "w", encoding="utf-8") as file:

        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n========== PHASE 2 COMPLETE ==========")

    print(f"Pages: {len(cleaned_documents)}")
    print(f"Tokens: {total_tokens}")
    print(f"Headings: {total_headings}")

    print(f"Output: {output_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="DocuMind AI - Phase 2"
    )

    parser.add_argument(
        "pdf_path",
        help="PDF document path"
    )

    args = parser.parse_args()

    process_pdf(args.pdf_path)
