
import fitz

from statistics import median
from pathlib import Path


def extract_document_structure(file_path: str) -> dict:

    structure = {}

    with fitz.open(file_path) as pdf:

        for page_number, page in enumerate(pdf, start=1):

            page_data = page.get_text("dict")

            lines = []
            font_sizes = []

            for block in page_data["blocks"]:

                if "lines" not in block:
                    continue

                for line in block["lines"]:

                    spans = line["spans"]

                    line_text = "".join(
                        span["text"] for span in spans
                    ).strip()

                    if not line_text:
                        continue

                    max_font_size = max(
                        span["size"] for span in spans
                    )

                    is_bold = any(
                        span["flags"] & 16
                        for span in spans
                    )

                    font_sizes.append(max_font_size)

                    lines.append({
                        "text": line_text,
                        "font_size": max_font_size,
                        "is_bold": is_bold
                    })

            if not font_sizes:
                structure[page_number] = []
                continue

            body_font_size = median(font_sizes)

            headings = []

            for line in lines:

                text = line["text"]
                font_size = line["font_size"]

                is_heading = (
                    font_size >= body_font_size * 1.2
                    or (
                        line["is_bold"]
                        and len(text) <= 120
                    )
                )

                if (
                    is_heading
                    and len(text) <= 150
                    and len(text.split()) <= 20
                    and text not in headings
                ):
                    headings.append(text)

            structure[page_number] = headings

    return structure


def enrich_document_metadata(
    documents: list,
    structure: dict
) -> list:

    for document in documents:

        page_number = document.metadata["page"]

        document.metadata["headings"] = structure.get(
            page_number,
            []
        )

    return documents
