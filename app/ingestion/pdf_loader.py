import fitz
import hashlib

import io
import os
import pytesseract

from PIL import Image
from pathlib import Path
from langchain_core.documents import Document

MIN_TEXT_CHARS = 40
MIXED_PAGE_TEXT_THRESHOLD = 100


def perform_ocr(page):

    pixmap = page.get_pixmap(dpi=300)

    image = Image.open(io.BytesIO(pixmap.tobytes("png")))

    ocr_data = pytesseract.image_to_data(
        image, output_type=pytesseract.Output.DICT, config="--psm 6"
    )

    text = pytesseract.image_to_string(image, config="--psm 6")

    confidences = [
        float(conf) for conf in ocr_data["conf"] if conf != "-1" and float(conf) >= 0
    ]

    average_confidence = sum(confidences) / len(confidences) if confidences else 0

    return text, round(average_confidence, 2)


def extract_pdf_pages(file_path: str) -> list[Document]:

    documents = []
    path = Path(file_path)

    source_id = hashlib.sha256(path.read_bytes()).hexdigest()

    if os.getenv("TESSERACT_CMD"):
        pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD")

    with fitz.open(path) as pdf:

        if pdf.needs_pass:
            raise ValueError("PDF is password protected.")

        total_pages = len(pdf)

        for page_number, page in enumerate(pdf, start=1):

            extracted_text = page.get_text("text").strip()

            has_images = bool(page.get_images(full=True))

            needs_ocr = len(extracted_text) < MIN_TEXT_CHARS or (
                has_images and len(extracted_text) < MIXED_PAGE_TEXT_THRESHOLD
            )

            extraction_method = "native"
            ocr_confidence = None

            if needs_ocr:

                try:
                    ocr_text, ocr_confidence = perform_ocr(page)

                    if len(ocr_text.strip()) > len(extracted_text):
                        extracted_text = ocr_text.strip()
                        extraction_method = "ocr"

                except pytesseract.TesseractNotFoundError as error:
                    raise RuntimeError(
                        "Tesseract is required for this PDF. "
                        "Install Tesseract OCR and configure PATH."
                    ) from error

            if not extracted_text:
                continue

            documents.append(
                Document(
                    page_content=extracted_text,
                    metadata={
                        "source": path.name,
                        "page": page_number,
                        "total_pages": total_pages,
                        "extraction_method": extraction_method,
                        "ocr_confidence": ocr_confidence,
                        "character_count": len(extracted_text),
                    },
                )
            )

    return documents
