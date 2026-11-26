
from pathlib import Path

MAX_FILE_SIZE_MB = 50


def validate_pdf(file_path: str) -> Path:
    """
    Validate PDF existence, extension and file size.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {file_path}"
        )

    if not path.is_file():
        raise ValueError("Provided path is not a file.")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    file_size_mb = path.stat().st_size / (1024 * 1024)

    if file_size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(
            f"PDF exceeds maximum size of {MAX_FILE_SIZE_MB} MB."
        )

    return path
