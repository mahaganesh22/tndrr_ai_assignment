from pathlib import Path

import fitz


def extract_pages(pdf_path: str | Path, minimum_pages: int = 30) -> list[dict]:
    """Extract non-empty text pages; page numbers are 1-based physical PDF pages."""
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    pages = []
    with fitz.open(path) as document:
        if len(document) < minimum_pages:
            raise ValueError(
                f"PDF has {len(document)} pages; at least {minimum_pages} are required."
            )
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append({"page": page_number, "text": text})

    if len(pages) < minimum_pages:
        raise ValueError(
            f"Only {len(pages)} pages contain extractable text; "
            f"at least {minimum_pages} are required."
        )
    return pages
