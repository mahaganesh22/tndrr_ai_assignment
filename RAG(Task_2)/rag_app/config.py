from dataclasses import dataclass, field
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    pdf_path: Path = APP_DIR / "nist_ai_rmf.pdf"
    questions_path: Path = APP_DIR / "15_test_questions_and_source_passages.csv"
    output_dir: Path = APP_DIR / "outputs"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    generation_model: str = "google/flan-t5-base"
    chunk_sizes: tuple[int, ...] = (256, 512)
    overlap: int = 50
    top_k: int = 5
    minimum_pages: int = 30
