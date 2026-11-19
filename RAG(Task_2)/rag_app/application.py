from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

from .chunking import create_chunks
from .config import Settings
from .document import extract_pages
from .vector_store import RetrievalIndex


@dataclass
class RAGApplication:
    settings: Settings

    def __post_init__(self):
        self.pages = extract_pages(
            self.settings.pdf_path, minimum_pages=self.settings.minimum_pages
        )
        self.embedding_model = SentenceTransformer(self.settings.embedding_model)
        # The model supports up to 512 tokens; this preserves the 512-token experiment.
        self.embedding_model.max_seq_length = max(self.settings.chunk_sizes)
        self.indexes: dict[int, RetrievalIndex] = {}

    def build_index(self, chunk_size: int) -> RetrievalIndex:
        if chunk_size not in self.settings.chunk_sizes:
            raise ValueError(
                f"Unsupported chunk size {chunk_size}; choose from {self.settings.chunk_sizes}"
            )
        if chunk_size not in self.indexes:
            chunks = create_chunks(
                self.pages,
                self.embedding_model.tokenizer,
                chunk_size,
                overlap=self.settings.overlap,
            )
            self.indexes[chunk_size] = RetrievalIndex.build(
                self.embedding_model, chunks, chunk_size
            )
        return self.indexes[chunk_size]
