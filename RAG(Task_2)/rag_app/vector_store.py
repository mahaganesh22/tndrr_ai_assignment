from dataclasses import dataclass

import faiss
import numpy as np


@dataclass
class RetrievalIndex:
    index: object
    chunks: list[dict]
    chunk_size: int

    @classmethod
    def build(cls, embedding_model, chunks: list[dict], chunk_size: int):
        if not chunks:
            raise ValueError("Cannot build a retrieval index without text chunks")
        vectors = embedding_model.encode(
            [chunk["text"] for chunk in chunks],
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index=index, chunks=chunks, chunk_size=chunk_size)

    def retrieve(self, query: str, embedding_model, top_k: int = 5) -> list[dict]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        query_vector = embedding_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)
        scores, ids = self.index.search(query_vector, min(top_k, self.index.ntotal))
        results = []
        for score, chunk_id in zip(scores[0], ids[0]):
            if chunk_id < 0:
                continue
            results.append(
                {
                    **self.chunks[int(chunk_id)],
                    "score": float(score),
                    "chunk_id": int(chunk_id),
                }
            )
        return results
