import re


def clean_text(text: str) -> str:
    text = re.sub(r"\x00", "", text)
    return re.sub(r"\s+", " ", text).strip()


def create_chunks(
    pages: list[dict], tokenizer, chunk_size: int, overlap: int = 50
) -> list[dict]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks = []
    step = chunk_size - overlap
    for page in pages:
        token_ids = tokenizer.encode(
            clean_text(page["text"]), add_special_tokens=False
        )
        for start in range(0, len(token_ids), step):
            end = min(start + chunk_size, len(token_ids))
            text = tokenizer.decode(token_ids[start:end], skip_special_tokens=True)
            if len(text.strip()) > 30:
                chunks.append(
                    {
                        "text": text,
                        "page": page["page"],
                        "start_token": start,
                        "end_token": end,
                    }
                )
            if end == len(token_ids):
                break
    return chunks
