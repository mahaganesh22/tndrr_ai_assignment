import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class LocalAnswerGenerator:
    """Optional local FLAN-T5 generator; retrieval works without this module."""

    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def answer(self, question: str, retrieved: list[dict]) -> str:
        context_parts = []
        remaining = 340
        for result in retrieved:
            ids = self.tokenizer.encode(result["text"], add_special_tokens=False)
            ids = ids[: min(100, remaining)]
            if ids:
                excerpt = self.tokenizer.decode(ids, skip_special_tokens=True)
                context_parts.append(f"[PDF page {result['page']}] {excerpt}")
                remaining -= len(ids)
            if remaining <= 0:
                break

        context_text = "\n\n".join(context_parts)
        prompt = (
            "Answer using only the context. Cite the PDF page. If the context does not "
            "answer the question, say the document does not contain sufficient information.\n\n"
            f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
