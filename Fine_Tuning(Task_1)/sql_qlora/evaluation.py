"""Greedy decoding and normalized exact-match evaluation."""

import re

import numpy as np
import torch

from .config import SYSTEM_PROMPT, user_prompt


def normalize_sql(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return normalized[:-1].rstrip() if normalized.endswith(";") else normalized


def make_prompt(example, tokenizer):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(example["context"], example["question"])},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def predict(model, tokenizer, example, max_new_tokens=256):
    prompt = make_prompt(example, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    completion = generated[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(completion, skip_special_tokens=True).strip()


def evaluate(model, tokenizer, examples, max_new_tokens=256):
    model.eval()
    rows = []
    for index, example in enumerate(examples):
        prediction = predict(model, tokenizer, example, max_new_tokens)
        reference = str(example["answer"])
        rows.append(
            {
                "test_index": index,
                "question": example["question"],
                "prediction": prediction,
                "reference": reference,
                "exact_match": normalize_sql(prediction) == normalize_sql(reference),
            }
        )
    correct = sum(row["exact_match"] for row in rows)
    return {
        "correct": correct,
        "total": len(rows),
        "exact_match": correct / len(rows) if rows else 0.0,
        "predictions": rows,
    }
