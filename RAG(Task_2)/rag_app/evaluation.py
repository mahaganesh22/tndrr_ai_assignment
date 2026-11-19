from pathlib import Path

import pandas as pd


def load_questions(csv_path: str | Path) -> list[dict]:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Question CSV not found: {path}")
    frame = pd.read_csv(path)
    required = {
        "Question",
        "Correct Source Page (PDF)",
        "Correct Source Page (printed)",
        "Correct Source Passage",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Question CSV is missing columns: {sorted(missing)}")
    return [
        {
            "question": row["Question"],
            "gold_page": int(row["Correct Source Page (PDF)"]),
            "gold_printed_page": int(row["Correct Source Page (printed)" ]),
            "gold_passage": row["Correct Source Passage"],
        }
        for _, row in frame.iterrows()
    ]


def evaluate_retriever(questions: list[dict], retrieval_index, embedding_model, top_k=5):
    rows = []
    for item in questions:
        retrieved = retrieval_index.retrieve(item["question"], embedding_model, top_k)
        retrieved_pages = [entry["page"] for entry in retrieved]
        rows.append(
            {
                "Question": item["question"],
                "Gold PDF Page": item["gold_page"],
                "Gold Printed Page": item["gold_printed_page"],
                "Retrieved Pages": retrieved_pages,
                "Hit@5": int(item["gold_page"] in retrieved_pages),
                "Best Similarity": retrieved[0]["score"] if retrieved else 0.0,
            }
        )
    return pd.DataFrame(rows)
