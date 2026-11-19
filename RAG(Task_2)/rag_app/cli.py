import argparse
import json
from pathlib import Path

import pandas as pd

from .application import RAGApplication
from .config import Settings
from .evaluation import evaluate_retriever, load_questions
from .generation import LocalAnswerGenerator


def _settings(args) -> Settings:
    defaults = Settings()
    return Settings(
        pdf_path=Path(args.pdf) if args.pdf else defaults.pdf_path,
        questions_path=Path(args.questions) if getattr(args, "questions", None) else defaults.questions_path,
        output_dir=Path(args.output_dir) if getattr(args, "output_dir", None) else defaults.output_dir,
        embedding_model=getattr(args, "embedding_model", None) or defaults.embedding_model,
        generation_model=getattr(args, "generation_model", None) or defaults.generation_model,
        chunk_sizes=defaults.chunk_sizes,
        overlap=args.overlap,
        top_k=args.top_k,
        minimum_pages=defaults.minimum_pages,
    )


def run_evaluation(args):
    settings = _settings(args)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    app = RAGApplication(settings)
    questions = load_questions(settings.questions_path)
    result_frames = []
    summary = []

    for size in settings.chunk_sizes:
        index = app.build_index(size)
        results = evaluate_retriever(
            questions, index, app.embedding_model, top_k=settings.top_k
        )
        results["Chunk Size"] = size
        result_frames.append(results)
        hits = int(results["Hit@5"].sum())
        recall = float(results["Hit@5"].mean()) if len(results) else 0.0
        summary.append({"Chunk Size": size, "Recall@5": recall, "Hits": hits})
        print(f"{size}-token Recall@5: {recall:.2%} ({hits}/{len(questions)})")

    combined = pd.concat(result_frames, ignore_index=True)
    summary_frame = pd.DataFrame(summary)
    combined.to_csv(settings.output_dir / "retrieval_evaluation.csv", index=False)
    summary_frame.to_csv(settings.output_dir / "chunk_size_comparison.csv", index=False)

    misses = combined[combined["Hit@5"] == 0].drop_duplicates("Question").head(2)
    question_map = {item["question"]: item for item in questions}
    failures = []
    for _, row in misses.iterrows():
        item = question_map[row["Question"]]
        failures.append(
            {
                "Question": item["question"],
                "Chunk Size": int(row["Chunk Size"]),
                "Correct Source Page (printed)": item["gold_printed_page"],
                "Correct Source Page (PDF)": item["gold_page"],
                "Correct Source Passage": item["gold_passage"],
                "Retrieved Pages": row["Retrieved Pages"],
                "Possible Fix": "Add BM25 candidates, retrieve a larger shortlist, then re-rank it.",
            }
        )
    pd.DataFrame(failures).to_csv(
        settings.output_dir / "retrieval_failure_analysis.csv", index=False
    )

    best = summary_frame.sort_values("Recall@5", ascending=False).iloc[0]
    print(f"Better result in this run: {int(best['Chunk Size'])}-token chunks")
    print(f"Reports written to: {settings.output_dir.resolve()}")


def run_ask(args):
    settings = _settings(args)
    app = RAGApplication(settings)
    index = app.build_index(args.chunk_size)
    retrieved = index.retrieve(args.question, app.embedding_model, top_k=settings.top_k)
    generator = LocalAnswerGenerator(settings.generation_model)
    response = {
        "question": args.question,
        "answer": generator.answer(args.question, retrieved),
        "sources": [
            {"pdf_page": item["page"], "score": round(item["score"], 4)}
            for item in retrieved
        ],
    }
    print(json.dumps(response, indent=2, ensure_ascii=False))


def build_parser():
    parser = argparse.ArgumentParser(description="RAG over the NIST AI RMF PDF")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, handler in (("evaluate", run_evaluation), ("ask", run_ask)):
        command = subparsers.add_parser(name)
        command.add_argument("--pdf", help="Path to a PDF (defaults to the included NIST PDF)")
        command.add_argument("--embedding-model", help="Sentence Transformers model name")
        command.add_argument("--overlap", type=int, default=50)
        command.add_argument("--top-k", type=int, default=5)
        if name == "evaluate":
            command.add_argument("--questions", help="Path to the test questions CSV")
            command.add_argument("--output-dir", help="Directory for evaluation CSVs")
        else:
            command.add_argument("--chunk-size", type=int, choices=(256, 512), default=256)
            command.add_argument("--generation-model", help="Local Transformers model name")
            command.add_argument("--question", required=True)
        command.set_defaults(handler=handler)
    return parser


def main():
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
