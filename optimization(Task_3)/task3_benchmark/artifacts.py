"""CSV, chart, quality worksheet, and submission report output."""
from pathlib import Path
import pandas as pd
from .config import MODEL_NAME, OUTPUT_DIR, TEST_PROMPTS

QUALITY_COLUMNS = [
    "Prompt_ID", "FP16_Correctness", "4BIT_Correctness",
    "FP16_Relevance", "4BIT_Relevance", "FP16_Clarity",
    "4BIT_Clarity", "Observations",
]


def build_quality_comparison(fp16_results, int4_results):
    return pd.DataFrame({
        "Prompt_ID": range(1, len(TEST_PROMPTS) + 1),
        "Prompt": TEST_PROMPTS,
        "FP16_Response": fp16_results["Response"].reset_index(drop=True),
        "4BIT_Response": int4_results["Response"].reset_index(drop=True),
    })


def load_or_create_quality_scores(output_dir: Path = OUTPUT_DIR):
    """Load existing human ratings, or create a blank 1-5 scoring worksheet."""
    path = output_dir / "quality_scores_template.csv"
    if path.exists():
        scores = pd.read_csv(path)
        if list(scores.columns) != QUALITY_COLUMNS:
            raise ValueError(f"Unexpected columns in {path}; preserve and fix the worksheet manually.")
        return scores
    scores = pd.DataFrame({
        "Prompt_ID": range(1, len(TEST_PROMPTS) + 1),
        **{column: [None] * len(TEST_PROMPTS) for column in QUALITY_COLUMNS[1:-1]},
        "Observations": [""] * len(TEST_PROMPTS),
    })
    scores.to_csv(path, index=False)
    return scores


def _quality_complete(scores):
    rating_columns = QUALITY_COLUMNS[1:-1]
    return (
        scores[rating_columns].notna().all().all()
        and scores["Observations"].fillna("").str.strip().ne("").all()
    )


def _save_charts(summary, output_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Charts skipped; matplotlib is unavailable: {exc}")
        return
    charts = [
        ("speed_comparison.png", "Average Tokens/Second", "Inference Speed", "Tokens per second"),
        ("memory_comparison.png", "Peak GPU Allocated (GB)", "GPU Memory Usage", "Peak allocated GPU memory (GB)"),
    ]
    for filename, metric, title, ylabel in charts:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(summary.columns, summary.loc[metric].values, color=["#4c78a8", "#f58518"])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=200)
        plt.close(fig)


def write_submission_report(summary, memory_saving, speed_difference, gpu_name,
                            quality_scores, output_dir: Path = OUTPUT_DIR):
    fp16_speed = summary.loc["Average Tokens/Second", "FP16"]
    int4_speed = summary.loc["Average Tokens/Second", "4-bit"]
    quality_status = (
        "Complete; see quality_scores_template.csv."
        if _quality_complete(quality_scores)
        else "Pending manual review. Compare all ten answer pairs in quality_comparison.csv, "
             "rate correctness, relevance, and clarity from 1 to 5, and record observations."
    )
    markdown = f"""# Task 3 — Local model and optimization

## Setup

- Model: `{MODEL_NAME}`
- GPU: {gpu_name}
- Evaluation: {len(TEST_PROMPTS)} identical prompts per setting; deterministic decoding; up to 200 generated tokens.

## Performance

| Setting | Mean generation speed (tokens/s) | Mean generation time (s) | Peak allocated GPU memory (GB) | Peak reserved GPU memory (GB) |
|---|---:|---:|---:|---:|
| FP16 | {fp16_speed:.3f} | {summary.loc['Average Generation Time', 'FP16']:.3f} | {summary.loc['Peak GPU Allocated (GB)', 'FP16']:.3f} | {summary.loc['Peak GPU Reserved (GB)', 'FP16']:.3f} |
| 4-bit NF4 | {int4_speed:.3f} | {summary.loc['Average Generation Time', '4-bit']:.3f} | {summary.loc['Peak GPU Allocated (GB)', '4-bit']:.3f} | {summary.loc['Peak GPU Reserved (GB)', '4-bit']:.3f} |

4-bit changed peak allocated memory by {memory_saving:.2f}% and mean speed by {speed_difference:.2f}% relative to FP16. These are measurements from this GPU and software stack; allocated and reserved memory are distinct metrics.

Per-prompt results: [FP16](fp16_results.csv), [4-bit](4bit_results.csv), [summary](performance_comparison.csv). Charts: [speed](speed_comparison.png), [memory](memory_comparison.png).

## Output quality

{quality_status}

Paired answers are saved in [quality_comparison.csv](quality_comparison.csv). Do not infer quality from performance metrics.

## KV cache

The KV cache holds attention key and value vectors already computed for earlier tokens. Reusing them avoids repeating that work during generation. The cache uses GPU memory and grows with sequence length and concurrent requests.

## Why quantization saves memory

Quantization stores model weights with fewer bits. Four-bit weights take less space than 16-bit weights, while metadata, activations, the KV cache, and runtime allocations also use memory.

## Serving 20 concurrent users on one 24 GB GPU

Assume one 24 GB GPU, a 3B model in 4-bit, 20 active requests, a 4096-token context cap, and a 512-token output cap. Use a serving engine with continuous batching and a paged KV cache, monitor cache capacity, and queue requests when full. This is a starting capacity plan; validate with representative prompt lengths because 20 active users does not guarantee 20 requests decode at full speed.
"""
    (output_dir / "Task_3_Submission.md").write_text(markdown, encoding="utf-8")
    (output_dir / "final_report.txt").write_text(markdown, encoding="utf-8")
    return markdown


def save_benchmark_artifacts(fp16_results, int4_results, summary,
                             memory_saving, speed_difference, gpu_name,
                             output_dir: Path = OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    fp16_results.to_csv(output_dir / "fp16_results.csv", index=False)
    int4_results.to_csv(output_dir / "4bit_results.csv", index=False)
    summary.to_csv(output_dir / "performance_comparison.csv")
    quality = build_quality_comparison(fp16_results, int4_results)
    quality.to_csv(output_dir / "quality_comparison.csv", index=False)
    scores = load_or_create_quality_scores(output_dir)
    _save_charts(summary, output_dir)
    report = write_submission_report(
        summary, memory_saving, speed_difference, gpu_name, scores, output_dir
    )
    return quality, scores, report
