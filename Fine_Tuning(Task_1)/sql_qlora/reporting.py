"""Persist evaluation reports and print the required 7B memory estimate."""

import json
from pathlib import Path


def save_comparison(base_result, tuned_result, report_dir: Path):
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "metric": "normalized_sql_exact_match",
        "base": {k: base_result[k] for k in ("correct", "total", "exact_match")},
        "fine_tuned": {
            k: tuned_result[k] for k in ("correct", "total", "exact_match")
        },
        "improvement_percentage_points": 100
        * (tuned_result["exact_match"] - base_result["exact_match"]),
    }
    (report_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (report_dir / "predictions.json").write_text(
        json.dumps(tuned_result["predictions"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def print_failures(tuned_result, count=3):
    failures = [row for row in tuned_result["predictions"] if not row["exact_match"]]
    for number, failure in enumerate(failures[:count], 1):
        print(f"\nFailure {number} (test row {failure['test_index']})")
        print("Question:", failure["question"])
        print("Prediction:", failure["prediction"])
        print("Reference:", failure["reference"])
        print(
            "Review whether the error is a literal database-value mismatch, "
            "wrong table/column, or incorrect filter/join/aggregation."
        )
    if not failures:
        print("No exact-match failures on this test run.")


def print_memory_estimate(parameters=7_000_000_000):
    full_gb = parameters * 16 / 1e9
    lora_base_gb = parameters * 2 / 1e9
    qlora_base_gb = parameters * 0.5 / 1e9
    print("Approximate 7B model-state memory (decimal GB):")
    print(f"Full Adam fine-tuning: 7B × 16 bytes = {full_gb:.1f} GB, plus activations/workspace")
    print(f"LoRA: 7B × 2 bytes = {lora_base_gb:.1f} GB frozen base, plus adapters/states/activations")
    print(f"QLoRA: 7B × 0.5 bytes = {qlora_base_gb:.1f} GB quantized base, plus metadata/adapters/activations")
    print("Rule-of-thumb peak ranges: full 100–120+ GB; LoRA 20–30+ GB; QLoRA 8–16+ GB.")
