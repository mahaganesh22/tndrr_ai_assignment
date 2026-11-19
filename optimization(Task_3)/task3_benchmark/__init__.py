"""Task 3 model benchmarking helpers."""

from .config import MODEL_NAME, OUTPUT_DIR, TEST_PROMPTS
from .modeling import load_model, clear_cuda_cache
from .benchmarking import benchmark_model, summarize_results, compare_summaries
from .artifacts import (
    build_quality_comparison,
    load_or_create_quality_scores,
    save_benchmark_artifacts,
    write_submission_report,
)

__all__ = [
    "MODEL_NAME", "OUTPUT_DIR", "TEST_PROMPTS", "load_model", "clear_cuda_cache",
    "benchmark_model", "summarize_results", "compare_summaries",
    "build_quality_comparison", "load_or_create_quality_scores",
    "save_benchmark_artifacts", "write_submission_report",
]
