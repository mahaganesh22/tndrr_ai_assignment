"""Shared benchmark settings and evaluation prompts."""
from pathlib import Path

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
TEST_PROMPTS = [
    "Explain what a REST API is in simple terms.",
    "What is the difference between SQL and NoSQL databases?",
    "Explain how a hash table works and its average time complexity.",
    "Write a Python function to check whether a number is prime.",
    "Explain the difference between supervised and unsupervised learning.",
    "What is the purpose of the KV cache in large language models?",
    "Explain how indexing improves database query performance.",
    "What happens when a user enters a URL into a browser?",
    "Explain the difference between concurrency and parallelism.",
    "Design a simple architecture for a URL shortening service.",
]
MAX_NEW_TOKENS = 200
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
