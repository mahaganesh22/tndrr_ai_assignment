"""Project configuration and shared task prompt."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset_name: str = "b-mc2/sql-create-context"
    output_dir: Path = PROJECT_DIR / "outputs" / "qwen-sql-qlora"
    report_dir: Path = PROJECT_DIR / "outputs" / "reports"
    seed: int = 42
    train_size: int = 5000
    validation_size: int = 500
    test_size: int = 100
    max_sequence_length: int = 1024
    max_new_tokens: int = 256
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    learning_rate: float = 2e-4
    epochs: int = 3
    train_batch_size: int = 4
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4

    @property
    def adapter_dir(self) -> Path:
        return self.output_dir / "adapter"


SYSTEM_PROMPT = (
    "You are an expert SQL query generator. Given a database schema and a "
    "natural language question, generate the correct SQL query. Return only "
    "the SQL query, without explanation or markdown."
)


def user_prompt(context: str, question: str) -> str:
    return (
        f"Database Schema:\n{context}\n\nQuestion:\n{question}\n\n"
        "Generate the SQL query."
    )
