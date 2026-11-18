"""Command-line pipeline for the Task 1 SQL QLoRA experiment."""

import gc

import torch
from transformers import set_seed

from .config import Config
from .data import format_splits, load_splits
from .evaluation import evaluate
from .modeling import load_adapter_model, load_tokenizer, load_training_model
from .reporting import print_failures, print_memory_estimate, save_comparison
from .training import train


def main():
    config = Config()
    set_seed(config.seed)
    print("Loading and splitting dataset...")
    splits = load_splits(config)
    print({name: len(split) for name, split in splits.items()})
    tokenizer = load_tokenizer(config)
    formatted = format_splits(splits, tokenizer)

    model, dtype = load_training_model(config)
    trainer, train_result = train(config, model, tokenizer, formatted)
    print("Training metrics:", train_result.metrics)

    # Release the trained model before loading each evaluation copy on GPU.
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    base_model = load_training_model_base(config, dtype)
    base_result = evaluate(
        base_model, tokenizer, splits["test"], config.max_new_tokens
    )
    print(f"Base: {base_result['correct']}/{base_result['total']} = {base_result['exact_match']:.1%}")
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    tuned_model = load_adapter_model(config, dtype)
    tuned_result = evaluate(
        tuned_model, tokenizer, splits["test"], config.max_new_tokens
    )
    print(f"Fine-tuned: {tuned_result['correct']}/{tuned_result['total']} = {tuned_result['exact_match']:.1%}")
    summary = save_comparison(base_result, tuned_result, config.report_dir)
    print("Comparison:", summary)
    print_failures(tuned_result)
    print_memory_estimate()


def load_training_model_base(config, dtype):
    """Load an unadapted base for a fair comparison with the tuned model."""
    from .modeling import load_base_model

    model = load_base_model(config, dtype)
    model.config.use_cache = True
    return model


if __name__ == "__main__":
    main()
