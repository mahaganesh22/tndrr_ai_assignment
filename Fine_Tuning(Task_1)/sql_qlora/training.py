"""TRL SFT trainer creation and adapter training."""

import inspect

import torch
from trl import SFTConfig, SFTTrainer

from .config import Config
from .modeling import cast_trainable_parameters_to_fp32


def _supported_kwargs(callable_obj, kwargs):
    parameters = inspect.signature(callable_obj).parameters
    return {k: v for k, v in kwargs.items() if k in parameters}


def make_trainer(config: Config, model, tokenizer, formatted_splits):
    training_kwargs = {
        "output_dir": str(config.output_dir),
        "num_train_epochs": config.epochs,
        "per_device_train_batch_size": config.train_batch_size,
        "per_device_eval_batch_size": config.eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "weight_decay": 0.01,
        "warmup_steps": 0,
        "lr_scheduler_type": "cosine",
        "logging_steps": 25,
        "eval_strategy": "epoch",
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        # Disable AMP/GradScaler for reliable T4 training. The frozen quantized
        # backbone still uses the compute dtype in BitsAndBytesConfig.
        "fp16": False,
        "bf16": False,
        "optim": "paged_adamw_8bit",
        "gradient_checkpointing": True,
        "max_length": config.max_sequence_length,
        "packing": False,
        "dataset_text_field": "text",
        "loss_type": "nll",
        "report_to": "none",
        "seed": config.seed,
    }
    args = SFTConfig(**_supported_kwargs(SFTConfig, training_kwargs))

    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": formatted_splits["train"],
        "eval_dataset": formatted_splits["validation"],
        "processing_class": tokenizer,
        # Older TRL versions called this argument tokenizer.
        "tokenizer": tokenizer,
    }
    trainer = SFTTrainer(**_supported_kwargs(SFTTrainer, trainer_kwargs))

    # Trainer/PEFT initialization may recast adapter parameters. Restore FP32
    # before Trainer lazily creates the optimizer.
    cast_trainable_parameters_to_fp32(trainer.model)
    if getattr(trainer.args, "fp16", False) or getattr(trainer.args, "bf16", False):
        raise RuntimeError("Mixed precision must be disabled for this T4-safe setup.")
    trainable_dtypes = {
        parameter.dtype
        for parameter in trainer.model.parameters()
        if parameter.requires_grad
    }
    print("Trainer AMP: disabled; trainable dtypes:", trainable_dtypes)
    if trainable_dtypes != {torch.float32}:
        raise RuntimeError(f"Expected FP32 adapter parameters, found {trainable_dtypes}.")
    return trainer


def train(config: Config, model, tokenizer, formatted_splits):
    config.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = make_trainer(config, model, tokenizer, formatted_splits)
    result = trainer.train()
    trainer.save_model(str(config.adapter_dir))
    tokenizer.save_pretrained(str(config.adapter_dir))
    return trainer, result
