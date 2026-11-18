"""Quantized model and LoRA adapter construction."""

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import Config


def compute_dtype():
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required. Select a GPU runtime in Colab.")
    capability = torch.cuda.get_device_capability(0)
    dtype = torch.bfloat16 if capability >= (8, 0) else torch.float16
    print(
        f"GPU: {torch.cuda.get_device_name(0)}; "
        f"compute capability: {capability[0]}.{capability[1]}; dtype: {dtype}"
    )
    return dtype


def load_tokenizer(config: Config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def quantization_config(dtype):
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )


def load_base_model(config: Config, dtype=None):
    dtype = dtype or compute_dtype()
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=quantization_config(dtype),
        device_map="auto",
        torch_dtype=dtype,
    )
    model.config.use_cache = False
    return model


def load_training_model(config: Config):
    dtype = compute_dtype()
    model = prepare_model_for_kbit_training(load_base_model(config, dtype))
    adapter_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, adapter_config)
    cast_trainable_parameters_to_fp32(model)
    model.print_trainable_parameters()
    return model, dtype


def cast_trainable_parameters_to_fp32(model):
    """Avoid low-precision adapter gradients on older GPUs such as Colab T4."""
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()


def load_adapter_model(config: Config, dtype=None):
    from peft import PeftModel

    base = load_base_model(config, dtype)
    base.config.use_cache = True
    return PeftModel.from_pretrained(base, str(config.adapter_dir))
