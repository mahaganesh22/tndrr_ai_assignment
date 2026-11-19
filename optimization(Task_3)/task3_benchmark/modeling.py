"""Model loading and cleanup."""
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from .config import MODEL_NAME


def load_model(quantization: str):
    """Load MODEL_NAME in FP16 or bitsandbytes NF4 4-bit mode."""
    quantization = quantization.upper()
    if quantization not in {"FP16", "4BIT"}:
        raise ValueError("quantization must be 'FP16' or '4BIT'")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this benchmark.")

    print(f"Loading {MODEL_NAME} in {quantization} mode...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    common = {"device_map": "auto"}
    if quantization == "FP16":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.float16, **common
        )
    else:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=quant_config, **common
        )
    model.eval()
    return model, tokenizer


def clear_cuda_cache():
    """Collect Python garbage and clear cached CUDA allocations."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
