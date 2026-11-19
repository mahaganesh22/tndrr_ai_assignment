"""Generation timing and GPU memory measurements."""
import time
import torch
import pandas as pd
from .config import TEST_PROMPTS, MAX_NEW_TOKENS

_GIB = 1024 ** 3


def benchmark_model(model, tokenizer, quantization: str, prompts=None):
    """Measure generation throughput and peak allocated/reserved CUDA memory."""
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this benchmark.")
    prompts = TEST_PROMPTS if prompts is None else prompts
    torch.cuda.synchronize()
    model_baseline = torch.cuda.memory_allocated() / _GIB
    results = []

    for prompt_id, prompt in enumerate(prompts, start=1):
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        input_tokens = inputs["input_ids"].shape[1]

        # Tokenization is excluded; count generation latency only.
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        duration = time.perf_counter() - start

        generated_ids = outputs[0][input_tokens:]
        generated_count = generated_ids.numel()
        peak_allocated = torch.cuda.max_memory_allocated() / _GIB
        peak_reserved = torch.cuda.max_memory_reserved() / _GIB
        results.append({
            "Configuration": quantization,
            "Prompt_ID": prompt_id,
            "Prompt": prompt,
            "Response": tokenizer.decode(generated_ids, skip_special_tokens=True),
            "Input_Tokens": input_tokens,
            "Generated_Tokens": generated_count,
            "Generation_Time_Seconds": duration,
            "Tokens_Per_Second": generated_count / duration if duration else 0,
            "Model_Baseline_Allocated_GB": model_baseline,
            "Peak_GPU_Allocated_GB": peak_allocated,
            "Peak_GPU_Allocated_Additional_GB": max(0, peak_allocated - model_baseline),
            "Peak_GPU_Reserved_GB": peak_reserved,
        })
        print(f"{quantization} prompt {prompt_id}/{len(prompts)}: "
              f"{generated_count / duration:.2f} tokens/s")
        del inputs, outputs, generated_ids
    return pd.DataFrame(results)


def summarize_results(results: pd.DataFrame) -> dict:
    """Return the aggregate metrics used in the submission report."""
    return {
        "Average Tokens/Second": results["Tokens_Per_Second"].mean(),
        "Average Generation Time": results["Generation_Time_Seconds"].mean(),
        "Average Generated Tokens": results["Generated_Tokens"].mean(),
        "Peak GPU Allocated (GB)": results["Peak_GPU_Allocated_GB"].max(),
        "Peak GPU Reserved (GB)": results["Peak_GPU_Reserved_GB"].max(),
    }


def compare_summaries(fp16_results: pd.DataFrame, int4_results: pd.DataFrame):
    """Build metric summary and calculate relative memory/speed changes."""
    summary = pd.DataFrame({
        "FP16": summarize_results(fp16_results),
        "4-bit": summarize_results(int4_results),
    }).round(3)
    fp16_memory = summary.loc["Peak GPU Allocated (GB)", "FP16"]
    int4_memory = summary.loc["Peak GPU Allocated (GB)", "4-bit"]
    fp16_speed = summary.loc["Average Tokens/Second", "FP16"]
    int4_speed = summary.loc["Average Tokens/Second", "4-bit"]
    memory_saving = (fp16_memory - int4_memory) / fp16_memory * 100
    speed_difference = (int4_speed - fp16_speed) / fp16_speed * 100
    return summary, memory_saving, speed_difference
