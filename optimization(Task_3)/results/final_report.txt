# Task 3 ? Local model and optimization

## Setup

- Model: `Qwen/Qwen2.5-3B-Instruct`
- GPU: Tesla T4 (recorded by the prior run)
- Ten identical prompts; deterministic decoding; 200 generated tokens per prompt.

## Measured results

| Setting | Mean tokens/s | Mean generation time | Peak allocated GPU memory | Peak reserved GPU memory |
|---|---:|---:|---:|---:|
| FP16 | 25.649 | 7.815 s | 5.767 GB | 5.832 GB |
| 4-bit NF4 | 12.538 | 15.955 s | 1.947 GB | 5.820 GB |

The recorded results show 66.24% lower peak allocated memory for 4-bit and a 51.12% reduction in tokens/s compared with FP16 on this Tesla T4 run. Reserved memory is reported separately from allocated memory. Per-prompt values are in `fp16_results.csv` and `4bit_results.csv`.

## Output quality

Quality comparison is **not recoverable from the provided output file**: it does not contain the generated answer texts or ratings. The response fields in `quality_comparison.csv` are intentionally blank, and the score fields in `quality_scores_template.csv` are unfilled. Do not submit these as a completed quality assessment. If the answer texts are available elsewhere, paste them into the comparison file, rate correctness, relevance, and clarity from 1 to 5 for each response, and add observations.

## KV cache

The KV cache holds attention key and value vectors already computed for earlier tokens. Reusing them avoids repeating that computation during generation. It uses GPU memory and grows with context length and concurrent requests.

## Why quantization saves memory

Quantization represents model weights with fewer bits. Four-bit weights take less storage than 16-bit weights, though metadata, activations, KV cache, and runtime allocations also use memory.

## Plan for 20 concurrent users on one 24 GB GPU

Assume one 24 GB GPU, a 3B model in 4-bit, 20 active requests, a 4096-token context limit, and a 512-token output limit. Serve with continuous batching and a paged KV cache, monitor cache capacity, and queue requests when full. This is a capacity plan to validate with representative traffic; 20 active users does not guarantee 20 requests can decode at full speed.

## Files

- `fp16_results.csv`, `4bit_results.csv`: recovered per-prompt performance values. Response text and input-token counts were not in the provided output snapshot.
- `performance_comparison.csv`: summary measurements.
- `quality_comparison.csv`, `quality_scores_template.csv`: quality-review files; ratings cannot be completed without the model responses.
- `optimization_outputs.md`: archive of the earlier captured notebook output.
