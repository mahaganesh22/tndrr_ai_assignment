# Archived notebook outputs — Task 3

Source: `optimization.ipynb` output snapshot before its outputs were cleared during code edits.
Model: `Qwen/Qwen2.5-3B-Instruct`  |  GPU: Tesla T4  |  Prompts: 10

This archive preserves the outputs that were present in the notebook snapshot. The quality-comparison cell had only emitted a pandas Styler object in that snapshot; its ten answer pairs were not rendered in the captured output, so those response texts cannot be reconstructed from the snapshot.

## Environment and prompt setup

- `Environment initialized.`
- `Number of evaluation prompts: 10`

## FP16 run

- Model loaded successfully.
- Prompt 1/10 | Speed: 23.13 tokens/sec | Time: 8.65s
- Prompt 2/10 | Speed: 25.99 tokens/sec | Time: 7.70s
- Prompt 3/10 | Speed: 25.30 tokens/sec | Time: 7.90s
- Prompt 4/10 | Speed: 27.25 tokens/sec | Time: 7.34s
- Prompt 5/10 | Speed: 24.93 tokens/sec | Time: 8.02s
- Prompt 6/10 | Speed: 25.65 tokens/sec | Time: 7.80s
- Prompt 7/10 | Speed: 26.78 tokens/sec | Time: 7.47s
- Prompt 8/10 | Speed: 24.98 tokens/sec | Time: 8.01s
- Prompt 9/10 | Speed: 27.21 tokens/sec | Time: 7.35s
- Prompt 10/10 | Speed: 25.25 tokens/sec | Time: 7.92s
- FP16 Experiment Completed.

| Prompt | Generated tokens | Generation time (s) | Tokens/s | Peak allocated GPU (GB) |
|---:|---:|---:|---:|---:|
| 1 | 200 | 8.645996 | 23.132095 | 5.766710 |
| 2 | 200 | 7.695231 | 25.990124 | 5.766710 |
| 3 | 200 | 7.903953 | 25.303794 | 5.766778 |
| 4 | 200 | 7.339474 | 27.249911 | 5.766778 |
| 5 | 200 | 8.021914 | 24.931705 | 5.766778 |
| 6 | 200 | 7.795768 | 25.654945 | 5.766778 |
| 7 | 200 | 7.467165 | 26.783926 | 5.766778 |
| 8 | 200 | 8.005120 | 24.984009 | 5.766778 |
| 9 | 200 | 7.350207 | 27.210119 | 5.766778 |
| 10 | 200 | 7.920253 | 25.251719 | 5.766778 |

Console also reported: `FP16 model memory released.`

## 4-bit run

- Model loaded successfully.
- Prompt 1/10 | Speed: 12.42 tokens/sec | Time: 16.11s
- Prompt 2/10 | Speed: 12.61 tokens/sec | Time: 15.86s
- Prompt 3/10 | Speed: 12.31 tokens/sec | Time: 16.25s
- Prompt 4/10 | Speed: 12.79 tokens/sec | Time: 15.64s
- Prompt 5/10 | Speed: 12.72 tokens/sec | Time: 15.72s
- Prompt 6/10 | Speed: 12.37 tokens/sec | Time: 16.17s
- Prompt 7/10 | Speed: 12.43 tokens/sec | Time: 16.09s
- Prompt 8/10 | Speed: 12.66 tokens/sec | Time: 15.79s
- Prompt 9/10 | Speed: 12.76 tokens/sec | Time: 15.68s
- Prompt 10/10 | Speed: 12.31 tokens/sec | Time: 16.24s
- 4-bit Experiment Completed.

| Prompt | Generated tokens | Generation time (s) | Tokens/s | Peak allocated GPU (GB) |
|---:|---:|---:|---:|---:|
| 1 | 200 | 16.108349 | 12.415921 | 1.946425 |
| 2 | 200 | 15.855580 | 12.613856 | 1.946425 |
| 3 | 200 | 16.246180 | 12.310586 | 1.946544 |
| 4 | 200 | 15.643141 | 12.785157 | 1.946544 |
| 5 | 200 | 15.724796 | 12.718766 | 1.946544 |
| 6 | 200 | 16.171768 | 12.367231 | 1.946544 |
| 7 | 200 | 16.087187 | 12.432254 | 1.946544 |
| 8 | 200 | 15.791883 | 12.664734 | 1.946544 |
| 9 | 200 | 15.679916 | 12.755170 | 1.946544 |
| 10 | 200 | 16.244529 | 12.311837 | 1.946544 |

## Summary and comparison outputs

| Metric | FP16 | 4-bit |
|---|---:|---:|
| Average Tokens/Second | 25.649 | 12.538 |
| Average Generation Time | 7.815 s | 15.955 s |
| Average Generated Tokens | 200.000 | 200.000 |
| Peak GPU Allocated (GB) | 5.767 | 1.947 |
| Peak GPU Reserved (GB) | 5.832 | 5.820 |

- `GPU Memory Saving: 66.24%`
- `Speed Difference: -51.12%`
- `FP16 is faster in this experiment.`
- The notebook saved the inference-speed and GPU-memory comparison charts.
- The quality-comparison display output was `<pandas.io.formats.style.Styler ...>`; its underlying response pairs were not printed in the captured output.
- `Quality evaluation template created.`

## Previously generated report text

```text
TASK 3 - LOCAL MODEL RUNNING AND OPTIMIZATION
==============================================

MODEL
-----
Model: Qwen/Qwen2.5-3B-Instruct
GPU: Tesla T4
Evaluation Prompts: 10

EXPERIMENT 1: FP16
------------------
Average Tokens/Second: 25.649
Average Generation Time: 7.815 seconds
Peak GPU Allocated Memory: 5.767 GB
Peak GPU Reserved Memory: 5.832 GB

EXPERIMENT 2: 4-BIT
-------------------
Average Tokens/Second: 12.538
Average Generation Time: 15.955 seconds
Peak GPU Allocated Memory: 1.947 GB
Peak GPU Reserved Memory: 5.820 GB

PERFORMANCE ANALYSIS
--------------------
GPU Memory Savings: 66.24%
Speed Difference: -51.12%

OUTPUT QUALITY
--------------
10 identical prompts were used for both configurations.
Responses are available in quality_comparison.csv.
Manual quality evaluation is required.

KV CACHE
--------
The KV cache stores previously calculated key and value
vectors from the attention mechanism. It avoids recomputing
them for earlier tokens during autoregressive generation.

QUANTIZATION
------------
Quantization reduces the number of bits used to represent
model weights. This reduces model memory requirements.
4-bit weights require significantly less storage than FP16
weights, although actual memory usage also depends on
quantization metadata and runtime overhead.

CONCURRENT SERVING PLAN
-----------------------
Target: 20 concurrent users
GPU: One 24 GB GPU

Proposed architecture:
1. Use a 3B or smaller 4-bit quantized model.
2. Serve requests using vLLM with continuous batching.
3. Configure a maximum sequence length of 4096 tokens.
4. Limit maximum output length to 512 tokens.
5. Use a maximum batch size of 20, subject to GPU memory.
6. Monitor KV cache usage and GPU utilization.
7. Queue requests when GPU capacity is exhausted.

Actual concurrency depends on prompt lengths, output lengths,
KV cache requirements, and GPU compute throughput.

RESULT FILES
------------
fp16_results.csv
4bit_results.csv
performance_comparison.csv
quality_comparison.csv
quality_scores_template.csv
speed_comparison.png
memory_comparison.png
```

The original notebook's captured output did not include the CSV contents, the plots themselves, or the generated answer texts; this archive preserves the displayed outputs and report text available from that snapshot.
