# Multimodal Processor Benchmark (`vllm bench mm-processor`)

## Overview

The mm-processor benchmark measures the latency of vLLM's multimodal processor
pipeline on a single GPU instance. It instruments each stage of the MM
processing path—from HuggingFace processor calls through prompt updates and
encoder forward passes—and reports per-stage percentile metrics alongside
end-to-end request latency.

**PRs:** [#29105](https://github.com/vllm-project/vllm/pull/29105) (initial benchmark), [#31655](https://github.com/vllm-project/vllm/pull/31655) (encoder forward pass + HF dataset support), [#32646](https://github.com/vllm-project/vllm/pull/32646) (E2E latency fix + warmup)

**Resolves:** [#24171](https://github.com/vllm-project/vllm/issues/24171), [#25450](https://github.com/vllm-project/vllm/issues/25450)

## What It Measures

The benchmark collects timing stats for these stages:

| Stage | Description | Unit |
|-------|-------------|------|
| `hf_processor_time` | Time in HuggingFace processor calls (tokenization, image processing) | ms |
| `hashing_time` | Time computing multimodal item hashes (for cache keying) | ms |
| `cache_lookup_time` | Time in cache lookups and merges | ms |
| `prompt_update_time` | Time applying prompt updates and finding placeholder tokens | ms |
| `preprocessor_total_time` | Total preprocessing time (sum of above stages) | ms |
| `encoder_forward_time` | Vision/audio encoder forward pass on GPU | ms |
| `num_encoder_calls` | Number of encoder invocations per request | count |

Plus end-to-end latency (TTFT + decode time) per request.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    mm_processor.py                           │
│  benchmark_multimodal_processor()                           │
│    1. Load model via LLM(enable_mm_processor_stats=True)    │
│    2. get_requests() → dataset sampling                     │
│    3. Warmup requests (optional)                            │
│    4. llm.chat() → run inference                            │
│    5. collect_mm_processor_stats()                          │
│       ├── preprocessing stats from InputProcessingContext   │
│       └── encoder stats from workers via collective_rpc    │
│    6. calculate_mm_processor_metrics() → percentiles        │
│    7. Report table + optional JSON output                   │
└─────────────────────────────────────────────────────────────┘
```

### Stats Collection

**Preprocessing stats** are collected via `InputProcessingContext` in
`vllm/multimodal/processing/context.py`. Each stage is timed using the
`timed_preprocessor_operation()` context manager, which records elapsed time
into a per-request `MultiModalProcessorTimingStats` object stored in a
thread-safe registry keyed by `request_id`.

**Encoder stats** are collected worker-side in `gpu_model_runner.py`. The
`embed_multimodal` call is timed, and per-request `encoder_forward_time` and
`num_encoder_calls` are stored in an `EncoderTimingStats` registry. These are
retrieved via `collective_rpc("get_encoder_timing_stats")` and merged with
preprocessing stats by request ID.

**Aggregation:** When multiple workers report encoder stats, the benchmark
takes the `max` across workers (since encoder calls run in parallel across TP
ranks), not the sum.

### Request ID Matching

In the V1 engine, encoder-side request IDs have a suffix appended to the
original preprocessing request ID. The merge logic strips the suffix
(`request_id.rpartition("-")[0]`) to match encoder stats back to their
preprocessing counterparts.

## CLI Usage

### Basic: Synthetic Random MM Data

```bash
vllm bench mm-processor \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --dataset-name random-mm \
    --num-prompts 10
```

### With HuggingFace Datasets

```bash
vllm bench mm-processor \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --dataset-name hf \
    --dataset-path lmarena-ai/VisionArena-Chat \
    --hf-split train \
    --num-prompts 10
```

### With Warmup and Custom Percentiles

```bash
vllm bench mm-processor \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --dataset-name random-mm \
    --num-prompts 20 \
    --num-warmups 3 \
    --metric-percentiles 50,90,95,99
```

### Save Results to JSON

```bash
vllm bench mm-processor \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --dataset-name random-mm \
    --num-prompts 10 \
    --output-json results.json
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | (required) | Model name or path |
| `--dataset-name` | `random-mm` | Dataset type: `random-mm` or `hf` |
| `--dataset-path` | None | HF dataset path (required when `--dataset-name hf`) |
| `--hf-subset` | None | HF dataset subset |
| `--hf-split` | None | HF dataset split (e.g., `train`, `test`) |
| `--num-prompts` | 10 | Number of prompts to process |
| `--num-warmups` | 1 | Number of warmup requests (excluded from stats) |
| `--output-len` | None | Override output length per request |
| `--metric-percentiles` | `99` | Comma-separated percentiles to report |
| `--output-json` | None | Path to save JSON results |
| `--disable-tqdm` | false | Disable progress bar |

Plus all standard `EngineArgs` and random multimodal dataset args
(`--random-mm-base-items-per-request`, `--random-mm-bucket-config`, etc.).

## Supported Datasets

| Dataset | `--dataset-name` | `--dataset-path` |
|---------|-----------------|-----------------|
| Random synthetic MM | `random-mm` | (not needed) |
| VisionArena | `hf` | `lmarena-ai/VisionArena-Chat` |
| MMVU | `hf` | `yale-nlp/MMVU` |
| LLaVA-OneVision-Data | `hf` | `lmms-lab/LLaVA-OneVision-Data` |

## Example Output

```
Starting multimodal processor benchmark...
Processing 1 warmup requests...
Processing 10 requests...

================================================================================
Multimodal Processor Benchmark Results
================================================================================

MM Processor Metrics:
              Stage (ms)    Mean  Median    Std    P99
       hf_processor_time  142.31  138.50  12.40  168.20
            hashing_time    1.20    1.15   0.30    1.80
        cache_lookup_time    0.45    0.42   0.10    0.65
      prompt_update_time    8.30    7.90   1.50   11.20
  preprocessor_total_time  152.26  148.00  13.80  180.00
    encoder_forward_time  118.50  115.20  10.30  140.00
    num_encoder_calls       2.00    2.00   0.00    2.00

Summary: 20 total encoder calls across 10 requests.

End-to-End Latency (ms):
 Metric  Value (ms)
   Mean      850.30
 Median      830.50
    Std       45.20
    P99      920.10
```

## Design Decisions

1. **Single GPU scope:** The benchmark runs on a single GPU instance.
   Multi-GPU distributed benchmarking was deferred—running the benchmark
   multiple times across GPUs is equivalent for processor-level metrics.

2. **Stats auto-enabled:** `enable_mm_processor_stats` is set to `True` by
   default in the benchmark CLI, so no manual flag is needed.

3. **Warmup support:** Warmup requests use a different random seed
   (`seed + 1`) and are excluded from metric calculation by skipping the first
   `num_warmup_reqs` entries in the stats registry.

4. **Count vs. timing metrics:** `num_encoder_calls` is treated as a unitless
   count (not converted to milliseconds) in both calculation and display.

5. **E2E latency calculation:** Computed as `TTFT + decode_time` where
   `decode_time = last_token_ts - first_token_ts`. Falls back to
   `total_time / num_completed` if detailed metrics are unavailable.

## Key Files

- `vllm/benchmarks/mm_processor.py` — Main benchmark implementation
- `vllm/entrypoints/cli/benchmark/mm_processor.py` — CLI subcommand wrapper
- `vllm/multimodal/processing/context.py` — Timing stats, `InputProcessingContext`
- `docs/cli/bench/mm_processor.md` — Auto-generated CLI docs
