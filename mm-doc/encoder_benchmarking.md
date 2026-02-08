# Encoder Forward Pass Benchmarking

## Overview

PR [#31655](https://github.com/vllm-project/vllm/pull/31655) extended the
mm-processor benchmark to measure vision encoder latency alongside the existing
preprocessing metrics. This captures the GPU-side cost of encoding multimodal
inputs (images, video frames) through the model's vision encoder.

**Resolves:** [#25450](https://github.com/vllm-project/vllm/issues/25450)

## What It Added

Before this PR, the mm-processor benchmark only measured CPU-side preprocessing
(HF processor, hashing, cache, prompt updates). After this PR, the benchmark
also captures:

- **`encoder_forward_time`** — Wall-clock time of the `embed_multimodal` call
  on GPU (seconds, displayed as ms)
- **`num_encoder_calls`** — Number of encoder invocations per request (unitless
  count)

## Implementation Details

### Worker-Side Instrumentation

Encoder timing is captured in `gpu_model_runner.py`:

```
embed_multimodal() call
  ├── start = time.perf_counter()
  ├── ... encoder forward pass ...
  └── elapsed = time.perf_counter() - start
      → stored in EncoderTimingStats registry
```

Each worker maintains a thread-safe `EncoderTimingStats` registry keyed by
`request_id`. The registry stores:
- `encoder_forward_time`: cumulative encoder time for the request
- `num_encoder_calls`: number of encoder invocations

The stats are exposed through `gpu_worker.py` via the
`get_encoder_timing_stats()` method.

### Stats Retrieval and Merging

The benchmark collects stats through `get_timing_stats_from_engine_client()`
in `vllm/multimodal/processing/context.py`:

```python
# 1. Get preprocessing stats from InputProcessingContext
preprocessing_stats = ctx.get_all_timing_stats()

# 2. Get encoder stats from all workers
encoder_stats = engine_client.collective_rpc("get_encoder_timing_stats")

# 3. Merge by request_id
merged_stats[request_id] = {**preprocessing_dict, **encoder_dict}
```

### Multi-Worker Aggregation

When running with tensor parallelism (TP > 1), multiple workers report
encoder stats for the same request. The aggregation uses `max` (not `sum`)
because encoder forward passes run in parallel across TP ranks:

```python
# Aggregate timing metrics across workers
encoder_stats[request_id]["encoder_forward_time"] = max(
    current_time, new_time
)
encoder_stats[request_id]["num_encoder_calls"] = max(
    current_calls, new_calls
)
```

### V1 Engine Request ID Matching

In the V1 engine, encoder-side request IDs have a suffix appended (e.g.,
`request-123-0`). The merge logic handles this by stripping the suffix:

```python
possible_original_id = request_id.rpartition("-")[0]
if possible_original_id in merged_stats:
    merged_stats[possible_original_id].update(enc_dict)
```

## Metric Renaming

This PR also renamed `total_time` → `preprocessor_total_time` to avoid
confusion between preprocessing total time and end-to-end request latency.

## HF Dataset Support

This PR added support for HuggingFace datasets in the mm-processor benchmark,
enabling benchmarking with real-world multimodal data:

```bash
vllm bench mm-processor \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --dataset-name hf \
    --dataset-path lmarena-ai/VisionArena-Chat \
    --hf-split train \
    --num-prompts 10
```

Supported HF datasets are the same as those supported by
`MultiModalConversationDataset` and `VisionArenaDataset`:
- `lmarena-ai/VisionArena-Chat`
- `yale-nlp/MMVU`
- `lmms-lab/LLaVA-OneVision-Data`

Validation ensures that only supported multimodal datasets are used:

```python
supported_mm_datasets = (
    VisionArenaDataset.SUPPORTED_DATASET_PATHS.keys()
    | MultiModalConversationDataset.SUPPORTED_DATASET_PATHS
)
```

## Encoder Summary Output

The benchmark reports an encoder summary line:

```
Summary: 20 total encoder calls across 10 requests.
```

This aggregates `num_encoder_calls` across all (non-warmup) requests to give
a quick overview of encoder workload.

## Key Files Modified

- `vllm/benchmarks/mm_processor.py` — Added encoder stat collection and display
- `vllm/multimodal/processing/context.py` — `get_timing_stats_from_engine_client()` merges preprocessing + encoder stats
- `vllm/v1/worker/gpu_model_runner.py` — `EncoderTimingStats`, timing around `embed_multimodal`
- `vllm/v1/worker/gpu_worker.py` — Exposes `get_encoder_timing_stats()` for RPC
