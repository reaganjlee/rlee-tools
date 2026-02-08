# Dataset Overview Table Update

## Current State (Incorrect)

In `docs/benchmarking/cli.md`, the dataset overview table shows:

```
| RandomMultiModal (Image/Video) | 🟡 | 🚧 | `synthetic` |
```

- 🟡 (Partial) for Online
- 🚧 (To be supported) for Offline

## What Changed

`RandomMultiModalDataset` is now fully supported in both online and offline
benchmarking:

**Online** (`vllm bench serve`):
- Supported via `--dataset-name random-mm` with `--backend openai-chat`
- Image and video sampling both work
- Was already partial (🟡) — now fully working (✅)

**Offline** (`vllm bench throughput`):
- `get_requests()` in `throughput.py` now handles `random-mm` (line ~420)
- Uses `vllm-chat` backend with `enable_multimodal_chat=True`
- All random-mm args are supported (`--random-mm-base-items-per-request`, etc.)
- Was 🚧 — now fully working (✅)

**Offline** (`vllm bench mm-processor`):
- `random-mm` is the *default* dataset for the mm-processor benchmark
- Uses `get_requests()` from `throughput.py` under the hood

## Proposed Change

In `docs/benchmarking/cli.md`, update line 23:

**Before:**
```markdown
| RandomMultiModal (Image/Video) | 🟡 | 🚧 | `synthetic` |
```

**After:**
```markdown
| RandomMultiModal (Image/Video) | ✅ | ✅ | `synthetic` |
```

## File to Edit

`/workspace/vllm-mm-doc/docs/benchmarking/cli.md` (line 23)

## New Offline Example to Add

Add this under the "Offline Throughput Benchmark" section, after the existing
VisionArena offline example:

```markdown
#### Synthetic Random Multimodal (random-mm)

Benchmark offline throughput with synthetic multimodal inputs (images/video):

```bash
vllm bench throughput \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --backend vllm-chat \
  --dataset-name random-mm \
  --num-prompts 100 \
  --random-input-len 300 \
  --random-output-len 40 \
  --random-mm-base-items-per-request 2 \
  --random-mm-limit-mm-per-prompt '{"image": 3, "video": 0}' \
  --random-mm-bucket-config '{(256, 256, 1): 0.7, (720, 1280, 1): 0.3}'
```

This generates synthetic RGB images and attaches them to random text prompts,
useful for stress-testing vision model throughput without downloading external
datasets.
```

## MM-Processor Benchmark Section to Add

A new section should be added under the benchmarking docs (after Offline
Throughput or as a sibling section):

```markdown
### 🔬 Multimodal Processor Benchmark

<details class="admonition abstract" markdown="1">
<summary>Show more</summary>

Benchmark the latency of vLLM's multimodal processor pipeline, measuring
per-stage timing (HF processor, hashing, cache lookup, prompt update,
encoder forward pass) and end-to-end request latency.

#### Basic Usage with Synthetic Data

```bash
vllm bench mm-processor \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --dataset-name random-mm \
    --num-prompts 10
```

#### With HuggingFace Dataset

```bash
vllm bench mm-processor \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --dataset-name hf \
    --dataset-path lmarena-ai/VisionArena-Chat \
    --hf-split train \
    --num-prompts 10
```

#### With Warmup and Custom Percentiles

```bash
vllm bench mm-processor \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --dataset-name random-mm \
    --num-prompts 20 \
    --num-warmups 3 \
    --metric-percentiles 50,90,95,99 \
    --output-json results.json
```

The benchmark reports per-stage metrics (mean, median, std, percentiles) for:
- HuggingFace processor time
- Multimodal item hashing time
- Cache lookup time
- Prompt update time
- Total preprocessing time
- Encoder forward pass time
- Number of encoder calls per request
- End-to-end request latency

</details>
```
