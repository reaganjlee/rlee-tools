# KV Cache Memory Leak Investigation

## Problem

Memory regression in vLLM between v0.11.0 (good) and v0.11.1 (bad). Memory accumulates across consecutive benchmark runs until OOM.

## Suspected Files

- `vllm/v1/core/block_pool.py`
- `vllm/v1/core/kv_cache_utils.py`
- `vllm/v1/core/encoder_cache_manager.py`
- `vllm/v1/request.py`

## Tools

- **bisect_verify.py** - Automates git bisect to find the regression commit. Runs 6 consecutive benchmarks per commit, tracking memory at each step.
- **tracemalloc_profile.py** - Memory profiling using Python's tracemalloc. Takes snapshots before/after each benchmark run to identify growing allocations.
- **tracemalloc_hook/** - Auto-loading hook that adds `/debug/snapshot` endpoint to the vLLM server for remote snapshot triggering.

## Test Workload

- **Model**: Qwen2.5-VL-3B-Instruct
- **Dataset**: lmarena-ai/VisionArena-Chat (1000 prompts)
- **Success**: 6 consecutive runs without memory errors

## Usage

```bash
# Activate vLLM environment
activate-vllm

# Run git bisect verification
python bisect_verify.py

# Run memory profiling
python tracemalloc_profile.py
```

## Build Notes

- `VLLM_USE_PRECOMPILED=1` only works for Python-only changes
- For C++/CUDA changes, must build from source (unset the flag or set to 0)

## Output

- `bisect_results.csv` - Results from bisect runs (in /workspace/vllm/)
- `snapshots/*.pickle` - Tracemalloc memory snapshots
- `logs/` - Per-commit logs from bisect runs (in /workspace/vllm/)
