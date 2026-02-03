# KV Cache Memory Leak Investigation - Latest Summary

**Last Updated**: 2026-02-03
**Status**: INVESTIGATION IN PROGRESS - NEED CLARIFICATION

## Overview

Investigating a memory regression in vLLM between v0.11.0 (good) and v0.11.1 (bad). Memory accumulates across consecutive benchmark runs until OOM.

**Test Setup**:
- Model: Qwen2.5-VL-3B-Instruct
- Dataset: lmarena-ai/VisionArena-Chat (1000 prompts)
- Success criteria: 6 consecutive runs without memory errors

## Current Status

### Profiling Complete

Ran memray profiling on vLLM server with 2 consecutive benchmark runs:

| Metric | After Run 1 | After Run 2 | Delta |
|--------|-------------|-------------|-------|
| Total allocations | 99.6M | 142.7M | +43.1M |
| Total memory allocated | 41.2 GB | 58.5 GB | **+17.3 GB** |
| Unique allocation sites | 59,818 | 60,313 | +495 |

**Conclusion**: Memory leak confirmed - ~17 GB growth per benchmark run.

### Artifacts Generated

- `memray_bench1.bin` - Capture after 1st benchmark run
- `memray_bench2.bin` - Capture after 2nd benchmark run
- `memray-stats-memray_bench1.bin.json` - Stats in JSON format
- `memray-stats-memray_bench2.bin.json` - Stats in JSON format
- `table_bench1.html` / `table_bench2.html` - Detailed allocation tables (99MB each)
- `flamegraph_bench1.html` / `flamegraph_bench2.html` - Flame graphs

## Key Finding: Leak is in Native Code

**Python heap is stable** - tracemalloc snapshots show ~27MB with no growth across 6 benchmark runs.

**System RAM grows ~17GB** - this means the leak is in:
- Native C++/CUDA allocations
- PyTorch tensors held by Python objects that aren't releasing their underlying memory
- Memory-mapped regions

## Cache Architecture (Multimodal Path)

Three separate caches are involved:

### 1. MM Processor Cache (`mm_processor_cache_gb`)
- **Default**: 4 GB
- **Location**: `vllm/multimodal/cache.py`
- **Purpose**: Caches preprocessed multimodal inputs (before encoding)
- **Eviction**: LRU-based, evicts when size exceeds capacity

### 2. Encoder Cache (`encoder_cache_size`)
- **Default**: `max_num_batched_tokens`
- **Scheduler**: `vllm/v1/core/encoder_cache_manager.py` - tracks slots
- **Worker**: `vllm/v1/worker/gpu_model_runner.py:423` - stores actual tensors in `self.encoder_cache: dict[str, torch.Tensor]`
- **Eviction**: Only evicts when `can_allocate()` needs space AND cache is full
- **Potential Issue**: If cache has plenty of space, old entries stay in GPU memory forever

### 3. KV Cache Block Pool
- **Location**: `vllm/v1/core/block_pool.py`
- **Purpose**: Manages KV cache blocks for attention
- **Eviction**: Reference counting, freed when ref_cnt hits 0

## Potential Leak Causes

1. **Encoder cache not evicting** - entries go to `freeable` but only move to `freed` when new allocations need space
2. **Large MM processor cache** - 4GB default may hold old tensors
3. **Tensor references** - Python objects may hold refs to tensors preventing GC

## Next Steps

1. **Test with `--mm-processor-cache-gb 0`** - Disable MM processor cache to isolate
2. **Profile with native traces** - Run memray with `--native` flag
3. **Trace encoder cache flow** - Verify `free_encoder_mm_hashes` is being propagated correctly

## Suspected Files

Based on CLAUDE.md:
- `vllm/v1/core/block_pool.py`
- `vllm/v1/core/kv_cache_utils.py`
- `vllm/v1/core/encoder_cache_manager.py`
- `vllm/v1/request.py`

## Current Status

### What We've Confirmed
1. **Python heap is stable** - tracemalloc shows ~27MB, no growth across 6 benchmark runs
2. **Leak is in native allocations** - System RAM grows but Python memory doesn't
3. **Three caches involved**: MM Processor Cache (4GB), Encoder Cache, KV Block Pool

### What We Investigated (INCORRECT PATH)
- Initially suspected commit f5f51e593 (PR #30475) - "Optimize encoder cache manager by operating with embeddings only"
- **BUT**: This PR is dated December 2025, which is AFTER v0.11.1 release
- So this cannot be the cause of the v0.11.0 → v0.11.1 regression

### Bisect Results Analysis
- All commits in `bisect_results.csv` show `status=good` and `growth_rate_decreased=True`
- This means memory growth slows down (stabilizes) for all tested commits
- **No clear bad/good boundary found yet**

## Open Questions

1. **What are the actual commit hashes for v0.11.0 and v0.11.1?**
   - Need to identify the exact commit range to investigate

2. **Is there a `target_commits.csv` file?**
   - The bisect_verify.py script mentions using this file for commit filtering

3. **What specific behavior indicates the leak?**
   - Bisect results show `growth_rate_decreased=True` for all commits
   - What metric/behavior distinguishes good vs bad commits?

4. **What is the current HEAD?**
   - Is main currently showing the leak?
   - Or are we testing against a specific version?

5. **What's the expected vs actual memory behavior?**
   - How much RAM growth is acceptable?
   - At what point does it become "leaking"?

## Architecture Understanding (for reference)

### Encoder Cache Flow
1. `EncoderCacheManager` (scheduler) tracks which entries should exist
2. `gpu_model_runner.encoder_cache` (worker) stores actual tensors
3. Entries freed via `free_encoder_mm_hashes` in scheduler output
4. Eviction only happens when `can_allocate()` needs space

### Potential Leak Points (still valid to investigate)
- Mismatch between scheduler bookkeeping and worker's actual tensor storage
- Entries staying in `freeable` without being evicted to `freed`
- Reference cycles preventing tensor garbage collection

## Next Steps

1. **Clarify version commits** - Get exact hashes for v0.11.0 and v0.11.1
2. **Identify bad commit** - Find a commit that shows the leak behavior
3. **Git bisect between versions** - Once we have good/bad commits
4. **Instrument caches** - Add logging to track cache sizes if needed

## Commands Reference

```bash
# Run bisect verification
cd /workspace/vllm
python /workspace/rlee-tools/kv-bug/bisect_verify.py

# Run with specific commit
python /workspace/rlee-tools/kv-bug/bisect_verify.py --commit <hash>

# Test with MM cache disabled (isolate encoder cache)
python /workspace/rlee-tools/kv-bug/bisect_verify.py --disable-mm-cache

# Compare tracemalloc snapshots
python /workspace/rlee-tools/kv-bug/compare_snapshots.py <snap1> <snap2>
```
