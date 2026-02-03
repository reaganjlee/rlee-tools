# Encoder Cache Memory Leak Analysis

## Summary

Memory leak in vLLM multimodal encoder cache. Initial hypothesis about unit mismatch was **INCORRECT** - the suspected PR was merged after v0.11.1 release.

## Initial Hypothesis (INCORRECT)

**Commit:** f5f51e593 - "[Core][MM] Optimize encoder cache manager by operating with embeddings only (#30475)"
**Date:** December 2025 - **AFTER v0.11.1 release**

This commit cannot be the cause of the v0.11.0 → v0.11.1 regression.

## Previous Analysis (for reference, may still be relevant for future)

This PR changed the encoder cache manager to count in **embeddings** instead of **tokens**, but the cache **size configuration** still uses tokens.

### The Mismatch

**Before PR #30475:**
- Cache size: measured in tokens
- Cache accounting: measured in tokens
- Units match ✓

**After PR #30475:**
- Cache size: still measured in tokens (`scheduler_config.encoder_cache_size = max_num_batched_tokens`)
- Cache accounting: now measured in embeddings (`get_num_encoder_embeds()`)
- **Units mismatch ✗**

### Code Flow

1. `scheduler_config.encoder_cache_size` defaults to `max_num_batched_tokens` (e.g., 8192 tokens)

2. `compute_mm_encoder_budget()` returns:
   ```python
   encoder_cache_size = max(
       scheduler_config.encoder_cache_size,  # TOKENS (8192)
       max_tokens_per_mm_item                 # EMBEDDINGS (1200)
   )
   # Returns 8192 (the token value)
   ```

3. `EncoderCacheManager.__init__(cache_size=8192)` - thinks it has 8192 "slots"

4. For each image, `get_num_encoder_embeds()` returns ~100-200 embeddings

5. Each image uses ~100-200 slots out of 8192 → cache can hold 40-80 images

6. **But intended behavior was ~8 images** (if 8192 tokens / 1000 tokens per image)

### Impact

- Cache holds 5-10x more entries than intended
- Eviction rarely triggers (`can_allocate()` almost always has enough `num_free_slots`)
- Entries stay in `freeable` forever, never move to `freed`
- Worker's `encoder_cache` dict grows unboundedly
- Memory accumulates across benchmark runs

## Evidence

1. **Python heap stable** - tracemalloc shows ~27MB, no growth across 6 benchmarks
2. **System RAM grows ~17GB** - native tensor allocations accumulating
3. **Encoder cache is the only cache without hard limit on worker side** - relies on scheduler to send `free_encoder_mm_hashes`

## Affected Files

- `vllm/v1/core/encoder_cache_manager.py` - cache manager counting in embeddings
- `vllm/config/scheduler.py:227-228` - cache size defaults to tokens
- `vllm/multimodal/profiling.py` - `get_mm_max_tokens()` now returns embeddings

## Proposed Fix

**Option A: Update cache size default to embeddings**
- Change `scheduler_config.encoder_cache_size` to be computed in embeddings
- Requires updating default calculation

**Option B: Convert tokens to embeddings in compute_mm_encoder_budget**
- Add conversion factor based on typical token-to-embedding ratio
- More complex, model-dependent

**Option C: Keep manager counting in tokens (revert part of PR)**
- Revert `get_num_encoder_embeds` back to `get_num_encoder_tokens`
- Loses the memory optimization from PR #30475

## Recommended Fix

Option A is cleanest. The PR intended to reduce memory by operating in embeddings, but missed updating the default cache size configuration.

The fix should ensure `encoder_cache_size` is computed in the same units (embeddings) as the manager uses.

---

## Current Investigation Status

**Date:** 2026-02-03

The above analysis was based on commit f5f51e593 which is from December 2025. This is AFTER the v0.11.1 release, so it cannot be the cause of the regression between v0.11.0 and v0.11.1.

### What We Know
1. Python heap is stable (~27MB) across benchmark runs
2. System RAM grows (~17GB) - leak is in native/tensor allocations
3. All commits in bisect_results.csv show "good" status

### Questions to Resolve
1. What are the exact commit hashes for v0.11.0 and v0.11.1?
2. Is there a target_commits.csv with the commit range to bisect?
3. What metric distinguishes a "leaking" commit from a "good" one?
4. Is main currently exhibiting the leak?

### Next Steps
1. Get clarification on version commits
2. Identify a commit that clearly shows the leak
3. Git bisect between known good and bad commits
4. Re-analyze the actual regression commit
