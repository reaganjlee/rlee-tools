# Fix Embedding Input Validation for Disabled Modalities

**Branch:** `skip-mod-2` in `reaganjlee/vllm`
**Base:** `fb1270f1f` (upstream main)
**Related PR:** #32493

## Problem Statement

When using `enable_mm_embeds=True` with `limit_mm_per_prompt={"image": 0}`, embeddings should work while raw images are rejected. This allows users to:
- Pass pre-computed image embeddings (bypassing the vision encoder)
- Save GPU memory by not loading the encoder (~0.58 GiB for LLaVA 1.5 7B)
- Disable raw image inputs for security/control

## Issues Discovered & Fixed

### Issue 1: Registry Text-Only Mode Blocks Embeddings (`registry.py`)

**File:** `vllm/multimodal/registry.py`

**Problem:** When all modality limits are 0, the registry enters text-only mode and disables all multimodal infrastructure. But with `enable_mm_embeds=True`, we still need that infrastructure to process pre-computed embeddings.

**Fix:** When `enable_mm_embeds=True`, skip the text-only short-circuit and return `True` to keep MM infrastructure alive.

```python
if all(mm_config.get_limit_per_prompt(modality) == 0 ...):
    # If enable_mm_embeds is True, we still need MM infrastructure
    # to process pre-computed embeddings even though encoder won't run
    if mm_config.enable_mm_embeds:
        return True
    logger.info_once("... running in text-only mode.")
    return False
```

### Issue 2: Encoder Cache Budget is (0, 0) (`budget.py`)

**File:** `vllm/multimodal/budget.py`

**Problem:** When all limits are 0, `mm_max_toks_per_item` is empty, so `compute_mm_encoder_budget()` returns (0, 0). With zero cache budget, the scheduler's `EncoderCacheManager.can_allocate()` always returns `False` — no free slots — and generation hangs indefinitely.

**Fix:** After `compute_mm_encoder_budget()`, fall back to scheduler defaults when `enable_mm_embeds=True` and budgets are (0, 0).

```python
mm_config = model_config.get_multimodal_config()
if (mm_config is not None and mm_config.enable_mm_embeds
        and encoder_compute_budget == 0
        and encoder_cache_size == 0):
    encoder_compute_budget = scheduler_config.max_num_encoder_input_tokens
    encoder_cache_size = scheduler_config.encoder_cache_size
```

**Note:** `max_num_encoder_input_tokens` and `encoder_cache_size` are derived from `max_num_batched_tokens` in `SchedulerConfig.__post_init__` — they are the standard defaults, not user-configurable values.

### Issue 3: Validation Logic Order (`context.py`)

**File:** `vllm/multimodal/processing/context.py` → `parse_mm_data()`

**Problem:** Upstream's validation had two separate loops: one checking for embedding inputs (raising error if `enable_mm_embeds=False`), then another validating all items against limits. This structure couldn't express "skip validation for embeddings when limit=0" — raw images with limit=0 would also need to be rejected.

**Fix:** Merge into a single loop that checks embedding type first, then applies the right logic:

```python
for modality, items in mm_items.items():
    if isinstance(items, (EmbeddingItems, DictEmbeddingItems)):
        if not mm_config.enable_mm_embeds:
            raise ValueError(f"You must set `--enable-mm-embeds` ...")
        if mm_config.get_limit_per_prompt(modality) == 0:
            logger.info("Skipping count validation for modality '%s' ...", modality)
            continue
    self.validate_num_items(modality, len(items))
```

### Issue 4: Chat API Item Tracker Validation (`chat_utils.py`)

**File:** `vllm/entrypoints/chat_utils.py` → `BaseMultiModalItemTracker.add()`

**Problem:** The chat API validates item counts as items are added. For embeddings with limit=0, this validation needs to be skipped. The check must verify the input is actually an embedding (via `_embeds` suffix) to avoid skipping validation for raw images.

**Fix:**
```python
mm_config = self.model_config.multimodal_config
if (
    mm_config is not None
    and mm_config.enable_mm_embeds
    and mm_config.get_limit_per_prompt(input_modality) == 0
    and original_modality.endswith("_embeds")
):
    pass  # Skip validation for embeddings with limit=0
else:
    self.mm_processor.info.validate_num_items(input_modality, num_items)
```

### Issue 5: Profiler Crash on Empty Modalities (`gpu_model_runner.py`)

**File:** `vllm/v1/worker/gpu_model_runner.py` → `profile_run()`

**Problem:** The Issue 2 fallback sets a non-zero `encoder_budget`, causing `profile_run()` to enter the profiling branch. But `mm_max_toks_per_item` is empty (no active modalities), so `get_modality_with_max_tokens()` calls `max()` on an empty dict → crash.

**Fix:** Check for empty `mm_max_toks_per_item` and skip profiling. There's no encoder to profile in embedding-only mode.

```python
if (encoder_budget := mm_budget.get_encoder_budget()) > 0:
    if not mm_budget.mm_max_toks_per_item:
        logger.info("Skipping encoder profiling for embedding-only mode ...")
    else:
        dummy_modality = mm_budget.get_modality_with_max_tokens()
        # ... existing profiling logic ...
```

### Issue 6: Scheduler Crash on Empty Modalities (`scheduler.py`)

**File:** `vllm/v1/core/sched/scheduler.py`

**Problem:** Same as Issue 5 but in the scheduler: `get_modality_with_max_tokens()` crashes on empty `mm_max_toks_per_item`.

**Fix:** Guard with empty dict check.

```python
self._num_encoder_max_input_tokens = (
    mm_budget.mm_max_toks_per_item[mm_budget.get_modality_with_max_tokens()]
    if mm_budget and mm_budget.mm_max_toks_per_item
    else 0
)
```

### Doc Update (`config/multimodal.py`)

Updated `enable_mm_embeds` docstring to clarify that only `limit=0` gets special treatment. Positive limits still apply to embeddings normally.

## Files Modified

| File | Issue | Description |
|------|-------|-------------|
| `vllm/multimodal/registry.py` | 1 | Keep MM infrastructure alive when embeds enabled |
| `vllm/multimodal/budget.py` | 2 | Encoder cache fallback for embedding-only mode |
| `vllm/multimodal/processing/context.py` | 3 | Validation logic: skip for embeddings with limit=0 |
| `vllm/entrypoints/chat_utils.py` | 4 | Chat API: skip validation for `_embeds` with limit=0 |
| `vllm/v1/worker/gpu_model_runner.py` | 5 | Skip encoder profiling when no modalities |
| `vllm/v1/core/sched/scheduler.py` | 6 | Guard empty dict in modality lookup |
| `vllm/config/multimodal.py` | — | Doc clarification |

## Expected Behaviors (All Verified)

| Config | Input Type | Expected Result |
|--------|-----------|-----------------|
| `enable_mm_embeds=True`, `limit=0` | Embedding | Works (skip validation) |
| `enable_mm_embeds=True`, `limit=0` | Raw image | `ValueError: At most 0 image(s)` |
| `enable_mm_embeds=True`, `limit=5` | Embedding | Validated against limit normally |
| `enable_mm_embeds=False`, `limit=0` | Embedding | `ValueError: must set --enable-mm-embeds` |
| `enable_mm_embeds=False`, `limit=0` | Raw image | `ValueError: At most 0 image(s)` |

## Testing

### Unit Tests (fast, no GPU needed)

```bash
source /workspace/vllm/.venv/bin/activate

# Disabled modality logic tests (17 tests)
pytest tests/multimodal/test_embed_disabled_modality_unit.py -v \
    --override-ini="confcutdir=tests/multimodal"

# Embedding shape validation tests (18 tests)
pytest tests/multimodal/test_embedding_shape_validation_unit.py -v \
    --override-ini="confcutdir=tests/multimodal"
```

### Integration Tests (requires GPU + llava-hf/llava-1.5-7b-hf)

```bash
source /workspace/vllm/.venv/bin/activate
python test_embedding_disabled_modality.py
# 6 passed, 0 failed, 0 errors
```

## Memory Verification

| Config | Memory | Notes |
|--------|--------|-------|
| `limit=0`, `enable_mm_embeds=True` | 12.55 GiB | Encoder not profiled |
| Default (no limit) | 13.13 GiB | Encoder loaded + profiled |
| **Savings** | **~0.58 GiB** | |

## Why the Encoder Cache is Needed for Pre-computed Embeddings

The "encoder cache" name is misleading — it's the **storage manager for all multimodal embeddings** during request processing, not just for caching encoder computation. Even pre-computed embeddings must be:
- **Stored** in GPU memory during generation
- **Tracked** by the scheduler so it knows when to feed them to the model
- **Freed** when the request finishes

Without a non-zero cache budget, the scheduler's `EncoderCacheManager.can_allocate()` always returns `False` (no free slots), so it never schedules the encoder inputs and the request hangs indefinitely.

## Why the Fallback is in `budget.py` (Not `get_mm_max_toks_per_item`)

A reviewer asked: why not modify `get_mm_max_toks_per_item` to return a synthetic entry instead of adding a budget fallback?

The problem is `mm_max_toks_per_item` flows into **profiling** (`gpu_model_runner.py`). If it contained `{"image": 576}`, the profiler would call `_get_mm_dummy_batch("image", N)` which creates dummy raw images and runs them through the encoder — hitting `ValueError: At most 0 image(s)`. The current approach correctly separates "we need cache space" (budget fallback) from "we have modalities to profile" (empty dict → skip profiling).

## Test Files

| File | Type | Count | Description |
|------|------|-------|-------------|
| `test_embed_disabled_modality_unit.py` | pytest | 17 | Unit tests for all fix logic |
| `test_embedding_shape_validation_unit.py` | pytest | 18 | Shape validation unit tests |
| `test_embedding_disabled_modality.py` | standalone | 6 | Full integration with LLM |
| `test_embedding_shape_validation.py` | pytest | — | OpenAI endpoint shape tests |
| `test_embed_simple.py` | standalone | — | Simple embedding smoke test |
| `test_embed_v0.py` | standalone | — | Earlier test script |
