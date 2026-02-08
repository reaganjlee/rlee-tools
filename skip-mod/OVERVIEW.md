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

**Problem:** When all limits are 0, the original code computed an empty `mm_max_toks_per_item`, got (0, 0) from `compute_mm_encoder_budget()`, then patched it up with a fallback hack using scheduler defaults. With zero cache budget (before the hack), the scheduler's `EncoderCacheManager.can_allocate()` always returns `False` — no free slots — and generation hangs indefinitely.

**Fix:** Refactored to classify modalities into two categories upfront, eliminating the fallback hack:

- **`tower_modalities`**: limit > 0, pass through the MM encoder tower
- **`embed_only_modalities`**: `enable_mm_embeds=True` and limit == 0, bypass the tower (pre-computed embeddings only)

The encoder budget is computed from the union (`active_modalities = tower | embed_only`), so `compute_mm_encoder_budget()` receives a non-empty dict and produces a proper non-zero budget directly. Per-prompt/per-batch limits and `mm_max_toks_per_item` are derived from tower-only modalities (used by profiler/scheduler).

```python
# Modalities that pass through the MM encoder tower
tower_modalities = {
    modality for modality in supported_mm_limits
    if mm_limits.get(modality, 0) > 0
}
# Modalities that bypass the tower (pre-computed embeddings only)
embed_only_modalities = {
    modality for modality in supported_mm_limits
    if enable_mm_embeds and mm_limits.get(modality, 0) == 0
}

active_modalities = tower_modalities | embed_only_modalities

# Encoder budget computed from ALL active modalities
encoder_compute_budget, encoder_cache_size = compute_mm_encoder_budget(
    scheduler_config, active_mm_max_toks_per_item,
)

# Per-prompt/per-batch limits from tower-only
self.mm_max_toks_per_item = tower_mm_max_toks_per_item
```

This is architecturally cleaner — the two concerns (cache space vs encoder compute) are explicitly separated, and the budget is computed properly from the start.

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

### Issue 5: Crash on Empty Tower Modalities (`gpu_model_runner.py`)

**File:** `vllm/v1/worker/gpu_model_runner.py` → `profile_run()` and `_dummy_mm_kwargs()`

**Problem:** When all modalities are embed-only, `mm_max_toks_per_item` (the tower-only dict) is empty but encoder budget is non-zero. Any code that calls `get_modality_with_max_tokens()` crashes with `max() arg is an empty sequence`.

**Fix:** Guard empty `mm_max_toks_per_item` in both `profile_run()` (already guarded) and `_dummy_mm_kwargs()`:

```python
def _dummy_mm_kwargs(self, num_seqs: int) -> BatchedTensorInputs:
    ...
    if not mm_budget.mm_max_toks_per_item:
        return {}  # No tower modalities (embed-only mode)

    dummy_modality = mm_budget.get_modality_with_max_tokens()
    return self._get_mm_dummy_batch(dummy_modality, num_seqs)
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

# Disabled modality logic tests (22 tests)
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

## Why Tower vs Embed-Only Classification (Not a Post-Hoc Fallback)

The original approach computed an empty `mm_max_toks_per_item`, got (0, 0) from `compute_mm_encoder_budget()`, then patched it up with a fallback hack. A reviewer pointed out this was architecturally unclean — the budget should be computed properly from the start.

The refactored approach classifies modalities upfront into **tower** (limit > 0, need encoder compute) and **embed-only** (limit == 0 with `enable_mm_embeds`, need cache space only). The encoder budget is computed from the union of both, so it gets a non-empty dict and produces proper values directly. Meanwhile, `mm_max_toks_per_item` (stored as the tower-only dict) remains empty in embed-only mode, correctly preventing the profiler from trying to create dummy inputs for disabled modalities.

This explicitly separates the two concerns:
- **Encoder cache space**: computed from all active modalities (tower + embed-only)
- **Encoder compute / profiling**: computed from tower modalities only

### Issue 7: Profiling Validation Rejects Dummy Inputs for Embed-Only Modalities

**Files:** `vllm/multimodal/processing/dummy_inputs.py`, `vllm/multimodal/processing/processor.py`

**Problem:** The Issue 2 fix adds `embed_only_modalities` (limit=0, embeds enabled) to `active_modalities`, which causes `get_mm_max_toks_per_item` to generate dummy inputs for them during budget computation. These dummy inputs are regular PIL images (not embeddings), so when they flow through `parse_mm_data`, the Issue 3 validation logic sees "1 image submitted, but limit is 0" and rejects it. The model fails to initialize.

This wasn't caught earlier because upstream only profiles modalities with limit > 0 — the dummy input path never runs for limit=0 modalities.

**Why profiling is needed:** Even though the encoder won't run for embed-only modalities, the profiler needs to determine the token count (e.g., 576 for LLaVA images) to size the encoder cache that stores pre-computed embeddings at runtime.

**Fix:** Pass `validate=False` to `parse_mm_data` at the two internal call sites that process dummy/re-parsed data:

1. `dummy_inputs.py:98` — generates synthetic dummy data for profiling during init
2. `processor.py:1398` — in `_get_cache_missing_items`, re-parses already-validated items during cache lookup

User-facing validation is unaffected. The entry points for actual user requests (`input_processor.py:243` and `preprocess.py:211`) still call `parse_mm_data` with the default `validate=True`.

```python
# dummy_inputs.py — profiling path
dummy_mm_items = self.info.parse_mm_data(dummy_mm_data, validate=False)

# processor.py — cache miss re-parse path
mm_missing_items = self.info.parse_mm_data(mm_missing_data, validate=False)
```

**Alternative approaches considered:**

- **Thread `validate` from budget layer:** Add a `validate` parameter through the full call chain (`budget.py` → `get_mm_max_toks_per_item` → `get_dummy_mm_inputs` → `get_dummy_processor_inputs` → `parse_mm_data`, and also through `apply` → `_cached_apply_hf_processor` → `_get_cache_missing_items` → `parse_mm_data`). Only `budget.py` passes `False` for embed-only modalities. More explicit about intent originating at the budget layer, but touches 7 functions across 5 files.
- **Avoid profiling embed-only modalities entirely:** Compute the token count without dummy inputs (e.g., from model config). LLaVA doesn't implement the fast path (`get_mm_max_tokens_per_item` returns `None`), so the dummy input pipeline is the only generic mechanism available.

See `PROFILING_VALIDATION_ISSUE.md` for full details on the alternatives.

## Files Modified (Updated)

| File | Issue | Description |
|------|-------|-------------|
| `vllm/multimodal/registry.py` | 1 | Keep MM infrastructure alive when embeds enabled |
| `vllm/multimodal/budget.py` | 2 | Tower vs embed-only modality classification for encoder budget |
| `vllm/multimodal/processing/context.py` | 3 | Validation logic: skip for embeddings with limit=0 |
| `vllm/entrypoints/chat_utils.py` | 4 | Chat API: skip validation for `_embeds` with limit=0 |
| `vllm/v1/worker/gpu_model_runner.py` | 5 | Guard empty tower modalities in profile_run and _dummy_mm_kwargs |
| `vllm/v1/core/sched/scheduler.py` | 6 | Guard empty dict in modality lookup |
| `vllm/multimodal/processing/dummy_inputs.py` | 7 | Skip validation for dummy profiling inputs |
| `vllm/multimodal/processing/processor.py` | 7 | Skip validation for cache-miss re-parsed items |
| `vllm/config/multimodal.py` | — | Doc clarification |

## Test Files

| File | Type | Count | Description |
|------|------|-------|-------------|
| `test_embed_disabled_modality_unit.py` | pytest | 22 | Unit tests for all fix logic |
| `test_embedding_shape_validation_unit.py` | pytest | 18 | Shape validation unit tests |
| `test_embedding_disabled_modality.py` | standalone | 6 | Full integration with LLM |
| `test_embedding_shape_validation.py` | pytest | — | OpenAI endpoint shape tests |
| `test_embed_simple.py` | standalone | — | Simple embedding smoke test |
| `test_embed_v0.py` | standalone | — | Earlier test script |
| `test_mm_embeds.py` | pytest (in vllm repo) | 3 | E2E test: model init + generation under embed-only config |

The e2e test (`tests/entrypoints/llm/test_mm_embeds.py`) verifies the model launches and generates with `enable_mm_embeds=True, limit_mm_per_prompt={"image": 0}`. This is the test that caught Issue 7 — the model couldn't even initialize without the `validate=False` fix.
