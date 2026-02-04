# Fix Embedding Input Validation for Disabled Modalities

**Branch:** `fix-embedding-validation-limit-zero` in `reaganjlee/vllm`

**Related PR:** #32493

## Problem Statement

When using `enable_mm_embeds=True` with `limit_mm_per_prompt={"image": 0}`, embeddings should work while raw images are rejected. This allows users to:
- Pass pre-computed image embeddings (bypassing the vision encoder)
- Save GPU memory by not loading the encoder
- Disable raw image inputs for security/control

## Issues Discovered

### Issue 1: Profiler Creates Dummy Images (`registry.py`)

**Location:** `vllm/multimodal/registry.py:176-187`

**Problem:** The PR's change included limit=0 modalities for profiling when `enable_mm_embeds=True`. The profiler creates dummy images to measure token counts, but these fail validation ("At most 0 image(s)").

**Fix:** Exclude limit=0 modalities from profiling. Embeddings don't need profiling since dimensions are user-defined.

```python
# BEFORE (PR's buggy code)
if enable_mm_embeds:
    modality_counts = {modality: 1 for modality in profiler_limits.keys()}
else:
    modality_counts = {modality: 1 for modality, limit in profiler_limits.items() if limit > 0}

# AFTER (fixed)
return profiler.get_mm_max_tokens(
    seq_len,
    {modality: 1 for modality, limit in profiler_limits.items() if limit > 0},
)
```

### Issue 2: Parser Doesn't Recognize `image_embeds` Key (`parse.py`)

**Location:** `vllm/multimodal/parse.py:652-664`

**Problem:** `parse_mm_data()` only recognized "audio", "image", "video" as valid keys. When users pass `{"image_embeds": tensor}`, it failed with "Unsupported modality: image_embeds".

**Fix:** Strip `_embeds` suffix and map to base modality.

```python
# BEFORE
for k, v in mm_data.items():
    if k not in subparsers:
        raise ValueError(f"Unsupported modality: {k}")
    ...

# AFTER
for k, v in mm_data.items():
    # Handle {modality}_embeds keys (e.g., "image_embeds" -> "image")
    if k.endswith("_embeds"):
        modality = k[:-7]  # Remove "_embeds" suffix
    else:
        modality = k
    
    if modality not in subparsers:
        raise ValueError(f"Unsupported modality: {k}")
    ...
```

### Issue 3: Encoder Cache Not Initialized (`utils.py`)

**Location:** `vllm/v1/worker/utils.py:48-60`

**Problem:** When all limits are 0, `max_tokens_by_modality` is empty, causing `compute_mm_encoder_budget` to return (0, 0). No encoder cache is created, but embeddings still need storage. This caused generation to hang indefinitely.

**Fix:** When `enable_mm_embeds=True` and cache budgets are 0, use scheduler defaults.

```python
# After computing encoder_compute_budget and encoder_cache_size...
mm_config = model_config.get_multimodal_config()
if (mm_config is not None and mm_config.enable_mm_embeds
        and encoder_compute_budget == 0 and encoder_cache_size == 0):
    encoder_compute_budget = scheduler_config.max_num_encoder_input_tokens
    encoder_cache_size = scheduler_config.encoder_cache_size
    logger.info(
        "enable_mm_embeds is True with all modality limits=0. "
        "Using default encoder cache settings for embeddings..."
    )
```

### Issue 4: Validation Logic Order (`processing.py`)

**Location:** `vllm/multimodal/processing.py:1582-1597`

**Problem:** Original logic had flawed order - it checked limit=0 before checking if input was an embedding, causing it to skip validation for ALL inputs (not just embeddings).

**Fix:** Check if input is embedding first, then apply appropriate validation.

```python
for modality, items in mm_items.items():
    if isinstance(items, (EmbeddingItems, DictEmbeddingItems)):
        if not mm_config.enable_mm_embeds:
            raise ValueError(f"You must set `--enable-mm-embeds` to input `{modality}_embeds`")
        if mm_config.get_limit_per_prompt(modality) == 0:
            logger.info(f"Skipping count validation for modality '{modality}' (embeddings with limit=0)")
            continue
    self.validate_num_items(modality, len(items))
```

## Files Modified

| File | Description |
|------|-------------|
| `vllm/multimodal/registry.py` | Remove limit=0 modality profiling |
| `vllm/multimodal/parse.py` | Handle `{modality}_embeds` keys |
| `vllm/v1/worker/utils.py` | Encoder cache fallback for embeddings |
| `vllm/multimodal/processing.py` | Fix validation logic order |

## Expected Behaviors

| Config | Input Type | Expected Result |
|--------|-----------|-----------------|
| `enable_mm_embeds=True`, `limit=0` | Embedding | ✓ Works (skip validation) |
| `enable_mm_embeds=True`, `limit=0` | Raw image | ✗ ValueError: At most 0 image(s) |
| `enable_mm_embeds=False`, `limit=0` | Embedding | ✗ ValueError: must set --enable-mm-embeds |
| `enable_mm_embeds=False`, `limit=0` | Raw image | ✗ ValueError: At most 0 image(s) |

## Testing

Run the test script:
```bash
source /workspace/vllm/.venv/bin/activate
python test_embed_simple.py
```

**Note:** The embedding format must be a **list of 2D tensors** (each tensor is `[num_tokens, hidden_dim]`), not a single 2D tensor.

## Memory Verification

- With `limit=0`: Model loads ~12.5 GiB (encoder not loaded)
- Without limit restriction: Model loads ~13.1 GiB (encoder loaded)

## Status

- [x] Issue 1 fixed (registry.py)
- [x] Issue 2 fixed (parse.py)
- [x] Issue 3 fixed (utils.py)
- [x] Issue 4 fixed (processing.py)
- [ ] Full test suite verification pending (Test 1 & 3 need revalidation after utils.py fix)
