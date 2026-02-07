# Profiling Validation Issue for Embed-Only Modalities

## The Issue

When `budget.py` computes the encoder cache budget, it needs to know how many tokens each modality produces (e.g., 576 for LLaVA images). It discovers this by generating a dummy image and processing it through the pipeline to count placeholder tokens.

The skip-mod-2 branch added `embed_only_modalities` (limit=0, embeds enabled) to the set of modalities that get profiled. This is correct — even though the encoder won't run, the cache still needs to be sized for storing pre-computed embeddings.

The problem: the dummy image generated for profiling is a regular PIL image, not an embedding. When it flows through `parse_mm_data`, the validation logic sees "1 image submitted, but limit is 0" and rejects it. The model can't even initialize.

This wasn't caught before because upstream only profiles modalities with limit > 0, so the dummy input never conflicts with the limit.

## Solutions

### Option A: `validate=False` in generic infrastructure

Change `dummy_inputs.py` and `processor.py` to pass `validate=False` to `parse_mm_data` at their call sites. Simple (2 lines changed), but modifies generic code that all models go through, not just the embed-only path.

**Files changed:** `vllm/multimodal/processing/dummy_inputs.py`, `vllm/multimodal/processing/processor.py`

### Option B: Thread `validate` from the budget layer

Add a `validate` parameter through the full call chain: `budget.py` -> `get_mm_max_toks_per_item` -> `get_dummy_mm_inputs` -> `get_dummy_processor_inputs` -> `parse_mm_data`, and also through `apply` -> `_cached_apply_hf_processor` -> `_get_cache_missing_items` -> `parse_mm_data`. Only `budget.py` passes `False`, for embed-only modalities. Explicit and the intent originates at the right layer, but touches 7 functions across 5 files.

**Files changed:** `vllm/multimodal/budget.py`, `vllm/multimodal/registry.py`, `vllm/multimodal/processing/dummy_inputs.py`, `vllm/multimodal/processing/processor.py`, `vllm/model_executor/models/voxtral.py`, `vllm/model_executor/models/pixtral.py`

### Option C: Avoid profiling embed-only modalities entirely

Don't send embed-only modalities through the dummy input pipeline at all. Instead, get the token count some other way — e.g., from `get_mm_max_tokens_per_item()` (a config-based fast path some models implement) or a hardcoded/computed default. The problem: LLaVA doesn't implement the fast path, and the dummy input pipeline is the only generic mechanism to discover the token count. So this would require either a model-specific override or a new generic mechanism.

**Files changed:** `vllm/multimodal/budget.py` (+ new mechanism TBD)
