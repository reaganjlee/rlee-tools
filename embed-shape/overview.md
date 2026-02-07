# embed-shape: Pass modality information through `embed_input_ids`

## Problem

In Qwen3-Omni, `embed_input_ids` needs to distinguish vision (image/video) embeddings from audio embeddings to handle deepstack feature splitting. The existing code inferred modality by checking `embedding.shape[-1] != hidden_size` — a fragile heuristic that only works because vision embeddings happen to have a different last dimension due to deepstack concatenation.

## Solution

Explicitly pass modality labels (e.g. `"image"`, `"video"`, `"audio"`) alongside the embeddings, threaded from the model runner where `MultiModalFeatureSpec.modality` is already available.

An optional `modality_types: list[str] | None = None` parameter was added to `embed_input_ids`. The `supports_kw` utility (from `vllm/utils/func_utils.py`) gates whether the kwarg is passed, so existing models that don't accept it are unaffected.

## Files modified

| File | Change |
|------|--------|
| `vllm/model_executor/models/interfaces.py` | Added `modality_types` param to `embed_input_ids` overload and concrete implementation in `SupportsMultiModal` |
| `vllm/v1/worker/gpu_model_runner.py` | `_gather_mm_embeddings` now returns a 3-tuple `(mm_embeds, is_mm_embed, mm_modalities)`. Main call site uses `supports_kw` to conditionally pass `modality_types` |
| `vllm/v1/spec_decode/eagle.py` | Widened `mm_embed_inputs` type to 3-tuple. Eagle proposer unpacks and conditionally passes `modality_types` via `supports_kw` |
| `vllm/model_executor/models/qwen3_omni_moe_thinker.py` | Accepts `modality_types`. When provided, uses `m in ("image", "video")` instead of shape heuristic. Falls back to shape check when `None` for backwards compatibility |

## Key design decisions

- **Backwards compatible**: `modality_types` defaults to `None` everywhere. Models that don't override `embed_input_ids` to accept it will never receive it (guarded by `supports_kw`).
- **No caching of `supports_kw`**: `supports_kw` is already `@lru_cache`-decorated, so repeated calls are free.
- **Fallback preserved**: Qwen3-Omni's `embed_input_ids` keeps the shape-based heuristic as a fallback when `modality_types is None`, ensuring any code path that doesn't thread modality still works.

## Existing utilities reused

- `supports_kw` from `vllm/utils/func_utils.py` — checks if a callable accepts a given kwarg
- `MultiModalFeatureSpec.modality` from `vllm/multimodal/inputs.py` — already tracks modality per item

## Branch and worktree

- **Worktree**: `/workspace/vllm-embed-shape`
- **Branch**: `embed-shape`
- **Base**: `bc32444b2` ([Kernel] Add enable_sm120_or_later for SM121 (DGX Spark) CUTLASS support)

## Testing

All tests pass (87/87 passed, 28 skipped):

| Test Suite | Result |
|---|---|
| `tests/model_executor/test_qwen3_omni.py` | 1/1 passed |
| `tests/multimodal/test_embedding_shape_validation_unit.py` | 18/18 passed |
| `tests/v1/worker/test_gpu_model_runner.py` | 23/23 passed |
| `tests/v1/spec_decode/test_eagle.py` | 45/45 passed, 28 skipped (TRITON_ATTN/TREE_ATTN variants requiring unavailable backends — pre-existing, unrelated) |

## Notes

- The original commit on `skip-mod-2` (`8385b5346`) mixed this change with `budget.py` modifications. The cherry-pick into `embed-shape` excluded `budget.py` and also removed a stray `mm_max_toks_per_item` guard that belonged to the budget refactor.
- Rebased onto updated upstream main (`bc32444b2`). A merge conflict in `gpu_model_runner.py` was resolved: upstream refactored `is_mm_embed` buffer access from `self.is_mm_embed.cpu` to `is_mm_embed_buf.cpu`. The resolution kept upstream's buffer naming and added the new `mm_modalities` tracking on top.
