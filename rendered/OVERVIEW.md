# Rendered Branch: Online APIs Pass Rendered Prompts via `_process_inputs()`

## Problem

Several online API endpoints in vLLM bypassed the standard input processing pipeline and passed raw `PromptType` or `TokensPrompt` directly to `engine_client.generate()`. This is inconsistent with how modern endpoints (Chat, Completions, Responses) work, where prompts go through `_process_inputs()` to create an `EngineCoreRequest` before reaching the engine.

The standard pipeline is:
1. Construct a `TokensPrompt` / `PromptType`
2. `_process_inputs()` converts it to an `EngineCoreRequest` (handles multimodal processing, validation, LoRA, truncation)
3. `engine_client.generate(EngineCoreRequest, ...)`

## Changes

All changes are in `/workspace/vllm-rendered` (the `rendered` worktree/branch).

### 1. Disaggregated Serving (`vllm/entrypoints/serve/disagg/serving.py`)

- Removed the TODO comment: `"Change to EngineCoreRequest once Renderer work is completed"`
- Added `await self._process_inputs()` call before `engine_client.generate()` to convert the `TokensPrompt` into an `EngineCoreRequest`
- `engine_client.generate()` now receives the `engine_request` instead of the raw prompt, plus `tokenization_kwargs` from `_process_inputs()`

### 2. Speech-to-Text (`vllm/entrypoints/openai/speech_to_text.py`)

- Converted a list comprehension into an explicit loop so each prompt can be `await`ed through `_process_inputs()`
- Each audio chunk's prompt now goes through `_process_inputs()` to create an `EngineCoreRequest` before being passed to `engine_client.generate()`
- Uses `trace_headers=None, priority=0` (this endpoint has no trace headers or priority support)

### Not Updated

- **Beam Search** (`serving_engine.py`): Intentionally left as-is. It's an iterative algorithm that builds `TokensPrompt` from accumulated tokens each step; adding `_process_inputs()` per iteration would add overhead with no benefit.

## Pattern Reference

Both changes follow the same pattern used in `serving_completion.py:227-247` and `serving_chat.py:400`:

```python
engine_request, tokenization_kwargs = await self._process_inputs(
    request_id,
    engine_prompt,
    sampling_params,
    lora_request=lora_request,
    trace_headers=trace_headers,
    priority=priority,
)

result_generator = self.engine_client.generate(
    engine_request,
    sampling_params,
    request_id,
    lora_request=lora_request,
    trace_headers=trace_headers,
    priority=priority,
    tokenization_kwargs=tokenization_kwargs,
)
```

## Verification

- `ruff check` passes on both modified files
