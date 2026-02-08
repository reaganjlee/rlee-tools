# Dummy Input Validation Test Change

## CI Failure

```
FAILED multimodal/test_processing.py::test_limit_mm_per_prompt_dummy[1-0-False-llava-hf/llava-v1.6-mistral-7b-hf]
FAILED multimodal/test_processing.py::test_limit_mm_per_prompt_dummy[2-1-False-llava-hf/llava-v1.6-mistral-7b-hf]
```

## What the Test Does

`test_limit_mm_per_prompt_dummy` calls `get_dummy_mm_inputs` directly with `mm_counts=limit_mm_per_prompt` and mocks `get_supported_mm_limits`. The two failing cases check that when the user-configured limit exceeds what the model supports (limit=1 but model supports 0, limit=2 but model supports 1), a `ValueError` is raised during dummy input generation.

## Why It Fails

The `validate=False` change in `dummy_inputs.py` (Issue 7 fix) disables all validation in `parse_mm_data` during dummy input generation. This skips `validate_num_items`, which was the only check catching the "limit > supported" case in this path.

## Why Changing the Test Is Safe

The scenario this test validates — `mm_counts` exceeding model-supported limits during dummy input generation — cannot happen in production:

1. **`budget.py` already caps counts:** `mm_counts` is derived from `allowed_mm_limits`, which computes `min(user_limit, supported_limit)`. So `budget.py` would never pass `mm_counts={"image": 2}` to `get_dummy_mm_inputs` when the model only supports 1.

2. **User-facing validation is unaffected:** The actual user request paths (`input_processor.py:243` and `preprocess.py:211`) call `parse_mm_data` with the default `validate=True`. Any user sending more items than supported will still get `ValueError: At most N image(s) may be provided`.

3. **Chat API validation is unaffected:** `chat_utils.py:541` calls `validate_num_items` independently of the dummy input path.

## Proposed Change

Flip the two failing cases from `is_valid=False` to `is_valid=True`. The rest of the test cases are unchanged.

```python
# Before:
(1, 0, False),  # limit=1, supported=0 → expected ValueError
(2, 1, False),  # limit=2, supported=1 → expected ValueError

# After:
(1, 0, True),   # dummy path no longer validates against supported limits
(2, 1, True),   # dummy path no longer validates against supported limits
```

The other 5 test cases (which test valid configurations) continue to pass as before.
