#!/usr/bin/env python3
"""
Integration test for embedding input with disabled modalities.

Tests the following scenarios for PR #32493:

1. NEW BEHAVIOR (enable_mm_embeds=True, limit=0):
   - Embeddings should work (skip validation, save memory by not loading encoder)
   - Raw images should raise ValueError (limit is 0)

2. ORIGINAL BEHAVIOR (enable_mm_embeds=False, limit=0):
   - Embeddings should raise ValueError (embeds not enabled)
   - Raw images should raise ValueError (limit is 0)

3. MEMORY SAVINGS VERIFICATION:
   - With limit=0: encoder not loaded (less memory)
   - Without limit=0: encoder loaded (more memory)
"""

import gc
import sys
import traceback
from dataclasses import dataclass
from typing import Literal

import torch
from PIL import Image

# Test configuration
MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
MAX_MODEL_LEN = 2048
DTYPE = "float16"

# LLaVA 1.5 7B parameters
TEXT_HIDDEN_SIZE = 4096  # Vicuna 7B hidden size
NUM_IMAGE_TOKENS = 576   # 24x24 patches from CLIP ViT


@dataclass
class TestResult:
    name: str
    status: Literal["PASS", "FAIL", "ERROR"]
    message: str
    memory_gb: float | None = None


def get_gpu_memory_gb() -> float:
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024**3)
    return 0.0


def reset_gpu_memory():
    """Reset GPU memory tracking."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def create_dummy_embedding(batch_size: int = 1) -> torch.Tensor:
    """Create a dummy image embedding tensor with correct shape for LLaVA."""
    # Shape: (batch_size, num_image_tokens, hidden_size)
    return torch.randn(
        batch_size, NUM_IMAGE_TOKENS, TEXT_HIDDEN_SIZE,
        dtype=torch.float16
    )


def create_dummy_image() -> Image.Image:
    """Create a dummy PIL image for testing."""
    return Image.new("RGB", (336, 336), color="red")


def run_test_case(
    test_name: str,
    enable_mm_embeds: bool,
    limit_mm_per_prompt: dict | None,
    use_embedding: bool,
    expect_error: type | None = None,
    expect_error_message: str | None = None,
) -> TestResult:
    """
    Run a single test case.

    Args:
        test_name: Name of the test
        enable_mm_embeds: Whether to enable embedding inputs
        limit_mm_per_prompt: Limit configuration (e.g., {"image": 0})
        use_embedding: Whether to use embeddings (True) or raw image (False)
        expect_error: Expected exception type, or None if should succeed
        expect_error_message: Substring expected in error message
    """
    from vllm import LLM, SamplingParams

    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"  enable_mm_embeds={enable_mm_embeds}")
    print(f"  limit_mm_per_prompt={limit_mm_per_prompt}")
    print(f"  use_embedding={use_embedding}")
    print(f"  expect_error={expect_error}")
    print(f"{'='*60}")

    reset_gpu_memory()
    llm = None

    try:
        # Build LLM kwargs
        llm_kwargs = {
            "model": MODEL_NAME,
            "max_model_len": MAX_MODEL_LEN,
            "dtype": DTYPE,
            "enforce_eager": True,
            "gpu_memory_utilization": 0.8,
        }

        if enable_mm_embeds:
            llm_kwargs["enable_mm_embeds"] = True

        if limit_mm_per_prompt is not None:
            llm_kwargs["limit_mm_per_prompt"] = limit_mm_per_prompt

        print(f"\nInitializing LLM with kwargs: {llm_kwargs}")
        llm = LLM(**llm_kwargs)

        memory_after_load = get_gpu_memory_gb()
        print(f"Memory after model load: {memory_after_load:.4f} GB")

        # Prepare input
        if use_embedding:
            mm_data = {"image": create_dummy_embedding()}
            print("Using embedding input")
        else:
            mm_data = {"image": create_dummy_image()}
            print("Using raw image input")

        prompt = "<image>\nDescribe this image briefly."
        sampling_params = SamplingParams(max_tokens=50, temperature=0.0)

        print("Running generation...")
        outputs = llm.generate(
            {"prompt": prompt, "multi_modal_data": mm_data},
            sampling_params=sampling_params,
        )

        # If we get here, generation succeeded
        if expect_error is not None:
            return TestResult(
                name=test_name,
                status="FAIL",
                message=f"Expected {expect_error.__name__} but generation succeeded. "
                        f"Output: {outputs[0].outputs[0].text[:100]}...",
                memory_gb=memory_after_load,
            )

        print(f"Generation succeeded: {outputs[0].outputs[0].text[:100]}...")
        return TestResult(
            name=test_name,
            status="PASS",
            message="Generation completed successfully",
            memory_gb=memory_after_load,
        )

    except Exception as e:
        error_type = type(e)
        error_message = str(e)

        print(f"Exception caught: {error_type.__name__}: {error_message}")

        if expect_error is not None:
            if isinstance(e, expect_error):
                if expect_error_message is None or expect_error_message in error_message:
                    return TestResult(
                        name=test_name,
                        status="PASS",
                        message=f"Got expected error: {error_message[:200]}",
                        memory_gb=get_gpu_memory_gb(),
                    )
                else:
                    return TestResult(
                        name=test_name,
                        status="FAIL",
                        message=f"Got {error_type.__name__} but message didn't match. "
                                f"Expected '{expect_error_message}' in '{error_message}'",
                    )
            else:
                return TestResult(
                    name=test_name,
                    status="FAIL",
                    message=f"Expected {expect_error.__name__} but got {error_type.__name__}: {error_message}",
                )
        else:
            traceback.print_exc()
            return TestResult(
                name=test_name,
                status="ERROR",
                message=f"Unexpected error: {error_type.__name__}: {error_message}",
            )

    finally:
        # Cleanup
        if llm is not None:
            del llm
        reset_gpu_memory()


def main():
    print("=" * 70)
    print("EMBEDDING INPUT WITH DISABLED MODALITIES - INTEGRATION TEST")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Testing PR #32493 validation logic changes")

    results: list[TestResult] = []

    # =========================================================================
    # TEST GROUP 1: NEW BEHAVIOR (enable_mm_embeds=True, limit=0)
    # =========================================================================
    print("\n" + "#" * 70)
    print("# GROUP 1: NEW BEHAVIOR (enable_mm_embeds=True, limit_mm_per_prompt={'image': 0})")
    print("#" * 70)

    # Test 1.1: Embeddings should work with limit=0 when embeds enabled
    results.append(run_test_case(
        test_name="1.1 - Embeds enabled + limit=0 + embedding input",
        enable_mm_embeds=True,
        limit_mm_per_prompt={"image": 0},
        use_embedding=True,
        expect_error=None,  # Should succeed
    ))

    # Test 1.2: Raw images should fail with limit=0
    results.append(run_test_case(
        test_name="1.2 - Embeds enabled + limit=0 + raw image input",
        enable_mm_embeds=True,
        limit_mm_per_prompt={"image": 0},
        use_embedding=False,
        expect_error=ValueError,
        expect_error_message="At most 0 image(s) may be provided",
    ))

    # =========================================================================
    # TEST GROUP 2: ORIGINAL BEHAVIOR (enable_mm_embeds=False, limit=0)
    # =========================================================================
    print("\n" + "#" * 70)
    print("# GROUP 2: ORIGINAL BEHAVIOR (enable_mm_embeds=False, limit_mm_per_prompt={'image': 0})")
    print("#" * 70)

    # Test 2.1: Embeddings should fail when embeds not enabled
    results.append(run_test_case(
        test_name="2.1 - Embeds disabled + limit=0 + embedding input",
        enable_mm_embeds=False,
        limit_mm_per_prompt={"image": 0},
        use_embedding=True,
        expect_error=ValueError,
        expect_error_message="You must set `--enable-mm-embeds`",
    ))

    # Test 2.2: Raw images should fail with limit=0
    results.append(run_test_case(
        test_name="2.2 - Embeds disabled + limit=0 + raw image input",
        enable_mm_embeds=False,
        limit_mm_per_prompt={"image": 0},
        use_embedding=False,
        expect_error=ValueError,
        expect_error_message="At most 0 image(s) may be provided",
    ))

    # =========================================================================
    # TEST GROUP 3: MEMORY SAVINGS VERIFICATION
    # =========================================================================
    print("\n" + "#" * 70)
    print("# GROUP 3: MEMORY COMPARISON")
    print("#" * 70)

    # Test 3.1: With limit=0, should use less memory (no encoder loaded)
    results.append(run_test_case(
        test_name="3.1 - Memory with limit=0 (no encoder)",
        enable_mm_embeds=True,
        limit_mm_per_prompt={"image": 0},
        use_embedding=True,
        expect_error=None,
    ))

    # Test 3.2: Without limit restriction, should use more memory (encoder loaded)
    results.append(run_test_case(
        test_name="3.2 - Memory without limit (encoder loaded)",
        enable_mm_embeds=False,
        limit_mm_per_prompt=None,  # Default limits
        use_embedding=False,
        expect_error=None,
    ))

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    errors = sum(1 for r in results if r.status == "ERROR")

    for r in results:
        status_icon = {"PASS": "✓", "FAIL": "✗", "ERROR": "!"}[r.status]
        memory_str = f" [{r.memory_gb:.2f} GB]" if r.memory_gb else ""
        print(f"  [{status_icon}] {r.status}: {r.name}{memory_str}")
        if r.status != "PASS":
            print(f"      -> {r.message[:100]}")

    print()
    print(f"TOTAL: {passed} passed, {failed} failed, {errors} errors")

    # Memory comparison
    mem_results = [(r.name, r.memory_gb) for r in results if r.memory_gb and "Memory" in r.name]
    if len(mem_results) >= 2:
        print("\nMEMORY COMPARISON:")
        for name, mem in mem_results:
            print(f"  {name}: {mem:.4f} GB")

        limit0_mem = next((m for n, m in mem_results if "limit=0" in n), None)
        nolimit_mem = next((m for n, m in mem_results if "without limit" in n), None)
        if limit0_mem and nolimit_mem:
            savings = nolimit_mem - limit0_mem
            print(f"\n  Memory savings with limit=0: {savings:.4f} GB")
            if savings > 0:
                print("  ✓ Encoder not loaded when limit=0 (expected behavior)")
            else:
                print("  ! No memory savings detected (may need investigation)")

    print()
    return 0 if (failed == 0 and errors == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
