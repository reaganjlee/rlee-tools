"""
Test script to verify embedding input validation for disabled modalities.

Expected behaviors:
| Config                              | Input Type | Expected Result                    |
|-------------------------------------|------------|------------------------------------|
| enable_mm_embeds=True, limit=0      | Embedding  | Works (skip validation)            |
| enable_mm_embeds=True, limit=0      | Raw image  | ValueError: At most 0 image(s)     |
| enable_mm_embeds=False, limit=0     | Embedding  | ValueError: must set --enable-mm-embeds |
| enable_mm_embeds=False, limit=0     | Raw image  | ValueError: At most 0 image(s)     |
"""

import numpy as np
import torch


def test_embedding_with_embeds_enabled_limit_zero():
    """Test: enable_mm_embeds=True, limit=0, input=embedding -> Should work"""
    from vllm import LLM, SamplingParams

    print("\n" + "=" * 70)
    print("TEST 1: enable_mm_embeds=True, limit=0, input=embedding")
    print("Expected: Should work (skip validation)")
    print("=" * 70)

    try:
        llm = LLM(
            model="llava-hf/llava-1.5-7b-hf",
            limit_mm_per_prompt={"image": 0},
            enable_mm_embeds=True,
            max_model_len=2048,
            enforce_eager=True,
        )

        # Create a dummy embedding (576 tokens x 4096 hidden dim for LLaVA 1.5)
        dummy_embedding = torch.randn(576, 4096, dtype=torch.float16)

        prompt = "<image>\nUSER: Describe this image.\nASSISTANT:"
        sampling_params = SamplingParams(max_tokens=50, temperature=0.0)

        outputs = llm.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {"image_embeds": dummy_embedding},
            },
            sampling_params=sampling_params,
        )

        print(f"SUCCESS: Generated output: {outputs[0].outputs[0].text[:100]}...")
        return True

    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        return False


def test_raw_image_with_embeds_enabled_limit_zero():
    """Test: enable_mm_embeds=True, limit=0, input=raw image -> Should fail"""
    from vllm import LLM, SamplingParams
    from PIL import Image

    print("\n" + "=" * 70)
    print("TEST 2: enable_mm_embeds=True, limit=0, input=raw image")
    print("Expected: ValueError (At most 0 image(s))")
    print("=" * 70)

    try:
        llm = LLM(
            model="llava-hf/llava-1.5-7b-hf",
            limit_mm_per_prompt={"image": 0},
            enable_mm_embeds=True,
            max_model_len=2048,
            enforce_eager=True,
        )

        # Create a dummy raw image
        dummy_image = Image.new("RGB", (224, 224), color="red")

        prompt = "<image>\nUSER: Describe this image.\nASSISTANT:"
        sampling_params = SamplingParams(max_tokens=50, temperature=0.0)

        outputs = llm.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {"image": dummy_image},
            },
            sampling_params=sampling_params,
        )

        print(f"FAILED: Should have raised ValueError, but got output: {outputs[0].outputs[0].text[:100]}...")
        return False

    except ValueError as e:
        if "At most 0" in str(e) or "at most 0" in str(e).lower():
            print(f"SUCCESS: Got expected ValueError: {e}")
            return True
        else:
            print(f"FAILED: Got ValueError but wrong message: {e}")
            return False
    except Exception as e:
        print(f"FAILED: Got unexpected exception: {type(e).__name__}: {e}")
        return False


def test_embedding_with_embeds_disabled_limit_zero():
    """Test: enable_mm_embeds=False, limit=0, input=embedding -> Should fail"""
    from vllm import LLM, SamplingParams

    print("\n" + "=" * 70)
    print("TEST 3: enable_mm_embeds=False, limit=0, input=embedding")
    print("Expected: ValueError (must set --enable-mm-embeds)")
    print("=" * 70)

    try:
        llm = LLM(
            model="llava-hf/llava-1.5-7b-hf",
            limit_mm_per_prompt={"image": 0},
            enable_mm_embeds=False,  # Disabled
            max_model_len=2048,
            enforce_eager=True,
        )

        # Create a dummy embedding
        dummy_embedding = torch.randn(576, 4096, dtype=torch.float16)

        prompt = "<image>\nUSER: Describe this image.\nASSISTANT:"
        sampling_params = SamplingParams(max_tokens=50, temperature=0.0)

        outputs = llm.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {"image_embeds": dummy_embedding},
            },
            sampling_params=sampling_params,
        )

        print(f"FAILED: Should have raised ValueError, but got output: {outputs[0].outputs[0].text[:100]}...")
        return False

    except ValueError as e:
        if "enable-mm-embeds" in str(e).lower() or "enable_mm_embeds" in str(e):
            print(f"SUCCESS: Got expected ValueError: {e}")
            return True
        else:
            print(f"FAILED: Got ValueError but wrong message: {e}")
            return False
    except Exception as e:
        print(f"FAILED: Got unexpected exception: {type(e).__name__}: {e}")
        return False


def test_raw_image_with_embeds_disabled_limit_zero():
    """Test: enable_mm_embeds=False, limit=0, input=raw image -> Should fail"""
    from vllm import LLM, SamplingParams
    from PIL import Image

    print("\n" + "=" * 70)
    print("TEST 4: enable_mm_embeds=False, limit=0, input=raw image")
    print("Expected: ValueError (At most 0 image(s))")
    print("=" * 70)

    try:
        llm = LLM(
            model="llava-hf/llava-1.5-7b-hf",
            limit_mm_per_prompt={"image": 0},
            enable_mm_embeds=False,
            max_model_len=2048,
            enforce_eager=True,
        )

        # Create a dummy raw image
        dummy_image = Image.new("RGB", (224, 224), color="red")

        prompt = "<image>\nUSER: Describe this image.\nASSISTANT:"
        sampling_params = SamplingParams(max_tokens=50, temperature=0.0)

        outputs = llm.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {"image": dummy_image},
            },
            sampling_params=sampling_params,
        )

        print(f"FAILED: Should have raised ValueError, but got output: {outputs[0].outputs[0].text[:100]}...")
        return False

    except ValueError as e:
        if "At most 0" in str(e) or "at most 0" in str(e).lower():
            print(f"SUCCESS: Got expected ValueError: {e}")
            return True
        else:
            print(f"FAILED: Got ValueError but wrong message: {e}")
            return False
    except Exception as e:
        print(f"FAILED: Got unexpected exception: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("EMBEDDING INPUT VALIDATION TEST SUITE")
    print("=" * 70)

    # Run tests one at a time (each creates a new LLM instance)
    results = []

    # Test 1: Embedding with embeds enabled and limit=0 (should work)
    results.append(("Test 1: embed + enabled + limit=0", test_embedding_with_embeds_enabled_limit_zero()))

    # Test 2: Raw image with embeds enabled and limit=0 (should fail)
    results.append(("Test 2: raw + enabled + limit=0", test_raw_image_with_embeds_enabled_limit_zero()))

    # Test 3: Embedding with embeds disabled and limit=0 (should fail)
    results.append(("Test 3: embed + disabled + limit=0", test_embedding_with_embeds_disabled_limit_zero()))

    # Test 4: Raw image with embeds disabled and limit=0 (should fail)
    results.append(("Test 4: raw + disabled + limit=0", test_raw_image_with_embeds_disabled_limit_zero()))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED!")
        sys.exit(1)
