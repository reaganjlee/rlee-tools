# SPDX-License-Identifier: Apache-2.0
"""
E2E test: model launches and generates under enable_mm_embeds=True + limit=0.

This guards against the scheduler hang (Issue 2) and the various crashes
(Issues 5/6) described in OVERVIEW.md. The critical assertion is that the
model starts up, accepts a pre-computed embedding, and produces output
tokens without hanging.

Requires: GPU, llava-hf/llava-1.5-7b-hf weights.

Run:
    source /workspace/vllm/.venv/bin/activate
    pytest test_e2e_embed_disabled_modality.py -v -x --timeout=300
"""

import weakref

import pytest
import torch
from PIL import Image

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.distributed import cleanup_dist_env_and_memory

MODEL = "llava-hf/llava-1.5-7b-hf"
PROMPT = "USER: <image>\nDescribe this image briefly.\nASSISTANT:"
TEXT_ONLY_PROMPT = "USER: What is 2 + 2?\nASSISTANT:"

# LLaVA 1.5 7B dimensions (for synthetic embeddings)
NUM_IMAGE_TOKENS = 576
TEXT_HIDDEN_SIZE = 4096


@pytest.fixture(scope="module")
def embed_only_llm():
    """LLM with enable_mm_embeds=True, limit_mm_per_prompt={"image": 0}.

    This is the configuration that previously caused hangs/crashes.
    Module-scoped so the model is loaded once for all tests in this file.
    """
    llm = LLM(
        model=MODEL,
        max_model_len=2048,
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        enable_mm_embeds=True,
        limit_mm_per_prompt={"image": 0},
    )
    yield weakref.proxy(llm)
    del llm
    cleanup_dist_env_and_memory()


@pytest.fixture
def real_embedding():
    """Pre-computed embedding from vllm's test assets (stop_sign)."""
    return ImageAsset("stop_sign").image_embeds


@pytest.fixture
def synthetic_embedding():
    """Synthetic embedding with the correct shape for LLaVA 1.5."""
    return torch.randn(1, NUM_IMAGE_TOKENS, TEXT_HIDDEN_SIZE, dtype=torch.float16)


@pytest.fixture
def raw_image():
    """A real PIL image that should be rejected when limit=0."""
    return ImageAsset("stop_sign").pil_image


# --------------------------------------------------------------------------- #
# Core e2e: model launches and generates with embeddings
# --------------------------------------------------------------------------- #

@pytest.mark.timeout(120)
def test_model_generates_with_real_embedding(embed_only_llm, real_embedding):
    """The happy path: pre-computed embedding in, tokens out, no hang."""
    outputs = embed_only_llm.generate(
        {"prompt": PROMPT, "multi_modal_data": {"image": real_embedding}},
        sampling_params=SamplingParams(max_tokens=32, temperature=0.0),
    )
    assert len(outputs) == 1
    text = outputs[0].outputs[0].text
    assert len(text) > 0, "Model produced empty output"


@pytest.mark.timeout(120)
def test_model_generates_with_synthetic_embedding(
    embed_only_llm, synthetic_embedding
):
    """Synthetic random embedding — output will be gibberish but must not hang."""
    outputs = embed_only_llm.generate(
        {"prompt": PROMPT, "multi_modal_data": {"image": synthetic_embedding}},
        sampling_params=SamplingParams(max_tokens=16, temperature=0.0),
    )
    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].text) > 0


# --------------------------------------------------------------------------- #
# Validation: raw images rejected, text-only still works
# --------------------------------------------------------------------------- #

def test_raw_image_rejected(embed_only_llm, raw_image):
    """Raw image input must be rejected when limit=0."""
    with pytest.raises(ValueError, match=r"At most 0 image\(s\)"):
        embed_only_llm.generate(
            {"prompt": PROMPT, "multi_modal_data": {"image": raw_image}},
            sampling_params=SamplingParams(max_tokens=16),
        )


@pytest.mark.timeout(120)
def test_text_only_prompt_works(embed_only_llm):
    """Text-only prompts must not regress under this config."""
    outputs = embed_only_llm.generate(
        TEXT_ONLY_PROMPT,
        sampling_params=SamplingParams(max_tokens=16, temperature=0.0),
    )
    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].text) > 0


# --------------------------------------------------------------------------- #
# Batch: multiple embedding requests in one call
# --------------------------------------------------------------------------- #

@pytest.mark.timeout(180)
def test_batch_embedding_requests(embed_only_llm, real_embedding):
    """Multiple embedding requests in a single generate() call.

    Guards against scheduler issues that only manifest with batched requests.
    """
    batch = [
        {"prompt": PROMPT, "multi_modal_data": {"image": real_embedding}}
        for _ in range(3)
    ]
    outputs = embed_only_llm.generate(
        batch,
        sampling_params=SamplingParams(max_tokens=16, temperature=0.0),
    )
    assert len(outputs) == 3
    for out in outputs:
        assert len(out.outputs[0].text) > 0
