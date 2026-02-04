#!/usr/bin/env python3
"""
Simple test for embedding input with limit=0 using V0 engine.
"""
import os
os.environ["VLLM_USE_V1"] = "0"

import torch


def main():
    from vllm import LLM, SamplingParams

    MODEL_NAME = "llava-hf/llava-1.5-7b-hf"

    # LLaVA 1.5 7B uses 4096 hidden size and 576 image tokens
    def create_embedding():
        return torch.randn(1, 576, 4096, dtype=torch.float16)

    print("=" * 60)
    print("Test: enable_mm_embeds=True, limit=0, embedding input (V0 engine)")
    print("=" * 60)

    llm = LLM(
        model=MODEL_NAME,
        max_model_len=2048,
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        enable_mm_embeds=True,
        limit_mm_per_prompt={"image": 0},
    )

    print("Model loaded successfully")
    print("Creating embedding and running generation...")

    embedding = create_embedding()
    print(f"Embedding shape: {embedding.shape}, dtype: {embedding.dtype}")

    outputs = llm.generate(
        {"prompt": "<image>\nDescribe this image.", "multi_modal_data": {"image": embedding}},
        sampling_params=SamplingParams(max_tokens=30, temperature=0.0),
    )

    print(f"\nGeneration result: {outputs[0].outputs[0].text}")
    print("\nTest PASSED!")


if __name__ == "__main__":
    main()
