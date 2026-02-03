#!/usr/bin/env python3
"""Test that a 3D embedding tensor (with batch dim) is correctly classified as EmbeddingItems."""

import sys
sys.path.insert(0, "/workspace/vllm-skip-mod-2")

import torch
from vllm.multimodal.parse import (
    MultiModalDataParser,
    EmbeddingItems,
    DictEmbeddingItems,
)

parser = MultiModalDataParser()
embedding_2d = torch.randn(100, 768)  # Single 2D tensor (seq_len, hidden_size)
embedding_3d = embedding_2d.unsqueeze(0)  # Add batch dim: (1, seq_len, hidden_size)
mm_items = parser.parse_mm_data({"image": embedding_3d})
items = mm_items["image"]

print(f"items type: {type(items).__name__}")
print(f"isinstance(items, (EmbeddingItems, DictEmbeddingItems)): {isinstance(items, (EmbeddingItems, DictEmbeddingItems))}")
