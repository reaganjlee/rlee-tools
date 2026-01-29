#!/usr/bin/env python3
"""
Mock for Qwen2.5-VL vision encoder to speed up bisect verification.
Activated via VLLM_MOCK_VISION_ENCODER=1 environment variable.

This module monkey-patches the embed_multimodal() method to return dummy tensors
instead of running the expensive vision encoder forward pass. This preserves
all downstream code paths (scheduler, KV cache, decoder) while skipping the
encoder computation.

Usage:
    Set VLLM_MOCK_VISION_ENCODER=1 and ensure this module is in PYTHONPATH.
    The mock will be applied automatically when the qwen2_5_vl module is imported.
"""
import os
import sys
import importlib.abc
import importlib.machinery
from functools import wraps

MOCK_ENABLED = os.environ.get("VLLM_MOCK_VISION_ENCODER", "0") == "1"
_PATCHED = False


def _apply_patch():
    """Apply mock patch to Qwen2_5_VLForConditionalGeneration.embed_multimodal"""
    global _PATCHED
    if _PATCHED:
        return

    try:
        import torch
        from vllm.model_executor.models import qwen2_5_vl

        original_embed_multimodal = qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.embed_multimodal

        @wraps(original_embed_multimodal)
        def mock_embed_multimodal(self, **kwargs):
            """Return dummy tensors with correct shapes instead of running encoder."""
            mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
            if not mm_input_by_modality:
                return []

            # Get config values for shape computation
            spatial_merge_size = self.config.vision_config.spatial_merge_size
            out_hidden_size = self.config.vision_config.out_hidden_size
            device = next(self.parameters()).device
            dtype = torch.bfloat16
            if hasattr(self, 'visual') and self.visual is not None:
                try:
                    dtype = next(self.visual.parameters()).dtype
                except StopIteration:
                    pass

            multimodal_embeddings = ()

            for modality in mm_input_by_modality:
                multimodal_input = mm_input_by_modality[modality]

                if modality == "image":
                    grid_thw = multimodal_input["image_grid_thw"]
                elif modality == "video":
                    grid_thw = multimodal_input["video_grid_thw"]
                else:
                    continue

                # Compute token counts: (t * h * w) // merge_size^2 for each item
                sizes = (grid_thw.prod(-1) // spatial_merge_size // spatial_merge_size).tolist()

                for num_tokens in sizes:
                    dummy_embed = torch.zeros(
                        (num_tokens, out_hidden_size),
                        dtype=dtype,
                        device=device
                    )
                    multimodal_embeddings += (dummy_embed,)

            return multimodal_embeddings

        qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.embed_multimodal = mock_embed_multimodal
        _PATCHED = True
        print("[MOCK] Vision encoder mock enabled - returning dummy embeddings", flush=True)

    except ImportError as e:
        print(f"[MOCK] Warning: Could not patch qwen2_5_vl: {e}", flush=True)
    except Exception as e:
        print(f"[MOCK] Warning: Error applying patch: {e}", flush=True)


class MockEncoderFinder(importlib.abc.MetaPathFinder):
    """
    Import hook that patches qwen2_5_vl module after it's loaded.
    """
    def find_module(self, fullname, path=None):
        if fullname == "vllm.model_executor.models.qwen2_5_vl":
            return MockEncoderLoader()
        return None


class MockEncoderLoader(importlib.abc.Loader):
    """
    Loader that imports the real module and then applies the patch.
    """
    def load_module(self, fullname):
        # Remove ourselves temporarily to avoid recursion
        sys.meta_path = [p for p in sys.meta_path if not isinstance(p, MockEncoderFinder)]

        try:
            # Import the real module
            module = importlib.import_module(fullname)

            # Apply the patch after the module is loaded
            _apply_patch()

            return module
        finally:
            # Re-add ourselves for any future imports (shouldn't happen, but just in case)
            if MOCK_ENABLED and not any(isinstance(p, MockEncoderFinder) for p in sys.meta_path):
                sys.meta_path.insert(0, MockEncoderFinder())


def install_hook():
    """Install the import hook to patch qwen2_5_vl when it's imported."""
    if not MOCK_ENABLED:
        return

    # Check if already installed
    if any(isinstance(p, MockEncoderFinder) for p in sys.meta_path):
        return

    # Check if module is already loaded
    if "vllm.model_executor.models.qwen2_5_vl" in sys.modules:
        _apply_patch()
    else:
        # Install hook for when it gets imported
        sys.meta_path.insert(0, MockEncoderFinder())
        print("[MOCK] Import hook installed for vision encoder mock", flush=True)


# Auto-install hook when module is imported
if MOCK_ENABLED:
    install_hook()
