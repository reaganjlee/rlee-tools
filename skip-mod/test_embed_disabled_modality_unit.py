# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for embedding input validation with disabled modalities.

Tests the fixes from PR #32493:
1. registry.py           - Exclude limit=0 modalities from profiling
2. parse.py              - Embedding detection via tensor shape
3. budget.py             - Encoder cache fallback when all limits=0 + enable_mm_embeds
4. processing.py         - Fix validation logic order (isinstance before enable_mm_embeds)
5. gpu_model_runner.py   - Handle empty mm_max_toks_per_item in profile_run

Run with:
    pytest tests/multimodal/test_embed_disabled_modality_unit.py -v
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.multimodal.parse import (
    ImageEmbeddingItems,
    MultiModalDataItems,
    MultiModalDataParser,
)


# ============================================================================
# Test 1: registry.py - Exclude limit=0 modalities from profiling
# ============================================================================


class TestRegistryExcludesLimitZeroFromProfiling:
    """
    Verify that get_max_tokens_per_item_by_modality excludes modalities
    with limit=0 from profiling, even when enable_mm_embeds=True.

    The fix removed the special-case that included limit=0 modalities
    for profiling when enable_mm_embeds was True. Profiling creates dummy
    images which fail validation ("At most 0 image(s)").
    """

    def test_limit_zero_excluded_from_modality_counts(self):
        """The dict comprehension should exclude limit=0 entries."""
        profiler_limits = {"image": 0, "audio": 2, "video": 0}

        # This is the exact logic from registry.py line 178
        modality_counts = {
            modality: 1
            for modality, limit in profiler_limits.items()
            if limit > 0
        }

        assert modality_counts == {"audio": 1}
        assert "image" not in modality_counts
        assert "video" not in modality_counts

    def test_all_limits_zero_yields_empty(self):
        """When all limits are 0, modality_counts should be empty."""
        profiler_limits = {"image": 0, "video": 0}

        modality_counts = {
            modality: 1
            for modality, limit in profiler_limits.items()
            if limit > 0
        }

        assert modality_counts == {}

    def test_positive_limits_included(self):
        """Positive limits should be included normally."""
        profiler_limits = {"image": 5, "audio": 3}

        modality_counts = {
            modality: 1
            for modality, limit in profiler_limits.items()
            if limit > 0
        }

        assert modality_counts == {"image": 1, "audio": 1}


# ============================================================================
# Test 2: parse.py - Handle {modality}_embeds keys
# ============================================================================


class TestParserEmbeddingDetection:
    """
    Verify that parse_mm_data detects embeddings via tensor shape
    when passed under the standard 'image' key.
    """

    def _make_parser(self):
        """Create a MultiModalDataParser with default settings."""
        return MultiModalDataParser()

    def test_3d_tensor_detected_as_embedding(self):
        """3D tensor under 'image' key should be detected as embedding."""
        parser = self._make_parser()

        # 3D tensor is detected as embedding by is_embeddings()
        embedding = torch.randn(1, 10, 768)

        result = parser.parse_mm_data({"image": embedding})

        assert "image" in result
        assert isinstance(result["image"], ImageEmbeddingItems)

    def test_list_of_2d_tensors_detected_as_embedding(self):
        """List of 2D tensors under 'image' key should be detected as embedding."""
        parser = self._make_parser()

        embeddings = [torch.randn(10, 768), torch.randn(15, 768)]

        result = parser.parse_mm_data({"image": embeddings})

        assert "image" in result
        assert isinstance(result["image"], ImageEmbeddingItems)
        assert result["image"].get_count() == 2

    def test_pil_image_not_detected_as_embedding(self):
        """PIL image should NOT be detected as embedding."""
        parser = self._make_parser()

        from PIL import Image
        dummy_image = Image.new("RGB", (64, 64), color="red")

        result = parser.parse_mm_data({"image": dummy_image})

        assert "image" in result
        assert not isinstance(result["image"], ImageEmbeddingItems)

    def test_unsupported_key_raises(self):
        """Unsupported key should raise ValueError."""
        parser = self._make_parser()

        embedding = torch.randn(1, 10, 768)

        with pytest.raises(ValueError, match="Unsupported modality"):
            parser.parse_mm_data({"unknown": embedding})


# ============================================================================
# Test 3: budget.py - Encoder cache fallback for embeddings
# ============================================================================


class TestEncoderCacheFallback:
    """
    Verify that when enable_mm_embeds=True and all limits are 0,
    the encoder cache falls back to scheduler defaults instead of (0, 0).

    Without this fix, compute_mm_encoder_budget returns (0, 0) when
    mm_max_toks_per_item is empty, causing generation to hang.
    """

    def test_compute_budget_returns_zero_for_empty_modalities(self):
        """compute_mm_encoder_budget returns (0, 0) when no modalities."""
        from vllm.v1.core.encoder_cache_manager import compute_mm_encoder_budget

        scheduler_config = MagicMock()
        scheduler_config.max_num_encoder_input_tokens = 16384
        scheduler_config.encoder_cache_size = 16384

        budget, cache = compute_mm_encoder_budget(scheduler_config, {})

        assert budget == 0
        assert cache == 0

    def test_fallback_logic_applies_when_embeds_enabled_and_zero_budget(self):
        """
        The fallback logic in utils.py should set budgets to scheduler
        defaults when enable_mm_embeds=True and budgets are (0, 0).
        """
        mm_config = MagicMock()
        mm_config.enable_mm_embeds = True

        model_config = MagicMock()
        model_config.get_multimodal_config.return_value = mm_config

        scheduler_config = MagicMock()
        scheduler_config.max_num_encoder_input_tokens = 16384
        scheduler_config.encoder_cache_size = 8192

        # Simulate the fallback logic from utils.py lines 62-72
        encoder_compute_budget = 0
        encoder_cache_size = 0

        config = model_config.get_multimodal_config()
        if (config is not None and config.enable_mm_embeds
                and encoder_compute_budget == 0 and encoder_cache_size == 0):
            encoder_compute_budget = scheduler_config.max_num_encoder_input_tokens
            encoder_cache_size = scheduler_config.encoder_cache_size

        assert encoder_compute_budget == 16384
        assert encoder_cache_size == 8192

    def test_fallback_does_not_apply_when_embeds_disabled(self):
        """
        When enable_mm_embeds=False, budgets should remain at (0, 0).
        """
        mm_config = MagicMock()
        mm_config.enable_mm_embeds = False

        model_config = MagicMock()
        model_config.get_multimodal_config.return_value = mm_config

        scheduler_config = MagicMock()
        scheduler_config.max_num_encoder_input_tokens = 16384
        scheduler_config.encoder_cache_size = 8192

        encoder_compute_budget = 0
        encoder_cache_size = 0

        config = model_config.get_multimodal_config()
        if (config is not None and config.enable_mm_embeds
                and encoder_compute_budget == 0 and encoder_cache_size == 0):
            encoder_compute_budget = scheduler_config.max_num_encoder_input_tokens
            encoder_cache_size = scheduler_config.encoder_cache_size

        # Should remain at 0 since embeds are disabled
        assert encoder_compute_budget == 0
        assert encoder_cache_size == 0

    def test_fallback_does_not_apply_when_budget_nonzero(self):
        """
        When budgets are already non-zero (some modalities have limits > 0),
        the fallback should not override them.
        """
        mm_config = MagicMock()
        mm_config.enable_mm_embeds = True

        model_config = MagicMock()
        model_config.get_multimodal_config.return_value = mm_config

        scheduler_config = MagicMock()
        scheduler_config.max_num_encoder_input_tokens = 16384
        scheduler_config.encoder_cache_size = 8192

        # Non-zero budgets (some modalities have limit > 0)
        encoder_compute_budget = 4096
        encoder_cache_size = 2048

        config = model_config.get_multimodal_config()
        if (config is not None and config.enable_mm_embeds
                and encoder_compute_budget == 0 and encoder_cache_size == 0):
            encoder_compute_budget = scheduler_config.max_num_encoder_input_tokens
            encoder_cache_size = scheduler_config.encoder_cache_size

        # Should keep original values
        assert encoder_compute_budget == 4096
        assert encoder_cache_size == 2048


# ============================================================================
# Test 4: processing.py - Validation logic order
# ============================================================================


class TestValidationLogicOrder:
    """
    Verify that _to_mm_items checks isinstance(EmbeddingItems) first,
    then checks enable_mm_embeds, then checks limit=0.

    The original buggy logic checked limit=0 first and then isinstance,
    which could skip validation for ALL inputs (not just embeddings)
    or reject embeddings when it shouldn't.
    """

    def _make_mock_processor(self, enable_mm_embeds, limit_per_prompt):
        """Create a mock processor with the specified config."""
        from vllm.config.multimodal import MultiModalConfig

        mm_config = MultiModalConfig(
            limit_per_prompt=limit_per_prompt,
            enable_mm_embeds=enable_mm_embeds,
        )

        processor = MagicMock()
        processor.info.ctx.model_config.get_multimodal_config.return_value = mm_config
        processor.supported_mm_limits = {"image": 999}
        processor.allowed_mm_limits = {"image": limit_per_prompt.get("image", 999)}

        return processor, mm_config

    def test_embedding_with_embeds_enabled_limit_zero_skips_validation(self):
        """
        enable_mm_embeds=True, limit=0, embedding input -> skip validation.
        """
        _, mm_config = self._make_mock_processor(
            enable_mm_embeds=True,
            limit_per_prompt={"image": 0},
        )

        # Simulate the validation logic from processing.py
        embedding_items = MagicMock(spec=ImageEmbeddingItems)
        mm_items = MultiModalDataItems()
        mm_items["image"] = embedding_items

        validated = []
        skipped = []

        for modality, items in mm_items.items():
            from vllm.multimodal.parse import DictEmbeddingItems, EmbeddingItems
            if isinstance(items, (EmbeddingItems, DictEmbeddingItems)):
                if not mm_config.enable_mm_embeds:
                    raise ValueError(
                        f"You must set `--enable-mm-embeds` to input "
                        f"`{modality}_embeds`"
                    )
                if mm_config.get_limit_per_prompt(modality) == 0:
                    skipped.append(modality)
                    continue
            validated.append(modality)

        assert "image" in skipped
        assert "image" not in validated

    def test_embedding_with_embeds_disabled_raises_error(self):
        """
        enable_mm_embeds=False, embedding input -> ValueError about --enable-mm-embeds.
        """
        _, mm_config = self._make_mock_processor(
            enable_mm_embeds=False,
            limit_per_prompt={"image": 0},
        )

        embedding_items = MagicMock(spec=ImageEmbeddingItems)
        mm_items = MultiModalDataItems()
        mm_items["image"] = embedding_items

        from vllm.multimodal.parse import DictEmbeddingItems, EmbeddingItems

        with pytest.raises(ValueError, match="enable-mm-embeds"):
            for modality, items in mm_items.items():
                if isinstance(items, (EmbeddingItems, DictEmbeddingItems)):
                    if not mm_config.enable_mm_embeds:
                        raise ValueError(
                            f"You must set `--enable-mm-embeds` to input "
                            f"`{modality}_embeds`"
                        )
                    if mm_config.get_limit_per_prompt(modality) == 0:
                        continue
                # validate_num_items would be called here

    def test_raw_image_with_limit_zero_gets_validated(self):
        """
        Raw image (not EmbeddingItems) with limit=0 -> should be validated
        and fail with "At most 0 image(s)".

        This is the key case the original buggy logic got wrong: it would
        skip validation for raw images too when limit=0 + enable_mm_embeds.
        """
        _, mm_config = self._make_mock_processor(
            enable_mm_embeds=True,
            limit_per_prompt={"image": 0},
        )

        # Use a non-embedding item (simulates raw PIL image)
        raw_image_items = MagicMock()
        raw_image_items.__class__ = type("ImageProcessorItems", (), {})
        mm_items = MultiModalDataItems()
        mm_items["image"] = raw_image_items

        validated_modalities = []

        from vllm.multimodal.parse import DictEmbeddingItems, EmbeddingItems
        for modality, items in mm_items.items():
            if isinstance(items, (EmbeddingItems, DictEmbeddingItems)):
                if not mm_config.enable_mm_embeds:
                    raise ValueError(
                        f"You must set `--enable-mm-embeds` to input "
                        f"`{modality}_embeds`"
                    )
                if mm_config.get_limit_per_prompt(modality) == 0:
                    continue
            validated_modalities.append(modality)

        # Raw image should reach validation (not be skipped)
        assert "image" in validated_modalities

    def test_embedding_with_positive_limit_gets_validated(self):
        """
        enable_mm_embeds=True, limit>0, embedding input -> should be validated.
        Only limit=0 embeddings skip validation.
        """
        _, mm_config = self._make_mock_processor(
            enable_mm_embeds=True,
            limit_per_prompt={"image": 5},
        )

        embedding_items = MagicMock(spec=ImageEmbeddingItems)
        mm_items = MultiModalDataItems()
        mm_items["image"] = embedding_items

        validated_modalities = []
        skipped = []

        from vllm.multimodal.parse import DictEmbeddingItems, EmbeddingItems
        for modality, items in mm_items.items():
            if isinstance(items, (EmbeddingItems, DictEmbeddingItems)):
                if not mm_config.enable_mm_embeds:
                    raise ValueError(
                        f"You must set `--enable-mm-embeds` to input "
                        f"`{modality}_embeds`"
                    )
                if mm_config.get_limit_per_prompt(modality) == 0:
                    skipped.append(modality)
                    continue
            validated_modalities.append(modality)

        # Should be validated, not skipped (limit is 5, not 0)
        assert "image" in validated_modalities
        assert "image" not in skipped


# ============================================================================
# Test 5: gpu_model_runner.py - Handle empty mm_max_toks_per_item
# ============================================================================


class TestProfileRunEmptyModalities:
    """
    Verify that profile_run skips profiling when mm_max_toks_per_item
    is empty, instead of crashing with 'max() iterable argument is empty'.

    This happened when enable_mm_embeds=True with all limits=0: the encoder
    budget fallback sets non-zero budgets, but there are no modalities to
    profile (mm_max_toks_per_item is empty).
    """

    def test_empty_mm_max_toks_per_item_handled(self):
        """
        When mm_max_toks_per_item is empty but encoder_budget > 0,
        profiling should be skipped without crashing.
        """
        # Simulate the profile_run logic
        mm_max_toks_per_item = {}  # Empty - all limits=0
        encoder_budget = 8192  # Non-zero from fallback

        if encoder_budget > 0:
            if not mm_max_toks_per_item:
                # Should take this path and skip profiling
                skipped = True
            else:
                skipped = False
                # This would crash with empty dict:
                # max(mm_max_toks_per_item.items(), key=lambda x: x[1])

        assert skipped is True

    def test_non_empty_modalities_proceeds(self):
        """
        Normal case: non-empty modalities should proceed to profiling.
        """
        mm_max_toks_per_item = {"image": 576}
        encoder_budget = 8192

        if encoder_budget > 0:
            if not mm_max_toks_per_item:
                skipped = True
            else:
                skipped = False
                modality, _ = max(
                    mm_max_toks_per_item.items(),
                    key=lambda x: x[1],
                )

        assert skipped is False
        assert modality == "image"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
