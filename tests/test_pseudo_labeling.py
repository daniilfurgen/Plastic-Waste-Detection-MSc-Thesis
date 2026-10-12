"""
Unit tests for pseudo-label score thresholding, per-box weights and image sampling.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import polars as pl
import pytest
import torch
from ultralytics.engine.results import Boxes

from ssod.pseudo_labeling import (
    BinThreshold,
    PseudoLabelGenerator,
    assign_size_bin,
    box_weights,
    fit_bin_threshold,
    ground_modes,
    ground_threshold,
    input_scale,
    sample_image_files,
    select_pseudo_labels,
)


def _scores_from_histogram(counts: list[int], lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Expand per-bin counts into scores placed at each bin centre."""
    edges = np.linspace(lo, hi, len(counts) + 1)
    centres = (edges[:-1] + edges[1:]) / 2.0
    return np.repeat(centres, counts)


def _bimodal_scores(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.05, 0.05, 20000).clip(0.0, 1.0)
    objects = rng.normal(0.85, 0.05, 8000).clip(0.0, 1.0)
    return np.concatenate([noise, objects])


class TestGroundThreshold:
    """Test valley detection between the noise and object score modes."""

    def test_bimodal_returns_valley(self):
        threshold = ground_threshold(_bimodal_scores())
        assert threshold is not None
        assert 0.25 < threshold < 0.7

    def test_object_mode_dominant(self):
        # Later SSL epochs: the object mode outgrows the noise mode
        rng = np.random.default_rng(1)
        noise = rng.normal(0.02, 0.03, 5000).clip(0.0, 1.0)
        objects = rng.normal(0.85, 0.03, 20000).clip(0.0, 1.0)
        threshold = ground_threshold(np.concatenate([noise, objects]))
        assert threshold is not None
        assert 0.1 < threshold < 0.8

    def test_observed_epoch_one_medium_histogram(self):
        # Medium-bin candidate histogram over [0, 1] (21 bins) from run ssod-v2-100, epoch 1
        counts = [
            17900, 1998, 1268, 951, 775, 714, 623, 573, 570, 577, 590,
            602, 644, 698, 779, 985, 1465, 2535, 3202, 1643, 0,
        ]
        threshold = ground_threshold(_scores_from_histogram(counts))
        assert threshold is not None
        assert 0.3 <= threshold <= 0.5

    def test_clipped_monotonic_range_has_no_valley(self):
        # The same scores restricted to [0.5, 1.0] rise monotonically: no valley must be reported
        rng = np.random.default_rng(2)
        scores = rng.triangular(0.5, 0.9, 0.9, 10000)
        assert ground_threshold(scores, lo=0.5) is None

    def test_empty_scores(self):
        assert ground_threshold(np.array([], dtype=float)) is None


class TestGroundModes:
    """Test that the object mode is reported alongside the valley."""

    def test_valley_below_object_mode(self):
        scores = _bimodal_scores()
        modes = ground_modes(scores)
        assert modes is not None
        valley, object_mode = modes
        assert valley < object_mode
        assert object_mode == pytest.approx(0.85, abs=0.05)
        assert valley == ground_threshold(scores)

    def test_no_modes_for_monotonic_scores(self):
        assert ground_modes(_scores_from_histogram(list(range(1, 22)))) is None


class TestBoxWeights:
    """Test threshold-aware per-box weights."""

    def test_zero_at_threshold_one_at_mode(self):
        weights = box_weights(np.array([0.3, 0.45, 0.6, 0.9]), threshold=0.3, object_mode=0.6)
        np.testing.assert_allclose(weights, [0.0, 0.5, 1.0, 1.0], atol=1e-6)

    def test_monotonic_in_score(self):
        scores = np.linspace(0.2, 1.0, 50)
        weights = box_weights(scores, threshold=0.2, object_mode=0.7)
        assert np.all(np.diff(weights) >= 0)

    def test_invariant_to_shifted_distribution(self):
        # A more confident Teacher shifts threshold and mode together: relative position keeps its weight
        low = box_weights(np.array([0.35]), threshold=0.3, object_mode=0.5)
        high = box_weights(np.array([0.75]), threshold=0.7, object_mode=0.9)
        np.testing.assert_allclose(low, high, atol=1e-6)

    def test_degenerate_gap_gives_full_weight(self):
        weights = box_weights(np.array([0.5, 0.9]), threshold=0.5, object_mode=0.5)
        np.testing.assert_allclose(weights, [1.0, 1.0])


class TestSampleImageFiles:
    """Test per-epoch uniform image sampling."""

    @pytest.fixture
    def files(self):
        return [Path(f"frame_{i:05d}.jpg") for i in range(1000)]

    def test_deterministic_per_seed(self, files):
        assert sample_image_files(files, 100, seed=7) == sample_image_files(files, 100, seed=7)

    def test_differs_across_seeds_and_keeps_order(self, files):
        first = sample_image_files(files, 100, seed=42)
        second = sample_image_files(files, 100, seed=43)
        assert len(first) == 100
        assert len(set(first)) == 100
        assert first != second
        assert first == sorted(first)

    @pytest.mark.parametrize("limit", [None, 5000])
    def test_no_sampling_returns_all(self, files, limit):
        assert sample_image_files(files, limit, seed=0) == files


class _FakeTeacher:
    """One small (score 0.9) and one large (score 0.5) box per image, both flip-symmetric; counts calls."""

    def __init__(self):
        self.predict_calls = 0

    def predict(self, images, **kwargs):
        self.predict_calls += 1
        return [
            SimpleNamespace(
                orig_shape=image.shape[:2],
                boxes=Boxes(
                    torch.tensor([[123.0, 10.0, 133.0, 20.0, 0.9, 0.0], [28.0, 28.0, 228.0, 228.0, 0.5, 0.0]]),
                    image.shape[:2],
                ),
            )
            for image in images
        ]


class TestGeneratorSwitches:
    """Test the size-bin thresholding and multi-view switches of PseudoLabelGenerator."""

    @pytest.fixture
    def generate(self, tmp_path):
        images = tmp_path / "images"
        images.mkdir()
        for i in range(3):
            cv2.imwrite(str(images / f"frame_{i}.png"), np.zeros((256, 256, 3), np.uint8))

        def _generate(modes, **switches):
            teacher = _FakeTeacher()
            generator = PseudoLabelGenerator(teacher, batch_size=8, half=False, workers=1, **switches)
            with patch("ssod.pseudo_labeling.ground_modes", side_effect=modes) as mocked:
                out = generator.generate(images, tmp_path / "labels")
            return teacher, mocked, out

        return _generate

    def test_per_bin_thresholds_by_default(self, generate):
        _, mocked, out = generate([(0.3, 0.8)] * 3)
        assert mocked.call_count == 3
        assert len(out.filtered_df) == 6

    def test_shared_threshold_without_size_bins(self, generate):
        _, mocked, out = generate([(0.7, 0.9)], size_bins=False)
        assert mocked.call_count == 1
        assert len(mocked.call_args.args[0]) == 6  # every candidate, all bins
        assert set(out.thresholds.values()) == {0.7}
        # the large boxes (score 0.5) fall below the shared threshold
        assert out.filtered_df["size_bin"].unique().to_list() == ["small"]

    def test_single_view_skips_flip_inference(self, generate):
        multi, _, _ = generate([(0.3, 0.8)] * 3)
        single, _, out = generate([(0.3, 0.8)] * 3, multi_view=False)
        assert multi.predict_calls == 2
        assert single.predict_calls == 1
        assert len(out.filtered_df) == 6

    def test_input_scale_bins_follow_network_input(self, generate):
        # 256 px images at imgsz=64 are shrunk 4x: the 200 px box (large in original pixels)
        # becomes 50 px at the input, i.e. medium
        _, _, scaled = generate([(0.3, 0.8)] * 3, imgsz=64)
        _, _, original = generate([(0.3, 0.8)] * 3, imgsz=64, input_scale_bins=False)
        assert sorted(scaled.filtered_df["size_bin"].unique().to_list()) == ["medium", "small"]
        assert sorted(original.filtered_df["size_bin"].unique().to_list()) == ["large", "small"]


class TestSizeBins:
    """Test COCO size-bin assignment at the network input scale."""

    @pytest.mark.parametrize(
        ("width", "height", "expected"),
        [(4000, 3000, 0.16), (1280, 720, 0.5)],
    )
    def test_input_scale_uses_longer_side(self, width, height, expected):
        assert input_scale(width, height, 640) == pytest.approx(expected)

    def test_scale_moves_box_between_bins(self):
        # 100x100 px box on a 4000x3000 image: large in original pixels, 16x16 px at the input
        assert assign_size_bin(100, 100) == "large"
        assert assign_size_bin(100, 100, input_scale(4000, 3000, 640)) == "small"
        assert assign_size_bin(40, 40, 1.0) == "medium"


class TestFitBinThreshold:
    """Test the per-bin ground threshold and its fallback."""

    def test_valley_threshold(self):
        rng = np.random.default_rng(0)
        scores = np.concatenate([rng.uniform(0.0, 0.1, 2000), rng.uniform(0.75, 0.95, 400)])
        fitted = fit_bin_threshold(scores, conf_floor=0.0, fallback=0.4)
        assert fitted.from_valley
        assert 0.1 < fitted.threshold < 0.75
        assert fitted.object_mode > fitted.threshold

    def test_fallback_without_valley(self):
        scores = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        with patch("ssod.pseudo_labeling.ground_modes", return_value=None):
            fitted = fit_bin_threshold(scores, conf_floor=0.0, fallback=0.4)
        assert fitted == BinThreshold(0.4, float(np.percentile(scores, 90)), False)

    def test_fallback_object_mode_without_scores_above(self):
        fitted = fit_bin_threshold(np.array([0.1, 0.2]), conf_floor=0.0, fallback=0.4)
        assert (fitted.threshold, fitted.object_mode) == (0.4, 1.0)


class TestSelectPseudoLabels:
    """Test per-bin filtering and soft weights of the candidate table."""

    @pytest.fixture
    def candidates(self):
        return pl.DataFrame(
            {"size_bin": ["small", "small", "large", "large"], "score": [0.2, 0.6, 0.5, 0.9]},
            schema={"size_bin": pl.String, "score": pl.Float32},
        )

    @pytest.fixture
    def thresholds(self):
        return {
            "small": BinThreshold(0.3, 0.7, True),
            "large": BinThreshold(0.5, 0.9, True),
        }

    def test_filters_by_bin_threshold(self, candidates, thresholds):
        labels = select_pseudo_labels(candidates, thresholds, soft_weights=False)
        assert labels["score"].to_list() == [np.float32(0.6), np.float32(0.5), np.float32(0.9)]
        assert labels["weight"].to_list() == [1.0, 1.0, 1.0]

    def test_soft_weights(self, candidates, thresholds):
        labels = select_pseudo_labels(candidates, thresholds, soft_weights=True)
        np.testing.assert_allclose(labels["weight"].to_numpy(), [0.75, 0.0, 1.0], atol=1e-6)

    def test_no_thresholds_gives_empty_table(self, candidates):
        labels = select_pseudo_labels(candidates, {}, soft_weights=True)
        assert labels.is_empty()
        assert "weight" in labels.columns
