"""
Unit tests for pseudo-label score thresholding, per-box weights and image sampling.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
from ultralytics.engine.results import Boxes

# Ensure code directory is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from ssod.pseudo_labeling import (  # noqa: E402
    PseudoLabelGenerator,
    assign_size_bin,
    box_weights,
    ground_modes,
    ground_threshold,
    input_scale,
    sample_image_files,
)


def _scores_from_histogram(counts: list[int], lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Expand per-bin counts into scores placed at each bin centre."""
    edges = np.linspace(lo, hi, len(counts) + 1)
    centres = (edges[:-1] + edges[1:]) / 2.0
    return np.repeat(centres, counts)


class TestGroundThreshold(unittest.TestCase):
    """Test valley detection between the noise and object score modes."""

    def test_bimodal_returns_valley(self):
        rng = np.random.default_rng(0)
        noise = rng.normal(0.05, 0.05, 20000).clip(0.0, 1.0)
        objects = rng.normal(0.85, 0.05, 8000).clip(0.0, 1.0)
        threshold = ground_threshold(np.concatenate([noise, objects]))
        self.assertIsNotNone(threshold)
        self.assertGreater(threshold, 0.25)
        self.assertLess(threshold, 0.7)

    def test_object_mode_dominant(self):
        # Later SSL epochs: the object mode outgrows the noise mode
        rng = np.random.default_rng(1)
        noise = rng.normal(0.02, 0.03, 5000).clip(0.0, 1.0)
        objects = rng.normal(0.85, 0.03, 20000).clip(0.0, 1.0)
        threshold = ground_threshold(np.concatenate([noise, objects]))
        self.assertIsNotNone(threshold)
        self.assertGreater(threshold, 0.1)
        self.assertLess(threshold, 0.8)

    def test_observed_epoch_one_medium_histogram(self):
        # Medium-bin candidate histogram over [0, 1] (21 bins) from run ssod-v2-100, epoch 1
        counts = [
            17900, 1998, 1268, 951, 775, 714, 623, 573, 570, 577, 590,
            602, 644, 698, 779, 985, 1465, 2535, 3202, 1643, 0,
        ]
        threshold = ground_threshold(_scores_from_histogram(counts))
        self.assertIsNotNone(threshold)
        self.assertGreaterEqual(threshold, 0.3)
        self.assertLessEqual(threshold, 0.5)

    def test_clipped_monotonic_range_has_no_valley(self):
        # The same scores restricted to [0.5, 1.0] rise monotonically: no valley must be reported
        rng = np.random.default_rng(2)
        scores = rng.triangular(0.5, 0.9, 0.9, 10000)
        self.assertIsNone(ground_threshold(scores, lo=0.5))

    def test_empty_scores(self):
        self.assertIsNone(ground_threshold(np.array([], dtype=float)))


if __name__ == "__main__":
    unittest.main()

class TestGroundModes(unittest.TestCase):
    """Test that the object mode is reported alongside the valley."""

    def test_valley_below_object_mode(self):
        rng = np.random.default_rng(0)
        noise = rng.normal(0.05, 0.05, 20000).clip(0.0, 1.0)
        objects = rng.normal(0.85, 0.05, 8000).clip(0.0, 1.0)
        modes = ground_modes(np.concatenate([noise, objects]))
        self.assertIsNotNone(modes)
        valley, object_mode = modes
        self.assertLess(valley, object_mode)
        self.assertAlmostEqual(object_mode, 0.85, delta=0.05)
        self.assertEqual(valley, ground_threshold(np.concatenate([noise, objects])))

    def test_no_modes_for_monotonic_scores(self):
        self.assertIsNone(ground_modes(_scores_from_histogram(list(range(1, 22)))))


class TestBoxWeights(unittest.TestCase):
    """Test threshold-aware per-box weights."""

    def test_zero_at_threshold_one_at_mode(self):
        weights = box_weights(np.array([0.3, 0.45, 0.6, 0.9]), threshold=0.3, object_mode=0.6)
        np.testing.assert_allclose(weights, [0.0, 0.5, 1.0, 1.0], atol=1e-6)

    def test_monotonic_in_score(self):
        scores = np.linspace(0.2, 1.0, 50)
        weights = box_weights(scores, threshold=0.2, object_mode=0.7)
        self.assertTrue(np.all(np.diff(weights) >= 0))

    def test_invariant_to_shifted_distribution(self):
        # A more confident Teacher shifts threshold and mode together: relative position keeps its weight
        low = box_weights(np.array([0.35]), threshold=0.3, object_mode=0.5)
        high = box_weights(np.array([0.75]), threshold=0.7, object_mode=0.9)
        np.testing.assert_allclose(low, high, atol=1e-6)

    def test_degenerate_gap_gives_full_weight(self):
        weights = box_weights(np.array([0.5, 0.9]), threshold=0.5, object_mode=0.5)
        np.testing.assert_allclose(weights, [1.0, 1.0])


class TestSampleImageFiles(unittest.TestCase):
    """Test per-epoch uniform image sampling."""

    def setUp(self):
        self.files = [Path(f"frame_{i:05d}.jpg") for i in range(1000)]

    def test_deterministic_per_seed(self):
        self.assertEqual(
            sample_image_files(self.files, 100, seed=7),
            sample_image_files(self.files, 100, seed=7),
        )

    def test_differs_across_seeds_and_keeps_order(self):
        first = sample_image_files(self.files, 100, seed=42)
        second = sample_image_files(self.files, 100, seed=43)
        self.assertEqual(len(first), 100)
        self.assertEqual(len(set(first)), 100)
        self.assertNotEqual(first, second)
        self.assertEqual(first, sorted(first))

    def test_no_sampling_returns_all(self):
        self.assertEqual(sample_image_files(self.files, None, seed=0), self.files)
        self.assertEqual(sample_image_files(self.files, 5000, seed=0), self.files)


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


class TestGeneratorSwitches(unittest.TestCase):
    """Test the size-bin thresholding and multi-view switches of PseudoLabelGenerator."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.images = self.test_dir / "images"
        self.images.mkdir()
        for i in range(3):
            cv2.imwrite(str(self.images / f"frame_{i}.png"), np.zeros((256, 256, 3), np.uint8))

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _generate(self, modes, **switches):
        teacher = _FakeTeacher()
        generator = PseudoLabelGenerator(teacher, batch_size=8, half=False, workers=1, **switches)
        with patch("ssod.pseudo_labeling.ground_modes", side_effect=modes) as mocked:
            out = generator.generate(self.images, self.test_dir / "labels")
        return teacher, mocked, out

    def test_per_bin_thresholds_by_default(self):
        _, mocked, out = self._generate([(0.3, 0.8), (0.3, 0.8), (0.3, 0.8)])
        self.assertEqual(mocked.call_count, 3)
        self.assertEqual(len(out["filtered_df"]), 6)

    def test_shared_threshold_without_size_bins(self):
        _, mocked, out = self._generate([(0.7, 0.9)], size_bins=False)
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(len(mocked.call_args.args[0]), 6)  # every candidate, all bins
        self.assertEqual(set(out["thresholds"].values()), {0.7})
        # the large boxes (score 0.5) fall below the shared threshold
        self.assertEqual(out["filtered_df"]["size_bin"].unique().to_list(), ["small"])

    def test_single_view_skips_flip_inference(self):
        multi, _, _ = self._generate([(0.3, 0.8)] * 3)
        single, _, out = self._generate([(0.3, 0.8)] * 3, multi_view=False)
        self.assertEqual(multi.predict_calls, 2)
        self.assertEqual(single.predict_calls, 1)
        self.assertEqual(len(out["filtered_df"]), 6)

    def test_input_scale_bins_follow_network_input(self):
        # 256 px images at imgsz=64 are shrunk 4x: the 200 px box (large in original pixels)
        # becomes 50 px at the input, i.e. medium
        _, _, scaled = self._generate([(0.3, 0.8)] * 3, imgsz=64)
        _, _, original = self._generate([(0.3, 0.8)] * 3, imgsz=64, input_scale_bins=False)
        self.assertEqual(sorted(scaled["filtered_df"]["size_bin"].unique().to_list()), ["medium", "small"])
        self.assertEqual(sorted(original["filtered_df"]["size_bin"].unique().to_list()), ["large", "small"])


class TestSizeBins(unittest.TestCase):
    """Test COCO size-bin assignment at the network input scale."""

    def test_input_scale_uses_longer_side(self):
        self.assertAlmostEqual(input_scale(4000, 3000, 640), 0.16)
        self.assertAlmostEqual(input_scale(1280, 720, 640), 0.5)

    def test_scale_moves_box_between_bins(self):
        # 100x100 px box on a 4000x3000 image: large in original pixels, 16x16 px at the input
        self.assertEqual(assign_size_bin(100, 100), "large")
        self.assertEqual(assign_size_bin(100, 100, input_scale(4000, 3000, 640)), "small")
        self.assertEqual(assign_size_bin(40, 40, 1.0), "medium")

