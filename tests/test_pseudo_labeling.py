"""
Unit tests for pseudo-label score thresholding.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

# Ensure code directory is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from ssod.pseudo_labeling import ground_threshold  # noqa: E402


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
