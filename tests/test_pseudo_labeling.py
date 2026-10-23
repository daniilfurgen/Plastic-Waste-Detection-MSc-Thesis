"""
Unit tests for online pseudo-labelling: score-histogram thresholding and its EMA, per-box
weights, multi-view candidates, per-step labelling, the histogram log and image sampling.
"""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from ssod.pseudo_labeling import (
    SIZE_BINS,
    BinThreshold,
    Detections,
    OnlinePseudoLabeler,
    PseudoLabelLog,
    ScoreHistograms,
    StepLabels,
    assign_size_bins,
    box_weights,
    erased_fraction,
    fit_bin_threshold,
    ground_modes,
    merge_views,
    sample_image_files,
    score_histogram,
    size_bin_index,
    truncate_histogram_log,
    unflip,
)


def _bimodal_hist(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.05, 0.05, 20000).clip(0.0, 1.0)
    objects = rng.normal(0.85, 0.05, 8000).clip(0.0, 1.0)
    return score_histogram(np.concatenate([noise, objects]))


class TestGroundModes:
    """Test valley detection between the noise and object modes of a score histogram."""

    def test_bimodal_returns_valley_below_object_mode(self):
        modes = ground_modes(_bimodal_hist())
        assert modes is not None
        valley, object_mode = modes
        assert 0.25 < valley < 0.7
        assert object_mode == pytest.approx(0.85, abs=0.05)

    def test_invariant_to_normalisation(self):
        hist = _bimodal_hist()
        assert ground_modes(hist) == ground_modes(hist / hist.sum())

    def test_object_mode_dominant(self):
        # Later SSL epochs: the object mode outgrows the noise mode
        rng = np.random.default_rng(1)
        noise = rng.normal(0.02, 0.03, 5000).clip(0.0, 1.0)
        objects = rng.normal(0.85, 0.03, 20000).clip(0.0, 1.0)
        modes = ground_modes(score_histogram(np.concatenate([noise, objects])))
        assert modes is not None
        assert 0.1 < modes[0] < 0.8

    def test_observed_epoch_one_medium_histogram(self):
        # Medium-bin candidate histogram over [0, 1] (21 bins) from run ssod-v2-100, epoch 1
        counts = np.array([
            17900, 1998, 1268, 951, 775, 714, 623, 573, 570, 577, 590,
            602, 644, 698, 779, 985, 1465, 2535, 3202, 1643, 0,
        ])
        modes = ground_modes(counts)
        assert modes is not None
        assert 0.3 <= modes[0] <= 0.5

    def test_clipped_monotonic_range_has_no_valley(self):
        # Scores restricted to [0.5, 1.0] rise monotonically: no valley must be reported
        rng = np.random.default_rng(2)
        hist = score_histogram(rng.triangular(0.5, 0.9, 0.9, 10000), lo=0.5)
        assert ground_modes(hist, lo=0.5) is None

    def test_empty_histogram(self):
        assert ground_modes(np.zeros(21)) is None


class TestFitBinThreshold:
    """Test the per-bin ground threshold and its fallback."""

    def test_valley_threshold(self):
        rng = np.random.default_rng(0)
        hist = score_histogram(np.concatenate([rng.uniform(0.0, 0.1, 2000), rng.uniform(0.75, 0.95, 400)]))
        fitted = fit_bin_threshold(hist, conf_floor=0.0, fallback=0.4)
        assert fitted.from_valley
        assert 0.1 < fitted.threshold < 0.75
        assert fitted.object_mode > fitted.threshold

    def test_fallback_object_mode_from_mass_above_it(self):
        hist = np.zeros(10)
        hist[[6, 8]] = [9, 1]  # centres 0.65 and 0.85 over [0, 1]
        with patch("ssod.pseudo_labeling.ground_modes", return_value=None):
            fitted = fit_bin_threshold(hist, conf_floor=0.0, fallback=0.4)
        assert fitted == BinThreshold(0.4, pytest.approx(0.65), False)

    def test_fallback_object_mode_without_mass_above(self):
        hist = np.zeros(10)
        hist[1] = 5
        fitted = fit_bin_threshold(hist, conf_floor=0.0, fallback=0.4)
        assert (fitted.threshold, fitted.object_mode, fitted.from_valley) == (0.4, 1.0, False)


class TestBoxWeights:
    """Test threshold-aware per-box weights."""

    def test_zero_at_threshold_one_at_mode(self):
        weights = box_weights(torch.tensor([0.3, 0.45, 0.6, 0.9]), torch.tensor(0.3), torch.tensor(0.6))
        torch.testing.assert_close(weights, torch.tensor([0.0, 0.5, 1.0, 1.0]))

    def test_per_box_thresholds(self):
        weights = box_weights(
            torch.tensor([0.35, 0.75]), torch.tensor([0.3, 0.7]), torch.tensor([0.5, 0.9])
        )
        torch.testing.assert_close(weights[0], weights[1])  # same relative position, same weight

    def test_degenerate_gap_gives_full_weight(self):
        weights = box_weights(torch.tensor([0.5, 0.9]), torch.tensor(0.5), torch.tensor(0.5))
        torch.testing.assert_close(weights, torch.tensor([1.0, 1.0]))


class TestSizeBins:
    """Test COCO size-bin assignment by box area."""

    def test_torch_matches_numpy(self):
        areas = np.array([0.0, 100.0, 32.0**2, 2000.0, 96.0**2, 1e6])
        names = np.array(list(SIZE_BINS))[size_bin_index(torch.tensor(areas)).numpy()]
        assert names.tolist() == assign_size_bins(areas).tolist()
        assert names.tolist() == ["small", "small", "medium", "medium", "large", "large"]


class TestScoreHistograms:
    """Test the EMA of normalised score histograms and its thresholds."""

    @staticmethod
    def _counts(small=None, medium=None, large=None, n_bins=21) -> np.ndarray:
        rows = [np.zeros(n_bins) if c is None else np.asarray(c, dtype=float) for c in (small, medium, large)]
        return np.stack(rows)

    def test_reset_normalises(self):
        hists = ScoreHistograms(conf_floor=0.0)
        hists.reset(self._counts(small=np.arange(21)))
        assert hists.densities["small"].sum() == pytest.approx(1.0)
        assert hists.densities["medium"].sum() == 0.0

    def test_update_is_ema_of_normalised_histograms(self):
        hists = ScoreHistograms(conf_floor=0.0)
        start = np.zeros(21)
        start[0] = 4
        hists.reset(self._counts(small=start, medium=start, large=start))
        batch = np.zeros(21)
        batch[20] = 10
        hists.update(self._counts(small=batch, large=batch), decay=0.9)

        expected = 0.9 * start / 4 + 0.1 * batch / 10
        np.testing.assert_allclose(hists.densities["small"], expected)
        np.testing.assert_allclose(hists.densities["large"], expected)
        # no medium candidates in the batch: the medium histogram is unchanged
        np.testing.assert_allclose(hists.densities["medium"], start / 4)

    def test_empty_histogram_starts_from_first_batch(self):
        hists = ScoreHistograms(conf_floor=0.0)
        batch = np.zeros(21)
        batch[3] = 2
        hists.update(self._counts(large=batch), decay=0.9996)
        np.testing.assert_allclose(hists.densities["large"], batch / 2)

    def test_shared_histogram_without_size_bins(self):
        hists = ScoreHistograms(conf_floor=0.0, size_bins=False)
        hists.reset(self._counts(small=_bimodal_hist(0), large=_bimodal_hist(1)))
        assert list(hists.densities) == ["all"]
        thresholds = hists.thresholds()
        assert list(thresholds) == list(SIZE_BINS)
        assert len(set(thresholds.values())) == 1
        assert thresholds["small"].from_valley

    def test_state_round_trip(self):
        hists = ScoreHistograms(conf_floor=0.0)
        hists.reset(self._counts(small=_bimodal_hist(), medium=np.arange(21)))
        restored = ScoreHistograms(conf_floor=0.0)
        restored.load_state_dict(hists.state_dict())
        for name in SIZE_BINS:
            np.testing.assert_array_equal(restored.densities[name], hists.densities[name])
        assert restored.thresholds() == hists.thresholds()


def _detections(boxes, scores, classes=None, image=None) -> Detections:
    n = len(scores)
    return Detections(
        torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        torch.tensor(scores, dtype=torch.float32),
        torch.tensor(classes or [0] * n),
        torch.tensor(image or [0] * n),
    )


class TestViews:
    """Test flip un-mapping and view merging."""

    def test_unflip_maps_boxes_back(self):
        flipped = unflip(_detections([[10.0, 5.0, 30.0, 25.0]], [0.9]), width=100)
        torch.testing.assert_close(flipped.boxes, torch.tensor([[70.0, 5.0, 90.0, 25.0]]))

    def test_merge_views_per_image_and_class(self):
        box = [10.0, 10.0, 50.0, 50.0]
        normal = _detections([box, box], [0.9, 0.8], classes=[0, 1], image=[0, 0])
        flipped = _detections([box, box], [0.7, 0.6], classes=[0, 0], image=[0, 1])
        merged = merge_views([normal, flipped], iou_threshold=0.5)
        # image 0 class 0 keeps the higher score; class 1 and image 1 are separate groups
        assert sorted(merged.scores.tolist()) == pytest.approx([0.6, 0.8, 0.9])


class TestErasedFraction:
    """Test the share of each box covered by erased patches."""

    def test_fraction_per_image(self):
        detections = _detections(
            [[0, 0, 10, 10], [0, 0, 10, 10], [20, 20, 30, 30]], [0.9, 0.9, 0.9], image=[0, 1, 0]
        )
        patches = torch.tensor([[0.0, 0, 0, 10, 5]])  # top half of the first box, image 0 only
        fraction = erased_fraction(detections, patches, (64, 64))
        torch.testing.assert_close(fraction, torch.tensor([0.5, 0.0, 0.0]))


class _FakeTeacher(nn.Module):
    """
    E2E-style Teacher: on 256 px images, a flip-symmetric small box (score 0.9), medium box
    (score 0.2) and large box (score 0.5) per image. Counts forward calls.
    """

    end2end = True

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        rows = torch.tensor([
            [123.0, 10.0, 133.0, 20.0, 0.9, 0.0],
            [108.0, 100.0, 148.0, 140.0, 0.2, 0.0],
            [28.0, 28.0, 228.0, 228.0, 0.5, 0.0],
        ], device=images.device)
        return rows.expand(len(images), -1, -1), {}


THRESHOLDS = {
    "small": BinThreshold(0.3, 0.8, True),
    "medium": BinThreshold(0.3, 0.8, True),
    "large": BinThreshold(0.4, 0.6, True),
}


def _batch(erase_boxes=None) -> dict:
    """GT image 0 with one box, unlabeled images 1 and 2."""
    return {
        "img": torch.zeros(3, 3, 256, 256),
        "teacher_img": torch.zeros(2, 3, 256, 256, dtype=torch.uint8),
        "is_pseudo": torch.tensor([False, True, True]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
        "cls": torch.tensor([[0.0]]),
        "batch_idx": torch.tensor([0.0]),
        "pl_weight": torch.tensor([1.0]),
        "erase_boxes": torch.zeros((0, 5)) if erase_boxes is None else erase_boxes,
    }


class TestOnlinePseudoLabeler:
    """Test per-step labelling of a batch's unlabeled images."""

    def test_appends_kept_boxes_to_their_images(self):
        batch = _batch()
        step = OnlinePseudoLabeler().label(_FakeTeacher(), batch, THRESHOLDS)

        assert batch["batch_idx"].tolist() == [0.0, 1.0, 1.0, 2.0, 2.0]  # sorted by image, as the loss expects
        assert len(batch["bboxes"]) == len(batch["cls"]) == len(batch["pl_weight"]) == 5
        # the large box (200 px at 256 px) in normalised xywh
        large = batch["bboxes"][1:][batch["pl_weight"][1:] < 1]
        torch.testing.assert_close(large, torch.tensor([[0.5, 0.5, 200 / 256, 200 / 256]] * 2))
        # soft weights: small 0.9 >= mode 0.8 -> 1; large (0.5 - 0.4) / 0.2 = 0.5; GT keeps 1
        assert sorted(batch["pl_weight"].tolist()) == pytest.approx([0.5, 0.5, 1.0, 1.0, 1.0])

        assert step.kept.tolist() == [2, 0, 2]
        assert step.labeled_images == 2
        # the histograms include the rejected medium candidates
        assert step.counts.sum(axis=1).tolist() == [2, 2, 2]

    def test_drops_mostly_erased_boxes(self):
        batch = _batch(erase_boxes=torch.tensor([[1.0, 20, 20, 236, 236]]))  # covers image 1's large box
        step = OnlinePseudoLabeler().label(_FakeTeacher(), batch, THRESHOLDS)
        assert batch["batch_idx"].tolist() == [0.0, 1.0, 2.0, 2.0]
        assert step.kept.tolist() == [2, 0, 1]
        assert step.counts.sum() == 6  # erased candidates still count in the histograms

    def test_hard_labels_without_soft_weights(self):
        batch = _batch()
        OnlinePseudoLabeler(soft_weights=False).label(_FakeTeacher(), batch, THRESHOLDS)
        assert batch["pl_weight"].tolist() == [1.0] * 5

    def test_single_view_runs_one_forward(self):
        multi, single = _FakeTeacher(), _FakeTeacher()
        OnlinePseudoLabeler().label(multi, _batch(), THRESHOLDS)
        OnlinePseudoLabeler(multi_view=False).label(single, _batch(), THRESHOLDS)
        assert (multi.calls, single.calls) == (2, 1)

    def test_batch_without_unlabeled_images_is_unchanged(self):
        batch = _batch()
        batch["is_pseudo"] = torch.tensor([False, False, False])
        teacher = _FakeTeacher()
        step = OnlinePseudoLabeler().label(teacher, batch, THRESHOLDS)
        assert teacher.calls == 0
        assert batch["batch_idx"].tolist() == [0.0]
        assert step.counts.sum() == 0 and step.kept.sum() == 0


class TestPseudoLabelLog:
    """Test the per-epoch summary columns and the histogram log."""

    @pytest.fixture
    def log(self):
        log = PseudoLabelLog()
        counts = np.zeros((3, 21), dtype=np.int64)
        counts[0, 2] = 5
        log.add(StepLabels(counts, np.array([3, 1, 0]), np.array([1.5, 1.0, 0.0]), 2), THRESHOLDS)
        log.add(StepLabels(counts, np.array([1, 0, 0]), np.array([0.5, 0.0, 0.0]), 1), THRESHOLDS)
        return log

    def test_columns(self, log):
        columns = log.columns()
        assert columns["pl_images"] == 3
        assert columns["pl_boxes"] == 5
        assert columns["pl_share/small"] == pytest.approx(0.8)
        assert columns["pl_weight/small"] == pytest.approx(0.5)
        assert columns["pl_weight/large"] == 0.0
        assert columns["threshold/large"] == pytest.approx(0.4)
        assert columns["object_mode/large"] == pytest.approx(0.6)

    def test_save_and_truncate(self, log, tmp_path):
        hists = ScoreHistograms(conf_floor=0.0)
        for epoch in (1, 2):
            log.save(tmp_path, epoch, hists)

        with open(tmp_path / "histograms.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2 * len(SIZE_BINS) * 21
        small = [r for r in rows if r["epoch"] == "1" and r["size_bin"] == "small"]
        assert int(small[2]["counts"]) == 10  # summed over the epoch's steps
        assert float(small[0]["bin_lo"]) == 0.0 and float(small[-1]["bin_hi"]) == 1.0

        steps = np.load(tmp_path / "histograms" / "epoch_1.npz")
        assert steps["counts"].shape == (2, 3, 21)
        assert steps["thresholds"].shape == steps["kept"].shape == (2, 3)

        truncate_histogram_log(tmp_path, 1)
        with open(tmp_path / "histograms.csv", encoding="utf-8") as f:
            assert {r["epoch"] for r in csv.DictReader(f)} == {"1"}
        assert not (tmp_path / "histograms" / "epoch_2.npz").exists()


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
