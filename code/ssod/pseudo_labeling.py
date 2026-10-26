"""
Online pseudo-labelling for the Mean Teacher pipeline (Size-Stratified Ground Thresholding).

At every training step the Teacher labels the unlabeled images of the batch:

1. Multi-view inference: the Teacher predicts on the weak view of each image (same geometry as
   the Student's view, without photometric augmentation) and on its horizontal flip; flipped
   boxes are mapped back and both views are merged with class-wise NMS.
2. Size-stratified ground thresholding: candidates are split into COCO size bins by their area
   in the batch image (the network input scale). Each bin keeps an EMA of its normalised score
   histogram, and its threshold is the valley between the noise and object modes of it.
3. Soft weights: each kept box is weighted by how far its score lies between its bin's threshold
   (weight 0) and object mode (weight 1).

The histograms start from the candidates of a uniform image sample after the warm-up and are
updated after every step with the normalised histograms of all the batch's candidates, kept or
not: p'_t = decay * p'_{t-1} + (1 - decay) * p_t.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torchvision.ops import batched_nms  # type: ignore[import-untyped]
from ultralytics.utils.nms import non_max_suppression


# ---------------------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------------------

# COCO object size bins by box area in px² (set on images whose longer side is about 640 px)
SIZE_BINS: dict[str, tuple[float, float]] = {
    "small": (0.0, 32.0**2),
    "medium": (32.0**2, 96.0**2),
    "large": (96.0**2, float("inf")),
}
SHARED_HISTOGRAM = "all"  # name of the single histogram used when size bins are off
N_SCORE_BINS = 21
VIEW_NMS_IOU = 0.7  # Ultralytics' default NMS IoU within one view (models without an E2E head)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------------------------
# Size bins
# ---------------------------------------------------------------------------------------------


def assign_size_bins(areas: np.ndarray) -> np.ndarray:
    """COCO size bin name for each box area (px²)."""
    names = np.array(list(SIZE_BINS))
    upper_limits = [hi for _, hi in SIZE_BINS.values()][:-1]
    return names[np.searchsorted(upper_limits, areas, side="right")]


def size_bin_index(areas: torch.Tensor) -> torch.Tensor:
    """Index into SIZE_BINS of each box area (px²)."""
    upper_limits = torch.tensor(
        [hi for _, hi in SIZE_BINS.values()][:-1], dtype=areas.dtype, device=areas.device
    )
    return torch.bucketize(areas, upper_limits, right=True)


# ---------------------------------------------------------------------------------------------
# Score-histogram ground thresholding and soft weights
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BinThreshold:
    """Score threshold of one size bin and the score of its object mode."""

    threshold: float
    object_mode: float
    from_valley: bool  # False: no valley was found and the fallback threshold is used


def bin_centres(n_bins: int, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    edges = np.linspace(lo, hi, n_bins + 1)
    return (edges[:-1] + edges[1:]) / 2.0


def score_histogram(
    scores: Sequence[float] | np.ndarray, lo: float = 0.0, n_bins: int = N_SCORE_BINS
) -> np.ndarray:
    """Candidate counts in `n_bins` equal score bins over [lo, 1]."""
    counts, _ = np.histogram(scores, bins=n_bins, range=(lo, 1.0))
    return counts


def ground_modes(
    hist: np.ndarray,
    lo: float = 0.0,
    hi: float = 1.0,
    smooth: int = 3,
) -> tuple[float, float] | None:
    """
    Locate the valley between the low-score (noise) mode and the high-score (object) mode of a
    score histogram (counts or a normalised density) spanning [lo, hi].

    The histogram must span the whole candidate score range: the noise mode sits near the
    candidate confidence floor, so clipping the range (e.g. to [0.5, 1.0]) removes the valley.
    The two highest local maxima of a moving-average-smoothed histogram are taken as the modes
    and the centres of the lowest bin between them (valley) and of the upper mode are returned.
    Returns None when there is no valley (fewer than two modes, or a flat valley).
    """
    counts = np.asarray(hist, dtype=np.float64)
    if counts.sum() <= 0:
        return None

    if smooth > 1:
        padded = np.pad(counts, smooth // 2, mode="edge")
        counts = np.convolve(padded, np.ones(smooth) / smooth, mode="valid")

    left_neighbours = np.concatenate(([-np.inf], counts[:-1]))
    right_neighbours = np.concatenate((counts[1:], [-np.inf]))
    peaks = np.flatnonzero(
        (counts > 0) & (counts >= left_neighbours) & (counts >= right_neighbours)
    )
    if len(peaks) < 2:
        return None

    first, second = sorted(peaks[np.argsort(counts[peaks])[-2:]])
    valley_index = int(np.argmin(counts[first : second + 1])) + first
    if counts[valley_index] >= min(counts[first], counts[second]):
        return None

    centres = bin_centres(len(counts), lo, hi)
    return float(centres[valley_index]), float(centres[second])


def fit_bin_threshold(hist: np.ndarray, conf_floor: float, fallback: float) -> BinThreshold:
    """
    Ground threshold of one size bin from its score histogram over [conf_floor, 1]. Without a
    valley the fixed `fallback` threshold is used and the object mode is approximated by the
    90th percentile of the histogram mass above it.
    """
    modes = ground_modes(hist, lo=conf_floor)
    if modes is not None:
        return BinThreshold(threshold=modes[0], object_mode=modes[1], from_valley=True)

    centres = bin_centres(len(hist), conf_floor)
    above = np.where(centres >= fallback, np.asarray(hist, dtype=np.float64), 0.0)
    if above.sum() <= 0:
        return BinThreshold(threshold=fallback, object_mode=1.0, from_valley=False)
    cdf = np.cumsum(above) / above.sum()
    object_mode = float(centres[np.searchsorted(cdf, 0.9)])
    return BinThreshold(threshold=fallback, object_mode=object_mode, from_valley=False)


def box_weights(scores: torch.Tensor, threshold: torch.Tensor, object_mode: torch.Tensor) -> torch.Tensor:
    """
    Threshold-aware pseudo-box weights: 0 at the size bin's threshold, rising linearly to 1
    at the bin's object mode, 1 above it. Normalising by the valley-to-mode gap keeps the
    weights independent of where the threshold sits and of how confident the Teacher is.
    `threshold` and `object_mode` are per box (or broadcastable scalars).
    """
    gap = object_mode - threshold
    ramp = ((scores - threshold) / gap.clamp(min=1e-3)).clamp(0.0, 1.0)
    return torch.where(gap < 1e-3, torch.ones_like(scores), ramp).float()


class ScoreHistograms:
    """
    EMA of the normalised candidate score histogram of each COCO size bin (one shared histogram
    when size bins are off), and the ground thresholds derived from it.
    """

    def __init__(
        self,
        conf_floor: float = 0.001,
        fallback: float = 0.4,
        size_bins: bool = True,
        n_bins: int = N_SCORE_BINS,
    ) -> None:
        self.conf_floor = conf_floor
        self.fallback = fallback
        self.size_bins = size_bins
        self.n_bins = n_bins
        self.names = list(SIZE_BINS) if size_bins else [SHARED_HISTOGRAM]
        self.densities = {name: np.zeros(n_bins) for name in self.names}

    def merge_counts(self, counts: np.ndarray) -> dict[str, np.ndarray]:
        """Per-histogram counts from (size bins x score bins) candidate counts."""
        if not self.size_bins:
            return {SHARED_HISTOGRAM: counts.sum(axis=0)}
        return dict(zip(self.names, counts))

    def reset(self, counts: np.ndarray) -> None:
        """Start the histograms from (size bins x score bins) candidate counts."""
        for name, c in self.merge_counts(counts).items():
            total = c.sum()
            self.densities[name] = c / total if total > 0 else np.zeros(self.n_bins)

    def update(self, counts: np.ndarray, decay: float) -> None:
        """
        EMA step with the normalised histograms of a batch's candidates. A histogram whose batch
        has no candidates keeps its previous vector; an empty one is started from the batch's.
        """
        for name, c in self.merge_counts(counts).items():
            total = c.sum()
            if total == 0:
                continue
            density = c / total
            previous = self.densities[name]
            self.densities[name] = (
                density if previous.sum() == 0 else decay * previous + (1.0 - decay) * density
            )

    def thresholds(self) -> dict[str, BinThreshold]:
        """Ground threshold of every size bin (the shared one for all bins when size bins are off)."""
        fitted = {
            name: fit_bin_threshold(d, self.conf_floor, self.fallback)
            for name, d in self.densities.items()
        }
        if not self.size_bins:
            return {size_bin: fitted[SHARED_HISTOGRAM] for size_bin in SIZE_BINS}
        return fitted

    def state_dict(self) -> dict[str, Any]:
        return {name: d.tolist() for name, d in self.densities.items()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.densities = {name: np.asarray(state[name], dtype=np.float64) for name in self.names}


# ---------------------------------------------------------------------------------------------
# Multi-view candidate inference
# ---------------------------------------------------------------------------------------------


@dataclass
class Detections:
    """Candidate boxes (x1, y1, x2, y2 in batch-image pixels) of several images."""

    boxes: torch.Tensor
    scores: torch.Tensor
    classes: torch.Tensor
    image: torch.Tensor  # index of the image each box belongs to

    def __len__(self) -> int:
        return len(self.scores)

    @property
    def areas(self) -> torch.Tensor:
        return (self.boxes[:, 2] - self.boxes[:, 0]) * (self.boxes[:, 3] - self.boxes[:, 1])

    def __getitem__(self, keep: torch.Tensor) -> Detections:
        return Detections(self.boxes[keep], self.scores[keep], self.classes[keep], self.image[keep])


def detections_from_outputs(outputs: list[torch.Tensor]) -> Detections:
    """Flatten per-image (n, 6) [x1, y1, x2, y2, score, class] predictions."""
    image = torch.cat(
        [torch.full((len(o),), i, dtype=torch.long, device=o.device) for i, o in enumerate(outputs)]
    )
    rows = torch.cat(outputs).float()
    return Detections(rows[:, :4], rows[:, 4], rows[:, 5].long(), image)


def unflip(detections: Detections, width: int) -> Detections:
    """Map boxes predicted on a horizontally flipped view back to the original view."""
    boxes = detections.boxes.clone()
    boxes[:, [0, 2]] = width - detections.boxes[:, [2, 0]]
    return Detections(boxes, detections.scores, detections.classes, detections.image)


def merge_views(views: Sequence[Detections], iou_threshold: float) -> Detections:
    """Merge the detections of all views with NMS per image and class (highest scores first)."""
    merged = Detections(
        boxes=torch.cat([v.boxes for v in views]),
        scores=torch.cat([v.scores for v in views]),
        classes=torch.cat([v.classes for v in views]),
        image=torch.cat([v.image for v in views]),
    )
    if len(merged) == 0:
        return merged
    groups = merged.image * (int(merged.classes.max()) + 1) + merged.classes
    keep = batched_nms(merged.boxes, merged.scores, groups, iou_threshold)
    return merged[keep]


def erased_fraction(
    detections: Detections, erase_boxes: torch.Tensor, image_size: tuple[int, int]
) -> torch.Tensor:
    """
    Fraction of each box covered by the erased patches of its image. `erase_boxes` is (M, 5)
    [image, x1, y1, x2, y2] in batch-image pixels.
    """
    fraction = torch.zeros(len(detections), device=detections.boxes.device)
    if len(detections) == 0 or len(erase_boxes) == 0:
        return fraction
    height, width = image_size
    for image in erase_boxes[:, 0].long().unique().tolist():
        in_image = detections.image == image
        if not in_image.any():
            continue
        mask = torch.zeros((height + 1, width + 1), device=detections.boxes.device)
        for x1, y1, x2, y2 in erase_boxes[erase_boxes[:, 0] == image, 1:].round().long().tolist():
            mask[y1 + 1 : y2 + 1, x1 + 1 : x2 + 1] = 1.0
        integral = mask.cumsum(0).cumsum(1)
        boxes = detections.boxes[in_image].round().long()
        x1, x2 = boxes[:, 0].clamp(0, width), boxes[:, 2].clamp(0, width)
        y1, y2 = boxes[:, 1].clamp(0, height), boxes[:, 3].clamp(0, height)
        covered = integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
        area = ((x2 - x1) * (y2 - y1)).float()
        fraction[in_image] = torch.where(area > 0, covered / area.clamp(min=1.0), 0.0)
    return fraction


# ---------------------------------------------------------------------------------------------
# Per-step labelling
# ---------------------------------------------------------------------------------------------


@dataclass
class StepLabels:
    """Pseudo-labels of one training step, per size bin."""

    counts: np.ndarray  # (size bins, score bins): all candidates, kept or not
    kept: np.ndarray  # (size bins,): pseudo boxes added to the batch
    weight_sums: np.ndarray  # (size bins,): sum of their weights
    labeled_images: int  # unlabeled images with at least one pseudo box


class OnlinePseudoLabeler:
    """Teacher pseudo-labels for the unlabeled images of a training batch."""

    def __init__(
        self,
        conf: float = 0.001,
        iou_threshold: float = 0.5,
        multi_view: bool = True,
        soft_weights: bool = True,
        erase_drop_ratio: float = 0.5,
        n_bins: int = N_SCORE_BINS,
    ) -> None:
        self.conf = conf  # candidate confidence floor
        self.iou_threshold = iou_threshold  # NMS IoU for merging the views
        self.multi_view = multi_view  # False: normal view only
        self.soft_weights = soft_weights
        self.erase_drop_ratio = erase_drop_ratio  # drop pseudo boxes at least this fraction erased
        self.n_bins = n_bins

    @torch.no_grad()
    def candidates(self, model: torch.nn.Module, imgs: torch.Tensor) -> Detections:
        """Teacher candidates of uint8 (B, 3, H, W) images on the normal (and flipped) view."""
        images = imgs.float() / 255.0
        views = [self._predict(model, images)]
        if self.multi_view:
            views.append(unflip(self._predict(model, images.flip(-1)), images.shape[-1]))
            return merge_views(views, self.iou_threshold)
        return views[0]

    def _predict(self, model: torch.nn.Module, images: torch.Tensor) -> Detections:
        with torch.autocast(images.device.type, enabled=images.device.type == "cuda"):
            preds = model(images)
        outputs = non_max_suppression(
            preds,
            conf_thres=self.conf,
            iou_thres=VIEW_NMS_IOU,
            end2end=getattr(model, "end2end", False),
        )
        return detections_from_outputs(outputs)

    def count(self, detections: Detections) -> np.ndarray:
        """(size bins x score bins) counts of candidate scores over [conf, 1]."""
        bins = size_bin_index(detections.areas)
        return np.stack([
            torch.histc(detections.scores[bins == b], bins=self.n_bins, min=self.conf, max=1.0)
            .cpu()
            .numpy()
            for b in range(len(SIZE_BINS))
        ]).astype(np.int64)

    def label(
        self,
        model: torch.nn.Module,
        batch: dict[str, Any],
        thresholds: dict[str, BinThreshold],
    ) -> StepLabels:
        """
        Pseudo-label the batch's unlabeled images (`is_pseudo`) from their weak views
        (`teacher_img`) and append the boxes kept by their size bin's threshold to the batch
        targets (`bboxes`, `cls`, `batch_idx`, `pl_weight`), dropping boxes mostly erased in the
        Student view. Returns the step's candidate histograms and pseudo-label counts.
        """
        n_size_bins = len(SIZE_BINS)
        rows = torch.where(batch["is_pseudo"])[0]
        if len(rows) == 0:
            return StepLabels(
                np.zeros((n_size_bins, self.n_bins), dtype=np.int64),
                np.zeros(n_size_bins, dtype=np.int64),
                np.zeros(n_size_bins),
                0,
            )

        detections = self.candidates(model, batch["teacher_img"])
        counts = self.count(detections)

        bins = size_bin_index(detections.areas)
        device = detections.scores.device
        threshold = torch.tensor([thresholds[b].threshold for b in SIZE_BINS], device=device)[bins]
        object_mode = torch.tensor([thresholds[b].object_mode for b in SIZE_BINS], device=device)[bins]
        keep = detections.scores >= threshold

        height, width = batch["img"].shape[-2:]
        erase_boxes = batch.get("erase_boxes")
        if erase_boxes is not None and len(erase_boxes) > 0:
            # Patches are indexed by batch row, detections by position among the unlabeled rows
            position = torch.full((len(batch["is_pseudo"]),), -1, dtype=torch.long, device=device)
            position[rows.to(device)] = torch.arange(len(rows), device=device)
            patches = erase_boxes.to(device).clone()
            patches[:, 0] = position[patches[:, 0].long()].float()
            patches = patches[patches[:, 0] >= 0]
            keep &= erased_fraction(detections, patches, (height, width)) < self.erase_drop_ratio

        kept = detections[keep]
        bins, threshold, object_mode = bins[keep], threshold[keep], object_mode[keep]
        weights = (
            box_weights(kept.scores, threshold, object_mode)
            if self.soft_weights
            else torch.ones_like(kept.scores)
        )
        self._append_targets(batch, kept, rows, weights, (height, width))

        return StepLabels(
            counts=counts,
            kept=torch.bincount(bins, minlength=n_size_bins).cpu().numpy(),
            weight_sums=torch.bincount(bins, weights=weights, minlength=n_size_bins).cpu().numpy(),
            labeled_images=int(kept.image.unique().numel()),
        )

    @staticmethod
    def _append_targets(
        batch: dict[str, Any],
        kept: Detections,
        rows: torch.Tensor,
        weights: torch.Tensor,
        image_size: tuple[int, int],
    ) -> None:
        """
        Append pseudo boxes (normalised xywh) to the batch targets of their images, keeping the
        targets sorted by image as Ultralytics' loss preprocessing expects.
        """
        height, width = image_size
        x1, y1, x2, y2 = kept.boxes.clamp(min=0).unbind(1)
        x1, x2 = x1.clamp(max=width), x2.clamp(max=width)
        y1, y2 = y1.clamp(max=height), y2.clamp(max=height)
        xywh = torch.stack(
            [(x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height],
            dim=1,
        )
        device = batch["bboxes"].device
        batch["bboxes"] = torch.cat([batch["bboxes"], xywh.to(device, batch["bboxes"].dtype)])
        batch["cls"] = torch.cat(
            [batch["cls"], kept.classes.view(-1, 1).to(device, batch["cls"].dtype)]
        )
        batch["batch_idx"] = torch.cat(
            [batch["batch_idx"], rows.to(kept.image.device)[kept.image].to(device, batch["batch_idx"].dtype)]
        )
        batch["pl_weight"] = torch.cat([batch["pl_weight"], weights.to(device, batch["pl_weight"].dtype)])
        order = torch.argsort(batch["batch_idx"], stable=True)
        for key in ("bboxes", "cls", "batch_idx", "pl_weight"):
            batch[key] = batch[key][order]


# ---------------------------------------------------------------------------------------------
# Per-epoch pseudo-label log
# ---------------------------------------------------------------------------------------------


@dataclass
class PseudoLabelLog:
    """
    Pseudo-label statistics and histograms of one epoch: summary columns for results.csv,
    the end-of-epoch histograms for histograms.csv and every step's raw candidate counts,
    pseudo-label counts and thresholds for histograms/epoch_<k>.npz.
    """

    counts: list[np.ndarray] = field(default_factory=list)
    kept: list[np.ndarray] = field(default_factory=list)
    weight_sums: list[np.ndarray] = field(default_factory=list)
    thresholds: list[np.ndarray] = field(default_factory=list)
    object_modes: list[np.ndarray] = field(default_factory=list)
    labeled_images: int = 0

    def add(self, step: StepLabels, thresholds: dict[str, BinThreshold]) -> None:
        self.counts.append(step.counts)
        self.kept.append(step.kept)
        self.weight_sums.append(step.weight_sums)
        self.thresholds.append(np.array([thresholds[b].threshold for b in SIZE_BINS]))
        self.object_modes.append(np.array([thresholds[b].object_mode for b in SIZE_BINS]))
        self.labeled_images += step.labeled_images

    def columns(self) -> dict[str, float]:
        """Summary columns: labeled images, boxes, per-bin share and mean weight, mean thresholds."""
        n_size_bins = len(SIZE_BINS)
        kept = np.sum(self.kept, axis=0) if self.kept else np.zeros(n_size_bins)
        weight_sums = np.sum(self.weight_sums, axis=0) if self.weight_sums else np.zeros(n_size_bins)
        thresholds = np.mean(self.thresholds, axis=0) if self.thresholds else np.zeros(n_size_bins)
        object_modes = np.mean(self.object_modes, axis=0) if self.object_modes else np.zeros(n_size_bins)
        columns: dict[str, float] = {
            "pl_images": float(self.labeled_images),
            "pl_boxes": float(kept.sum()),
        }
        for i, b in enumerate(SIZE_BINS):
            columns[f"threshold/{b}"] = float(thresholds[i])
        for i, b in enumerate(SIZE_BINS):
            columns[f"object_mode/{b}"] = float(object_modes[i])
        for i, b in enumerate(SIZE_BINS):
            columns[f"pl_share/{b}"] = float(kept[i] / max(1.0, kept.sum()))
        for i, b in enumerate(SIZE_BINS):
            columns[f"pl_weight/{b}"] = float(weight_sums[i] / kept[i]) if kept[i] else 0.0
        return columns

    def save(self, exp_dir: Path, epoch: int, histograms: ScoreHistograms) -> None:
        """Append the epoch's histograms to histograms.csv and save its per-step arrays."""
        n_size_bins = len(SIZE_BINS)
        epoch_counts = (
            np.sum(self.counts, axis=0)
            if self.counts
            else np.zeros((n_size_bins, histograms.n_bins), dtype=np.int64)
        )
        append_histograms(exp_dir, epoch, histograms, epoch_counts)
        steps_dir = exp_dir / "histograms"
        steps_dir.mkdir(parents=True, exist_ok=True)
        empty = np.zeros((0, n_size_bins))
        np.savez_compressed(
            steps_dir / f"epoch_{epoch}.npz",
            size_bins=np.array(list(SIZE_BINS)),
            counts=np.asarray(self.counts, dtype=np.int32).reshape(-1, n_size_bins, histograms.n_bins),
            kept=np.asarray(self.kept, dtype=np.int32).reshape(-1, n_size_bins),
            thresholds=np.asarray(self.thresholds).reshape(-1, n_size_bins) if self.thresholds else empty,
            object_modes=np.asarray(self.object_modes).reshape(-1, n_size_bins) if self.object_modes else empty,
        )


HISTOGRAM_FIELDS = ["epoch", "size_bin", "bin", "bin_lo", "bin_hi", "ema", "counts"]


def append_histograms(
    exp_dir: Path, epoch: int, histograms: ScoreHistograms, counts: np.ndarray
) -> None:
    """
    Append one epoch to histograms.csv: each histogram's EMA density and the raw candidate
    counts (size bins x score bins, merged like the histograms) behind it in that epoch.
    """
    path = exp_dir / "histograms.csv"
    write_header = not path.is_file() or path.stat().st_size == 0
    edges = np.linspace(histograms.conf_floor, 1.0, histograms.n_bins + 1)
    merged = histograms.merge_counts(counts)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(HISTOGRAM_FIELDS)
        for name in histograms.names:
            for i in range(histograms.n_bins):
                writer.writerow([
                    epoch, name, i, f"{edges[i]:.6f}", f"{edges[i + 1]:.6f}",
                    f"{histograms.densities[name][i]:.8g}", int(merged[name][i]),
                ])


def truncate_histogram_log(exp_dir: Path, last_epoch: int) -> None:
    """Drop histograms.csv rows and step files of epochs after `last_epoch` (on resume)."""
    path = exp_dir / "histograms.csv"
    if path.is_file():
        with open(path, encoding="utf-8", newline="") as f:
            rows = [row for row in csv.DictReader(f) if int(row["epoch"]) <= last_epoch]
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=HISTOGRAM_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    for step_file in (exp_dir / "histograms").glob("epoch_*.npz"):
        if int(step_file.stem.removeprefix("epoch_")) > last_epoch:
            step_file.unlink()


# ---------------------------------------------------------------------------------------------
# Unlabeled image pool
# ---------------------------------------------------------------------------------------------


def list_images(images_dir: Path) -> list[Path]:
    """All image files under `images_dir`, sorted."""
    return sorted(
        p for p in Path(images_dir).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def sample_image_files(
    image_files: list[Path],
    sample_images: int | None,
    seed: int,
) -> list[Path]:
    """
    Uniform random sample (without replacement, original order kept) of `sample_images` files.
    Sampling whole images uniformly keeps the pseudo-box size-bin distribution unbiased.
    A sample larger than the pool takes whole passes over it plus a uniform remainder, so every
    image appears k or k + 1 times. Returns all files when `sample_images` is None or not positive.
    """
    if sample_images is None or sample_images <= 0 or not image_files:
        return image_files
    passes, remainder = divmod(sample_images, len(image_files))
    rng = np.random.default_rng(seed)
    chosen = np.sort(np.concatenate([
        np.repeat(np.arange(len(image_files)), passes),
        rng.choice(len(image_files), size=remainder, replace=False),
    ]))
    return [image_files[i] for i in chosen]
