"""
Pseudo-label generation for the Mean Teacher pipeline.

For a set of unlabeled images the Teacher model produces candidate boxes with a very low confidence
floor, and the confident ones are kept as pseudo-labels:

1. Multi-view inference: the Teacher predicts on each image and on its horizontal flip; flipped
   boxes are mapped back and both views are merged with class-wise NMS.
2. Size-stratified ground thresholding: candidates are split into COCO size bins (box areas taken
   at the network input scale), and each bin gets its own score threshold at the valley between
   the noise and object modes of its score histogram.
3. Soft weights: each kept box is weighted by how far its score lies between its bin's threshold
   (weight 0) and object mode (weight 1).

Pseudo-labels are written as YOLO label files, with the weight as an optional 6th column.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np
import polars as pl
import torch
from torchvision.ops import batched_nms  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore[import-untyped]
import typer
from ultralytics import YOLO

try:
    from ssod.utilities import get_logger, input_scale
except ImportError:
    from utilities import get_logger, input_scale  # type: ignore[import-not-found,no-redef]


# ---------------------------------------------------------------------------------------------
# Constants and schemas
# ---------------------------------------------------------------------------------------------

# COCO object size bins by box area in px² (set on images whose longer side is about 640 px)
SIZE_BINS: dict[str, tuple[float, float]] = {
    "small": (0.0, 32.0**2),
    "medium": (32.0**2, 96.0**2),
    "large": (96.0**2, float("inf")),
}

# One row per candidate box; also the schema of the saved candidate Parquet files
CANDIDATE_SCHEMA: dict[str, Any] = {
    "image_stem": pl.String,
    "label_path": pl.String,
    "img_width": pl.Int32,
    "img_height": pl.Int32,
    "class_id": pl.Int32,
    "score": pl.Float32,
    "x1": pl.Float32,
    "y1": pl.Float32,
    "x2": pl.Float32,
    "y2": pl.Float32,
    "w": pl.Float32,
    "h": pl.Float32,
    "size_bin": pl.String,
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

app = typer.Typer(
    name="pseudo-label-generator",
    help="Generate pseudo-labels via multi-view inference and COCO size-bin ground thresholding.",
    add_completion=False,
)


# ---------------------------------------------------------------------------------------------
# Input scale and size bins
# ---------------------------------------------------------------------------------------------


def assign_size_bins(areas: np.ndarray) -> np.ndarray:
    """COCO size bin name for each box area (px²)."""
    names = np.array(list(SIZE_BINS))
    upper_limits = [hi for _, hi in SIZE_BINS.values()][:-1]
    return names[np.searchsorted(upper_limits, areas, side="right")]


def assign_size_bin(w: float, h: float, scale: float = 1.0) -> str:
    """
    COCO size bin of a box. COCO's bin limits were set on images whose longer side is about 640 px,
    so `scale` maps the box to the network input (see `input_scale`) before binning; with
    scale=1 the bin is taken in original image pixels.
    """
    return str(assign_size_bins(np.array([w * h * scale**2]))[0])


# ---------------------------------------------------------------------------------------------
# Score-histogram ground thresholding and soft weights
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BinThreshold:
    """Score threshold of one size bin and the score of its object mode."""

    threshold: float
    object_mode: float
    from_valley: bool  # False: no valley was found and the fallback threshold is used


def ground_modes(
    scores: Sequence[float] | np.ndarray,
    lo: float = 0.0,
    hi: float = 1.0,
    n_bins: int = 21,
    smooth: int = 3,
) -> tuple[float, float] | None:
    """
    Locate the valley between the low-score (noise) mode and the high-score (object) mode.

    The histogram must span the whole candidate score range: the noise mode sits near the
    candidate confidence floor, so clipping the range (e.g. to [0.5, 1.0]) removes the valley.
    The two highest local maxima of a moving-average-smoothed histogram are taken as the modes
    and the centres of the lowest bin between them (valley) and of the upper mode are returned.
    Returns None when there is no valley (fewer than two modes, or a flat valley).
    """
    hist, bin_edges = np.histogram(scores, bins=n_bins, range=(lo, hi))
    if hist.sum() == 0:
        return None

    counts = hist.astype(np.float64)
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

    centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    return float(centres[valley_index]), float(centres[second])


def ground_threshold(
    scores: Sequence[float] | np.ndarray,
    lo: float = 0.0,
    hi: float = 1.0,
    n_bins: int = 21,
    smooth: int = 3,
) -> float | None:
    """Valley between the noise and object modes (see `ground_modes`), or None if there is none."""
    modes = ground_modes(scores, lo=lo, hi=hi, n_bins=n_bins, smooth=smooth)
    return None if modes is None else modes[0]


def fit_bin_threshold(scores: np.ndarray, conf_floor: float, fallback: float) -> BinThreshold:
    """
    Ground threshold of one size bin. Without a valley the fixed `fallback` threshold is used and
    the object mode is approximated by the 90th percentile of the scores above it.
    """
    modes = ground_modes(scores, lo=conf_floor)
    if modes is not None:
        return BinThreshold(threshold=modes[0], object_mode=modes[1], from_valley=True)

    above = scores[scores >= fallback]
    object_mode = float(np.percentile(above, 90)) if len(above) else 1.0
    return BinThreshold(threshold=fallback, object_mode=object_mode, from_valley=False)


def box_weights(
    scores: np.ndarray,
    threshold: float,
    object_mode: float,
) -> np.ndarray:
    """
    Threshold-aware pseudo-box weights: 0 at the size bin's threshold, rising linearly to 1
    at the bin's object mode, 1 above it. Normalising by the valley-to-mode gap keeps the
    weights independent of where the threshold sits and of how confident the Teacher is.
    """
    gap = object_mode - threshold
    if gap < 1e-3:
        return np.ones_like(scores, dtype=np.float32)
    return np.clip((scores - threshold) / gap, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------------------------
# Multi-view candidate inference
# ---------------------------------------------------------------------------------------------


@dataclass
class Detections:
    """Boxes (x1, y1, x2, y2 in image pixels), scores and class ids of one image."""

    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray

    def __len__(self) -> int:
        return len(self.scores)


def detections_from_result(result: Any, flipped: bool = False) -> Detections:
    """
    Teacher detections of one image view. Boxes predicted on a horizontally flipped view are
    mapped back to the original image. Coordinates are kept in float64.
    """
    if result.boxes is None or len(result.boxes) == 0:
        return Detections(np.empty((0, 4)), np.empty(0), np.empty(0, dtype=np.int64))

    boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float64)
    if flipped:
        width = float(result.orig_shape[1])
        boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
    return Detections(
        boxes=boxes,
        scores=result.boxes.conf.detach().cpu().numpy().astype(np.float64),
        classes=result.boxes.cls.detach().cpu().numpy().astype(np.int64),
    )


def merge_views(views: Sequence[Detections], iou_threshold: float) -> Detections:
    """Merge the detections of all views with class-wise NMS (highest scores first)."""
    boxes = np.concatenate([view.boxes for view in views])
    scores = np.concatenate([view.scores for view in views])
    classes = np.concatenate([view.classes for view in views])
    if len(scores) == 0:
        return Detections(boxes, scores, classes)

    keep = batched_nms(
        torch.from_numpy(boxes).float(),
        torch.from_numpy(scores).float(),
        torch.from_numpy(classes),
        iou_threshold,
    ).numpy()
    return Detections(boxes[keep], scores[keep], classes[keep])


def candidate_frame(
    detections: Detections,
    image_path: Path,
    label_path: Path,
    image_size: tuple[int, int],
    bin_scale: float,
) -> pl.DataFrame:
    """Candidate rows of one image; `bin_scale` maps box areas to the scale they are binned at."""
    img_width, img_height = image_size
    x1, y1, x2, y2 = detections.boxes.T
    w, h = x2 - x1, y2 - y1
    return pl.DataFrame(
        {
            "image_stem": [image_path.stem] * len(detections),
            "label_path": [str(label_path)] * len(detections),
            "img_width": [img_width] * len(detections),
            "img_height": [img_height] * len(detections),
            "class_id": detections.classes,
            "score": detections.scores,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "w": w,
            "h": h,
            "size_bin": assign_size_bins(w * h * bin_scale**2),
        },
        schema=CANDIDATE_SCHEMA,
    )


# ---------------------------------------------------------------------------------------------
# Pseudo-label selection
# ---------------------------------------------------------------------------------------------


def select_pseudo_labels(
    candidates: pl.DataFrame,
    thresholds: dict[str, BinThreshold],
    soft_weights: bool,
) -> pl.DataFrame:
    """
    Keep the candidates at or above their size bin's threshold and add a `weight` column: the
    threshold-aware soft weight, or 1 for every box when soft weights are off.
    """
    parts = []
    for size_bin, bin_threshold in thresholds.items():
        kept = candidates.filter(
            (pl.col("size_bin") == size_bin) & (pl.col("score") >= bin_threshold.threshold)
        )
        weights = (
            box_weights(kept["score"].to_numpy(), bin_threshold.threshold, bin_threshold.object_mode)
            if soft_weights
            else np.ones(len(kept), dtype=np.float32)
        )
        parts.append(kept.with_columns(weight=pl.Series(weights, dtype=pl.Float32)))
    if not parts:
        return candidates.clear().with_columns(weight=pl.lit(None, dtype=pl.Float32))
    return pl.concat(parts)


def size_bin_stats(labels: pl.DataFrame) -> dict[str, dict[str, float]]:
    """Share of pseudo-labels and their mean weight in each size bin."""
    stats = {}
    for size_bin in SIZE_BINS:
        weights = labels.filter(pl.col("size_bin") == size_bin)["weight"]
        stats[size_bin] = {
            "share": len(weights) / max(1, len(labels)),
            "mean_weight": float(weights.mean()) if len(weights) else 0.0,  # type: ignore[arg-type]
        }
    return stats


# ---------------------------------------------------------------------------------------------
# Dataset files and label I/O
# ---------------------------------------------------------------------------------------------


def resolve_dataset_paths(
    dataset_path: Path | str,
    labels_dir: Path | str | None = None,
) -> tuple[Path, Path]:
    """
    Images directory (the `images` subdirectory if there is one) and labels directory (given, or
    the images path with its last `images` component replaced by `labels`).
    """
    base_path = Path(dataset_path).resolve()
    if (base_path / "images").is_dir() and base_path.name != "images":
        images_dir = base_path / "images"
    else:
        images_dir = base_path

    if labels_dir is not None:
        return images_dir, Path(labels_dir).resolve()
    if "images" in images_dir.parts:
        parts = list(images_dir.parts)
        last = max(i for i, part in enumerate(parts) if part == "images")
        parts[last] = "labels"
        return images_dir, Path(*parts)
    return images_dir, images_dir.parent / "labels"


def list_images(images_dir: Path) -> list[Path]:
    """All image files under `images_dir`, sorted."""
    return sorted(
        p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def sample_image_files(
    image_files: list[Path],
    sample_images: int | None,
    seed: int,
) -> list[Path]:
    """
    Uniform random sample (without replacement, original order kept) of `sample_images` files.
    Sampling whole images uniformly keeps the pseudo-box size-bin distribution unbiased.
    Returns all files when `sample_images` is None or not smaller than the pool.
    """
    if sample_images is None or not 0 < sample_images < len(image_files):
        return image_files
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(image_files), size=sample_images, replace=False))
    return [image_files[i] for i in chosen]


def clean_dataset_annotations(
    labels_dir: Path | str,
    dataset_root: Path | str | None = None,
    logger: logging.Logger | None = None,
) -> int:
    """
    Remove all existing .txt label files and stale cache files in labels_dir and dataset_root.
    Returns the number of deleted .txt files.
    """
    labels_path = Path(labels_dir)
    deleted_txt = 0
    if labels_path.is_dir():
        for txt_file in labels_path.rglob("*.txt"):
            if txt_file.is_file():
                txt_file.unlink()
                deleted_txt += 1

    cache_dirs = [labels_path] + ([Path(dataset_root)] if dataset_root is not None else [])
    deleted_caches = 0
    for cache_dir in cache_dirs:
        if cache_dir.is_dir():
            for cache_file in cache_dir.rglob("*.cache"):
                if cache_file.is_file():
                    cache_file.unlink()
                    deleted_caches += 1

    labels_path.mkdir(parents=True, exist_ok=True)
    if logger:
        logger.info(
            f"Cleaned annotations in {labels_path}: deleted {deleted_txt} .txt files, "
            f"{deleted_caches} cache files."
        )
    return deleted_txt


def read_images(
    paths: Sequence[Path], pool: ThreadPoolExecutor
) -> list[tuple[Path, np.ndarray]]:
    """Decode images in parallel, skipping unreadable files."""
    images = pool.map(lambda path: cv2.imread(str(path)), paths)
    return [(path, image) for path, image in zip(paths, images) if image is not None]


def with_yolo_columns(labels: pl.DataFrame, with_weights: bool) -> pl.DataFrame:
    """
    Add normalised YOLO box columns and the label-file `line`
    (`class x_center y_center width height [weight]`).
    """
    labels = labels.with_columns(
        x_center=((pl.col("x1") + pl.col("x2")) / (2.0 * pl.col("img_width"))).clip(0.0, 1.0),
        y_center=((pl.col("y1") + pl.col("y2")) / (2.0 * pl.col("img_height"))).clip(0.0, 1.0),
        w_norm=((pl.col("x2") - pl.col("x1")) / pl.col("img_width")).clip(0.0, 1.0),
        h_norm=((pl.col("y2") - pl.col("y1")) / pl.col("img_height")).clip(0.0, 1.0),
    )
    fields = [
        pl.col("class_id").cast(pl.String),
        *(pl.col(c).round(6).cast(pl.String) for c in ("x_center", "y_center", "w_norm", "h_norm")),
        *([pl.col("weight").round(4).cast(pl.String)] if with_weights else []),
    ]
    return labels.with_columns(line=pl.concat_str(fields, separator=" "))


def write_yolo_labels(labels: pl.DataFrame) -> int:
    """Write one YOLO label file per image (images without pseudo-labels get no file)."""
    files = labels.group_by("label_path").agg(content=pl.col("line").str.join("\n"))
    for row in files.iter_rows(named=True):
        target = Path(row["label_path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(row["content"] + "\n", encoding="utf-8")
    return len(files)


def save_candidates(candidates: pl.DataFrame, path: Path | str) -> None:
    """Save raw candidate detections as Parquet (for offline analysis)."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    candidates.write_parquet(destination)


# ---------------------------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------------------------


@dataclass
class ImagePool:
    """Unlabeled images to pseudo-label and where their label files go."""

    images_dir: Path
    labels_dir: Path
    files: list[Path]

    def label_path(self, image_path: Path) -> Path:
        return self.labels_dir / image_path.relative_to(self.images_dir).with_suffix(".txt")


@dataclass(frozen=True)
class PseudoLabelResult:
    """Outcome of one pseudo-labeling pass."""

    images_processed: int
    total_candidates: int
    candidates_df: pl.DataFrame
    filtered_df: pl.DataFrame
    thresholds: dict[str, float] = field(default_factory=dict)
    object_modes: dict[str, float] = field(default_factory=dict)
    bin_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    labeled_images: int = 0
    labels_written: int = 0


class PseudoLabelGenerator:
    """
    Generates pseudo-labels for unlabeled images using batched multi-view candidate inference
    and COCO size-bin ground thresholding.
    """

    def __init__(
        self,
        teacher_model: YOLO | str | Path,
        batch_size: int = 32,
        half: bool = True,
        workers: int = 4,
        conf: float = 0.001,
        iou_threshold: float = 0.5,
        fallback_threshold: float = 0.4,
        soft_weights: bool = True,
        size_bins: bool = True,
        multi_view: bool = True,
        imgsz: int = 640,
        input_scale_bins: bool = True,
        device: str | torch.device | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.model = YOLO(str(teacher_model)) if isinstance(teacher_model, (str, Path)) else teacher_model
        self.batch_size = max(1, batch_size)
        self.half = half
        self.workers = max(1, workers)
        self.conf = conf  # candidate confidence floor
        self.iou_threshold = iou_threshold  # NMS IoU for merging the views
        self.fallback_threshold = fallback_threshold  # threshold of a bin without a score valley
        self.soft_weights = soft_weights
        self.size_bins = size_bins  # False: one shared threshold for all boxes
        self.multi_view = multi_view  # False: normal view only
        self.imgsz = imgsz
        self.input_scale_bins = input_scale_bins  # False: size bins in original image pixels
        self.device = device
        self.logger = logger or get_logger()
        self._image_lists: dict[Path, list[Path]] = {}

    def generate(
        self,
        images_dir: Path | str,
        labels_dir: Path | str | None = None,
        max_images: int | None = None,
        save_candidates_path: Path | str | None = None,
        sample_images: int | None = None,
        sample_seed: int = 0,
    ) -> PseudoLabelResult:
        """
        Pseudo-label the images in `images_dir` (or a uniform random sample of `sample_images` of
        them) and write the labels to `labels_dir`, replacing any existing ones.
        """
        pool = self._image_pool(images_dir, labels_dir, max_images, sample_images, sample_seed)
        clean_dataset_annotations(pool.labels_dir, pool.images_dir.parent, self.logger)

        candidates = self._collect_candidates(pool)
        if save_candidates_path is not None:
            save_candidates(candidates, save_candidates_path)
        if candidates.is_empty():
            self.logger.warning("No candidate detections were found.")
            return self._empty_result(pool, candidates)

        thresholds = self._fit_thresholds(candidates)
        labels = select_pseudo_labels(candidates, thresholds, self.soft_weights)
        labels = with_yolo_columns(labels, with_weights=self.soft_weights)
        labels_written = write_yolo_labels(labels)

        result = PseudoLabelResult(
            images_processed=len(pool.files),
            total_candidates=len(candidates),
            candidates_df=candidates,
            filtered_df=labels,
            thresholds={b: t.threshold for b, t in thresholds.items()},
            object_modes={b: t.object_mode for b, t in thresholds.items()},
            bin_stats=size_bin_stats(labels),
            labeled_images=labels["image_stem"].n_unique(),
            labels_written=labels_written,
        )
        self._log_summary(result, pool)
        return result

    def _image_pool(
        self,
        images_dir: Path | str,
        labels_dir: Path | str | None,
        max_images: int | None,
        sample_images: int | None,
        sample_seed: int,
    ) -> ImagePool:
        """Resolve the dataset directories and choose the images of this pass."""
        resolved_images, resolved_labels = resolve_dataset_paths(images_dir, labels_dir)
        if resolved_images not in self._image_lists:
            self._image_lists[resolved_images] = list_images(resolved_images)
        files = self._image_lists[resolved_images]
        if max_images is not None and max_images > 0:
            files = files[:max_images]

        chosen = sample_image_files(files, sample_images, sample_seed)
        self.logger.info(
            f"Pseudo-labeling {len(chosen)} of {len(files)} images in {resolved_images} "
            f"(seed={sample_seed})."
        )
        return ImagePool(resolved_images, resolved_labels, chosen)

    def _collect_candidates(self, pool: ImagePool) -> pl.DataFrame:
        """Batched multi-view Teacher inference over the pool -> one row per candidate box."""
        self.model.predictor = None  # rebuild the predictor for the current Teacher weights
        frames = [pl.DataFrame(schema=CANDIDATE_SCHEMA)]
        batches = [
            pool.files[i : i + self.batch_size] for i in range(0, len(pool.files), self.batch_size)
        ]
        with ThreadPoolExecutor(max_workers=self.workers) as io_pool, tqdm(
            total=len(pool.files), desc="Multi-view inference"
        ) as progress:
            for batch in batches:
                images = read_images(batch, io_pool)
                if images:
                    frames += self._batch_candidates(images, pool, io_pool)
                progress.update(len(batch))

        candidates = pl.concat(frames)
        self.logger.info(f"Extracted {len(candidates)} candidate detections.")
        return candidates

    def _batch_candidates(
        self,
        images: list[tuple[Path, np.ndarray]],
        pool: ImagePool,
        io_pool: ThreadPoolExecutor,
    ) -> list[pl.DataFrame]:
        """Candidates of each image in a batch, merged over its views."""
        views_per_image = self._predict_views([image for _, image in images], io_pool)
        frames = []
        for (path, image), views in zip(images, views_per_image):
            img_height, img_width = image.shape[:2]
            bin_scale = input_scale(img_width, img_height, self.imgsz) if self.input_scale_bins else 1.0
            detections = merge_views(views, self.iou_threshold)
            frames.append(
                candidate_frame(detections, path, pool.label_path(path), (img_width, img_height), bin_scale)
            )
        return frames

    def _predict_views(
        self, images: list[np.ndarray], io_pool: ThreadPoolExecutor
    ) -> list[list[Detections]]:
        """Teacher detections of every image on the normal (and, if multi-view, flipped) view."""
        predict_args: dict[str, Any] = {
            "batch": len(images),
            "imgsz": self.imgsz,
            "half": self.half,
            "conf": self.conf,
            "verbose": False,
        }
        if self.device is not None:
            predict_args["device"] = self.device

        normal = [detections_from_result(r) for r in self.model.predict(images, **predict_args)]
        if not self.multi_view:
            return [[view] for view in normal]

        flips = list(io_pool.map(lambda image: cv2.flip(image, 1), images))
        flipped = [
            detections_from_result(r, flipped=True) for r in self.model.predict(flips, **predict_args)
        ]
        return [[n, f] for n, f in zip(normal, flipped)]

    def _fit_thresholds(self, candidates: pl.DataFrame) -> dict[str, BinThreshold]:
        """Ground threshold of each size bin (one shared threshold when size bins are off)."""
        if not self.size_bins:
            shared = self._fit_logged(candidates["score"], "All size bins")
            return {size_bin: shared for size_bin in SIZE_BINS}
        return {
            size_bin: self._fit_logged(
                candidates.filter(pl.col("size_bin") == size_bin)["score"], f"Size bin '{size_bin}'"
            )
            for size_bin in SIZE_BINS
        }

    def _fit_logged(self, scores: pl.Series, group: str) -> BinThreshold:
        values = scores.to_numpy().astype(float)
        fitted = fit_bin_threshold(values, self.conf, self.fallback_threshold)
        if fitted.from_valley:
            self.logger.info(
                f"{group}: count={len(values)}, ground threshold={fitted.threshold:.4f}, "
                f"object mode={fitted.object_mode:.4f}"
            )
        else:
            self.logger.warning(
                f"{group}: no score valley found (count={len(values)}); "
                f"using fallback threshold={fitted.threshold:.4f}"
            )
        return fitted

    def _empty_result(self, pool: ImagePool, candidates: pl.DataFrame) -> PseudoLabelResult:
        labels = with_yolo_columns(select_pseudo_labels(candidates, {}, False), self.soft_weights)
        return PseudoLabelResult(
            images_processed=len(pool.files),
            total_candidates=0,
            candidates_df=candidates,
            filtered_df=labels,
        )

    def _log_summary(self, result: PseudoLabelResult, pool: ImagePool) -> None:
        boxes = len(result.filtered_df)
        self.logger.info(
            f"Kept {boxes} of {result.total_candidates} candidates as pseudo-labels "
            f"({result.labeled_images} labeled images, "
            f"{boxes / max(1, result.labeled_images):.2f} boxes/image)."
        )
        self.logger.info(
            "Pseudo-label size-bin shares / mean weights: "
            + ", ".join(
                f"{b}={s['share']:.3f}/{s['mean_weight']:.3f}" for b, s in result.bin_stats.items()
            )
        )
        self.logger.info(f"Saved {result.labels_written} YOLO label files into {pool.labels_dir}")


# ---------------------------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------------------------


@app.command()
def main(
    images: Path = typer.Option(
        ...,
        "--images",
        "-i",
        help="Path to directory containing unlabeled images.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    model: Path = typer.Option(
        ...,
        "--model",
        "-m",
        help="Path to the YOLO Teacher model weights.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    labels: Optional[Path] = typer.Option(
        None,
        "--labels",
        "-l",
        help="Path to output labels directory (auto-inferred if omitted).",
    ),
    save_candidates_path: Optional[Path] = typer.Option(
        None,
        "--save-candidates",
        "-s",
        help="Optional destination path to save raw candidate detections as Parquet.",
    ),
    max_images: Optional[int] = typer.Option(
        None,
        "--max-images",
        "-n",
        help="Maximum number of images to process (useful for test runs).",
        min=1,
    ),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Inference batch size.", min=1),
    half: bool = typer.Option(
        True, "--half/--no-half", help="Use FP16 half-precision inference on GPU."
    ),
    workers: int = typer.Option(
        4, "--workers", "-w", help="Number of threads for image decoding and flipping.", min=1
    ),
    conf: float = typer.Option(
        0.001, "--conf", "-c", help="Candidate confidence floor.", min=0.0, max=1.0
    ),
    iou: float = typer.Option(
        0.5, "--iou", help="NMS IoU threshold for merging the views.", min=0.0, max=1.0
    ),
    fallback_threshold: float = typer.Option(
        0.4,
        "--fallback-threshold",
        help="Score threshold used for a size bin when its score histogram has no valley.",
        min=0.0,
        max=1.0,
    ),
    soft_weights: bool = typer.Option(
        True,
        "--soft-weights/--no-soft-weights",
        help="Write threshold-aware per-box weights as a 6th label column.",
    ),
    size_bins: bool = typer.Option(
        True,
        "--size-bins/--no-size-bins",
        help="Ground threshold per COCO size bin (off: one threshold for all boxes).",
    ),
    multi_view: bool = typer.Option(
        True,
        "--multi-view/--no-multi-view",
        help="Aggregate the normal and horizontally flipped views (off: normal view only).",
    ),
    imgsz: int = typer.Option(640, "--imgsz", help="Teacher inference image size."),
    input_scale_bins: bool = typer.Option(
        True,
        "--input-scale-bins/--no-input-scale-bins",
        help="Take COCO size bins at the network input scale (off: in original image pixels).",
    ),
    device: Optional[str] = typer.Option(
        None, "--device", "-d", help="PyTorch device identifier (e.g. 'cuda:0', 'cpu')."
    ),
) -> None:
    """Run pseudo-label generation on an unannotated image dataset."""
    generator = PseudoLabelGenerator(
        teacher_model=model,
        batch_size=batch_size,
        half=half,
        workers=workers,
        conf=conf,
        iou_threshold=iou,
        fallback_threshold=fallback_threshold,
        soft_weights=soft_weights,
        size_bins=size_bins,
        multi_view=multi_view,
        imgsz=imgsz,
        input_scale_bins=input_scale_bins,
        device=device,
    )
    generator.generate(
        images_dir=images,
        labels_dir=labels,
        max_images=max_images,
        save_candidates_path=save_candidates_path,
    )


if __name__ == "__main__":
    app()
