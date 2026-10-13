"""
COCO evaluation and prediction harmonization metrics module for YOLO Mean Teacher training.
Computes standard COCO AP/AR metrics (all scales, small, medium, large) and peak F1 score.
The COCO size bins are applied at the network input scale, since the 32²/96² px thresholds assume
COCO-sized images (longer side ~640 px) and the datasets here are larger and of mixed resolution.

`evaluate_coco` reads as the pipeline: load predictions (harmonised to the GT ids) → COCOeval with
input-scale areas → summary statistics and peak F1.
"""

from __future__ import annotations

from collections.abc import Callable
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

try:
    from ssod.utilities import get_logger, input_scale
except ImportError:
    from utilities import get_logger, input_scale  # type: ignore[import-not-found,no-redef]

Prediction = dict[str, Any]

# Names of COCOeval.stats, in its order
SUMMARY_STATS = (
    "mAP_50_95", "mAP_50", "mAP_75", "mAP_s", "mAP_m", "mAP_l",
    "AR_1", "AR_10", "AR_100", "AR_s", "AR_m", "AR_l",
)
PEAK_STATS = ("peak_f1", "peak_precision", "peak_recall")
METRIC_NAMES = SUMMARY_STATS + PEAK_STATS
RECALL_THRESHOLDS = np.linspace(0, 1, 101)  # COCOeval's recall grid


def zero_metrics() -> dict[str, float]:
    """Metrics reported when there is nothing to evaluate."""
    return dict.fromkeys(METRIC_NAMES, 0.0)


def read_json(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------
# Prediction id harmonisation
# ---------------------------------------------------------------------------------------------


def image_id_resolver(images: list[dict[str, Any]]) -> Callable[[Prediction], Any]:
    """
    Maps a prediction to its GT image id: by file name, then by file stem, then by its own
    `image_id` read as a stem (Ultralytics' default). Unresolved predictions keep their id.
    """
    by_name = {img["file_name"]: img["id"] for img in images}
    by_stem = {Path(img["file_name"]).stem: img["id"] for img in images}

    def resolve(pred: Prediction) -> Any:
        if file_name := pred.get("file_name"):
            if file_name in by_name:
                return by_name[file_name]
            if Path(file_name).stem in by_stem:
                return by_stem[Path(file_name).stem]
        return by_stem.get(str(pred.get("image_id")), pred.get("image_id"))

    return resolve


def category_id_mapper(
    gt_category_ids: set[int], default_category_id: int, category_offset: int
) -> Callable[[Prediction], int]:
    """Undoes Ultralytics' category offset; ids unknown to the GT fall back to the default."""

    def map_category(pred: Prediction) -> int:
        category = pred.get("category_id", default_category_id) - category_offset
        return category if category in gt_category_ids else default_category_id

    return map_category


def fix_prediction_ids(
    predictions_path: Path | str,
    coco_gt_path: Path | str,
    default_category_id: int = 0,
    category_offset: int = 1,
) -> list[Prediction]:
    """
    Ultralytics predictions.json harmonised with the reference COCO annotations.json: image ids
    matched by file name, and category ids mapped back to the GT's.

    For non-COCO datasets Ultralytics writes class index + 1 as category_id, while the GT
    annotations here use the YOLO class index; `category_offset` undoes that. Ids that are still
    unknown to the GT fall back to `default_category_id`. The file itself is not modified, so
    predictions can be evaluated any number of times.
    """
    if not Path(predictions_path).exists():
        return []
    predictions = read_json(predictions_path)
    if not predictions:
        return []
    coco_gt = read_json(coco_gt_path)

    resolve_image = image_id_resolver(coco_gt.get("images", []))
    map_category = category_id_mapper(
        {cat["id"] for cat in coco_gt.get("categories", [])}, default_category_id, category_offset
    )
    return [
        {**pred, "category_id": map_category(pred), "image_id": resolve_image(pred)}
        for pred in predictions
    ]


# ---------------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------------


def scale_areas(coco: COCO, images: dict[int, dict[str, Any]], imgsz: int) -> None:
    """
    Rescale every annotation's `area` in place to the network input (longer side -> imgsz), so
    COCOeval's size bins see the boxes at the scale the detector does. Per image, as resolutions
    differ within a dataset.
    """
    for ann in coco.anns.values():
        img = images[ann["image_id"]]
        ann["area"] *= input_scale(img["width"], img["height"], imgsz) ** 2


def run_cocoeval(gt_path: Path, predictions: list[Prediction], imgsz: int | None) -> COCOeval:
    """Evaluated, accumulated and summarised bbox COCOeval; areas at `imgsz` unless None."""
    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(predictions)
    if imgsz is not None:
        for coco in (coco_gt, coco_dt):
            scale_areas(coco, coco_gt.imgs, imgsz)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


def peak_f1(precision: np.ndarray) -> dict[str, float]:
    """
    Peak F1 (and its precision and recall) on the class-averaged precision-recall curve at
    IoU 0.50, all areas, 100 detections. `precision` is COCOeval's [T, R, K, A, M] array.
    """
    per_class = precision[0, :, :, 0, 2]
    evaluated = per_class[:, (per_class > -1).any(axis=0)]  # categories present in the GT
    if not evaluated.shape[1]:
        return dict.fromkeys(PEAK_STATS, 0.0)
    curve = evaluated.mean(axis=1)
    valid = curve > -1
    if not valid.any():
        return dict.fromkeys(PEAK_STATS, 0.0)

    p, r = curve[valid], RECALL_THRESHOLDS[valid]
    f1 = 2 * p * r / (p + r + 1e-16)
    best = int(np.argmax(f1))
    return dict(zip(PEAK_STATS, map(float, (f1[best], p[best], r[best]))))


def coco_metrics(coco_eval: COCOeval) -> dict[str, float]:
    """Named summary statistics and peak F1 from an accumulated COCOeval."""
    return {
        **dict(zip(SUMMARY_STATS, map(float, coco_eval.stats))),
        **peak_f1(coco_eval.eval["precision"]),
    }


def load_predictions(
    pred_path: Path,
    gt_path: Path,
    fix_ids: bool,
    default_category_id: int,
    category_offset: int,
) -> list[Prediction]:
    if not fix_ids:
        return read_json(pred_path)
    return fix_prediction_ids(
        pred_path, gt_path, default_category_id=default_category_id, category_offset=category_offset
    )


def evaluate_coco(
    coco_gt_path: Path | str,
    predictions_path: Path | str,
    fix_ids: bool = True,
    default_category_id: int = 0,
    category_offset: int = 1,
    imgsz: int | None = 640,
    logger: logging.Logger | None = None,
) -> dict[str, float]:
    """
    Evaluate bounding box predictions using pycocotools COCOeval.

    Box areas for the small/medium/large bins are taken at the network input scale `imgsz`
    (`None`: original pixels). Metrics over all sizes do not depend on it. Missing or unreadable
    inputs and failed evaluations are logged and give zeroed metrics.

    Returns:
        Dictionary containing standard COCO AP/AR metrics:
        - mAP_50_95, mAP_50, mAP_75, mAP_s, mAP_m, mAP_l
        - AR_1, AR_10, AR_100, AR_s, AR_m, AR_l
        - peak_f1, peak_precision, peak_recall (at IoU=0.50)
    """
    log = logger or get_logger()
    gt_path, pred_path = Path(coco_gt_path), Path(predictions_path)

    for path, what in ((gt_path, "COCO ground truth file"), (pred_path, "Predictions JSON file")):
        if not path.exists():
            log.warning(f"{what} not found: {path}; returning zeroed metrics.")
            return zero_metrics()

    try:
        predictions = load_predictions(
            pred_path, gt_path, fix_ids, default_category_id, category_offset
        )
    except Exception as e:
        log.warning(f"Failed to read predictions from {pred_path}: {e}")
        return zero_metrics()
    if not predictions:
        log.warning(f"No predictions found in {pred_path}; returning zeroed COCO metrics.")
        return zero_metrics()

    try:
        return coco_metrics(run_cocoeval(gt_path, predictions, imgsz))
    except Exception as e:
        log.warning(f"COCO evaluation failed on {pred_path}: {e}")
        return zero_metrics()
