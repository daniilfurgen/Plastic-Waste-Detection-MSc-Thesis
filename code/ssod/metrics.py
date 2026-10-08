"""
COCO evaluation and prediction harmonization metrics module for YOLO Mean Teacher training.
Computes standard COCO AP/AR metrics (all scales, small, medium, large) and peak F1 score.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

try:
    from ssod.utilities import get_logger
except ImportError:
    from utilities import get_logger  # type: ignore[import-not-found,no-redef]


def fix_prediction_ids(
    predictions_path: Path | str,
    coco_gt_path: Path | str,
    default_category_id: int = 0,
) -> list[dict[str, Any]]:
    """
    Harmonize image_id and category_id in Ultralytics predictions.json
    with the reference COCO ground truth annotations.json.
    """
    pred_path = Path(predictions_path)
    gt_path = Path(coco_gt_path)

    if not pred_path.exists():
        return []

    with open(pred_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    if not predictions:
        return []

    with open(gt_path, "r", encoding="utf-8") as f:
        coco_gt = json.load(f)

    img_name_to_id = {img["file_name"]: img["id"] for img in coco_gt.get("images", [])}
    stem_to_id = {
        Path(img["file_name"]).stem: img["id"] for img in coco_gt.get("images", [])
    }

    # Ultralytics writes 0-based class indices as category_id for non-COCO datasets; keep them
    # when the GT uses the same ids (multi-class), otherwise fall back to the default category
    gt_category_ids = {cat["id"] for cat in coco_gt.get("categories", [])}

    for pred in predictions:
        if pred.get("category_id") not in gt_category_ids:
            pred["category_id"] = default_category_id
        file_name = pred.get("file_name")
        if file_name and file_name in img_name_to_id:
            pred["image_id"] = img_name_to_id[file_name]
        elif file_name and Path(file_name).stem in stem_to_id:
            pred["image_id"] = stem_to_id[Path(file_name).stem]
        elif "image_id" in pred and str(pred["image_id"]) in stem_to_id:
            pred["image_id"] = stem_to_id[str(pred["image_id"])]

    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f)

    return predictions


def evaluate_coco(
    coco_gt_path: Path | str,
    predictions_path: Path | str,
    fix_ids: bool = True,
    default_category_id: int = 0,
    logger: logging.Logger | None = None,
) -> dict[str, float]:
    """
    Evaluate bounding box predictions using pycocotools COCOeval.

    Returns:
        Dictionary containing standard COCO AP/AR metrics:
        - mAP_50_95, mAP_50, mAP_75, mAP_s, mAP_m, mAP_l
        - AR_1, AR_10, AR_100, AR_s, AR_m, AR_l
        - peak_f1, peak_precision, peak_recall (at IoU=0.50)
    """
    log = logger or get_logger()
    gt_path = Path(coco_gt_path)
    pred_path = Path(predictions_path)

    default_zero_metrics = {
        "mAP_50_95": 0.0,
        "mAP_50": 0.0,
        "mAP_75": 0.0,
        "mAP_s": 0.0,
        "mAP_m": 0.0,
        "mAP_l": 0.0,
        "AR_1": 0.0,
        "AR_10": 0.0,
        "AR_100": 0.0,
        "AR_s": 0.0,
        "AR_m": 0.0,
        "AR_l": 0.0,
        "peak_f1": 0.0,
        "peak_precision": 0.0,
        "peak_recall": 0.0,
    }

    if not gt_path.exists():
        log.warning(
            f"COCO ground truth file not found: {gt_path}; returning zeroed metrics."
        )
        return default_zero_metrics

    if not pred_path.exists():
        log.warning(
            f"Predictions JSON file not found: {pred_path}; returning zeroed metrics."
        )
        return default_zero_metrics

    if fix_ids:
        fix_prediction_ids(pred_path, gt_path, default_category_id=default_category_id)

    # Check if predictions file contains detections
    try:
        with open(pred_path, "r", encoding="utf-8") as f:
            preds = json.load(f)
    except Exception as e:
        log.warning(f"Failed to read predictions from {pred_path}: {e}")
        return default_zero_metrics

    if not preds:
        log.warning(
            f"No predictions found in {pred_path}; returning zeroed COCO metrics."
        )
        return default_zero_metrics

    try:
        coco_gt = COCO(str(gt_path))
        coco_dt = coco_gt.loadRes(str(pred_path))

        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = coco_eval.stats

        # Peak F1 Calculation at IoU=0.50
        # precision array dimensions: [T, R, K, A, M]
        # T=0 (IoU=0.50), K=0 (first category), A=0 (all areas), M=2 (maxDets=100)
        precisions = coco_eval.eval["precision"][0, :, 0, 0, 2]
        recalls = np.linspace(0, 1, 101)
        valid_idx = precisions > -1

        if np.any(valid_idx):
            p = precisions[valid_idx]
            r = recalls[valid_idx]
            f1_scores = 2 * (p * r) / (p + r + 1e-16)
            peak_idx = int(np.argmax(f1_scores))
            peak_f1 = float(f1_scores[peak_idx])
            peak_precision = float(p[peak_idx])
            peak_recall = float(r[peak_idx])
        else:
            peak_f1, peak_precision, peak_recall = 0.0, 0.0, 0.0

        return {
            "mAP_50_95": float(stats[0]),
            "mAP_50": float(stats[1]),
            "mAP_75": float(stats[2]),
            "mAP_s": float(stats[3]),
            "mAP_m": float(stats[4]),
            "mAP_l": float(stats[5]),
            "AR_1": float(stats[6]),
            "AR_10": float(stats[7]),
            "AR_100": float(stats[8]),
            "AR_s": float(stats[9]),
            "AR_m": float(stats[10]),
            "AR_l": float(stats[11]),
            "peak_f1": peak_f1,
            "peak_precision": peak_precision,
            "peak_recall": peak_recall,
        }
    except Exception as e:
        log.warning(f"COCO evaluation failed on {pred_path}: {e}")
        return default_zero_metrics
