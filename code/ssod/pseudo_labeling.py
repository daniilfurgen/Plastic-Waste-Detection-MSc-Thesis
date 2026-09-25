import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms
import numpy as np
from ultralytics import YOLO

SIZE_BINS = {
    "small": (0, 32**2),
    "medium": (32**2, 96**2),
    "large": (96**2, float("inf")),
}


def _hflip(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError("image must be a single CHW tensor")
    return image.flip(-1)


def _upscale(image: torch.Tensor, factor: float = 2) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError("image must be a single CHW tensor")
    if factor <= 0:
        raise ValueError("factor must be positive")

    height, width = image.shape[-2:]
    size = (round(height * factor), round(width * factor))
    return F.interpolate(
        image.unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def _inverse_transform(preds, name: str):
    if name not in {"normal", "flip", "scale", "flip_scale"}:
        raise ValueError(f"unknown view transform: {name}")
    if not preds:
        return []

    result = preds[0]
    boxes = result.boxes.xyxy.detach().cpu().tolist()
    scores = result.boxes.conf.detach().cpu().tolist()
    classes = result.boxes.cls.detach().cpu().tolist()
    view_height, view_width = result.orig_shape
    scale = 2 if "scale" in name else 1
    width = view_width / scale

    detections = []
    for box, score, class_id in zip(boxes, scores, classes):
        x1, y1, x2, y2 = (coordinate / scale for coordinate in box)
        if "flip" in name:
            x1, x2 = width - x2, width - x1

        detections.append(
            {
                "class": int(class_id),
                "score": float(score),
                "box": [x1, y1, x2, y2],
                "w": x2 - x1,
                "h": y2 - y1,
            }
        )
    return detections


def _nms_aggregate(prediction_groups, iou_threshold: float = 0.5):
    detections = [detection for group in prediction_groups for detection in group]
    if not detections:
        return []

    boxes = torch.tensor([detection["box"] for detection in detections])
    scores = torch.tensor([detection["score"] for detection in detections])
    classes = torch.tensor(
        [detection["class"] for detection in detections], dtype=torch.long
    )
    keep = batched_nms(boxes, scores, classes, iou_threshold)

    return [detections[index] for index in keep.tolist()]


def assign_size_bin(w: float, h: float):
    area = w * h
    for name, (lo, hi) in SIZE_BINS.items():
        if lo <= area < hi:
            return name
    return "large"


def ground_threshold(scores, lo=0.0, hi=1.0, n_bins=21):
    if len(scores) == 0:
        return hi
    hist, edges = np.histogram(scores, bins=n_bins, range=(lo, hi))
    return edges[np.argmin(hist)]


def generate_candidates_multiview(teacher_model: YOLO, image: torch.Tensor):
    """One unlabeled image -> aggregated candidate boxes via 4 views."""
    views = {
        "normal": image,
        "flip": _hflip(image),
        "scale": _upscale(image, factor=2),
        "flip_scale": _upscale(_hflip(image), factor=2),
    }
    raw_preds = {}
    for name, view_img in views.items():
        preds = teacher_model.predict(view_img, conf=0.001)
        raw_preds[name] = _inverse_transform(preds, name)

    merged = _nms_aggregate(list(raw_preds.values()))
    return merged  # list of {class, score, box, w, h}
