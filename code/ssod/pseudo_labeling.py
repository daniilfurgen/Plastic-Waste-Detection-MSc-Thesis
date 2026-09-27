from concurrent.futures import ThreadPoolExecutor
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
    from ssod.utilities import get_logger
except ImportError:
    from utilities import get_logger  # type: ignore[import-not-found,no-redef]

SIZE_BINS: dict[str, tuple[float, float]] = {
    "small": (0.0, 32.0**2),
    "medium": (32.0**2, 96.0**2),
    "large": (96.0**2, float("inf")),
}

EMPTY_FILTERED_SCHEMA: dict[str, Any] = {
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
    "x_center": pl.Float32,
    "y_center": pl.Float32,
    "w_norm": pl.Float32,
    "h_norm": pl.Float32,
    "line": pl.String,
}

app = typer.Typer(
    name="pseudo-label-generator",
    help="Generate pseudo-labels via multi-view inference and COCO size-bin ground thresholding.",
    add_completion=False,
)


def _hflip(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    if isinstance(image, torch.Tensor):
        if image.ndim != 3:
            raise ValueError("image must be a single CHW tensor")
        return image.flip(-1)
    elif isinstance(image, np.ndarray):
        return cv2.flip(image, 1)
    raise TypeError(f"Unsupported image type: {type(image)}")


def _inverse_transform(preds: Any, name: str) -> list[dict[str, Any]]:
    if name not in {"normal", "flip", "scale", "flip_scale"}:
        raise ValueError(f"unknown view transform: {name}")
    if not preds:
        return []

    result = preds[0] if isinstance(preds, (list, tuple)) else preds
    if result.boxes is None or len(result.boxes) == 0:
        return []

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
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "w": float(x2 - x1),
                "h": float(y2 - y1),
            }
        )
    return detections


def _nms_aggregate(
    prediction_groups: Sequence[list[dict[str, Any]]],
    iou_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    detections = [detection for group in prediction_groups for detection in group]
    if not detections:
        return []

    boxes = torch.tensor(
        [detection["box"] for detection in detections], dtype=torch.float32
    )
    scores = torch.tensor(
        [detection["score"] for detection in detections], dtype=torch.float32
    )
    classes = torch.tensor(
        [detection["class"] for detection in detections], dtype=torch.long
    )
    keep = batched_nms(boxes, scores, classes, iou_threshold)

    return [detections[index] for index in keep.tolist()]


def assign_size_bin(w: float, h: float) -> str:
    area = w * h
    for name, (lo, hi) in SIZE_BINS.items():
        if lo <= area < hi:
            return name
    return "large"


def ground_threshold(
    scores: Sequence[float] | np.ndarray,
    lo: float = 0.0,
    hi: float = 1.0,
    n_bins: int = 21,
) -> float:
    hist, bin_edges = np.histogram(scores, bins=n_bins, range=(lo, hi))

    left_max_index = np.argmax(hist[: len(hist) // 2])
    right_max_index = np.argmax(hist[len(hist) // 2 :]) + len(hist) // 2

    local_min_index = np.argmin(hist[left_max_index:right_max_index]) + left_max_index
    return float(bin_edges[local_min_index])


def generate_candidates_multiview(
    teacher_model: YOLO,
    image: torch.Tensor | np.ndarray | str | Path,
    conf: float = 0.001,
    iou_threshold: float = 0.5,
    half: bool = True,
    device: str | torch.device | None = None,
) -> list[dict[str, Any]]:
    """One unlabeled image -> aggregated candidate boxes via 2 views (normal and flip)."""
    if isinstance(image, (str, Path)):
        img_np = cv2.imread(str(image))
        if img_np is None:
            raise FileNotFoundError(f"Failed to read image at {image}")
    elif isinstance(image, np.ndarray):
        img_np = image
    elif isinstance(image, torch.Tensor):
        if image.ndim == 4 and image.shape[0] == 1:
            image = image.squeeze(0)
        img_np = image.permute(1, 2, 0).cpu().numpy()
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    views = {
        "normal": img_np,
        "flip": cv2.flip(img_np, 1),
    }

    raw_preds = {}
    for name, view_np in views.items():
        predict_kwargs: dict[str, Any] = {
            "conf": conf,
            "half": half,
            "verbose": False,
        }
        if device is not None:
            predict_kwargs["device"] = device
        preds = teacher_model.predict(view_np, **predict_kwargs)
        raw_preds[name] = _inverse_transform(preds, name)

    merged = _nms_aggregate(list(raw_preds.values()), iou_threshold=iou_threshold)
    return merged


def resolve_dataset_paths(
    dataset_path: Path | str,
    labels_dir: Path | str | None = None,
) -> tuple[Path, Path, list[Path]]:
    """
    Resolve images directory, labels directory, and find image files.
    """
    base_path = Path(dataset_path).resolve()

    if (base_path / "images").is_dir() and base_path.name != "images":
        images_dir = base_path / "images"
    else:
        images_dir = base_path

    if labels_dir is not None:
        resolved_labels_dir = Path(labels_dir).resolve()
    else:
        if "images" in images_dir.parts:
            parts = list(images_dir.parts)
            idx = max(i for i, part in enumerate(parts) if part == "images")
            parts[idx] = "labels"
            resolved_labels_dir = Path(*parts)
        else:
            resolved_labels_dir = images_dir.parent / "labels"

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = sorted(
        [
            p
            for p in images_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in valid_exts
        ]
    )

    return images_dir, resolved_labels_dir, image_files


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
    if labels_path.exists() and labels_path.is_dir():
        for txt_file in labels_path.rglob("*.txt"):
            if txt_file.is_file():
                txt_file.unlink()
                deleted_txt += 1

    dirs_to_check = [labels_path]
    if dataset_root is not None:
        dirs_to_check.append(Path(dataset_root))

    deleted_caches = 0
    for d in dirs_to_check:
        if d.exists() and d.is_dir():
            for cache_file in d.rglob("*.cache"):
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
        device: str | torch.device | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if isinstance(teacher_model, (str, Path)):
            self.model = YOLO(str(teacher_model))
        else:
            self.model = teacher_model

        self.batch_size = max(1, batch_size)
        self.half = half
        self.workers = max(1, workers)
        self.conf = conf
        self.iou_threshold = iou_threshold
        self.device = device
        self.logger = logger or get_logger()
        self._cached_images_dir: Path | None = None
        self._cached_image_files: list[Path] | None = None

    def generate(
        self,
        images_dir: Path | str,
        labels_dir: Path | str | None = None,
        max_images: int | None = None,
        save_candidates_path: Path | str | None = None,
    ) -> dict[str, Any]:
        """
        Run full pseudo-labeling pipeline:
        1. Clean existing annotations in labels_dir and stale caches.
        2. Generate candidates for all images using batched multi-view inference.
        3. Collect candidates into Polars DataFrame.
        4. Optionally export raw candidates to Parquet.
        5. Compute optimal ground threshold per COCO size bin.
        6. Filter candidates by threshold and save YOLO .txt labels (omitting empty files).
        """
        if hasattr(self.model, "predictor"):
            self.model.predictor = None

        resolved_img_dir, resolved_lbl_dir, image_files = resolve_dataset_paths(
            images_dir, labels_dir
        )

        if (
            self._cached_image_files is None
            or self._cached_images_dir != resolved_img_dir
        ):
            self._cached_images_dir = resolved_img_dir
            self._cached_image_files = image_files
        else:
            image_files = self._cached_image_files

        if max_images is not None and max_images > 0:
            image_files = image_files[:max_images]

        self.logger.info(
            f"Found {len(image_files)} images in {resolved_img_dir} to process."
        )

        # 1. Clean existing annotations from scratch
        clean_dataset_annotations(
            labels_dir=resolved_lbl_dir,
            dataset_root=resolved_img_dir.parent,
            logger=self.logger,
        )

        if not image_files:
            self.logger.warning("No images found to process.")
            return {
                "images_processed": 0,
                "total_candidates": 0,
                "thresholds": {},
                "labels_written": 0,
                "candidates_df": pl.DataFrame(),
                "filtered_df": pl.DataFrame(),
            }

        # 2. Extract multi-view candidates across all images using batched inference
        records: dict[str, list[Any]] = {
            "image_stem": [],
            "label_path": [],
            "img_width": [],
            "img_height": [],
            "class_id": [],
            "score": [],
            "x1": [],
            "y1": [],
            "x2": [],
            "y2": [],
            "w": [],
            "h": [],
            "size_bin": [],
        }

        self.logger.info(
            f"Running multi-view candidate inference (batch_size={self.batch_size}, "
            f"half={self.half}, workers={self.workers})..."
        )

        chunks = [
            image_files[i : i + self.batch_size]
            for i in range(0, len(image_files), self.batch_size)
        ]

        total_batches = len(chunks)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            with tqdm(total=len(image_files), desc="Multi-view Inference") as pbar:
                for chunk_paths in chunks:
                    # Parallel image decode via thread pool
                    cv_imgs = list(pool.map(lambda p: cv2.imread(str(p)), chunk_paths))
                    # AVX2 SIMD flip with None guard
                    cv_flipped = list(
                        pool.map(
                            lambda im: cv2.flip(im, 1) if im is not None else None,
                            cv_imgs,
                        )
                    )

                    valid_items = [
                        (p, im, flp)
                        for p, im, flp in zip(chunk_paths, cv_imgs, cv_flipped)
                        if im is not None and flp is not None
                    ]
                    if not valid_items:
                        pbar.update(len(chunk_paths))
                        continue

                    v_paths, v_imgs, v_flips = zip(*valid_items)

                    predict_kwargs: dict[str, Any] = {
                        "batch": len(v_paths),
                        "half": self.half,
                        "conf": self.conf,
                        "verbose": False,
                    }
                    if self.device is not None:
                        predict_kwargs["device"] = self.device

                    preds_norm = self.model.predict(list(v_imgs), **predict_kwargs)
                    preds_flip = self.model.predict(list(v_flips), **predict_kwargs)

                    for j, img_path in enumerate(v_paths):
                        im = v_imgs[j]
                        img_height, img_width = im.shape[:2]

                        rel_path = img_path.relative_to(resolved_img_dir)
                        target_label = resolved_lbl_dir / rel_path.with_suffix(".txt")

                        raw = [
                            _inverse_transform([preds_norm[j]], "normal"),
                            _inverse_transform([preds_flip[j]], "flip"),
                        ]
                        candidates = _nms_aggregate(
                            raw, iou_threshold=self.iou_threshold
                        )

                        for cand in candidates:
                            sb = assign_size_bin(cand["w"], cand["h"])
                            records["image_stem"].append(img_path.stem)
                            records["label_path"].append(str(target_label))
                            records["img_width"].append(img_width)
                            records["img_height"].append(img_height)
                            records["class_id"].append(cand["class"])
                            records["score"].append(cand["score"])
                            records["x1"].append(cand["box"][0])
                            records["y1"].append(cand["box"][1])
                            records["x2"].append(cand["box"][2])
                            records["y2"].append(cand["box"][3])
                            records["w"].append(cand["w"])
                            records["h"].append(cand["h"])
                            records["size_bin"].append(sb)

                    pbar.update(len(chunk_paths))

        candidates_df = pl.DataFrame(
            records,
            schema={
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
            },
        )

        total_candidates = len(candidates_df)
        self.logger.info(
            f"Extracted {total_candidates} total multi-view candidate detections across {total_batches} batches."
        )

        # 3. Save raw candidates to Parquet if specified
        if save_candidates_path is not None:
            parquet_dest = Path(save_candidates_path)
            parquet_dest.parent.mkdir(parents=True, exist_ok=True)
            candidates_df.write_parquet(parquet_dest)
            self.logger.info(f"Saved raw candidate detections to {parquet_dest}")

        if total_candidates == 0:
            self.logger.warning("No candidate detections were found.")
            return {
                "images_processed": len(image_files),
                "total_candidates": 0,
                "thresholds": {},
                "labels_written": 0,
                "candidates_df": candidates_df,
                "filtered_df": pl.DataFrame(schema=EMPTY_FILTERED_SCHEMA),
            }

        # 4. Compute optimal ground threshold per COCO size bin
        thresholds: dict[str, float] = {}
        filter_expr = pl.lit(False)

        for sb in SIZE_BINS:
            sb_scores = (
                candidates_df.filter(pl.col("size_bin") == sb)["score"]
                .to_numpy()
                .astype(float)
            )
            opt_thresh = ground_threshold(sb_scores)
            thresholds[sb] = opt_thresh
            self.logger.info(
                f"Size bin '{sb}': count={len(sb_scores)}, ground threshold={opt_thresh:.4f}"
            )
            filter_expr = filter_expr | (
                (pl.col("size_bin") == sb) & (pl.col("score") >= opt_thresh)
            )

        # 5. Filter candidates
        filtered_df = candidates_df.filter(filter_expr)
        self.logger.info(
            f"Filtered candidates from {total_candidates} to {len(filtered_df)} "
            f"using ground thresholds."
        )

        # 6. Save YOLO format annotations (omit empty txt files)
        labels_written = 0
        if len(filtered_df) > 0:
            filtered_df = filtered_df.with_columns(
                x_center=(
                    (pl.col("x1") + pl.col("x2")) / (2.0 * pl.col("img_width"))
                ).clip(0.0, 1.0),
                y_center=(
                    (pl.col("y1") + pl.col("y2")) / (2.0 * pl.col("img_height"))
                ).clip(0.0, 1.0),
                w_norm=((pl.col("x2") - pl.col("x1")) / pl.col("img_width")).clip(
                    0.0, 1.0
                ),
                h_norm=((pl.col("y2") - pl.col("y1")) / pl.col("img_height")).clip(
                    0.0, 1.0
                ),
            )

            filtered_df = filtered_df.with_columns(
                line=pl.concat_str(
                    [
                        pl.col("class_id").cast(pl.String),
                        pl.lit(" "),
                        pl.col("x_center").round(6).cast(pl.String),
                        pl.lit(" "),
                        pl.col("y_center").round(6).cast(pl.String),
                        pl.lit(" "),
                        pl.col("w_norm").round(6).cast(pl.String),
                        pl.lit(" "),
                        pl.col("h_norm").round(6).cast(pl.String),
                    ]
                )
            )

            grouped = filtered_df.group_by("label_path").agg(
                content=pl.col("line").str.join("\n")
            )

            for row in grouped.iter_rows(named=True):
                target_file = Path(row["label_path"])
                target_file.parent.mkdir(parents=True, exist_ok=True)
                target_file.write_text(row["content"] + "\n", encoding="utf-8")
                labels_written += 1
        else:
            filtered_df = pl.DataFrame(schema=EMPTY_FILTERED_SCHEMA)

        self.logger.info(
            f"Saved {labels_written} YOLO annotation files into {resolved_lbl_dir}"
        )

        return {
            "images_processed": len(image_files),
            "total_candidates": total_candidates,
            "thresholds": thresholds,
            "labels_written": labels_written,
            "candidates_df": candidates_df,
            "filtered_df": filtered_df,
        }


def generate_pseudo_labels(
    model: YOLO | str | Path,
    images: Path | str,
    labels: Path | str | None = None,
    save_candidates_path: Path | str | None = None,
    max_images: int | None = None,
    batch_size: int = 32,
    half: bool = True,
    workers: int = 4,
    conf: float = 0.001,
    iou: float = 0.5,
    device: str | torch.device | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Functional convenience wrapper for PseudoLabelGenerator."""
    generator = PseudoLabelGenerator(
        teacher_model=model,
        batch_size=batch_size,
        half=half,
        workers=workers,
        conf=conf,
        iou_threshold=iou,
        device=device,
        logger=logger,
    )
    return generator.generate(
        images_dir=images,
        labels_dir=labels,
        max_images=max_images,
        save_candidates_path=save_candidates_path,
    )


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
        Path("code/noisy_student_pipeline/YOLO26m-best-naive.pt"),
        "--model",
        "-m",
        help="Path to pretrained YOLO teacher model weights.",
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
    save_candidates: Optional[Path] = typer.Option(
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
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        "-b",
        help="Inference batch size.",
        min=1,
    ),
    half: bool = typer.Option(
        True,
        "--half/--no-half",
        help="Use FP16 half-precision inference on GPU.",
    ),
    workers: int = typer.Option(
        4,
        "--workers",
        "-w",
        help="Number of worker threads for parallel I/O and SIMD transforms.",
        min=1,
    ),
    conf: float = typer.Option(
        0.001,
        "--conf",
        "-c",
        help="Multiview candidate confidence threshold.",
        min=0.0,
        max=1.0,
    ),
    iou: float = typer.Option(
        0.5,
        "--iou",
        help="IoU threshold for multi-view candidate NMS aggregation.",
        min=0.0,
        max=1.0,
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        "-d",
        help="PyTorch device identifier (e.g. 'cuda:0', 'cpu').",
    ),
) -> None:
    """Run pseudo-label generation on an unannotated image dataset."""
    generate_pseudo_labels(
        model=model,
        images=images,
        labels=labels,
        save_candidates_path=save_candidates,
        max_images=max_images,
        batch_size=batch_size,
        half=half,
        workers=workers,
        conf=conf,
        iou=iou,
        device=device,
    )


if __name__ == "__main__":
    app()
