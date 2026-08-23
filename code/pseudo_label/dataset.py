"""
Dataset preparation, image path resolution, and pseudo-label generation.
"""

from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .utils import load_yaml, save_yaml

if TYPE_CHECKING:
    from ultralytics import YOLO


def resolve_image_list(
    data_cfg: dict[str, Any], split: str, base_yaml_path: Path
) -> list[Path]:
    """
    Return absolute image paths for a dataset split.
    Handles both directory paths and .txt list files (Ultralytics formats).
    """
    root = Path(data_cfg.get("path", "."))
    if not root.is_absolute():
        root = (base_yaml_path.parent / root).resolve()

    entry = root / data_cfg[split]
    if entry.is_dir():
        return sorted(
            p for p in entry.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    if entry.is_file():
        lines = entry.read_text().splitlines()
        return [
            Path(line) if Path(line).is_absolute() else (root / line).resolve()
            for line in lines
            if line.strip()
        ]

    return []


def sync_pseudo_labels(
    source_labels_dir: Path,
    target_labels_dir: Path,
    clean_target: bool = True,
    logger: logging.Logger | None = None,
) -> int:
    """
    Synchronize pseudo-label .txt files from a source directory into target_labels_dir.
    Optionally cleans target directory first to prevent stale labels.
    """
    log = logger or logging.getLogger("pseudo_pipeline")
    target_labels_dir.mkdir(parents=True, exist_ok=True)

    if clean_target:
        for old_txt in target_labels_dir.glob("*.txt"):
            old_txt.unlink()

    copied = 0
    if source_labels_dir.exists():
        for txt in source_labels_dir.glob("*.txt"):
            shutil.copy2(txt, target_labels_dir / txt.name)
            copied += 1

    log.info(f"Synchronized {copied} label files into {target_labels_dir}")
    return copied


def make_pseudo_labels(
    model: YOLO,
    predict_dir: Path,
    video_images_dir: Path,
    video_labels_dir: Path,
    conf: float,
    logger: logging.Logger | None = None,
) -> int:
    """
    Run inference on unlabeled video frames and save YOLO-format .txt labels.
    Uses stream=True to keep memory flat across thousands of frames.
    """
    log = logger or logging.getLogger("pseudo_pipeline")
    predict_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        f"Generating pseudo-labels on {video_images_dir} with conf>={conf:.2f}..."
    )

    results = model.predict(
        source=str(video_images_dir),
        conf=conf,
        save=False,
        save_txt=True,
        save_conf=False,
        project=str(predict_dir.parent),
        name=predict_dir.name,
        exist_ok=True,
        stream=True,  # Keeps memory flat
        verbose=False,
    )

    # Consume the generator to run inference
    for _ in results:
        pass

    src_label_dir = predict_dir / "labels"
    moved = sync_pseudo_labels(
        src_label_dir,
        video_labels_dir,
        clean_target=True,
        logger=log,
    )

    log.info(f"{moved} frames received pseudo-labels at conf>={conf:.2f}")
    return moved


def build_round_yaml(
    round_idx: int,
    orig_data_yaml: Path,
    video_images_dir: Path,
    video_labels_dir: Path,
    round_dir: Path,
    max_pseudo_ratio: float = 1.0,
    seed: int = 42,
    logger: logging.Logger | None = None,
) -> Path:
    """
    Merge original human-labeled training set with this round's pseudo-labeled frames.
    Validation set remains pinned to the original human-labeled val set.
    """
    log = logger or logging.getLogger("pseudo_pipeline")
    orig_cfg = load_yaml(orig_data_yaml)
    orig_train_imgs = resolve_image_list(orig_cfg, "train", orig_data_yaml)

    # Find pseudo images that have corresponding label txt files
    pseudo_imgs = [
        video_images_dir / f"{p.stem}.jpg"
        for p in video_labels_dir.glob("*.txt")
    ]
    pseudo_imgs = [p for p in pseudo_imgs if p.exists()]

    # Deterministically shuffle and sample up to max_pseudo_ratio
    max_pseudo = max(0, int(len(orig_train_imgs) * max_pseudo_ratio))
    random.Random(seed + round_idx).shuffle(pseudo_imgs)
    pseudo_imgs = pseudo_imgs[:max_pseudo]

    # Write merged image list
    round_dir.mkdir(parents=True, exist_ok=True)
    merged_list_path = round_dir / "train_merged.txt"
    with open(merged_list_path, "w", encoding="utf-8") as f:
        for p in orig_train_imgs + pseudo_imgs:
            f.write(f"{p}\n")

    # Write round data.yaml
    round_cfg = dict(orig_cfg)
    round_cfg["train"] = str(merged_list_path)  # Point train to merged list; val is untouched
    round_yaml_path = round_dir / "data.yaml"
    save_yaml(round_cfg, round_yaml_path)

    log.info(
        f"Round {round_idx}: train set = {len(orig_train_imgs)} real + "
        f"{len(pseudo_imgs)} pseudo = {len(orig_train_imgs) + len(pseudo_imgs)} images"
    )
    return round_yaml_path


def build_tuning_subset_yaml(
    orig_data_yaml: Path,
    tune_dir: Path,
    fraction: float = 0.10,
    seed: int = 42,
    logger: logging.Logger | None = None,
) -> Path:
    """
    Subsample the training split for fast hyperparameter tuning.
    Validation split remains full-size.
    """
    log = logger or logging.getLogger("pseudo_pipeline")
    orig_cfg = load_yaml(orig_data_yaml)
    train_imgs = resolve_image_list(orig_cfg, "train", orig_data_yaml)

    rng = random.Random(seed)
    k = max(1, int(len(train_imgs) * fraction))
    subset = rng.sample(train_imgs, k)

    tune_dir.mkdir(parents=True, exist_ok=True)
    subset_list_path = tune_dir / "train_subset.txt"
    with open(subset_list_path, "w", encoding="utf-8") as f:
        for p in subset:
            f.write(f"{p}\n")

    subset_cfg = dict(orig_cfg)
    subset_cfg["train"] = str(subset_list_path)
    subset_yaml_path = tune_dir / "data_subset.yaml"
    save_yaml(subset_cfg, subset_yaml_path)

    log.info(
        f"Tuning subset: {k}/{len(train_imgs)} train images ({fraction:.0%}), seed={seed}"
    )
    return subset_yaml_path
