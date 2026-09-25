"""
Semi-Supervised Object Detection (SSOD) dataset handler for YOLO Mean Teacher training.

Supports:
1. Simultaneous loading of annotated Ground-Truth (train & val) and Pseudo-Labeled (train only).
2. Augmentation policy:
   - HSV, Scale, Translate, and Fliplr applied to both GT and Pseudo datasets.
   - Mosaic applied strictly to GT dataset, closed during the last 10% of epochs.
   - All other augmentations (MixUp, CutMix, CopyPaste, degrees, shear, perspective, flipud, Albumentations) disabled.
3. Dynamic pseudo-label accommodation:
   - Refreshing pseudo-labels dynamically at epoch start (from disk or in-memory Polars DataFrame).
4. Epoch item sequencing:
   - 25% of GT items uniformly selected for the tail of each epoch.
   - 75% of GT items shuffled with pseudo-labeled items in the head of each epoch.
5. Collate function exposing standard YOLO tensors plus `is_pseudo` sample mask.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import yaml

from ultralytics.data.augment import (
    Compose,
    Format,
    LetterBox,
    Mosaic,
    RandomFlip,
    RandomHSV,
    RandomPerspective,
)
from ultralytics.data.dataset import YOLODataset

try:
    from ssod.utilities import get_logger
except ImportError:
    from utilities import get_logger  # type: ignore[import-not-found,no-redef]


VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class SSODAugmentConfig:
    """Hyperparameters and configuration for SSOD augmentations and epoch policies."""

    # Color jitter
    hsv_h: float = 0.01257
    hsv_s: float = 0.50473
    hsv_v: float = 0.45312

    # Spatial augmentations
    translate: float = 0.11679
    scale: float = 0.51241
    fliplr: float = 0.49473

    # Ground-truth mosaic augmentation
    mosaic: float = 1.0

    # SSOD epoch policy
    tail_ratio: float = 0.25  # 25% of GT uniformly placed at tail of epoch
    close_mosaic_ratio: float = 0.10  # Disable mosaic during last 10% of epochs


def build_gt_transforms(
    dataset: Any,
    imgsz: int = 640,
    hyp: SSODAugmentConfig | None = None,
    mosaic: bool = True,
) -> Compose:
    """
    Build transforms for Ground-Truth training dataset.

    Includes Mosaic (if mosaic=True), Scale & Translate (RandomPerspective),
    RandomHSV, and RandomFlip(fliplr).
    All other augmentations are disabled.
    """
    cfg = hyp or SSODAugmentConfig()

    affine = RandomPerspective(
        degrees=0.0,
        translate=cfg.translate,
        scale=cfg.scale,
        shear=0.0,
        perspective=0.0,
        pre_transform=LetterBox(new_shape=(imgsz, imgsz)),
    )

    if mosaic and cfg.mosaic > 0.0:
        mosaic_transform = Mosaic(dataset, imgsz=imgsz, p=cfg.mosaic)
        pre_transform = Compose([mosaic_transform, affine])
    else:
        pre_transform = affine

    transforms = [
        pre_transform,
        RandomHSV(hgain=cfg.hsv_h, sgain=cfg.hsv_s, vgain=cfg.hsv_v),
        RandomFlip(direction="horizontal", p=cfg.fliplr),
        Format(
            bbox_format="xywh",
            normalize=True,
            return_mask=False,
            return_keypoint=False,
            return_obb=False,
            batch_idx=True,
            mask_ratio=4,
            mask_overlap=True,
            bgr=0.0,
        ),
    ]
    return Compose(transforms)


def build_pseudo_transforms(
    dataset: Any,
    imgsz: int = 640,
    hyp: SSODAugmentConfig | None = None,
) -> Compose:
    """
    Build transforms for Pseudo-Labeled training dataset.

    Mosaic is NEVER included. Includes Scale & Translate (RandomPerspective),
    RandomHSV, and RandomFlip(fliplr). All other augmentations disabled.
    """
    cfg = hyp or SSODAugmentConfig()

    affine = RandomPerspective(
        degrees=0.0,
        translate=cfg.translate,
        scale=cfg.scale,
        shear=0.0,
        perspective=0.0,
        pre_transform=LetterBox(new_shape=(imgsz, imgsz)),
    )

    transforms = [
        affine,
        RandomHSV(hgain=cfg.hsv_h, sgain=cfg.hsv_s, vgain=cfg.hsv_v),
        RandomFlip(direction="horizontal", p=cfg.fliplr),
        Format(
            bbox_format="xywh",
            normalize=True,
            return_mask=False,
            return_keypoint=False,
            return_obb=False,
            batch_idx=True,
            mask_ratio=4,
            mask_overlap=True,
            bgr=0.0,
        ),
    ]
    return Compose(transforms)


def build_val_transforms(imgsz: int = 640) -> Compose:
    """Build validation transforms: LetterBox resize + Format only (no augmentations)."""
    return Compose(
        [
            LetterBox(new_shape=(imgsz, imgsz), scaleup=False),
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=False,
                return_keypoint=False,
                return_obb=False,
                batch_idx=True,
                mask_ratio=4,
                mask_overlap=True,
                bgr=0.0,
            ),
        ]
    )


class PseudoYOLODataset(YOLODataset):
    """
    YOLO dataset tailored for dynamic pseudo-labeled pools.
    Supports in-memory Polars DataFrame updates as well as disk .txt re-scans.
    Supports filter_empty=True (only training on images with pseudo-labels).
    Gracefully handles empty pseudo-label sets (len=0).
    """

    def __init__(
        self,
        images_dir: Path | str,
        labels_dir: Path | str,
        data: dict[str, Any] | None = None,
        imgsz: int = 640,
        hyp: SSODAugmentConfig | None = None,
        filter_empty: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        self.images_dir = Path(images_dir).resolve()
        self.labels_dir = Path(labels_dir).resolve()
        self.filter_empty = filter_empty
        self.logger = logger or get_logger()
        self.augment_cfg = hyp or SSODAugmentConfig()

        data_dict = data or {"names": {0: "object"}, "nc": 1}

        # Initialize BaseDataset attributes without calling full YOLODataset.__init__ caching
        # to support dynamic/empty initial state
        super().__init__(
            img_path=str(self.images_dir),
            imgsz=imgsz,
            data=data_dict,
            task="detect",
            augment=True,
        )

        # Build initial label index
        self.reload(filter_empty=self.filter_empty)

    def get_labels(self) -> list[dict[str, Any]]:
        """Override standard label loading to load from self.labels_dir."""
        return self._load_labels_from_disk(filter_empty=self.filter_empty)

    def _find_image_files(self) -> list[Path]:
        if not self.images_dir.exists():
            return []
        return sorted(
            [
                p
                for p in self.images_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS
            ]
        )

    def _load_labels_from_disk(self, filter_empty: bool = True) -> list[dict[str, Any]]:
        img_files = self._find_image_files()
        labels_list: list[dict[str, Any]] = []
        im_files_list: list[str] = []

        for img_path in img_files:
            rel = img_path.relative_to(self.images_dir)
            label_path = self.labels_dir / rel.with_suffix(".txt")

            boxes: list[list[float]] = []
            classes: list[int] = []

            if label_path.exists() and label_path.is_file():
                content = label_path.read_text(encoding="utf-8").strip()
                if content:
                    for line in content.splitlines():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(float(parts[0]))
                            xc, yc, w, h = (float(v) for v in parts[1:5])
                            classes.append(cls_id)
                            boxes.append([xc, yc, w, h])

            if filter_empty and len(boxes) == 0:
                continue

            # Read image shape (fast header or imread fallback)
            im = cv2.imread(str(img_path))
            if im is None:
                continue
            h, w = im.shape[:2]

            cls_arr = (
                np.array(classes, dtype=np.float32).reshape(-1, 1)
                if classes
                else np.empty((0, 1), dtype=np.float32)
            )
            box_arr = (
                np.array(boxes, dtype=np.float32).reshape(-1, 4)
                if boxes
                else np.empty((0, 4), dtype=np.float32)
            )

            record = {
                "im_file": str(img_path),
                "shape": (h, w),
                "cls": cls_arr,
                "bboxes": box_arr,
                "segments": [],
                "keypoints": None,
                "normalized": True,
                "bbox_format": "xywh",
            }
            labels_list.append(record)
            im_files_list.append(str(img_path))

        self.labels = labels_list
        self.im_files = im_files_list
        self.ni = len(labels_list)
        return labels_list

    def load_from_dataframe(
        self,
        df: pl.DataFrame,
        filter_empty: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Populate dataset labels directly from an in-memory Polars DataFrame.
        Expected schema columns: 'image_stem' or 'label_path', 'class_id',
        'x_center', 'y_center', 'w_norm', 'h_norm', 'img_width', 'img_height'.
        """
        all_imgs = {p.stem: p for p in self._find_image_files()}
        labels_list: list[dict[str, Any]] = []
        im_files_list: list[str] = []

        if len(df) == 0:
            if not filter_empty:
                for stem, img_path in all_imgs.items():
                    im = cv2.imread(str(img_path))
                    if im is None:
                        continue
                    h, w = im.shape[:2]
                    record = {
                        "im_file": str(img_path),
                        "shape": (h, w),
                        "cls": np.empty((0, 1), dtype=np.float32),
                        "bboxes": np.empty((0, 4), dtype=np.float32),
                        "segments": [],
                        "keypoints": None,
                        "normalized": True,
                        "bbox_format": "xywh",
                    }
                    labels_list.append(record)
                    im_files_list.append(str(img_path))
            self.labels = labels_list
            self.im_files = im_files_list
            self.ni = len(labels_list)
            return labels_list

        detected_stems = set()
        for keys, group_df in df.group_by("image_stem"):
            stem = (
                keys[0]
                if isinstance(keys, tuple)
                else (keys if isinstance(keys, str) else group_df["image_stem"][0])
            )

            img_path = all_imgs.get(stem)
            if img_path is None:
                continue

            detected_stems.add(stem)
            img_width = group_df["img_width"][0]
            img_height = group_df["img_height"][0]

            classes = group_df["class_id"].to_numpy().astype(np.float32).reshape(-1, 1)
            xc = group_df["x_center"].to_numpy().astype(np.float32)
            yc = group_df["y_center"].to_numpy().astype(np.float32)
            wn = group_df["w_norm"].to_numpy().astype(np.float32)
            hn = group_df["h_norm"].to_numpy().astype(np.float32)
            bboxes = np.stack([xc, yc, wn, hn], axis=1)

            record = {
                "im_file": str(img_path),
                "shape": (int(img_height), int(img_width)),
                "cls": classes,
                "bboxes": bboxes,
                "segments": [],
                "keypoints": None,
                "normalized": True,
                "bbox_format": "xywh",
            }
            labels_list.append(record)
            im_files_list.append(str(img_path))

        if not filter_empty:
            for stem, img_path in all_imgs.items():
                if stem not in detected_stems:
                    im = cv2.imread(str(img_path))
                    if im is None:
                        continue
                    h, w = im.shape[:2]
                    record = {
                        "im_file": str(img_path),
                        "shape": (h, w),
                        "cls": np.empty((0, 1), dtype=np.float32),
                        "bboxes": np.empty((0, 4), dtype=np.float32),
                        "segments": [],
                        "keypoints": None,
                        "normalized": True,
                        "bbox_format": "xywh",
                    }
                    labels_list.append(record)
                    im_files_list.append(str(img_path))

        self.labels = labels_list
        self.im_files = im_files_list
        self.ni = len(labels_list)
        return labels_list

    def reload(
        self,
        labels_dir: Path | str | None = None,
        filtered_df: pl.DataFrame | None = None,
        filter_empty: bool | None = None,
    ) -> int:
        """
        Refresh pseudo-labels either from an in-memory DataFrame or from disk .txt files.
        Returns the number of loaded pseudo-labeled images.
        """
        if labels_dir is not None:
            self.labels_dir = Path(labels_dir).resolve()
        if filter_empty is not None:
            self.filter_empty = filter_empty

        if filtered_df is not None:
            self.load_from_dataframe(filtered_df, filter_empty=self.filter_empty)
        else:
            self._load_labels_from_disk(filter_empty=self.filter_empty)

        return len(self.labels)


class SSODSampler(Sampler[int]):
    """
    Epoch-aware sampler for Semi-Supervised Object Detection (SSOD).

    Each epoch:
      1. Uniformly select 25% of GT items (without replacement) to place at the tail.
      2. The remaining 75% of GT items are mixed and shuffled with all pseudo-labeled items.
      3. Yields indices: [ Shuffled(75% GT + 100% Pseudo) ] + [ 25% GT tail ].
    """

    def __init__(
        self,
        gt_len: int,
        pl_len: int,
        tail_ratio: float = 0.25,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.gt_len = gt_len
        self.pl_len = pl_len
        self.tail_ratio = tail_ratio
        self.seed = seed
        self.epoch = 0
        self._indices: list[int] = []
        self._generate_indices()

    def set_epoch(self, epoch: int) -> None:
        """Update epoch index and regenerate deterministic sampling sequence."""
        self.epoch = epoch
        self._generate_indices()

    def update_lengths(self, gt_len: int, pl_len: int) -> None:
        """Update dataset lengths when pseudo-labels change."""
        self.gt_len = gt_len
        self.pl_len = pl_len
        self._generate_indices()

    def _generate_indices(self) -> list[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        n_gt = self.gt_len
        n_pl = self.pl_len

        if n_gt == 0 and n_pl == 0:
            self._indices = []
            return self._indices

        if n_gt == 0:
            # No GT items; only pseudo items
            pl_indices = list(range(n_pl))
            rng.shuffle(pl_indices)
            self._indices = pl_indices
            return self._indices

        # Uniformly partition GT into 25% tail and 75% head
        n_tail = (
            max(1, round(n_gt * self.tail_ratio))
            if (n_gt > 1 and self.tail_ratio > 0.0)
            else (1 if self.tail_ratio > 0.0 else 0)
        )
        if n_tail > n_gt:
            n_tail = n_gt

        perm_gt = rng.permutation(n_gt)
        gt_tail = perm_gt[:n_tail].tolist()
        gt_head = perm_gt[n_tail:].tolist()

        # Pseudo items have global index range [n_gt, n_gt + n_pl - 1]
        pl_indices = list(range(n_gt, n_gt + n_pl))

        # 75% GT + all PL items shuffled together
        head_pool = gt_head + pl_indices
        rng.shuffle(head_pool)

        # Sequence: [ Shuffled Head Pool ] + [ Uniformly Selected GT Tail ]
        self._indices = head_pool + gt_tail
        return self._indices

    def __iter__(self):
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)


class SSODDataset(Dataset):
    """
    Combined PyTorch Dataset for SSOD training.

    Wraps Ground-Truth dataset and Pseudo-Labeled dataset, coordinating
    separate augmentation pipelines (Mosaic on GT, no Mosaic on Pseudo),
    dynamic pseudo-label reloading, and sample source tracking (`is_pseudo`).
    """

    def __init__(
        self,
        gt_dataset: YOLODataset,
        pl_dataset: PseudoYOLODataset,
        augment_cfg: SSODAugmentConfig | None = None,
        imgsz: int = 640,
    ) -> None:
        super().__init__()
        self.gt_dataset = gt_dataset
        self.pl_dataset = pl_dataset
        self.augment_cfg = augment_cfg or SSODAugmentConfig()
        self.imgsz = imgsz
        self.is_mosaic_closed = False

        # Build transform pipelines
        self.transforms_gt_open = build_gt_transforms(
            gt_dataset, imgsz=imgsz, hyp=self.augment_cfg, mosaic=True
        )
        self.transforms_gt_closed = build_gt_transforms(
            gt_dataset, imgsz=imgsz, hyp=self.augment_cfg, mosaic=False
        )
        self.transforms_pl = build_pseudo_transforms(
            pl_dataset, imgsz=imgsz, hyp=self.augment_cfg
        )

        # Set initial transforms
        self.gt_dataset.transforms = self.transforms_gt_open
        self.pl_dataset.transforms = self.transforms_pl

    def close_mosaic(self) -> None:
        """Close Mosaic augmentation on GT dataset (for last 10% of epochs)."""
        self.gt_dataset.transforms = self.transforms_gt_closed
        self.is_mosaic_closed = True

    def open_mosaic(self) -> None:
        """Enable Mosaic augmentation on GT dataset."""
        self.gt_dataset.transforms = self.transforms_gt_open
        self.is_mosaic_closed = False

    def reload_pseudo_labels(
        self,
        labels_dir: Path | str | None = None,
        filtered_df: pl.DataFrame | None = None,
        filter_empty: bool = True,
    ) -> int:
        """Refresh pseudo-labels in pl_dataset."""
        count = self.pl_dataset.reload(
            labels_dir=labels_dir,
            filtered_df=filtered_df,
            filter_empty=filter_empty,
        )
        return count

    def __len__(self) -> int:
        return len(self.gt_dataset) + len(self.pl_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        n_gt = len(self.gt_dataset)
        if index < n_gt:
            sample = self.gt_dataset[index]
            sample["is_pseudo"] = False
            sample["source"] = 0  # 0 for Ground Truth
        else:
            sample = self.pl_dataset[index - n_gt]
            sample["is_pseudo"] = True
            sample["source"] = 1  # 1 for Pseudo
        return sample

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Collate batch items into standard Ultralytics YOLO tensors while
        preserving `is_pseudo` and `source` tensors for Mean Teacher loss.
        """
        is_pseudo_list = [b["is_pseudo"] for b in batch]
        source_list = [b["source"] for b in batch]

        # Strip custom non-tensor metadata keys before calling standard YOLODataset.collate_fn
        standard_batch = []
        for b in batch:
            clean = {k: v for k, v in b.items() if k not in {"is_pseudo", "source"}}
            standard_batch.append(clean)

        collated = YOLODataset.collate_fn(standard_batch)
        collated["is_pseudo"] = torch.tensor(is_pseudo_list, dtype=torch.bool)
        collated["source"] = torch.tensor(source_list, dtype=torch.long)
        return collated


class SSODDatasetHandler:
    """
    High-level orchestrator for SSOD datasets and dataloaders in Mean Teacher training.

    Coordinates:
    - GT Train dataset & GT Val dataset
    - Pseudo-labeled Train dataset
    - SSODSampler with 25% GT tail / 75% GT + PL head sequencing
    - Automatic mosaic closing during the last 10% of epochs
    - Dynamic pseudo-label reloading at epoch boundaries
    """

    def __init__(
        self,
        gt_data_yaml: Path | str | None = None,
        gt_train_images: Path | str | None = None,
        gt_train_labels: Path | str | None = None,
        gt_val_images: Path | str | None = None,
        gt_val_labels: Path | str | None = None,
        pl_images_dir: Path | str | None = None,
        pl_labels_dir: Path | str | None = None,
        class_names: dict[int, str] | list[str] | None = None,
        imgsz: int = 640,
        batch_size: int = 10,
        workers: int = 4,
        seed: int = 42,
        augment_cfg: SSODAugmentConfig | None = None,
        tail_ratio: float = 0.25,
        close_mosaic_ratio: float = 0.10,
        filter_empty_pseudo: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or get_logger()
        self.imgsz = imgsz
        self.batch_size = max(1, batch_size)
        self.workers = max(0, workers)
        self.seed = seed
        self.augment_cfg = augment_cfg or SSODAugmentConfig(
            tail_ratio=tail_ratio, close_mosaic_ratio=close_mosaic_ratio
        )
        self.tail_ratio = tail_ratio
        self.close_mosaic_ratio = close_mosaic_ratio
        self.filter_empty_pseudo = filter_empty_pseudo

        # Resolve paths & class metadata
        data_cfg = self._resolve_data_config(
            gt_data_yaml=gt_data_yaml,
            gt_train_images=gt_train_images,
            gt_train_labels=gt_train_labels,
            gt_val_images=gt_val_images,
            gt_val_labels=gt_val_labels,
            pl_images_dir=pl_images_dir,
            pl_labels_dir=pl_labels_dir,
            class_names=class_names,
        )

        self.gt_train_images = data_cfg["gt_train_images"]
        self.gt_val_images = data_cfg["gt_val_images"]
        self.pl_images_dir = data_cfg["pl_images_dir"]
        self.pl_labels_dir = data_cfg["pl_labels_dir"]
        self.data_dict = data_cfg["data_dict"]

        # 1. Initialize Ground-Truth Train Dataset
        self.logger.info(
            f"Initializing Ground-Truth Train Dataset from {self.gt_train_images}..."
        )
        self.gt_train_dataset = YOLODataset(
            img_path=str(self.gt_train_images),
            imgsz=self.imgsz,
            data=self.data_dict,
            task="detect",
            augment=True,
        )

        # 2. Initialize Ground-Truth Val Dataset
        self.val_dataset: YOLODataset | None = None
        if self.gt_val_images is not None and Path(self.gt_val_images).exists():
            self.logger.info(
                f"Initializing Ground-Truth Val Dataset from {self.gt_val_images}..."
            )
            self.val_dataset = YOLODataset(
                img_path=str(self.gt_val_images),
                imgsz=self.imgsz,
                data=self.data_dict,
                task="detect",
                augment=False,
                rect=True,
            )
            self.val_dataset.transforms = build_val_transforms(self.imgsz)

        # 3. Initialize Pseudo-Labeled Dataset
        self.logger.info(
            f"Initializing Pseudo-Labeled Dataset (images={self.pl_images_dir}, labels={self.pl_labels_dir})..."
        )
        self.pl_dataset = PseudoYOLODataset(
            images_dir=self.pl_images_dir,
            labels_dir=self.pl_labels_dir,
            data=self.data_dict,
            imgsz=self.imgsz,
            hyp=self.augment_cfg,
            filter_empty=self.filter_empty_pseudo,
            logger=self.logger,
        )

        # 4. Initialize Combined SSODDataset
        self.train_dataset = SSODDataset(
            gt_dataset=self.gt_train_dataset,
            pl_dataset=self.pl_dataset,
            augment_cfg=self.augment_cfg,
            imgsz=self.imgsz,
        )

        # 5. Initialize SSODSampler
        self.sampler = SSODSampler(
            gt_len=len(self.gt_train_dataset),
            pl_len=len(self.pl_dataset),
            tail_ratio=self.tail_ratio,
            seed=self.seed,
        )

        self.logger.info(
            f"SSOD Dataset initialized: {len(self.gt_train_dataset)} GT train items, "
            f"{len(self.pl_dataset)} initial pseudo items (total={len(self.train_dataset)})."
        )

    def _resolve_data_config(
        self,
        gt_data_yaml: Path | str | None,
        gt_train_images: Path | str | None,
        gt_train_labels: Path | str | None,
        gt_val_images: Path | str | None,
        gt_val_labels: Path | str | None,
        pl_images_dir: Path | str | None,
        pl_labels_dir: Path | str | None,
        class_names: dict[int, str] | list[str] | None,
    ) -> dict[str, Any]:
        """Resolve dataset configuration from YAML or explicit path arguments."""
        data_dict: dict[str, Any] = {}
        resolved_gt_train = gt_train_images
        resolved_gt_val = gt_val_images

        if gt_data_yaml is not None:
            yaml_path = Path(gt_data_yaml).resolve()
            with open(yaml_path, "r", encoding="utf-8") as f:
                raw_cfg = yaml.safe_load(f)

            root = Path(raw_cfg.get("path", "."))
            if not root.is_absolute():
                root = (yaml_path.parent / root).resolve()

            if "train" in raw_cfg:
                resolved_gt_train = (root / raw_cfg["train"]).resolve()
            if "val" in raw_cfg:
                resolved_gt_val = (root / raw_cfg["val"]).resolve()

            raw_names = raw_cfg.get("names", {0: "object"})
            if isinstance(raw_names, list):
                names_dict = {i: n for i, n in enumerate(raw_names)}
            else:
                names_dict = {int(k): str(v) for k, v in raw_names.items()}

            data_dict = {
                "names": names_dict,
                "nc": raw_cfg.get("nc", len(names_dict)),
                "path": str(root),
            }
        else:
            if class_names is not None:
                if isinstance(class_names, list):
                    names_dict = {i: n for i, n in enumerate(class_names)}
                else:
                    names_dict = {int(k): str(v) for k, v in class_names.items()}
            else:
                names_dict = {0: "object"}
            data_dict = {
                "names": names_dict,
                "nc": len(names_dict),
            }

        if resolved_gt_train is None:
            raise ValueError(
                "Ground-truth train images path must be specified via gt_data_yaml or gt_train_images."
            )

        if pl_images_dir is None:
            raise ValueError(
                "Pseudo-labeled images dir (pl_images_dir) must be specified."
            )
        resolved_pl_images = Path(pl_images_dir).resolve()

        if pl_labels_dir is not None:
            resolved_pl_labels = Path(pl_labels_dir).resolve()
        else:
            resolved_pl_labels = resolved_pl_images.parent / "labels"

        return {
            "gt_train_images": resolved_gt_train,
            "gt_val_images": resolved_gt_val,
            "pl_images_dir": resolved_pl_images,
            "pl_labels_dir": resolved_pl_labels,
            "data_dict": data_dict,
        }

    def reload_pseudo_labels(
        self,
        labels_dir: Path | str | None = None,
        filtered_df: pl.DataFrame | None = None,
        filter_empty: bool | None = None,
    ) -> int:
        """
        Reload pseudo-labels dynamically for the current epoch and synchronize the sampler.
        """
        count = self.train_dataset.reload_pseudo_labels(
            labels_dir=labels_dir,
            filtered_df=filtered_df,
            filter_empty=self.filter_empty_pseudo
            if filter_empty is None
            else filter_empty,
        )
        self.sampler.update_lengths(
            gt_len=len(self.gt_train_dataset), pl_len=len(self.pl_dataset)
        )
        self.logger.info(
            f"Reloaded pseudo-labels: {count} active pseudo images. "
            f"Total epoch dataset length: {len(self.train_dataset)}."
        )
        return count

    def set_epoch(
        self,
        epoch: int,
        total_epochs: int,
        dataloader: DataLoader | None = None,
    ) -> None:
        """
        Update epoch index, evaluate mosaic close condition for last 10% of epochs,
        and re-seed sampler sequence.
        """
        close_epochs = max(1, math.ceil(self.close_mosaic_ratio * total_epochs))
        is_last_10_percent = epoch >= (total_epochs - close_epochs)

        if is_last_10_percent:
            if not self.train_dataset.is_mosaic_closed:
                self.logger.info(
                    f"Epoch {epoch}/{total_epochs}: Entering final 10% of epochs "
                    f"(close_mosaic threshold={total_epochs - close_epochs}). Disabling Mosaic."
                )
                self.train_dataset.close_mosaic()
                if dataloader is not None and hasattr(dataloader, "reset"):
                    dataloader.reset()
        else:
            if self.train_dataset.is_mosaic_closed:
                self.logger.info(
                    f"Epoch {epoch}/{total_epochs}: Re-enabling Mosaic augmentation on Ground-Truth."
                )
                self.train_dataset.open_mosaic()
                if dataloader is not None and hasattr(dataloader, "reset"):
                    dataloader.reset()

        self.sampler.set_epoch(epoch)

    def get_train_loader(
        self,
        pin_memory: bool = True,
        drop_last: bool = False,
    ) -> DataLoader:
        """
        Construct PyTorch DataLoader for SSOD training using SSODSampler
        and SSODDataset.collate_fn.
        """
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,
            num_workers=self.workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            collate_fn=SSODDataset.collate_fn,
            drop_last=drop_last,
            persistent_workers=(self.workers > 0),
        )

    def get_val_loader(
        self,
        batch_size: int | None = None,
        pin_memory: bool = True,
    ) -> DataLoader | None:
        """Construct PyTorch DataLoader for Ground-Truth validation evaluation."""
        if self.val_dataset is None:
            return None
        bs = batch_size or (self.batch_size * 2)
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            collate_fn=YOLODataset.collate_fn,
        )
