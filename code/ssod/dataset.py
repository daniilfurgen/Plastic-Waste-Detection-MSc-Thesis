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

    # Strong Student augmentation on pseudo-labeled images (weak/strong asymmetry)
    pl_strong: bool = False
    strong_hsv_h: float = 0.03
    strong_hsv_s: float = 0.7
    strong_hsv_v: float = 0.6
    blur_p: float = 0.3
    blur_max_kernel: int = 7
    erase_p: float = 0.5
    erase_max_patches: int = 3
    erase_area: tuple[float, float] = (0.01, 0.08)
    erase_drop_ratio: float = 0.5  # drop pseudo boxes at least this fraction erased

    # SSOD epoch policy
    tail_ratio: float = 0.25  # 25% of GT uniformly placed at tail of epoch
    close_mosaic_ratio: float = 0.10  # Disable mosaic during last 10% of epochs


class RandomGaussianBlur:
    """Blur the image with a random odd Gaussian kernel with probability p."""

    def __init__(self, p: float = 0.3, max_kernel: int = 7) -> None:
        self.p = p
        self.max_kernel = max(3, max_kernel)

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if np.random.random() >= self.p:
            return labels
        kernel = int(np.random.choice(np.arange(3, self.max_kernel + 1, 2)))
        labels["img"] = cv2.GaussianBlur(labels["img"], (kernel, kernel), 0)
        return labels


class BoxAwareRandomErasing:
    """
    Erase random rectangular patches with probability p and drop boxes whose area is
    at least `drop_ratio` erased, so the Student is not asked to detect invisible objects.
    Must run after RandomPerspective (instances in absolute pixel coordinates).
    """

    def __init__(
        self,
        p: float = 0.5,
        max_patches: int = 3,
        area: tuple[float, float] = (0.01, 0.08),
        drop_ratio: float = 0.5,
    ) -> None:
        self.p = p
        self.max_patches = max(1, max_patches)
        self.area = area
        self.drop_ratio = drop_ratio

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if np.random.random() >= self.p:
            return labels

        img = labels["img"]
        h, w = img.shape[:2]
        erased = np.zeros((h, w), dtype=bool)

        # Place patches inside the image content, not on letterbox / affine border padding (value 114)
        content = (img != 114).any(axis=2)
        rows, cols = np.flatnonzero(content.any(axis=1)), np.flatnonzero(content.any(axis=0))
        if len(rows) == 0 or len(cols) == 0:
            return labels
        cy0, cy1, cx0, cx1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        ch, cw = cy1 - cy0, cx1 - cx0

        for _ in range(np.random.randint(1, self.max_patches + 1)):
            patch_area = np.random.uniform(*self.area) * ch * cw
            aspect = np.exp(np.random.uniform(np.log(0.3), np.log(3.3)))
            ph = int(min(ch, round(math.sqrt(patch_area * aspect))))
            pw = int(min(cw, round(math.sqrt(patch_area / aspect))))
            if ph < 1 or pw < 1:
                continue
            y0 = cy0 + np.random.randint(0, ch - ph + 1)
            x0 = cx0 + np.random.randint(0, cw - pw + 1)
            img[y0 : y0 + ph, x0 : x0 + pw] = np.random.randint(
                0, 256, (ph, pw, img.shape[2]), dtype=img.dtype
            )
            erased[y0 : y0 + ph, x0 : x0 + pw] = True

        instances = labels["instances"]
        if len(instances) > 0:
            instances.convert_bbox(format="xyxy")
            if instances.normalized:
                instances.denormalize(w, h)
            boxes = np.round(instances.bboxes).astype(int)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
            keep = np.ones(len(boxes), dtype=bool)
            for k, (x1, y1, x2, y2) in enumerate(boxes):
                if x2 > x1 and y2 > y1:
                    keep[k] = erased[y1:y2, x1:x2].mean() < self.drop_ratio
            labels["instances"] = instances[keep]
            labels["cls"] = labels["cls"][keep]

        labels["img"] = img
        return labels


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
    RandomHSV, and RandomFlip(fliplr). With `pl_strong`, HSV gains are raised and
    Gaussian blur plus box-aware random erasing are added. All other augmentations disabled.
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

    if cfg.pl_strong:
        # Erasing must directly follow the affine step: it detects padding by its exact fill value (114)
        photometric = [
            BoxAwareRandomErasing(
                p=cfg.erase_p,
                max_patches=cfg.erase_max_patches,
                area=cfg.erase_area,
                drop_ratio=cfg.erase_drop_ratio,
            ),
            RandomHSV(hgain=cfg.strong_hsv_h, sgain=cfg.strong_hsv_s, vgain=cfg.strong_hsv_v),
            RandomGaussianBlur(p=cfg.blur_p, max_kernel=cfg.blur_max_kernel),
        ]
    else:
        photometric = [RandomHSV(hgain=cfg.hsv_h, sgain=cfg.hsv_s, vgain=cfg.hsv_v)]

    transforms = [
        affine,
        *photometric,
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


def split_pl_weight(sample: dict[str, Any]) -> dict[str, Any]:
    """Split a (N, 2) [class_id, weight] `cls` tensor into `cls` (N, 1) and `pl_weight` (N,)."""
    cls = sample["cls"]
    if cls.ndim == 2 and cls.shape[1] == 2:
        sample["cls"] = cls[:, :1].contiguous()
        sample["pl_weight"] = cls[:, 1].contiguous().float()
    else:
        sample["pl_weight"] = torch.ones(cls.shape[0], dtype=torch.float32)
    return sample


class PseudoYOLODataset(YOLODataset):
    """
    YOLO dataset tailored for dynamic pseudo-labeled pools.
    Supports in-memory Polars DataFrame updates as well as disk .txt re-scans.
    Supports filter_empty=True (only training on images with pseudo-labels).
    Gracefully handles empty pseudo-label sets (len=0).
    Label `cls` arrays are (N, 2): [class_id, per-box loss weight]. Every transform indexes
    `cls` row-wise alongside the boxes, so the weight stays aligned through augmentation;
    __getitem__ splits it back into `cls` (N, 1) and `pl_weight` (N,).
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
        self._cached_image_files: list[Path] | None = None
        self._cached_stem_to_path: dict[str, Path] | None = None

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
        self._sync_buffers()

    def _sync_buffers(self) -> None:
        """Synchronize BaseDataset image caching and dimension buffers to match current items."""
        n = len(self.im_files)
        self.ni = n
        self.ims = [None] * n
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        self.im_hw0 = [None] * n
        self.im_hw = [None] * n
        self.buffer = []
        batch_size = getattr(self, "batch_size", 16) or 16
        augment = getattr(self, "augment", True)
        self.max_buffer_length = min((self.ni, batch_size * 8, 1000)) if augment else 0

    def load_image(
        self, i: int, rect_mode: bool = True
    ) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """Load an image from dataset index 'i' with defensive bounds checking and buffer synchronization."""
        if i < 0 or i >= len(self.im_files):
            raise IndexError(
                f"Index {i} out of bounds for PseudoYOLODataset with {len(self.im_files)} images."
            )

        if len(self.ims) != len(self.im_files) or len(self.npy_files) != len(
            self.im_files
        ):
            self._sync_buffers()

        im = self.ims[i]
        f = self.im_files[i]
        fn = self.npy_files[i]

        if im is None:
            if fn.exists():
                try:
                    im = np.load(fn)
                except Exception as e:
                    self.logger.warning(
                        f"Removing corrupt *.npy image file {fn} due to: {e}"
                    )
                    Path(fn).unlink(missing_ok=True)
                    im = cv2.imread(f, flags=self.cv2_flag)
            else:
                im = cv2.imread(f, flags=self.cv2_flag)

            if im is None:
                raise FileNotFoundError(f"Image Not Found: {f}")

            h0, w0 = im.shape[:2]
            if rect_mode:
                r = self.imgsz / max(h0, w0)
                if r != 1:
                    w, h = (
                        min(math.ceil(w0 * r), self.imgsz),
                        min(math.ceil(h0 * r), self.imgsz),
                    )
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):
                im = cv2.resize(
                    im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR
                )

            if im.ndim == 2:
                im = im[..., None]

            if self.augment:
                self.ims[i] = im
                self.im_hw0[i] = (h0, w0)
                self.im_hw[i] = im.shape[:2]
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        return split_pl_weight(sample)

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """YOLODataset collation plus `pl_weight` concatenated in box order."""
        weights = [b.pop("pl_weight") for b in batch]
        collated = YOLODataset.collate_fn(batch)
        collated["pl_weight"] = torch.cat(weights, 0).float()
        return collated

    def get_labels(self) -> list[dict[str, Any]]:
        """Override standard label loading to load from self.labels_dir."""
        return self._load_labels_from_disk(filter_empty=self.filter_empty)

    def _find_image_files(self) -> list[Path]:
        if self._cached_image_files is not None:
            return self._cached_image_files
        if not self.images_dir.exists():
            self._cached_image_files = []
            return []
        self._cached_image_files = sorted(
            [
                p
                for p in self.images_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS
            ]
        )
        return self._cached_image_files

    def _get_stem_to_path_map(self) -> dict[str, Path]:
        if self._cached_stem_to_path is not None:
            return self._cached_stem_to_path
        self._cached_stem_to_path = {p.stem: p for p in self._find_image_files()}
        return self._cached_stem_to_path

    def clear_cache(self) -> None:
        self._cached_image_files = None
        self._cached_stem_to_path = None

    def _load_labels_from_disk(self, filter_empty: bool = True) -> list[dict[str, Any]]:
        img_files = self._find_image_files()
        labels_list: list[dict[str, Any]] = []
        im_files_list: list[str] = []

        for img_path in img_files:
            rel = img_path.relative_to(self.images_dir)
            label_path = self.labels_dir / rel.with_suffix(".txt")

            boxes: list[list[float]] = []
            classes: list[int] = []
            weights: list[float] = []

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
                            weights.append(float(parts[5]) if len(parts) >= 6 else 1.0)

            if filter_empty and len(boxes) == 0:
                continue

            # Read image shape (fast header or imread fallback)
            im = cv2.imread(str(img_path))
            if im is None:
                continue
            h, w = im.shape[:2]

            cls_arr = (
                np.stack([classes, weights], axis=1).astype(np.float32)
                if classes
                else np.empty((0, 2), dtype=np.float32)
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
        self._sync_buffers()
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
        all_imgs = self._get_stem_to_path_map()
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
                        "cls": np.empty((0, 2), dtype=np.float32),
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
            self._sync_buffers()
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

            weights = (
                group_df["weight"].to_numpy()
                if "weight" in group_df.columns
                else np.ones(len(group_df))
            )
            classes = np.stack(
                [group_df["class_id"].to_numpy(), weights], axis=1
            ).astype(np.float32)
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
                        "cls": np.empty((0, 2), dtype=np.float32),
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
        self._sync_buffers()
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
      2. The remaining 75% of GT items (repeated `gt_repeat` times) are mixed and shuffled
         with all pseudo-labeled items.
      3. Yields indices: [ Shuffled(gt_repeat x 75% GT + 100% Pseudo) ] + [ 25% GT tail ].
    """

    def __init__(
        self,
        gt_len: int,
        pl_len: int,
        tail_ratio: float = 0.25,
        seed: int = 42,
        gt_repeat: int = 1,
    ) -> None:
        super().__init__()
        self.gt_len = gt_len
        self.pl_len = pl_len
        self.tail_ratio = tail_ratio
        self.seed = seed
        self.gt_repeat = max(1, gt_repeat)
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

        # gt_repeat x 75% GT + all PL items shuffled together
        head_pool = gt_head * self.gt_repeat + pl_indices
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
        if "pl_weight" not in sample:
            sample["pl_weight"] = torch.ones(sample["cls"].shape[0], dtype=torch.float32)
        return sample

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Collate batch items into standard Ultralytics YOLO tensors while
        preserving `is_pseudo` and `source` tensors for Mean Teacher loss.
        """
        is_pseudo_list = [b["is_pseudo"] for b in batch]
        source_list = [b["source"] for b in batch]
        weight_list = [
            b.get("pl_weight", torch.ones(b["cls"].shape[0], dtype=torch.float32))
            for b in batch
        ]

        # Strip custom metadata keys before calling standard YOLODataset.collate_fn
        standard_batch = []
        for b in batch:
            clean = {
                k: v for k, v in b.items() if k not in {"is_pseudo", "source", "pl_weight"}
            }
            standard_batch.append(clean)

        collated = YOLODataset.collate_fn(standard_batch)
        collated["is_pseudo"] = torch.tensor(is_pseudo_list, dtype=torch.bool)
        collated["source"] = torch.tensor(source_list, dtype=torch.long)
        collated["pl_weight"] = torch.cat(weight_list, 0).float()  # aligned with bboxes
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
        gt_repeat: int = 1,
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
            gt_repeat=gt_repeat,
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
        # A ratio of 0 keeps Mosaic open for the whole run
        close_epochs = (
            max(1, math.ceil(self.close_mosaic_ratio * total_epochs))
            if self.close_mosaic_ratio > 0
            else 0
        )
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
        persistent_workers: bool = True,
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
            persistent_workers=persistent_workers and (self.workers > 0),
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
