"""
Semi-Supervised Object Detection (SSOD) dataset handler for YOLO Mean Teacher training.

Supports:
1. Simultaneous loading of annotated Ground-Truth (train & val) and unlabeled images (train only),
   which the Teacher pseudo-labels online at every training step.
2. Augmentation policy:
   - HSV, Scale, Translate, and Fliplr applied to both GT and unlabeled images.
   - Mosaic applied strictly to GT dataset, closed during the last 10% of epochs.
   - Unlabeled images also return a weak Teacher view: the same geometry (scale, translate,
     flip) as the Student's view, without photometric augmentation, so Teacher boxes are
     already in Student coordinates. Erased patches of the Student view are recorded so mostly
     erased pseudo boxes can be dropped after labelling.
   - All other augmentations (MixUp, CutMix, CopyPaste, degrees, shear, perspective, flipud, Albumentations) disabled.
3. A fresh image list for the unlabeled pool at every epoch start.
4. Epoch item sequencing:
   - GT items (repeated `gt_repeat` times) shuffled with unlabeled items.
   - Optional GT tail (`tail_ratio`, off by default): a uniformly selected fraction of GT placed,
     unmixed, at the end of each epoch.
5. Collate function exposing standard YOLO tensors plus `is_pseudo` sample mask, Teacher views
   and erased patches.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
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
    erase_drop_ratio: float = 0.5  # pseudo boxes at least this fraction erased are dropped

    # SSOD epoch policy
    tail_ratio: float = 0.0  # fraction of GT uniformly placed at tail of epoch (0 = no tail)
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


class RandomErasing:
    """
    Erase random rectangular patches with probability p and record them (xyxy pixels) in
    `labels["erase_boxes"]`, so pseudo boxes produced later for the image can be dropped when
    mostly erased and the Student is not asked to detect invisible objects. Must run after the
    geometric augmentations (it detects padding by its exact fill value, 114).
    """

    def __init__(
        self,
        p: float = 0.5,
        max_patches: int = 3,
        area: tuple[float, float] = (0.01, 0.08),
    ) -> None:
        self.p = p
        self.max_patches = max(1, max_patches)
        self.area = area

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        labels["erase_boxes"] = np.zeros((0, 4), dtype=np.float32)
        if np.random.random() >= self.p:
            return labels

        img = labels["img"]
        patches = []

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
            patches.append([x0, y0, x0 + pw, y0 + ph])

        if patches:
            labels["erase_boxes"] = np.array(patches, dtype=np.float32)
        labels["img"] = img
        return labels


class TeacherView:
    """Keep a copy of the current (weakly augmented) image as the Teacher's input."""

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        labels["teacher_img"] = labels["img"].copy()  # later photometric steps work in place
        return labels


class FormatTeacherView:
    """Convert the Teacher view like `Format` does with the image (HWC BGR -> CHW RGB tensor)."""

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        img = labels["teacher_img"]
        if img.ndim < 3:
            img = img[..., None]
        labels["teacher_img"] = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)[::-1]))
        labels["erase_boxes"] = torch.from_numpy(labels.get("erase_boxes", np.zeros((0, 4), np.float32)))
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


def build_unlabeled_transforms(
    dataset: Any,
    imgsz: int = 640,
    hyp: SSODAugmentConfig | None = None,
) -> Compose:
    """
    Build transforms for the unlabeled training images.

    Mosaic is NEVER included. The geometric steps (Scale & Translate via RandomPerspective, then
    RandomFlip(fliplr)) are shared by both views; the image after them is kept as the weak
    Teacher view (`teacher_img`). RandomHSV is then applied for the Student; with `pl_strong`,
    HSV gains are raised and random erasing plus Gaussian blur are added. All other
    augmentations disabled.
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
        # Erasing must precede the photometric steps: it detects padding by its exact fill value (114)
        photometric = [
            RandomErasing(p=cfg.erase_p, max_patches=cfg.erase_max_patches, area=cfg.erase_area),
            RandomHSV(hgain=cfg.strong_hsv_h, sgain=cfg.strong_hsv_s, vgain=cfg.strong_hsv_v),
            RandomGaussianBlur(p=cfg.blur_p, max_kernel=cfg.blur_max_kernel),
        ]
    else:
        photometric = [RandomHSV(hgain=cfg.hsv_h, sgain=cfg.hsv_s, vgain=cfg.hsv_v)]

    transforms = [
        affine,
        RandomFlip(direction="horizontal", p=cfg.fliplr),
        TeacherView(),
        *photometric,
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
        FormatTeacherView(),
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


class UnlabeledDataset(YOLODataset):
    """
    YOLO dataset over a changing list of unlabeled images (set each epoch with `set_images`).
    Samples carry no boxes: the Teacher pseudo-labels them during training from their weak view
    (`teacher_img`). Gracefully handles an empty image list (len=0).
    """

    def __init__(
        self,
        images_dir: Path | str,
        data: dict[str, Any] | None = None,
        imgsz: int = 640,
        hyp: SSODAugmentConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.images_dir = Path(images_dir).resolve()
        self.logger = logger or get_logger()
        self.augment_cfg = hyp or SSODAugmentConfig()
        self._image_files: list[Path] = []

        data_dict = data or {"names": {0: "object"}, "nc": 1}
        super().__init__(
            img_path=str(self.images_dir),
            imgsz=imgsz,
            data=data_dict,
            task="detect",
            augment=True,
        )
        self.set_images([])

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
                f"Index {i} out of bounds for UnlabeledDataset with {len(self.im_files)} images."
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

    def get_labels(self) -> list[dict[str, Any]]:
        """Labels of the current image list (no boxes)."""
        return [
            {
                "im_file": str(path),
                "cls": np.empty((0, 1), dtype=np.float32),
                "bboxes": np.empty((0, 4), dtype=np.float32),
                "segments": [],
                "keypoints": None,
                "normalized": True,
                "bbox_format": "xywh",
            }
            for path in self._image_files
        ]

    def set_images(self, image_files: Sequence[Path | str]) -> int:
        """Replace the image list (e.g. with a fresh sample of the pool). Returns its length."""
        self._image_files = [Path(p).resolve() for p in image_files]
        self.labels = self.get_labels()
        self.im_files = [label["im_file"] for label in self.labels]
        self._sync_buffers()
        return len(self.labels)

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """YOLODataset collation plus stacked Teacher views and (M, 5) [image, x1, y1, x2, y2] erased patches."""
        teacher_imgs, erase_boxes = split_teacher_inputs(batch)
        collated = YOLODataset.collate_fn(batch)
        collated["teacher_img"] = teacher_imgs
        collated["erase_boxes"] = erase_boxes
        return collated


def split_teacher_inputs(batch: list[dict[str, Any]]) -> tuple[torch.Tensor | None, torch.Tensor]:
    """
    Pop the Teacher views and erased patches of a batch's unlabeled samples: the views stacked
    in batch order (None without unlabeled samples) and the patches as (M, 5)
    [batch index, x1, y1, x2, y2].
    """
    teacher_imgs = [b.pop("teacher_img") for b in batch if "teacher_img" in b]
    patches = [
        torch.cat([torch.full((len(p), 1), float(i)), p.float()], dim=1)
        for i, b in enumerate(batch)
        if (p := b.pop("erase_boxes", None)) is not None
    ]
    return (
        torch.stack(teacher_imgs) if teacher_imgs else None,
        torch.cat(patches) if patches else torch.zeros((0, 5)),
    )


class SSODSampler(Sampler[int]):
    """
    Epoch-aware sampler for Semi-Supervised Object Detection (SSOD).

    Each epoch:
      1. Uniformly select a `tail_ratio` fraction of GT items (without replacement) for the
         tail; none with the default `tail_ratio=0`.
      2. The remaining GT items (repeated `gt_repeat` times) are mixed and shuffled with all
         pseudo-labeled items.
      3. Yields indices: [ Shuffled(gt_repeat x head GT + 100% Pseudo) ] + [ GT tail ].
    """

    def __init__(
        self,
        gt_len: int,
        pl_len: int,
        tail_ratio: float = 0.0,
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

        # Uniformly partition GT into tail (tail_ratio) and head
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

    Wraps Ground-Truth dataset and unlabeled dataset, coordinating
    separate augmentation pipelines (Mosaic on GT, no Mosaic and a weak Teacher view on
    unlabeled images) and sample source tracking (`is_pseudo`).
    """

    def __init__(
        self,
        gt_dataset: YOLODataset,
        pl_dataset: UnlabeledDataset,
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
        self.transforms_pl = build_unlabeled_transforms(
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
        Collate batch items into standard Ultralytics YOLO tensors while preserving the
        `is_pseudo` and `source` tensors for the Mean Teacher loss, the stacked Teacher views of
        the unlabeled samples (`teacher_img`, in batch order; None without unlabeled samples),
        their erased patches (`erase_boxes`, (M, 5) [batch index, x1, y1, x2, y2]) and GT box
        weights of 1 (`pl_weight`; pseudo boxes are appended by the labeler).
        """
        is_pseudo_list = [b["is_pseudo"] for b in batch]
        source_list = [b["source"] for b in batch]

        # Strip custom metadata keys before calling standard YOLODataset.collate_fn
        standard_batch = [
            {k: v for k, v in b.items() if k not in {"is_pseudo", "source"}} for b in batch
        ]
        teacher_imgs, erase_boxes = split_teacher_inputs(standard_batch)

        collated = YOLODataset.collate_fn(standard_batch)
        collated["is_pseudo"] = torch.tensor(is_pseudo_list, dtype=torch.bool)
        collated["source"] = torch.tensor(source_list, dtype=torch.long)
        collated["teacher_img"] = teacher_imgs
        collated["erase_boxes"] = erase_boxes
        collated["pl_weight"] = torch.ones(collated["cls"].shape[0], dtype=torch.float32)
        return collated


class SSODDatasetHandler:
    """
    High-level orchestrator for SSOD datasets and dataloaders in Mean Teacher training.

    Coordinates:
    - GT Train dataset & GT Val dataset
    - Unlabeled Train dataset (a fresh image list every epoch)
    - SSODSampler with GT + unlabeled shuffled sequencing (optional GT tail)
    - Automatic mosaic closing during the last 10% of epochs
    """

    def __init__(
        self,
        gt_data_yaml: Path | str | None = None,
        gt_train_images: Path | str | None = None,
        gt_train_labels: Path | str | None = None,
        gt_val_images: Path | str | None = None,
        gt_val_labels: Path | str | None = None,
        pl_images_dir: Path | str | None = None,
        class_names: dict[int, str] | list[str] | None = None,
        imgsz: int = 640,
        batch_size: int = 10,
        workers: int = 4,
        seed: int = 42,
        augment_cfg: SSODAugmentConfig | None = None,
        tail_ratio: float = 0.0,
        close_mosaic_ratio: float = 0.10,
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

        # Resolve paths & class metadata
        data_cfg = self._resolve_data_config(
            gt_data_yaml=gt_data_yaml,
            gt_train_images=gt_train_images,
            gt_train_labels=gt_train_labels,
            gt_val_images=gt_val_images,
            gt_val_labels=gt_val_labels,
            pl_images_dir=pl_images_dir,
            class_names=class_names,
        )

        self.gt_train_images = data_cfg["gt_train_images"]
        self.gt_val_images = data_cfg["gt_val_images"]
        self.pl_images_dir = data_cfg["pl_images_dir"]
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

        # 3. Initialize Unlabeled Dataset (empty until the first `set_pl_images`)
        self.logger.info(f"Initializing Unlabeled Dataset (images={self.pl_images_dir})...")
        self.pl_dataset = UnlabeledDataset(
            images_dir=self.pl_images_dir,
            data=self.data_dict,
            imgsz=self.imgsz,
            hyp=self.augment_cfg,
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
            f"{len(self.pl_dataset)} initial unlabeled items (total={len(self.train_dataset)})."
        )

    def _resolve_data_config(
        self,
        gt_data_yaml: Path | str | None,
        gt_train_images: Path | str | None,
        gt_train_labels: Path | str | None,
        gt_val_images: Path | str | None,
        gt_val_labels: Path | str | None,
        pl_images_dir: Path | str | None,
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
                "Unlabeled images dir (pl_images_dir) must be specified."
            )

        return {
            "gt_train_images": resolved_gt_train,
            "gt_val_images": resolved_gt_val,
            "pl_images_dir": Path(pl_images_dir).resolve(),
            "data_dict": data_dict,
        }

    def set_pl_images(self, image_files: Sequence[Path | str]) -> int:
        """Set the unlabeled images of the current epoch and synchronize the sampler."""
        count = self.pl_dataset.set_images(image_files)
        self.sampler.update_lengths(
            gt_len=len(self.gt_train_dataset), pl_len=len(self.pl_dataset)
        )
        self.logger.info(
            f"Unlabeled images this epoch: {count}. "
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

    def get_unlabeled_loader(self, pin_memory: bool = True) -> DataLoader:
        """
        DataLoader over the current unlabeled images alone, with the same transforms as in
        training (used to build the initial score histograms from the Teacher views).
        """
        return DataLoader(
            dataset=self.pl_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            collate_fn=UnlabeledDataset.collate_fn,
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
