"""
Automated unit and integration tests for SSOD dataset handler, sampler, and transform builders.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any
import unittest
from unittest import mock

import cv2
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from ultralytics.data.augment import (
    Format,
    LetterBox,
    Mosaic,
    RandomFlip,
    RandomHSV,
    RandomPerspective,
)
from ultralytics.data.dataset import YOLODataset

# Ensure code directory is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from ssod.dataset import (  # noqa: E402
    BoxAwareRandomErasing,
    PseudoYOLODataset,
    RandomGaussianBlur,
    SSODAugmentConfig,
    SSODDataset,
    SSODDatasetHandler,
    SSODSampler,
    build_gt_transforms,
    build_pseudo_transforms,
    build_val_transforms,
    split_pl_weight,
)


def _has_transform(transforms_obj: Any, cls_type: type) -> bool:
    """Recursively check whether a transform type is present in Compose."""
    if isinstance(transforms_obj, cls_type):
        return True
    if hasattr(transforms_obj, "transforms"):
        return any(_has_transform(t, cls_type) for t in transforms_obj.transforms)
    return False


def _create_synthetic_image(path: Path, width: int = 100, height: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _create_synthetic_label(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines) + ("\n" if lines else "")
    path.write_text(content, encoding="utf-8")


class TestSSODTransforms(unittest.TestCase):
    """Test augmentation configuration and selective transform pipelines."""

    def test_gt_transforms_open_and_closed(self):
        cfg = SSODAugmentConfig(
            hsv_h=0.015,
            hsv_s=0.5,
            hsv_v=0.4,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
            mosaic=1.0,
        )

        class DummyDataset:
            cache = None

        dummy_dataset = DummyDataset()
        # 1. Open mosaic
        transforms_open = build_gt_transforms(dummy_dataset, imgsz=640, hyp=cfg, mosaic=True)
        self.assertTrue(_has_transform(transforms_open, Mosaic), "GT open transforms must include Mosaic")
        self.assertTrue(_has_transform(transforms_open, RandomPerspective), "GT open transforms must include RandomPerspective")
        self.assertTrue(_has_transform(transforms_open, RandomHSV), "GT open transforms must include RandomHSV")
        self.assertTrue(_has_transform(transforms_open, RandomFlip), "GT open transforms must include RandomFlip")
        self.assertTrue(_has_transform(transforms_open, Format), "GT open transforms must include Format")

        # 2. Closed mosaic
        transforms_closed = build_gt_transforms(dummy_dataset, imgsz=640, hyp=cfg, mosaic=False)
        self.assertFalse(_has_transform(transforms_closed, Mosaic), "GT closed transforms must NOT include Mosaic")
        self.assertTrue(_has_transform(transforms_closed, RandomPerspective))
        self.assertTrue(_has_transform(transforms_closed, RandomHSV))
        self.assertTrue(_has_transform(transforms_closed, RandomFlip))

    def test_pseudo_transforms_never_include_mosaic(self):
        cfg = SSODAugmentConfig(mosaic=1.0)
        class DummyDataset:
            cache = None
        dummy_dataset = DummyDataset()
        transforms_pseudo = build_pseudo_transforms(dummy_dataset, imgsz=640, hyp=cfg)

        self.assertFalse(_has_transform(transforms_pseudo, Mosaic), "Pseudo transforms must NEVER include Mosaic")
        self.assertTrue(_has_transform(transforms_pseudo, RandomPerspective))
        self.assertTrue(_has_transform(transforms_pseudo, RandomHSV))
        self.assertTrue(_has_transform(transforms_pseudo, RandomFlip))

    def test_pseudo_strong_transforms(self):
        class DummyDataset:
            cache = None
        weak = build_pseudo_transforms(DummyDataset(), imgsz=640, hyp=SSODAugmentConfig())
        self.assertFalse(_has_transform(weak, RandomGaussianBlur))
        self.assertFalse(_has_transform(weak, BoxAwareRandomErasing))

        strong = build_pseudo_transforms(DummyDataset(), imgsz=640, hyp=SSODAugmentConfig(pl_strong=True))
        self.assertFalse(_has_transform(strong, Mosaic))
        self.assertTrue(_has_transform(strong, RandomGaussianBlur))
        self.assertTrue(_has_transform(strong, BoxAwareRandomErasing))
        self.assertTrue(_has_transform(strong, RandomFlip))

    def test_box_aware_erasing_drops_only_covered_boxes(self):
        from ultralytics.utils.instance import Instances

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        boxes = np.array([[10, 10, 30, 30], [60, 60, 90, 90]], dtype=np.float32)
        labels = {
            "img": img,
            "cls": np.array([[0.0], [0.0]], dtype=np.float32),
            "instances": Instances(
                boxes,
                segments=np.zeros((0, 1000, 2), dtype=np.float32),
                bbox_format="xyxy",
                normalized=False,
            ),
        }
        erasing = BoxAwareRandomErasing(p=1.0, max_patches=1, area=(0.3, 0.3), drop_ratio=0.5)
        # Force a single square patch (~55x55) anchored at (0, 0): covers box 1, misses box 2
        original_randint = np.random.randint

        def fixed_randint(low, high=None, size=None, dtype=int):
            if size is not None:
                return original_randint(low, high, size=size, dtype=dtype)
            return 1 if high == 2 else 0

        with mock.patch.object(np.random, "randint", side_effect=fixed_randint), \
                mock.patch.object(np.random, "uniform", side_effect=lambda a, b: 0.0 if a < 0 else a):
            out = erasing(labels)

        self.assertEqual(len(out["instances"]), len(out["cls"]))
        kept = out["instances"].bboxes.tolist()
        self.assertNotIn([10.0, 10.0, 30.0, 30.0], kept, "Box under the patch must be dropped")
        self.assertIn([60.0, 60.0, 90.0, 90.0], kept, "Box outside the patch must be kept")

    def test_val_transforms_only_letterbox_and_format(self):
        transforms_val = build_val_transforms(imgsz=640)
        self.assertTrue(_has_transform(transforms_val, LetterBox))
        self.assertTrue(_has_transform(transforms_val, Format))
        self.assertFalse(_has_transform(transforms_val, Mosaic))
        self.assertFalse(_has_transform(transforms_val, RandomPerspective))
        self.assertFalse(_has_transform(transforms_val, RandomHSV))
        self.assertFalse(_has_transform(transforms_val, RandomFlip))


class TestSSODSampler(unittest.TestCase):
    """Test 25% GT tail selection and 75% GT + PL head shuffling."""

    def test_partitioning_and_proportions(self):
        n_gt = 100
        n_pl = 50
        tail_ratio = 0.25

        sampler = SSODSampler(gt_len=n_gt, pl_len=n_pl, tail_ratio=tail_ratio, seed=42)
        indices = list(sampler)

        self.assertEqual(len(indices), n_gt + n_pl, "Sampler must return total items = N_gt + N_pl")
        self.assertEqual(len(set(indices)), n_gt + n_pl, "All indices must be unique within an epoch")

        n_tail = round(n_gt * tail_ratio)
        n_head = len(indices) - n_tail

        head_indices = indices[:n_head]
        tail_indices = indices[n_head:]

        self.assertEqual(len(tail_indices), 25, "Tail must contain exactly 25 items")
        # Tail items must strictly belong to GT (indices 0 to 99)
        for idx in tail_indices:
            self.assertTrue(0 <= idx < n_gt, f"Tail item {idx} must be a GT index (< {n_gt})")

        # Head items must contain the remaining 75 GT items and all 50 PL items
        gt_in_head = [idx for idx in head_indices if idx < n_gt]
        pl_in_head = [idx for idx in head_indices if idx >= n_gt]

        self.assertEqual(len(gt_in_head), 75, "Head must contain exactly 75 GT items")
        self.assertEqual(len(pl_in_head), 50, "Head must contain all 50 PL items")
        self.assertEqual(set(pl_in_head), set(range(n_gt, n_gt + n_pl)), "All PL indices must be in head")

        # Tail and head GT items must be disjoint
        self.assertTrue(set(gt_in_head).isdisjoint(set(tail_indices)), "Head GT and Tail GT must be mutually exclusive")

    def test_gt_repeat_scales_head_gt_only(self):
        n_gt, n_pl = 100, 50
        sampler = SSODSampler(gt_len=n_gt, pl_len=n_pl, tail_ratio=0.25, seed=42, gt_repeat=3)
        indices = list(sampler)

        self.assertEqual(len(indices), 3 * 75 + 25 + n_pl)
        tail = indices[-25:]
        head = indices[:-25]
        self.assertTrue(all(idx < n_gt for idx in tail))
        head_gt = [idx for idx in head if idx < n_gt]
        self.assertEqual(len(head_gt), 3 * 75)
        self.assertEqual(len(set(head_gt)), 75)
        self.assertTrue(set(head_gt).isdisjoint(tail))
        self.assertEqual(sorted(idx for idx in head if idx >= n_gt), list(range(n_gt, n_gt + n_pl)))

    def test_reproducibility_with_epoch(self):
        sampler = SSODSampler(gt_len=40, pl_len=20, tail_ratio=0.25, seed=123)

        sampler.set_epoch(1)
        seq_epoch1_first = list(sampler)

        sampler.set_epoch(1)
        seq_epoch1_second = list(sampler)
        self.assertEqual(seq_epoch1_first, seq_epoch1_second, "Same epoch must yield identical sequence")

        sampler.set_epoch(2)
        seq_epoch2 = list(sampler)
        self.assertNotEqual(seq_epoch1_first, seq_epoch2, "Different epochs must yield different sequences")

    def test_zero_pseudo_labels(self):
        n_gt = 20
        n_pl = 0
        sampler = SSODSampler(gt_len=n_gt, pl_len=n_pl, tail_ratio=0.25, seed=42)
        indices = list(sampler)

        self.assertEqual(len(indices), n_gt)
        tail = indices[-5:]
        self.assertEqual(len(tail), 5)
        for idx in tail:
            self.assertTrue(0 <= idx < n_gt)

    def test_update_lengths(self):
        sampler = SSODSampler(gt_len=20, pl_len=5, tail_ratio=0.25, seed=42)
        self.assertEqual(len(sampler), 25)

        # PL dataset expands from 5 to 15
        sampler.update_lengths(gt_len=20, pl_len=15)
        self.assertEqual(len(sampler), 35)
        indices = list(sampler)
        self.assertEqual(len(indices), 35)
        # Check PL range is now [20, 34]
        pl_items = [idx for idx in indices if idx >= 20]
        self.assertEqual(set(pl_items), set(range(20, 35)))


class TestSSODDatasetIntegration(unittest.TestCase):
    """End-to-end integration tests with synthetic image & label files."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

        # Setup GT Train directories
        self.gt_train_img_dir = self.temp_dir / "gt" / "images" / "train"
        self.gt_train_lbl_dir = self.temp_dir / "gt" / "labels" / "train"
        # Setup GT Val directories
        self.gt_val_img_dir = self.temp_dir / "gt" / "images" / "val"
        self.gt_val_lbl_dir = self.temp_dir / "gt" / "labels" / "val"
        # Setup Pseudo directories
        self.pl_img_dir = self.temp_dir / "pseudo" / "images"
        self.pl_lbl_dir = self.temp_dir / "pseudo" / "labels"

        # Create 4 GT Train samples
        for i in range(4):
            _create_synthetic_image(self.gt_train_img_dir / f"gt_train_{i}.jpg", width=128, height=128)
            _create_synthetic_label(
                self.gt_train_lbl_dir / f"gt_train_{i}.txt",
                [f"0 0.{i+1} 0.{i+1} 0.2 0.2"],
            )

        # Create 2 GT Val samples
        for i in range(2):
            _create_synthetic_image(self.gt_val_img_dir / f"gt_val_{i}.jpg", width=128, height=128)
            _create_synthetic_label(
                self.gt_val_lbl_dir / f"gt_val_{i}.txt",
                ["0 0.5 0.5 0.3 0.3"],
            )

        # Create 3 Pseudo samples (2 with labels, 1 without)
        for i in range(3):
            _create_synthetic_image(self.pl_img_dir / f"pseudo_{i}.jpg", width=128, height=128)

        _create_synthetic_label(self.pl_lbl_dir / "pseudo_0.txt", ["0 0.2 0.2 0.1 0.1"])
        _create_synthetic_label(self.pl_lbl_dir / "pseudo_1.txt", ["0 0.4 0.4 0.2 0.2"])
        # pseudo_2 has no label file initially

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pseudo_dataset_filter_empty(self):
        # With filter_empty=True, only pseudo_0 and pseudo_1 are included
        pl_ds = PseudoYOLODataset(
            images_dir=self.pl_img_dir,
            labels_dir=self.pl_lbl_dir,
            filter_empty=True,
        )
        self.assertEqual(len(pl_ds), 2)

        # With filter_empty=False, all 3 images are included
        pl_ds_all = PseudoYOLODataset(
            images_dir=self.pl_img_dir,
            labels_dir=self.pl_lbl_dir,
            filter_empty=False,
        )
        self.assertEqual(len(pl_ds_all), 3)

    def test_ssod_dataset_item_access_and_collate(self):
        gt_ds = YOLODataset(
            img_path=str(self.gt_train_img_dir),
            imgsz=128,
            data={"names": {0: "waste"}, "nc": 1},
            augment=True,
        )
        pl_ds = PseudoYOLODataset(
            images_dir=self.pl_img_dir,
            labels_dir=self.pl_lbl_dir,
            data={"names": {0: "waste"}, "nc": 1},
            imgsz=128,
            filter_empty=True,
        )

        ssod_ds = SSODDataset(gt_dataset=gt_ds, pl_dataset=pl_ds, imgsz=128)
        self.assertEqual(len(ssod_ds), 4 + 2)

        # Check GT item
        item_gt = ssod_ds[0]
        self.assertFalse(item_gt["is_pseudo"])
        self.assertEqual(item_gt["source"], 0)
        self.assertIn("img", item_gt)
        self.assertIn("bboxes", item_gt)

        # Check Pseudo item
        item_pl = ssod_ds[4]
        self.assertTrue(item_pl["is_pseudo"])
        self.assertEqual(item_pl["source"], 1)

        # Check Collate Fn
        batch = ssod_ds.collate_fn([ssod_ds[0], ssod_ds[4]])
        self.assertIn("img", batch)
        self.assertEqual(batch["img"].shape[0], 2)
        self.assertIn("is_pseudo", batch)
        self.assertTrue(isinstance(batch["is_pseudo"], torch.Tensor))
        self.assertEqual(batch["is_pseudo"].tolist(), [False, True])
        self.assertEqual(batch["source"].tolist(), [0, 1])

    def test_dynamic_dataframe_reloading(self):
        pl_ds = PseudoYOLODataset(
            images_dir=self.pl_img_dir,
            labels_dir=self.pl_lbl_dir,
            filter_empty=True,
        )
        self.assertEqual(len(pl_ds), 2)

        # Simulate PseudoLabelGenerator candidate output dataframe for pseudo_2
        df = pl.DataFrame({
            "image_stem": ["pseudo_2"],
            "class_id": [0],
            "x_center": [0.55],
            "y_center": [0.65],
            "w_norm": [0.25],
            "h_norm": [0.35],
            "img_width": [128],
            "img_height": [128],
        })

        pl_ds.reload(filtered_df=df, filter_empty=True)
        self.assertEqual(len(pl_ds), 1)
        self.assertEqual(pl_ds.im_files[0], str(self.pl_img_dir / "pseudo_2.jpg"))

    def test_handler_lifecycle_and_mosaic_close(self):
        handler = SSODDatasetHandler(
            gt_train_images=self.gt_train_img_dir,
            gt_val_images=self.gt_val_img_dir,
            pl_images_dir=self.pl_img_dir,
            pl_labels_dir=self.pl_lbl_dir,
            imgsz=128,
            batch_size=2,
            workers=0,
            close_mosaic_ratio=0.10,
        )

        self.assertEqual(len(handler.gt_train_dataset), 4)
        self.assertEqual(len(handler.pl_dataset), 2)
        self.assertEqual(len(handler.train_dataset), 6)

        train_loader = handler.get_train_loader()
        val_loader = handler.get_val_loader()

        self.assertIsNotNone(val_loader)
        self.assertEqual(len(val_loader.dataset), 2)

        total_epochs = 10
        # Epoch 0 (first 90%): Mosaic must be open
        handler.set_epoch(epoch=0, total_epochs=total_epochs)
        self.assertFalse(handler.train_dataset.is_mosaic_closed)

        # Epoch 8: 8 < 9, Mosaic open
        handler.set_epoch(epoch=8, total_epochs=total_epochs)
        self.assertFalse(handler.train_dataset.is_mosaic_closed)

        # Epoch 9: 9 >= 9 (10 - 1), entering last 10%, Mosaic closed!
        handler.set_epoch(epoch=9, total_epochs=total_epochs)
        self.assertTrue(handler.train_dataset.is_mosaic_closed)

        # Iterate 1 batch to verify tensor execution
        for b in train_loader:
            self.assertEqual(b["img"].shape, (2, 3, 128, 128))
            self.assertEqual(len(b["is_pseudo"]), 2)
            break

    def test_pseudo_dataset_buffer_sync_from_empty(self):
        # Create empty pseudo directory with images only (no labels)
        empty_lbl_dir = self.temp_dir / "empty_pseudo_labels"
        empty_lbl_dir.mkdir(parents=True, exist_ok=True)

        pl_ds = PseudoYOLODataset(
            images_dir=self.pl_img_dir,
            labels_dir=empty_lbl_dir,
            filter_empty=True,
        )
        self.assertEqual(len(pl_ds), 0)
        self.assertEqual(len(pl_ds.ims), 0)
        self.assertEqual(len(pl_ds.npy_files), 0)

        # Now populate label files for both pseudo images
        _create_synthetic_label(empty_lbl_dir / "pseudo_1.txt", ["0 0.5 0.5 0.2 0.2"])
        _create_synthetic_label(empty_lbl_dir / "pseudo_2.txt", ["0 0.6 0.6 0.3 0.3"])

        count = pl_ds.reload()
        self.assertEqual(count, 2)
        self.assertEqual(len(pl_ds), 2)
        self.assertEqual(len(pl_ds.ims), 2)
        self.assertEqual(len(pl_ds.npy_files), 2)
        self.assertEqual(len(pl_ds.im_hw0), 2)
        self.assertEqual(len(pl_ds.im_hw), 2)

        # Accessing all items must succeed without IndexError
        item0 = pl_ds[0]
        self.assertIn("img", item0)
        item1 = pl_ds[1]
        self.assertIn("img", item1)

    def test_pseudo_dataset_dataframe_buffer_sync_from_empty(self):
        empty_lbl_dir = self.temp_dir / "empty_pseudo_labels_df"
        empty_lbl_dir.mkdir(parents=True, exist_ok=True)

        pl_ds = PseudoYOLODataset(
            images_dir=self.pl_img_dir,
            labels_dir=empty_lbl_dir,
            filter_empty=True,
        )
        self.assertEqual(len(pl_ds), 0)
        self.assertEqual(len(pl_ds.ims), 0)

        df = pl.DataFrame({
            "image_stem": ["pseudo_1", "pseudo_2"],
            "class_id": [0, 0],
            "x_center": [0.5, 0.6],
            "y_center": [0.5, 0.6],
            "w_norm": [0.2, 0.3],
            "h_norm": [0.2, 0.3],
            "img_width": [128, 128],
            "img_height": [128, 128],
        })
        pl_ds.reload(filtered_df=df, filter_empty=True)
        self.assertEqual(len(pl_ds), 2)
        self.assertEqual(len(pl_ds.ims), 2)
        self.assertEqual(len(pl_ds.npy_files), 2)

        item = pl_ds[1]
        self.assertIn("img", item)

    def test_pseudo_dataset_multiprocessing_dataloader_after_reload(self):
        empty_lbl_dir = self.temp_dir / "empty_pseudo_labels_mp"
        empty_lbl_dir.mkdir(parents=True, exist_ok=True)

        pl_ds = PseudoYOLODataset(
            images_dir=self.pl_img_dir,
            labels_dir=empty_lbl_dir,
            filter_empty=True,
        )
        self.assertEqual(len(pl_ds), 0)

        _create_synthetic_label(empty_lbl_dir / "pseudo_1.txt", ["0 0.5 0.5 0.2 0.2"])
        _create_synthetic_label(empty_lbl_dir / "pseudo_2.txt", ["0 0.6 0.6 0.3 0.3"])
        pl_ds.reload()

        loader = DataLoader(
            pl_ds,
            batch_size=2,
            num_workers=2,
            collate_fn=pl_ds.collate_fn,
            persistent_workers=False,
        )
        batches = list(loader)
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0]["img"].shape, (2, 3, 640, 640))

    def test_handler_multiprocessing_dynamic_reload(self):
        empty_pl_lbl = self.temp_dir / "handler_empty_pl_labels"
        empty_pl_lbl.mkdir(parents=True, exist_ok=True)

        handler = SSODDatasetHandler(
            gt_train_images=self.gt_train_img_dir,
            gt_val_images=self.gt_val_img_dir,
            pl_images_dir=self.pl_img_dir,
            pl_labels_dir=empty_pl_lbl,
            imgsz=128,
            batch_size=2,
            workers=2,
            close_mosaic_ratio=0.10,
        )
        # GT=4, PL=0
        self.assertEqual(len(handler.gt_train_dataset), 4)
        self.assertEqual(len(handler.pl_dataset), 0)
        self.assertEqual(len(handler.train_dataset), 4)

        # Epoch 1: pseudo labels generated
        _create_synthetic_label(empty_pl_lbl / "pseudo_1.txt", ["0 0.5 0.5 0.2 0.2"])
        _create_synthetic_label(empty_pl_lbl / "pseudo_2.txt", ["0 0.6 0.6 0.3 0.3"])
        reloaded_count = handler.reload_pseudo_labels()
        self.assertEqual(reloaded_count, 2)
        self.assertEqual(len(handler.train_dataset), 6)

        train_loader = handler.get_train_loader()
        all_pseudo_flags: list[bool] = []
        for batch in train_loader:
            self.assertEqual(batch["img"].shape, (2, 3, 128, 128))
            all_pseudo_flags.extend(batch["is_pseudo"].tolist())

        self.assertEqual(len(all_pseudo_flags), 6)
        self.assertEqual(sum(all_pseudo_flags), 2)  # exactly 2 pseudo items


if __name__ == "__main__":
    unittest.main()


class TestPseudoBoxWeights(unittest.TestCase):
    """Per-box pseudo-label weights must stay aligned with their boxes through augmentation and collation."""

    # Distinct widths (all near the centre, so augmentation never clips them) mapped to distinct weights
    BOXES = [(0.44, 0.10), (0.56, 0.14), (0.50, 0.18)]  # (x_center, width)
    WEIGHTS = [0.2, 0.5, 0.9]

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.pl_img_dir = self.temp_dir / "pseudo" / "images"
        self.pl_lbl_dir = self.temp_dir / "pseudo" / "labels"
        for i in range(4):
            _create_synthetic_image(self.pl_img_dir / f"pseudo_{i}.jpg", width=256, height=256)
            _create_synthetic_label(
                self.pl_lbl_dir / f"pseudo_{i}.txt",
                [f"0 {x} 0.5 {w} {w} {wt}" for (x, w), wt in zip(self.BOXES, self.WEIGHTS)],
            )
        _create_synthetic_image(self.pl_img_dir / "pseudo_empty.jpg", width=256, height=256)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _assert_aligned(self, batch: dict[str, Any]) -> None:
        n = batch["bboxes"].shape[0]
        self.assertEqual(batch["cls"].shape, (n, 1))
        self.assertEqual(batch["pl_weight"].shape, (n,))
        self.assertTrue(torch.all(batch["cls"] == 0))
        for image_index in batch["batch_idx"].unique():
            mask = batch["batch_idx"] == image_index
            widths = batch["bboxes"][mask][:, 2]
            weights = batch["pl_weight"][mask]
            order = torch.argsort(widths)
            # Width ordering is preserved by scale/translate/flip, so weights must follow it
            self.assertTrue(torch.all(torch.diff(weights[order]) > 0), (widths, weights))
            for w in weights.tolist():
                self.assertTrue(any(abs(w - ref) < 1e-4 for ref in self.WEIGHTS))

    def _dataset(self, cfg: SSODAugmentConfig, filter_empty: bool = True) -> PseudoYOLODataset:
        ds = PseudoYOLODataset(
            images_dir=self.pl_img_dir,
            labels_dir=self.pl_lbl_dir,
            imgsz=128,
            hyp=cfg,
            filter_empty=filter_empty,
        )
        ds.transforms = build_pseudo_transforms(ds, imgsz=128, hyp=cfg)
        return ds

    def test_weights_follow_boxes_weak_and_strong(self):
        for cfg in (SSODAugmentConfig(), SSODAugmentConfig(pl_strong=True, erase_p=1.0)):
            ds = self._dataset(cfg)
            np.random.seed(0)
            seen_boxes = 0
            for _ in range(5):
                loader = DataLoader(ds, batch_size=4, collate_fn=PseudoYOLODataset.collate_fn)
                for batch in loader:
                    self._assert_aligned(batch)
                    seen_boxes += batch["bboxes"].shape[0]
            self.assertGreater(seen_boxes, 20, "augmentation must leave boxes to check")

    def test_empty_samples_collate(self):
        ds = self._dataset(SSODAugmentConfig(), filter_empty=False)
        self.assertEqual(len(ds), 5)
        batch = next(iter(DataLoader(ds, batch_size=5, collate_fn=PseudoYOLODataset.collate_fn)))
        self._assert_aligned(batch)
        empty = split_pl_weight({"cls": torch.zeros(0, 1)})
        self.assertEqual(empty["pl_weight"].shape, (0,))

    def test_dataframe_weights_through_ssod_collate(self):
        ds = self._dataset(SSODAugmentConfig())
        rows = [
            {
                "image_stem": f"pseudo_{i}",
                "class_id": 0,
                "x_center": x,
                "y_center": 0.5,
                "w_norm": w,
                "h_norm": w,
                "img_width": 256,
                "img_height": 256,
                "weight": wt,
            }
            for i in range(4)
            for (x, w), wt in zip(self.BOXES, self.WEIGHTS)
        ]
        ds.load_from_dataframe(pl.DataFrame(rows))
        samples = []
        for i in range(len(ds)):
            sample = ds[i]
            sample["is_pseudo"] = True
            sample["source"] = 1
            samples.append(sample)
        batch = SSODDataset.collate_fn(samples)
        self._assert_aligned(batch)
        self.assertGreater(batch["bboxes"].shape[0], 0)
        self.assertTrue(torch.all(batch["is_pseudo"]))

