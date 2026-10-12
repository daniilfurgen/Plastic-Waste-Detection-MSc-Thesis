"""
Automated unit and integration tests for SSOD dataset handler, sampler, and transform builders.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np
import polars as pl
import pytest
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
from ultralytics.utils.instance import Instances

from ssod.dataset import (
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


class DummyDataset:
    cache = None


class TestSSODTransforms:
    """Test augmentation configuration and selective transform pipelines."""

    def test_gt_transforms_open_mosaic(self):
        cfg = SSODAugmentConfig(
            hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, translate=0.1, scale=0.5, fliplr=0.5, mosaic=1.0
        )
        transforms = build_gt_transforms(DummyDataset(), imgsz=640, hyp=cfg, mosaic=True)
        for expected in (Mosaic, RandomPerspective, RandomHSV, RandomFlip, Format):
            assert _has_transform(transforms, expected), f"GT open transforms must include {expected.__name__}"

    def test_gt_transforms_closed_mosaic(self):
        cfg = SSODAugmentConfig(
            hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, translate=0.1, scale=0.5, fliplr=0.5, mosaic=1.0
        )
        transforms = build_gt_transforms(DummyDataset(), imgsz=640, hyp=cfg, mosaic=False)
        assert not _has_transform(transforms, Mosaic), "GT closed transforms must NOT include Mosaic"
        for expected in (RandomPerspective, RandomHSV, RandomFlip):
            assert _has_transform(transforms, expected)

    def test_pseudo_transforms_never_include_mosaic(self):
        transforms = build_pseudo_transforms(DummyDataset(), imgsz=640, hyp=SSODAugmentConfig(mosaic=1.0))
        assert not _has_transform(transforms, Mosaic), "Pseudo transforms must NEVER include Mosaic"
        for expected in (RandomPerspective, RandomHSV, RandomFlip):
            assert _has_transform(transforms, expected)

    def test_pseudo_weak_transforms_have_no_strong_augmentation(self):
        weak = build_pseudo_transforms(DummyDataset(), imgsz=640, hyp=SSODAugmentConfig())
        assert not _has_transform(weak, RandomGaussianBlur)
        assert not _has_transform(weak, BoxAwareRandomErasing)

    def test_pseudo_strong_transforms(self):
        strong = build_pseudo_transforms(DummyDataset(), imgsz=640, hyp=SSODAugmentConfig(pl_strong=True))
        assert not _has_transform(strong, Mosaic)
        assert _has_transform(strong, RandomGaussianBlur)
        assert _has_transform(strong, BoxAwareRandomErasing)
        assert _has_transform(strong, RandomFlip)

    def test_box_aware_erasing_drops_only_covered_boxes(self):
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

        with (
            mock.patch.object(np.random, "randint", side_effect=fixed_randint),
            mock.patch.object(np.random, "uniform", side_effect=lambda a, b: 0.0 if a < 0 else a),
        ):
            out = erasing(labels)

        assert len(out["instances"]) == len(out["cls"])
        kept = out["instances"].bboxes.tolist()
        assert [10.0, 10.0, 30.0, 30.0] not in kept, "Box under the patch must be dropped"
        assert [60.0, 60.0, 90.0, 90.0] in kept, "Box outside the patch must be kept"

    def test_val_transforms_only_letterbox_and_format(self):
        transforms = build_val_transforms(imgsz=640)
        assert _has_transform(transforms, LetterBox)
        assert _has_transform(transforms, Format)
        for absent in (Mosaic, RandomPerspective, RandomHSV, RandomFlip):
            assert not _has_transform(transforms, absent)


class TestSSODSampler:
    """Test 25% GT tail selection and 75% GT + PL head shuffling."""

    def test_partitioning_and_proportions(self):
        n_gt, n_pl, tail_ratio = 100, 50, 0.25

        indices = list(SSODSampler(gt_len=n_gt, pl_len=n_pl, tail_ratio=tail_ratio, seed=42))

        assert len(indices) == n_gt + n_pl, "Sampler must return total items = N_gt + N_pl"
        assert len(set(indices)) == n_gt + n_pl, "All indices must be unique within an epoch"

        n_tail = round(n_gt * tail_ratio)
        n_head = len(indices) - n_tail
        head_indices = indices[:n_head]
        tail_indices = indices[n_head:]

        assert len(tail_indices) == 25, "Tail must contain exactly 25 items"
        assert all(0 <= idx < n_gt for idx in tail_indices), "Tail items must be GT indices"

        # Head items must contain the remaining 75 GT items and all 50 PL items
        gt_in_head = [idx for idx in head_indices if idx < n_gt]
        pl_in_head = [idx for idx in head_indices if idx >= n_gt]

        assert len(gt_in_head) == 75, "Head must contain exactly 75 GT items"
        assert len(pl_in_head) == 50, "Head must contain all 50 PL items"
        assert set(pl_in_head) == set(range(n_gt, n_gt + n_pl)), "All PL indices must be in head"
        assert set(gt_in_head).isdisjoint(tail_indices), "Head GT and Tail GT must be mutually exclusive"

    def test_gt_repeat_scales_head_gt_only(self):
        n_gt, n_pl = 100, 50
        indices = list(SSODSampler(gt_len=n_gt, pl_len=n_pl, tail_ratio=0.25, seed=42, gt_repeat=3))

        assert len(indices) == 3 * 75 + 25 + n_pl
        tail = indices[-25:]
        head = indices[:-25]
        assert all(idx < n_gt for idx in tail)
        head_gt = [idx for idx in head if idx < n_gt]
        assert len(head_gt) == 3 * 75
        assert len(set(head_gt)) == 75
        assert set(head_gt).isdisjoint(tail)
        assert sorted(idx for idx in head if idx >= n_gt) == list(range(n_gt, n_gt + n_pl))

    def test_reproducibility_with_epoch(self):
        sampler = SSODSampler(gt_len=40, pl_len=20, tail_ratio=0.25, seed=123)

        sampler.set_epoch(1)
        first = list(sampler)
        sampler.set_epoch(1)
        second = list(sampler)
        assert first == second, "Same epoch must yield identical sequence"

        sampler.set_epoch(2)
        assert first != list(sampler), "Different epochs must yield different sequences"

    def test_zero_pseudo_labels(self):
        n_gt = 20
        indices = list(SSODSampler(gt_len=n_gt, pl_len=0, tail_ratio=0.25, seed=42))

        assert len(indices) == n_gt
        tail = indices[-5:]
        assert len(tail) == 5
        assert all(0 <= idx < n_gt for idx in tail)

    def test_update_lengths(self):
        sampler = SSODSampler(gt_len=20, pl_len=5, tail_ratio=0.25, seed=42)
        assert len(sampler) == 25

        # PL dataset expands from 5 to 15
        sampler.update_lengths(gt_len=20, pl_len=15)
        assert len(sampler) == 35
        indices = list(sampler)
        assert len(indices) == 35
        # PL range is now [20, 34]
        assert {idx for idx in indices if idx >= 20} == set(range(20, 35))


@pytest.fixture
def dirs(tmp_path, make_image, make_label):
    """Synthetic GT (4 train / 2 val) and pseudo (3 images, 2 labelled) dataset layout."""
    layout = SimpleNamespace(
        root=tmp_path,
        gt_train_img=tmp_path / "gt" / "images" / "train",
        gt_train_lbl=tmp_path / "gt" / "labels" / "train",
        gt_val_img=tmp_path / "gt" / "images" / "val",
        gt_val_lbl=tmp_path / "gt" / "labels" / "val",
        pl_img=tmp_path / "pseudo" / "images",
        pl_lbl=tmp_path / "pseudo" / "labels",
    )
    for i in range(4):
        make_image(layout.gt_train_img / f"gt_train_{i}.jpg", width=128, height=128)
        make_label(layout.gt_train_lbl / f"gt_train_{i}.txt", [f"0 0.{i+1} 0.{i+1} 0.2 0.2"])
    for i in range(2):
        make_image(layout.gt_val_img / f"gt_val_{i}.jpg", width=128, height=128)
        make_label(layout.gt_val_lbl / f"gt_val_{i}.txt", ["0 0.5 0.5 0.3 0.3"])
    for i in range(3):
        make_image(layout.pl_img / f"pseudo_{i}.jpg", width=128, height=128)
    make_label(layout.pl_lbl / "pseudo_0.txt", ["0 0.2 0.2 0.1 0.1"])
    make_label(layout.pl_lbl / "pseudo_1.txt", ["0 0.4 0.4 0.2 0.2"])
    # pseudo_2 has no label file initially
    return layout


@pytest.fixture
def empty_pl_lbl(dirs):
    path = dirs.root / "empty_pseudo_labels"
    path.mkdir()
    return path


@pytest.fixture
def populate_pl_labels(make_label):
    """Write label files for pseudo_1 and pseudo_2 into the given directory."""

    def _populate(directory):
        make_label(directory / "pseudo_1.txt", ["0 0.5 0.5 0.2 0.2"])
        make_label(directory / "pseudo_2.txt", ["0 0.6 0.6 0.3 0.3"])

    return _populate


class TestSSODDatasetIntegration:
    """End-to-end integration tests with synthetic image & label files."""

    def test_pseudo_dataset_filter_empty(self, dirs):
        # With filter_empty=True, only pseudo_0 and pseudo_1 are included
        pl_ds = PseudoYOLODataset(images_dir=dirs.pl_img, labels_dir=dirs.pl_lbl, filter_empty=True)
        assert len(pl_ds) == 2

        # With filter_empty=False, all 3 images are included
        pl_ds_all = PseudoYOLODataset(images_dir=dirs.pl_img, labels_dir=dirs.pl_lbl, filter_empty=False)
        assert len(pl_ds_all) == 3

    def test_ssod_dataset_item_access_and_collate(self, dirs):
        data = {"names": {0: "waste"}, "nc": 1}
        gt_ds = YOLODataset(img_path=str(dirs.gt_train_img), imgsz=128, data=data, augment=True)
        pl_ds = PseudoYOLODataset(
            images_dir=dirs.pl_img, labels_dir=dirs.pl_lbl, data=data, imgsz=128, filter_empty=True
        )

        ssod_ds = SSODDataset(gt_dataset=gt_ds, pl_dataset=pl_ds, imgsz=128)
        assert len(ssod_ds) == 4 + 2

        item_gt = ssod_ds[0]
        assert not item_gt["is_pseudo"]
        assert item_gt["source"] == 0
        assert "img" in item_gt
        assert "bboxes" in item_gt

        item_pl = ssod_ds[4]
        assert item_pl["is_pseudo"]
        assert item_pl["source"] == 1

        batch = ssod_ds.collate_fn([ssod_ds[0], ssod_ds[4]])
        assert batch["img"].shape[0] == 2
        assert isinstance(batch["is_pseudo"], torch.Tensor)
        assert batch["is_pseudo"].tolist() == [False, True]
        assert batch["source"].tolist() == [0, 1]

    def test_dynamic_dataframe_reloading(self, dirs):
        pl_ds = PseudoYOLODataset(images_dir=dirs.pl_img, labels_dir=dirs.pl_lbl, filter_empty=True)
        assert len(pl_ds) == 2

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
        assert len(pl_ds) == 1
        assert pl_ds.im_files[0] == str(dirs.pl_img / "pseudo_2.jpg")

    @pytest.mark.integration
    def test_handler_lifecycle_and_mosaic_close(self, dirs):
        handler = SSODDatasetHandler(
            gt_train_images=dirs.gt_train_img,
            gt_val_images=dirs.gt_val_img,
            pl_images_dir=dirs.pl_img,
            pl_labels_dir=dirs.pl_lbl,
            imgsz=128,
            batch_size=2,
            workers=0,
            close_mosaic_ratio=0.10,
        )

        assert len(handler.gt_train_dataset) == 4
        assert len(handler.pl_dataset) == 2
        assert len(handler.train_dataset) == 6

        train_loader = handler.get_train_loader()
        val_loader = handler.get_val_loader()

        assert val_loader is not None
        assert len(val_loader.dataset) == 2

        total_epochs = 10
        # Epoch 0 (first 90%): Mosaic must be open
        handler.set_epoch(epoch=0, total_epochs=total_epochs)
        assert not handler.train_dataset.is_mosaic_closed

        # Epoch 8: 8 < 9, Mosaic open
        handler.set_epoch(epoch=8, total_epochs=total_epochs)
        assert not handler.train_dataset.is_mosaic_closed

        # Epoch 9: 9 >= 9 (10 - 1), entering last 10%, Mosaic closed
        handler.set_epoch(epoch=9, total_epochs=total_epochs)
        assert handler.train_dataset.is_mosaic_closed

        # A ratio of 0 never closes Mosaic, not even in the final epoch
        handler.close_mosaic_ratio = 0.0
        handler.set_epoch(epoch=0, total_epochs=total_epochs)
        handler.set_epoch(epoch=total_epochs - 1, total_epochs=total_epochs)
        assert not handler.train_dataset.is_mosaic_closed

        # Iterate 1 batch to verify tensor execution
        batch = next(iter(train_loader))
        assert batch["img"].shape == (2, 3, 128, 128)
        assert len(batch["is_pseudo"]) == 2

    def test_pseudo_dataset_buffer_sync_from_empty(self, dirs, empty_pl_lbl, populate_pl_labels):
        pl_ds = PseudoYOLODataset(images_dir=dirs.pl_img, labels_dir=empty_pl_lbl, filter_empty=True)
        assert len(pl_ds) == 0
        assert len(pl_ds.ims) == 0
        assert len(pl_ds.npy_files) == 0

        populate_pl_labels(empty_pl_lbl)

        assert pl_ds.reload() == 2
        assert len(pl_ds) == 2
        assert len(pl_ds.ims) == 2
        assert len(pl_ds.npy_files) == 2
        assert len(pl_ds.im_hw0) == 2
        assert len(pl_ds.im_hw) == 2

        # Accessing all items must succeed without IndexError
        assert "img" in pl_ds[0]
        assert "img" in pl_ds[1]

    def test_pseudo_dataset_dataframe_buffer_sync_from_empty(self, dirs, empty_pl_lbl):
        pl_ds = PseudoYOLODataset(images_dir=dirs.pl_img, labels_dir=empty_pl_lbl, filter_empty=True)
        assert len(pl_ds) == 0
        assert len(pl_ds.ims) == 0

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
        assert len(pl_ds) == 2
        assert len(pl_ds.ims) == 2
        assert len(pl_ds.npy_files) == 2

        assert "img" in pl_ds[1]

    @pytest.mark.slow
    def test_pseudo_dataset_multiprocessing_dataloader_after_reload(
        self, dirs, empty_pl_lbl, populate_pl_labels
    ):
        pl_ds = PseudoYOLODataset(images_dir=dirs.pl_img, labels_dir=empty_pl_lbl, filter_empty=True)
        assert len(pl_ds) == 0

        populate_pl_labels(empty_pl_lbl)
        pl_ds.reload()

        loader = DataLoader(
            pl_ds,
            batch_size=2,
            num_workers=2,
            collate_fn=pl_ds.collate_fn,
            persistent_workers=False,
        )
        batches = list(loader)
        assert len(batches) == 1
        assert batches[0]["img"].shape == (2, 3, 640, 640)

    @pytest.mark.slow
    def test_handler_multiprocessing_dynamic_reload(self, dirs, empty_pl_lbl, populate_pl_labels):
        handler = SSODDatasetHandler(
            gt_train_images=dirs.gt_train_img,
            gt_val_images=dirs.gt_val_img,
            pl_images_dir=dirs.pl_img,
            pl_labels_dir=empty_pl_lbl,
            imgsz=128,
            batch_size=2,
            workers=2,
            close_mosaic_ratio=0.10,
        )
        # GT=4, PL=0
        assert len(handler.gt_train_dataset) == 4
        assert len(handler.pl_dataset) == 0
        assert len(handler.train_dataset) == 4

        # Epoch 1: pseudo labels generated
        populate_pl_labels(empty_pl_lbl)
        assert handler.reload_pseudo_labels() == 2
        assert len(handler.train_dataset) == 6

        all_pseudo_flags: list[bool] = []
        for batch in handler.get_train_loader():
            assert batch["img"].shape == (2, 3, 128, 128)
            all_pseudo_flags.extend(batch["is_pseudo"].tolist())

        assert len(all_pseudo_flags) == 6
        assert sum(all_pseudo_flags) == 2  # exactly 2 pseudo items


class TestPseudoBoxWeights:
    """Per-box pseudo-label weights must stay aligned with their boxes through augmentation and collation."""

    # Distinct widths (all near the centre, so augmentation never clips them) mapped to distinct weights
    BOXES = [(0.44, 0.10), (0.56, 0.14), (0.50, 0.18)]  # (x_center, width)
    WEIGHTS = [0.2, 0.5, 0.9]

    @pytest.fixture
    def pl_dirs(self, tmp_path, make_image, make_label):
        img_dir = tmp_path / "pseudo" / "images"
        lbl_dir = tmp_path / "pseudo" / "labels"
        for i in range(4):
            make_image(img_dir / f"pseudo_{i}.jpg", width=256, height=256)
            make_label(
                lbl_dir / f"pseudo_{i}.txt",
                [f"0 {x} 0.5 {w} {w} {wt}" for (x, w), wt in zip(self.BOXES, self.WEIGHTS)],
            )
        make_image(img_dir / "pseudo_empty.jpg", width=256, height=256)
        return SimpleNamespace(img=img_dir, lbl=lbl_dir)

    @pytest.fixture
    def make_dataset(self, pl_dirs):
        def _dataset(cfg: SSODAugmentConfig, filter_empty: bool = True) -> PseudoYOLODataset:
            ds = PseudoYOLODataset(
                images_dir=pl_dirs.img,
                labels_dir=pl_dirs.lbl,
                imgsz=128,
                hyp=cfg,
                filter_empty=filter_empty,
            )
            ds.transforms = build_pseudo_transforms(ds, imgsz=128, hyp=cfg)
            return ds

        return _dataset

    def _assert_aligned(self, batch: dict[str, Any]) -> None:
        n = batch["bboxes"].shape[0]
        assert batch["cls"].shape == (n, 1)
        assert batch["pl_weight"].shape == (n,)
        assert torch.all(batch["cls"] == 0)
        for image_index in batch["batch_idx"].unique():
            mask = batch["batch_idx"] == image_index
            widths = batch["bboxes"][mask][:, 2]
            weights = batch["pl_weight"][mask]
            order = torch.argsort(widths)
            # Width ordering is preserved by scale/translate/flip, so weights must follow it
            assert torch.all(torch.diff(weights[order]) > 0), (widths, weights)
            for w in weights.tolist():
                assert any(abs(w - ref) < 1e-4 for ref in self.WEIGHTS)

    @pytest.mark.parametrize(
        "cfg",
        [SSODAugmentConfig(), SSODAugmentConfig(pl_strong=True, erase_p=1.0)],
        ids=["weak", "strong"],
    )
    def test_weights_follow_boxes(self, make_dataset, cfg):
        ds = make_dataset(cfg)
        np.random.seed(0)
        seen_boxes = 0
        for _ in range(5):
            loader = DataLoader(ds, batch_size=4, collate_fn=PseudoYOLODataset.collate_fn)
            for batch in loader:
                self._assert_aligned(batch)
                seen_boxes += batch["bboxes"].shape[0]
        assert seen_boxes > 20, "augmentation must leave boxes to check"

    def test_empty_samples_collate(self, make_dataset):
        ds = make_dataset(SSODAugmentConfig(), filter_empty=False)
        assert len(ds) == 5
        batch = next(iter(DataLoader(ds, batch_size=5, collate_fn=PseudoYOLODataset.collate_fn)))
        self._assert_aligned(batch)
        empty = split_pl_weight({"cls": torch.zeros(0, 1)})
        assert empty["pl_weight"].shape == (0,)

    def test_dataframe_weights_through_ssod_collate(self, make_dataset):
        ds = make_dataset(SSODAugmentConfig())
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
        assert batch["bboxes"].shape[0] > 0
        assert torch.all(batch["is_pseudo"])
