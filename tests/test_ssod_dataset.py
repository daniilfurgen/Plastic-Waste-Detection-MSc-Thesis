"""
Automated unit and integration tests for SSOD dataset handler, sampler, and transform builders.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np
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

from ssod.dataset import (
    RandomErasing,
    RandomGaussianBlur,
    SSODAugmentConfig,
    SSODDataset,
    SSODDatasetHandler,
    SSODSampler,
    TeacherView,
    UnlabeledDataset,
    build_gt_transforms,
    build_unlabeled_transforms,
    build_val_transforms,
)


def _has_transform(transforms_obj: Any, cls_type: type) -> bool:
    """Recursively check whether a transform type is present in Compose."""
    if isinstance(transforms_obj, cls_type):
        return True
    if hasattr(transforms_obj, "transforms"):
        return any(_has_transform(t, cls_type) for t in transforms_obj.transforms)
    return False


def _transform_types(transforms_obj: Any) -> list[type]:
    return [type(t) for t in transforms_obj.transforms]


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

    def test_unlabeled_transforms_never_include_mosaic(self):
        transforms = build_unlabeled_transforms(DummyDataset(), imgsz=640, hyp=SSODAugmentConfig(mosaic=1.0))
        assert not _has_transform(transforms, Mosaic), "Unlabeled transforms must NEVER include Mosaic"
        for expected in (RandomPerspective, RandomHSV, RandomFlip, TeacherView):
            assert _has_transform(transforms, expected)

    def test_teacher_view_after_geometry_before_photometric(self):
        for cfg in (SSODAugmentConfig(), SSODAugmentConfig(pl_strong=True)):
            types = _transform_types(build_unlabeled_transforms(DummyDataset(), imgsz=640, hyp=cfg))
            teacher = types.index(TeacherView)
            assert types.index(RandomPerspective) < teacher
            assert types.index(RandomFlip) < teacher
            assert teacher < types.index(RandomHSV)

    def test_unlabeled_weak_transforms_have_no_strong_augmentation(self):
        weak = build_unlabeled_transforms(DummyDataset(), imgsz=640, hyp=SSODAugmentConfig())
        assert not _has_transform(weak, RandomGaussianBlur)
        assert not _has_transform(weak, RandomErasing)

    def test_unlabeled_strong_transforms(self):
        strong = build_unlabeled_transforms(DummyDataset(), imgsz=640, hyp=SSODAugmentConfig(pl_strong=True))
        assert not _has_transform(strong, Mosaic)
        assert _has_transform(strong, RandomGaussianBlur)
        assert _has_transform(strong, RandomErasing)
        assert _has_transform(strong, RandomFlip)

    def test_erasing_records_patches(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        erasing = RandomErasing(p=1.0, max_patches=1, area=(0.3, 0.3))
        # Force a single square patch (~55x55) anchored at (0, 0)
        original_randint = np.random.randint

        def fixed_randint(low, high=None, size=None, dtype=int):
            if size is not None:
                return original_randint(low, high, size=size, dtype=dtype)
            return 1 if high == 2 else 0

        with (
            mock.patch.object(np.random, "randint", side_effect=fixed_randint),
            mock.patch.object(np.random, "uniform", side_effect=lambda a, b: 0.0 if a < 0 else a),
        ):
            out = erasing({"img": img})

        assert out["erase_boxes"].tolist() == [[0.0, 0.0, 55.0, 55.0]]

    def test_erasing_skipped_records_no_patches(self):
        out = RandomErasing(p=0.0)({"img": np.zeros((10, 10, 3), dtype=np.uint8)})
        assert out["erase_boxes"].shape == (0, 4)

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
    """Synthetic GT (4 train / 2 val) dataset and an unlabeled pool of 3 images."""
    layout = SimpleNamespace(
        root=tmp_path,
        gt_train_img=tmp_path / "gt" / "images" / "train",
        gt_train_lbl=tmp_path / "gt" / "labels" / "train",
        gt_val_img=tmp_path / "gt" / "images" / "val",
        gt_val_lbl=tmp_path / "gt" / "labels" / "val",
        pl_img=tmp_path / "pseudo" / "images",
    )
    for i in range(4):
        make_image(layout.gt_train_img / f"gt_train_{i}.jpg", width=128, height=128)
        make_label(layout.gt_train_lbl / f"gt_train_{i}.txt", [f"0 0.{i+1} 0.{i+1} 0.2 0.2"])
    for i in range(2):
        make_image(layout.gt_val_img / f"gt_val_{i}.jpg", width=128, height=128)
        make_label(layout.gt_val_lbl / f"gt_val_{i}.txt", ["0 0.5 0.5 0.3 0.3"])
    layout.pl_files = []
    for i in range(3):
        make_image(layout.pl_img / f"pseudo_{i}.jpg", width=128, height=96)
        layout.pl_files.append(layout.pl_img / f"pseudo_{i}.jpg")
    return layout


def _unlabeled(dirs, cfg: SSODAugmentConfig | None = None, imgsz: int = 128) -> UnlabeledDataset:
    cfg = cfg or SSODAugmentConfig()
    ds = UnlabeledDataset(images_dir=dirs.pl_img, imgsz=imgsz, hyp=cfg)
    ds.transforms = build_unlabeled_transforms(ds, imgsz=imgsz, hyp=cfg)
    return ds


class TestUnlabeledDataset:
    """Unlabeled samples: no boxes, a weak Teacher view with the Student's geometry."""

    def test_starts_empty_and_follows_image_list(self, dirs):
        ds = _unlabeled(dirs)
        assert len(ds) == 0
        assert len(ds.ims) == len(ds.npy_files) == 0

        assert ds.set_images(dirs.pl_files[1:]) == 2
        assert ds.im_files == [str(p) for p in dirs.pl_files[1:]]
        assert len(ds.ims) == len(ds.npy_files) == len(ds.im_hw0) == len(ds.im_hw) == 2
        assert ds.set_images(dirs.pl_files) == 3

    def test_sample_has_teacher_view_and_no_boxes(self, dirs):
        ds = _unlabeled(dirs)
        ds.set_images(dirs.pl_files)
        sample = ds[0]
        assert sample["bboxes"].shape == (0, 4)
        assert sample["teacher_img"].shape == sample["img"].shape == (3, 128, 128)
        assert sample["teacher_img"].dtype == torch.uint8
        assert sample["erase_boxes"].shape == (0, 4)

    def test_teacher_view_shares_geometry_without_photometric_changes(self, dirs):
        # Without colour jitter both views are identical, flips and affine included
        cfg = SSODAugmentConfig(hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, fliplr=0.5)
        ds = _unlabeled(dirs, cfg)
        ds.set_images(dirs.pl_files)
        np.random.seed(0)
        for i in range(len(ds)):
            sample = ds[i]
            assert torch.equal(sample["img"], sample["teacher_img"])

        # With colour jitter only the Student view changes
        ds = _unlabeled(dirs, SSODAugmentConfig(hsv_v=0.9))
        ds.set_images(dirs.pl_files)
        assert any(not torch.equal(ds[i]["img"], ds[i]["teacher_img"]) for i in range(len(ds)))

    def test_erased_patches_only_in_student_view(self, dirs):
        cfg = SSODAugmentConfig(
            pl_strong=True, erase_p=1.0, blur_p=0.0, strong_hsv_h=0.0, strong_hsv_s=0.0, strong_hsv_v=0.0,
        )
        ds = _unlabeled(dirs, cfg)
        ds.set_images(dirs.pl_files)
        sample = ds[0]
        assert len(sample["erase_boxes"]) > 0
        outside = torch.ones(sample["img"].shape[1:], dtype=torch.bool)
        for x1, y1, x2, y2 in sample["erase_boxes"].long().tolist():
            outside[y1:y2, x1:x2] = False
        assert torch.equal(sample["img"][:, outside], sample["teacher_img"][:, outside])
        assert not torch.equal(sample["img"], sample["teacher_img"])

    def test_collate_stacks_teacher_views_and_patches(self, dirs):
        cfg = SSODAugmentConfig(pl_strong=True, erase_p=1.0)
        ds = _unlabeled(dirs, cfg)
        ds.set_images(dirs.pl_files)
        batch = next(iter(DataLoader(ds, batch_size=3, collate_fn=UnlabeledDataset.collate_fn)))
        assert batch["teacher_img"].shape == (3, 3, 128, 128)
        assert batch["erase_boxes"].shape[1] == 5
        assert set(batch["erase_boxes"][:, 0].tolist()) <= {0.0, 1.0, 2.0}

    @pytest.mark.slow
    def test_multiprocessing_dataloader_after_set_images(self, dirs):
        ds = _unlabeled(dirs, imgsz=640)
        ds.set_images(dirs.pl_files[:2])
        loader = DataLoader(
            ds, batch_size=2, num_workers=2, collate_fn=ds.collate_fn, persistent_workers=False
        )
        batches = list(loader)
        assert len(batches) == 1
        assert batches[0]["img"].shape == batches[0]["teacher_img"].shape == (2, 3, 640, 640)


class TestSSODDatasetIntegration:
    """End-to-end integration tests with synthetic image & label files."""

    def test_ssod_dataset_item_access_and_collate(self, dirs):
        data = {"names": {0: "waste"}, "nc": 1}
        gt_ds = YOLODataset(img_path=str(dirs.gt_train_img), imgsz=128, data=data, augment=True)
        pl_ds = UnlabeledDataset(images_dir=dirs.pl_img, data=data, imgsz=128)
        pl_ds.set_images(dirs.pl_files[:2])

        ssod_ds = SSODDataset(gt_dataset=gt_ds, pl_dataset=pl_ds, imgsz=128)
        assert len(ssod_ds) == 4 + 2

        item_gt = ssod_ds[0]
        assert not item_gt["is_pseudo"]
        assert item_gt["source"] == 0
        assert "teacher_img" not in item_gt

        item_pl = ssod_ds[4]
        assert item_pl["is_pseudo"]
        assert item_pl["source"] == 1

        batch = ssod_ds.collate_fn([ssod_ds[0], ssod_ds[4], ssod_ds[5]])
        assert batch["img"].shape[0] == 3
        assert batch["is_pseudo"].tolist() == [False, True, True]
        assert batch["source"].tolist() == [0, 1, 1]
        # Teacher views of the unlabeled samples only; GT boxes weighted 1
        assert batch["teacher_img"].shape == (2, 3, 128, 128)
        assert batch["erase_boxes"].shape == (0, 5)
        assert batch["pl_weight"].shape == (batch["bboxes"].shape[0],)
        assert torch.all(batch["pl_weight"] == 1)
        assert set(batch["batch_idx"].tolist()) == {0.0}

    def test_gt_only_batch_has_no_teacher_views(self, dirs):
        data = {"names": {0: "waste"}, "nc": 1}
        gt_ds = YOLODataset(img_path=str(dirs.gt_train_img), imgsz=128, data=data, augment=True)
        pl_ds = UnlabeledDataset(images_dir=dirs.pl_img, data=data, imgsz=128)
        ssod_ds = SSODDataset(gt_dataset=gt_ds, pl_dataset=pl_ds, imgsz=128)
        batch = ssod_ds.collate_fn([ssod_ds[0], ssod_ds[1]])
        assert batch["teacher_img"] is None

    @pytest.mark.integration
    def test_handler_lifecycle_and_mosaic_close(self, dirs):
        handler = SSODDatasetHandler(
            gt_train_images=dirs.gt_train_img,
            gt_val_images=dirs.gt_val_img,
            pl_images_dir=dirs.pl_img,
            imgsz=128,
            batch_size=2,
            workers=0,
            close_mosaic_ratio=0.10,
        )

        assert len(handler.gt_train_dataset) == 4
        assert len(handler.pl_dataset) == 0
        assert handler.set_pl_images(dirs.pl_files[:2]) == 2
        assert len(handler.train_dataset) == 6
        assert len(handler.sampler) == 6

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

        # Iterate the epoch to verify tensor execution
        n_pseudo = 0
        for batch in train_loader:
            assert batch["img"].shape == (2, 3, 128, 128)
            n_pseudo += int(batch["is_pseudo"].sum())
            n_views = 0 if batch["teacher_img"] is None else len(batch["teacher_img"])
            assert n_views == int(batch["is_pseudo"].sum())
        assert n_pseudo == 2

        # The unlabeled-only loader serves the Teacher views of the current images
        views = next(iter(handler.get_unlabeled_loader()))["teacher_img"]
        assert views.shape == (2, 3, 128, 128)

    @pytest.mark.slow
    def test_handler_multiprocessing_new_images_per_epoch(self, dirs):
        handler = SSODDatasetHandler(
            gt_train_images=dirs.gt_train_img,
            gt_val_images=dirs.gt_val_img,
            pl_images_dir=dirs.pl_img,
            imgsz=128,
            batch_size=2,
            workers=2,
            close_mosaic_ratio=0.10,
        )
        assert len(handler.train_dataset) == 4

        for files in (dirs.pl_files[:2], dirs.pl_files):
            handler.set_pl_images(files)
            all_pseudo_flags: list[bool] = []
            for batch in handler.get_train_loader(persistent_workers=False):
                assert batch["img"].shape[1:] == (3, 128, 128)
                all_pseudo_flags.extend(batch["is_pseudo"].tolist())
            assert len(all_pseudo_flags) == 4 + len(files)
            assert sum(all_pseudo_flags) == len(files)
