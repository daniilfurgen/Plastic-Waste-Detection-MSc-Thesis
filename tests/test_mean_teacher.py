"""
Automated unit and integration tests for YOLO Mean Teacher training loop.
"""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace

from ssod.mean_teacher import (
    MeanTeacherConfig,
    compute_batch_loss,
    configure_student_criterion,
    discover_coco_gt_path,
    pl_teacher_due,
    split_batch,
    update_teacher_ema,
)
from ssod.metrics import evaluate_coco, fix_prediction_ids


class TestMeanTeacherEMA:
    """Test step-level Exponential Moving Average parameter updates."""

    def test_update_teacher_ema_floating_point(self):
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 2, bias=False)
                self.register_buffer("counter", torch.tensor(0, dtype=torch.long))

        teacher = SimpleNet()
        student = SimpleNet()

        with torch.no_grad():
            teacher.fc.weight.fill_(1.0)
            teacher.counter.fill_(10)
            student.fc.weight.fill_(3.0)
            student.counter.fill_(20)

        # Decay = 0.9: new teacher = 0.9 * 1.0 + 0.1 * 3.0 = 1.2
        update_teacher_ema(teacher, student, decay=0.9)

        torch.testing.assert_close(teacher.fc.weight, torch.full((2, 4), 1.2), atol=1e-5, rtol=0)
        # Non-floating buffer copied directly
        assert teacher.counter.item() == 20


@pytest.fixture
def student(yolo26n_path):
    """Fresh trainable YOLO26n network; function-scoped because tests backpropagate through it."""
    net = YOLO(yolo26n_path).model
    net.args = IterableSimpleNamespace(**dict(DEFAULT_CFG_DICT))
    net.train()
    for p in net.parameters():
        p.requires_grad = True
    return net


@pytest.fixture
def mixed_batch() -> dict:
    return {
        "img": torch.zeros(3, 3, 640, 640),
        "bboxes": torch.tensor(
            [[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1], [0.6, 0.4, 0.2, 0.3], [0.4, 0.6, 0.2, 0.2]]
        ),
        "cls": torch.zeros(4, 1),
        "batch_idx": torch.tensor([0.0, 1.0, 1.0, 2.0]),
        "is_pseudo": torch.tensor([False, True, True]),
        "pl_weight": torch.tensor([1.0, 0.5, 0.8, 1.0]),
    }


class TestMeanTeacherLoss:
    """Test batch splitting and default YOLO loss with unsupervised weighting."""

    def test_split_batch(self):
        batch = {
            "img": torch.zeros(3, 3, 64, 64, dtype=torch.float32),
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]], dtype=torch.float32),
            "cls": torch.tensor([[0.0], [0.0]], dtype=torch.float32),
            "batch_idx": torch.tensor([0.0, 2.0], dtype=torch.float32),
        }
        sliced = split_batch(batch, torch.tensor([True, False, True]))

        assert sliced["img"].shape[0] == 2
        assert sliced["bboxes"].shape[0] == 2
        # batch_idx remapped from [0, 2] to [0, 1]
        assert sliced["batch_idx"].tolist() == [0.0, 1.0]

    @pytest.mark.parametrize("unsupervised_weight", [1.0, 0.5])
    def test_compute_batch_loss_backward(self, student, unsupervised_weight):
        batch = {
            "img": torch.zeros(2, 3, 640, 640, dtype=torch.float32),
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
            "cls": torch.tensor([[0.0]], dtype=torch.float32),
            "batch_idx": torch.tensor([0.0], dtype=torch.float32),
            "is_pseudo": torch.tensor([False, True]),
        }
        loss, _ = compute_batch_loss(student, batch, unsupervised_weight=unsupervised_weight)
        assert loss.requires_grad
        loss.backward()

    def test_split_batch_slices_pl_weight(self):
        batch = {
            "img": torch.zeros(3, 3, 64, 64),
            "bboxes": torch.rand(4, 4),
            "cls": torch.zeros(4, 1),
            "batch_idx": torch.tensor([0.0, 1.0, 1.0, 2.0]),
            "pl_weight": torch.tensor([0.1, 0.2, 0.3, 0.4]),
        }
        sliced = split_batch(batch, torch.tensor([False, True, True]))
        torch.testing.assert_close(sliced["pl_weight"], torch.tensor([0.2, 0.3, 0.4]))
        assert sliced["batch_idx"].tolist() == [0.0, 0.0, 1.0]

    def test_weighted_loss_with_unsupervised_split(self, student):
        configure_student_criterion(student)
        batch = {
            "img": torch.zeros(2, 3, 640, 640),
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]]),
            "cls": torch.zeros(2, 1),
            "batch_idx": torch.tensor([0.0, 1.0]),
            "is_pseudo": torch.tensor([False, True]),
            "pl_weight": torch.tensor([1.0, 0.4]),
        }
        loss, _ = compute_batch_loss(student, batch, unsupervised_weight=0.5)
        assert torch.isfinite(loss)
        loss.backward()

    def test_pl_cls_only_masks_pseudo_box_weights(self, student, mixed_batch):
        configure_student_criterion(student)
        loss, _ = compute_batch_loss(student, mixed_batch, pl_cls_only=True)
        torch.testing.assert_close(mixed_batch["box_weight"], torch.tensor([1.0, 0.0, 0.0, 0.0]))
        assert torch.isfinite(loss)
        loss.backward()

    def test_pl_cls_only_with_unsupervised_split(self, student, mixed_batch):
        configure_student_criterion(student)
        loss, _ = compute_batch_loss(student, mixed_batch, unsupervised_weight=0.5, pl_cls_only=True)
        assert torch.isfinite(loss)
        pseudo_half = split_batch(mixed_batch, mixed_batch["is_pseudo"])
        torch.testing.assert_close(pseudo_half["box_weight"], torch.zeros(3))
        torch.testing.assert_close(pseudo_half["pl_weight"], torch.tensor([0.5, 0.8, 1.0]))

    def test_pl_cls_only_off_adds_no_box_weight(self, student, mixed_batch):
        configure_student_criterion(student)
        compute_batch_loss(student, mixed_batch)
        assert "box_weight" not in mixed_batch

    def test_configure_student_criterion_pins_e2e_gains(self, student):
        criterion = configure_student_criterion(student)
        assert student.criterion is criterion
        # YOLO26 is end-to-end: gains must match the end of Ultralytics' decay schedule
        assert criterion.o2m == pytest.approx(criterion.final_o2m)
        assert criterion.o2o == pytest.approx(1.0 - criterion.final_o2m)
        assert criterion.o2o > criterion.o2m


class TestCOCOMetrics:
    """Test COCO prediction ID fixing and pycocotools evaluation."""

    @pytest.fixture
    def coco_files(self, tmp_path):
        gt_path = tmp_path / "annotations.json"
        pred_path = tmp_path / "predictions.json"

        coco_gt = {
            "images": [
                {"id": 1, "file_name": "img_1.jpg", "width": 640, "height": 640},
                {"id": 2, "file_name": "img_2.jpg", "width": 640, "height": 640},
            ],
            "categories": [{"id": 0, "name": "bottle"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 0, "bbox": [100, 100, 50, 50], "area": 2500, "iscrowd": 0},
                {"id": 2, "image_id": 2, "category_id": 0, "bbox": [200, 200, 80, 80], "area": 6400, "iscrowd": 0},
            ],
        }
        gt_path.write_text(json.dumps(coco_gt), encoding="utf-8")

        preds = [
            {"image_id": 1, "file_name": "img_1.jpg", "category_id": 0, "bbox": [102, 101, 48, 52], "score": 0.95},
            {"image_id": 2, "file_name": "img_2.jpg", "category_id": 0, "bbox": [198, 202, 82, 78], "score": 0.88},
        ]
        pred_path.write_text(json.dumps(preds), encoding="utf-8")
        return gt_path, pred_path

    def test_fix_prediction_ids(self, coco_files):
        gt_path, pred_path = coco_files
        fixed = fix_prediction_ids(pred_path, gt_path, default_category_id=0)
        assert len(fixed) == 2
        assert fixed[0]["image_id"] == 1
        assert fixed[1]["image_id"] == 2

    def test_evaluate_coco(self, coco_files):
        gt_path, pred_path = coco_files
        metrics = evaluate_coco(coco_gt_path=gt_path, predictions_path=pred_path, default_category_id=0)

        for key in ("mAP_50_95", "mAP_50", "mAP_s", "mAP_m", "mAP_l", "peak_f1"):
            assert key in metrics
        assert metrics["mAP_50"] > 0.8
        assert metrics["peak_f1"] > 0.8


class TestCOCOMultiClass:
    """Ultralytics' 1-based category ids map back to the GT's YOLO class ids."""

    @pytest.fixture
    def coco_files(self, tmp_path):
        gt_path = tmp_path / "annotations.json"
        pred_path = tmp_path / "predictions.json"
        gt = {
            "images": [{"id": 1, "file_name": "a.jpg", "width": 640, "height": 640}],
            "categories": [{"id": 0, "name": "glass"}, {"id": 1, "name": "metal"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 10, 50, 50], "area": 2500, "iscrowd": 0},
                {"id": 2, "image_id": 1, "category_id": 1, "bbox": [200, 200, 60, 60], "area": 3600, "iscrowd": 0},
            ],
        }
        gt_path.write_text(json.dumps(gt))
        preds = [
            {"image_id": "a", "file_name": "a.jpg", "category_id": 1, "bbox": [10, 10, 50, 50], "score": 0.9},
            {"image_id": "a", "file_name": "a.jpg", "category_id": 2, "bbox": [200, 200, 60, 60], "score": 0.9},
            {"image_id": "a", "file_name": "a.jpg", "category_id": 8, "bbox": [400, 400, 20, 20], "score": 0.5},
        ]
        pred_path.write_text(json.dumps(preds))
        return gt_path, pred_path

    def test_class_ids_mapped(self, coco_files):
        gt_path, pred_path = coco_files
        fixed = fix_prediction_ids(pred_path, gt_path, default_category_id=0)
        assert [p["category_id"] for p in fixed] == [0, 1, 0]  # unknown id -> default
        assert all(p["image_id"] == 1 for p in fixed)

    def test_predictions_file_unchanged(self, coco_files):
        gt_path, pred_path = coco_files
        before = pred_path.read_text()
        fix_prediction_ids(pred_path, gt_path)
        evaluate_coco(gt_path, pred_path)
        assert pred_path.read_text() == before

    def test_multiclass_map(self, coco_files):
        gt_path, pred_path = coco_files
        metrics = evaluate_coco(gt_path, pred_path, default_category_id=0)
        assert metrics["mAP_50"] > 0.99


class TestCOCOSizeBins:
    """COCO size bins are applied at the network input scale, per image."""

    @pytest.fixture
    def coco_files(self, tmp_path):
        gt_path = tmp_path / "annotations.json"
        pred_path = tmp_path / "predictions.json"
        # A 40x40 box is medium in original pixels; at imgsz 640 it is 20x20 on the 1280-px image
        # (small) and 40x40 on the 640-px image (medium).
        gt = {
            "images": [
                {"id": 1, "file_name": "big.jpg", "width": 1280, "height": 720},
                {"id": 2, "file_name": "native.jpg", "width": 640, "height": 480},
            ],
            "categories": [{"id": 0, "name": "bottle"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 0, "bbox": [100, 100, 40, 40], "area": 1600, "iscrowd": 0},
                {"id": 2, "image_id": 2, "category_id": 0, "bbox": [100, 100, 40, 40], "area": 1600, "iscrowd": 0},
            ],
        }
        gt_path.write_text(json.dumps(gt))
        preds = [
            {"image_id": 1, "file_name": "big.jpg", "category_id": 1, "bbox": [100, 100, 40, 40], "score": 0.9},
            {"image_id": 2, "file_name": "native.jpg", "category_id": 1, "bbox": [100, 100, 40, 40], "score": 0.9},
        ]
        pred_path.write_text(json.dumps(preds))
        return gt_path, pred_path

    def test_original_pixels(self, coco_files):
        metrics = evaluate_coco(*coco_files, imgsz=None)
        assert metrics["mAP_s"] == -1.0  # no small GT
        assert metrics["AR_m"] == pytest.approx(1.0)

    def test_input_scale_per_image(self, coco_files):
        metrics = evaluate_coco(*coco_files, imgsz=640)
        assert metrics["mAP_s"] == pytest.approx(1.0)
        assert metrics["AR_s"] == pytest.approx(1.0)
        assert metrics["mAP_m"] == pytest.approx(1.0)  # the 640-px image keeps its box medium

    def test_overall_metrics_independent_of_scale(self, coco_files):
        native = evaluate_coco(*coco_files, imgsz=None)
        scaled = evaluate_coco(*coco_files, imgsz=640)
        for key in ("mAP_50_95", "mAP_50", "mAP_75", "AR_100", "peak_f1"):
            assert scaled[key] == native[key]


def test_discover_coco_gt_prefers_dataset_annotations(tmp_path):
    """The dataset's own val annotations win over the FloW fallback next to its YAML."""
    for ds in ("flow_img", "ucwd"):
        (tmp_path / ds / "val").mkdir(parents=True)
        (tmp_path / ds / "val" / "annotations.json").write_text("{}")
    yaml_path = tmp_path / "ucwd.yaml"
    yaml_path.write_text(f"path: {tmp_path / 'ucwd'}\ntrain: train/images\nval: val/images\n")
    assert discover_coco_gt_path(yaml_path) == tmp_path / "ucwd" / "val" / "annotations.json"


class TestPseudoLabelTeacherRefresh:
    """The pseudo-labelling Teacher snapshot is refreshed every k epochs, starting at the first."""

    def test_every_epoch_by_default(self):
        assert all(pl_teacher_due(epoch, 1) for epoch in range(10))

    @pytest.mark.parametrize(
        ("epochs", "k", "expected"),
        [(10, 3, [0, 3, 6, 9]), (15, 8, [0, 8])],
    )
    def test_every_k_epochs(self, epochs, k, expected):
        assert [e for e in range(epochs) if pl_teacher_due(e, k)] == expected


def test_mean_teacher_config_initialization():
    cfg = MeanTeacherConfig(
        model_name="yolo26n.pt",
        warmup_epochs=5,
        ssl_epochs=20,
        batch_size=8,
        ema_decay=0.999,
        unsupervised_weight=0.75,
    )
    assert cfg.warmup_epochs == 5
    assert cfg.ssl_epochs == 20
    assert cfg.batch_size == 8
    assert cfg.ema_decay == 0.999
    assert cfg.unsupervised_weight == 0.75
