"""
Automated unit and integration tests for YOLO Mean Teacher training loop.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace

# Ensure code directory is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from ssod.mean_teacher import (  # noqa: E402
    MeanTeacherConfig,
    compute_batch_loss,
    split_batch,
    update_teacher_ema,
)
from ssod.metrics import evaluate_coco, fix_prediction_ids  # noqa: E402


def _create_synthetic_image(path: Path, width: int = 128, height: int = 128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _create_synthetic_label(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines) + ("\n" if lines else "")
    path.write_text(content, encoding="utf-8")


class TestMeanTeacherEMA(unittest.TestCase):
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

        expected_weight = torch.full((2, 4), 1.2)
        self.assertTrue(torch.allclose(teacher.fc.weight, expected_weight, atol=1e-5))
        # Non-floating buffer copied directly
        self.assertEqual(teacher.counter.item(), 20)


class TestMeanTeacherLoss(unittest.TestCase):
    """Test batch splitting and default YOLO loss with unsupervised weighting."""

    def setUp(self):
        self.model = YOLO("yolo26n.pt")
        cfg_dict = dict(DEFAULT_CFG_DICT)
        self.model.model.args = IterableSimpleNamespace(**cfg_dict)
        self.model.model.train()
        for p in self.model.model.parameters():
            p.requires_grad = True

    def test_split_batch(self):
        img = torch.zeros(3, 3, 64, 64, dtype=torch.float32)
        bboxes = torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]], dtype=torch.float32)
        cls = torch.tensor([[0.0], [0.0]], dtype=torch.float32)
        batch_idx = torch.tensor([0.0, 2.0], dtype=torch.float32)
        mask = torch.tensor([True, False, True])

        batch = {"img": img, "bboxes": bboxes, "cls": cls, "batch_idx": batch_idx}
        sliced = split_batch(batch, mask)

        self.assertEqual(sliced["img"].shape[0], 2)
        self.assertEqual(sliced["bboxes"].shape[0], 2)
        # batch_idx remapped from [0, 2] to [0, 1]
        self.assertEqual(sliced["batch_idx"].tolist(), [0.0, 1.0])

    def test_compute_batch_loss_backward(self):
        img = torch.zeros(2, 3, 640, 640, dtype=torch.float32)
        bboxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32)
        cls = torch.tensor([[0.0]], dtype=torch.float32)
        batch_idx = torch.tensor([0.0], dtype=torch.float32)
        is_pseudo = torch.tensor([False, True])

        batch = {
            "img": img,
            "bboxes": bboxes,
            "cls": cls,
            "batch_idx": batch_idx,
            "is_pseudo": is_pseudo,
        }

        # 1. Uniform loss
        loss, _ = compute_batch_loss(self.model.model, batch, unsupervised_weight=1.0)
        self.assertTrue(loss.requires_grad)
        loss.backward()

        # 2. Weighted loss
        self.model.model.zero_grad()
        loss_w, _ = compute_batch_loss(self.model.model, batch, unsupervised_weight=0.5)
        self.assertTrue(loss_w.requires_grad)
        loss_w.backward()


class TestCOCOMetrics(unittest.TestCase):
    """Test COCO prediction ID fixing and pycocotools evaluation."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.gt_json_path = self.temp_dir / "annotations.json"
        self.pred_json_path = self.temp_dir / "predictions.json"

        # Create minimal COCO ground truth
        coco_gt = {
            "images": [
                {"id": 1, "file_name": "img_1.jpg", "width": 640, "height": 640},
                {"id": 2, "file_name": "img_2.jpg", "width": 640, "height": 640},
            ],
            "categories": [{"id": 0, "name": "bottle"}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 0,
                    "bbox": [100, 100, 50, 50],
                    "area": 2500,
                    "iscrowd": 0,
                },
                {
                    "id": 2,
                    "image_id": 2,
                    "category_id": 0,
                    "bbox": [200, 200, 80, 80],
                    "area": 6400,
                    "iscrowd": 0,
                },
            ],
        }
        with open(self.gt_json_path, "w", encoding="utf-8") as f:
            json.dump(coco_gt, f)

        # Create matching predictions
        preds = [
            {
                "image_id": 1,
                "file_name": "img_1.jpg",
                "category_id": 0,
                "bbox": [102, 101, 48, 52],
                "score": 0.95,
            },
            {
                "image_id": 2,
                "file_name": "img_2.jpg",
                "category_id": 0,
                "bbox": [198, 202, 82, 78],
                "score": 0.88,
            },
        ]
        with open(self.pred_json_path, "w", encoding="utf-8") as f:
            json.dump(preds, f)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_fix_prediction_ids(self):
        fixed = fix_prediction_ids(
            self.pred_json_path, self.gt_json_path, default_category_id=0
        )
        self.assertEqual(len(fixed), 2)
        self.assertEqual(fixed[0]["image_id"], 1)
        self.assertEqual(fixed[1]["image_id"], 2)

    def test_evaluate_coco(self):
        metrics = evaluate_coco(
            coco_gt_path=self.gt_json_path,
            predictions_path=self.pred_json_path,
            default_category_id=0,
        )

        self.assertIn("mAP_50_95", metrics)
        self.assertIn("mAP_50", metrics)
        self.assertIn("mAP_s", metrics)
        self.assertIn("mAP_m", metrics)
        self.assertIn("mAP_l", metrics)
        self.assertIn("peak_f1", metrics)
        self.assertGreater(metrics["mAP_50"], 0.8)
        self.assertGreater(metrics["peak_f1"], 0.8)


class TestMeanTeacherIntegration(unittest.TestCase):
    """Integration test verifying MeanTeacherConfig and dataset orchestration."""

    def test_config_initialization(self):
        cfg = MeanTeacherConfig(
            model_name="yolo26n.pt",
            warmup_epochs=5,
            ssl_epochs=20,
            batch_size=8,
            ema_decay=0.999,
            unsupervised_weight=0.75,
        )
        self.assertEqual(cfg.warmup_epochs, 5)
        self.assertEqual(cfg.ssl_epochs, 20)
        self.assertEqual(cfg.batch_size, 8)
        self.assertEqual(cfg.ema_decay, 0.999)
        self.assertEqual(cfg.unsupervised_weight, 0.75)


if __name__ == "__main__":
    unittest.main()
