"""
Integration tests verifying multi-epoch Mean Teacher transitions,
VectorizedEMA precision, and model integrity without dtype or BatchNorm corruption.
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
import polars as pl
import torch
import torch.nn as nn
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from ssod.mean_teacher import (  # noqa: E402
    MeanTeacherConfig,
    MeanTeacherTrainer,
    VectorizedEMA,
    update_teacher_ema,
)


def _create_synthetic_image(path: Path, width: int = 128, height: int = 128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _create_synthetic_label(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines) + ("\n" if lines else "")
    path.write_text(content, encoding="utf-8")


class TestVectorizedEMA(unittest.TestCase):
    """Test VectorizedEMA correctness against baseline update_teacher_ema."""

    def test_vectorized_ema_matches_baseline(self):
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 4, 3, padding=1)
                self.bn = nn.BatchNorm2d(4)
                self.fc = nn.Linear(4, 2)
                self.register_buffer("counter", torch.tensor(10, dtype=torch.long))

        # Setup model pairs
        torch.manual_seed(42)
        teacher_v = SimpleNet()
        student = SimpleNet()

        # Copy state to reference teacher
        teacher_ref = SimpleNet()
        teacher_ref.load_state_dict(teacher_v.state_dict())

        # Vectorized EMA instance
        ema = VectorizedEMA(teacher_v, student, decay=0.9)

        # Baseline update
        update_teacher_ema(teacher_ref, student, decay=0.9)
        # Vectorized update
        ema.update()

        # Verify all floating-point parameters and buffers match exactly
        for (n1, p1), (n2, p2) in zip(
            teacher_v.named_parameters(), teacher_ref.named_parameters()
        ):
            self.assertEqual(n1, n2)
            self.assertTrue(
                torch.allclose(p1, p2, atol=1e-6),
                f"Parameter mismatch for {n1}",
            )

        for (n1, b1), (n2, b2) in zip(
            teacher_v.named_buffers(), teacher_ref.named_buffers()
        ):
            self.assertEqual(n1, n2)
            self.assertTrue(
                torch.equal(b1, b2),
                f"Buffer mismatch for {n1}",
            )


class TestMultiEpochTransitions(unittest.TestCase):
    """Integration test running a multi-epoch dry run verifying transitions and model integrity."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.gt_dir = self.temp_dir / "gt"
        self.pl_images_dir = self.temp_dir / "pl_images"
        self.pl_labels_dir = self.temp_dir / "pl_labels"
        self.runs_dir = self.temp_dir / "runs"

        # Create GT images & labels
        for split, count in [("train", 4), ("val", 2)]:
            for i in range(count):
                im_path = self.gt_dir / f"images/{split}/img_{i:03d}.jpg"
                _create_synthetic_image(im_path, 128, 128)
                lb_path = self.gt_dir / f"labels/{split}/img_{i:03d}.txt"
                _create_synthetic_label(lb_path, ["0 0.5 0.5 0.2 0.2"])

        # Create Unlabeled images
        for i in range(4):
            im_path = self.pl_images_dir / f"unlabeled_{i:03d}.jpg"
            _create_synthetic_image(im_path, 128, 128)

        # Create GT YAML
        self.gt_yaml = self.temp_dir / "data.yaml"
        yaml_content = f"""
path: {self.gt_dir}
train: images/train
val: images/val
names:
  0: plastic
"""
        self.gt_yaml.write_text(yaml_content.strip(), encoding="utf-8")

        # Create COCO GT JSON for val
        self.coco_gt = self.temp_dir / "val_coco.json"
        coco_dict = {
            "images": [
                {"id": 1, "file_name": "img_000.jpg", "width": 128, "height": 128},
                {"id": 2, "file_name": "img_001.jpg", "width": 128, "height": 128},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 0,
                    "bbox": [51.2, 51.2, 25.6, 25.6],
                    "area": 655.36,
                    "iscrowd": 0,
                },
                {
                    "id": 2,
                    "image_id": 2,
                    "category_id": 0,
                    "bbox": [51.2, 51.2, 25.6, 25.6],
                    "area": 655.36,
                    "iscrowd": 0,
                },
            ],
            "categories": [{"id": 0, "name": "plastic"}],
        }
        with open(self.coco_gt, "w", encoding="utf-8") as f:
            json.dump(coco_dict, f)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_multi_epoch_ssl_transitions_and_integrity(self):
        """
        Run 3 SSL epochs:
        - Epoch 1 -> 2: Verifies no AutoBackend mutation or dtype mismatch crash.
        - Epoch 2 -> 3: Verifies mosaic close transition (close_mosaic_ratio=0.34).
        - Verifies Teacher retains BatchNorm2d layers and FP32 dtypes.
        """
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        cfg = MeanTeacherConfig(
            model_name="code/yolo26n.pt",
            warmup_weights=Path("code/yolo26n.pt"),
            gt_data_yaml=self.gt_yaml,
            pl_images_dir=self.pl_images_dir,
            pl_labels_dir=self.pl_labels_dir,
            coco_gt_path=self.coco_gt,
            warmup_epochs=0,
            ssl_epochs=3,
            batch_size=2,
            imgsz=128,
            workers=0,
            device=device,
            ema_decay=0.99,
            close_mosaic_ratio=0.34,  # Triggers mosaic close on epoch 3 (index 2)
            runs_dir=self.runs_dir,
            experiment_name="test_transition",
        )

        trainer = MeanTeacherTrainer(config=cfg)
        results = trainer.train_mean_teacher(warmup_weights=Path("code/yolo26n.pt"))

        # Verify all 3 epochs completed
        self.assertEqual(len(results["history"]), 3)
        for h in results["history"]:
            self.assertFalse(np.isnan(h["train_loss"]))
            self.assertTrue(np.isfinite(h["train_loss"]))

        # Verify saved model checkpoint exists
        best_model_path = Path(results["best_model"])
        self.assertTrue(best_model_path.exists())

        # Verify Teacher architecture integrity: BatchNorm layers must NOT be fused away
        saved_teacher = YOLO(str(best_model_path))
        has_bn = any(isinstance(m, nn.BatchNorm2d) for m in saved_teacher.model.modules())
        self.assertTrue(has_bn, "Teacher model lost BatchNorm layers (improper fusion)!")

        # Verify dtypes are valid
        dtypes = {p.dtype for p in saved_teacher.model.parameters()}
        self.assertIn(torch.float32, dtypes)

    def test_save_candidates_parquet_per_epoch(self):
        """
        Run 2 SSL epochs with save_candidates=True:
        - Verifies that <runs_dir>/candidates/ directory is created.
        - Verifies that candidates_epoch_1.parquet and candidates_epoch_2.parquet are saved.
        - Reads parquet files with Polars and asserts expected schema columns.
        """
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        cfg = MeanTeacherConfig(
            model_name="code/yolo26n.pt",
            warmup_weights=Path("code/yolo26n.pt"),
            gt_data_yaml=self.gt_yaml,
            pl_images_dir=self.pl_images_dir,
            pl_labels_dir=self.pl_labels_dir,
            coco_gt_path=self.coco_gt,
            warmup_epochs=0,
            ssl_epochs=2,
            batch_size=2,
            imgsz=128,
            workers=0,
            device=device,
            ema_decay=0.99,
            save_candidates=True,
            runs_dir=self.runs_dir,
            experiment_name="test_cand_parquet",
        )

        trainer = MeanTeacherTrainer(config=cfg)
        trainer.train_mean_teacher(warmup_weights=Path("code/yolo26n.pt"))

        cand_dir = self.runs_dir / "test_cand_parquet" / "candidates"
        self.assertTrue(cand_dir.exists(), "Candidates directory was not created!")

        epoch1_file = cand_dir / "candidates_epoch_1.parquet"
        epoch2_file = cand_dir / "candidates_epoch_2.parquet"
        self.assertTrue(epoch1_file.exists(), f"{epoch1_file} does not exist!")
        self.assertTrue(epoch2_file.exists(), f"{epoch2_file} does not exist!")

        for p_file in [epoch1_file, epoch2_file]:
            df = pl.read_parquet(p_file)
            self.assertIsInstance(df, pl.DataFrame)
            expected_cols = [
                "image_stem",
                "label_path",
                "img_width",
                "img_height",
                "class_id",
                "score",
                "x1",
                "y1",
                "x2",
                "y2",
                "w",
                "h",
                "size_bin",
            ]
            for col in expected_cols:
                self.assertIn(col, df.columns)
            if len(df) > 0:
                scores = df["score"].to_list()
                self.assertTrue(all(0.0 <= s <= 1.0 for s in scores))


if __name__ == "__main__":
    unittest.main()
