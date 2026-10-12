"""
Integration tests verifying multi-epoch Mean Teacher transitions,
VectorizedEMA precision, and model integrity without dtype or BatchNorm corruption.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import torch
import torch.nn as nn
from ultralytics import YOLO

from ssod.mean_teacher import (
    MeanTeacherConfig,
    MeanTeacherTrainer,
    VectorizedEMA,
    update_teacher_ema,
)


def test_vectorized_ema_matches_baseline():
    """VectorizedEMA must reproduce the baseline update_teacher_ema."""

    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 3, padding=1)
            self.bn = nn.BatchNorm2d(4)
            self.fc = nn.Linear(4, 2)
            self.register_buffer("counter", torch.tensor(10, dtype=torch.long))

    torch.manual_seed(42)
    teacher_v = SimpleNet()
    student = SimpleNet()

    # Copy state to reference teacher
    teacher_ref = SimpleNet()
    teacher_ref.load_state_dict(teacher_v.state_dict())

    ema = VectorizedEMA(teacher_v, student, decay=0.9)
    update_teacher_ema(teacher_ref, student, decay=0.9)
    ema.update()

    for (n1, p1), (n2, p2) in zip(teacher_v.named_parameters(), teacher_ref.named_parameters()):
        assert n1 == n2
        torch.testing.assert_close(p1, p2, atol=1e-6, rtol=0, msg=f"Parameter mismatch for {n1}")

    for (n1, b1), (n2, b2) in zip(teacher_v.named_buffers(), teacher_ref.named_buffers()):
        assert n1 == n2
        assert torch.equal(b1, b2), f"Buffer mismatch for {n1}"


@pytest.fixture
def env(tmp_path, yolo26n_path, device, make_image, make_label):
    """Tiny synthetic GT dataset, unlabeled pool and COCO val annotations for a dry run."""
    gt_dir = tmp_path / "gt"
    pl_images_dir = tmp_path / "pl_images"

    for split, count in [("train", 4), ("val", 2)]:
        for i in range(count):
            make_image(gt_dir / f"images/{split}/img_{i:03d}.jpg", 128, 128)
            make_label(gt_dir / f"labels/{split}/img_{i:03d}.txt", ["0 0.5 0.5 0.2 0.2"])

    for i in range(4):
        make_image(pl_images_dir / f"unlabeled_{i:03d}.jpg", 128, 128)

    gt_yaml = tmp_path / "data.yaml"
    gt_yaml.write_text(
        f"path: {gt_dir}\ntrain: images/train\nval: images/val\nnames:\n  0: plastic",
        encoding="utf-8",
    )

    coco_gt = tmp_path / "val_coco.json"
    coco_gt.write_text(
        json.dumps({
            "images": [
                {"id": 1, "file_name": "img_000.jpg", "width": 128, "height": 128},
                {"id": 2, "file_name": "img_001.jpg", "width": 128, "height": 128},
            ],
            "annotations": [
                {"id": i, "image_id": i, "category_id": 0, "bbox": [51.2, 51.2, 25.6, 25.6],
                 "area": 655.36, "iscrowd": 0}
                for i in (1, 2)
            ],
            "categories": [{"id": 0, "name": "plastic"}],
        }),
        encoding="utf-8",
    )

    def make_config(**overrides) -> MeanTeacherConfig:
        return MeanTeacherConfig(
            **{
                "model_name": yolo26n_path,
                "warmup_weights": Path(yolo26n_path),
                "gt_data_yaml": gt_yaml,
                "pl_images_dir": pl_images_dir,
                "pl_labels_dir": tmp_path / "pl_labels",
                "coco_gt_path": coco_gt,
                "warmup_epochs": 0,
                "batch_size": 2,
                "imgsz": 128,
                "workers": 0,
                "device": device,
                "ema_decay": 0.99,
                "runs_dir": tmp_path / "runs",
                **overrides,
            }
        )

    return SimpleNamespace(
        make_config=make_config, runs_dir=tmp_path / "runs", weights=Path(yolo26n_path)
    )


@pytest.mark.slow
@pytest.mark.integration
class TestMultiEpochTransitions:
    """Integration test running a multi-epoch dry run verifying transitions and model integrity."""

    def test_multi_epoch_ssl_transitions_and_integrity(self, env):
        """
        Run 3 SSL epochs:
        - Epoch 1 -> 2: Verifies no AutoBackend mutation or dtype mismatch crash.
        - Epoch 2 -> 3: Verifies mosaic close transition (close_mosaic_ratio=0.34).
        - Verifies Teacher retains BatchNorm2d layers and FP32 dtypes.
        """
        cfg = env.make_config(
            ssl_epochs=3,
            close_mosaic_ratio=0.34,  # Triggers mosaic close on epoch 3 (index 2)
            experiment_name="test_transition",
        )
        results = MeanTeacherTrainer(config=cfg).train_mean_teacher(warmup_weights=env.weights)

        # Epoch-0 baseline row plus all 3 epochs completed
        assert [h["epoch"] for h in results["history"]] == [0, 1, 2, 3]
        for h in results["history"]:
            assert np.isfinite(h["train_loss"])

        best_model_path = Path(results["best_model"])
        assert best_model_path.exists()

        # BatchNorm layers must NOT be fused away
        saved_teacher = YOLO(str(best_model_path))
        has_bn = any(isinstance(m, nn.BatchNorm2d) for m in saved_teacher.model.modules())
        assert has_bn, "Teacher model lost BatchNorm layers (improper fusion)!"

        dtypes = {p.dtype for p in saved_teacher.model.parameters()}
        assert torch.float32 in dtypes

    def test_save_candidates_parquet_per_epoch(self, env):
        """
        Run 2 SSL epochs with save_candidates=True:
        - Verifies that <runs_dir>/candidates/ directory is created.
        - Verifies that candidates_epoch_1.parquet and candidates_epoch_2.parquet are saved.
        - Reads parquet files with Polars and asserts expected schema columns.
        """
        cfg = env.make_config(ssl_epochs=2, save_candidates=True, experiment_name="test_cand_parquet")
        MeanTeacherTrainer(config=cfg).train_mean_teacher(warmup_weights=env.weights)

        cand_dir = env.runs_dir / "test_cand_parquet" / "candidates"
        assert cand_dir.exists(), "Candidates directory was not created!"

        expected_cols = [
            "image_stem", "label_path", "img_width", "img_height", "class_id", "score",
            "x1", "y1", "x2", "y2", "w", "h", "size_bin",
        ]
        for epoch in (1, 2):
            p_file = cand_dir / f"candidates_epoch_{epoch}.parquet"
            assert p_file.exists(), f"{p_file} does not exist!"
            df = pl.read_parquet(p_file)
            for col in expected_cols:
                assert col in df.columns
            if len(df) > 0:
                assert all(0.0 <= s <= 1.0 for s in df["score"].to_list())
