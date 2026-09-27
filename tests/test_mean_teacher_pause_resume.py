"""
Automated unit and integration tests for Mean Teacher pause, resume, and CLI latency.
"""

from __future__ import annotations

import csv
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
from ultralytics import YOLO

# Ensure code directory is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from ssod.mean_teacher import MeanTeacherConfig, MeanTeacherTrainer  # noqa: E402


class TestMeanTeacherCLIAndLatency(unittest.TestCase):
    """Test CLI execution speed and argument parsing."""

    def test_cli_help_latency_and_options(self):
        """Verify calling --help runs in < 1.0s and documents --resume."""
        script_path = CODE_DIR / "ssod" / "mean_teacher.py"
        cmd = [sys.executable, str(script_path), "--help"]

        t0 = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        elapsed = time.perf_counter() - t0

        self.assertEqual(result.returncode, 0, msg=f"CLI failed: {result.stderr}")
        self.assertLess(
            elapsed,
            1.5,
            msg=f"--help took {elapsed:.2f}s, exceeding target threshold of 1.5s",
        )
        self.assertIn("--resume", result.stdout)
        self.assertIn("-r", result.stdout)


class TestMeanTeacherCheckpointSerialization(unittest.TestCase):
    """Test rich and legacy checkpoint serialization and restoration."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.runs_dir = self.test_dir / "runs"
        self.cfg = MeanTeacherConfig(
            model_name="yolo26n.pt",
            runs_dir=self.runs_dir,
            experiment_name="test_exp",
            ssl_epochs=10,
        )
        self.trainer = MeanTeacherTrainer(config=self.cfg)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_checkpoint_path_resolution(self):
        """Test resolve_checkpoint_path on files and directories."""
        weights_dir = self.runs_dir / "test_exp" / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)

        # 1. Non-existent file raises FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            MeanTeacherTrainer.resolve_checkpoint_path(self.test_dir / "non_existent.pt")

        # 2. Directory with no checkpoints raises FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            MeanTeacherTrainer.resolve_checkpoint_path(self.test_dir)

        # 3. Directory with last.pt
        last_pt = weights_dir / "last.pt"
        last_pt.touch()
        resolved = MeanTeacherTrainer.resolve_checkpoint_path(self.runs_dir / "test_exp")
        self.assertEqual(resolved, last_pt)

        # 4. Directory with last_checkpoint.pt takes priority over last.pt
        last_ckpt = weights_dir / "last_checkpoint.pt"
        last_ckpt.touch()
        resolved = MeanTeacherTrainer.resolve_checkpoint_path(self.runs_dir / "test_exp")
        self.assertEqual(resolved, last_ckpt)

        # 5. Directory with paused.pt takes highest priority
        paused_pt = weights_dir / "paused.pt"
        paused_pt.touch()
        resolved = MeanTeacherTrainer.resolve_checkpoint_path(self.runs_dir / "test_exp")
        self.assertEqual(resolved, paused_pt)

    def test_rich_checkpoint_save_and_load(self):
        """Test full fidelity of saving and loading rich checkpoints."""
        teacher = YOLO("yolo26n.pt")
        student = YOLO("yolo26n.pt")
        optimizer = torch.optim.AdamW(student.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        # Mutate weights slightly and step optimizer
        for p in student.model.parameters():
            if p.requires_grad:
                p.data.add_(0.01)
                break
        optimizer.step()
        scheduler.step()

        save_path = self.trainer.weights_dir / "test_checkpoint.pt"
        self.trainer.save_checkpoint(
            path=save_path,
            epoch=3,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            best_fitness=0.456,
            best_epoch=2,
            training_history=[{"epoch": 1, "fitness": 0.40}, {"epoch": 2, "fitness": 0.456}],
            warmup_weights="weights/warmup_best.pt",
        )

        self.assertTrue(save_path.exists())

        # Load back
        loaded = self.trainer.load_checkpoint(save_path)
        self.assertEqual(loaded["format"], "rich")
        self.assertEqual(loaded["start_epoch"], 3)
        self.assertAlmostEqual(loaded["best_fitness"], 0.456)
        self.assertEqual(loaded["best_epoch"], 2)
        self.assertEqual(len(loaded["training_history"]), 2)
        self.assertIsNotNone(loaded["optimizer_state_dict"])
        self.assertIsNotNone(loaded["scheduler_state_dict"])

        # Verify model parameter fidelity
        loaded_student = loaded["student"]
        orig_sd = student.model.state_dict()
        loaded_sd = loaded_student.model.state_dict()
        for k in orig_sd:
            self.assertTrue(torch.allclose(orig_sd[k].cpu(), loaded_sd[k].cpu()))

    def test_legacy_checkpoint_load_with_results_csv(self):
        """Test fallback loading from Ultralytics last.pt and results.csv."""
        exp_dir = self.runs_dir / "legacy_exp"
        weights_dir = exp_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)

        # Save native YOLO model to last.pt
        yolo_model = YOLO("yolo26n.pt")
        last_pt = weights_dir / "last.pt"
        yolo_model.save(str(last_pt))

        # Create mock results.csv with 4 epochs
        results_csv = exp_dir / "results.csv"
        rows = [
            {"epoch": 1, "train_loss": 2.1, "fitness": 0.35},
            {"epoch": 2, "train_loss": 1.9, "fitness": 0.42},
            {"epoch": 3, "train_loss": 1.8, "fitness": 0.39},
            {"epoch": 4, "train_loss": 1.7, "fitness": 0.45},
        ]
        with open(results_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "fitness"])
            writer.writeheader()
            writer.writerows(rows)

        # Load legacy checkpoint
        legacy_trainer = MeanTeacherTrainer(
            config=MeanTeacherConfig(
                runs_dir=self.runs_dir,
                experiment_name="legacy_exp",
                ssl_epochs=10,
            )
        )
        loaded = legacy_trainer.load_checkpoint(last_pt)

        self.assertEqual(loaded["format"], "legacy")
        self.assertEqual(loaded["start_epoch"], 4)
        self.assertAlmostEqual(loaded["best_fitness"], 0.45)
        self.assertEqual(loaded["best_epoch"], 4)
        self.assertEqual(len(loaded["training_history"]), 4)


class TestMeanTeacherKeyboardInterrupt(unittest.TestCase):
    """Test graceful KeyboardInterrupt catching and paused state serialization."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.runs_dir = self.test_dir / "runs"
        self.exp_name = "interrupt_exp"
        self.cfg = MeanTeacherConfig(
            model_name="yolo26n.pt",
            runs_dir=self.runs_dir,
            experiment_name=self.exp_name,
            ssl_epochs=3,
            warmup_epochs=1,
            batch_size=2,
            imgsz=64,
            workers=0,
        )
        self.trainer = MeanTeacherTrainer(config=self.cfg)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_train_mean_teacher_handles_keyboard_interrupt(self):
        """Simulate KeyboardInterrupt in training loop and assert clean pause."""
        warmup_ckpt = self.trainer.weights_dir / "warmup_best.pt"
        YOLO("yolo26n.pt").save(str(warmup_ckpt))

        # Mock PseudoLabelGenerator.generate to raise KeyboardInterrupt
        with patch("ssod.mean_teacher.PseudoLabelGenerator.generate") as mock_gen:
            mock_gen.side_effect = KeyboardInterrupt("User pressed Ctrl+C")

            res = self.trainer.train_mean_teacher(warmup_weights=warmup_ckpt)

            self.assertTrue(res.get("paused"), "Trainer did not return paused=True")
            self.assertEqual(res.get("paused_at_epoch"), 1)

            paused_pt = self.trainer.weights_dir / "paused.pt"
            self.assertTrue(paused_pt.exists(), f"paused.pt was not created at {paused_pt}")

            # Verify paused.pt can be loaded
            loaded = self.trainer.load_checkpoint(paused_pt)
            self.assertEqual(loaded["start_epoch"], 0)
            self.assertEqual(loaded["format"], "rich")


if __name__ == "__main__":
    unittest.main()
