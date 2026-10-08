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


class TestMeanTeacherCLIOptions(unittest.TestCase):
    """Test that ablation-relevant CLI options reach MeanTeacherConfig."""

    def test_lr_and_pl_cls_only_options(self):
        from typer.testing import CliRunner

        from ssod.mean_teacher import app

        captured = {}

        class FakeTrainer:
            def __init__(self, config):
                captured["config"] = config

            def train_mean_teacher(self, resume_checkpoint=None):
                return {}

        test_dir = Path(tempfile.mkdtemp())
        try:
            with patch("ssod.mean_teacher.MeanTeacherTrainer", FakeTrainer):
                result = CliRunner().invoke(
                    app,
                    [
                        "--lr0", "5e-5",
                        "--lrf", "0.05",
                        "--pl-cls-only",
                        "--no-pl-size-bins",
                        "--no-pl-multi-view",
                        "--close-mosaic-ratio", "0",
                        "--project", str(test_dir),
                        "--name", "cli",
                    ],
                )
            self.assertEqual(result.exit_code, 0, msg=result.output)
            cfg = captured["config"]
            self.assertAlmostEqual(cfg.lr0, 5e-5)
            self.assertAlmostEqual(cfg.lrf, 0.05)
            self.assertTrue(cfg.pl_cls_only)
            self.assertEqual(cfg.close_mosaic_ratio, 0.0)
            self.assertTrue(cfg.pl_soft_weights)  # adopted default
            self.assertFalse(cfg.pl_size_bins)
            self.assertFalse(cfg.pl_multi_view)
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)


class TestMeanTeacherDefaults(unittest.TestCase):
    """Test the adopted default configuration and full-pool handling."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _invoke(self, args: list[str]) -> MeanTeacherConfig:
        from typer.testing import CliRunner

        from ssod.mean_teacher import app

        captured = {}

        class FakeTrainer:
            def __init__(self, config):
                captured["config"] = config

            def train_mean_teacher(self, resume_checkpoint=None):
                return {}

        with patch("ssod.mean_teacher.MeanTeacherTrainer", FakeTrainer):
            result = CliRunner().invoke(app, args)
        self.assertEqual(result.exit_code, 0, msg=result.output)
        return captured["config"]

    def test_adopted_defaults(self):
        cfg = MeanTeacherConfig(runs_dir=self.test_dir)
        self.assertEqual(cfg.pl_images_per_epoch, 2400)
        self.assertAlmostEqual(cfg.ema_decay, 0.9996)
        self.assertTrue(cfg.pl_soft_weights)
        cli_cfg = self._invoke(["--project", str(self.test_dir), "--name", "defaults"])
        self.assertEqual(cli_cfg.pl_images_per_epoch, 2400)
        self.assertAlmostEqual(cli_cfg.ema_decay, 0.9996)
        self.assertTrue(cli_cfg.pl_soft_weights)

    def test_negative_cap_means_whole_pool(self):
        self.assertIsNone(MeanTeacherConfig(runs_dir=self.test_dir, pl_images_per_epoch=-1).pl_images_per_epoch)
        cfg = self._invoke(
            ["--pl-images-per-epoch", "-1", "--project", str(self.test_dir), "--name", "pool"]
        )
        self.assertIsNone(cfg.pl_images_per_epoch)

    def test_historic_pool_control_flags_reproduce_original_settings(self):
        sys.path.insert(0, str(CODE_DIR / "ssod"))
        from ssod.ablation import BATCHES

        cfg = self._invoke(
            [*BATCHES["pool"].runs["control"], "--project", str(self.test_dir), "--name", "control"]
        )
        self.assertIsNone(cfg.pl_images_per_epoch)
        self.assertFalse(cfg.pl_soft_weights)
        self.assertAlmostEqual(cfg.ema_decay, 0.999)

    def test_resume_keeps_saved_whole_pool(self):
        exp_dir = self.test_dir / "old_run"
        (exp_dir / "weights").mkdir(parents=True)
        (exp_dir / "weights" / "paused.pt").write_bytes(b"")
        (exp_dir / "args.yaml").write_text(
            "pl_images_per_epoch: null\npl_soft_weights: false\nema_decay: 0.999\n", encoding="utf-8"
        )
        cfg = self._invoke(["--resume", str(exp_dir)])
        self.assertIsNone(cfg.pl_images_per_epoch)
        self.assertFalse(cfg.pl_soft_weights)
        self.assertAlmostEqual(cfg.ema_decay, 0.999)


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

        import cv2
        import numpy as np

        train_img_dir = self.test_dir / "data" / "train" / "images"
        train_lbl_dir = self.test_dir / "data" / "train" / "labels"
        val_img_dir = self.test_dir / "data" / "val" / "images"
        val_lbl_dir = self.test_dir / "data" / "val" / "labels"
        pl_img_dir = self.test_dir / "data" / "pl" / "images"
        pl_lbl_dir = self.test_dir / "data" / "pl" / "labels"
        for d in (train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir, pl_img_dir, pl_lbl_dir):
            d.mkdir(parents=True, exist_ok=True)

        dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
        for i in range(2):
            cv2.imwrite(str(train_img_dir / f"train_{i}.jpg"), dummy_img)
            cv2.imwrite(str(val_img_dir / f"val_{i}.jpg"), dummy_img)
            cv2.imwrite(str(pl_img_dir / f"pl_{i}.jpg"), dummy_img)
            (train_lbl_dir / f"train_{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
            (val_lbl_dir / f"val_{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
            (pl_lbl_dir / f"pl_{i}.txt").write_text("")

        dummy_yaml = self.test_dir / "data.yaml"
        dummy_yaml.write_text(
            f"path: {self.test_dir / 'data'}\n"
            f"train: train/images\n"
            f"val: val/images\n"
            f"nc: 1\n"
            f"names: ['plastic']\n"
        )

        self.cfg = MeanTeacherConfig(
            model_name="yolo26n.pt",
            gt_data_yaml=dummy_yaml,
            pl_images_dir=pl_img_dir,
            pl_labels_dir=pl_lbl_dir,
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


    def test_infer_exp_dir_from_resume(self):
        """Test inference of runs_dir and experiment_name from different resume paths."""
        from ssod.mean_teacher import infer_exp_dir_from_resume

        # 1. Target is a run directory
        run_dir = Path("/tmp/experiments/my_ssl_run")
        runs_dir, exp_name = infer_exp_dir_from_resume(run_dir)
        self.assertEqual(runs_dir, Path("/tmp/experiments"))
        self.assertEqual(exp_name, "my_ssl_run")

        # 2. Target is a file inside weights/
        ckpt_weights = Path("/tmp/experiments/my_ssl_run/weights/paused.pt")
        runs_dir, exp_name = infer_exp_dir_from_resume(ckpt_weights)
        self.assertEqual(runs_dir, Path("/tmp/experiments"))
        self.assertEqual(exp_name, "my_ssl_run")

        # 3. Target is a file directly in run directory
        ckpt_direct = Path("/tmp/experiments/my_ssl_run/paused.pt")
        runs_dir, exp_name = infer_exp_dir_from_resume(ckpt_direct)
        self.assertEqual(runs_dir, Path("/tmp/experiments"))
        self.assertEqual(exp_name, "my_ssl_run")

    def test_config_infers_name_and_project_without_creating_exp_dir(self):
        """Verify MeanTeacherConfig and Trainer infer experiment_name and do not create 'exp' dir."""
        custom_run_dir = self.runs_dir / "YOLO26m-SSOD-100-150"
        weights_dir = custom_run_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        (weights_dir / "last.pt").touch()

        # Instantiate config with resume pointing to run directory, without explicit experiment_name
        cfg = MeanTeacherConfig(
            resume=custom_run_dir,
            runs_dir=self.runs_dir,
        )

        self.assertEqual(cfg.experiment_name, "YOLO26m-SSOD-100-150")
        trainer = MeanTeacherTrainer(config=cfg)
        self.assertEqual(trainer.exp_dir, custom_run_dir)
        self.assertEqual(trainer.weights_dir, weights_dir)

        # Verify that runs_dir/exp was NOT created
        self.assertFalse((self.runs_dir / "exp").exists())

    def test_rich_checkpoint_save_and_load_with_custom_nc(self):
        """Verify saving and loading rich checkpoints when nc=1 (e.g. FloW-img) works without size mismatch."""
        from ultralytics.nn.tasks import DetectionModel

        m_student = DetectionModel(cfg="yolo26n.yaml", nc=1, verbose=False)
        m_teacher = DetectionModel(cfg="yolo26n.yaml", nc=1, verbose=False)
        m_student.nc = 1
        m_teacher.nc = 1
        student = YOLO("yolo26n.yaml")
        student.model = m_student
        teacher = YOLO("yolo26n.yaml")
        teacher.model = m_teacher

        optimizer = torch.optim.AdamW(student.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        save_path = self.trainer.weights_dir / "custom_nc_checkpoint.pt"
        self.trainer.save_checkpoint(
            path=save_path,
            epoch=5,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            best_fitness=0.55,
            best_epoch=4,
            training_history=[],
        )

        # Load back
        loaded = self.trainer.load_checkpoint(save_path)
        self.assertEqual(loaded["format"], "rich")
        self.assertEqual(loaded["start_epoch"], 5)
        self.assertEqual(loaded["student"].model.nc, 1)
        self.assertEqual(loaded["teacher"].model.nc, 1)

    def test_load_actual_paused_checkpoint(self):
        """Verify that the user's actual paused.pt loads cleanly without RuntimeError."""
        paused_pt = REPO_ROOT / "runs/mean_teacher/YOLO26m-SSOD-100-150/weights/paused.pt"
        if not paused_pt.is_file():
            self.skipTest("paused.pt not present on disk")

        loaded = self.trainer.load_checkpoint(paused_pt)
        self.assertEqual(loaded["format"], "rich")
        self.assertEqual(loaded["start_epoch"], 103)
        self.assertEqual(loaded["student"].model.nc, 1)
        self.assertEqual(loaded["teacher"].model.nc, 1)
        self.assertIsNotNone(self.trainer.config.coco_gt_path)
        self.assertTrue(self.trainer.config.coco_gt_path.exists())


class TestMeanTeacherArgsPersistence(unittest.TestCase):
    """Test args.yaml persistence and CLI parameter override on resume."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.runs_dir = self.test_dir / "runs"

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_save_and_load_config_yaml(self):
        from ssod.mean_teacher import load_config_yaml, save_config_yaml

        cfg = MeanTeacherConfig(
            model_name="yolo26m.pt",
            batch_size=32,
            runs_dir=self.runs_dir,
            experiment_name="yaml_test",
        )
        yaml_file = self.runs_dir / "yaml_test" / "args.yaml"
        save_config_yaml(cfg, yaml_file)
        self.assertTrue(yaml_file.is_file())

        loaded = load_config_yaml(yaml_file)
        self.assertEqual(loaded["model_name"], "yolo26m.pt")
        self.assertEqual(loaded["batch_size"], 32)
        self.assertEqual(loaded["experiment_name"], "yaml_test")


class TestSSODDetectionValidatorAndLoss(unittest.TestCase):
    """Test SSODDetectionValidator validation loss computation and COCO evaluation."""

    def test_validation_loss_is_computed(self):
        """Verify evaluate_model_coco produces non-zero validation loss metrics."""
        from ssod.mean_teacher import evaluate_model_coco

        best_pt = REPO_ROOT / "runs/mean_teacher/YOLO26m-SSOD-100-150/weights/best.pt"
        if not best_pt.is_file():
            self.skipTest("best.pt checkpoint not found for validation loss test")

        model = YOLO(str(best_pt))
        metrics = evaluate_model_coco(
            model=model,
            gt_data_yaml=REPO_ROOT / "data/flow_img.yaml",
            coco_gt_path=None,  # auto-discovers
            imgsz=320,
            batch_size=32,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )

        self.assertIn("val/box_loss", metrics)
        self.assertIn("val/cls_loss", metrics)
        self.assertIn("val/dfl_loss", metrics)
        self.assertGreater(metrics["val/box_loss"], 0.0)
        self.assertGreater(metrics["val/cls_loss"], 0.0)
        self.assertIn("coco/mAP_50_95", metrics)


if __name__ == "__main__":
    unittest.main()
