"""
Automated unit and integration tests for Mean Teacher pause, resume, and CLI latency.
"""

from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import pytest
import torch
from typer.testing import CliRunner
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

from ssod.mean_teacher import (
    MeanTeacherConfig,
    MeanTeacherTrainer,
    app,
    evaluate_model_coco,
    infer_exp_dir_from_resume,
    load_config_yaml,
    save_config_yaml,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = REPO_ROOT / "code"


@pytest.fixture
def runs_dir(tmp_path) -> Path:
    return tmp_path / "runs"


@pytest.fixture
def invoke_cli(monkeypatch):
    """Run the Typer CLI with a fake trainer and return the MeanTeacherConfig it received."""

    def _invoke(args: list[str]) -> MeanTeacherConfig:
        captured = {}

        class FakeTrainer:
            def __init__(self, config):
                captured["config"] = config

            def train_mean_teacher(self, resume_checkpoint=None):
                return {}

        monkeypatch.setattr("ssod.mean_teacher.MeanTeacherTrainer", FakeTrainer)
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 0, result.output
        return captured["config"]

    return _invoke


@pytest.fixture
def trainer(yolo26n_path, runs_dir) -> MeanTeacherTrainer:
    cfg = MeanTeacherConfig(
        model_name=yolo26n_path,
        runs_dir=runs_dir,
        experiment_name="test_exp",
        ssl_epochs=10,
    )
    return MeanTeacherTrainer(config=cfg)


@pytest.mark.slow
def test_cli_help_latency_and_options():
    """Verify calling --help runs in < 1.5s and documents --resume."""
    cmd = [sys.executable, str(CODE_DIR / "ssod" / "mean_teacher.py"), "--help"]

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - t0

    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    assert elapsed < 1.5, f"--help took {elapsed:.2f}s, exceeding target threshold of 1.5s"
    assert "--resume" in result.stdout
    assert "-r" in result.stdout


def test_lr_and_pl_cls_only_options(invoke_cli, tmp_path):
    """Ablation-relevant CLI options reach MeanTeacherConfig."""
    cfg = invoke_cli([
        "--lr0", "5e-5",
        "--lrf", "0.05",
        "--pl-cls-only",
        "--no-pl-size-bins",
        "--no-pl-multi-view",
        "--seed", "7",
        "--pl-teacher-refresh-epochs", "3",
        "--close-mosaic-ratio", "0",
        "--project", str(tmp_path),
        "--name", "cli",
    ])
    assert cfg.lr0 == pytest.approx(5e-5)
    assert cfg.lrf == pytest.approx(0.05)
    assert cfg.pl_cls_only
    assert cfg.close_mosaic_ratio == 0.0
    assert cfg.pl_soft_weights  # adopted default
    assert not cfg.pl_size_bins
    assert not cfg.pl_multi_view
    assert cfg.seed == 7
    assert cfg.pl_teacher_refresh_epochs == 3


class TestMeanTeacherDefaults:
    """Test the adopted default configuration and full-pool handling."""

    def test_adopted_defaults(self, invoke_cli, tmp_path):
        cfg = MeanTeacherConfig(runs_dir=tmp_path)
        assert cfg.pl_images_per_epoch == 2400
        assert cfg.ema_decay == pytest.approx(0.9996)
        assert cfg.pl_soft_weights
        cli_cfg = invoke_cli(["--project", str(tmp_path), "--name", "defaults"])
        assert cli_cfg.pl_images_per_epoch == 2400
        assert cli_cfg.ema_decay == pytest.approx(0.9996)
        assert cli_cfg.pl_soft_weights

    def test_negative_cap_means_whole_pool(self, invoke_cli, tmp_path):
        assert MeanTeacherConfig(runs_dir=tmp_path, pl_images_per_epoch=-1).pl_images_per_epoch is None
        cfg = invoke_cli(["--pl-images-per-epoch", "-1", "--project", str(tmp_path), "--name", "pool"])
        assert cfg.pl_images_per_epoch is None

    def test_historic_pool_control_flags_reproduce_original_settings(self, invoke_cli, tmp_path):
        from ssod.ablation import BATCHES

        cfg = invoke_cli([*BATCHES["pool"].runs["control"], "--project", str(tmp_path), "--name", "control"])
        assert cfg.pl_images_per_epoch is None
        assert not cfg.pl_soft_weights
        assert cfg.ema_decay == pytest.approx(0.999)

    def test_resume_keeps_saved_whole_pool(self, invoke_cli, tmp_path):
        exp_dir = tmp_path / "old_run"
        (exp_dir / "weights").mkdir(parents=True)
        (exp_dir / "weights" / "paused.pt").write_bytes(b"")
        (exp_dir / "args.yaml").write_text(
            "pl_images_per_epoch: null\npl_soft_weights: false\nema_decay: 0.999\n", encoding="utf-8"
        )
        cfg = invoke_cli(["--resume", str(exp_dir)])
        assert cfg.pl_images_per_epoch is None
        assert not cfg.pl_soft_weights
        assert cfg.ema_decay == pytest.approx(0.999)


class TestMeanTeacherCheckpointSerialization:
    """Test rich and legacy checkpoint serialization and restoration."""

    def test_checkpoint_path_resolution(self, trainer, tmp_path, runs_dir):
        """Test resolve_checkpoint_path on files and directories."""
        exp_dir = runs_dir / "test_exp"
        weights_dir = exp_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)

        # Non-existent file and directory without checkpoints raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            MeanTeacherTrainer.resolve_checkpoint_path(tmp_path / "non_existent.pt")
        with pytest.raises(FileNotFoundError):
            MeanTeacherTrainer.resolve_checkpoint_path(tmp_path)

        # last.pt < last_checkpoint.pt < paused.pt in priority
        for name in ("last.pt", "last_checkpoint.pt", "paused.pt"):
            expected = weights_dir / name
            expected.touch()
            assert MeanTeacherTrainer.resolve_checkpoint_path(exp_dir) == expected

    def test_rich_checkpoint_save_and_load(self, trainer, yolo26n_path):
        """Test full fidelity of saving and loading rich checkpoints."""
        teacher = YOLO(yolo26n_path)
        student = YOLO(yolo26n_path)
        optimizer = torch.optim.AdamW(student.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        # Mutate weights slightly and step optimizer
        for p in student.model.parameters():
            if p.requires_grad:
                p.data.add_(0.01)
                break
        optimizer.step()
        scheduler.step()

        save_path = trainer.weights_dir / "test_checkpoint.pt"
        trainer.save_checkpoint(
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
        assert save_path.exists()

        loaded = trainer.load_checkpoint(save_path)
        assert loaded["format"] == "rich"
        assert loaded["start_epoch"] == 3
        assert loaded["best_fitness"] == pytest.approx(0.456)
        assert loaded["best_epoch"] == 2
        assert len(loaded["training_history"]) == 2
        assert loaded["optimizer_state_dict"] is not None
        assert loaded["scheduler_state_dict"] is not None

        # Verify model parameter fidelity
        orig_sd = student.model.state_dict()
        loaded_sd = loaded["student"].model.state_dict()
        for k in orig_sd:
            assert torch.allclose(orig_sd[k].cpu(), loaded_sd[k].cpu()), k

    def test_legacy_checkpoint_load_with_results_csv(self, yolo26n_path, runs_dir):
        """Test fallback loading from Ultralytics last.pt and results.csv."""
        exp_dir = runs_dir / "legacy_exp"
        weights_dir = exp_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)

        last_pt = weights_dir / "last.pt"
        YOLO(yolo26n_path).save(str(last_pt))

        rows = [
            {"epoch": 1, "train_loss": 2.1, "fitness": 0.35},
            {"epoch": 2, "train_loss": 1.9, "fitness": 0.42},
            {"epoch": 3, "train_loss": 1.8, "fitness": 0.39},
            {"epoch": 4, "train_loss": 1.7, "fitness": 0.45},
        ]
        with open(exp_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "fitness"])
            writer.writeheader()
            writer.writerows(rows)

        legacy_trainer = MeanTeacherTrainer(
            config=MeanTeacherConfig(runs_dir=runs_dir, experiment_name="legacy_exp", ssl_epochs=10)
        )
        loaded = legacy_trainer.load_checkpoint(last_pt)

        assert loaded["format"] == "legacy"
        assert loaded["start_epoch"] == 4
        assert loaded["best_fitness"] == pytest.approx(0.45)
        assert loaded["best_epoch"] == 4
        assert len(loaded["training_history"]) == 4

    def test_rich_checkpoint_save_and_load_with_custom_nc(self, trainer):
        """Saving and loading rich checkpoints with nc=1 (e.g. FloW-img) works without size mismatch."""
        student = YOLO("yolo26n.yaml")
        teacher = YOLO("yolo26n.yaml")
        student.model = DetectionModel(cfg="yolo26n.yaml", nc=1, verbose=False)
        teacher.model = DetectionModel(cfg="yolo26n.yaml", nc=1, verbose=False)
        student.model.nc = 1
        teacher.model.nc = 1

        optimizer = torch.optim.AdamW(student.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        save_path = trainer.weights_dir / "custom_nc_checkpoint.pt"
        trainer.save_checkpoint(
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

        loaded = trainer.load_checkpoint(save_path)
        assert loaded["format"] == "rich"
        assert loaded["start_epoch"] == 5
        assert loaded["student"].model.nc == 1
        assert loaded["teacher"].model.nc == 1

    @pytest.mark.needs_artifacts
    def test_load_actual_paused_checkpoint(self, trainer):
        """Verify that the user's actual paused.pt loads cleanly without RuntimeError."""
        paused_pt = REPO_ROOT / "runs/mean_teacher/YOLO26m-SSOD-100-150/weights/paused.pt"
        if not paused_pt.is_file():
            pytest.skip("paused.pt not present on disk")

        loaded = trainer.load_checkpoint(paused_pt)
        assert loaded["format"] == "rich"
        assert loaded["start_epoch"] == 103
        assert loaded["student"].model.nc == 1
        assert loaded["teacher"].model.nc == 1
        assert trainer.config.coco_gt_path is not None
        assert trainer.config.coco_gt_path.exists()


class TestMeanTeacherKeyboardInterrupt:
    """Test graceful KeyboardInterrupt catching and paused state serialization."""

    @pytest.fixture
    def interrupt_trainer(self, tmp_path, runs_dir, yolo26n_path) -> MeanTeacherTrainer:
        data = tmp_path / "data"
        train_img, train_lbl = data / "train" / "images", data / "train" / "labels"
        val_img, val_lbl = data / "val" / "images", data / "val" / "labels"
        pl_img, pl_lbl = data / "pl" / "images", data / "pl" / "labels"
        for d in (train_img, train_lbl, val_img, val_lbl, pl_img, pl_lbl):
            d.mkdir(parents=True)

        dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
        for i in range(2):
            cv2.imwrite(str(train_img / f"train_{i}.jpg"), dummy_img)
            cv2.imwrite(str(val_img / f"val_{i}.jpg"), dummy_img)
            cv2.imwrite(str(pl_img / f"pl_{i}.jpg"), dummy_img)
            (train_lbl / f"train_{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
            (val_lbl / f"val_{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
            (pl_lbl / f"pl_{i}.txt").write_text("")

        dummy_yaml = tmp_path / "data.yaml"
        dummy_yaml.write_text(
            f"path: {data}\ntrain: train/images\nval: val/images\nnc: 1\nnames: ['plastic']\n"
        )

        return MeanTeacherTrainer(
            config=MeanTeacherConfig(
                model_name=yolo26n_path,
                gt_data_yaml=dummy_yaml,
                pl_images_dir=pl_img,
                pl_labels_dir=pl_lbl,
                runs_dir=runs_dir,
                experiment_name="interrupt_exp",
                ssl_epochs=3,
                warmup_epochs=1,
                batch_size=2,
                imgsz=64,
                workers=0,
            )
        )

    @pytest.mark.slow
    def test_train_mean_teacher_handles_keyboard_interrupt(
        self, interrupt_trainer, yolo26n_path, monkeypatch
    ):
        """Simulate KeyboardInterrupt in training loop and assert clean pause."""
        warmup_ckpt = interrupt_trainer.weights_dir / "warmup_best.pt"
        YOLO(yolo26n_path).save(str(warmup_ckpt))

        def interrupt(*args, **kwargs):
            raise KeyboardInterrupt("User pressed Ctrl+C")

        monkeypatch.setattr("ssod.mean_teacher.PseudoLabelGenerator.generate", interrupt)

        res = interrupt_trainer.train_mean_teacher(warmup_weights=warmup_ckpt)

        assert res.get("paused"), "Trainer did not return paused=True"
        assert res.get("paused_at_epoch") == 1

        paused_pt = interrupt_trainer.weights_dir / "paused.pt"
        assert paused_pt.exists(), f"paused.pt was not created at {paused_pt}"

        loaded = interrupt_trainer.load_checkpoint(paused_pt)
        assert loaded["start_epoch"] == 0
        assert loaded["format"] == "rich"


@pytest.mark.parametrize(
    "target",
    [
        "/tmp/experiments/my_ssl_run",  # a run directory
        "/tmp/experiments/my_ssl_run/weights/paused.pt",  # a file inside weights/
        "/tmp/experiments/my_ssl_run/paused.pt",  # a file directly in the run directory
    ],
    ids=["run-dir", "weights-file", "run-dir-file"],
)
def test_infer_exp_dir_from_resume(target):
    """Test inference of runs_dir and experiment_name from different resume paths."""
    runs_dir, exp_name = infer_exp_dir_from_resume(Path(target))
    assert runs_dir == Path("/tmp/experiments")
    assert exp_name == "my_ssl_run"


def test_config_infers_name_and_project_without_creating_exp_dir(runs_dir):
    """MeanTeacherConfig and Trainer infer experiment_name and do not create an 'exp' dir."""
    custom_run_dir = runs_dir / "YOLO26m-SSOD-100-150"
    weights_dir = custom_run_dir / "weights"
    weights_dir.mkdir(parents=True)
    (weights_dir / "last.pt").touch()

    # Resume points to a run directory, without explicit experiment_name
    cfg = MeanTeacherConfig(resume=custom_run_dir, runs_dir=runs_dir)

    assert cfg.experiment_name == "YOLO26m-SSOD-100-150"
    trainer = MeanTeacherTrainer(config=cfg)
    assert trainer.exp_dir == custom_run_dir
    assert trainer.weights_dir == weights_dir
    assert not (runs_dir / "exp").exists()


def test_save_and_load_config_yaml(runs_dir):
    """args.yaml persistence round-trips the config."""
    cfg = MeanTeacherConfig(
        model_name="yolo26m.pt",
        batch_size=32,
        runs_dir=runs_dir,
        experiment_name="yaml_test",
    )
    yaml_file = runs_dir / "yaml_test" / "args.yaml"
    save_config_yaml(cfg, yaml_file)
    assert yaml_file.is_file()

    loaded = load_config_yaml(yaml_file)
    assert loaded["model_name"] == "yolo26m.pt"
    assert loaded["batch_size"] == 32
    assert loaded["experiment_name"] == "yaml_test"


@pytest.mark.needs_artifacts
def test_validation_loss_is_computed(device):
    """Verify evaluate_model_coco produces non-zero validation loss metrics."""
    best_pt = REPO_ROOT / "runs/mean_teacher/YOLO26m-SSOD-100-150/weights/best.pt"
    if not best_pt.is_file():
        pytest.skip("best.pt checkpoint not found for validation loss test")

    metrics = evaluate_model_coco(
        model=YOLO(str(best_pt)),
        gt_data_yaml=REPO_ROOT / "data/flow_img.yaml",
        coco_gt_path=None,  # auto-discovers
        imgsz=320,
        batch_size=32,
        device=device,
    )

    for key in ("val/box_loss", "val/cls_loss", "val/dfl_loss", "coco/mAP_50_95"):
        assert key in metrics
    assert metrics["val/box_loss"] > 0.0
    assert metrics["val/cls_loss"] > 0.0
