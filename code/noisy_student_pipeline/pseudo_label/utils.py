"""
System, logging, and state management utilities for the SSL pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import psutil
import torch
import yaml


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load and parse a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: Any, path: Path | str) -> None:
    """Save data structure to a YAML file, creating parent directories if needed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def configure_system(cudnn_benchmark: bool = False) -> None:
    """
    Configure PyTorch and cuDNN environment.
    benchmark=False avoids runtime engine errors on some driver/GPU combos.
    """
    torch.backends.cudnn.benchmark = cudnn_benchmark


def setup_logging(
    log_file: Path | str | None = None,
    level: int = logging.INFO,
    logger_name: str = "pseudo_pipeline",
) -> logging.Logger:
    """Set up and configure console and file loggers."""
    log = logging.getLogger(logger_name)
    log.setLevel(level)

    # Avoid duplicate handlers if setup_logging is called multiple times
    if not log.handlers:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level)
        log.addHandler(stream_handler)

        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            log.addHandler(file_handler)

    # Ensure log messages don't propagate multiple times to root logger
    log.propagate = False
    return log


def log_environment_info(logger: logging.Logger | None = None) -> None:
    """Log current system RAM and GPU VRAM stats to help monitor resources."""
    log = logger or logging.getLogger("pseudo_pipeline")

    try:
        ram_total = psutil.virtual_memory().total / (1024**3)
        ram_avail = psutil.virtual_memory().available / (1024**3)
        ram_info = f"RAM: {ram_avail:.1f} GB available / {ram_total:.1f} GB total"
    except Exception:
        ram_info = "RAM: N/A"

    gpu_info = "GPU: None (CPU mode)"
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        alloc_vram = torch.cuda.memory_allocated(0) / (1024**3)
        res_vram = torch.cuda.memory_reserved(0) / (1024**3)
        gpu_info = (
            f"GPU: {name} | VRAM: {alloc_vram:.2f} GB allocated, "
            f"{res_vram:.2f} GB reserved / {total_vram:.1f} GB total"
        )
    log.info(f"Resource status: {gpu_info} | {ram_info}")


class RunStateTracker:
    """Tracks and persists experiment state and round-by-round metrics."""

    def __init__(self, run_dir: Path | str) -> None:
        self.run_dir = Path(run_dir)
        self.summary_file = self.run_dir / "summary.yaml"

    def load_summary(self) -> list[dict[str, Any]]:
        """Load summary history from summary.yaml if it exists."""
        if self.summary_file.exists():
            data = load_yaml(self.summary_file)
            if isinstance(data, list):
                return data
        return []

    def save_summary(self, summary: list[dict[str, Any]]) -> None:
        """Save summary history to summary.yaml."""
        save_yaml(summary, self.summary_file)

    def record_round(
        self,
        round_idx: int,
        weights: str,
        mAP50: float,
        mAP50_95: float,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Record or update metrics for a given round."""
        summary = self.load_summary()

        entry: dict[str, Any] = {
            "round": round_idx,
            "weights": str(weights),
            "mAP50": float(mAP50),
            "mAP50_95": float(mAP50_95),
        }
        if extra:
            entry.update(extra)

        # Replace existing entry if present, else append
        existing_idx = next(
            (i for i, item in enumerate(summary) if item.get("round") == round_idx),
            None,
        )
        if existing_idx is not None:
            summary[existing_idx] = entry
        else:
            summary.append(entry)

        # Sort by round index
        summary.sort(key=lambda item: item.get("round", 0))
        self.save_summary(summary)

    def get_last_completed_round(self) -> int | None:
        """
        Return the highest round index that successfully completed training
        and has both a valid checkpoint (best.pt) and a record in summary.yaml.
        """
        summary = self.load_summary()
        completed_rounds = []
        for item in summary:
            r = item.get("round")
            if r is not None:
                ckpt = self.get_checkpoint_for_round(r)
                if ckpt is not None and ckpt.exists():
                    completed_rounds.append(r)

        return max(completed_rounds) if completed_rounds else None

    def get_checkpoint_for_round(self, round_idx: int) -> Path | None:
        """Return path to best.pt for the given round, or None if not found."""
        ckpt = self.run_dir / f"round_{round_idx}_train" / "weights" / "best.pt"
        return ckpt if ckpt.exists() else None

    def get_last_checkpoint_for_round(self, round_idx: int) -> Path | None:
        """Return path to last.pt for the given round (for mid-round resume), or None."""
        ckpt = self.run_dir / f"round_{round_idx}_train" / "weights" / "last.pt"
        return ckpt if ckpt.exists() else None
