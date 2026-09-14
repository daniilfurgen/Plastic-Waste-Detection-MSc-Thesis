"""
Configuration dataclasses and utilities for the Semi-Supervised Learning (SSL) pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PipelineConfig:
    """Configuration for the iterative pseudo-labeling pipeline."""

    # Run metadata & directories
    experiment_name: str = "YOLO26m-P2-SSL w/ pretrained weights"
    root_dir: Path = field(default_factory=lambda: Path.cwd().resolve())

    # Dataset paths (relative to root_dir unless absolute)
    orig_data_yaml: Path = field(default_factory=lambda: Path("data/flow_img.yaml"))
    video_images_dir: Path = field(
        default_factory=lambda: Path("data/video_frames/images/all")
    )
    video_labels_dir: Path = field(
        default_factory=lambda: Path("data/video_frames/labels/all")
    )

    # Output run directories
    runs_dir: Path = field(default_factory=lambda: Path("runs"))
    pseudo_run_dir: Path = field(
        default_factory=lambda: Path("runs/pseudo/yolo26m-p2-ssl")
    )
    tune_run_dir: Path = field(default_factory=lambda: Path("runs/tune"))
    log_file: Path = field(default_factory=lambda: Path("pipeline.log"))

    # Model architecture & initialization
    model_size: str = "m"  # "n", "s", "m", "l", "x"
    p2_head: bool = True  # use P2 small-object detection head
    base_configuration: str | None = None  # overrides f"yolo26{model_size}-p2.yaml"
    pretrained_weights: str | None = None  # overrides f"yolo26{model_size}.pt"

    # Training hyperparameters
    batch_size: int = 10  # Optimal batch size benchmarked for RTX 4070 (12GB)
    workers: int = 4  # DataLoader workers
    n_rounds: int = 3  # Total number of pseudo-labeling rounds (rounds 0, 1, 2)
    epochs_per_round: int = 150  # Max epochs per training round
    patience: int = 25  # Early stopping patience
    imgsz: int = 640  # Input image resolution
    device: str | int | None = None  # Device selection (e.g. 0, "cuda:0", "cpu")

    # Pseudo-labeling parameters
    conf_schedule: list[float] = field(default_factory=lambda: [0.85, 0.80, 0.75])
    max_pseudo_ratio: float = 1.0  # Max ratio of pseudo-labeled images to real images
    seed: int = 42

    # Evaluation parameters
    coco_gt_path: Path | None = field(
        default_factory=lambda: Path("data/flow_img/val/annotations.json")
    )
    enable_coco_eval: bool = (
        True  # Compute standard COCO metrics (mAP_s, mAP_m, mAP_l, etc.)
    )

    # Hyperparameter tuning parameters
    tune_fraction: float = 0.10  # Fraction of train images used for tuning
    tune_seed: int = 42
    tune_epochs: int = 30
    tune_iterations: int = 10

    # Extra hyperparameter overrides (e.g. lr0, mosaic, etc.)
    hyp_overrides: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Resolve all paths relative to root_dir if not already absolute."""
        self.root_dir = Path(self.root_dir).resolve()
        self.orig_data_yaml = self._resolve(self.orig_data_yaml)
        self.video_images_dir = self._resolve(self.video_images_dir)
        self.video_labels_dir = self._resolve(self.video_labels_dir)
        self.runs_dir = self._resolve(self.runs_dir)
        self.pseudo_run_dir = self._resolve(self.pseudo_run_dir)
        self.tune_run_dir = self._resolve(self.tune_run_dir)
        self.log_file = self._resolve(self.log_file)
        if self.coco_gt_path is not None:
            self.coco_gt_path = self._resolve(self.coco_gt_path)

        if not self.conf_schedule:
            self.conf_schedule = [0.85]

    def _resolve(self, path: Path | str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else (self.root_dir / p).resolve()

    def get_base_configuration(self) -> str:
        """Return the base YAML model architecture configuration."""
        if self.base_configuration:
            return self.base_configuration
        return (
            f"yolo26{self.model_size}-p2.yaml"
            if self.p2_head
            else f"yolo26{self.model_size}.yaml"
        )

    def get_pretrained_weights(self) -> str:
        """Return the initial pre-trained checkpoint filename or path."""
        if self.pretrained_weights:
            return self.pretrained_weights
        return f"yolo26{self.model_size}.pt"

    def get_conf_for_round(self, round_idx: int) -> float:
        """
        Safely return confidence threshold for the given round index.
        If round_idx exceeds the length of conf_schedule, clamps to the last value.
        """
        if round_idx < len(self.conf_schedule):
            return self.conf_schedule[round_idx]
        return self.conf_schedule[-1]

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a dictionary with serializable path strings."""
        data = asdict(self)
        for key, val in data.items():
            if isinstance(val, Path):
                data[key] = str(val)
        return data

    def save_yaml(self, path: Path | str) -> None:
        """Save configuration dictionary to a YAML file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: Path | str) -> PipelineConfig:
        """Load configuration from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}

        # Convert string paths back to Path objects
        path_keys = {
            "root_dir",
            "orig_data_yaml",
            "video_images_dir",
            "video_labels_dir",
            "runs_dir",
            "pseudo_run_dir",
            "tune_run_dir",
            "log_file",
            "coco_gt_path",
        }
        for k in path_keys:
            if k in raw_data and raw_data[k] is not None:
                raw_data[k] = Path(raw_data[k])

        return cls(**raw_data)
