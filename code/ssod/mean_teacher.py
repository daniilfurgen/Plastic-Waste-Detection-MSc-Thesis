"""
Mean Teacher Semi-Supervised Object Detection (SSOD) training loop for YOLO.

Features:
1. Supervised Warm-up Phase:
   - Trains Teacher on Ground-Truth data only for n-epochs (or loads from --warmup-weights).
   - Initializes Student model from trained Teacher weights (theta_S <- theta_T).
2. Mean Teacher Training Loop:
   - Starts each epoch by generating pseudo-labels on unlabeled pool using PseudoLabelGenerator.
   - Synchronizes SSODDatasetHandler with 25% GT tail / 75% GT+PL head sequencing and 10% mosaic closing.
   - Trains Student with YOLO default detection loss (with optional unsupervised weight lambda_u).
   - Updates Teacher EMA at every training step.
   - Evaluates Teacher on GT validation split every epoch:
     extracts all Ultralytics default metrics and pycocotools COCO metrics.
   - Selects and saves best.pt checkpoint based on Ultralytics fitness.
3. Pause & Resume Lifecycle:
   - Catches KeyboardInterrupt gracefully, saving clean uncorrupted weights/paused.pt.
   - Saves full training state checkpoint (weights/last_checkpoint.pt) every epoch alongside native YOLO weights/last.pt.
   - Supports CLI --resume / -r from paused checkpoint, last_checkpoint.pt, or legacy last.pt + results.csv.
"""

from __future__ import annotations

from copy import deepcopy
import csv
from dataclasses import asdict, dataclass, field
import logging
from pathlib import Path
import random
import re
import sys
import time
from typing import TYPE_CHECKING, Any, Optional

import typer

try:
    from ssod.utilities import get_logger
except ImportError:
    try:
        from utilities import get_logger  # type: ignore[import-not-found,no-redef]
    except ImportError:
        REPO_CODE_DIR = Path(__file__).resolve().parent.parent
        if str(REPO_CODE_DIR) not in sys.path:
            sys.path.insert(0, str(REPO_CODE_DIR))
        from ssod.utilities import get_logger  # type: ignore[import-not-found,no-redef]

if TYPE_CHECKING:
    import numpy as np
    import torch
    import torch.nn as nn
    from tqdm import tqdm
    from ultralytics import YOLO
    from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
    from ssod.dataset import SSODAugmentConfig, SSODDatasetHandler
    from ssod.metrics import evaluate_coco
    from ssod.pseudo_labeling import PseudoLabelGenerator

# Runtime placeholders populated on-demand by _load_heavy_dependencies()
torch: Any = None
nn: Any = None
np: Any = None
tqdm: Any = None
YOLO: Any = None
DEFAULT_CFG_DICT: Any = None
IterableSimpleNamespace: Any = None
SSODAugmentConfig: Any = None
SSODDatasetHandler: Any = None
evaluate_coco: Any = None
PseudoLabelGenerator: Any = None


def _load_heavy_dependencies() -> None:
    """Lazily load heavy scientific and CV libraries to ensure instantaneous CLI --help."""
    global torch, nn, np, tqdm, YOLO, DEFAULT_CFG_DICT, IterableSimpleNamespace
    global SSODAugmentConfig, SSODDatasetHandler, evaluate_coco, PseudoLabelGenerator

    if torch is not None:
        return

    import numpy as _np
    import torch as _torch
    import torch.nn as _nn
    from tqdm import tqdm as _tqdm  # type: ignore[import-untyped]
    from ultralytics import YOLO as _YOLO
    from ultralytics.utils import (
        DEFAULT_CFG_DICT as _DEFAULT_CFG_DICT,
        IterableSimpleNamespace as _IterableSimpleNamespace,
    )

    try:
        from ssod.dataset import (
            SSODAugmentConfig as _SSODAugmentConfig,
            SSODDatasetHandler as _SSODDatasetHandler,
        )
        from ssod.metrics import evaluate_coco as _evaluate_coco
        from ssod.pseudo_labeling import PseudoLabelGenerator as _PseudoLabelGenerator
    except ImportError:
        from dataset import (  # type: ignore[import-not-found,no-redef]
            SSODAugmentConfig as _SSODAugmentConfig,
            SSODDatasetHandler as _SSODDatasetHandler,
        )
        from metrics import evaluate_coco as _evaluate_coco  # type: ignore[import-not-found,no-redef]
        from pseudo_labeling import PseudoLabelGenerator as _PseudoLabelGenerator  # type: ignore[import-not-found,no-redef]

    torch = _torch
    nn = _nn
    np = _np
    tqdm = _tqdm
    YOLO = _YOLO
    DEFAULT_CFG_DICT = _DEFAULT_CFG_DICT
    IterableSimpleNamespace = _IterableSimpleNamespace
    SSODAugmentConfig = _SSODAugmentConfig
    SSODDatasetHandler = _SSODDatasetHandler
    evaluate_coco = _evaluate_coco
    PseudoLabelGenerator = _PseudoLabelGenerator


def __getattr__(name: str) -> Any:
    """Module-level attribute lookup triggering on-demand dependency loading."""
    _load_heavy_dependencies()
    if name in globals() and globals()[name] is not None:
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


app = typer.Typer(
    name="mean-teacher",
    help="Mean Teacher Semi-Supervised Training for YOLO.",
    add_completion=False,
)


def infer_exp_dir_from_resume(resume_path: Path | str) -> tuple[Path, str]:
    """
    Infer (runs_dir, experiment_name) from a checkpoint file or directory path.

    Examples:
      - 'runs/mean_teacher/YOLO26m-SSOD-100-150' -> (Path('runs/mean_teacher'), 'YOLO26m-SSOD-100-150')
      - 'runs/mean_teacher/YOLO26m-SSOD-100-150/weights/last.pt' -> (Path('runs/mean_teacher'), 'YOLO26m-SSOD-100-150')
      - 'runs/mean_teacher/YOLO26m-SSOD-100-150/paused.pt' -> (Path('runs/mean_teacher'), 'YOLO26m-SSOD-100-150')
    """
    target = Path(resume_path).resolve()
    if target.suffix in (".pt", ".pth") or target.is_file():
        parent = target.parent
        if parent.name == "weights":
            return parent.parent.parent, parent.parent.name
        return parent.parent, parent.name
    return target.parent, target.name


@dataclass
class MeanTeacherConfig:
    """Configuration for Mean Teacher training."""

    # Models & weights
    model_name: str = "yolo26n.pt"
    warmup_weights: Path | None = None
    resume: Path | None = None

    # Datasets
    gt_data_yaml: Path = field(default_factory=lambda: Path("data/flow_img.yaml"))
    pl_images_dir: Path = field(
        default_factory=lambda: Path("data/video_frames/images/all")
    )
    pl_labels_dir: Path = field(
        default_factory=lambda: Path("data/video_frames/labels/all")
    )
    coco_gt_path: Path | None = field(
        default_factory=lambda: Path("data/flow_img/val/annotations.json")
    )

    # Schedules & dimensions
    warmup_epochs: int = 10
    ssl_epochs: int = 50
    batch_size: int = 16
    imgsz: int = 640
    workers: int = 4
    device: str | int | None = None

    # Optimizer & Scheduler
    lr0: float = 0.0002
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    amp: bool = True

    # EMA & SSL policies
    ema_decay: float = 0.999
    tail_ratio: float = 0.25
    close_mosaic_ratio: float = 0.10
    unsupervised_weight: float = 1.0

    # Pseudo-labeling parameters
    pseudo_conf: float = 0.001
    pseudo_iou: float = 0.5
    pseudo_half: bool = True
    max_pseudo_images: int | None = None
    save_candidates: bool = False
    candidates_dir_name: str = "candidates"

    # Logging and output
    runs_dir: Path = field(default_factory=lambda: Path("runs/mean_teacher"))
    experiment_name: str = "exp"
    seed: int = 42

    def __post_init__(self) -> None:
        self.gt_data_yaml = Path(self.gt_data_yaml).resolve()
        self.pl_images_dir = Path(self.pl_images_dir).resolve()
        self.pl_labels_dir = Path(self.pl_labels_dir).resolve()
        if self.coco_gt_path is not None:
            self.coco_gt_path = Path(self.coco_gt_path).resolve()
        if self.warmup_weights is not None:
            self.warmup_weights = Path(self.warmup_weights).resolve()
        if self.resume is not None:
            self.resume = Path(self.resume).resolve()
            # If experiment_name was left at default "exp", infer from resume path
            if self.experiment_name == "exp":
                inferred_runs_dir, inferred_exp_name = infer_exp_dir_from_resume(
                    self.resume
                )
                self.experiment_name = inferred_exp_name
                # If runs_dir was left at default "runs/mean_teacher", infer too
                if Path(self.runs_dir).resolve() == Path("runs/mean_teacher").resolve():
                    self.runs_dir = inferred_runs_dir
        self.runs_dir = Path(self.runs_dir).resolve()


class VectorizedEMA:
    """
    High-throughput Exponential Moving Average (EMA) for Teacher-Student models.
    Pre-pairs parameters and buffers once at initialization to eliminate
    the state_dict() dictionary overhead and utilizes vectorized torch._foreach operations.
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        decay: float = 0.999,
    ) -> None:
        _load_heavy_dependencies()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.decay = decay

        self.fp_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.non_fp_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

        s_params = dict(student_model.named_parameters())
        for name, t_p in teacher_model.named_parameters():
            if name in s_params:
                s_p = s_params[name]
                if t_p.dtype.is_floating_point:
                    self.fp_pairs.append((t_p, s_p))
                else:
                    self.non_fp_pairs.append((t_p, s_p))

        s_buffers = dict(student_model.named_buffers())
        for name, t_b in teacher_model.named_buffers():
            if name in s_buffers:
                s_b = s_buffers[name]
                if t_b.dtype.is_floating_point:
                    self.fp_pairs.append((t_b, s_b))
                else:
                    self.non_fp_pairs.append((t_b, s_b))

        self.t_fp_tensors = [p[0] for p in self.fp_pairs]
        self.s_fp_tensors = [p[1] for p in self.fp_pairs]

    def update(self) -> None:
        if not self.fp_pairs:
            return
        with torch.no_grad():
            torch._foreach_mul_(self.t_fp_tensors, self.decay)
            torch._foreach_add_(
                self.t_fp_tensors, self.s_fp_tensors, alpha=1.0 - self.decay
            )
            for t_b, s_b in self.non_fp_pairs:
                t_b.copy_(s_b)


def update_teacher_ema(
    teacher_model: nn.Module,
    student_model: nn.Module,
    decay: float = 0.999,
) -> None:
    """
    Update teacher model parameters via Exponential Moving Average at every step:
    theta_T = decay * theta_T + (1 - decay) * theta_S.
    Non-floating-point buffers (e.g. num_batches_tracked) are copied directly.
    """
    _load_heavy_dependencies()
    with torch.no_grad():
        t_fp = []
        s_fp = []
        s_params = dict(student_model.named_parameters())
        s_buffers = dict(student_model.named_buffers())

        for name, t_p in teacher_model.named_parameters():
            if name in s_params:
                s_p = s_params[name]
                if t_p.dtype.is_floating_point:
                    t_fp.append(t_p)
                    s_fp.append(s_p.detach().to(t_p.device))
                else:
                    t_p.copy_(s_p.detach())

        for name, t_b in teacher_model.named_buffers():
            if name in s_buffers:
                s_b = s_buffers[name]
                if t_b.dtype.is_floating_point:
                    t_fp.append(t_b)
                    s_fp.append(s_b.detach().to(t_b.device))
                else:
                    t_b.copy_(s_b.detach())

        if t_fp:
            torch._foreach_mul_(t_fp, decay)
            torch._foreach_add_(t_fp, s_fp, alpha=1.0 - decay)


def split_batch(batch: dict[str, Any], mask: torch.Tensor) -> dict[str, Any]:
    """Slice a collated batch dictionary to only include samples matching boolean mask."""
    _load_heavy_dependencies()
    indices = torch.where(mask)[0]
    new_batch: dict[str, Any] = {"img": batch["img"][mask]}

    if "bboxes" in batch and len(batch["bboxes"]) > 0:
        b_mask = torch.isin(batch["batch_idx"], indices.float())
        new_batch["bboxes"] = batch["bboxes"][b_mask]
        new_batch["cls"] = batch["cls"][b_mask]
        idx_map = {idx.item(): i for i, idx in enumerate(indices)}
        remapped = [idx_map[int(x.item())] for x in batch["batch_idx"][b_mask]]
        new_batch["batch_idx"] = torch.tensor(
            remapped, dtype=torch.float32, device=batch["batch_idx"].device
        )
    else:
        new_batch["bboxes"] = torch.empty((0, 4), device=batch["img"].device)
        new_batch["cls"] = torch.empty((0, 1), device=batch["img"].device)
        new_batch["batch_idx"] = torch.empty((0,), device=batch["img"].device)

    return new_batch


def compute_batch_loss(
    model: nn.Module,
    batch: dict[str, Any],
    unsupervised_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute YOLO default detection loss on batch.
    If unsupervised_weight != 1.0 and both GT and Pseudo items are present,
    applies unsupervised_weight to pseudo-labeled loss.
    Returns (loss_tensor, loss_items_detach).
    """
    _load_heavy_dependencies()
    bs = batch["img"].shape[0]
    is_pseudo = batch.get("is_pseudo")

    if (
        unsupervised_weight == 1.0
        or is_pseudo is None
        or not torch.is_tensor(is_pseudo)
    ):
        loss_tuple = model.loss(batch)
        loss_sum = loss_tuple[0].sum() / max(1, bs)
        loss_items = loss_tuple[1]
        return loss_sum, loss_items

    has_gt = (~is_pseudo).any()
    has_pl = is_pseudo.any()

    if has_gt and has_pl:
        b_gt = split_batch(batch, ~is_pseudo)
        b_pl = split_batch(batch, is_pseudo)
        n_gt = b_gt["img"].shape[0]
        n_pl = b_pl["img"].shape[0]

        l_gt_tuple = model.loss(b_gt)
        l_pl_tuple = model.loss(b_pl)

        l_gt = l_gt_tuple[0].sum()
        l_pl = l_pl_tuple[0].sum()

        eff_n = n_gt + (unsupervised_weight * n_pl)
        loss_sum = (l_gt + (unsupervised_weight * l_pl)) / max(1e-6, eff_n)
        loss_items = l_gt_tuple[1]
        return loss_sum, loss_items
    elif has_pl and not has_gt:
        loss_tuple = model.loss(batch)
        loss_sum = (loss_tuple[0].sum() / max(1, bs)) * unsupervised_weight
        return loss_sum, loss_tuple[1]
    else:
        loss_tuple = model.loss(batch)
        loss_sum = loss_tuple[0].sum() / max(1, bs)
        return loss_sum, loss_tuple[1]


def evaluate_model_coco(
    model: YOLO,
    gt_data_yaml: Path | str,
    coco_gt_path: Path | str | None,
    imgsz: int = 640,
    batch_size: int = 16,
    device: str | int | None = None,
    save_dir: Path | str | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, float]:
    """
    Evaluate YOLO model on validation split, extracting Ultralytics metrics,
    computing fitness, and evaluating pycocotools COCO metrics.
    """
    _load_heavy_dependencies()
    log = logger or get_logger()
    val_kwargs: dict[str, Any] = {
        "data": str(gt_data_yaml),
        "split": "val",
        "save_json": True,
        "batch": batch_size,
        "imgsz": imgsz,
        "verbose": False,
        "plots": False,
    }
    if device is not None:
        val_kwargs["device"] = device
    if save_dir is not None:
        val_kwargs["project"] = str(Path(save_dir).parent)
        val_kwargs["name"] = str(Path(save_dir).name)
        val_kwargs["exist_ok"] = True

    val_results = model.val(**val_kwargs)
    res_dict = val_results.results_dict

    # Extract Ultralytics metrics & fitness
    metrics: dict[str, float] = {
        "metrics/precision(B)": float(res_dict.get("metrics/precision(B)", 0.0)),
        "metrics/recall(B)": float(res_dict.get("metrics/recall(B)", 0.0)),
        "metrics/mAP50(B)": float(res_dict.get("metrics/mAP50(B)", 0.0)),
        "metrics/mAP50-95(B)": float(res_dict.get("metrics/mAP50-95(B)", 0.0)),
        "val/box_loss": float(res_dict.get("val/box_loss", 0.0)),
        "val/cls_loss": float(res_dict.get("val/cls_loss", 0.0)),
        "val/dfl_loss": float(res_dict.get("val/dfl_loss", 0.0)),
        "fitness": float(
            res_dict.get(
                "fitness",
                0.1 * float(res_dict.get("metrics/mAP50(B)", 0.0))
                + 0.9 * float(res_dict.get("metrics/mAP50-95(B)", 0.0)),
            )
        ),
    }

    # Evaluate COCO metrics if COCO ground truth JSON is available
    if coco_gt_path is not None and Path(coco_gt_path).exists():
        preds_json = Path(val_results.save_dir) / "predictions.json"
        coco_res = evaluate_coco(
            coco_gt_path=coco_gt_path,
            predictions_path=preds_json,
            logger=log,
        )
        for k, v in coco_res.items():
            metrics[f"coco/{k}"] = float(v)

    return metrics


class MeanTeacherTrainer:
    """
    Orchestrates Supervised Warm-up and Mean Teacher SSL training loops.
    Supports pause/resume lifecycle via KeyboardInterrupt handling and checkpoint management.
    """

    def __init__(
        self,
        config: MeanTeacherConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        _load_heavy_dependencies()
        self.config = config or MeanTeacherConfig()
        self.exp_dir = self.config.runs_dir / self.config.experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir = self.exp_dir / "weights"
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logger or get_logger(
            log_file=self.exp_dir / "train.log",
            level=logging.INFO,
        )

        # Set device
        if self.config.device is not None:
            self.device = torch.device(self.config.device)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.logger.info(
            f"Initialized MeanTeacherTrainer on device={self.device} (runs_dir={self.exp_dir})"
        )

    def save_checkpoint(
        self,
        path: Path | str,
        epoch: int,
        student: YOLO,
        teacher: YOLO,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: Any,
        best_fitness: float,
        best_epoch: int,
        training_history: list[dict[str, Any]],
        warmup_weights: Path | str | None = None,
        is_interrupted: bool = False,
    ) -> None:
        """Atomically serialize full SSOD training state dictionary to disk."""
        _load_heavy_dependencies()
        dest = Path(path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp_dest = dest.with_suffix(".pt.tmp")

        rng_state = {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }

        use_scaler = (
            scaler is not None and getattr(scaler, "is_enabled", lambda: False)()
        )
        checkpoint: dict[str, Any] = {
            "epoch": epoch,
            "total_epochs": self.config.ssl_epochs,
            "student_state_dict": {
                k: v.cpu().clone() for k, v in student.model.state_dict().items()
            },
            "teacher_state_dict": {
                k: v.cpu().clone() for k, v in teacher.model.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if use_scaler else None,
            "best_fitness": best_fitness,
            "best_epoch": best_epoch,
            "training_history": deepcopy(training_history),
            "config": asdict(self.config),
            "warmup_weights": str(warmup_weights)
            if warmup_weights is not None
            else None,
            "rng_state": rng_state,
            "is_interrupted": is_interrupted,
        }

        torch.save(checkpoint, tmp_dest)
        tmp_dest.replace(dest)

    @staticmethod
    def resolve_checkpoint_path(resume_target: Path | str) -> Path:
        """Resolve a file or directory path into an existing checkpoint .pt file."""
        target = Path(resume_target).resolve()
        if target.is_file():
            return target
        if target.is_dir():
            candidates = [
                target / "weights" / "paused.pt",
                target / "weights" / "last_checkpoint.pt",
                target / "weights" / "last.pt",
                target / "paused.pt",
                target / "last_checkpoint.pt",
                target / "last.pt",
            ]
            for cand in candidates:
                if cand.is_file():
                    return cand
            raise FileNotFoundError(
                f"Could not locate a checkpoint in directory: {target}. Checked: {[str(c) for c in candidates]}"
            )
        raise FileNotFoundError(f"Checkpoint path does not exist: {target}")

    def load_checkpoint(self, checkpoint_path: Path | str) -> dict[str, Any]:
        """
        Load checkpoint data, seamlessly handling rich checkpoints (paused.pt / last_checkpoint.pt)
        and legacy Ultralytics checkpoints (last.pt + results.csv).
        """
        _load_heavy_dependencies()
        ckpt_file = self.resolve_checkpoint_path(checkpoint_path)
        self.logger.info(f"Loading checkpoint from: {ckpt_file}")
        raw_ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)

        # Infer run directory if checkpoint is inside a runs directory
        inferred_runs_dir, inferred_exp_name = infer_exp_dir_from_resume(ckpt_file)
        inferred_exp_dir = inferred_runs_dir / inferred_exp_name
        if inferred_exp_dir != self.exp_dir:
            self.logger.info(
                f"Aligning experiment directory to checkpoint run: {inferred_exp_dir}"
            )
            self.exp_dir = inferred_exp_dir
            self.weights_dir = self.exp_dir / "weights"
            self.config.runs_dir = inferred_runs_dir
            self.config.experiment_name = inferred_exp_name

        # Check for results.csv
        csv_candidates = [
            self.exp_dir / "results.csv",
            ckpt_file.parent.parent / "results.csv",
            ckpt_file.parent / "results.csv",
        ]
        results_csv = next((c for c in csv_candidates if c.is_file()), None)

        # 1. Rich Checkpoint
        if isinstance(raw_ckpt, dict) and "student_state_dict" in raw_ckpt:
            start_epoch = int(raw_ckpt.get("epoch", 0))
            total_epochs = int(raw_ckpt.get("total_epochs", self.config.ssl_epochs))
            best_fitness = float(raw_ckpt.get("best_fitness", -1.0))
            best_epoch = int(raw_ckpt.get("best_epoch", -1))
            training_history = list(raw_ckpt.get("training_history", []))

            # Initialize models
            teacher = YOLO(self.config.model_name)
            student = YOLO(self.config.model_name)
            cfg_dict = deepcopy(DEFAULT_CFG_DICT)
            if hasattr(student, "overrides") and isinstance(student.overrides, dict):
                cfg_dict.update(student.overrides)
            student.model.args = IterableSimpleNamespace(**cfg_dict)
            teacher.model.args = IterableSimpleNamespace(**cfg_dict)

            student.model = student.model.to(self.device)
            teacher.model = teacher.model.to(self.device)

            student.model.load_state_dict(raw_ckpt["student_state_dict"])
            teacher.model.load_state_dict(raw_ckpt["teacher_state_dict"])

            return {
                "format": "rich",
                "checkpoint_file": ckpt_file,
                "start_epoch": start_epoch,
                "total_epochs": total_epochs,
                "teacher": teacher,
                "student": student,
                "optimizer_state_dict": raw_ckpt.get("optimizer_state_dict"),
                "scheduler_state_dict": raw_ckpt.get("scheduler_state_dict"),
                "scaler_state_dict": raw_ckpt.get("scaler_state_dict"),
                "best_fitness": best_fitness,
                "best_epoch": best_epoch,
                "training_history": training_history,
                "warmup_weights": raw_ckpt.get("warmup_weights"),
                "rng_state": raw_ckpt.get("rng_state"),
                "results_csv": results_csv,
            }

        # 2. Legacy Checkpoint (Ultralytics YOLO last.pt)
        self.logger.info(
            "Detected Ultralytics YOLO model file without rich state dictionary. "
            "Reconstructing state from model and results.csv..."
        )
        teacher = YOLO(str(ckpt_file))
        student = YOLO(str(ckpt_file))
        cfg_dict = deepcopy(DEFAULT_CFG_DICT)
        if hasattr(student, "overrides") and isinstance(student.overrides, dict):
            cfg_dict.update(student.overrides)
        student.model.args = IterableSimpleNamespace(**cfg_dict)
        teacher.model.args = IterableSimpleNamespace(**cfg_dict)

        student.model = student.model.to(self.device)
        teacher.model = teacher.model.to(self.device)

        completed_epochs = 0
        best_fitness = -1.0
        best_epoch = -1
        training_history = []

        if results_csv is not None and results_csv.is_file():
            with open(results_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        ep = int(row.get("epoch", 0))
                    except ValueError:
                        continue
                    completed_epochs = max(completed_epochs, ep)
                    fit = float(row.get("fitness", -1.0))
                    if fit > best_fitness:
                        best_fitness = fit
                        best_epoch = ep
                    record: dict[str, Any] = {}
                    for k, v in row.items():
                        try:
                            record[k] = (
                                float(v) if ("." in v or "e" in v.lower()) else int(v)
                            )
                        except ValueError:
                            record[k] = v
                    training_history.append(record)

        start_epoch = completed_epochs

        # Reconstruct total epochs from train.log if needed
        total_epochs = self.config.ssl_epochs
        train_log = self.exp_dir / "train.log"
        if train_log.is_file():
            try:
                log_text = train_log.read_text(encoding="utf-8")
                matches = re.findall(
                    r"Mean Teacher SSL Training \((\d+) epochs\)", log_text
                )
                if matches:
                    logged_epochs = int(matches[-1])
                    if total_epochs == 50 and logged_epochs > 50:
                        total_epochs = logged_epochs
            except Exception as e:
                self.logger.warning(f"Failed to read total epochs from train.log: {e}")

        if total_epochs <= start_epoch:
            total_epochs = max(total_epochs, start_epoch + 10)

        warmup_candidates = [
            self.weights_dir / "warmup_best.pt",
            ckpt_file.parent / "warmup_best.pt",
        ]
        warmup_weights = next((c for c in warmup_candidates if c.is_file()), None)

        return {
            "format": "legacy",
            "checkpoint_file": ckpt_file,
            "start_epoch": start_epoch,
            "total_epochs": total_epochs,
            "teacher": teacher,
            "student": student,
            "optimizer_state_dict": None,
            "scheduler_state_dict": None,
            "scaler_state_dict": None,
            "best_fitness": best_fitness,
            "best_epoch": best_epoch,
            "training_history": training_history,
            "warmup_weights": str(warmup_weights) if warmup_weights else None,
            "rng_state": None,
            "results_csv": results_csv,
        }

    def run_warmup(self) -> Path:
        """
        Execute Phase 1: Supervised warm-up on Ground-Truth dataset only.
        If warmup_weights was supplied and exists, skips training and returns path.
        """
        if (
            self.config.warmup_weights is not None
            and self.config.warmup_weights.exists()
        ):
            self.logger.info(
                f"Using existing warm-up checkpoint: {self.config.warmup_weights}. Skipping warm-up training."
            )
            return self.config.warmup_weights

        self.logger.info(
            f"=== Phase 1: Supervised Warm-up ({self.config.warmup_epochs} epochs) ==="
        )
        warmup_dir = self.exp_dir / "warmup"

        teacher_yolo = YOLO(self.config.model_name)
        train_kwargs: dict[str, Any] = {
            "data": str(self.config.gt_data_yaml),
            "epochs": self.config.warmup_epochs,
            "batch": self.config.batch_size,
            "imgsz": self.config.imgsz,
            "workers": self.config.workers,
            "project": str(warmup_dir.parent),
            "name": str(warmup_dir.name),
            "exist_ok": True,
            "verbose": True,
        }
        if self.config.device is not None:
            train_kwargs["device"] = self.config.device

        try:
            train_results = teacher_yolo.train(**train_kwargs)
        except KeyboardInterrupt:
            self.logger.warning(
                "Supervised warm-up interrupted by user (KeyboardInterrupt)."
            )
            warmup_last = Path(warmup_dir) / "weights" / "last.pt"
            if warmup_last.exists():
                self.logger.info(f"Warm-up checkpoint saved to: {warmup_last}")
            raise

        warmup_best = Path(train_results.save_dir) / "weights" / "best.pt"

        # Copy to main weights dir
        dest = self.weights_dir / "warmup_best.pt"
        if warmup_best.exists():
            torch.save(torch.load(warmup_best, weights_only=False), dest)
            self.logger.info(f"Warm-up complete. Saved warm-up weights to {dest}")
            return dest

        last_pt = Path(train_results.save_dir) / "weights" / "last.pt"
        torch.save(torch.load(last_pt, weights_only=False), dest)
        return dest

    def train_mean_teacher(
        self,
        warmup_weights: Path | str | None = None,
        resume_checkpoint: Path | str | None = None,
    ) -> dict[str, Any]:
        """
        Execute Phase 2: Mean Teacher Semi-Supervised training loop.
        Supports resuming from checkpoint and handles KeyboardInterrupt gracefully.
        """
        _load_heavy_dependencies()
        resume_target = resume_checkpoint or self.config.resume

        last_path = self.weights_dir / "last.pt"
        best_path = self.weights_dir / "best.pt"
        last_ckpt_path = self.weights_dir / "last_checkpoint.pt"

        # 1. Initialize or Resume State
        if resume_target is not None:
            self.logger.info(f"Resuming training from: {resume_target}")
            ckpt_data = self.load_checkpoint(resume_target)

            teacher = ckpt_data["teacher"]
            student = ckpt_data["student"]
            start_epoch = ckpt_data["start_epoch"]
            total_epochs = ckpt_data["total_epochs"]
            best_fitness = ckpt_data["best_fitness"]
            best_epoch = ckpt_data["best_epoch"]
            training_history = ckpt_data["training_history"]
            ckpt_path = (
                ckpt_data["warmup_weights"] or self.weights_dir / "warmup_best.pt"
            )

            self.logger.info(
                f"Resuming at Epoch {start_epoch + 1}/{total_epochs} "
                f"(best_fitness={best_fitness:.4f} at epoch {best_epoch})"
            )
        else:
            # Standard Start: Obtain warm-up weights
            ckpt_path = warmup_weights or self.run_warmup()
            self.logger.info(
                f"=== Phase 2: Mean Teacher SSL Training ({self.config.ssl_epochs} epochs) ==="
            )
            self.logger.info(f"Loading Teacher and Student models from {ckpt_path}...")

            teacher = YOLO(str(ckpt_path))
            student = YOLO(str(ckpt_path))

            cfg_dict = deepcopy(DEFAULT_CFG_DICT)
            if hasattr(student, "overrides") and isinstance(student.overrides, dict):
                cfg_dict.update(student.overrides)
            student.model.args = IterableSimpleNamespace(**cfg_dict)
            teacher.model.args = IterableSimpleNamespace(**cfg_dict)

            start_epoch = 0
            total_epochs = self.config.ssl_epochs
            best_fitness = -1.0
            best_epoch = -1
            training_history = []
            ckpt_data = None

        # Place models on device
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        student.model = student.model.to(self.device)
        teacher.model = teacher.model.to(self.device)
        teacher.model.eval()

        for p in student.model.parameters():
            p.requires_grad = True
        for p in teacher.model.parameters():
            p.requires_grad = False

        # Vectorized EMA tracker
        ema = VectorizedEMA(
            teacher_model=teacher.model,
            student_model=student.model,
            decay=self.config.ema_decay,
        )

        teacher.save(str(last_path))

        # 2. Initialize SSODDatasetHandler and PseudoLabelGenerator
        pseudo_generator = PseudoLabelGenerator(
            teacher_model=YOLO(str(last_path)),
            batch_size=self.config.batch_size * 2,
            half=self.config.pseudo_half and self.device.type == "cuda",
            workers=self.config.workers,
            conf=self.config.pseudo_conf,
            iou_threshold=self.config.pseudo_iou,
            device=self.device,
            logger=self.logger,
        )

        augment_cfg = SSODAugmentConfig(
            tail_ratio=self.config.tail_ratio,
            close_mosaic_ratio=self.config.close_mosaic_ratio,
        )

        dataset_handler = SSODDatasetHandler(
            gt_data_yaml=self.config.gt_data_yaml,
            pl_images_dir=self.config.pl_images_dir,
            pl_labels_dir=self.config.pl_labels_dir,
            imgsz=self.config.imgsz,
            batch_size=self.config.batch_size,
            workers=self.config.workers,
            seed=self.config.seed,
            augment_cfg=augment_cfg,
            logger=self.logger,
        )

        # 3. Setup Optimizer & Scheduler for Student
        optimizer = torch.optim.AdamW(
            student.model.parameters(),
            lr=self.config.lr0,
            betas=(self.config.momentum, 0.999),
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=self.config.lr0 * self.config.lrf,
        )

        is_bf16 = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if is_bf16 else torch.float16
        use_scaler = self.config.amp and (self.device.type == "cuda") and not is_bf16
        scaler = torch.amp.GradScaler(
            self.device.type,
            enabled=use_scaler,
        )

        # Restore optimizer / scheduler state if available
        if ckpt_data is not None:
            if ckpt_data.get("optimizer_state_dict") is not None:
                try:
                    optimizer.load_state_dict(ckpt_data["optimizer_state_dict"])
                    self.logger.info("Restored optimizer state from checkpoint.")
                except Exception as e:
                    self.logger.warning(f"Could not restore optimizer state: {e}")
            if ckpt_data.get("scheduler_state_dict") is not None:
                try:
                    scheduler.load_state_dict(ckpt_data["scheduler_state_dict"])
                    self.logger.info("Restored scheduler state from checkpoint.")
                except Exception as e:
                    self.logger.warning(f"Could not restore scheduler state: {e}")
            elif start_epoch > 0:
                # Fast forward scheduler for legacy checkpoints
                for _ in range(start_epoch):
                    scheduler.step()
                self.logger.info(
                    f"Advanced CosineAnnealingLR scheduler {start_epoch} steps to match resume epoch."
                )

            if use_scaler and ckpt_data.get("scaler_state_dict") is not None:
                try:
                    scaler.load_state_dict(ckpt_data["scaler_state_dict"])
                    self.logger.info("Restored GradScaler state from checkpoint.")
                except Exception as e:
                    self.logger.warning(f"Could not restore scaler state: {e}")

            if ckpt_data.get("rng_state") is not None:
                try:
                    rng = ckpt_data["rng_state"]
                    torch.set_rng_state(rng["torch"])
                    if torch.cuda.is_available() and rng.get("cuda") is not None:
                        torch.cuda.set_rng_state_all(rng["cuda"])
                    np.random.set_state(rng["numpy"])
                    random.setstate(rng["python"])
                    self.logger.info("Restored RNG state from checkpoint.")
                except Exception as e:
                    self.logger.warning(f"Could not restore RNG state: {e}")

        self.logger.info(
            f"Training precision: {'bfloat16 (native, scaler-free)' if is_bf16 else ('float16 with GradScaler' if self.config.amp else 'float32')}"
        )

        # 4. Results tracking and results.csv synchronization
        results_csv = self.exp_dir / "results.csv"
        csv_header_written = results_csv.exists()

        if results_csv.exists() and start_epoch > 0:
            lines = results_csv.read_text(encoding="utf-8").splitlines()
            if lines:
                header = lines[0]
                valid_lines = [header]
                for line in lines[1:]:
                    parts = line.split(",")
                    if parts and parts[0].isdigit():
                        if int(parts[0]) <= start_epoch:
                            valid_lines.append(line)
                results_csv.write_text("\n".join(valid_lines) + "\n", encoding="utf-8")
                self.logger.info(
                    f"Synchronized results.csv to last completed epoch {start_epoch}."
                )

        # 5. Training Loop with graceful KeyboardInterrupt handling
        try:
            for epoch in range(start_epoch, total_epochs):
                # Clean epoch-start snapshots in case of interruption
                clean_student_sd = {
                    k: v.cpu().clone() for k, v in student.model.state_dict().items()
                }
                clean_teacher_sd = {
                    k: v.cpu().clone() for k, v in teacher.model.state_dict().items()
                }
                clean_optimizer_sd = deepcopy(optimizer.state_dict())
                clean_scheduler_sd = deepcopy(scheduler.state_dict())
                clean_scaler_sd = deepcopy(scaler.state_dict()) if use_scaler else None
                clean_rng = {
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None,
                    "numpy": np.random.get_state(),
                    "python": random.getstate(),
                }

                epoch_start_time = time.time()
                self.logger.info(f"--- Epoch {epoch + 1}/{total_epochs} ---")

                # A. Snapshot teacher weights and isolate for pseudo-label generation
                teacher.save(str(last_path))
                inference_teacher = YOLO(str(last_path))
                pseudo_generator.model = inference_teacher
                if hasattr(pseudo_generator.model, "predictor"):
                    pseudo_generator.model.predictor = None

                candidates_path = None
                if self.config.save_candidates:
                    cand_dir = self.exp_dir / self.config.candidates_dir_name
                    cand_dir.mkdir(parents=True, exist_ok=True)
                    candidates_path = cand_dir / f"candidates_epoch_{epoch + 1}.parquet"

                self.logger.info(
                    f"Generating pseudo-labels with Teacher (conf={self.config.pseudo_conf})..."
                )
                gen_out = pseudo_generator.generate(
                    images_dir=self.config.pl_images_dir,
                    labels_dir=self.config.pl_labels_dir,
                    max_images=self.config.max_pseudo_images,
                    save_candidates_path=candidates_path,
                )
                dataset_handler.reload_pseudo_labels(
                    filtered_df=gen_out.get("filtered_df")
                )

                # B. Configure epoch (25% GT tail sampling, 10% mosaic closing)
                train_loader = dataset_handler.get_train_loader(persistent_workers=True)
                dataset_handler.set_epoch(
                    epoch=epoch,
                    total_epochs=total_epochs,
                    dataloader=train_loader,
                )

                # C. Student Training over Batches
                student.model.train()
                train_losses: list[float] = []

                pbar = tqdm(
                    train_loader,
                    desc=f"Epoch {epoch + 1}/{total_epochs} [Student Train]",
                    leave=False,
                )

                try:
                    for batch in pbar:
                        # Move tensors to device & normalize image to float [0, 1]
                        for k, v in batch.items():
                            if isinstance(v, torch.Tensor):
                                batch[k] = v.to(
                                    self.device, non_blocking=self.device.type == "cuda"
                                )
                        batch["img"] = batch["img"].float() / 255.0

                        optimizer.zero_grad()

                        with torch.amp.autocast(
                            self.device.type,
                            dtype=amp_dtype,
                            enabled=self.config.amp and self.device.type == "cuda",
                        ):
                            loss, loss_items = compute_batch_loss(
                                model=student.model,
                                batch=batch,
                                unsupervised_weight=self.config.unsupervised_weight,
                            )

                        if use_scaler:
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                student.model.parameters(), max_norm=10.0
                            )
                            if torch.isfinite(grad_norm):
                                scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                student.model.parameters(), max_norm=10.0
                            )
                            optimizer.step()

                        # D. Update Teacher EMA at EVERY STEP
                        ema.update()

                        train_losses.append(loss.item())
                        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                finally:
                    pbar.close()

                scheduler.step()
                mean_train_loss = float(np.mean(train_losses)) if train_losses else 0.0

                # E. Evaluate Teacher Model on GT Validation split
                self.logger.info("Evaluating Teacher model on validation split...")
                val_save_dir = self.exp_dir / "val" / f"val_epoch_{epoch + 1}"
                val_save_dir.mkdir(parents=True, exist_ok=True)

                teacher.save(str(last_path))
                val_teacher = YOLO(str(last_path))

                val_metrics = evaluate_model_coco(
                    model=val_teacher,
                    gt_data_yaml=self.config.gt_data_yaml,
                    coco_gt_path=self.config.coco_gt_path,
                    imgsz=self.config.imgsz,
                    batch_size=self.config.batch_size * 2,
                    device=self.device,
                    save_dir=val_save_dir,
                    logger=self.logger,
                )

                epoch_time = time.time() - epoch_start_time
                fitness = val_metrics.get("fitness", 0.0)
                map50 = val_metrics.get("metrics/mAP50(B)", 0.0)
                map50_95 = val_metrics.get("metrics/mAP50-95(B)", 0.0)

                self.logger.info(
                    f"Epoch {epoch + 1} Result ({epoch_time:.1f}s): "
                    f"Train Loss={mean_train_loss:.4f} | "
                    f"Teacher mAP50={map50:.4f}, mAP50-95={map50_95:.4f}, Fitness={fitness:.4f}"
                )

                # Record epoch log
                record = {
                    "epoch": epoch + 1,
                    "train_loss": mean_train_loss,
                    "epoch_time": epoch_time,
                    **val_metrics,
                }
                training_history.append(record)

                # Append to results.csv
                fieldnames = list(record.keys())
                if not csv_header_written:
                    with open(results_csv, "w", encoding="utf-8") as f:
                        f.write(",".join(fieldnames) + "\n")
                    csv_header_written = True

                with open(results_csv, "a", encoding="utf-8") as f:
                    f.write(",".join(str(record[k]) for k in fieldnames) + "\n")

                # Checkpoint management
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_epoch = epoch + 1
                    teacher.save(str(best_path))
                    self.logger.info(
                        f"New best model saved to {best_path} (fitness={best_fitness:.4f} at epoch {best_epoch})."
                    )

                # Save rich checkpoint at end of epoch
                self.save_checkpoint(
                    path=last_ckpt_path,
                    epoch=epoch + 1,
                    student=student,
                    teacher=teacher,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler if use_scaler else None,
                    best_fitness=best_fitness,
                    best_epoch=best_epoch,
                    training_history=training_history,
                    warmup_weights=ckpt_path,
                    is_interrupted=False,
                )

        except KeyboardInterrupt:
            self.logger.warning("=" * 70)
            self.logger.warning(
                "KeyboardInterrupt detected! Gracefully pausing Mean Teacher training..."
            )
            self.logger.warning("=" * 70)

            paused_path = self.weights_dir / "paused.pt"
            # Save clean state for this epoch so resuming restarts cleanly at beginning of epoch
            tmp_paused = paused_path.with_suffix(".pt.tmp")
            paused_checkpoint = {
                "epoch": epoch,
                "total_epochs": total_epochs,
                "student_state_dict": clean_student_sd,
                "teacher_state_dict": clean_teacher_sd,
                "optimizer_state_dict": clean_optimizer_sd,
                "scheduler_state_dict": clean_scheduler_sd,
                "scaler_state_dict": clean_scaler_sd,
                "best_fitness": best_fitness,
                "best_epoch": best_epoch,
                "training_history": deepcopy(training_history),
                "config": asdict(self.config),
                "warmup_weights": str(ckpt_path),
                "rng_state": clean_rng,
                "is_interrupted": True,
            }
            torch.save(paused_checkpoint, tmp_paused)
            tmp_paused.replace(paused_path)

            try:
                teacher.save(str(last_path))
            except Exception:
                pass

            self.logger.info(
                f"Training successfully paused at Epoch {epoch + 1}/{total_epochs}."
            )
            self.logger.info(f"Saved paused checkpoint to: {paused_path}")
            self.logger.info(
                f"To resume training from this paused state, run:\n"
                f"  uv run python code/ssod/mean_teacher.py --resume {paused_path}\n"
                f"or:\n"
                f"  uv run python code/ssod/mean_teacher.py --resume {self.exp_dir}"
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return {
                "paused": True,
                "paused_at_epoch": epoch + 1,
                "paused_checkpoint": str(paused_path),
                "best_fitness": best_fitness,
                "best_epoch": best_epoch,
                "results_csv": str(results_csv),
                "history": training_history,
            }

        self.logger.info(
            f"\n=== Training Complete ===\n"
            f"Best Fitness={best_fitness:.4f} observed at Epoch {best_epoch}.\n"
            f"Results saved to {self.exp_dir}"
        )
        return {
            "paused": False,
            "best_fitness": best_fitness,
            "best_epoch": best_epoch,
            "best_model": str(self.weights_dir / "best.pt"),
            "results_csv": str(results_csv),
            "history": training_history,
        }


@app.command()
def main(
    gt_yaml: Path = typer.Option(
        Path("data/flow_img.yaml"), "--gt-yaml", help="Path to GT dataset YAML."
    ),
    pl_images: Path = typer.Option(
        Path("data/video_frames/images/all"),
        "--pl-images",
        help="Path to unlabeled images dir.",
    ),
    pl_labels: Path = typer.Option(
        Path("data/video_frames/labels/all"),
        "--pl-labels",
        help="Path to pseudo-labels dir.",
    ),
    coco_gt: Optional[Path] = typer.Option(
        None, "--coco-gt", help="Path to reference COCO annotations JSON."
    ),
    model: str = typer.Option("yolo26m.pt", "--model", help="Base model weights."),
    warmup_weights: Optional[Path] = typer.Option(
        None, "--warmup-weights", help="Optional path to existing warm-up checkpoint."
    ),
    warmup_epochs: int = typer.Option(
        10, "--warmup-epochs", help="Supervised warm-up epochs."
    ),
    ssl_epochs: int = typer.Option(50, "--ssl-epochs", help="Mean Teacher SSL epochs."),
    batch: int = typer.Option(16, "--batch", help="Batch size."),
    imgsz: int = typer.Option(640, "--imgsz", help="Input image size."),
    workers: int = typer.Option(4, "--workers", help="DataLoader worker processes."),
    device: Optional[str] = typer.Option(None, "--device", help="PyTorch device."),
    ema_decay: float = typer.Option(
        0.999, "--ema-decay", help="Teacher EMA decay per step."
    ),
    unsupervised_weight: float = typer.Option(
        1.0, "--unsupervised-weight", help="Weight lambda_u for pseudo loss."
    ),
    max_pseudo_images: Optional[int] = typer.Option(
        None, "--max-pseudo-images", help="Max unlabeled images for pseudo-labeling."
    ),
    save_candidates: bool = typer.Option(
        False,
        "--save-candidates",
        help="Save raw multi-view pseudo-label candidate detections to Parquet files per epoch.",
    ),
    project: Optional[Path] = typer.Option(
        None,
        "--project",
        help="Output project directory [default: runs/mean_teacher or inferred from --resume].",
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        help="Experiment name [default: exp or inferred from --resume].",
    ),
    resume: Optional[Path] = typer.Option(
        None,
        "--resume",
        "-r",
        help="Path to paused checkpoint (.pt) or experiment directory to resume training from.",
    ),
) -> None:
    """Run Mean Teacher semi-supervised training."""
    if resume is not None:
        inferred_project, inferred_name = infer_exp_dir_from_resume(resume)
        if project is None:
            project = inferred_project
        if name is None:
            name = inferred_name
    else:
        if project is None:
            project = Path("runs/mean_teacher")
        if name is None:
            name = "exp"

    cfg = MeanTeacherConfig(
        model_name=model,
        warmup_weights=warmup_weights,
        resume=resume,
        gt_data_yaml=gt_yaml,
        pl_images_dir=pl_images,
        pl_labels_dir=pl_labels,
        coco_gt_path=coco_gt,
        warmup_epochs=warmup_epochs,
        ssl_epochs=ssl_epochs,
        batch_size=batch,
        imgsz=imgsz,
        workers=workers,
        device=device,
        ema_decay=ema_decay,
        unsupervised_weight=unsupervised_weight,
        max_pseudo_images=max_pseudo_images,
        save_candidates=save_candidates,
        runs_dir=project,
        experiment_name=name,
    )
    trainer = MeanTeacherTrainer(config=cfg)
    try:
        trainer.train_mean_teacher(resume_checkpoint=resume)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app()
