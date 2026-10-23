"""
Mean Teacher Semi-Supervised Object Detection (SSOD) training loop for YOLO.

Features:
1. Supervised Warm-up Phase:
   - Trains Teacher on Ground-Truth data only for n-epochs (or loads from --warmup-weights).
   - Initializes Student model from trained Teacher weights (theta_S <- theta_T).
2. Mean Teacher Training Loop (online Size-Stratified Ground Thresholding):
   - After the warm-up, builds the initial normalised score histogram of each COCO size bin
     from the Teacher's candidates on a uniform sample of the unlabeled pool.
   - Starts each epoch with a fresh uniform sample of unlabeled images and synchronizes
     SSODDatasetHandler epoch sequencing (GT shuffled with unlabeled images; optional GT tail)
     and 10% mosaic closing.
   - At every step: derives the per-bin thresholds from the histograms, pseudo-labels the
     batch's unlabeled images with the Teacher (OnlinePseudoLabeler), trains the Student with
     YOLO default detection loss (with optional unsupervised weight lambda_u), updates the
     Teacher EMA, then the histogram EMA with the batch's candidate histograms.
   - Logs the histograms every epoch (histograms.csv, histograms/epoch_<k>.npz).
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
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
    from ssod.dataset import SSODAugmentConfig, SSODDatasetHandler
    from ssod.metrics import evaluate_coco
    from ssod import pseudo_labeling
    from ssod.losses import WeightedDetectionLoss

# Runtime placeholders populated on-demand by _load_heavy_dependencies()
torch: Any = None
nn: Any = None
np: Any = None
tqdm: Any = None
YOLO: Any = None
DetectionModel: Any = None
DetectionValidator: Any = None
DEFAULT_CFG_DICT: Any = None
IterableSimpleNamespace: Any = None
SSODAugmentConfig: Any = None
SSODDatasetHandler: Any = None
evaluate_coco: Any = None
pseudo_labeling: Any = None
WeightedDetectionLoss: Any = None
E2ELoss: Any = None


def _load_heavy_dependencies() -> None:
    """Lazily load heavy scientific and CV libraries to ensure instantaneous CLI --help."""
    global torch, nn, np, tqdm, YOLO, DetectionModel, DetectionValidator, DEFAULT_CFG_DICT, IterableSimpleNamespace
    global SSODAugmentConfig, SSODDatasetHandler, evaluate_coco, pseudo_labeling
    global WeightedDetectionLoss, E2ELoss

    if torch is not None:
        return

    import numpy as _np
    import torch as _torch
    import torch.nn as _nn
    from tqdm import tqdm as _tqdm  # type: ignore[import-untyped]
    from ultralytics import YOLO as _YOLO
    from ultralytics.models.yolo.detect import DetectionValidator as _DetectionValidator
    from ultralytics.nn.tasks import DetectionModel as _DetectionModel
    from ultralytics.utils.loss import E2ELoss as _E2ELoss
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
        from ssod import pseudo_labeling as _pseudo_labeling
        from ssod.losses import WeightedDetectionLoss as _WeightedDetectionLoss
    except ImportError:
        from dataset import (  # type: ignore[import-not-found,no-redef]
            SSODAugmentConfig as _SSODAugmentConfig,
            SSODDatasetHandler as _SSODDatasetHandler,
        )
        from metrics import evaluate_coco as _evaluate_coco  # type: ignore[import-not-found,no-redef]
        import pseudo_labeling as _pseudo_labeling  # type: ignore[import-not-found,no-redef]
        from losses import WeightedDetectionLoss as _WeightedDetectionLoss  # type: ignore[import-not-found,no-redef]

    torch = _torch
    nn = _nn
    np = _np
    tqdm = _tqdm
    YOLO = _YOLO
    DetectionModel = _DetectionModel
    DetectionValidator = _DetectionValidator
    DEFAULT_CFG_DICT = _DEFAULT_CFG_DICT
    IterableSimpleNamespace = _IterableSimpleNamespace
    SSODAugmentConfig = _SSODAugmentConfig
    SSODDatasetHandler = _SSODDatasetHandler
    evaluate_coco = _evaluate_coco
    pseudo_labeling = _pseudo_labeling
    WeightedDetectionLoss = _WeightedDetectionLoss
    E2ELoss = _E2ELoss


def __getattr__(name: str) -> Any:
    """Module-level attribute lookup triggering on-demand dependency loading."""
    _load_heavy_dependencies()
    if name in globals() and globals()[name] is not None:
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def discover_coco_gt_path(gt_data_yaml: Path | str | None) -> Path | None:
    """Auto-discover reference COCO annotations JSON path from gt_data_yaml or standard locations."""
    candidates: list[Path] = []
    if gt_data_yaml is not None:
        yaml_path = Path(gt_data_yaml).resolve()
        if yaml_path.is_file():
            try:
                import yaml
                with open(yaml_path, "r", encoding="utf-8") as f:
                    ydata = yaml.safe_load(f)
                if isinstance(ydata, dict) and "path" in ydata:
                    base_path = Path(ydata["path"])
                    candidates.extend([
                        base_path / "val" / "annotations.json",
                        base_path / "annotations.json",
                    ])
            except Exception:
                pass
        candidates.extend([
            yaml_path.parent / "val" / "annotations.json",
            yaml_path.parent / "flow_img" / "val" / "annotations.json",
        ])
    candidates.append(Path("data/flow_img/val/annotations.json").resolve())

    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def save_config_yaml(config: MeanTeacherConfig, path: Path | str) -> None:
    """Save MeanTeacherConfig to YAML file."""
    import yaml
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    cfg_dict = asdict(config)
    serializable: dict[str, Any] = {}
    for k, v in cfg_dict.items():
        if isinstance(v, Path):
            serializable[k] = str(v)
        elif isinstance(v, (int, float, bool, str)) or v is None:
            serializable[k] = v
        else:
            serializable[k] = str(v)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(serializable, f, sort_keys=False, default_flow_style=False)


def load_config_yaml(path: Path | str) -> dict[str, Any]:
    """Load configuration dictionary from YAML file."""
    import yaml
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


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
    ema_decay: float = 0.9996
    tail_ratio: float = 0.0  # GT fraction placed unmixed at the epoch end; 0 mixes all GT
    close_mosaic_ratio: float = 0.10
    unsupervised_weight: float = 1.0
    unsupervised_warmup_epochs: int = 0  # linear lambda_u ramp-up; 0 disables
    gt_repeat: int = 1  # times the GT head is repeated per epoch
    pl_strong_aug: bool = False  # strong Student augmentation on pseudo-labeled images
    # Fresh uniform unlabeled image sample per epoch (also the size of the initial histogram
    # sample); None (or negative) = whole pool, 0 = none (GT-only control)
    pl_images_per_epoch: int | None = 2400
    pl_soft_weights: bool = True  # threshold-aware per-box PL loss weights
    pl_cls_only: bool = False  # pseudo boxes train classification only (no box/DFL regression)
    pl_size_bins: bool = True  # ground threshold per COCO size bin (False: one for all boxes)
    pl_multi_view: bool = True  # Teacher inference on normal + flipped views (False: normal only)

    # Pseudo-labeling parameters
    pseudo_conf: float = 0.001  # candidate confidence floor (and lower end of the histograms)
    pseudo_iou: float = 0.5  # NMS IoU for merging the views
    pseudo_fallback_threshold: float = 0.4
    max_pseudo_images: int | None = None  # cap on the unlabeled pool (first N images)

    # Logging and output
    runs_dir: Path = field(default_factory=lambda: Path("runs/mean_teacher"))
    experiment_name: str = "exp"
    seed: int = 42

    def __post_init__(self) -> None:
        self.gt_data_yaml = Path(self.gt_data_yaml).resolve()
        self.pl_images_dir = Path(self.pl_images_dir).resolve()
        if self.coco_gt_path is None:
            self.coco_gt_path = discover_coco_gt_path(self.gt_data_yaml)
        elif self.coco_gt_path is not None:
            self.coco_gt_path = Path(self.coco_gt_path).resolve()
        if self.warmup_weights is not None:
            self.warmup_weights = Path(self.warmup_weights).resolve()
        if self.pl_images_per_epoch is not None and self.pl_images_per_epoch < 0:
            self.pl_images_per_epoch = None
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
        for key in ("pl_weight", "box_weight"):
            if key in batch:
                new_batch[key] = batch[key][b_mask]
        idx_map = {idx.item(): i for i, idx in enumerate(indices)}
        remapped = [idx_map[int(x.item())] for x in batch["batch_idx"][b_mask]]
        new_batch["batch_idx"] = torch.tensor(
            remapped, dtype=torch.float32, device=batch["batch_idx"].device
        )
    else:
        new_batch["bboxes"] = torch.empty((0, 4), device=batch["img"].device)
        new_batch["cls"] = torch.empty((0, 1), device=batch["img"].device)
        new_batch["batch_idx"] = torch.empty((0,), device=batch["img"].device)
        for key in ("pl_weight", "box_weight"):
            if key in batch:
                new_batch[key] = torch.empty((0,), device=batch["img"].device)

    return new_batch


def configure_student_criterion(model: nn.Module) -> Any:
    """
    Build the Student criterion explicitly and pin YOLO26 E2E loss gains to their end-of-warm-up values.

    Ultralytics' E2ELoss starts at o2m=0.8 / o2o=0.2 and relies on the trainer calling criterion.update()
    every epoch to decay towards o2m=final_o2m (0.1). The custom SSL loop never calls update(), so a lazily
    created criterion would under-weight the one-to-one head used at inference for the whole SSL phase.
    Uses WeightedDetectionLoss (per-box `pl_weight` importance weights; identical to the stock
    loss when the batch carries no weights) for both E2E branches.
    Must be called after the model is on its target device and model.args is set.
    """
    _load_heavy_dependencies()
    if getattr(model, "end2end", False):
        criterion = E2ELoss(model, loss_fn=WeightedDetectionLoss)
    else:
        criterion = WeightedDetectionLoss(model)
    if hasattr(criterion, "final_o2m"):
        criterion.o2m = criterion.final_o2m
        criterion.o2o = max(criterion.total - criterion.o2m, 0.0)
    model.criterion = criterion
    return criterion


def initial_candidate_counts(
    teacher_model: nn.Module,
    labeler: Any,
    loader: Any,
    device: torch.device,
) -> np.ndarray:
    """
    (size bins x score bins) Teacher candidate counts over every image of `loader`, taken on
    the same weak views as during training; the start of the score-histogram EMA.
    """
    counts = None
    for batch in tqdm(loader, desc="Initial score histograms", leave=False):
        imgs = batch["teacher_img"].to(device, non_blocking=device.type == "cuda")
        batch_counts = labeler.count(labeler.candidates(teacher_model, imgs))
        counts = batch_counts if counts is None else counts + batch_counts
    if counts is None:
        return np.zeros((len(pseudo_labeling.SIZE_BINS), labeler.n_bins), dtype=np.int64)
    return counts


def compute_batch_loss(
    model: nn.Module,
    batch: dict[str, Any],
    unsupervised_weight: float = 1.0,
    pl_cls_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute YOLO default detection loss on batch.
    If unsupervised_weight != 1.0 and both GT and Pseudo items are present,
    applies unsupervised_weight to pseudo-labeled loss.
    If pl_cls_only, pseudo boxes contribute to the classification term only: their
    box/DFL weight (`batch["box_weight"]`) is 0, while GT boxes keep theirs.
    Returns (loss_tensor, loss_items_detach).
    """
    _load_heavy_dependencies()
    bs = batch["img"].shape[0]
    is_pseudo = batch.get("is_pseudo")

    if pl_cls_only and torch.is_tensor(is_pseudo) and "bboxes" in batch:
        n_boxes = batch["bboxes"].shape[0]
        pl_weight = batch.get("pl_weight")
        if pl_weight is None:
            pl_weight = torch.ones(n_boxes, device=batch["bboxes"].device)
        box_is_gt = ~is_pseudo.to(batch["batch_idx"].device)[batch["batch_idx"].long()]
        batch["box_weight"] = pl_weight * box_is_gt.to(pl_weight.dtype)

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


class SSODDetectionValidator:
    """Stub wrapper allowing SSODDetectionValidator to be referenced before heavy imports."""
    pass


def _create_ssod_detection_validator() -> type:
    """Instantiate SSODDetectionValidator subclassing DetectionValidator on-demand."""
    _load_heavy_dependencies()

    class _SSODDetectionValidator(DetectionValidator):
        """
        Custom DetectionValidator that computes real validation loss (box, cls, dfl)
        on the GT validation split while retaining full predictions.json export for pycocotools COCO evaluation.
        """

        def __call__(self, trainer=None, model=None):
            self._raw_model = deepcopy(model) if model is not None else None
            return super().__call__(trainer=trainer, model=model)

        def init_metrics(self, model):
            super().init_metrics(model)
            self.val_loss = torch.zeros(3, device=self.device)
            self.val_loss_count = 0
            if self._raw_model is not None:
                self._raw_model = self._raw_model.to(self.device)
                cfg = deepcopy(DEFAULT_CFG_DICT)
                self._raw_model.args = IterableSimpleNamespace(**cfg)
                self._raw_model.criterion = None

        def update_metrics(self, preds, batch):
            super().update_metrics(preds, batch)
            if self._raw_model is not None:
                if getattr(self.args, "half", False) and self.device.type != "cpu":
                    self._raw_model.half()
                else:
                    self._raw_model.float()
                with torch.no_grad():
                    loss_tuple = self._raw_model.loss(batch)
                    self.val_loss += loss_tuple[1].detach()
                    self.val_loss_count += 1

        def finalize_metrics(self):
            super().finalize_metrics()
            if self.val_loss_count > 0:
                mean_loss = (self.val_loss / self.val_loss_count).cpu().tolist()
                loss_dict = {
                    "val/box_loss": float(mean_loss[0]),
                    "val/cls_loss": float(mean_loss[1]),
                    "val/dfl_loss": float(mean_loss[2]),
                }
                self.metrics.val_loss = loss_dict
                self.metrics.results_dict.update(loss_dict)

    return _SSODDetectionValidator


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
    computing real validation loss, fitness, and evaluating pycocotools COCO metrics.
    """
    _load_heavy_dependencies()
    log = logger or get_logger()

    validator_cls = _create_ssod_detection_validator()
    val_kwargs: dict[str, Any] = {
        "validator": validator_cls,
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
    val_loss_dict = getattr(val_results, "val_loss", {})

    # Extract Ultralytics metrics & fitness
    metrics: dict[str, float] = {
        "metrics/precision(B)": float(res_dict.get("metrics/precision(B)", 0.0)),
        "metrics/recall(B)": float(res_dict.get("metrics/recall(B)", 0.0)),
        "metrics/mAP50(B)": float(res_dict.get("metrics/mAP50(B)", 0.0)),
        "metrics/mAP50-95(B)": float(res_dict.get("metrics/mAP50-95(B)", 0.0)),
        "val/box_loss": float(val_loss_dict.get("val/box_loss", res_dict.get("val/box_loss", 0.0))),
        "val/cls_loss": float(val_loss_dict.get("val/cls_loss", res_dict.get("val/cls_loss", 0.0))),
        "val/dfl_loss": float(val_loss_dict.get("val/dfl_loss", res_dict.get("val/dfl_loss", 0.0))),
        "fitness": float(
            res_dict.get(
                "fitness",
                0.1 * float(res_dict.get("metrics/mAP50(B)", 0.0))
                + 0.9 * float(res_dict.get("metrics/mAP50-95(B)", 0.0)),
            )
        ),
    }

    # Auto-discover COCO annotations if not provided
    effective_coco_gt = coco_gt_path
    if effective_coco_gt is None:
        effective_coco_gt = discover_coco_gt_path(gt_data_yaml)

    # Evaluate COCO metrics if COCO ground truth JSON is available
    if effective_coco_gt is not None and Path(effective_coco_gt).exists():
        preds_json = Path(val_results.save_dir) / "predictions.json"
        coco_res = evaluate_coco(
            coco_gt_path=effective_coco_gt,
            predictions_path=preds_json,
            imgsz=imgsz,
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
        if not (self.exp_dir / "args.yaml").exists():
            self.save_args()

    def save_args(self) -> None:
        """Serialize current MeanTeacherConfig to runs_dir/args.yaml."""
        args_path = self.exp_dir / "args.yaml"
        save_config_yaml(self.config, args_path)

    @staticmethod
    def _infer_nc_from_state_dict(state_dict: dict[str, Any]) -> int:
        """Infer number of classes nc from state_dict classification head weights."""
        for key in reversed(list(state_dict.keys())):
            if "cv3" in key and "weight" in key and hasattr(state_dict[key], "shape") and state_dict[key].ndim == 4:
                return int(state_dict[key].shape[0])
        return 1

    @staticmethod
    def _resolve_model_yaml(model_name: str) -> str:
        """Resolve model name (e.g. yolo26m.pt) to model yaml name (e.g. yolo26m.yaml)."""
        stem = Path(model_name).stem
        if not stem.endswith(".yaml"):
            return f"{stem}.yaml"
        return stem

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
        score_histograms: dict[str, Any] | None = None,
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
            "nc": getattr(student.model, "nc", 1),
            "model_name": self.config.model_name,
            "warmup_weights": str(warmup_weights)
            if warmup_weights is not None
            else None,
            "rng_state": rng_state,
            "is_interrupted": is_interrupted,
            "score_histograms": deepcopy(score_histograms),
        }

        torch.save(checkpoint, tmp_dest)
        tmp_dest.replace(dest)
        self.save_args()

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

            # Restore config if available in checkpoint or args.yaml
            saved_cfg = raw_ckpt.get("config")
            args_yaml = self.exp_dir / "args.yaml"
            if args_yaml.is_file():
                try:
                    loaded_yaml_cfg = load_config_yaml(args_yaml)
                    if saved_cfg is None:
                        saved_cfg = loaded_yaml_cfg
                    elif isinstance(saved_cfg, dict):
                        saved_cfg = {**loaded_yaml_cfg, **saved_cfg}
                except Exception as e:
                    self.logger.warning(f"Failed to load {args_yaml}: {e}")

            if isinstance(saved_cfg, dict):
                for k, v in saved_cfg.items():
                    if hasattr(self.config, k):
                        curr_v = getattr(self.config, k)
                        if k == "coco_gt_path" and curr_v is None and v is not None:
                            setattr(self.config, k, Path(v).resolve() if v else None)
                        elif k == "model_name" and v:
                            setattr(self.config, k, str(v))
                        elif k in ("gt_data_yaml", "pl_images_dir") and v:
                            setattr(self.config, k, Path(v).resolve())
                        elif k in ("unsupervised_weight", "ema_decay", "tail_ratio", "close_mosaic_ratio", "lr0", "lrf", "pseudo_fallback_threshold"):
                            setattr(self.config, k, float(v))
                        elif k in ("ssl_epochs", "warmup_epochs", "batch_size", "imgsz", "workers", "unsupervised_warmup_epochs", "gt_repeat", "seed"):
                            setattr(self.config, k, int(v))
                        elif k in ("pl_strong_aug", "pl_soft_weights", "pl_cls_only", "pl_size_bins", "pl_multi_view"):
                            setattr(self.config, k, bool(v))
                        elif k == "pl_images_per_epoch":
                            setattr(self.config, k, int(v) if v is not None else None)

            if self.config.coco_gt_path is None:
                self.config.coco_gt_path = discover_coco_gt_path(self.config.gt_data_yaml)

            self.save_args()

            # Locate base model template with matching architecture and nc
            base_candidates = [
                ckpt_file.parent / "last.pt",
                self.weights_dir / "last.pt",
                ckpt_file.parent / "warmup_best.pt",
                self.weights_dir / "warmup_best.pt",
                ckpt_file.parent / "best.pt",
                self.weights_dir / "best.pt",
                raw_ckpt.get("warmup_weights"),
            ]
            base_model_path = next(
                (c for c in base_candidates if c and Path(c).is_file()), None
            )

            if base_model_path is not None:
                self.logger.info(f"Instantiating model templates from base checkpoint: {base_model_path}")
                teacher = YOLO(str(base_model_path))
                student = YOLO(str(base_model_path))
            else:
                nc = int(raw_ckpt.get("nc") or self._infer_nc_from_state_dict(raw_ckpt["student_state_dict"]))
                model_yaml = self._resolve_model_yaml(self.config.model_name)
                self.logger.info(f"Instantiating DetectionModel({model_yaml}, nc={nc})")
                teacher = YOLO(model_yaml)
                teacher.model = DetectionModel(cfg=model_yaml, nc=nc, verbose=False)
                teacher.model.nc = nc
                student = YOLO(model_yaml)
                student.model = DetectionModel(cfg=model_yaml, nc=nc, verbose=False)
                student.model.nc = nc

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
                "score_histograms": raw_ckpt.get("score_histograms"),
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
            "score_histograms": None,
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
        from ultralytics.utils.torch_utils import init_seeds

        # Global RNGs drive augmentation and DataLoader worker seeds; sampling and ordering use
        # their own generators seeded from the same value
        init_seeds(self.config.seed)
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

        criterion = configure_student_criterion(student.model)
        if hasattr(criterion, "o2m"):
            self.logger.info(
                f"Student E2E loss gains: o2m={criterion.o2m:.2f}, o2o={criterion.o2o:.2f}"
            )

        # Vectorized EMA tracker
        ema = VectorizedEMA(
            teacher_model=teacher.model,
            student_model=student.model,
            decay=self.config.ema_decay,
        )

        teacher.save(str(last_path))

        # 2. Initialize SSODDatasetHandler, online pseudo-labeler and score histograms
        use_pseudo_labels = self.config.pl_images_per_epoch != 0
        augment_cfg = SSODAugmentConfig(
            tail_ratio=self.config.tail_ratio,
            close_mosaic_ratio=self.config.close_mosaic_ratio,
            pl_strong=self.config.pl_strong_aug,
        )

        dataset_handler = SSODDatasetHandler(
            gt_data_yaml=self.config.gt_data_yaml,
            pl_images_dir=self.config.pl_images_dir,
            imgsz=self.config.imgsz,
            batch_size=self.config.batch_size,
            workers=self.config.workers,
            seed=self.config.seed,
            augment_cfg=augment_cfg,
            tail_ratio=self.config.tail_ratio,
            close_mosaic_ratio=self.config.close_mosaic_ratio,
            gt_repeat=self.config.gt_repeat,
            logger=self.logger,
        )

        labeler = pseudo_labeling.OnlinePseudoLabeler(
            conf=self.config.pseudo_conf,
            iou_threshold=self.config.pseudo_iou,
            multi_view=self.config.pl_multi_view,
            soft_weights=self.config.pl_soft_weights,
            erase_drop_ratio=augment_cfg.erase_drop_ratio,
        )
        histograms = pseudo_labeling.ScoreHistograms(
            conf_floor=self.config.pseudo_conf,
            fallback=self.config.pseudo_fallback_threshold,
            size_bins=self.config.pl_size_bins,
            n_bins=labeler.n_bins,
        )
        pl_pool: list[Path] = []
        if use_pseudo_labels:
            pl_pool = pseudo_labeling.list_images(self.config.pl_images_dir)
            if self.config.max_pseudo_images is not None and self.config.max_pseudo_images > 0:
                pl_pool = pl_pool[: self.config.max_pseudo_images]
            self.logger.info(f"Unlabeled pool: {len(pl_pool)} images in {self.config.pl_images_dir}.")

        def build_initial_histograms() -> None:
            """Start the histograms from the Teacher's candidates on a uniform pool sample."""
            dataset_handler.set_pl_images(
                pseudo_labeling.sample_image_files(
                    pl_pool, self.config.pl_images_per_epoch, self.config.seed
                )
            )
            counts = initial_candidate_counts(
                teacher.model, labeler, dataset_handler.get_unlabeled_loader(), self.device
            )
            histograms.reset(counts)
            pseudo_labeling.append_histograms(self.exp_dir, 0, histograms, counts)
            self.logger.info(
                f"Initial score histograms from {len(dataset_handler.pl_dataset)} images "
                f"({int(counts.sum())} candidates): "
                + ", ".join(
                    f"{b}: threshold={t.threshold:.3f}, object mode={t.object_mode:.3f}"
                    + ("" if t.from_valley else " (fallback)")
                    for b, t in histograms.thresholds().items()
                )
            )

        if use_pseudo_labels:
            saved_histograms = ckpt_data.get("score_histograms") if ckpt_data is not None else None
            if saved_histograms is not None:
                histograms.load_state_dict(saved_histograms)
                pseudo_labeling.truncate_histogram_log(self.exp_dir, start_epoch)
                self.logger.info("Restored score histograms from checkpoint.")
            else:
                pseudo_labeling.truncate_histogram_log(self.exp_dir, -1)
                build_initial_histograms()

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
        csv_header_written = results_csv.exists() and results_csv.stat().st_size > 0

        if results_csv.exists() and start_epoch > 0:
            lines = results_csv.read_text(encoding="utf-8").splitlines()
            if lines:
                old_header = lines[0].split(",")
                expected_threshold_cols = [
                    "threshold/small",
                    "threshold/medium",
                    "threshold/large",
                ]
                missing_cols = [c for c in expected_threshold_cols if c not in old_header]

                valid_rows = []
                for line in lines[1:]:
                    parts = line.split(",")
                    if parts and parts[0].isdigit() and int(parts[0]) <= start_epoch:
                        row_dict = dict(zip(old_header, parts))
                        for mc in missing_cols:
                            row_dict[mc] = "0.0"
                        valid_rows.append(row_dict)

                new_header = list(old_header)
                if missing_cols:
                    insert_idx = 3 if len(new_header) > 3 else len(new_header)
                    for mc in reversed(missing_cols):
                        new_header.insert(insert_idx, mc)

                out_lines = [",".join(new_header)]
                for r in valid_rows:
                    out_lines.append(",".join(str(r.get(c, "0.0")) for c in new_header))
                results_csv.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
                self.logger.info(
                    f"Synchronized and aligned results.csv to last completed epoch {start_epoch}."
                )

        def append_result(record: dict[str, Any]) -> None:
            """Append a row in the existing header's column order, extending the header if needed."""
            nonlocal csv_header_written
            if not csv_header_written:
                with open(results_csv, "w", encoding="utf-8") as f:
                    f.write(",".join(record.keys()) + "\n")
                csv_header_written = True
            lines = results_csv.read_text(encoding="utf-8").splitlines()
            header = lines[0].split(",")
            new_cols = [k for k in record if k not in header]
            if new_cols:
                rows = [line + ",0.0" * len(new_cols) for line in lines[1:] if line]
                header += new_cols
                results_csv.write_text(
                    "\n".join([",".join(header), *rows]) + "\n", encoding="utf-8"
                )
            with open(results_csv, "a", encoding="utf-8") as f:
                f.write(",".join(str(record.get(k, 0.0)) for k in header) + "\n")

        # Baseline: evaluate the starting Teacher with the same evaluator as SSL epochs
        if resume_target is None and start_epoch == 0:
            self.logger.info("Evaluating starting Teacher (epoch 0 baseline)...")
            teacher.save(str(last_path))
            baseline_metrics = evaluate_model_coco(
                model=YOLO(str(last_path)),
                gt_data_yaml=self.config.gt_data_yaml,
                coco_gt_path=self.config.coco_gt_path,
                imgsz=self.config.imgsz,
                batch_size=self.config.batch_size * 2,
                device=self.device,
                save_dir=self.exp_dir / "val" / "val_epoch_0",
                logger=self.logger,
            )
            self.logger.info(
                f"Epoch 0 Baseline: Teacher mAP50={baseline_metrics.get('metrics/mAP50(B)', 0.0):.4f}, "
                f"mAP50-95={baseline_metrics.get('metrics/mAP50-95(B)', 0.0):.4f}"
            )
            # Pseudo-label columns hold the initial thresholds (no pseudo-labels trained yet)
            baseline_log = pseudo_labeling.PseudoLabelLog()
            if use_pseudo_labels:
                baseline_log.thresholds.append(
                    np.array([t.threshold for t in histograms.thresholds().values()])
                )
                baseline_log.object_modes.append(
                    np.array([t.object_mode for t in histograms.thresholds().values()])
                )
            baseline_record = {
                "epoch": 0,
                "train_loss": 0.0,
                "epoch_time": 0.0,
                **baseline_log.columns(),
                **baseline_metrics,
            }
            training_history.append(baseline_record)
            append_result(baseline_record)

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
                clean_histograms = histograms.state_dict() if use_pseudo_labels else None
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

                # A. Fresh uniform sample of unlabeled images (pseudo-labelled online, per step)
                if use_pseudo_labels:
                    dataset_handler.set_pl_images(
                        pseudo_labeling.sample_image_files(
                            pl_pool, self.config.pl_images_per_epoch, self.config.seed + epoch + 1
                        )
                    )
                else:
                    # Supervised fine-tuning control: same loop, Teacher EMA and schedule, no pseudo-labels
                    self.logger.info("Pseudo-labeling disabled (pl_images_per_epoch=0): GT-only epoch.")
                    dataset_handler.set_pl_images([])
                pl_log = pseudo_labeling.PseudoLabelLog()

                # B. Configure epoch (GT/PL sequencing, 10% mosaic closing)
                train_loader = dataset_handler.get_train_loader(persistent_workers=True)
                dataset_handler.set_epoch(
                    epoch=epoch,
                    total_epochs=total_epochs,
                    dataloader=train_loader,
                )

                # C. Student Training over Batches
                unsupervised_weight = self.config.unsupervised_weight
                if self.config.unsupervised_warmup_epochs > 0:
                    ramp = min(1.0, (epoch + 1) / self.config.unsupervised_warmup_epochs)
                    unsupervised_weight *= ramp
                    self.logger.info(f"Unsupervised weight (ramp-up): {unsupervised_weight:.4f}")

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

                        # Teacher pseudo-labels from the current per-bin thresholds
                        if use_pseudo_labels:
                            thresholds = histograms.thresholds()
                            step_labels = labeler.label(teacher.model, batch, thresholds)

                        optimizer.zero_grad()

                        with torch.amp.autocast(
                            self.device.type,
                            dtype=amp_dtype,
                            enabled=self.config.amp and self.device.type == "cuda",
                        ):
                            loss, loss_items = compute_batch_loss(
                                model=student.model,
                                batch=batch,
                                unsupervised_weight=unsupervised_weight,
                                pl_cls_only=self.config.pl_cls_only,
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

                        # D. Update Teacher EMA at EVERY STEP, then the score-histogram EMA
                        ema.update()
                        if use_pseudo_labels:
                            histograms.update(step_labels.counts, decay=self.config.ema_decay)
                            pl_log.add(step_labels, thresholds)

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

                # Record epoch log with pseudo-label statistics and mean thresholds
                record = {
                    "epoch": epoch + 1,
                    "train_loss": mean_train_loss,
                    "epoch_time": epoch_time,
                    **pl_log.columns(),
                    **val_metrics,
                }
                training_history.append(record)

                append_result(record)
                if use_pseudo_labels:
                    pl_log.save(self.exp_dir, epoch + 1, histograms)
                    self.logger.info(
                        f"Pseudo-labels: {record['pl_boxes']:.0f} boxes on {record['pl_images']:.0f} images; "
                        + ", ".join(
                            f"{b}: threshold={record[f'threshold/{b}']:.3f}, share={record[f'pl_share/{b}']:.3f}, "
                            f"mean weight={record[f'pl_weight/{b}']:.3f}"
                            for b in pseudo_labeling.SIZE_BINS
                        )
                    )

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
                    score_histograms=histograms.state_dict() if use_pseudo_labels else None,
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
                "nc": getattr(student.model, "nc", 1),
                "model_name": self.config.model_name,
                "warmup_weights": str(ckpt_path),
                "rng_state": clean_rng,
                "is_interrupted": True,
                "score_histograms": clean_histograms,
            }
            torch.save(paused_checkpoint, tmp_paused)
            tmp_paused.replace(paused_path)
            self.save_args()

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
    ctx: typer.Context,
    gt_yaml: Path = typer.Option(
        Path("data/flow_img.yaml"), "--gt-yaml", help="Path to GT dataset YAML."
    ),
    pl_images: Path = typer.Option(
        Path("data/video_frames/images/all"),
        "--pl-images",
        help="Path to unlabeled images dir.",
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
        0.9996, "--ema-decay", help="Teacher EMA decay per step."
    ),
    lr0: float = typer.Option(0.0002, "--lr0", help="Initial Student learning rate (AdamW)."),
    lrf: float = typer.Option(
        0.01, "--lrf", help="Final learning rate as a fraction of lr0 (cosine schedule)."
    ),
    unsupervised_weight: float = typer.Option(
        1.0, "--unsupervised-weight", help="Weight lambda_u for pseudo loss."
    ),
    unsupervised_warmup_epochs: int = typer.Option(
        0,
        "--unsupervised-warmup-epochs",
        help="Epochs of linear ramp-up for lambda_u (0 disables).",
    ),
    tail_ratio: float = typer.Option(
        0.0,
        "--tail-ratio",
        help="Fraction of GT placed, unmixed, at the tail of each epoch (0 mixes all GT with the pseudo-labels).",
    ),
    close_mosaic_ratio: float = typer.Option(
        0.10,
        "--close-mosaic-ratio",
        help="Fraction of final epochs with GT Mosaic disabled (0 keeps Mosaic on throughout).",
        min=0.0,
        max=1.0,
    ),
    gt_repeat: int = typer.Option(
        1, "--gt-repeat", help="Times the GT head is repeated in each epoch.", min=1
    ),
    pl_strong_aug: bool = typer.Option(
        False,
        "--pl-strong-aug/--no-pl-strong-aug",
        help="Strong Student augmentation (HSV, blur, erasing) on pseudo-labeled images.",
    ),
    pl_images_per_epoch: int = typer.Option(
        2400,
        "--pl-images-per-epoch",
        help=(
            "Train on a fresh uniform random sample of this many unlabeled images each epoch, "
            "pseudo-labelled online; the initial score histograms use a sample of the same size "
            "(0 disables pseudo-labels: supervised fine-tuning control; -1 uses the whole pool)."
        ),
        min=-1,
    ),
    pl_cls_only: bool = typer.Option(
        False,
        "--pl-cls-only/--no-pl-cls-only",
        help="Train pseudo boxes for classification only (no box/DFL regression towards Teacher boxes).",
    ),
    pl_soft_weights: bool = typer.Option(
        True,
        "--pl-soft-weights/--no-pl-soft-weights",
        help="Weight each pseudo box's loss by its score between its size bin's threshold and object mode.",
    ),
    pl_size_bins: bool = typer.Option(
        True,
        "--pl-size-bins/--no-pl-size-bins",
        help="Pseudo-label ground threshold per COCO size bin (off: one threshold for all boxes).",
    ),
    pl_multi_view: bool = typer.Option(
        True,
        "--pl-multi-view/--no-pl-multi-view",
        help="Teacher pseudo-labels from the normal and flipped views (off: normal view only).",
    ),
    max_pseudo_images: Optional[int] = typer.Option(
        None, "--max-pseudo-images", help="Cap on the unlabeled pool (its first N images)."
    ),
    pseudo_fallback_threshold: float = typer.Option(
        0.4,
        "--pseudo-fallback-threshold",
        help="Pseudo-label score threshold for a size bin whose score histogram has no valley.",
    ),
    project: Optional[Path] = typer.Option(
        None,
        "--project",
        help="Output project directory [default: runs/mean_teacher or inferred from --resume].",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Seed for pseudo-label image sampling, epoch ordering and augmentation RNGs.",
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
    saved_cfg: dict[str, Any] = {}
    if resume is not None:
        inferred_project, inferred_name = infer_exp_dir_from_resume(resume)
        if project is None:
            project = inferred_project
        if name is None:
            name = inferred_name
        exp_dir = project / name
        args_yaml_path = exp_dir / "args.yaml"
        if args_yaml_path.is_file():
            try:
                saved_cfg = load_config_yaml(args_yaml_path)
            except Exception:
                pass
        if not saved_cfg:
            try:
                _load_heavy_dependencies()
                resolved_ckpt = MeanTeacherTrainer.resolve_checkpoint_path(resume)
                ckpt_dict = torch.load(resolved_ckpt, map_location="cpu", weights_only=False)
                if isinstance(ckpt_dict, dict) and "config" in ckpt_dict and isinstance(ckpt_dict["config"], dict):
                    saved_cfg = ckpt_dict["config"]
            except Exception:
                pass
    else:
        if project is None:
            project = Path("runs/mean_teacher")
        if name is None:
            name = "exp"

    def was_specified(param_name: str) -> bool:
        src = ctx.get_parameter_source(param_name)
        return src is not None and getattr(src, "name", "") == "COMMANDLINE"

    def pick_val(param_name: str, cli_val: Any, cfg_key: str, default_val: Any) -> Any:
        if was_specified(param_name):
            return cli_val
        if saved_cfg and cfg_key in saved_cfg and saved_cfg[cfg_key] is not None:
            return saved_cfg[cfg_key]
        return cli_val if cli_val is not None else default_val

    chosen_model = pick_val("model", model, "model_name", "yolo26m.pt")
    chosen_warmup_weights = pick_val("warmup_weights", warmup_weights, "warmup_weights", None)
    chosen_gt_yaml = pick_val("gt_yaml", gt_yaml, "gt_data_yaml", Path("data/flow_img.yaml"))
    chosen_pl_images = pick_val("pl_images", pl_images, "pl_images_dir", Path("data/video_frames/images/all"))
    chosen_coco_gt = pick_val("coco_gt", coco_gt, "coco_gt_path", None)
    chosen_warmup_epochs = int(pick_val("warmup_epochs", warmup_epochs, "warmup_epochs", 10))
    chosen_ssl_epochs = int(pick_val("ssl_epochs", ssl_epochs, "ssl_epochs", 50))
    chosen_batch = int(pick_val("batch", batch, "batch_size", 16))
    chosen_imgsz = int(pick_val("imgsz", imgsz, "imgsz", 640))
    chosen_workers = int(pick_val("workers", workers, "workers", 4))
    chosen_device = pick_val("device", device, "device", None)
    chosen_ema_decay = float(pick_val("ema_decay", ema_decay, "ema_decay", 0.9996))
    chosen_lr0 = float(pick_val("lr0", lr0, "lr0", 0.0002))
    chosen_lrf = float(pick_val("lrf", lrf, "lrf", 0.01))
    chosen_unsupervised_weight = float(pick_val("unsupervised_weight", unsupervised_weight, "unsupervised_weight", 1.0))
    chosen_unsupervised_warmup_epochs = int(pick_val("unsupervised_warmup_epochs", unsupervised_warmup_epochs, "unsupervised_warmup_epochs", 0))
    chosen_tail_ratio = float(pick_val("tail_ratio", tail_ratio, "tail_ratio", 0.0))
    chosen_close_mosaic_ratio = float(pick_val("close_mosaic_ratio", close_mosaic_ratio, "close_mosaic_ratio", 0.10))
    chosen_gt_repeat = int(pick_val("gt_repeat", gt_repeat, "gt_repeat", 1))
    chosen_pl_strong_aug = bool(pick_val("pl_strong_aug", pl_strong_aug, "pl_strong_aug", False))
    if not was_specified("pl_images_per_epoch") and saved_cfg and "pl_images_per_epoch" in saved_cfg:
        # A saved None means "whole pool" and must survive resume
        chosen_pl_images_per_epoch = saved_cfg["pl_images_per_epoch"]
    else:
        chosen_pl_images_per_epoch = pl_images_per_epoch
    if chosen_pl_images_per_epoch is not None:
        chosen_pl_images_per_epoch = int(chosen_pl_images_per_epoch)
        if chosen_pl_images_per_epoch < 0:
            chosen_pl_images_per_epoch = None
    chosen_pl_cls_only = bool(pick_val("pl_cls_only", pl_cls_only, "pl_cls_only", False))
    chosen_pl_soft_weights = bool(pick_val("pl_soft_weights", pl_soft_weights, "pl_soft_weights", True))
    chosen_pl_size_bins = bool(pick_val("pl_size_bins", pl_size_bins, "pl_size_bins", True))
    chosen_pl_multi_view = bool(pick_val("pl_multi_view", pl_multi_view, "pl_multi_view", True))
    chosen_max_pseudo_images = pick_val("max_pseudo_images", max_pseudo_images, "max_pseudo_images", None)
    if chosen_max_pseudo_images is not None:
        chosen_max_pseudo_images = int(chosen_max_pseudo_images)
    chosen_pseudo_fallback_threshold = float(pick_val("pseudo_fallback_threshold", pseudo_fallback_threshold, "pseudo_fallback_threshold", 0.4))
    chosen_seed = int(pick_val("seed", seed, "seed", 42))

    cfg = MeanTeacherConfig(
        model_name=str(chosen_model),
        warmup_weights=Path(chosen_warmup_weights) if chosen_warmup_weights else None,
        resume=resume,
        gt_data_yaml=Path(chosen_gt_yaml),
        pl_images_dir=Path(chosen_pl_images),
        coco_gt_path=Path(chosen_coco_gt) if chosen_coco_gt else None,
        warmup_epochs=chosen_warmup_epochs,
        ssl_epochs=chosen_ssl_epochs,
        batch_size=chosen_batch,
        imgsz=chosen_imgsz,
        workers=chosen_workers,
        device=chosen_device,
        ema_decay=chosen_ema_decay,
        lr0=chosen_lr0,
        lrf=chosen_lrf,
        unsupervised_weight=chosen_unsupervised_weight,
        unsupervised_warmup_epochs=chosen_unsupervised_warmup_epochs,
        tail_ratio=chosen_tail_ratio,
        close_mosaic_ratio=chosen_close_mosaic_ratio,
        gt_repeat=chosen_gt_repeat,
        pl_strong_aug=chosen_pl_strong_aug,
        pl_images_per_epoch=chosen_pl_images_per_epoch,
        pl_soft_weights=chosen_pl_soft_weights,
        pl_cls_only=chosen_pl_cls_only,
        pl_size_bins=chosen_pl_size_bins,
        pl_multi_view=chosen_pl_multi_view,
        max_pseudo_images=chosen_max_pseudo_images,
        pseudo_fallback_threshold=chosen_pseudo_fallback_threshold,
        seed=chosen_seed,
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
