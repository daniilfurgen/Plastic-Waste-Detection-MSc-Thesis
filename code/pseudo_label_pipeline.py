"""
Iterative semi-supervised (pseudo-labeling) training pipeline for YOLO26,
built entirely on Ultralytics' built-in train() / predict() / tune() methods.

Designed to run UNATTENDED. All Ultralytics calls below are blocking, so this
script just runs start-to-finish on its own -- launch it with e.g.

    nohup python pseudo_label_pipeline.py > /dev/null 2>&1 &

or inside tmux/screen, or as a batch/Slurm job. Progress goes to pipeline.log
and a final summary.yaml so you can review results after the fact rather
than watching the terminal.

IMPORTANT: run a tiny dry run first (see bottom of file) before launching the
full unattended version -- this catches path/config bugs in minutes instead
of after several wasted hours.

Expected disk layout (edit CONFIG below to match yours):

  data/
    flow_img.yaml                     <- your existing labeled FloW-img config
    flow_img/
      images/train/*.jpg  images/val/*.jpg
      labels/train/*.txt  labels/val/*.txt
    video_frames/
      images/all/*.jpg                <- extracted, UNLABELED video frames
      labels/all/                     <- populated by this script

  runs/
    tune/                             <- hyperparameter search output
    pseudo/round_0/, round_1/, ...    <- per-round training output
"""

from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

# --------------------------------------------------------------------------- #
# CONFIG -- edit this block for your setup
# --------------------------------------------------------------------------- #

# Disable cuDNN's algorithm autotuner. This is the most common practical fix
# for "RuntimeError: GET/FIND was unable to find an engine to execute this
# computation" -- cuDNN's benchmark-mode algorithm search can fail outright
# on some driver/cuDNN/GPU combinations, and gets more likely to misfire
# once GPU memory is fragmented from earlier runs in the same process.
# Costs a little raw speed, buys a lot of reliability for an unattended job.
import torch

torch.backends.cudnn.benchmark = False

# Configure the name of the experiment run.
EXRUN_NAME = "YOLO26m-P2-SSL w/ pretrained weights (10 rounds)"

# Configure paths relative to the repo root.
ROOT = Path.cwd().resolve()
ORIG_DATA_YAML = ROOT / "data" / "flow_img.yaml"

VIDEO_IMAGES_DIR = ROOT / "data" / "video_frames" / "images" / "all"
VIDEO_LABELS_DIR = ROOT / "data" / "video_frames" / "labels" / "all"

RUNS_DIR = ROOT / "runs"
PSEUDO_RUN_DIR = RUNS_DIR / "pseudo" / "yolo26m-p2-10-rounds"
TUNE_RUN_DIR = RUNS_DIR / "tune"

# Configure the YOLO26 model size and whether to use the P2 small-object detection head.
# P2 head is a part of FPN that has a higher resolution feature map for detecting small objects.
# If your dataset has many small objects, consider using P2_HEAD = True.
SIZE = "m"  # "n", "s", "m", "l", "x" -- YOLO26 model size
P2_HEAD = True  # use P2 small-object detection head (yolo26*-p2.yaml/.pt)
BASE_CONFIGURATION = f"yolo26{SIZE}-p2.yaml" if P2_HEAD else f"yolo26{SIZE}.yaml"
PRETRAINED_WEIGHTS = f"yolo26{SIZE}.pt"

BATCH = 10  # Optimal batch size benchmarked for RTX 4070 (12GB) to avoid RAM spillover
WORKERS = 4  # DataLoader workers
N_ROUNDS = 10
EPOCHS_PER_ROUND = 150
PATIENCE = 25  # early stopping -- avoids burning unattended hours on a plateau
IMG_SIZE = 640  # higher than the 640 default helps small floating debris

# Confidence threshold for accepting a pseudo-label, one entry per round.
# Start conservative (fewer, more reliable boxes); loosen as the model improves.
CONF_SCHEDULE = [0.85, 0.80, 0.75][:N_ROUNDS]

# Cap pseudo-labeled images per round relative to the real labeled set, so
# pseudo data can't drown out the ground-truth signal (standard STAC/Soft
# Teacher-style precaution).
MAX_PSEUDO_RATIO = 1.0  # at most 1x as many pseudo images as real train images

TUNE_FRACTION = 0.10  # subset size for hyperparameter search
TUNE_SEED = 42

LOG_FILE = ROOT / "pipeline.log"

file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[file_handler, logging.StreamHandler()],
)
log = logging.getLogger("pseudo_pipeline")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_image_list(data_cfg: dict, split: str) -> list[Path]:
    """Return absolute image paths for a split, whether the yaml points to a
    directory or a .txt file listing image paths (both are valid Ultralytics
    formats)."""
    root = Path(data_cfg.get("path", "."))
    if not root.is_absolute():
        root = ORIG_DATA_YAML.parent / root
    entry = root / data_cfg[split]
    if entry.is_dir():
        return sorted(
            p for p in entry.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
    lines = entry.read_text().splitlines()
    return [
        Path(line) if Path(line).is_absolute() else root / line
        for line in lines
        if line.strip()
    ]


def make_pseudo_labels(model: YOLO, round_idx: int, conf: float) -> int:
    """Run inference on unlabeled video frames; save YOLO-format label .txt
    files into VIDEO_LABELS_DIR (mirroring the images/ -> labels/ convention
    Ultralytics expects). Returns the number of frames that received at
    least one confident detection."""
    predict_dir = PSEUDO_RUN_DIR / f"round_{round_idx}" / "predict"
    results = model.predict(
        source=str(VIDEO_IMAGES_DIR),
        conf=conf,
        save=False,
        save_txt=True,
        save_conf=False,
        project=str(predict_dir.parent),
        name="predict",
        exist_ok=True,
        stream=True,  # generator -- keeps memory flat over many frames
        verbose=False,
    )
    for _ in results:  # must consume the generator to actually run inference
        pass

    VIDEO_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    src_label_dir = predict_dir / "labels"
    moved = 0
    if src_label_dir.exists():
        for txt in src_label_dir.glob("*.txt"):
            shutil.copy(txt, VIDEO_LABELS_DIR / txt.name)
            moved += 1
    log.info(
        f"Round {round_idx}: {moved} frames received pseudo-labels at conf>={conf}"
    )
    return moved


def build_round_yaml(round_idx: int, max_pseudo: int) -> Path:
    """Merge original labeled training images with this round's pseudo-
    labeled frames into a combined train list. Validation always stays
    pinned to the ORIGINAL human-labeled val set -- never validate on
    pseudo-labels."""
    orig_cfg = load_yaml(ORIG_DATA_YAML)
    orig_train_imgs = resolve_image_list(orig_cfg, "train")

    pseudo_imgs = [
        VIDEO_IMAGES_DIR / f"{p.stem}.jpg" for p in VIDEO_LABELS_DIR.glob("*.txt")
    ]
    pseudo_imgs = [p for p in pseudo_imgs if p.exists()]

    random.Random(round_idx).shuffle(pseudo_imgs)
    pseudo_imgs = pseudo_imgs[:max_pseudo]

    merged_list_path = PSEUDO_RUN_DIR / f"round_{round_idx}" / "train_merged.txt"
    merged_list_path.parent.mkdir(parents=True, exist_ok=True)
    with open(merged_list_path, "w") as f:
        for p in orig_train_imgs + pseudo_imgs:
            f.write(str(p) + "\n")

    round_cfg = dict(orig_cfg)
    round_cfg["train"] = str(merged_list_path)  # val entry untouched

    round_yaml_path = PSEUDO_RUN_DIR / f"round_{round_idx}" / "data.yaml"
    with open(round_yaml_path, "w") as f:
        yaml.safe_dump(round_cfg, f)

    log.info(
        f"Round {round_idx}: train set = {len(orig_train_imgs)} real + "
        f"{len(pseudo_imgs)} pseudo = {len(orig_train_imgs) + len(pseudo_imgs)} images"
    )
    return round_yaml_path


def build_tuning_subset_yaml(
    fraction: float = TUNE_FRACTION, seed: int = TUNE_SEED
) -> Path:
    """Uniformly subsample the TRAIN split only, for fast hyperparameter
    search. Validation stays full-size and fixed, so trial rankings aren't
    themselves noisy from an undersized val set."""
    orig_cfg = load_yaml(ORIG_DATA_YAML)
    train_imgs = resolve_image_list(orig_cfg, "train")

    rng = random.Random(seed)
    k = max(1, int(len(train_imgs) * fraction))
    subset = rng.sample(train_imgs, k)

    subset_list_path = TUNE_RUN_DIR / "train_subset.txt"
    subset_list_path.parent.mkdir(parents=True, exist_ok=True)
    with open(subset_list_path, "w") as f:
        for p in subset:
            f.write(str(p) + "\n")

    subset_cfg = dict(orig_cfg)
    subset_cfg["train"] = str(subset_list_path)
    subset_yaml_path = TUNE_RUN_DIR / "data_subset.yaml"
    with open(subset_yaml_path, "w") as f:
        yaml.safe_dump(subset_cfg, f)

    log.info(
        f"Tuning subset: {k}/{len(train_imgs)} train images ({fraction:.0%}), seed={seed}"
    )
    return subset_yaml_path


def log_environment_info() -> None:
    """Log current system RAM and GPU VRAM stats to help monitor resources."""
    try:
        import psutil

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


# --------------------------------------------------------------------------- #
# Stage 1 (optional): hyperparameter tuning on the small subset
# --------------------------------------------------------------------------- #


def run_hyperparameter_tuning() -> dict:
    subset_yaml = build_tuning_subset_yaml()
    model = YOLO(BASE_CONFIGURATION)
    model.load(PRETRAINED_WEIGHTS)
    log.info("Starting hyperparameter tuning (genetic algorithm, fully automatic)...")
    log.info(
        f"Tuning config: batch={BATCH}, workers={WORKERS}, imgsz={IMG_SIZE}, "
        f"model={BASE_CONFIGURATION}, weights={PRETRAINED_WEIGHTS}"
    )
    log_environment_info()
    model.tune(
        data=str(subset_yaml),
        epochs=30,  # short trials -- only need relative ranking between configs
        iterations=10,  # generations of the genetic search; raise/lower per your budget
        batch=BATCH,
        workers=WORKERS,
        imgsz=IMG_SIZE,
        patience=10,
        project=str(TUNE_RUN_DIR),
        name="evolve",
        exist_ok=True,
        plots=False,
    )
    # The tuner writes a best-hyperparameters summary into the run dir.
    # Confirm the exact filename printed in your console / Ultralytics
    # version's tuning guide, then load it here, e.g.:
    best_hyp_path = TUNE_RUN_DIR / "evolve" / "best_hyperparameters.yaml"
    best_hyp = load_yaml(best_hyp_path) if best_hyp_path.exists() else {}
    log.info(f"Tuning complete. Best hyperparameters: {best_hyp}")
    return best_hyp


# --------------------------------------------------------------------------- #
# Stage 2: iterative pseudo-labeling rounds
# --------------------------------------------------------------------------- #


def run_pipeline(hyp_overrides: dict | None = None):
    hyp_overrides = hyp_overrides or {}
    configuration = BASE_CONFIGURATION
    weights = PRETRAINED_WEIGHTS
    summary_rows = []

    log.info(
        f"Pipeline config: batch={BATCH}, workers={WORKERS}, imgsz={IMG_SIZE}, "
        f"epochs_per_round={EPOCHS_PER_ROUND}, patience={PATIENCE}, "
        f"model={BASE_CONFIGURATION}, initial_weights={PRETRAINED_WEIGHTS}"
    )
    log_environment_info()

    for round_idx in range(N_ROUNDS):
        try:
            log.info(f"=== Round {round_idx}: training ===")
            model = YOLO(configuration)
            model.load(weights)

            if round_idx == 0:
                data_yaml = ORIG_DATA_YAML
            else:
                real_n = len(resolve_image_list(load_yaml(ORIG_DATA_YAML), "train"))
                data_yaml = build_round_yaml(
                    round_idx, max_pseudo=int(real_n * MAX_PSEUDO_RATIO)
                )

            log.info(
                f"Round {round_idx} training started: model={configuration}, "
                f"data={data_yaml}, batch={BATCH}, workers={WORKERS}, imgsz={IMG_SIZE}"
            )
            log_environment_info()

            train_results = model.train(
                data=str(data_yaml),
                epochs=EPOCHS_PER_ROUND,
                batch=BATCH,
                workers=WORKERS,
                patience=PATIENCE,
                imgsz=IMG_SIZE,
                project=str(PSEUDO_RUN_DIR),
                name=f"round_{round_idx}_train",
                exist_ok=True,
                **hyp_overrides,
            )
            configuration = Path(train_results.save_dir) / "weights" / "best.pt"  # type: ignore
            log.info(
                f"Round {round_idx} training finished. Best checkpoint: {configuration}"
            )

            # Always evaluate against the original, fixed human-labeled val set.
            metrics = model.val(data=str(ORIG_DATA_YAML), split="val")
            log.info(
                f"Round {round_idx} val: mAP50={metrics.box.map50:.4f} mAP50-95={metrics.box.map:.4f}"
            )
            summary_rows.append(
                {
                    "round": round_idx,
                    "weights": str(configuration),
                    "mAP50": float(metrics.box.map50),
                    "mAP50_95": float(metrics.box.map),
                }
            )

            if round_idx < N_ROUNDS - 1:
                log.info(
                    f"=== Round {round_idx}: generating pseudo-labels for next round ==="
                )
                make_pseudo_labels(
                    YOLO(configuration), round_idx, conf=CONF_SCHEDULE[round_idx]
                )

        except (
            Exception
        ) as e:  # keep going safely / fail loudly into the log, not silently
            log.exception(f"Round {round_idx} failed, stopping pipeline: {e}")
            break

    log.info(f"Pipeline finished. Summary: {summary_rows}")
    with open(PSEUDO_RUN_DIR / "summary.yaml", "w") as f:
        yaml.safe_dump(summary_rows, f)
    return summary_rows


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # --- DRY RUN FIRST ---
    # Before launching the real multi-hour unattended run, sanity-check the
    # whole thing end-to-end in minutes by temporarily setting, e.g.:
    #   N_ROUNDS = 1; EPOCHS_PER_ROUND = 2; TUNE_FRACTION = 0.02; iterations=3
    # This catches path typos / yaml mistakes / empty-pseudo-label edge cases
    # before they waste real compute time.
    try:
        log.info(f"== Starting pseudo-labeling pipeline: {EXRUN_NAME} ==")

        # best_hyp = run_hyperparameter_tuning()  # comment out to skip tuning
        best_hyp = yaml.safe_load(
            (TUNE_RUN_DIR / "evolve-6" / "best_hyperparameters.yaml").read_text()
        )
        run_pipeline(hyp_overrides=best_hyp)
    except KeyboardInterrupt:
        log.warning("Pipeline interrupted by user.")
    except Exception as e:
        log.exception(f"Pipeline failed: {e}")
