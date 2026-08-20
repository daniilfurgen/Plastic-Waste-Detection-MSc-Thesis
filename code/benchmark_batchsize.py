"""
Batch Size Benchmarking Script for YOLO Training.

Measures training throughput (images/sec), epoch latency, peak GPU VRAM,
and host system RAM across candidate batch sizes under exact model configurations.

Runs each candidate batch size in an isolated spawned process to:
1. Prevent CUDA memory fragmentation and cache accumulation between trials.
2. Catch Out-of-Memory (OOM) errors gracefully without crashing the benchmark.
3. Accurately measure true peak process and worker memory footprint.

Usage example:
    uv run python code/benchmark_batchsize.py --batch-sizes 4 8 12 16 20 24 32
"""

from __future__ import annotations

import argparse
import csv
import datetime
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

import psutil
import torch
import yaml

# Set up logger for the benchmark harness
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("batch_benchmark")


def _get_system_ram_gb() -> float:
    """Return total system RAM in GiB."""
    return psutil.virtual_memory().total / (1024**3)


def _get_gpu_info(device_idx: int = 0) -> dict[str, Any]:
    """Return GPU name and total VRAM in GiB if CUDA is available."""
    if not torch.cuda.is_available():
        return {"name": "CPU", "total_vram_gb": 0.0, "cuda_available": False}
    name = torch.cuda.get_device_name(device_idx)
    total_mem = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)
    return {"name": name, "total_vram_gb": total_mem, "cuda_available": True}


def _worker_train_test(
    config: dict[str, Any],
    result_queue: mp.Queue,
) -> None:
    """Target function executed in an isolated child process for one batch size test."""
    batch_size = config["batch_size"]
    model_cfg = config["model_cfg"]
    weights = config["weights"]
    data_yaml = config["data_yaml"]
    imgsz = config["imgsz"]
    workers = config["workers"]
    warmup_epochs = config["warmup_epochs"]
    test_epochs = config["test_epochs"]
    device = config["device"]
    benchmark_dir = Path(config["benchmark_dir"])

    total_epochs = warmup_epochs + test_epochs
    run_name = f"bs_{batch_size}"

    # Disable cuDNN benchmark autotuning to match production pipeline setting
    torch.backends.cudnn.benchmark = False

    process = psutil.Process(os.getpid())
    peak_ram_bytes = 0

    def sample_memory() -> int:
        nonlocal peak_ram_bytes
        try:
            mem = process.memory_info().rss
            for child in process.children(recursive=True):
                try:
                    mem += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            if mem > peak_ram_bytes:
                peak_ram_bytes = mem
        except Exception:
            pass
        return peak_ram_bytes

    try:
        from ultralytics import YOLO

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        sample_memory()

        # Instantiate model
        model = YOLO(model_cfg)
        if weights and Path(weights).exists():
            model.load(weights)

        # Track timing for warmup and test epochs
        epoch_durations: list[float] = []

        class TimingCallback:
            def __init__(self):
                self.epoch_start_time = 0.0

            def on_train_epoch_start(self, trainer):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self.epoch_start_time = time.perf_counter()
                sample_memory()

            def on_train_epoch_end(self, trainer):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                dur = time.perf_counter() - self.epoch_start_time
                epoch_durations.append(dur)
                sample_memory()

            def on_train_batch_end(self, trainer):
                sample_memory()

        timing_cb = TimingCallback()
        model.add_callback("on_train_epoch_start", timing_cb.on_train_epoch_start)
        model.add_callback("on_train_epoch_end", timing_cb.on_train_epoch_end)
        model.add_callback("on_train_batch_end", timing_cb.on_train_batch_end)

        # Train model for total_epochs
        model.train(
            data=str(data_yaml),
            epochs=total_epochs,
            batch=batch_size,
            imgsz=imgsz,
            workers=workers,
            device=device,
            project=str(benchmark_dir),
            name=run_name,
            exist_ok=True,
            save=False,
            plots=False,
            val=False,
            verbose=False,
        )

        sample_memory()

        # Extract timing from measured test epochs (discarding warmup)
        if len(epoch_durations) >= total_epochs:
            measured_epochs = epoch_durations[warmup_epochs:]
        else:
            measured_epochs = epoch_durations

        avg_epoch_time = (
            sum(measured_epochs) / len(measured_epochs) if measured_epochs else 0.0
        )

        # Get dataset image count
        from ultralytics.data.utils import check_det_dataset

        dataset_dict = check_det_dataset(data_yaml)
        train_count = dataset_dict.get("train", None)
        num_images = 1200  # default FloW-img train count
        if isinstance(train_count, list):
            num_images = len(train_count)
        elif isinstance(train_count, str) and Path(train_count).exists():
            p = Path(train_count)
            if p.is_file():
                num_images = len([line for line in p.read_text().splitlines() if line.strip()])
            elif p.is_dir():
                num_images = len(list(p.rglob("*.jpg")) + list(p.rglob("*.png")))

        throughput_img_per_sec = (
            (num_images / avg_epoch_time) if avg_epoch_time > 0 else 0.0
        )
        steps_per_epoch = (num_images + batch_size - 1) // batch_size
        step_latency_ms = (
            (avg_epoch_time / steps_per_epoch * 1000) if steps_per_epoch > 0 else 0.0
        )

        peak_vram_alloc_mb = 0.0
        peak_vram_res_mb = 0.0
        if torch.cuda.is_available():
            peak_vram_alloc_mb = torch.cuda.max_memory_allocated() / (1024**2)
            peak_vram_res_mb = torch.cuda.max_memory_reserved() / (1024**2)

        peak_ram_mb = peak_ram_bytes / (1024**2)

        result_queue.put(
            {
                "success": True,
                "batch_size": batch_size,
                "throughput_img_per_sec": round(throughput_img_per_sec, 2),
                "avg_epoch_time_s": round(avg_epoch_time, 3),
                "step_latency_ms": round(step_latency_ms, 2),
                "peak_vram_alloc_mb": round(peak_vram_alloc_mb, 1),
                "peak_vram_res_mb": round(peak_vram_res_mb, 1),
                "peak_ram_mb": round(peak_ram_mb, 1),
                "measured_epochs": [round(t, 3) for t in measured_epochs],
                "error": None,
            }
        )

    except torch.cuda.OutOfMemoryError as e:
        result_queue.put(
            {
                "success": False,
                "batch_size": batch_size,
                "throughput_img_per_sec": 0.0,
                "avg_epoch_time_s": 0.0,
                "step_latency_ms": 0.0,
                "peak_vram_alloc_mb": 0.0,
                "peak_vram_res_mb": 0.0,
                "peak_ram_mb": round(peak_ram_bytes / (1024**2), 1),
                "measured_epochs": [],
                "error": f"CUDA Out Of Memory: {e}",
            }
        )
    except Exception as e:
        error_msg = str(e)
        if "out of memory" in error_msg.lower():
            error_msg = f"CUDA Out Of Memory: {error_msg}"
        result_queue.put(
            {
                "success": False,
                "batch_size": batch_size,
                "throughput_img_per_sec": 0.0,
                "avg_epoch_time_s": 0.0,
                "step_latency_ms": 0.0,
                "peak_vram_alloc_mb": 0.0,
                "peak_vram_res_mb": 0.0,
                "peak_ram_mb": round(peak_ram_bytes / (1024**2), 1),
                "measured_epochs": [],
                "error": error_msg,
            }
        )


def run_benchmark_for_batch(config: dict[str, Any], timeout_s: int = 300) -> dict[str, Any]:
    """Run single batch test in isolated process with safety timeout and queue collection."""
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    proc = ctx.Process(
        target=_worker_train_test,
        args=(config, result_queue),
    )
    proc.start()
    proc.join(timeout=timeout_s)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
        return {
            "success": False,
            "batch_size": config["batch_size"],
            "throughput_img_per_sec": 0.0,
            "avg_epoch_time_s": 0.0,
            "step_latency_ms": 0.0,
            "peak_vram_alloc_mb": 0.0,
            "peak_vram_res_mb": 0.0,
            "peak_ram_mb": 0.0,
            "measured_epochs": [],
            "error": f"Process timed out after {timeout_s}s",
        }

    if not result_queue.empty():
        return result_queue.get()

    if proc.exitcode != 0:
        return {
            "success": False,
            "batch_size": config["batch_size"],
            "throughput_img_per_sec": 0.0,
            "avg_epoch_time_s": 0.0,
            "step_latency_ms": 0.0,
            "peak_vram_alloc_mb": 0.0,
            "peak_vram_res_mb": 0.0,
            "peak_ram_mb": 0.0,
            "measured_epochs": [],
            "error": f"Child process crashed with exit code {proc.exitcode}",
        }

    return {
        "success": False,
        "batch_size": config["batch_size"],
        "throughput_img_per_sec": 0.0,
        "avg_epoch_time_s": 0.0,
        "step_latency_ms": 0.0,
        "peak_vram_alloc_mb": 0.0,
        "peak_vram_res_mb": 0.0,
        "peak_ram_mb": 0.0,
        "measured_epochs": [],
        "error": "No result returned from worker process",
    }


def print_table(results: list[dict[str, Any]], total_vram_gb: float) -> None:
    """Print an aligned ASCII comparison table of benchmark results."""
    header = (
        f"{'Batch':>6} | "
        f"{'Speed (img/s)':>14} | "
        f"{'Epoch (s)':>10} | "
        f"{'Step (ms)':>10} | "
        f"{'Peak VRAM (MB)':>15} | "
        f"{'VRAM %':>8} | "
        f"{'Host RAM (MB)':>14} | "
        f"{'Status':<12}"
    )
    divider = "-" * len(header)

    print("\n" + divider)
    print(" BATCH SIZE BENCHMARK RESULTS")
    print(divider)
    print(header)
    print(divider)

    for r in results:
        bs = r["batch_size"]
        if r["success"]:
            speed = f"{r['throughput_img_per_sec']:.1f}"
            ep_time = f"{r['avg_epoch_time_s']:.2f}"
            step_time = f"{r['step_latency_ms']:.1f}"
            vram = f"{r['peak_vram_res_mb']:.0f}"
            vram_pct = (
                f"{(r['peak_vram_res_mb'] / (total_vram_gb * 1024) * 100):.1f}%"
                if total_vram_gb > 0
                else "N/A"
            )
            ram = f"{r['peak_ram_mb']:.0f}"
            status = "OK"
        else:
            speed = "N/A"
            ep_time = "N/A"
            step_time = "N/A"
            vram = "N/A"
            vram_pct = "N/A"
            ram = f"{r.get('peak_ram_mb', 0):.0f}" if r.get("peak_ram_mb") else "N/A"
            status = "OOM / Error"

        row = (
            f"{bs:>6} | "
            f"{speed:>14} | "
            f"{ep_time:>10} | "
            f"{step_time:>10} | "
            f"{vram:>15} | "
            f"{vram_pct:>8} | "
            f"{ram:>14} | "
            f"{status:<12}"
        )
        print(row)

    print(divider + "\n")


def save_results(
    results: list[dict[str, Any]],
    output_dir: Path,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    """Save benchmark results to structured YAML and CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    yaml_path = output_dir / f"batch_benchmark_{timestamp}.yaml"
    csv_path = output_dir / f"batch_benchmark_{timestamp}.csv"

    payload = {
        "metadata": metadata,
        "results": results,
    }
    with open(yaml_path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    fieldnames = [
        "batch_size",
        "success",
        "throughput_img_per_sec",
        "avg_epoch_time_s",
        "step_latency_ms",
        "peak_vram_alloc_mb",
        "peak_vram_res_mb",
        "peak_ram_mb",
        "error",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in fieldnames})

    return yaml_path, csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Quickly benchmark YOLO training throughput across batch sizes."
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[4, 8, 12, 16, 20, 24, 32],
        help="List of batch sizes to test (default: 4 8 12 16 20 24 32)",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="m",
        choices=["n", "s", "m", "l", "x"],
        help="YOLO26 model size (default: m)",
    )
    parser.add_argument(
        "--p2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use P2 small-object detection head (default: True)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size (default: 640)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/flow_img.yaml",
        help="Path to dataset YAML (default: data/flow_img.yaml)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Pretrained weights file (default: yolo26{size}.pt)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="DataLoader worker processes per trial (default: 4)",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=1,
        help="Warmup epochs before measurement (default: 1)",
    )
    parser.add_argument(
        "--test-epochs",
        type=int,
        default=2,
        help="Timed measurement epochs (default: 2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0" if torch.cuda.is_available() else "cpu",
        help="CUDA device index or 'cpu' (default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/benchmark",
        help="Directory to save benchmark summaries (default: runs/benchmark)",
    )

    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    data_yaml = (
        Path(args.data) if Path(args.data).is_absolute() else repo_root / args.data
    )
    if not data_yaml.exists():
        logger.error(f"Dataset YAML not found at: {data_yaml}")
        sys.exit(1)

    model_cfg = f"yolo26{args.size}-p2.yaml" if args.p2 else f"yolo26{args.size}.yaml"
    weights = args.weights or f"yolo26{args.size}.pt"
    if weights and not Path(weights).is_absolute():
        weights_path = repo_root / weights
        weights = str(weights_path) if weights_path.exists() else weights

    output_dir = (
        Path(args.output_dir)
        if Path(args.output_dir).is_absolute()
        else repo_root / args.output_dir
    )

    gpu_info = _get_gpu_info()
    sys_ram_gb = _get_system_ram_gb()

    logger.info("=" * 60)
    logger.info("YOLO BATCH SIZE BENCHMARK")
    logger.info("=" * 60)
    logger.info(f"Model: {model_cfg} (Pretrained: {weights})")
    logger.info(f"Image Resolution: {args.imgsz}x{args.imgsz}")
    logger.info(f"Dataset: {data_yaml}")
    logger.info(f"DataLoader Workers: {args.workers}")
    logger.info(f"Candidate Batch Sizes: {args.batch_sizes}")
    logger.info(f"Warmup / Test Epochs: {args.warmup_epochs} / {args.test_epochs}")
    logger.info(
        f"Hardware: GPU={gpu_info['name']} ({gpu_info['total_vram_gb']:.1f} GB VRAM) | Host RAM={sys_ram_gb:.1f} GB"
    )
    logger.info("=" * 60)

    results: list[dict[str, Any]] = []

    for bs in args.batch_sizes:
        logger.info(f"\n---> Testing batch size {bs} ...")
        trial_config = {
            "batch_size": bs,
            "model_cfg": model_cfg,
            "weights": weights,
            "data_yaml": str(data_yaml),
            "imgsz": args.imgsz,
            "workers": args.workers,
            "warmup_epochs": args.warmup_epochs,
            "test_epochs": args.test_epochs,
            "device": args.device,
            "benchmark_dir": str(output_dir / "temp_runs"),
        }

        res = run_benchmark_for_batch(trial_config)
        results.append(res)

        if res["success"]:
            logger.info(
                f"Batch {bs:2d} DONE: {res['throughput_img_per_sec']:.1f} img/s "
                f"({res['avg_epoch_time_s']:.2f}s/epoch, {res['step_latency_ms']:.1f}ms/step) "
                f"| Peak VRAM: {res['peak_vram_res_mb']:.0f} MB | Peak Host RAM: {res['peak_ram_mb']:.0f} MB"
            )
        else:
            logger.warning(f"Batch {bs:2d} FAILED: {res['error']}")

    # Clean up temporary training run outputs
    temp_dir = output_dir / "temp_runs"
    if temp_dir.exists():
        try:
            import shutil

            shutil.rmtree(temp_dir)
        except Exception:
            pass

    # Print summary table
    print_table(results, gpu_info["total_vram_gb"])

    # Determine recommended batch size (highest throughput among successful trials)
    successful_runs = [r for r in results if r["success"]]
    best_run = None
    if successful_runs:
        best_run = max(successful_runs, key=lambda x: x["throughput_img_per_sec"])
        logger.info("=" * 60)
        logger.info(
            f"RECOMMENDED OPTIMAL BATCH SIZE: {best_run['batch_size']} "
            f"({best_run['throughput_img_per_sec']:.1f} img/s, "
            f"Peak VRAM: {best_run['peak_vram_res_mb']:.0f} MB / {gpu_info['total_vram_gb'] * 1024:.0f} MB, "
            f"Peak RAM: {best_run['peak_ram_mb']:.0f} MB)"
        )
        logger.info("=" * 60)
    else:
        logger.error("No batch size completed successfully.")

    # Save results
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model": model_cfg,
        "weights": weights,
        "imgsz": args.imgsz,
        "workers": args.workers,
        "dataset": str(data_yaml),
        "gpu_name": gpu_info["name"],
        "total_vram_gb": round(gpu_info["total_vram_gb"], 2),
        "total_host_ram_gb": round(sys_ram_gb, 2),
        "recommended_batch_size": best_run["batch_size"] if best_run else None,
    }

    yaml_file, csv_file = save_results(results, output_dir, metadata)
    logger.info(f"Saved benchmark summary to:\n  {yaml_file}\n  {csv_file}")


if __name__ == "__main__":
    main()
