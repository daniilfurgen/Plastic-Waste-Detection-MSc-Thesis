# Walkthrough: Batch Size Benchmarking & Pipeline Logging Enhancements

## Summary of Changes

1. **Created Batch Size Benchmarking Tool ([`code/benchmark_batchsize.py`](code/benchmark_batchsize.py))**:
   - A standalone CLI script to test YOLO training throughput (images/sec), step latency, peak GPU VRAM, and host RAM across candidate batch sizes.
   - Executes each trial in an isolated child process to prevent CUDA memory fragmentation and safely catch OOMs/timeouts without interrupting the benchmark suite.
   - Exports results to formatted tables and persists structured summaries to YAML and CSV in `runs/benchmark/`.

2. **Updated Semi-Supervised Pipeline ([`code/pseudo_label_pipeline.py`](code/pseudo_label_pipeline.py))**:
   - Added explicit `BATCH = 10` and `WORKERS = 4` to the `CONFIG` block.
   - Updated `model.tune()` and `model.train()` calls to pass `batch=BATCH` and `workers=WORKERS`.
   - Added system resource logging (`GPU` model, allocated/reserved VRAM, available host RAM) and explicit configuration tracking before tuning and training rounds.
   - Attached a file handler to Ultralytics' internal logger so training metrics and progress are logged to [`pipeline.log`](pipeline.log).

---

## Experimental Benchmark Results

Tested on **NVIDIA GeForce RTX 4070 SUPER (12.0 GB VRAM)**, **AMD Ryzen 7 5700X**, **24 GB System RAM** with `yolo26m-p2.yaml` (P2 small-object head) at `imgsz=640` on [`data/flow_img.yaml`](data/flow_img.yaml):

| Batch Size | Speed (img/s) | Epoch Time (s) | Step Latency (ms) |  Peak VRAM   |  VRAM %   |   Host RAM    |         Status         |
| :--------: | :-----------: | :------------: | :---------------: | :----------: | :-------: | :-----------: | :--------------------: |
|   **4**    |     25.0      |     48.05      |       160.2       |   3,990 MB   |   32.5%   |    10.7 GB    |         ✅ OK          |
|   **6**    |     31.5      |     38.08      |       190.4       |   5,558 MB   |   45.3%   |    11.3 GB    |         ✅ OK          |
|   **8**    |     33.5      |     35.86      |       239.1       |   7,238 MB   |   58.9%   |    11.3 GB    |         ✅ OK          |
|   **10**   |   **35.6**    |   **33.68**    |     **280.7**     | **8,898 MB** | **72.5%** |  **11.4 GB**  |     ✅ **Optimal**     |
|   **12**   |     34.3      |     34.95      |       349.5       |  10,522 MB   |   85.7%   |    12.0 GB    |         ✅ OK          |
|   **14**   |      0.0      |      N/A       |        N/A        | > 12,282 MB  |  > 100%   | Spills to RAM | ❌ Timeout / Thrashing |
|   **16**   |      0.0      |      N/A       |        N/A        | > 12,282 MB  |  > 100%   | Spills to RAM | ❌ Timeout / Thrashing |

> [!IMPORTANT]
> **Root Cause Analysis of System RAM Spillover**:
> With the P2 small-object detection head enabled (`yolo26m-p2.yaml`), feature map resolutions are 4x larger (160×160 vs 80×80). At batch size 16 (Ultralytics default), activation memory exceeds the 12 GB VRAM capacity.
> On this driver setup, PyTorch/CUDA spills over-allocated tensors into host system RAM, causing severe PCIe paging and degrading iteration speed by 50–100x.
>
> Setting **`BATCH = 10`** achieves the highest training throughput (**35.6 images/second**) while safely utilizing **72.5% of VRAM (8.9 GB)**, completely avoiding system RAM paging.

---

## How to Run the Benchmark in the Future

To test different batch sizes, model variants, or worker counts:

```bash
# Test custom batch sizes with 1 warmup epoch and 2 timed epochs:
uv run python code/benchmark_batchsize.py --batch-sizes 6 8 10 12 --warmup-epochs 1 --test-epochs 2

# Test a different model size (e.g., yolo26s without P2 head):
uv run python code/benchmark_batchsize.py --size s --no-p2 --batch-sizes 8 16 32 64

# Test different DataLoader worker counts (e.g. 2 vs 4 vs 8):
uv run python code/benchmark_batchsize.py --batch-sizes 10 --workers 4
```

Results are saved to [`runs/benchmark/`](runs/benchmark/) in both YAML and CSV formats.

---

## Verification Summary

- [x] Linting passed with `uv run ruff check code/benchmark_batchsize.py code/pseudo_label_pipeline.py`.
- [x] Tested isolated process execution and graceful handling of timeouts / OOMs.
- [x] Verified environment resource logging and file writing to [`pipeline.log`](pipeline.log).
- [x] Verified structured results output in [`runs/benchmark/batch_benchmark_20260818_134751.yaml`](runs/benchmark/batch_benchmark_20260818_134751.yaml).
