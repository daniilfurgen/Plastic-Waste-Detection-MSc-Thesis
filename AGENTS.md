# Project Context: Computer Science Master's Thesis

This repository contains the Master's Thesis project of Daniil Sanzharov. The research focuses on automated plastic waste detection in aquatic environments using deep learning, optimized for Edge Computing.

## Project Structure

- `code/`: Python scripts and Jupyter notebooks for training and evaluating Computer Vision models.
  - Primary models: YOLO26 (via Ultralytics).
  - Dataset: FloW-img (locally stored in `code/FloW-img/`, git-ignored).
  - `ssod/`: Semi-Supervised Object Detection (SSOD) pipeline components.
    - `dataset.py`: Dual dataset handler (`SSODDatasetHandler`) and epoch-aware sampler (`SSODSampler`). Supports simultaneous loading of annotated Ground-Truth (train & val) and dynamic Pseudo-Labeled pools, strict selective augmentations (HSV, scale, translate, fliplr on both; Mosaic on GT only, automatically closed during the last 10% of epochs; optional strong Student augmentation on pseudo-labeled images — stronger HSV, Gaussian blur, box-aware random erasing; all other augmentations disabled), dynamic in-memory / disk pseudo-label reloading, 25% GT tail / 75% GT+PL head epoch sequencing with optional GT head repetition (`gt_repeat`), per-box pseudo-label weights carried through augmentation (as a second `cls` column) into `pl_weight`, and `is_pseudo` batch collation.
    - `mean_teacher.py`: Semi-supervised Mean Teacher training pipeline (`MeanTeacherTrainer`). Supports supervised warm-up on Ground-Truth data, initializing Student from Teacher weights, epoch-0 baseline evaluation of the starting Teacher, per-step Teacher EMA updates, Student training using YOLO default detection loss via `WeightedDetectionLoss` (YOLO26 E2E one-to-one/one-to-many gains pinned to their end-of-warm-up values; optional per-box pseudo-label importance weights; optional unsupervised weighting $\lambda_u$ with linear ramp-up), optional fresh uniform per-epoch sample of the unlabeled pool (`pl_images_per_epoch`), per-epoch Teacher evaluation on GT validation split logging Ultralytics and COCO metrics plus pseudo-label counts, size-bin shares and mean weights, and checkpoint selection via Ultralytics fitness. Includes Typer CLI.
    - `losses.py`: `WeightedDetectionLoss`, a `v8DetectionLoss` whose foreground (classification, IoU, DFL) terms are importance-weighted per box by `batch["pl_weight"]`; background anchors and normalisation are unchanged, so weight-0 boxes act as ignore regions. Identical to the stock loss without weights.
    - `metrics.py`: Standalone COCO evaluation and prediction ID harmonization module (`evaluate_coco`, `fix_prediction_ids`) computing multi-scale AP (`mAP_s`, `mAP_m`, `mAP_l`), AR, and peak $F_1$ score via `pycocotools.cocoeval`.
    - `pseudo_labeling.py`: High-throughput pseudo-label generator featuring 2-view candidate inference (native + SIMD horizontal flip), batched FP16 Tensor Core execution with multi-threaded I/O prefetching, Polars candidate aggregation, Parquet export, COCO size-bin ground thresholding (valley between the noise and object modes of the full-range score histogram, with a fallback threshold when no valley exists), optional threshold-aware per-box weights (0 at the bin's threshold, rising linearly to 1 at the bin's object mode; written as a 6th label column), optional uniform image sampling, clean-slate annotation cleanup, and Typer CLI.
    - `utilities.py`: Project-wide structured logging (`get_logger`) and shared utilities.
- `tests/`: Automated unit and integration test suite covering SSOD augmentations, samplers, EMA updates, loss functions, COCO evaluation, and dataset lifecycles.
- `paper/`: Raw Typst files for the thesis paper and conference abstracts.
  - `ВКР-Магістр-Санжаров.typ`: Main thesis document.
  - `sources.bib`: Bibliography in BibLaTeX format.
- `runs/`: Output directory for model training and evaluation results.

## Technical Stack

- **Environment Management:** [uv](https://github.com/astral-sh/uv). Use `uv` for all dependency management and execution.
- **Python Version:** 3.12+
- **Core Dependencies:** `ultralytics`, `supervision`, `torch`, `torchvision`, `polars`, `typer`, `onnxruntime-gpu`.
- **Document Generation:** Typst.

## Licensing

- **Source Code:** [MIT License](LICENSE)
- **Thesis Paper & Materials:** [Creative Commons Attribution 4.0 International (CC-BY 4.0)](LICENSE-PAPER)

## Foundational Mandates & Instructions

- **Academic Integrity:** All Typst files (`paper/*.typ`) are **READ-ONLY**. You must not modify the content, structure, or arguments of the thesis. The only exception is proofreading or grammar correction, and even then, only when **explicitly** directed by the user for a specific section.
- **SAHI Framework:** Although SAHI (Slicing Aided Hyper Inference) is present in some legacy scripts and was tested, it **should not be used** for further development or production inference. Experimental results showed poor performance on this specific dataset compared to native YOLO26.
- **Dependency Management:** Always use `uv` (e.g., `uv add`, `uv run`, `uv lock`). Never use `pip` or `venv` directly. The project environment is located in `.venv/`.
- **Notebooks:** When working with notebooks in `code/`, ensure the kernel is set to the `.venv` created by `uv`.
- **Past-experiments:** `code/noisy_student_pipeline/` and `code/sahi_experiments` contain code for the past experiments that demonstrated inadequate results. They can be looked up for reference what was done before, but they should not be used as reference for style or design of the current and future code.
