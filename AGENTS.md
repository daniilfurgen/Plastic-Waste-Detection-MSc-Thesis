# Project Context: Computer Science Master's Thesis

This repository contains the Master's Thesis project of Daniil Sanzharov. The research focuses on automated plastic waste detection in aquatic environments using deep learning, optimized for Edge Computing.

## Project Structure

- `code/`: Python scripts and Jupyter notebooks for training and evaluating Computer Vision models.
  - Primary models: YOLO26 (via Ultralytics).
  - Dataset: FloW-img (locally stored in `code/FloW-img/`, git-ignored).
- `paper/`: Raw Typst files for the thesis paper and conference abstracts.
  - `ВКР-Магістр-Санжаров.typ`: Main thesis document.
  - `sources.bib`: Bibliography in BibLaTeX format.
- `runs/`: Output directory for model training and evaluation results.

## Technical Stack

- **Environment Management:** [uv](https://github.com/astral-sh/uv). Use `uv` for all dependency management and execution.
- **Python Version:** 3.12+
- **Core Dependencies:** `ultralytics`, `supervision`, `torch`, `torchvision`, `onnxruntime-gpu`.
- **Document Generation:** Typst.

## Licensing

- **Source Code:** [MIT License](LICENSE)
- **Thesis Paper & Materials:** [Creative Commons Attribution 4.0 International (CC-BY 4.0)](LICENSE-PAPER)

## Foundational Mandates & Instructions

- **Academic Integrity:** All Typst files (`paper/*.typ`) are **READ-ONLY**. You must not modify the content, structure, or arguments of the thesis. The only exception is proofreading or grammar correction, and even then, only when **explicitly** directed by the user for a specific section.
- **SAHI Framework:** Although SAHI (Slicing Aided Hyper Inference) is present in some legacy scripts and was tested, it **should not be used** for further development or production inference. Experimental results showed poor performance on this specific dataset compared to native YOLO26.
- **Dependency Management:** Always use `uv` (e.g., `uv add`, `uv run`, `uv lock`). Never use `pip` or `venv` directly. The project environment is located in `.venv/`.
- **Notebooks:** When working with notebooks in `code/`, ensure the kernel is set to the `.venv` created by `uv`.
