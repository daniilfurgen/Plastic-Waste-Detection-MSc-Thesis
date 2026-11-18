# Plastic Waste Detection Algorithm for Coasts and Rivers Using Neural Models

🇺🇦 [Українська версія](README.md)

_(Алгоритм детекції пластикових відходів на узбережжях і річках з використанням нейронних моделей)_

## Project Description

This project is a Master's thesis in Computer Science. Its goal is to improve the accuracy of automated detection of small plastic waste in aerial imagery of rivers and sea coasts when only a limited amount of labelled data is available.

The main method is semi-supervised training of a **YOLO26** detector with the **Mean Teacher** scheme: a teacher model labels unlabelled frames, a student learns from ground-truth and pseudo-labels, and confidence thresholds are set separately for each object size category (Size-Stratified Ground Thresholding, SSGT). Experiments were run on **FloW-img** (rivers) and **UCWD** (coasts, UAV imagery) with 10–100 % of labels, against a control trained without pseudo-labels.

## Academic Definitions

- **Goal of the study:** improving the accuracy of detecting small plastic waste in aerial imagery of water bodies when few annotated data are available, through semi-supervised training of a detector.
- **Object of the study:** the process of automated detection and monitoring of plastic waste in water bodies using computer vision systems.
- **Subject of the study:** semi-supervised learning methods (Mean Teacher, size-dependent thresholds) for YOLO26 detectors on small objects.

## Status of the Paper

| Document | File | Status and content |
| --- | --- | --- |
| Master's thesis | [ВКР-Магістр-Санжаров.typ](paper/ВКР-Магістр-Санжаров.typ) | Main text (in Ukrainian): semi-supervised YOLO26 training (Mean Teacher + SSGT) on FloW-img and UCWD. Compiled PDF: [paper/output/ВКР-Магістр-Санжаров.pdf](paper/output/ВКР-Магістр-Санжаров.pdf). |
| ОДММТ-2026 conference paper | [ОДММТ-2026-Санжаров.typ](paper/ОДММТ-2026-Санжаров.typ) | Earlier paper (in Ukrainian): comparison of SAHI and YOLO26's built-in mechanisms on FloW-img. Does not reflect the current method. |
| Abstract (Тези) | [Тези-Доповідь-Санжаров.typ](paper/Тези-Доповідь-Санжаров.typ) | Earlier abstract on the same SAHI/YOLO26 study. |
| Collection of master's works | [Санжаров-Збірка-Робіт-Магістрів.typ](paper/Санжаров-Збірка-Робіт-Магістрів.typ) | Collection containing the earlier SAHI/YOLO26 work. |

Main results of the thesis (from the abstract):

- Gain in $\text{mAP}$ over the control at 10 % labels: **+0.020** on FloW-img and **+0.039** on UCWD.
- With all labels, the gain on FloW-img shrinks to **+0.003**.
- YOLO26m on FloW-img: $\text{mAP}$ **43.9 %**, $\text{mAP}_{50}$ **87.2 %**.
- The slow teacher (EMA 0.9996) was the only component with a clear contribution.
- SAHI is not used in the main experiments: in earlier tests it performed worse than native YOLO26.

### Structure of the Thesis Text

[ВКР-Магістр-Санжаров.typ](paper/ВКР-Магістр-Санжаров.typ) includes the following files from `paper/sections/`:

- [title.typ](paper/sections/title.typ) — title page and requirements;
- [abstract.typ](paper/sections/abstract.typ) — abstract in Ukrainian and English;
- [introduction.typ](paper/sections/introduction.typ) — introduction;
- [section-1.typ](paper/sections/section-1.typ) — analysis of the problem and detection methods: water pollution, detector architectures, small objects, semi-supervised learning;
- [section-2.typ](paper/sections/section-2.typ) — methodology: datasets, adaptive Mean Teacher;
- [section-3.typ](paper/sections/section-3.typ) — implementation and results: software implementation, development and ablation study, evaluation;
- [conclusions.typ](paper/sections/conclusions.typ) — conclusions.

## Layout of `paper/`

```
paper/
├── ВКР-Магістр-Санжаров.typ        # main thesis text
├── ОДММТ-2026-Санжаров.typ         # conference paper (earlier)
├── Тези-Доповідь-Санжаров.typ      # abstract (earlier)
├── Санжаров-Збірка-Робіт-Магістрів.typ  # collection of works
├── template.typ                    # shared template and styles
├── sections/                       # thesis sections
├── figures/                        # PDF figures and mean_teacher.typ
├── exp-results/                    # CSV files with raw experiment numbers
├── output/                         # compiled PDF, DOCX and PPTX files
├── sources.bib, sources.yaml       # bibliography
└── *.csl                           # citation styles (DSTU 8302:2015, APA)
```

- `exp-results/` — tables produced by `code/ssod/export_results.py` (main comparison, settings, per-epoch curves, components, architectures, label fractions).
- `figures/` — figures produced by `code/plot_results.py` (`uv run python code/plot_results.py all`).

### Compilation

From the repository root (requires [Typst](https://typst.app/)):

```bash
typst compile paper/ВКР-Магістр-Санжаров.typ paper/output/ВКР-Магістр-Санжаров.pdf
```

The files `paper/*.typ` and `paper/sections/*.typ` are the text of the thesis; edit them deliberately.

## Semi-Supervised Learning Pipeline (`code/ssod/`)

The pipeline in [code/ssod/](code/ssod/) improves YOLO26 on FloW-img by also learning from unlabelled frames of river videos.

| Module | Role |
| --- | --- |
| [mean_teacher.py](code/ssod/mean_teacher.py) | `MeanTeacherTrainer`: warm-up, semi-supervised loop, teacher EMA, evaluation, checkpoints, CLI. |
| [pseudo_labeling.py](code/ssod/pseudo_labeling.py) | Online labelling at every step: two-view teacher inference, size-bin thresholds, per-box weights, histogram log. |
| [dataset.py](code/ssod/dataset.py) | `SSODDatasetHandler` and `SSODSampler`: labelled and unlabelled data, separate augmentations, per-epoch sample order. |
| [losses.py](code/ssod/losses.py) | `WeightedDetectionLoss`: YOLO detection loss with per-box weights. |
| [metrics.py](code/ssod/metrics.py) | COCO evaluation (`pycocotools`): AP/AR, AP by size, peak $F_1$. |
| [ablation.py](code/ssod/ablation.py) | Batches of short ablation runs; results in [ABLATION.md](code/ssod/ABLATION.md). |
| [export_results.py](code/ssod/export_results.py) | Export of results to `paper/exp-results/`. |
| [recompute_coco.py](code/ssod/recompute_coco.py) | Recomputes COCO metrics from saved `predictions.json` files. |

### Training Procedure

1. **Warm-up.** Train YOLO26 on labelled data only; the teacher and the student are both initialised from these weights.
2. **Initial score histograms.** The teacher labels a uniform sample of unlabelled frames; per-size-bin score histograms are built.
3. **Baseline (epoch 0).** Evaluate the starting teacher on the validation split.
4. **Each SSL epoch.** Draw a fresh uniform sample of unlabelled frames; at every step: pseudo-labelling by the teacher → one optimiser step for the student on GT + PL → EMA update of the teacher and of the histograms. At the end of the epoch the teacher is evaluated; `best.pt` is chosen by Ultralytics fitness: $0.1\cdot\text{mAP}_{50} + 0.9\cdot\text{mAP}_{50\text{-}95}$.

### Key Parameters (defaults in the code)

| Parameter | Value |
| --- | --- |
| Optimiser | AdamW, $lr_0 = 2\times10^{-4}$, $\beta = (0.937, 0.999)$, weight decay $5\times10^{-4}$ |
| LR schedule | cosine, down to 1 % of $lr_0$ |
| Teacher EMA $\alpha$ | 0.9996 (per step) |
| Unlabelled frames per epoch | 4,800 (fresh uniform sample of a pool of ≈27k) |
| Pseudo-label weights | soft, enabled |
| $\lambda_u$ | 1.0, no ramp-up |
| GT-only tail (`tail_ratio`) | 0 |
| Mosaic | GT only; off during the last 10 % of epochs |
| Candidate confidence / NMS | 0.001 / 0.5 |
| Input size / batch size | 640 / 16 |

The full description of the method, loss equations and settings is in the thesis, section 2 (`paper/sections/section-2.typ`, in Ukrainian).

### Running

```bash
uv run python code/ssod/mean_teacher.py --help        # trainer CLI
uv run python code/ssod/ablation.py --dry-run         # list ablation commands
```

---

## Licensing

- **Source code:** [MIT License](LICENSE).
- **Thesis text and materials:** [Creative Commons Attribution 4.0 International (CC-BY 4.0)](LICENSE-PAPER).
