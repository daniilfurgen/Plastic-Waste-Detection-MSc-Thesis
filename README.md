# Алгоритм детекції пластикових відходів на узбережжях і річках з використанням нейронних моделей

_(Plastic Waste Detection Algorithm for Coasts and Rivers Using Neural Models)_

## Опис проекту / Project Description

**UA:** Цей проект є магістерською дисертацією в галузі комп'ютерних наук. Він присвячений розробці інтелектуального алгоритму на основі глибокого навчання для автоматизованого виявлення пластикових відходів у водному середовищі. Основна увага приділяється оптимізації детекції малорозмірних об'єктів та адаптації моделей для роботи на периферійних пристроях (Edge Computing).

**EN:** This project is a Master's thesis in Computer Science. It is dedicated to the development of an intelligent deep learning-based algorithm for automated plastic waste detection in aquatic environments. The primary focus is on optimizing small object detection and adapting models for Edge Computing.

---

## Академічні визначення / Academic Definitions

### Українською (UA)

- **Мета роботи:** підвищення ефективності екологічного моніторингу водних ресурсів шляхом розробки методу автоматизованого виявлення пластикових відходів на аерофотознімках, адаптованого до роботи з малорозмірними об'єктами в умовах обмежених обчислювальних ресурсів (Edge Computing).
- **Об’єкт дослідження:** процес автоматизованого аналізу візуальних даних (аерофотознімків) для екологічного моніторингу стану водних поверхонь.
- **Предмет дослідження:** методи та алгоритми глибокого навчання (Deep Learning) для детекції малорозмірних об'єктів із застосуванням підходів фрагментації зображень (slicing) та оптимізації інференсу.

### English (EN)

- **Goal of the study:** enhancing the efficiency of environmental monitoring of water resources by developing a method for automated plastic waste detection in aerial photographs, adapted for small-sized objects under conditions of limited computational resources (Edge Computing).
- **Object of the study:** the process of automated analysis of visual data (aerial photographs) for environmental monitoring of water surfaces.
- **Subject of the study:** Deep Learning methods and algorithms for small object detection using image fragmentation (slicing) and inference optimization approaches.

---

## Semi-Supervised Object Detection (SSOD) Pipeline

The pipeline in [code/ssod/](code/ssod/) improves a YOLO26 detector on FloW-img by also learning from unlabeled river video frames. It uses a **Mean Teacher** scheme: a Teacher model labels the unlabeled frames, a Student model trains on ground-truth (GT) plus these pseudo-labels (PL), and the Teacher is updated as a slow moving average of the Student.

### Components

| Module | Role |
| --- | --- |
| [mean_teacher.py](code/ssod/mean_teacher.py) | `MeanTeacherTrainer`: supervised warm-up, the SSL loop, Teacher EMA, per-epoch evaluation, checkpointing, pause/resume and the CLI. |
| [pseudo_labeling.py](code/ssod/pseudo_labeling.py) | Generates pseudo-labels with the Teacher: 2-view inference, per-size-bin thresholds and per-box weights. |
| [dataset.py](code/ssod/dataset.py) | `SSODDatasetHandler` and `SSODSampler`: loads GT and the PL pool with separate augmentations and orders samples within each epoch. |
| [losses.py](code/ssod/losses.py) | `WeightedDetectionLoss`: the YOLO detection loss with per-box importance weights. |
| [metrics.py](code/ssod/metrics.py) | COCO evaluation (`pycocotools`): AP/AR, AP by object size (`mAP_s/m/l`) and peak $F_1$. |
| [ablation.py](code/ssod/ablation.py) | Runs batches of short, comparable experiments. Results and rationale are in [ABLATION.md](code/ssod/ABLATION.md). |

### Training procedure

1. **Supervised warm-up.** Train the model on GT only with standard Ultralytics training, or load existing warm-up weights. Teacher and Student both start from these weights.
2. **Epoch-0 baseline.** Evaluate the starting Teacher on the GT validation split. Every SSL run is judged by its change from this baseline.
3. **For each SSL epoch:**
   1. **Pseudo-labeling.** Draw a fresh uniform sample of unlabeled frames and run the current Teacher on them (see below).
   2. **Student training.** Train one epoch on GT + PL with the weighted loss.
   3. **EMA update.** Update the Teacher from the Student after _every_ optimizer step.
   4. **Evaluation.** Evaluate the Teacher on the GT validation split and log Ultralytics and COCO metrics plus PL statistics (counts, size-bin shares, mean weights). `best.pt` is chosen by Ultralytics fitness:

$$
\text{fitness} = 0.1 \cdot \text{mAP}_{50} + 0.9 \cdot \text{mAP}_{50\text{-}95}
$$

### Pseudo-label generation

- **2-view inference.** The Teacher predicts on the original image and on its horizontal flip, with a very low candidate confidence ($0.001$). The flipped predictions are mapped back, and both views are merged with class-wise NMS (IoU $0.5$).
- **Size bins.** Candidates are grouped by COCO area bins: small $< 32^2$, medium $32^2$–$96^2$, large $\ge 96^2$ px².
- **Ground threshold per bin.** For each bin, build a histogram of candidate scores over the whole score range (21 bins, moving-average smoothing of width 3). The two highest peaks are the _noise mode_ (near the confidence floor) and the _object mode_. The threshold $\tau_b$ is the lowest point (valley) between them, and $m_b$ is the object mode. If there is no valley, $\tau_b = 0.4$ and $m_b$ is the 90th percentile of the scores above it.
- **Soft per-box weights** (on by default). A kept box with score $s \ge \tau_b$ gets

$$
w = \operatorname{clip}\!\left(\frac{s - \tau_b}{m_b - \tau_b},\ 0,\ 1\right)
$$

  so a box at the threshold is weighted 0 and a box at or above the bin's object mode is weighted 1. Each box is judged against the scores typical for its own size, so small objects aren't penalised for having generally lower confidence. The weight is stored as a 6th column in the YOLO label file.

### Loss function

The Student uses the standard YOLO detection loss (CIoU box, BCE classification and DFL terms), with each foreground term multiplied by its box's weight $w_j$ ($w_j = 1$ for GT boxes). For anchor $a$ assigned to box $j(a)$ with target score $t_a$:

$$
\mathcal{L}_{cls} = \frac{1}{T}\sum_a w_{j(a)}\,\mathrm{BCE}(p_a, t_a), \qquad
\mathcal{L}_{box} = \frac{1}{T}\sum_{a \in \mathrm{fg}} w_{j(a)}\, t_a\,\bigl(1 - \mathrm{CIoU}_a\bigr), \qquad
T = \max\Bigl(\sum_a t_a,\ 1\Bigr)
$$

DFL is weighted the same way as $\mathcal{L}_{box}$. Background anchors keep weight 1 and the normaliser $T$ is unchanged. This means a box with $w = 0$ acts as an **ignore region**: its anchors are pushed neither towards the box nor towards background. Without weights, the loss is identical to Ultralytics' `v8DetectionLoss`.

For end-to-end YOLO26, the one-to-many and one-to-one branches are combined with gains fixed at the values Ultralytics reaches at the end of supervised training:

$$
\mathcal{L}_{det} = 0.1\,\mathcal{L}_{o2m} + 0.9\,\mathcal{L}_{o2o}
$$

An optional unsupervised weight $\lambda_u$ (default 1, with an optional linear ramp-up over $E_r$ epochs) scales the pseudo-labeled images in a batch:

$$
\mathcal{L} = \frac{\mathcal{L}_{GT} + \lambda_u \mathcal{L}_{PL}}{n_{GT} + \lambda_u n_{PL}}, \qquad
\lambda_u^{(e)} = \lambda_u \cdot \min\!\left(1, \frac{e+1}{E_r}\right)
$$

### Teacher EMA

$$
\theta_T \leftarrow \alpha\,\theta_T + (1 - \alpha)\,\theta_S, \qquad \alpha = 0.9996
$$

The update runs after every optimizer step. Non-floating-point buffers are copied directly. The Teacher's time constant is $1/(1-\alpha) = 2{,}500$ steps, which is roughly 10 epochs at the default settings. A slower Teacher removed most of the late-training drift seen with $\alpha = 0.999$.

### Epoch composition and augmentation

- **Sample ordering** (`SSODSampler`). Each epoch, 25% of GT images are randomly chosen and placed at the **tail** of the epoch. The other 75% (repeated `gt_repeat` times, default 1) are shuffled together with all PL images to form the **head**:
  `[ shuffle(gt_repeat × 75% GT + PL) ] + [ 25% GT ]`.
  Each epoch therefore ends on clean GT data.
- **PL cap.** 2,400 unlabeled frames are freshly sampled per epoch from the pool of ~27k. With ~1.2k GT images, this gives roughly a 1 : 2 GT : PL ratio. Sampling whole images keeps the small/medium/large box mix unbiased.
- **Augmentations.** HSV, scale, translate and horizontal flip are applied to both GT and PL. Mosaic is applied **to GT only** and is turned off for the last 10% of SSL epochs. All other augmentations are disabled. An optional strong Student augmentation for PL images adds stronger HSV, Gaussian blur and random erasing; boxes that are at least 50% erased are dropped.

### Core hyperparameters (defaults)

| Parameter | Value |
| --- | --- |
| Optimizer | AdamW, $lr_0 = 2\times10^{-4}$, $\beta = (0.937, 0.999)$, weight decay $5\times10^{-4}$ |
| LR schedule | Cosine per epoch, from $lr_0$ to $lr_0 \cdot 0.01$ |
| Gradient clipping | max norm 10 |
| Teacher EMA decay $\alpha$ | 0.9996 (per step) |
| PL frames per epoch | 2,400 (fresh uniform sample) |
| Candidate confidence / merge NMS IoU | 0.001 / 0.5 |
| Fallback threshold (no valley) | 0.4 |
| Soft PL weights | on |
| $\lambda_u$ | 1.0, no ramp-up |
| GT tail ratio | 0.25 |
| Mosaic closing ratio | 0.10 (GT only) |
| E2E gains (o2m / o2o) | 0.1 / 0.9 |
| Image size / batch size | 640 / 16 |

### Running

```bash
uv run python code/ssod/mean_teacher.py --help        # trainer CLI
uv run python code/ssod/pseudo_labeling.py --help     # standalone pseudo-labeling
uv run python code/ssod/ablation.py --dry-run         # list ablation commands
```

---

## Ліцензування / Licensing

**UA:**
Цей проект використовує подвійне ліцензування:

- **Програмний код:** Розповсюджується на умовах [MIT License](LICENSE).
- **Текст дисертації та матеріали:** Ліцензовано на умовах [Creative Commons Attribution 4.0 International (CC-BY 4.0)](LICENSE-PAPER).

**EN:**
This project uses a dual-licensing structure:

- **Source Code:** Licensed under the [MIT License](LICENSE).
- **Thesis Paper & Materials:** Licensed under the [Creative Commons Attribution 4.0 International (CC-BY 4.0)](LICENSE-PAPER).
