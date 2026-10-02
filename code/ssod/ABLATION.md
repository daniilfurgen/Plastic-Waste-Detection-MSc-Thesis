# Mean Teacher ablation

[`ablation.py`](ablation.py) runs short, comparable Mean Teacher experiments to find out which
semi-supervised settings help YOLO26 on FloW-img and which hurt.

## Why

Long SSL runs (50–100 epochs) kept ending below their own starting point. The supervised warm-up
model scores 0.430 mAP50-95, and full SSL training slowly degraded it. Degradation is visible within
~10 epochs, so short runs are enough to compare settings, and many fit in one night.

## Adopted default configuration

Based on the batches below, the trainer's defaults are now:

| Setting | Default | Why |
|---|---|---|
| `--pl-images-per-epoch` | 2400 | Less pseudo-data per epoch was better at every step (27k → 4,800 → 2,400 frames). |
| `--pl-soft-weights` | on | Part of every best configuration since `pool`. |
| `--ema-decay` | 0.9996 | Removes most of the late drift; 0.9998 adds nothing. |

Earlier batches pin their original settings explicitly in `ablation.py` (EMA 0.999, hard labels,
full pool via `--pl-images-per-epoch -1`), so they reproduce exactly.

## How runs are kept comparable

Every run in a batch:

- starts from the same warm-up weights (`runs/mean_teacher/ssod-v3-50-50/weights/warmup_best.pt`),
  so there is no warm-up variance and no warm-up time. The exception is `fractions`, where each
  label fraction has its own warm-up, shared by that fraction's runs;
- trains for the same number of SSL epochs (default 15), so the cosine LR schedule and mosaic
  closing are identical;
- uses the same seed, and the standard FloW-img val split (so numbers stay comparable with other
  pipelines on this dataset);
- writes an **epoch 0** row: the starting Teacher evaluated before any SSL. Each run is judged by
  its change from that baseline.

Each batch changes one setting per run relative to a base configuration.

## Running

```bash
uv run python code/ssod/ablation.py                   # default batch: settle
uv run python code/ssod/ablation.py --batch fractions # label-fraction evaluation
uv run python code/ssod/ablation.py --batch pool      # an earlier batch
uv run python code/ssod/ablation.py --only cls-only   # a single run
uv run python code/ssod/ablation.py --dry-run         # print commands only
```

Runs execute one after another. Re-running the same command skips finished runs and resumes an
interrupted one (Ctrl+C pauses safely). A failed run is logged and the next one starts.
Don't run anything else that writes to `data/video_frames/labels/all` while a batch is running,
because every epoch rewrites the pseudo-labels there.

## Batches

### `pool`: how much pseudo-labelled data, and how to weight it

Output: `runs/mean_teacher/ablation/`. Started from full-pool training with hard labels, where
~27k pseudo-labelled video frames swamp 1,200 ground-truth images.

| Run | Setting | Question |
|---|---|---|
| `control` | full pool, hard labels | Reference: how fast does the original setup degrade? |
| `lambda-0.2` | full pool, pseudo loss × 0.2 | Is down-weighting the pseudo loss enough? |
| `cap` | 4,800 random frames per epoch | Does limiting pseudo data per epoch help? |
| `cap-soft` | `cap` + per-box soft weights | Does trusting confident boxes more help? |
| `cap-soft-strong` | `cap-soft` + strong augmentation | Does a weak/strong augmentation gap help? |

Result after 15 epochs (mAP50-95, baseline 0.430):

| Run | Best (epoch) | Last | Change |
|---|---|---|---|
| control | 0.438 (1) | 0.377 | −0.054 |
| lambda-0.2 | 0.433 (1) | 0.402 | −0.028 |
| cap | 0.444 (3) | 0.427 | −0.004 |
| **cap-soft** | **0.446 (5)** | **0.433** | **+0.003** |
| cap-soft-strong | 0.442 (5) | 0.418 | −0.012 |

Takeaways: capping the pool is the main fix, and soft weights add a little on top. Strong
augmentation hurts small objects. All capped runs still peak around epoch 3–5 and then drift down.
The Teacher produces more and more confident pseudo-boxes per frame while val mAP falls.
These are single-seed runs; differences under ~0.005 may be noise.

### `drift`: what causes the late decline

Output: `runs/mean_teacher/ablation-drift/`. Each run is `cap-soft` with one change, and each tests
one suspected cause of the drift. The existing `cap-soft` run is shown in the summary as the
reference and is not re-run.

| Run | Change | Hypothesis tested |
|---|---|---|
| `lr-5e-5` | learning rate 2e-4 → 5e-5 | The Student learns too fast and drags the Teacher with it. |
| `ema-0.9996` | Teacher EMA 0.999 → 0.9996 | The Teacher follows the Student too quickly (~3 → ~7 epochs memory). |
| `cls-only` | pseudo-boxes train classification only | Regressing on the Teacher's own box coordinates feeds its localisation errors back. |

Result after 15 epochs (mAP50-95, baseline 0.430):

| Run | Best (epoch) | Last | Drop after peak | Change |
|---|---|---|---|---|
| cap-soft (ref) | 0.446 (5) | 0.433 | −0.013 | +0.003 |
| **ema-0.9996** | 0.444 (5) | **0.439** | **−0.005** | **+0.008** |
| lr-5e-5 | 0.435 (5) | 0.424 | −0.011 | −0.006 |
| cls-only | 0.436 (3) | 0.427 | −0.009 | −0.003 |

Takeaways:
- **A slower Teacher removes most of the drift.** It holds 0.440–0.444 from epoch 5 to 13, with the
  best final recall (0.853).
- **A lower learning rate only learns less.**
- **Classification-only pseudo-boxes lose the localisation gain.** mAP75 stays flat, so the
  pseudo-box regression is useful.
- **Every run dips slightly in the last two epochs.** This coincides with GT Mosaic closing, but
  `settle` shows Mosaic isn't the cause.

### `settle`: confirm the slow-Teacher configuration

Output: `runs/mean_teacher/ablation-settle/`. Each run is `cap-soft` with EMA 0.9996 (the best
`drift` run, shown as reference) plus one change, except the control.

| Run | Change | Question |
|---|---|---|
| `gt-only-control` | no pseudo-labels; GT repeated 6× per epoch | Is the SSL gain more than what extra fine-tuning with the same schedule and Teacher EMA gives? |
| `ema-0.9998` | Teacher EMA 0.9996 → 0.9998 | Does an even slower Teacher still help, or has it flattened? |
| `ema-0.9996-mosaic-on` | GT Mosaic never closed | Is the end-of-run dip caused by Mosaic closing? |
| `ema-0.9996-cap-2400` | 2,400 instead of 4,800 frames per epoch | Does even less pseudo-labelled data help? |

The control runs the same loop, LR schedule and Teacher EMA, with the same number of optimizer
steps per epoch. Pseudo-labelled images are replaced by repeated ground truth. Without it, an SSL
gain could simply come from extra training.

Result after 15 epochs (mAP50-95, baseline 0.430; "late mean" = mean of epochs 8–15):

| Run | Best (epoch) | Last | Late mean | Last vs. control |
|---|---|---|---|---|
| gt-only-control | 0.439 (8) | 0.431 | 0.436 | — |
| ema-0.9996 (ref) | 0.444 (5) | 0.439 | 0.441 | +0.007 |
| ema-0.9998 | 0.440 (11) | 0.439 | 0.439 | +0.007 |
| ema-0.9996-mosaic-on | 0.446 (8) | 0.435 | 0.441 | +0.004 |
| **ema-0.9996-cap-2400** | **0.446 (12)** | **0.444** | **0.445** | **+0.013** |

Takeaways:
- **Extra fine-tuning alone is worth part of the earlier gain.** The control climbs to 0.439 before
  falling back. Against it, SSL with a 4,800-frame cap is worth only +0.004 to +0.007 at 100% labels.
- **A 2,400-frame cap is the best configuration.** It's the only run that rises steadily and holds
  (0.446 from epoch 10 to 13), at +0.013 over the control at epoch 15 and +0.010 on the late mean.
  It's also the fastest SSL run (1.6 min/epoch).
- **EMA 0.9998 adds nothing over 0.9996.**
- **Mosaic closing doesn't cause the end-of-run dip.** The Mosaic-on run and the GT-only control dip
  just as much in the last epochs, so the cause is common to the schedule.

### `fractions`: SSL gain vs. amount of labelled data

Output: `runs/mean_teacher/ablation-fractions/`. This is the standard SSOD evaluation protocol:
train with 10%, 25% and 50% of the FloW-img train labels, each with and without pseudo-labels. The
100% runs come from `drift` and `settle` and are shown as references.

How it's set up:

- **Subsets.** The GT subsets are nested random samples (10% ⊂ 25% ⊂ 50%, seed 0), written to
  `data/flow_img/splits/` with a matching `data/flow_img_<N>pct.yaml`. Val is unchanged.
- **Warm-ups.** Each fraction gets its own supervised warm-up, run first. Every warm-up gets the
  same number of optimizer steps as the full-label one (~3,750: 469, 197 and 99 epochs). Its
  epoch-0 row is the supervised baseline for that fraction.
- **`<N>pct-ssl`.** Uses the best configuration so far (`cap-soft` + EMA 0.9996). GT is repeated
  1/fraction times per epoch, so the GT:pseudo sample ratio matches the 100% run.
- **`<N>pct-gt-only`.** The matching control: no pseudo-labels, and GT repeated to give the same
  number of steps per epoch as the SSL run.

The batch took roughly 6 h. The 10% warm-up stopped early at epoch 321 of 469, because Ultralytics
stops after 100 epochs without improvement; that's the same selection rule as the other warm-ups.

Result after 15 epochs (mAP50-95; SSL used the 4,800-frame cap; 100% rows are the `drift` and
`settle` references):

| GT labels | Supervised (epoch 0) | GT-only control | SSL | SSL − control | SSL − control, late mean |
|---|---|---|---|---|---|
| 10% | 0.341 | 0.334 | **0.360** | +0.025 | +0.021 |
| 25% | 0.378 | 0.373 | **0.397** | +0.024 | +0.021 |
| 50% | 0.412 | 0.408 | **0.421** | +0.013 | +0.011 |
| 100% | 0.430 | 0.431 | 0.439 | +0.007 | +0.005 |

Takeaways:
- **The gain comes from the pseudo-labels, not from fine-tuning.** All step-matched controls end at
  or below their baseline, because repeated GT overfits. The SSL and control curves separate from
  epoch 2 onwards and never cross.
- **The gain grows as labels shrink,** as expected for SSOD. At 10% labels, recall rises from 0.717
  to 0.755 and `mAP_s` from 0.237 to 0.271 over the control.
- **SSL saves labels.** With 50% of the labels, SSL reaches 0.4295 at its best epoch, essentially
  the full-label supervised baseline (0.430). The final epoch is 0.421.
- **Single seed.** The 10–25% gaps (~0.02, consistent across epochs) are robust. Differences of
  ~0.005 at 100% are borderline.
- **These runs used the 4,800-frame cap.** Re-running them with the cap chosen by the architecture
  batches would put every point on the curve on the final configuration.

## Settings referenced above

| Flag | Meaning |
|---|---|
| `--pl-images-per-epoch N` | Pseudo-label a fresh random sample of N unlabeled frames each epoch (default 2,400) instead of all ~27k. Sampling whole images keeps the small/medium/large box mix unchanged. `0` disables pseudo-labels (GT-only control); `-1` uses the whole pool. |
| `--gt-repeat k` | Repeat the GT part of each epoch (except the 25% tail) k times. |
| `--gt-yaml` | Ground-truth dataset YAML (used for the label-fraction subsets). |
| `--pl-soft-weights` (default on) | Weight each pseudo-box's loss by its confidence relative to its size bin: 0 at the bin's threshold, 1 at the bin's typical high score. Small objects are judged against small-object scores. Boxes near the threshold get ~0 weight and are effectively ignored, not treated as background. |
| `--pl-strong-aug` | Stronger colour jitter, blur and random erasing on pseudo-labelled images only. |
| `--unsupervised-weight λ` | Multiply the pseudo-label loss by λ. |
| `--pl-cls-only` | Pseudo-boxes contribute to the classification loss only, with no box regression. Ground-truth boxes are unaffected. |
| `--lr0`, `--ema-decay` | Student learning rate and Teacher EMA decay (default 0.9996). |
| `--close-mosaic-ratio r` | Fraction of final epochs with GT Mosaic off (default 0.1; 0 keeps it on throughout). |

## Output

Each batch directory contains one folder per run and two shared files:

- **`summary.csv`**, updated after every run, with these columns:
  - `baseline_mAP50-95`: epoch 0
  - `last_mAP50-95` and `delta_last`: final epoch, and its change from the baseline
  - `best_mAP50-95` and `best_epoch`
  - `last_recall` and `last_mAP_s` (small objects)
  - `last_pl_boxes`: pseudo-boxes used in the final epoch
  - `mean_epoch_min`: minutes per epoch
- **`ablation.log`**: what was run, skipped, resumed or failed.

Each run folder holds:

- `results.csv`: per-epoch metrics, pseudo-label counts, size-bin shares and mean weights;
- `train.log` and `stdout.log`;
- `weights/best.pt`;
- `candidates/`: raw per-epoch Teacher detections as Parquet, for offline analysis.

Compare runs on val metrics only. Train loss isn't comparable across settings that change the loss
(e.g. `cls-only` drops the pseudo-box regression terms).
