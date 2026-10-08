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
| `--pl-images-per-epoch` | 2400 | The optimum for YOLO26 at full labels: 27k → 4,800 → 2,400 frames kept improving, and 1,200 loses the gain over the GT-only control (`architectures`). With 10–25% of the labels, 4,800 does slightly better (`fractions-2400`); that's left as future work. |
| `--pl-soft-weights` | on | Part of every best configuration since `pool`. |
| `--ema-decay` | 0.9996 | Removes most of the late drift; 0.9998 adds nothing. |

Earlier batches pin their original settings explicitly in `ablation.py` (EMA 0.999, hard labels,
full pool via `--pl-images-per-epoch -1`), so they reproduce exactly.

## How runs are kept comparable

Every run in a batch:

- starts from the same warm-up weights (`runs/mean_teacher/ssod-v3-50-50/weights/warmup_best.pt`),
  so there is no warm-up variance and no warm-up time. The exceptions are `fractions` and
  `fractions-2400`, where each label fraction has its own warm-up (shared by both batches), and
  `architectures`, where YOLO11 and YOLO12 have their own;
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
uv run python code/ssod/ablation.py --batch mechanisms      # leave-one-out of the PL mechanisms
uv run python code/ssod/ablation.py --batch add-one-in      # each component alone on a stripped base
uv run python code/ssod/ablation.py --batch seeds           # seed repeats of final / simple / minimal / gt-only
uv run python code/ssod/ablation.py --batch components      # leave-one-out from the tuned YOLO26m
uv run python code/ssod/ablation.py --batch fractions-2400  # label fractions, final configuration
uv run python code/ssod/ablation.py --batch fractions # label fractions, 4,800 cap
uv run python code/ssod/ablation.py --batch architectures && \
  uv run python code/ssod/ablation.py --batch architectures-control
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
- **These runs used the 4,800-frame cap.** `fractions-2400` repeats them on the final configuration.

### `architectures` and `architectures-control`: PL cap across YOLO26, YOLO11 and YOLO12

Outputs: `runs/mean_teacher/ablation-architectures/` (SSL) and
`runs/mean_teacher/ablation-architectures-control/` (GT-only controls). These choose the PL cap for
the final label-fraction runs, and compare YOLO26 with YOLO11 and YOLO12 under identical conditions:
the same data, warm-up budget (50 epochs) and SSL pipeline. The protocol is deliberately not tuned
per architecture.

| Run | Setting |
|---|---|
| `<arch>-cap4800`, `<arch>-cap2400`, `<arch>-cap1200` | adopted configuration (soft weights, EMA 0.9996) at each cap |
| `<arch>-gt-only` | no pseudo-labels; GT repeated 3× to match the steps of a 2,400-cap run |

- **YOLO26.** Uses the existing warm-up. Its 4,800 and 2,400 runs already exist with identical
  settings (`drift/ema-0.9996`, `settle/ema-0.9996-cap-2400`), so they're shown as references; only
  the 1,200 run and a 2,400-matched control are new.
- **YOLO11 and YOLO12.** Each gets a full-label supervised warm-up with the same data and
  Ultralytics arguments as the YOLO26 one (50 epochs). The warm-ups are stored in
  `runs/mean_teacher/warmups/` and shared by both batches.
- **Different loss path.** YOLO11 and YOLO12 aren't end-to-end models: they use the standard YOLO
  loss and NMS at inference.
- **Run order.** Run `architectures` first, so the warm-ups exist before the controls need them.

Result after 15 epochs (mAP50-95; "late mean" = mean of epochs 8–15; controls are step-matched to
the 2,400 cap):

| Model | Supervised (epoch 0) | GT-only control | Cap 4,800 | Cap 2,400 | Cap 1,200 |
|---|---|---|---|---|---|
| YOLO26m | 0.430 | 0.436 (late 0.436) | 0.439 (late 0.441) | **0.444 (late 0.445)** | 0.436 (late 0.436) |
| YOLO11m | 0.408 | **0.425 (late 0.423)** | 0.414 (late 0.417) | 0.420 (late 0.420) | 0.422 (late 0.421) |
| YOLO12m | 0.414 | **0.422 (late 0.422)** | 0.412 (late 0.415) | 0.416 (late 0.419) | 0.418 (late 0.419) |

Minutes per SSL epoch at caps 4,800 / 2,400 / 1,200: YOLO26 2.5 / 1.6 / 1.1, YOLO11
2.2 / 1.4 / 1.0, YOLO12 3.5 / 2.0 / 1.6.

Takeaways:
- **For YOLO26, 2,400 is the optimum.** It's the only cap that clearly beats the control: +0.009 at
  epoch 15 and +0.009 on the late mean. At 1,200 the gain disappears (0.4355 vs. the control's
  0.4356), and at 4,800 it's smaller and drifts. So pseudo-data volume has an inverted-U effect.
- **For YOLO11 and YOLO12, no cap beats the GT-only control.** Their ranking (1,200 ≥ 2,400 > 4,800)
  isn't an optimum: smaller caps only bring the run closer to the control, so less pseudo-data means
  less harm, not more gain.
- **Pseudo-label quality doesn't explain the difference.** On labelled val, each architecture's
  starting Teacher at its own first-epoch thresholds gives almost the same pseudo-label precision and
  recall (P 0.814–0.828, R 0.822–0.824; small boxes P 0.73–0.75 for all three).
- **Under the shared 50-epoch budget, YOLO11 and YOLO12 hadn't converged.**
  - They start lower (0.408 and 0.414 vs. 0.430), and their warm-ups were still improving at epoch 50
    (YOLO11's best epoch was its last).
  - Extra supervised fine-tuning is worth much more to them: their GT-only controls gain +0.017 and
    +0.008, against +0.005 for YOLO26. YOLO11's control is still rising at epoch 15.
  - The control sees ground truth 3× per epoch, while SSL runs see it once plus pseudo-labels, so an
    under-trained model gets more from the control.
- **YOLO26 comes out ahead under identical conditions.** It's the strongest supervised model at the
  same warm-up budget, the only one where SSL beats step-matched supervised fine-tuning, and the best
  result overall: 0.444, against 0.425 for the best YOLO11 run (its control).
- **Scope.** Whether SSL would help YOLO11 or YOLO12 after a longer, converged warm-up wasn't tested,
  because the point was a same-conditions comparison. For the thesis, state it as a limitation:
  "Under an equal 50-epoch warm-up budget, YOLO11m and YOLO12m had not yet converged, and additional
  supervised fine-tuning outperformed pseudo-labelling; whether SSL benefits them after full
  convergence was not investigated."
- **Controls are step-matched to the 2,400 cap only.** Cap 4,800 runs train ~1.7× more steps per
  epoch than the control and cap 1,200 runs ~0.75× as many. Compare each cap mainly within its row.
- **Single seed.** Gaps under ~0.005 (e.g. YOLO12's caps against each other) are within noise.

### `fractions-2400`: label fractions on the final configuration

Output: `runs/mean_teacher/ablation-fractions-2400/`. This repeats `fractions` with YOLO26 at the
adopted 2,400-frame cap, so every point on the label-efficiency curve uses the final configuration.

- **Warm-ups.** Reused from `fractions` (`runs/mean_teacher/ablation-fractions/<N>pct-warmup`),
  so the supervised baselines are identical and no warm-up time is spent.
- **`<N>pct-ssl`.** Cap 2,400, soft weights, EMA 0.9996. GT is repeated 1/fraction times (10, 4, 2),
  as in `fractions`.
- **`<N>pct-gt-only`.** No pseudo-labels; GT is repeated 34×, 14× and 7× to match the steps of the
  2,400-cap SSL runs.
- **100% references.** `settle/ema-0.9996-cap-2400` (SSL) and `architectures-control/yolo26-gt-only`
  (its 2,400-matched control).

The batch took about 2 h: ~1.5 min per SSL epoch and ~1.0 min per control epoch.

Result after 15 epochs (mAP50-95; "late mean" = mean of epochs 8–15):

| GT labels | Supervised (epoch 0) | GT-only control | SSL | SSL − control | SSL − control, late mean |
|---|---|---|---|---|---|
| 10% | 0.341 | 0.340 | **0.356** | +0.016 | +0.010 |
| 25% | 0.378 | 0.379 | **0.394** | +0.015 | +0.011 |
| 50% | 0.412 | 0.414 | **0.425** | +0.011 | +0.009 |
| 100% | 0.430 | 0.436 | **0.444** | +0.009 | +0.009 |

Takeaways:
- **SSL beats its step-matched control at every label fraction on the final configuration.** The
  gain grows as labels shrink. The curves are stable, with no post-peak drift at any fraction.
- **The SSL − control gaps are smaller than in `fractions`, partly because of the controls.** The
  2,400-matched controls repeat GT fewer times (34/14/7× vs. 60/24/12×), so they overfit less and
  score higher. Don't compare the gaps across the two batches; compare the SSL runs directly (below).

#### Cap vs. label fraction (open question)

Both batches share the same warm-ups, so their SSL runs are directly comparable:

| GT labels | Cap 4,800 (late mean / final) | Cap 2,400 (late mean / final) | Better cap |
|---|---|---|---|
| 10% | **0.362** / 0.360 | 0.354 / 0.356 | 4,800 (+0.007) |
| 25% | **0.399** / 0.397 | 0.393 / 0.394 | 4,800 (+0.006) |
| 50% | 0.425 / 0.421 | 0.425 / **0.425** | tie on late mean; 4,800 peaks at 0.4295 (epoch 8) then drifts |
| 100% | 0.441 / 0.439 | **0.445** / 0.444 | 2,400 (+0.005) |

- **The best amount of pseudo-data per epoch shrinks as labels grow.** At 10% and 25%, 4,800 is ahead
  at every sampled epoch: with only 120–300 unique labelled images, fresh pseudo-labelled frames are
  worth more than their noise.
- **Drift sets in sooner as the supervised Teacher gets stronger.** At 50%, the larger cap peaks
  higher but drifts like the 100% runs. At 100% the drift wins outright.
- **Not pursued in this thesis.** Choosing the cap per fraction on the val split would be optimistic,
  because the same split is used for reporting. Testing it properly needs more points (e.g. 10% at
  9,600) and seeds. The main results use the fixed 2,400 default, and this table is reported as a
  sensitivity analysis. How the optimal pseudo-data volume depends on label budget and Teacher
  strength is left for future work.

### `components`: what each part of the final configuration contributes

Output: `runs/mean_teacher/ablation-components/`. Every earlier batch started from our untuned
50-epoch warm-up (`auto` optimizer → AdamW; 0.431 COCO mAP50-95). The supervised YOLO26m from the SAHI
experiments (`code/sahi_experiments/yolo26m-FloW-plain.pt`) was hyperparameter-tuned and trained
with MuSGD, and scores **0.448** on the same val split and evaluator. That's slightly above our
best Mean Teacher run (0.446 best epoch / 0.444 final).

This batch therefore starts every run from the tuned model, so the final configuration is measured
against the strongest supervised baseline. Each run removes one component (leave-one-out):

| Run | Change from `final` | Component tested |
|---|---|---|
| `final` | — (cap 2,400, soft weights, EMA 0.9996) | reference |
| `gt-only` | no pseudo-labels; GT repeated 3× (2,400-matched steps) | pseudo-labels as a whole |
| `no-soft-weights` | hard pseudo-labels | soft per-box weights |
| `fast-ema` | Teacher EMA 0.999 | slow Teacher |
| `no-cap` | whole pool (~27k frames per epoch) | pseudo-label cap |

`no-cap` runs last because it's ~7× slower per epoch. Expect about 4 h in total.

Result after 15 epochs (mAP50-95, baseline 0.448; "late mean" = mean of epochs 8–15):

| Run | Best (epoch) | Last | Late mean | mAP_s, last |
|---|---|---|---|---|
| **gt-only** | 0.459 (6) | **0.453** | **0.456** | 0.310 |
| no-soft-weights | **0.461 (7)** | 0.447 | 0.452 | **0.322** |
| final | 0.458 (3) | 0.441 | 0.446 | 0.314 |
| fast-ema | 0.459 (1) | 0.430 | 0.432 | 0.305 |
| no-cap | 0.444 (1) | 0.410 | 0.413 | 0.281 |

Takeaways:
- **From the tuned model, pseudo-labels no longer beat the GT-only control.** `final` ends 0.012
  below it, 0.010 on the late mean. The control gains about the same as from the untuned warm-up
  (+0.005 at epoch 15). What changed is the pseudo-label contribution: a stronger Teacher's labels
  carry less the Student doesn't already know, while its errors still feed back.
- **Pseudo-labels still help small objects and hurt the rest.** At epoch 15, against the control,
  `mAP_s` is +0.004 (`final`) and +0.012 (`no-soft-weights`), while `mAP_m` is −0.021 and −0.015,
  and `mAP75` is lower for `final` (0.369 vs. 0.385).
- **The cap and the slow Teacher still matter.** Without the cap the run collapses (−0.037); EMA
  0.999 peaks at epoch 1 and drifts to 0.430. The drift signature is unchanged: pseudo-boxes per
  epoch grow (4.9k → 6.4k for `final`, → 7.9k for `fast-ema`) while val mAP falls.
- **Hard labels beat soft weights here (+0.006 late mean), the opposite of `pool` (−0.006).** Both
  gaps are at single-seed noise level, so neither shows whether soft weights help.
- **The best epochs (0.459–0.461) are within noise of each other** and are picked on the val split
  they're reported on, so they're optimistic.

Because SSL doesn't beat the control from this start, the batch can't measure what each component
contributes to an SSL gain. That's measured from the untuned warm-up in `mechanisms`.

### `mechanisms`: what the pseudo-labelling and sampling mechanisms contribute

Output: `runs/mean_teacher/ablation-mechanisms/`. Starts from the shared untuned warm-up (0.430),
where the final configuration beats its step-matched GT-only control by +0.009 (`architectures`).
Each run removes one mechanism (leave-one-out). Both existing runs are shown as references.

| Run | Change from `final` | Mechanism tested |
|---|---|---|
| `no-size-bins` | `--no-pl-size-bins`: one ground threshold (and object mode for soft weights) over all candidates instead of per COCO size bin | size-stratified ground thresholding |
| `single-view` | `--no-pl-multi-view`: Teacher infers on the normal view only, no flipped view | multi-view inference |
| `no-gt-tail` | `--tail-ratio 0`: all GT shuffled with the pseudo-labels; same GT samples per epoch | GT epoch tail |
| `no-soft-weights` | hard pseudo-labels | soft per-box weights |
| `final-rerun` | none | run-to-run noise of a single seed |
| `final (ref)` | `settle/ema-0.9996-cap-2400` | — |
| `gt-only (ref)` | `architectures-control/yolo26-gt-only` (2,400-matched) | pseudo-labels as a whole |

`final-rerun` repeats the reference with the same seed. How far it lands from `final (ref)` is the
smallest difference the other runs can be trusted to show. Expect about 2 h in total.

### `add-one-in`: what each component does on its own

Output: `runs/mean_teacher/ablation-add-one-in/`. `mechanisms` removes one component at a time, so
a component whose job another one already does looks useless there. Soft weights are the example:
in `pool` (EMA 0.999) they mainly slowed the decline after the peak (best epoch +0.002, last epoch
+0.006), and once the slow EMA and the 2,400 cap removed that decline, they showed nothing.

This batch starts from a stripped Mean Teacher that shares only the 2,400 cap with the final
configuration, then switches each component back on alone. Same untuned warm-up (0.430).

| Run | Configuration |
|---|---|
| `minimal` | cap 2,400; one shared threshold, normal view only, no GT tail, hard labels, EMA 0.999 |
| `plus-slow-ema` | `minimal` + EMA 0.9996 |
| `plus-soft-weights` | `minimal` + soft per-box weights |
| `plus-size-bins` | `minimal` + per-size-bin ground thresholds |
| `plus-multi-view` | `minimal` + flipped Teacher view |
| `plus-gt-tail` | `minimal` + 25% GT tail |
| `minimal-rerun` | `minimal` again: run-to-run noise |
| `final (ref)`, `final-rerun (ref)`, `gt-only (ref)` | everything on, twice; no pseudo-labels |

The slow EMA counts as a component here, with the original 0.999 as "off". With the fast Teacher
the stripped run is expected to peak early and then decline, so compare runs mainly on their best
epoch. Expect about 2.5 h in total.

### `seeds`: is the full configuration better than a simple one?

Output: `runs/mean_teacher/ablation-seeds/`. In `mechanisms` and `add-one-in`, two identical runs
differed by up to 0.0036 mAP50-95 at the best epoch, as much as any single component's effect. This
batch repeats the configurations the conclusions rest on with seeds 43 and 44 (`--seed`), from the
same untuned warm-up. The existing seed-42 runs complete each set.

| Config | Configuration | Seed-42 runs |
|---|---|---|
| `final` | cap 2,400, soft weights, size bins, multi-view, GT tail, EMA 0.9996 | `settle/ema-0.9996-cap-2400`, `mechanisms/final-rerun` |
| `simple` | cap 2,400 and EMA 0.9996 only (shared threshold, single view, no GT tail, hard labels) | `add-one-in/plus-slow-ema` |
| `minimal` | cap 2,400 only, EMA 0.999 | `add-one-in/minimal`, `add-one-in/minimal-rerun` |
| `gt-only` | no pseudo-labels, 2,400-matched steps | `architectures-control/yolo26-gt-only` |

Questions, compared on mean ± standard deviation over seeds:
- `final` vs. `simple`: do soft weights, size bins, multi-view and the GT tail together add
  anything, or only complexity? If they don't separate, the simpler pipeline is the better one.
- `simple` vs. `minimal`: what the slow Teacher adds, at the best epoch and after it.
- `simple` and `final` vs. `gt-only`: is the SSL gain larger than the seed spread?

`--seed` sets the per-epoch pseudo-label sample, the epoch ordering and, since this batch, the
global Python/NumPy/PyTorch RNGs that drive augmentation and DataLoader workers. The seed-42 runs
predate the global seeding, so they're independent samples, not exact repeats. Expect about 3 h.

## Exporting results for the thesis

```bash
uv run python code/ssod/export_results.py
```

[`export_results.py`](export_results.py) reads the run folders and writes raw-number CSVs into
`paper/exp-results/`. Formatting and labels are left to Typst. Tables whose runs don't exist yet are
skipped with a warning, so re-run it as batches finish.

All metrics in these files are COCO metrics (pycocotools). They can differ from the Ultralytics
mAP50-95 quoted in this README by about ±0.001.

| File | Contents |
|---|---|
| `ssod_main_comparison.csv` | Tuned supervised YOLO26m, its step-matched GT-only fine-tuning (`components/gt-only`, best epoch), naive self-training (best round) and Mean Teacher (`components/final`, best epoch): full COCO metrics. |
| `ssod_self_training_rounds.csv` | The naive self-training baseline per round. Its method is iterative self-training with hard pseudo-labels, not Noisy Student (no noise, no student growth). |
| `ssod_mt_settings_summary.csv` | One row per development run (`pool` → `architectures`, YOLO26m): its settings (cap, soft weights, EMA, variant) plus supervised, final, late-mean and best mAP50-95. |
| `ssod_mt_settings_curves.csv` | The same runs per epoch, in long format, for training-curve plots. |
| `ssod_mt_long_runs_curves.csv` | The 50–100-epoch runs before the ablation (v2 with the threshold and loss-gain bugs, v3 with the full pool). |
| `ssod_components.csv`, `ssod_components_curves.csv` | The `components` batch, including the change from `final`. |
| `ssod_mechanisms.csv`, `ssod_mechanisms_curves.csv` | The `mechanisms` batch with its `final` and `gt-only` references, including the change from `final`. |
| `ssod_add_one_in.csv`, `ssod_add_one_in_curves.csv` | The `add-one-in` batch with its references, including the change from `minimal`. |
| `ssod_seed_runs.csv`, `ssod_seed_summary.csv` | One row per seed run of `final`, `simple`, `minimal` and `gt-only` (best-epoch metrics, late mean, final), and their mean and standard deviation per configuration. |
| `ssod_architectures.csv` | Supervised, GT-only control and each cap for YOLO26m, YOLO11m and YOLO12m. |
| `ssod_label_fractions.csv` | Supervised, GT-only control and SSL at 10/25/50/100% labels, for both caps. |

In the curve files, `pl_cap` is 27302 for the whole pool and 0 for GT-only controls; epoch 0 is the
supervised starting point.

## Settings referenced above

| Flag | Meaning |
|---|---|
| `--pl-images-per-epoch N` | Pseudo-label a fresh random sample of N unlabeled frames each epoch (default 2,400) instead of all ~27k. Sampling whole images keeps the small/medium/large box mix unchanged. `0` disables pseudo-labels (GT-only control); `-1` uses the whole pool. |
| `--gt-repeat k` | Repeat the GT part of each epoch (except the 25% tail) k times. |
| `--gt-yaml` | Ground-truth dataset YAML (used for the label-fraction subsets). |
| `--model` | Architecture name recorded with the run; the weights themselves come from `--warmup-weights`. |
| `--pl-soft-weights` (default on) | Weight each pseudo-box's loss by its confidence relative to its size bin: 0 at the bin's threshold, 1 at the bin's typical high score. Small objects are judged against small-object scores. Boxes near the threshold get ~0 weight and are effectively ignored, not treated as background. |
| `--no-pl-size-bins` | Use one ground threshold (and one object mode) for all pseudo-boxes instead of one per COCO size bin. |
| `--no-pl-multi-view` | The Teacher pseudo-labels from the normal view only, without the horizontally flipped view. |
| `--tail-ratio r` | Fraction of GT placed, unmixed, at the end of each epoch (default 0.25; 0 mixes all GT with the pseudo-labels). |
| `--pl-strong-aug` | Stronger colour jitter, blur and random erasing on pseudo-labelled images only. |
| `--unsupervised-weight λ` | Multiply the pseudo-label loss by λ. |
| `--pl-cls-only` | Pseudo-boxes contribute to the classification loss only, with no box regression. Ground-truth boxes are unaffected. |
| `--seed n` | Seed for the per-epoch pseudo-label sample, epoch ordering and augmentation (default 42). |
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
