"""Figures of the results chapter (section 3) of the thesis.

Reads the per-epoch `results.csv` of the finished Mean Teacher runs (and the exported tables in
`paper/exp-results/`) and writes vector PDFs into `paper/figures/`. Every figure is 16 cm wide, the
text width of the thesis, and uses the same fonts and colours as `plot_size_distribution.py`.

Figures:
  label_efficiency    mAP vs. share of labels, SSL vs. GT-only control (FloW-img and UCWD)
  training_curves     mAP per epoch at 10% of the labels (FloW-img and UCWD)
  caps_images_seen    mAP vs. the number of pseudo-labelled images seen, PL caps 2400/4800/9600
  ssgt_histograms     score histograms and thresholds of the three COCO size bins
  ablation_forest     leave-one-out differences from the full configuration (100% and 10% of labels)
  architectures       SSL gain over the GT-only control per detector (needs the `arch-online` batch)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import typer
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs/mean_teacher"
FIGURES = ROOT / "paper/figures"
EXP_RESULTS = ROOT / "paper/exp-results"
SEEDS = (42, 43, 44)
LATE = range(11, 16)  # epochs averaged for the reported value
CM = 1 / 2.54
WIDTH_CM = 16
FONT = "STIX Two Text"
# Categorical slots validated for colour-vision deficiency (same as plot_size_distribution.py).
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6b6b6b"
INK = "#262626"
METRIC = "coco/mAP_50_95"
METRIC_S = "coco/mAP_s"

app = typer.Typer(add_completion=False, help=__doc__)


# ---- helpers ------------------------------------------------------------------------------


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": FONT,
            "font.size": 10,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,  # keep the PDF text selectable
            "axes.edgecolor": INK,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )


def comma(ax: plt.Axes, axis: str = "y", digits: int = 2) -> None:
    """Ukrainian decimal comma on tick labels."""
    formatter = FuncFormatter(lambda v, _: f"{v:.{digits}f}".replace(".", ","))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(formatter)


def load_run(path: Path) -> pl.DataFrame | None:
    file = path / "results.csv"
    return pl.read_csv(file) if file.is_file() else None


def load_runs(pattern: str, base: Path) -> list[pl.DataFrame]:
    """Runs `<pattern>-seed42..44` of a batch directory; missing seeds are skipped."""
    runs = [load_run(base / f"{pattern}-seed{seed}") for seed in SEEDS]
    return [run for run in runs if run is not None]


def curve(runs: list[pl.DataFrame], column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Epochs, mean and sample std over seeds of a per-epoch column."""
    epochs = runs[0]["epoch"].to_numpy()
    values = np.stack([run[column].to_numpy() for run in runs])
    std = values.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros(values.shape[1])
    return epochs, values.mean(axis=0), std


def late_mean(runs: list[pl.DataFrame], column: str = METRIC) -> tuple[float, float]:
    """Mean and std over seeds of each run's mean over the late epochs."""
    per_seed = [run.filter(pl.col("epoch").is_in(list(LATE)))[column].mean() for run in runs]
    return float(np.mean(per_seed)), float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0


def start_value(runs: list[pl.DataFrame], column: str = METRIC) -> float:
    return float(np.mean([run.filter(pl.col("epoch") == 0)[column][0] for run in runs]))


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"{name}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    typer.echo(f"Saved {out}")


# ---- label efficiency ---------------------------------------------------------------------

FLOW = RUNS / "ablation-online-fractions"
UCWD = RUNS / "ablation-ucwd-online-fractions"
FRACTIONS = (10, 25, 50, 100)


def flow_runs(pct: int, method: str) -> list[pl.DataFrame]:
    if pct == 100 and method == "ssl":
        return load_runs("cap-4800", RUNS / "ablation-online-caps")
    return load_runs(f"{pct}pct-{method}", FLOW)


def label_efficiency_table(runs_of, fractions) -> dict[int, dict[str, tuple[float, float]]]:
    table: dict[int, dict[str, tuple[float, float]]] = {}
    for pct in fractions:
        ssl, control = runs_of(pct, "ssl"), runs_of(pct, "gt-only")
        if not ssl or not control:
            continue
        table[pct] = {"ssl": late_mean(ssl), "gt-only": late_mean(control), "start": (start_value(ssl), 0.0)}
    return table


def ucwd_runs(pct: int, method: str) -> list[pl.DataFrame]:
    return load_runs(f"{pct}pct-{method}", UCWD)


@app.command()
def label_efficiency() -> None:
    """mAP (epochs 11-15, mean ± std over seeds) vs. the share of labels."""
    style()
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_CM * CM, 5.4 * CM + 1.2 * CM), sharex=True)
    ucwd_full = float(
        next(r for r in pl.read_csv(EXP_RESULTS / "ssod_ucwd.csv").iter_rows(named=True)
             if r["label_pct"] == 100 and r["method"] == "supervised")["map50_95"]
    )
    for ax, title, runs_of, fractions in (
        (axes[0], "FloW-img", flow_runs, FRACTIONS),
        (axes[1], "UCWD", ucwd_runs, (10, 25, 50)),
    ):
        table = label_efficiency_table(runs_of, fractions)
        xs = list(table)
        for key, colour, marker, label in (
            ("ssl", BLUE, "o", "Mean Teacher"),
            ("gt-only", ORANGE, "s", "Контроль без псевдоміток"),
            ("start", GREY, "_", "Початок (розігрів)"),
        ):
            mean = [table[x][key][0] for x in xs]
            std = [table[x][key][1] for x in xs]
            if key == "start":
                ax.plot(xs, mean, color=colour, linestyle=":", marker=marker, markersize=9, label=label)
            else:
                ax.errorbar(xs, mean, yerr=std, color=colour, marker=marker, markersize=5,
                            capsize=2.5, linewidth=1.4, label=label)
        if title == "UCWD":
            ax.plot([100], [ucwd_full], color=GREY, marker="_", markersize=9)
            ax.annotate("100 %: розігрів", (100, ucwd_full), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=8, color=GREY)
            ax.set_ylim(top=ucwd_full + 0.02)
        ax.set_xscale("log")
        ax.set_xlim(8, 125)
        ax.set_xticks([10, 25, 50, 100], ["10", "25", "50", "100"])
        ax.minorticks_off()
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Частка розміченої вибірки, %")
        comma(ax, digits=2)
    axes[0].set_ylabel("mAP (епохи 11–15)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "label_efficiency")


# ---- training curves ----------------------------------------------------------------------


@app.command()
def training_curves(pct: int = 10) -> None:
    """Per-epoch mAP of SSL and the GT-only control at `pct` % of the labels."""
    style()
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_CM * CM, 6.6 * CM))
    for ax, title, runs_of in ((axes[0], "FloW-img", flow_runs), (axes[1], "UCWD", ucwd_runs)):
        for method, colour, label in (("ssl", BLUE, "Mean Teacher"), ("gt-only", ORANGE, "Контроль без псевдоміток")):
            runs = runs_of(pct, method)
            if not runs:
                continue
            epochs, mean, std = curve(runs, METRIC)
            ax.plot(epochs, mean, color=colour, linewidth=1.5, label=label)
            ax.fill_between(epochs, mean - std, mean + std, color=colour, alpha=0.2, linewidth=0)
        ax.set_title(f"{title}, {pct} % розмітки", fontsize=11)
        ax.set_xlabel("Епоха")
        ax.set_xticks([0, 5, 10, 15])
        comma(ax, digits=3)
    axes[0].set_ylabel("mAP")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "training_curves")


# ---- pseudo-label cap ---------------------------------------------------------------------


@app.command()
def caps_images_seen() -> None:
    """mAP vs. cumulative pseudo-labelled images seen, for three PL caps (online SSGT)."""
    style()
    base = RUNS / "ablation-online-caps"
    fig, ax = plt.subplots(figsize=(WIDTH_CM * CM, 6.0 * CM))
    for cap, colour in ((2400, BLUE), (4800, ORANGE), (9600, AQUA)):
        runs = load_runs(f"cap-{cap}", base)
        if not runs:
            continue
        epochs, mean, std = curve(runs, METRIC)
        seen = np.cumsum(np.mean([run["pl_images"].to_numpy() for run in runs], axis=0)) / 1000
        ax.plot(seen, mean, color=colour, linewidth=1.5, marker="o", markersize=2.5,
                label=f"{cap:,}".replace(",", " ") + " кадрів на епоху")
        ax.fill_between(seen, mean - std, mean + std, color=colour, alpha=0.18, linewidth=0)
    ax.set_xlabel("Кількість побачених псевдорозмічених зображень, тис.")
    ax.set_ylabel("mAP")
    comma(ax, digits=3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "caps_images_seen")


# ---- SSGT histograms ----------------------------------------------------------------------


@app.command()
def ssgt_histograms(run: str = "ablation-online-caps/cap-4800-seed42") -> None:
    """Teacher score histograms per size bin at the start and the end of training, with thresholds."""
    style()
    run_dir = RUNS / run
    hist = pl.read_csv(run_dir / "histograms.csv")
    results = pl.read_csv(run_dir / "results.csv")
    last = int(hist["epoch"].max())
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_CM * CM, 5.2 * CM), sharey=True)
    for ax, size_bin, title in zip(axes, ("small", "medium", "large"), ("Малі", "Середні", "Великі")):
        for epoch, colour, label in ((0, GREY, "початок"), (last, BLUE, f"епоха {last}")):
            rows = hist.filter((pl.col("epoch") == epoch) & (pl.col("size_bin") == size_bin)).sort("bin")
            edges = np.append(rows["bin_lo"].to_numpy(), rows["bin_hi"].to_numpy()[-1])
            ax.stairs(rows["ema"].to_numpy(), edges, color=colour, linewidth=1.3, label=label)
        row = results.filter(pl.col("epoch") == last)
        threshold, mode = float(row[f"threshold/{size_bin}"][0]), float(row[f"object_mode/{size_bin}"][0])
        ax.axvline(threshold, color=ORANGE, linestyle="--", linewidth=1.1, label=r"поріг $\tau_b$")
        ax.axvline(mode, color=AQUA, linestyle="-.", linewidth=1.1, label=r"мода $m_b$")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Коефіцієнт впевненості")
        ax.set_xlim(0, 1)
        comma(ax, axis="x", digits=1)
    axes[0].set_ylabel("Частка кандидатів")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "ssgt_histograms")


# ---- component ablation -------------------------------------------------------------------

COMPONENTS = {
    "no-multi-view": "без дзеркального вигляду",
    "no-size-bins": "без розмірних категорій",
    "no-soft-weights": "без вагових коефіцієнтів",
    "fast-ema": "EMA 0,999 (швидкий вчитель)",
    "gt-only": "без псевдоміток",
}


def differences(batch: Path, configs: dict[str, str]) -> list[tuple[str, float, float]]:
    """Late-mean difference from `final` and its standard error (3 seeds, independent runs)."""
    final = late_mean(load_runs("final", batch))
    out = []
    for name, label in configs.items():
        runs = load_runs(name, batch)
        if not runs:
            continue
        mean, std = late_mean(runs)
        se = np.hypot(std, final[1]) / np.sqrt(len(runs))
        out.append((label, mean - final[0], float(se)))
    return out


@app.command()
def ablation_forest() -> None:
    """Late-mean mAP difference of each leave-one-out run from the full configuration."""
    style()
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_CM * CM, 6.0 * CM), sharex=True)
    panels = (
        (axes[0], "100 % розмітки", RUNS / "ablation-loo-seeds", COMPONENTS),
        (axes[1], "10 % розмітки", RUNS / "ablation-fractions-loo",
         {**COMPONENTS, "cap-4800": "4800 кадрів на епоху"}),
    )
    for ax, title, batch, configs in panels:
        rows = differences(batch, configs)
        ys = np.arange(len(rows))[::-1]
        for y, (label, delta, se) in zip(ys, rows):
            colour = ORANGE if label == COMPONENTS["gt-only"] else BLUE
            ax.errorbar(delta * 1000, y, xerr=1.96 * se * 1000, color=colour, marker="o", markersize=4.5,
                        capsize=2.5, linewidth=1.3)
        ax.axvline(0, color=INK, linewidth=0.8)
        ax.set_yticks(ys, [label for label, _, _ in rows], fontsize=8.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r"Зміна mAP, $\times 10^{-3}$")
        ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−")))
    fig.tight_layout()
    save(fig, "ablation_forest")


# ---- architectures ------------------------------------------------------------------------


@app.command()
def architectures() -> None:
    """SSL minus GT-only control (late mAP and mAP_s) for each detector at 10 % and 100 % of the labels."""
    style()
    base = RUNS / "ablation-arch-online"
    detectors = (("yolov8", "YOLOv8m"), ("yolo11", "YOLO11m"), ("yolo26", "YOLO26m"))

    def runs_for(arch: str, pct: int, method: str) -> list[pl.DataFrame]:
        if arch == "yolo26":
            return flow_runs(pct, method)
        return load_runs(f"{arch}-{pct}pct-{method}", base)

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_CM * CM, 6.2 * CM), sharey=True)
    width = 0.36
    for ax, pct in zip(axes, (10, 100)):
        for i, (metric, colour, label) in enumerate(((METRIC, BLUE, "mAP"), (METRIC_S, AQUA, r"$\mathrm{mAP}_S$"))):
            deltas, errs = [], []
            for arch, _ in detectors:
                ssl, control = runs_for(arch, pct, "ssl"), runs_for(arch, pct, "gt-only")
                if not ssl or not control:
                    deltas.append(np.nan)
                    errs.append(0.0)
                    continue
                per_seed = [
                    run.filter(pl.col("epoch").is_in(list(LATE)))[metric].mean() for run in ssl
                ]
                per_control = [
                    run.filter(pl.col("epoch").is_in(list(LATE)))[metric].mean() for run in control
                ]
                deltas.append(float(np.mean(per_seed) - np.mean(per_control)))
                errs.append(float(np.hypot(np.std(per_seed, ddof=1) if len(per_seed) > 1 else 0,
                                           np.std(per_control, ddof=1) if len(per_control) > 1 else 0)))
            x = np.arange(len(detectors)) + (i - 0.5) * width
            ax.bar(x, deltas, width, yerr=errs, color=colour, capsize=2.5, label=label,
                   error_kw={"linewidth": 1.0})
        ax.axhline(0, color=INK, linewidth=0.8)
        ax.set_xticks(range(len(detectors)), [name for _, name in detectors])
        ax.set_title(f"{pct} % розмітки", fontsize=11)
        comma(ax, digits=3)
    axes[0].set_ylabel("Приріст відносно контролю")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "architectures")


@app.command()
def all() -> None:  # noqa: A001
    """Write every figure that has its data."""
    for fn in (label_efficiency, training_curves, caps_images_seen, ssgt_histograms, ablation_forest):
        fn()
    try:
        architectures()
    except Exception as exc:  # the arch-online batch may not be finished yet
        typer.echo(f"Skipped architectures: {exc}")


if __name__ == "__main__":
    app()
