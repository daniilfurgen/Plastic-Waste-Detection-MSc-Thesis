"""Pie charts of the COCO object-size distribution in FloW-img and UCWD.

One figure, two pairs of pies: box areas at the original image resolution and
at the network input scale (longer image side -> imgsz), the same normalisation
the SSOD pipeline uses for its size bins. All splits of a dataset are pooled into one
distribution.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import typer

ROOT = Path(__file__).resolve().parent.parent
DATASETS = {
    "FloW-img": ROOT / "data/flow_img",
    "UCWD": ROOT / "data/ucwd",
}
SMALL_MAX, MEDIUM_MAX = 32**2, 96**2
SIZE_BINS = ("small", "medium", "large")
# First three categorical slots (blue, orange, aqua); validated for colour-vision deficiency.
COLORS = {"small": "#2a78d6", "medium": "#eb6834", "large": "#1baf7a"}
TEXT_COLORS = {"small": "white", "medium": "#1a1a1a", "large": "#1a1a1a"}
LABELS = {
    "uk": {
        "small": "Малі",
        "medium": "Середні",
        "large": "Великі",
        "original": "Оригінальний масштаб",
        "input": "Нормалізований масштаб мережі ({imgsz} px)",
        "decimal": ",",
    },
    "en": {
        "small": "Small",
        "medium": "Medium",
        "large": "Large",
        "original": "Original scale",
        "input": "Network input scale ({imgsz} px)",
        "decimal": ".",
    },
}
# STIX Two Text has no "≥" glyph, so that one sign comes from the STIX math font.
RANGES = {"small": "<32²", "medium": "32²–96²", "large": r"$\geq$96²"}

CM = 1 / 2.54
FIG_WIDTH_CM, FIG_HEIGHT_CM = 16, 5.6
# Pie panel in data units: room on the right for the labels of slivers.
PANEL_XLIM, PANEL_YLIM = (-1.2, 1.75), (-1.05, 1.05)
FONT = "STIX Two Text"
TITLE_SIZE, LABEL_SIZE = 12, 10
OUTLINE = {"edgecolor": "#262626", "linewidth": 1.0}


def load_boxes(annotations: Path, imgsz: int | None) -> pl.DataFrame:
    """One row per annotation with its area at the network input scale."""
    coco = json.loads(annotations.read_text())
    images = pl.DataFrame(
        {
            "image_id": [img["id"] for img in coco["images"]],
            "width": [img["width"] for img in coco["images"]],
            "height": [img["height"] for img in coco["images"]],
        }
    )
    boxes = pl.DataFrame(
        {
            "image_id": [ann["image_id"] for ann in coco["annotations"]],
            "area": [float(ann["area"]) for ann in coco["annotations"]],
        }
    )
    boxes = boxes.join(images, on="image_id", how="left")
    # Vectorised input_scale(): the letterbox factor that maps the longer side to imgsz.
    scale = 1.0 if imgsz is None else imgsz / pl.max_horizontal("width", "height")
    return boxes.with_columns((pl.col("area") * scale**2).alias("input_area"))


def size_counts(dataset_dir: Path, imgsz: int | None) -> dict[str, int]:
    """Object count per COCO size bin, pooled over every split of the dataset."""
    boxes = pl.concat(
        load_boxes(path, imgsz)
        for path in sorted(dataset_dir.glob("*/annotations.json"))
    )
    bins = (
        pl.when(pl.col("input_area") < SMALL_MAX)
        .then(pl.lit("small"))
        .when(pl.col("input_area") < MEDIUM_MAX)
        .then(pl.lit("medium"))
        .otherwise(pl.lit("large"))
    )
    counts = dict(boxes.group_by(bins.alias("size")).len().iter_rows())
    return {size: counts.get(size, 0) for size in SIZE_BINS}


def draw_pie(ax, counts: dict[str, int], decimal: str) -> None:
    values = [counts[size] for size in SIZE_BINS]
    total = sum(values)
    wedges, _ = ax.pie(
        values,
        colors=[COLORS[size] for size in SIZE_BINS],
        startangle=0,
        counterclock=False,
        wedgeprops=OUTLINE,
    )
    for wedge, size, value in zip(wedges, SIZE_BINS, values):
        share = value / total
        if share == 0:
            continue
        # Slivers (next to 3 o'clock) get their label outside the pie.
        inside = share >= 0.06
        radius = 0.62 if inside else 1.08
        angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
        x, y = radius * wedge.r * np.array([np.cos(angle), np.sin(angle)])
        ax.text(
            x,
            y,
            f"{share:.1%}".replace(".", decimal),
            ha="center" if inside else "left",
            va="center",
            fontsize=LABEL_SIZE,
            color=TEXT_COLORS[size] if inside else "#1a1a1a",
        )
    ax.set(xlim=PANEL_XLIM, ylim=PANEL_YLIM, aspect="equal")


def plot(
    panels: dict[str, dict[str, dict[str, int]]],
    labels: dict[str, str],
    output: Path,
) -> None:
    """One row of pies: a header per scale over a pair of datasets."""
    plt.rcParams.update(
        {
            "font.family": FONT,
            "font.size": LABEL_SIZE,
            "mathtext.fontset": "stix",
            # Embed fonts as TrueType so the PDF text stays selectable.
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(FIG_WIDTH_CM * CM, FIG_HEIGHT_CM * CM))
    n_cols = sum(len(datasets) for datasets in panels.values())
    col_width = 1 / n_cols
    pie_bottom, pie_height = 0.19, 0.53
    col = 0
    for header, datasets in panels.items():
        group_left = col * col_width
        group_right = (col + len(datasets)) * col_width
        fig.text(
            (group_left + group_right) / 2,
            0.95,
            header,
            ha="center",
            va="top",
            fontsize=TITLE_SIZE,
        )
        rule_y, rule_pad = 0.84, 0.02
        fig.add_artist(
            plt.Line2D(
                [group_left + rule_pad, group_right - rule_pad],
                [rule_y, rule_y],
                color="#262626",
                linewidth=0.6,
            )
        )
        for name, counts in datasets.items():
            ax = fig.add_axes((col * col_width, pie_bottom, col_width, pie_height))
            draw_pie(ax, counts, labels["decimal"])
            total = f"{sum(counts.values()):,}".replace(",", "\u2009")
            ax.set_title(f"{name} (n = {total})", fontsize=LABEL_SIZE, pad=3)
            col += 1

    fig.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[size], **OUTLINE)
            for size in SIZE_BINS
        ],
        labels=[f"{labels[size]} ({RANGES[size]} px)" for size in SIZE_BINS],
        fontsize=LABEL_SIZE,
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=len(SIZE_BINS),
        frameon=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main(
    output: Path = typer.Option(
        ROOT / "paper/figures/object_size_distribution.pdf", help="Output PDF path."
    ),
    imgsz: int = typer.Option(640, help="Network input size the areas are scaled to."),
    lang: str = typer.Option("uk", help="Label language: uk or en."),
) -> None:
    """Plot the distribution at the original resolution and at the network input scale."""
    labels = LABELS[lang]
    scales = {
        labels["original"]: None,
        labels["input"].format(imgsz=imgsz): imgsz,
    }
    panels = {
        header: {name: size_counts(path, scale) for name, path in DATASETS.items()}
        for header, scale in scales.items()
    }
    for header, datasets in panels.items():
        for name, counts in datasets.items():
            typer.echo(f"{header} / {name}: {counts}")
    plot(panels, labels, output)
    typer.echo(f"Saved {output}")


if __name__ == "__main__":
    typer.run(main)
