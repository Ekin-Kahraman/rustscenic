"""Plot completed runs: python make_figure.py (requires matplotlib).

Lines connect measured cell counts; no performance model or inference is run.
"""
import csv
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
METHODS = ("rustscenic", "arboreto_sparse", "arboreto_mp_sparse")
NAMES = ("RustScenic", "arboreto · Dask", "arboreto · multiprocessing")
SIZES = (100000, 500000, 1306127)
# Match the documented symbol colour and existing README workflow palette.
# See site_docs/assets/branding/README.txt and generate_workflow.py.
BRAND_ORANGE = "#D46F40"
BRAND_TEAL = "#449F95"


def measurements():
    with (ROOT / "scaling-measurements.csv").open(newline="") as f:
        raw = list(csv.DictReader(f))
    rows = [{"cells": int(r["cells"]), "seed": int(r["seed"]), "method": r["method"],
             "analysis_s": float(r["analysis_s"]),
             "analysis_peak_pss_gb": float(r["analysis_peak_pss_mb"]) / 1000} for r in raw]
    summary = json.loads((ROOT / "summary.json").read_text())
    provenance = json.loads((ROOT / "scaling-provenance.json").read_text())
    assert len(rows) == 15
    for key in ("genes", "tfs", "threads"):
        assert provenance[key] == summary[key]
    medians = {}
    for method in METHODS:
        medians[method] = {}
        for count in SIZES:
            selected = [r for r in rows if r["method"] == method and r["cells"] == count]
            expected = provenance["repeats_by_cell_count"][str(count)]
            assert len(selected) == len(expected)
            assert {r["seed"] for r in selected} == set(expected)
            values = {key: statistics.median(r[key] for r in selected)
                      for key in ("analysis_s", "analysis_peak_pss_gb")}
            if count == SIZES[-1]:
                for key, value in values.items():
                    assert value > 0 and math.isclose(value, summary["medians"][method][key])
            medians[method][count] = values
    return rows, medians, summary


def draw(rows, medians, summary, *, dark=False, mobile=False):
    if dark:
        bg, fg, muted, grid = "#141414", "#E6EDF3", "#9DA7B3", "#303B47"
        colors = (BRAND_ORANGE, "#9DA7B3", "#9DA7B3")
        accent = BRAND_TEAL
    else:
        bg, fg, muted, grid = "#FFFFFF", "#24292F", "#57616C", "#D8DFE5"
        colors = (BRAND_ORANGE, "#57616C", "#9DA7B3")
        accent = BRAND_TEAL
    fig = plt.figure(figsize=(8.4, 12.4) if mobile else (15.4, 8.6), facecolor=bg)
    if mobile:
        axes = [fig.add_axes((.13, .52, .82, .265)), fig.add_axes((.13, .16, .82, .265))]
        fig.text(.07, .97, "RustScenic: scaling to\n1.3 million real cells",
                 color=fg, fontsize=21, fontweight="bold", va="top")
        fig.text(.07, .895, "Mouse-brain RNA · 2,095 genes · 256 TFs · 16 CPUs", color=muted, fontsize=11)
        legend_y = .875
    else:
        axes = [fig.add_axes((.075, .25, .385, .47)), fig.add_axes((.58, .25, .385, .47))]
        fig.text(.055, .955, "RustScenic: scaling to 1.3 million real cells",
                 color=fg, fontsize=26, fontweight="bold", va="top")
        fig.text(.055, .887, "Mouse-brain RNA · 2,095 genes · 256 TFs · 16 CPU cores",
                 color=muted, fontsize=13)
        legend_y = .844
    styles = (("o", "-", 3.5), ("s", "--", 2.1), ("^", ":", 2.2)) if dark else (
        ("o", "-", 2.8), ("s", "--", 1.8), ("^", ":", 2.0))
    handles = [Line2D([0], [0], color=c, marker=m, linestyle=ls, linewidth=lw,
                      markersize=6, label=n) for c, (m, ls, lw), n in zip(colors, styles, NAMES)]
    legend = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.055, legend_y),
                        ncol=2 if mobile else 3, frameon=False,
                        fontsize=11.2 if mobile else 12, handlelength=2.8, columnspacing=2.2)
    for label in legend.get_texts():
        label.set_color(fg)
    final = SIZES[-1]
    rust, dask, mp = (medians[m][final] for m in METHODS)
    speedup = dask["analysis_s"] / rust["analysis_s"]
    reduction = 100 * (1 - rust["analysis_peak_pss_gb"] / dask["analysis_peak_pss_gb"])
    assert math.isclose(speedup, summary["speedup_ratio_of_medians"])
    assert math.isclose(reduction, summary["pss_reduction_percent"])
    panels = [
        ("analysis_s", 60, "A  Runtime", f"{speedup:.2f}× faster", "Analysis time (min)", 112, (0, 20, 40, 60, 80, 100)),
        ("analysis_peak_pss_gb", 1, "B  Memory", f"{reduction:.1f}% lower memory", "Peak memory (GB, PSS)", 16, (0, 4, 8, 12, 16)),
    ]
    for panel, (ax, (key, divisor, title, improvement, ylabel, ymax, yticks)) in enumerate(zip(axes, panels)):
        ax.set_facecolor(bg)
        for method, color, (marker, style, lw) in zip(METHODS, colors, styles):
            xs = [count / 1e6 for count in SIZES]
            ys = [medians[method][count][key] / divisor for count in SIZES]
            ax.plot(xs, ys, color=color, marker=marker, linestyle=style, linewidth=lw,
                    markersize=6.5 if dark else 5.5, zorder=4 if method == "rustscenic" else 3)
            runs = [r[key] / divisor for r in rows if r["method"] == method and r["cells"] == final]
            ax.scatter([final / 1e6] * len(runs), runs, s=31 if dark else 24, facecolor=color,
                       edgecolor=bg, linewidth=.65, zorder=6)
            ax.vlines(final / 1e6, min(runs), max(runs), color=color, linewidth=1.2, zorder=5)
            value = ys[-1]
            y_label = {"rustscenic": value, "arboreto_sparse": 104, "arboreto_mp_sparse": 86}[method] if panel == 0 else value
            annotation = f"{value:.1f}" if panel == 0 else f"{value:.2f}"
            ax.annotate(annotation, xy=(xs[-1], value), xytext=(1.365, y_label), va="center",
                fontsize=10.8 if mobile and dark else 10.2 if mobile else 12.2 if dark else 11.4, color=color,
                fontweight="bold" if method == "rustscenic" else "normal",
                arrowprops=dict(arrowstyle="-", color=color, lw=.7, shrinkA=2, shrinkB=5) if y_label != value else None)
        ax.set_title(title, loc="left", fontsize=14 if mobile else 15.5, fontweight="bold", color=fg, pad=20)
        ax.text(1, 1.065, improvement, transform=ax.transAxes, ha="right",
                fontsize=15 if mobile else 17, color=accent, fontweight="bold")
        ax.set(xlim=(0, 1.64), ylim=(0, ymax), yticks=yticks,
               xticks=(0, .5, 1.0, 1.306127), xticklabels=("0", "0.5", "1.0", "1.3"))
        ax.set_xlabel("Cells (millions)", fontsize=12, labelpad=11, color=fg)
        ax.set_ylabel(ylabel, fontsize=12, labelpad=11, color=fg)
        ax.grid(axis="y", color=grid, linewidth=.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=fg, labelsize=10.5, length=3.5, pad=6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("bottom", "left"):
            ax.spines[spine].set_color(muted)
            ax.spines[spine].set_linewidth(.8)
    if mobile:
        footers = [
            "Gains vs Dask at 1.3M. 100k/500k: one seed; 1.3M: three seeds.",
            "Core GRN inference · preparation/loading excluded · sampled PSS",
        ]
        for y, line in zip((.075, .045), footers):
            fig.text(.07, y, line, color=muted, fontsize=10)
    else:
        fig.text(.075, .125, "Gains vs arboreto Dask at 1.3M. One seed at 100k/500k; three seeds at 1.3M. Points: runs; lines: medians.",
                 color=muted, fontsize=11.5)
        fig.text(.075, .080, "Core GRN inference; preparation/loading excluded. Memory: peak sampled process-tree PSS. Full conditions linked below.",
                 color=muted, fontsize=11.5)
    suffix = ("-mobile" if mobile else "") + ("-dark" if dark else "") + "-brand"
    svg_path = ROOT / f"figure{suffix}.svg"
    fig.savefig(svg_path, facecolor=bg,
        metadata={"Date": None, "Title": "RustScenic: measured scaling on real single-cell data",
                  "Description": "Runtime and memory at 100,000, 500,000 and 1,306,127 real cells. Both arboreto routes shown."})
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    if not mobile and not dark:
        fig.savefig(ROOT / "figure-brand.png", dpi=180, facecolor=bg,
            metadata={"Software": "RustScenic scientific benchmark visualization"})
    plt.close(fig)


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path", "svg.hashsalt": "rustscenic-atlas-20261026"})
    observations, values, provenance = measurements()
    for narrow in (False, True):
        for night in (False, True):
            draw(observations, values, provenance, dark=night, mobile=narrow)
    print("Generated scientific scaling figures from 15 validated completed runs.")
