"""Render the case study from auditable, donor-level source tables."""
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def render(folder):
    donor = pd.read_csv(folder / "lineage-donor-means.csv")
    trace = pd.read_csv(folder / "regulator-trace.csv")
    orange, ink, grey = "#D46F40", "#24292F", "#57616C"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "path", "pdf.fonttype": 42,
                         "svg.hashsalt": "rustscenic-case-study", "text.color": ink,
                         "axes.labelcolor": ink, "xtick.color": ink, "ytick.color": ink})
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 7), gridspec_kw={"width_ratios": [1.3, 1]})
    fig.subplots_adjust(left=.21, right=.96, bottom=.23, top=.74, wspace=.55)
    fig.text(.045, .94, "Biological case study: blood-cell regulatory programmes", fontsize=18, weight="bold", color=ink)
    fig.text(.045, .89, "Saved networks and scores · no model refitting · all five predefined regulator checks shown", color=ink, fontsize=11)
    a, b = axes
    a.set_title("A  Useful lineage activity", loc="left", pad=22, weight="bold", color=ink)
    panels = [("BMMC", "SPI1", "Gene activity", "SPI1 · myeloid RNA"),
              ("BMMC", "GATA1", "Gene activity", "GATA1 · erythroid RNA"),
              ("BMMC", "SPI1", "Observed-region activity", "SPI1 · myeloid ATAC"),
              ("BMMC", "GATA1", "Observed-region activity", "GATA1 · erythroid ATAC"),
              ("Kang", "SPI1", "Gene activity", "SPI1 · external PBMC RNA")]
    for y, (cohort, tf, measure, _) in enumerate(panels):
        query = donor[(donor.training_cells == 20000) & donor.cohort.eq(cohort) & donor.TF.eq(tf)]
        values = query[query.method.eq("rustscenic") & query.measurement.eq(measure)].auroc.to_numpy()
        a.plot([values.min(), values.max()], [y, y], color=orange, lw=2)
        a.scatter(values, y + np.linspace(-.07, .07, len(values)), color=orange, s=22, zorder=3)
        a.text(1.035, y, f"{values.mean():.3f}", va="center", color=ink, weight="bold", fontsize=10)
        if y < 2:
            for method, measurement, marker in [("expression", "TF expression", "s"),
                                               ("rustscenic", "Random-target gene activity", "D")]:
                vals = query[query.method.eq(method) & query.measurement.eq(measurement)].auroc
                a.scatter(vals, np.full(len(vals), y), s=28, marker=marker,
                          facecolor="white", edgecolor=grey, zorder=4)
    a.axvline(.5, color=grey, ls=":", lw=1)
    a.set(xlim=(0, 1.12), ylim=(4.6, -.6), yticks=range(5),
          yticklabels=[x[3] for x in panels], xticks=[0, .25, .5, .75, 1], xlabel="Lineage AUROC (higher is better)")
    a.legend(handles=[Line2D([], [], marker="o", color=orange, lw=0, label="RustScenic"),
                      Line2D([], [], marker="s", color=grey, markerfacecolor="white", lw=0, label="TF expression"),
                      Line2D([], [], marker="D", color=grey, markerfacecolor="white", lw=0, label="Random targets")],
             loc="upper left", bbox_to_anchor=(-.03, 1.11), ncol=3, frameon=False, fontsize=8.5,
             columnspacing=1.0, handletextpad=.35)
    b.set_title("B  Why three regulators are missing", loc="left", pad=22, weight="bold", color=ink)
    markers = ["SPI1", "GATA1", "PAX5", "EBF1", "TCF7"]
    for y, tf in enumerate(markers):
        for method, offset, colour, symbol in [("rustscenic", -.11, orange, "o"), ("reference", .11, grey, "s")]:
            values = trace[trace.TF.eq(tf) & trace.method.eq(method)].max_nes
            b.plot([values.min(), values.max()], [y + offset, y + offset], color=colour, lw=1.5)
            b.scatter(values, np.full(len(values), y + offset), color=colour, marker=symbol, s=15)
    b.axvline(3, color=ink, ls="--", lw=1)
    b.text(3.3, 4.65, "Enrichment cutoff = 3", fontsize=9, color=ink)
    b.set(xlim=(0, 17), ylim=(5, -.6), yticks=range(5), yticklabels=markers,
          xticks=[0, 3, 6, 9, 12, 15], xlabel="Best motif enrichment score per run (NES)")
    b.legend(handles=[Line2D([], [], marker="o", color=orange, lw=0, label="RustScenic"),
                      Line2D([], [], marker="s", color=grey, lw=0, label="SCENIC+")],
             loc="upper left", bbox_to_anchor=(-.03, 1.11), frameon=False, ncol=2, fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(grey)
        ax.tick_params(length=3, color=grey)
        ax.grid(axis="x", color="#D8DFE5", linewidth=.8)
        ax.set_axisbelow(True)
    fig.text(.045, .14, "A: 20,000-cell training networks; dots = donor means over three seeds (3 BMMC donors; 8 external donors).", fontsize=10, color=ink)
    fig.text(.045, .103, "RNA and observed ATAC scores recover myeloid/erythroid programmes. Self-gene removal preserves the RNA signal.", fontsize=10, color=ink)
    fig.text(.045, .066, "B: both training sizes and all seeds. Missing regulators have motifs and initial edges, but fail the unchanged enrichment cutoff.", fontsize=10, color=ink)
    fig.text(.045, .029, "Post hoc case study. AUROC measures lineage discrimination, not the fraction of correct regulatory edges. Full comparisons accompany this figure.", fontsize=9, color=ink)
    for suffix in ("png", "svg", "pdf"):
        path = folder / f"biological-case-study.{suffix}"
        metadata = {"Date": None} if suffix == "svg" else (
            {"CreationDate": None, "ModDate": None} if suffix == "pdf" else None)
        fig.savefig(path, dpi=180, facecolor="white", metadata=metadata)
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    render(parser.parse_args().folder)
