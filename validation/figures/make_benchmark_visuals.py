"""Generate README and docs visuals from committed benchmark data.

Outputs:
  - site_docs/assets/rustscenic-proof-strip.svg
  - site_docs/assets/rustscenic-memory-scale.svg
  - site_docs/assets/rustscenic-opengraph.png

The script intentionally keeps the figures evidence-first: all plotted
RustScenic numbers come from `benchmark_visuals.csv`, which points back to
committed validation artefacts.

Requires matplotlib and Pillow for local regeneration.
"""
from __future__ import annotations

import csv
import html
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "validation" / "figures" / "benchmark_visuals.csv"
ASSET_DIR = ROOT / "site_docs" / "assets"


def _rows() -> list[dict[str, str]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def write_proof_strip(rows: list[dict[str, str]]) -> None:
    real_runs = [r for r in rows if r["kind"] == "real"]
    synth_runs = [r for r in rows if r["kind"] == "synthetic"]
    max_synth_rss = max(_float(r, "peak_rss_gb") for r in synth_runs)
    max_synth_cells = max(int(r["n_cells"]) for r in synth_runs)

    cards = [
        ("pip install", "rustscenic", "Single install path"),
        ("5", "runtime deps", "numpy, pandas, scipy +2"),
        ("No", "Java / dask / CUDA", "Modern Python plus CPU"),
        (f"{max_synth_cells // 1000}k", "synthetic cells", f"Peak RSS {max_synth_rss:.2f} GB"),
        (str(len(real_runs)), "real multiome runs", "All 7 stages non-empty"),
    ]

    width, height = 1280, 360
    card_w, card_h = 220, 150
    x0, gap = 54, 22
    y = 130
    accents = ["#1f7a6d", "#2c5aa0", "#a85b18", "#6b5b95", "#546a2f"]

    def esc(text: str) -> str:
        return html.escape(text, quote=True)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        "<title id=\"title\">RustScenic validation summary</title>",
        "<desc id=\"desc\">One install, five runtime dependencies, no Java dask or CUDA, and 100k to 200k synthetic multiome runs below 8 GB peak RSS.</desc>",
        '<rect width="1280" height="360" fill="#f7f8f5"/>',
        '<rect x="0" y="0" width="1280" height="8" fill="#1f7a6d"/>',
        '<text x="54" y="58" font-family="Arial, sans-serif" font-size="34" font-weight="700" fill="#202124">RustScenic: SCENIC+ compute path in one package</text>',
        '<text x="54" y="93" font-family="Arial, sans-serif" font-size="18" fill="#4b5563">Evidence-backed install, memory and real-data validation summary. Full commands and caveats live in site_docs/benchmarks.md.</text>',
    ]
    for idx, (top, main, sub) in enumerate(cards):
        x = x0 + idx * (card_w + gap)
        accent = accents[idx]
        parts.extend(
            [
                f'<rect x="{x}" y="{y}" width="{card_w}" height="{card_h}" rx="10" fill="#ffffff" stroke="#d8ded8"/>',
                f'<rect x="{x}" y="{y}" width="{card_w}" height="8" rx="4" fill="{accent}"/>',
                f'<text x="{x + 18}" y="{y + 56}" font-family="Arial, sans-serif" font-size="34" font-weight="700" fill="#202124">{esc(top)}</text>',
                f'<text x="{x + 18}" y="{y + 88}" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="{accent}">{esc(main)}</text>',
                f'<text x="{x + 18}" y="{y + 120}" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">{esc(sub)}</text>',
            ]
        )
    parts.extend(
        [
            '<text x="54" y="325" font-family="Arial, sans-serif" font-size="13" fill="#5f6368">Scale rows are synthetic workload proofs. Legacy SCENIC+ memory is a reported baseline, not a controlled head-to-head run.</text>',
            "</svg>",
        ]
    )
    (ASSET_DIR / "rustscenic-proof-strip.svg").write_text("\n".join(parts), encoding="utf-8")


def write_memory_scale(rows: list[dict[str, str]]) -> None:
    plotted = [r for r in rows if r["kind"] in {"real", "synthetic"}]
    labels = ["PBMC\n3k real", "Brain\n5k real", "PBMC\n10k real", "100k\nsynthetic", "200k\nsynthetic"]
    cells = [int(r["n_cells"]) for r in plotted]
    rss = [_float(r, "peak_rss_gb") for r in plotted]
    mins = [_float(r, "runtime_s") / 60.0 for r in plotted]
    colours = ["#2c5aa0" if r["kind"] == "real" else "#1f7a6d" for r in plotted]

    fig, ax1 = plt.subplots(figsize=(9.6, 4.8), dpi=150)
    fig.patch.set_facecolor("#f7f8f5")
    ax1.set_facecolor("#f7f8f5")

    x = range(len(plotted))
    bars = ax1.bar(x, rss, color=colours, width=0.62)
    ax1.set_ylim(0, 8.5)
    ax1.set_ylabel("Peak RSS (GB)")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels)
    ax1.grid(axis="y", color="#d7ddd7", linewidth=0.8)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.spines["left"].set_color("#889188")
    ax1.spines["bottom"].set_color("#889188")

    for rect, value, minutes in zip(bars, rss, mins):
        ax1.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.12,
            f"{value:.2f} GB\n{minutes:.1f} min",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#202124",
        )

    ax1.set_title(
        "Full-pipeline evidence: real multiome plus synthetic atlas-scale runs",
        loc="left",
        fontsize=13,
        fontweight="bold",
        color="#202124",
        pad=14,
    )

    legend_handles = [
        plt.Line2D([0], [0], color="#2c5aa0", lw=8, label="Real public multiome"),
        plt.Line2D([0], [0], color="#1f7a6d", lw=8, label="Synthetic scale proof"),
    ]
    ax1.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=8.5)
    fig.text(
        0.08,
        0.060,
        "Labels show peak RSS and wall time. Source data: validation/figures/benchmark_visuals.csv.",
        fontsize=8.2,
        color="#5f6368",
    )
    fig.text(
        0.08,
        0.035,
        "Reported legacy SCENIC+ memory >40 GB is a reported baseline, not a controlled head-to-head run.",
        fontsize=8.2,
        color="#5f6368",
    )
    fig.tight_layout(rect=(0, 0.15, 1, 1))

    fig.savefig(ASSET_DIR / "rustscenic-memory-scale.svg", format="svg", metadata={"Date": None})
    fig.savefig(ASSET_DIR / "rustscenic-memory-scale.png", format="png", metadata={"Date": None})
    plt.close(fig)


def write_opengraph(rows: list[dict[str, str]]) -> None:
    synth = [r for r in rows if r["kind"] == "synthetic"]
    max_rss = max(_float(r, "peak_rss_gb") for r in synth)
    max_cells = max(int(r["n_cells"]) for r in synth)

    w, h = 1200, 630
    img = Image.new("RGB", (w, h), "#f7f8f5")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, w, 18), fill="#1f7a6d")
    draw.rectangle((54, 88, 1146, 542), fill="#ffffff", outline="#d8ded8", width=2)
    draw.rectangle((54, 88, 1146, 106), fill="#2c5aa0")

    title = _font(72, bold=True)
    subtitle = _font(34, bold=False)
    metric = _font(44, bold=True)
    small = _font(26, bold=False)
    tiny = _font(22, bold=False)

    draw.text((92, 146), "RustScenic", font=title, fill="#202124")
    draw.text((96, 235), "SCENIC+ compute path in one Python package", font=subtitle, fill="#2f3b45")

    metrics = [
        ("pip install", "one command"),
        ("5 deps", "runtime core"),
        ("No Java", "no dask / CUDA"),
        (f"{max_cells // 1000}k cells", f"{max_rss:.2f} GB RSS"),
    ]
    x_positions = [96, 356, 616, 876]
    for x, (top, bottom) in zip(x_positions, metrics):
        draw.rounded_rectangle((x, 328, x + 220, 442), radius=14, fill="#f7f8f5", outline="#d8ded8", width=2)
        draw.text((x + 18, 350), top, font=metric, fill="#1f7a6d")
        draw.text((x + 20, 404), bottom, font=small, fill="#4b5563")

    draw.text((96, 492), "Benchmarks, commands, hardware and caveats committed in repo docs.", font=tiny, fill="#5f6368")
    img.save(ASSET_DIR / "rustscenic-opengraph.png")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    rows = _rows()
    write_proof_strip(rows)
    write_memory_scale(rows)
    write_opengraph(rows)
    print(f"wrote assets under {ASSET_DIR}")


if __name__ == "__main__":
    main()
