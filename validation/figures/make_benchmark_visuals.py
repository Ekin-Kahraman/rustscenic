"""Generate README and docs visuals from committed benchmark data.

Outputs:
  - site_docs/assets/rustscenic-proof-strip.svg
  - site_docs/assets/rustscenic-memory-context.svg
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


def _strip_trailing_whitespace(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def write_proof_strip(rows: list[dict[str, str]]) -> None:
    real_runs = [r for r in rows if r["kind"] == "real"]
    synth_runs = [r for r in rows if r["kind"] == "synthetic"]
    max_synth_rss = max(_float(r, "peak_rss_gb") for r in synth_runs)
    max_synth_cells = max(int(r["n_cells"]) for r in synth_runs)

    cards = [
        ("pip install", "rustscenic", "Single install path"),
        ("5", "runtime deps", "numpy, pandas, scipy +2"),
        ("No", "Java / dask / CUDA", "Modern Python plus CPU"),
        (f"{max_synth_cells // 1000}k", "synthetic cells", f"Peak memory {max_synth_rss:.2f} GB"),
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


def write_memory_context(rows: list[dict[str, str]]) -> None:
    real = [r for r in rows if r["kind"] == "real"]
    synthetic = [r for r in rows if r["kind"] == "synthetic"]
    reported = next(r for r in rows if r["kind"] == "reported")

    max_real = max(real, key=lambda r: _float(r, "peak_rss_gb"))
    max_synthetic = max(synthetic, key=lambda r: _float(r, "peak_rss_gb"))
    labels = [
        "RustScenic real public E2E\nmax observed",
        "RustScenic synthetic E2E\n100k to 200k",
        "Legacy SCENIC+ stack\nreported around 100k",
    ]
    values = [
        _float(max_real, "peak_rss_gb"),
        _float(max_synthetic, "peak_rss_gb"),
        _float(reported, "peak_rss_gb"),
    ]
    annotations = [
        f"{values[0]:.2f} GB peak RSS\n{int(max_real['n_cells']):,} cells, all stages",
        f"{values[1]:.2f} GB peak RSS\n200,000 cells, all stages",
        ">40 GB peak RSS\nreported baseline only",
    ]
    colours = ["#2c5aa0", "#1f7a6d", "#87908a"]

    fig, ax = plt.subplots(figsize=(9.6, 4.8), dpi=150)
    fig.patch.set_facecolor("#f7f8f5")
    ax.set_facecolor("#f7f8f5")

    y = range(len(labels))
    bars = ax.barh(y, values, color=colours, height=0.52)
    ax.set_xlim(0, 45)
    ax.invert_yaxis()
    ax.set_xlabel("Peak resident memory (GB), lower is better")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.grid(axis="x", color="#d7ddd7", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#889188")
    ax.spines["bottom"].set_color("#889188")

    for rect, text in zip(bars, annotations):
        ax.text(
            rect.get_width() + 0.65,
            rect.get_y() + rect.get_height() / 2,
            text,
            ha="left",
            va="center",
            fontsize=9.4,
            color="#202124",
        )

    ax.set_title(
        "Peak-memory context: measured RustScenic rows stay below 8 GB",
        loc="left",
        fontsize=13,
        fontweight="bold",
        color="#202124",
        pad=14,
    )
    fig.text(
        0.08,
        0.060,
        "Source data: validation/figures/benchmark_visuals.csv. RustScenic rows are measured validation artefacts.",
        fontsize=8.2,
        color="#5f6368",
    )
    fig.text(
        0.08,
        0.035,
        "The legacy SCENIC+ row is reported context, not a controlled head-to-head run.",
        fontsize=8.2,
        color="#5f6368",
    )
    fig.tight_layout(rect=(0, 0.15, 1, 1))

    svg_path = ASSET_DIR / "rustscenic-memory-context.svg"
    fig.savefig(svg_path, format="svg", metadata={"Date": None})
    fig.savefig(ASSET_DIR / "rustscenic-memory-context.png", format="png", metadata={"Date": None})
    plt.close(fig)
    _strip_trailing_whitespace(svg_path)


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
        (f"{max_cells // 1000}k cells", f"{max_rss:.2f} GB RAM"),
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
    write_memory_context(rows)
    write_opengraph(rows)
    print(f"wrote assets under {ASSET_DIR}")


if __name__ == "__main__":
    main()
