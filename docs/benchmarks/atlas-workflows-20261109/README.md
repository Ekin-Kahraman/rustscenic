# Verified atlas workflow benchmarks

Three seeds per size, 21 matched pairs, 16 CPUs. Figures use the individual
measurements in [the consolidated evidence](../../paper/evidence/scaling/consolidated-runs.csv).
Lines connect medians; dots show every run. No fitted or projected timings.
Palette, font sizes, canvas, panel positions, line weights, markers and spacing
match the [live README graph template](../huangfu-atlas-20261026/make_figure.py).
RustScenic uses orange `#D46F40`; gains use teal `#449F95`. The original
light/dark backgrounds and mobile layout are retained. Only the data, labels
and axis ranges vary.

- [RNA+ATAC vs SCENIC+](scenicplus.png): 817,740 cells; 14.376× faster and
  57.062% lower sampled peak process-tree PSS. 5,000 genes, 20,000 regions,
  256 TFs. Three sizes from 100,000 to 817,740 cells.
- [RNA vs pySCENIC](pyscenic.png): 1,306,127 cells; 3.979× faster and 96.617%
  lower sampled peak PSS. 3,000 genes, 256 TFs. Two measured sizes.
- [Earlier core-GRN vs arboreto](../huangfu-atlas-20261026/benchmark-source-record.md):
  retained separately because its feature panel and timing scope differ.

Prepared-input loading, analysis and standard export are timed. Raw-data
preparation, startup and final hashing are excluded. Matched pairs share a
host/allocation; CPU models differ across some sizes. RustScenic 0.5.0 with
the archived adapter, pySCENIC 0.12.1 and SCENIC+ 1.0a2 were measured.

The paired workflows use different linking, topic, pruning and assembly
algorithms. The largest paired comparison has only five to six common final
TF signatures. Its timing advantage is not evidence of identical outputs or
quality-matched regulatory prediction. The smaller donor-held-out biological
experiments and later scoring confirmation are reported separately.

Each figure has light/dark and desktop/mobile SVG/PNG variants, plus a light
PDF. Rebuild after `python validation/paper/build.py` with:

```bash
python validation/paper/benchmark_figures.py
```

[Methods and biological results](../../paper/RESULTS.md) ·
[Source audit](../../paper/evidence/scaling/AUDIT.txt)
