# Independent debate — what rustscenic does NOT yet prove

Honest self-review of the gaps between "the tests pass" and "this is a
production-ready SCENIC+ replacement." Written after the Apr 22 burst
that closed the scope gap vs the aertslab stack, to name the validation
we still owe before a v1.0 claim.

## 1. Synthetic-only smoke coverage

`tests/test_full_scenicplus_smoke.py` walks every public API end to end
but on generated data: 120 cells, 3 programmes, fragments dropped by
cluster membership, expression driven by the same activity vector.
That isolates interface breakage (peak_id key mismatches between
modules, column schema drift, chrom-convention drops) — the class of
bug Fuaad hit on Kamath et al. 2022.

What it does **not** catch:
- **Real 10x Multiome parsing edge cases** — barcode-prefix conventions,
  `+` / `-` strand annotations, doublet-marker cells, cells with < 10
  fragments total.
- **Real biological noise** — our synthetic correlation is ≈ 0.8; real
  multiome peak↔gene correlations are typically 0.1–0.3 for true links,
  well inside the `min_abs_corr=0.1` default but below the `0.2` we
  used in the smoke test.
- **Distribution shifts across chromosomes** — the smoke test uses
  chr1 only, so the Rust hashmap indexing and the Python `_normalise_chrom`
  helper are not exercised on chrY, chrM, or the 1..22 + X multi-chrom
  loop that dominates real wall-clock.

**Gap to close:** add one integration test that runs the full pipeline
on a real multiome sample (e.g. 10x public 10k PBMC multiome, 4 GB)
pinned as a fixture under `validation/multiome/`.

## 2. Peak calling has no MACS2 cross-check

`rustscenic.preproc.call_peaks` implements Corces 2018's
density-window / iterative-overlap-rejection algorithm. Our only
validation is the synthetic recovery check in the smoke test: peak
centroid inside the programme's ± 500 bp window. That proves the code
produces *a* peak somewhere in the right neighbourhood.

What we haven't shown:
- Peak count within a factor of 2 of MACS2 on the same fragments.
- Precision-recall of our peaks vs a MACS2 gold standard on ENCODE.
- Behaviour at extreme pseudobulk sizes (single-cluster, 100k cells
  per pseudobulk).

**Why it matters:** pycisTopic's downstream topic model is trained on
the called peaks. A 4× peak-count mismatch vs MACS2 would shift
topic-peak assignments silently.

**Gap to close:** run `call_peaks` on ENCODE 10x Multiome PBMC, compare
against MACS2-broadPeak. Target: ≥ 70 % F1 at 50 bp IoU tolerance.

## 3. eRegulon intersection thresholds are defaults, not tuned

`build_eregulons` defaults to `min_target_genes=5`,
`min_enhancer_links=2`, `use_grn_intersection=True`,
`cistarget_auc_threshold=0.05`. The smoke test uses
`min_target_genes=2, min_enhancer_links=1` because synthetic regulons
are smaller — the defaults have never been validated against scenicplus'
eRegulons on the same input.

**Concrete risk:** if scenicplus produces 150 eRegulons on Kamath et
al. and rustscenic produces 20 because of the intersection threshold,
Fuaad's next message will be "I'm getting 1/5 the regulons." The
silent-regression class we already built guards against would help here
only if rustscenic emits a warning when the cut gets catastrophic.

**Gap to close:** add `build_eregulons` diagnostic warning when
`len(eregulons) < 0.5 × len(unique_cistarget_tfs)`, and document the
threshold-tuning path.

## 4. "one install" is from GitHub, not PyPI yet

PyPI trusted-publisher config requires the user's PyPI account state.
The short-term install path is GitHub: either `pip install git+...@v0.3.3`
or a platform-specific wheel from the latest tagged Release. But:
- `pip install rustscenic` from PyPI returns nothing yet.
- Release wheel names are platform-locked. Users need the matching
  macOS / Linux, x86_64 / aarch64 wheel.
- The release workflow must keep publishing GitHub Release assets even
  while PyPI remains gated.

**Gap to close:** either (a) resolve PyPI trusted-publishing config
(user action), or (b) keep GitHub Release wheels as the official
distribution path and keep README examples platform-specific.

## 5. Scaling proof covers GRN only

PR #12 (partition buffer pool) fixed the super-linear scaling we
observed on GRN specifically, and the CI regression test in `tests/`
guards only that slope. We have *not* benchmarked scaling for:
- `fragments_to_matrix` on 100k cells × 500k peaks.
- `call_peaks` with 10 clusters × 100k cells.
- `enhancer.link_peaks_to_genes` on 100k peaks × 30 genes × 50k cells.
- `build_eregulons` on 100k links (though this is trivial pandas bookkeeping).

**Concrete risk:** the Python-side `link_peaks_to_genes` function
densifies `atac_X` (`_densify` at `python/rustscenic/enhancer.py:282`)
— a 100k × 50k float32 matrix is 20 GB. A user doing real multiome
will hit OOM here, not in the Rust core.

**Gap to close:** either (a) port `link_peaks_to_genes` to Rust with
sparse peak_vec × dense gene_block correlation, or (b) at minimum,
emit a clear error when the densified block would exceed available
RAM.

## 6. No test for `data.download_motif_rankings`

The orchestrator (`pipeline.run`) accepts a motif rankings path; the
`data` helper fetches the aertslab-hosted feather file. We ship the
code but have no test that the URL is alive, the checksum matches,
or that the fetched rankings join to `rna.var_names` under common
species conventions.

**Concrete risk:** aertslab moves the file to a new URL; rustscenic
silently fails with "motif rankings not found" — or worse, a silent
empty DataFrame.

**Gap to close:** add a weekly scheduled CI job that pings the URL
and runs the join-test with a tiny fixture. Skip on PR CI (network
flakiness).

## 7. Convention-mismatch guards are Python-only

PR #24 added silent-zero guards: ENSEMBL-in-var_names auto-detect,
ENSEMBL-in-rankings warning, UCSC-vs-Ensembl chrom normalisation.
These live in `python/rustscenic/_gene_resolution.py` and
`python/rustscenic/enhancer.py`. The **Rust** layer (peak calling,
fragment parsing, cistarget scoring) has chrom normalisation on the
`peaks` side via `normalise_chrom` in `rustscenic-preproc/src/peaks.rs`,
but gene-name resolution is entirely Python — Rust never sees gene
symbols directly, so this is correct by construction. Worth stating
so nobody looks for an equivalent Rust helper and assumes we missed
something.

## 8. What "independent debate" is NOT testing

- **Memory regression over time** — we have one 100k-cell Ziegler
  RSS number from a single run; no nightly baseline.
- **Numerical stability across BLAS versions** — we pin numpy >= 1.21
  but haven't tested MKL vs OpenBLAS on the same dataset.
- **Windows support** — we claim Linux + macOS only; the code likely
  builds on Windows via maturin but nobody has tried it.
- **Seurat interop path** — `docs/seurat-interop.md` exists but the
  scope is one function; we have not tested a real Seurat → rustscenic
  pipeline.

## Priority ranking for the next sprint

1. **Real-data multiome integration test** (gap 1) — highest value,
   lowest effort, directly addresses the class of bug Fuaad hit.
2. **Densification OOM guard in enhancer** (gap 5) — prevents a bad
   first-impression crash on real 100k multiome.
3. **MACS2 cross-check on ENCODE** (gap 2) — load-bearing for the
   "MACS2-free" claim in the README.
4. **eRegulon diagnostic warning** (gap 3) — cheap, prevents silent
   regulon-count regressions.
5. **Motif rankings URL monitor** (gap 6) — costs one scheduled CI
   job, prevents a class of supply-chain failures.

Everything else is nice-to-have until v1.0.
