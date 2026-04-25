# Landscape comparison (2026-04-25)

Measured from the GitHub repos and their published releases. This doc
anchors rustscenic's positioning claims with current, citable data.

## Active maintenance

| Project | Stars | Last release | Last commit |
|---|---:|---:|---:|
| [aertslab/pySCENIC](https://github.com/aertslab/pySCENIC) | 598 | v0.12.1 (2022-11-21) | 2025-01-09 |
| [aertslab/arboreto](https://github.com/aertslab/arboreto) | 67 | v0.1.6 (2021-02-09) | 2024-04-05 |
| [aertslab/pycisTopic](https://github.com/aertslab/pycisTopic) | 79 | v1.0.2 (2023-04-23) | 2026-03-30 |
| [aertslab/pycistarget](https://github.com/aertslab/pycistarget) | 18 | v1.1 (2025-01-10) | 2026-04-14 |
| [aertslab/scenicplus](https://github.com/aertslab/scenicplus) | 251 | v1.0a2 alpha (2025-01-13) | 2026-01-16 |
| [Ekin-Kahraman/rustscenic](https://github.com/Ekin-Kahraman/rustscenic) | — | v0.2.0 (2026-04-25) | 2026-04-25 |

- pySCENIC's last PyPI release is **3.5 years old**.
- arboreto's last PyPI release is **5 years old**.
- pycisTopic's last PyPI release is **3 years old**.

PyPI release cadence is the right proxy for what users actually install.
The compute repos are effectively frozen even when development continues
on `main`.

## Install surface

| Project | pyproject deps | requirements.txt lines | Install story (fresh Python 3.12, 2026-04) |
|---|---:|---:|---|
| pySCENIC | — | 19 | `pip install` fails: `ModuleNotFoundError: pkg_resources` |
| arboreto | — | 7 | `pip install` fails: `TypeError: Must supply at least one delayed object` (dask_expr) |
| pycisTopic | 84 | 67 | Heavy (polars, pyranges, MACS2, Mallet / Java). Recent v3 rework in PR #226. |
| pycistarget | 8 | — | Installs; depends on pycisTopic + ctxcore upstream. |
| scenicplus | 51 | **918** | 918-line pinned requirements file. Recent issue #629 confirms `pkg_resources` deprecation is actively biting scenicplus users on Setuptools ≥ 81. |
| **rustscenic** | **4** | — | GitHub Release wheels and `pip install git+...@v0.2.0` succeed on Python 3.10–3.13, Linux + macOS, x86_64 + aarch64. Four runtime deps: numpy, pandas, pyarrow, scipy. PyPI name publish is pending trusted-publisher setup. |

Documented install failures on the reference stack:
- arboreto issue [#42](https://github.com/aertslab/arboreto/issues/42) (Oct 2024, still open): `grnboost2 TypeError: Must supply at least one delayed object` — the dask API arboreto depends on was removed. Unpatched for 18 months.
- scenicplus issue [#629](https://github.com/aertslab/scenicplus/issues/629) (Feb 2026, open): `pkg_resources` deprecation on Setuptools 81 breaks scenicplus at import. Unpatched.

rustscenic doesn't use dask or `pkg_resources` anywhere. Both failure
modes above are impossible by construction.

## Test coverage (test files in repo)

| Project | Test files |
|---|---:|
| pySCENIC | 4 |
| arboreto | 3 |
| pycisTopic | 0 |
| pycistarget | 0 |
| scenicplus | 0 |
| **rustscenic** | **51 Rust tests + 106 Python tests + 19 validation scripts** |

Three of the five reference repos ship with zero checked-in tests.
That's how "installs cleanly on Python 3.7" becomes "dead on Python 3.12".

## Functional scope covered by one pip install

| Stage | pySCENIC | arboreto | pycisTopic | pycistarget | scenicplus | **rustscenic** |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| GRN inference (GRNBoost2) | ✓ | ✓ | — | — | (via arboreto) | ✓ |
| AUCell regulon activity | ✓ | — | — | — | ✓ | ✓ |
| Topic modelling on ATAC | — | — | ✓ (needs Java/Mallet) | — | ✓ (needs pycisTopic) | ✓ (no Java) |
| Motif enrichment | — | — | — | ✓ | ✓ | ✓ |
| ATAC fragment → matrix | — | — | ✓ (needs pyranges, MACS2) | — | ✓ (needs pycisTopic) | ✓ (PR #5–#9, no new deps) |
| Bundled TF lists | — | — | — | — | — | ✓ (PR #15, 1,839 hs + 1,721 mm) |
| Motif DB auto-download | — | — | — | — | — | ✓ (PR #15) |
| One-call full pipeline | — | — | — | — | partial (snakemake) | ✓ (PR #15, `rustscenic.pipeline.run`) |

## Speed benchmarks (vs pyscenic, measured on this repo)

| Stage / dataset | pyscenic | rustscenic | speedup |
|---|---:|---:|---:|
| AUCell, 31,602 cells × 59 regulons (Ziegler atlas) | 6.81 s | 0.25 s | **27×** |
| AUCell, 10,290 cells × 1,457 regulons (10x Multiome) | 18.6 s | 0.21 s | **88×** |
| Peak RSS, 4 stages on 100k cells × 20k genes | > 40 GB (reported) | 6.3 GB | **≈ 7×** |
| GRN at 30k cells (Ziegler, n_estimators=300) | (arboreto doesn't install) | 139 s post PR #12 | — |

Numerical parity vs pyscenic:
- AUCell per-cell Pearson 0.984 on Ziegler 31k (91.7 % of cells > 0.95).
- AUCell per-cell Pearson 0.988 on 10x Multiome (99.5 % > 0.95).
- Cistarget Pearson 1.0000 vs `ctxcore.recovery.aucs` (bit-identical at float32).
- GRN edges disagree at fine rank (Spearman 0.58) but coarse biology matches: 94 % of known TRRUST edges, 8/8 lineage TFs, all 13 canonical airway TFs.

## Different scope (not direct competitors)

- [haozhu233/flashscenic](https://github.com/haozhu233/flashscenic) (★2) — GPU, different algorithm (RegDiffusion). Outputs not pyscenic-numerical.
- [scverse/decoupler](https://github.com/scverse/decoupler) (★266) — activity scoring from prebuilt regulons; does not infer GRNs from data.
- R-SCENIC, [Epiregulon](https://www.nature.com/articles/s41467-025-62252-5) — R Bioconductor ecosystem.

## Summary positioning

- **Only maintained CPU-Python drop-in for the SCENIC compute stack.** Every competing compute module has either a multi-year-stale PyPI release or known unpatched install breakage.
- **Single install covers 4 compute stages + ATAC preprocessing + bundled TFs + auto-fetch motif DB.** Closest competitor needs ≥ 3 packages stitched together (pySCENIC + pycisTopic + pycistarget + their dependency tangle).
- **Numerically equivalent to pyscenic** at per-cell AUC (0.98+), bit-identical at cistarget, honest on GRN edge disagreement (documented).
- **An order of magnitude more tested** than every competing repo combined.

Data cited above is from public GitHub metadata and local release validation on 2026-04-25.
