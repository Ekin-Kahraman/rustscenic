# Landscape comparison (archived 2026-04-25 snapshot)

Measured from the GitHub repos and their published releases. This doc
anchors the April 2026 positioning work. Use the README and
`site_docs/benchmarks.md` for the current release claims.

## Active maintenance

| Project | Stars | Last release | Last commit |
|---|---:|---:|---:|
| [aertslab/pySCENIC](https://github.com/aertslab/pySCENIC) | 598 | v0.12.1 (2022-11-21) | 2025-01-09 |
| [aertslab/arboreto](https://github.com/aertslab/arboreto) | 67 | v0.1.6 (2021-02-09) | 2024-04-05 |
| [aertslab/pycisTopic](https://github.com/aertslab/pycisTopic) | 79 | v1.0.2 (2023-04-23) | 2026-03-30 |
| [aertslab/pycistarget](https://github.com/aertslab/pycistarget) | 18 | v1.1 (2025-01-10) | 2026-04-14 |
| [aertslab/scenicplus](https://github.com/aertslab/scenicplus) | 251 | v1.0a2 alpha (2025-01-13) | 2026-01-16 |
| [Ekin-Kahraman/rustscenic](https://github.com/Ekin-Kahraman/rustscenic) | n/a | v0.4.6 (2026-05-18) | 2026-05-18 |

- pySCENIC's last PyPI release is **3.5 years old**.
- arboreto's last PyPI release is **5 years old**.
- pycisTopic's last PyPI release is **3 years old**.

PyPI release cadence is the right proxy for what users actually install.
The compute repos are effectively frozen even when development continues
on `main`.

## Install surface

| Project | pyproject deps | requirements.txt lines | Install story (fresh Python 3.12, 2026-04) |
|---|---:|---:|---|
| pySCENIC | n/a | 19 | `pip install` fails: `ModuleNotFoundError: pkg_resources` |
| arboreto | n/a | 7 | `pip install` fails: `TypeError: Must supply at least one delayed object` (dask_expr) |
| pycisTopic | 84 | 67 | Heavy (polars, pyranges, MACS2, Mallet / Java). Recent v3 rework in PR #226. |
| pycistarget | 8 | n/a | Installs; depends on pycisTopic + ctxcore upstream. |
| scenicplus | 51 | **918** | 918-line pinned requirements file. Recent issue #629 confirms `pkg_resources` deprecation is actively biting scenicplus users on Setuptools ≥ 81. |
| **rustscenic** | **5** | n/a | `pip install rustscenic` (PyPI) on Python 3.10 to 3.13, with Linux, macOS and Windows x64 wheels in the v0.4.6 release. Five runtime deps: numpy, pandas, pyarrow, scipy, anndata. |

Documented install failures on the reference stack:
- arboreto issue [#42](https://github.com/aertslab/arboreto/issues/42) (Oct 2024, still open): `grnboost2 TypeError: Must supply at least one delayed object`; the dask API arboreto depends on was removed. Unpatched for 18 months.
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
| **rustscenic** | **57 Rust tests + 144 Python tests + 39 validation scripts** |

Three of the five reference repos ship with zero checked-in tests.
That's how "installs cleanly on Python 3.7" becomes "dead on Python 3.12".

## Functional scope covered by one install

| Stage | pySCENIC | arboreto | pycisTopic | pycistarget | scenicplus | **rustscenic** |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| GRN inference (GRNBoost2) | ✓ | ✓ | - | - | (via arboreto) | ✓ |
| AUCell regulon activity | ✓ | - | - | - | ✓ | ✓ |
| Topic modelling on ATAC | - | - | ✓ (needs Java/Mallet) | - | ✓ (needs pycisTopic) | ✓ (no Java) |
| Motif enrichment | - | - | - | ✓ | ✓ | ✓ |
| ATAC fragment to matrix | - | - | ✓ (needs pyranges, MACS2) | - | ✓ (needs pycisTopic) | ✓ |
| Bundled TF lists | - | - | - | - | - | ✓ (1,839 hs + 1,721 mm) |
| Motif DB auto-download | - | - | - | - | - | ✓ |
| One-call pipeline runner | - | - | - | - | partial (snakemake) | ✓ (`rustscenic.pipeline.run`; region cistarget supported) |

## Speed benchmarks (vs pyscenic, measured on this repo)

| Stage / dataset | pyscenic | rustscenic | speedup |
|---|---:|---:|---:|
| AUCell, 31,602 cells × 59 regulons (Ziegler atlas) | 6.81 s | 0.25 s | **27×** |
| AUCell, 10,290 cells × 1,457 regulons (10x Multiome) | 18.6 s | 0.21 s | **88×** |
| Peak RSS, 4 stages on 100k cells × 20k genes | > 40 GB (reported) | 6.3 GB | **≈ 7×** |
| GRN at 30k cells (Ziegler, n_estimators=300) | (arboreto doesn't install) | 139 s post PR #12 | - |

Numerical parity vs pyscenic:
- AUCell per-cell Pearson 0.984 on Ziegler 31k (91.7 % of cells > 0.95).
- AUCell per-cell Pearson 0.988 on 10x Multiome (99.5 % > 0.95).
- Cistarget Pearson 1.0000 vs `ctxcore.recovery.aucs` (bit-identical at float32).
- GRN edges disagree at fine rank (Spearman 0.58) but coarse biology matches: 94 % of known TRRUST edges, 8/8 lineage TFs, all 13 canonical airway TFs.

## Different scope (not direct competitors)

- [haozhu233/flashscenic](https://github.com/haozhu233/flashscenic) (★2): GPU, different algorithm (RegDiffusion). Outputs are not pyscenic-numerical.
- [scverse/decoupler](https://github.com/scverse/decoupler) (★266): activity scoring from prebuilt regulons; does not infer GRNs from data.
- R-SCENIC, [Epiregulon](https://www.nature.com/articles/s41467-025-62252-5): R Bioconductor ecosystem.

## Summary positioning

- **Single-install implementation track for the practical SCENIC / SCENIC+ compute path.** The project goal is one CPU package for GRN, AUCell, motif enrichment, ATAC preprocessing, topics, enhancer-gene links, and eRegulons.
- **Maintained CPU-Python option in this repo's validation scope.** The comparison above is a repo and release snapshot, not a formal market claim.
- **Single install covers four compute stages plus ATAC preprocessing, bundled TFs and motif DB helpers.** The next strict-parity milestone is a real scenicplus head-to-head using region-ranking databases.
- **Numerically equivalent to pyscenic** at per-cell AUC (0.98+), bit-identical at cistarget, honest on GRN edge disagreement (documented).
- **Substantial checked-in tests and validation scripts**, with current CI status reported in the README.

Data cited above is from public GitHub metadata and local release validation, with the original snapshot taken on 2026-04-25 and rustscenic release status refreshed on 2026-05-16.
