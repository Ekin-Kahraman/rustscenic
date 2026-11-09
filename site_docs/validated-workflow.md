# Run the audited workflow

Turn prepared RNA, or paired RNA+ATAC, into **motif-supported regulatory networks and per-cell activity**. One command writes the networks, activity tables, regulator coverage and run manifest.

RustScenic v0.6.0 exposes the native recipes used in the paper with the verified seeded AUCell correction. It does not change the inference kernels or require pySCENIC, SCENIC+, Dask or Java. The existing flexible `pipeline` command remains available.

Install with `python -m pip install --upgrade rustscenic`. The `validated` command requires v0.6.0 or later.

## RNA

```bash
rustscenic validated --rna rna.h5ad --tfs hs \
  --motif-rankings gene-rankings.feather \
  --motif-annotations motifs.tbl \
  --output results-rna --threads 16
```

Use `mm` for mouse or a newline-separated TF file. RNA `X` must hold prepared non-negative expression, normally library-size normalised and log1p transformed. Use unique gene symbols in `var_names`, matching the TF list and motif database. Cell identifiers must also be unique. Select features before running; the command does not silently normalise or subsample.

Supply a complete motif universe from a species-matched gene-ranking database. Normal aertslab `.tbl` annotations work directly. The recipe uses signed top-50 modules, NES ≥3, annotated motif support and at least five retained genes. Unsupported candidates are never substituted into the activity matrix.

## Paired RNA+ATAC

```bash
rustscenic validated --rna rna.h5ad --atac atac.h5ad --tfs hs \
  --region-motif-rankings region-rankings.feather \
  --region-mapping peak-to-database.tsv \
  --motif-annotations motifs.tbl --gene-coords gene-coordinates.tsv \
  --activity-mode directional --output results-paired --threads 16
```

Additional inputs:

| Input | Required contents |
| --- | --- |
| `atac.h5ad` | Raw non-negative integer peak counts; exactly the RNA cells in the same order; unique `chr:start-end` peak names. |
| `peak-to-database.tsv` | `Target` = input peak ID; `Query` = overlapping ranking-database region ID. Multiple overlaps may be listed. Use the same genome build and a documented overlap rule. For identical region IDs, an identity mapping is sufficient. |
| `gene-coordinates.tsv` | `gene`, `chrom`, `tss`; gene symbols match RNA and TSS positions use zero-based coordinates. |
| `motifs.tbl` | Standard motif, gene and `description` fields; the paired recipe retains direct or orthologous annotations without motif-similarity-only support. |

The recipe fits 30 Gibbs topics for 500 iterations, tests the top 1,000 regions per topic for motif support, links imputed accessibility to genes within 150 kb, and assembles eRegulons. Activity is scored from the **retained eRegulon gene and region sets**.

`--activity-mode directional` keeps positively and negatively associated target sets in separate columns, such as `SPI1_activator` and `SPI1_repressor`. Use both columns for downstream analysis rather than subtracting them: subtraction loses information when both programmes are active. These names describe expression associations, not proven activation or repression. Only the selected representation is scored and stored. Omitting the option preserves the historical paired TF unions; RNA already retains separate directions.

Imputed accessibility uses temporary disk storage in the output directory, approximately `cells × active peaks × 4` bytes. For example, 100,000 cells × 20,000 peaks needs about 8 GB of scratch disk. Linking and scoring use bounded batches. Scratch files are removed when finished; final networks and activity tables remain.

## Read the results

```python
import json
import pandas as pd

manifest = json.load(open("results-rna/manifest.json"))
activity = pd.read_parquet("results-rna/gene_activity.parquet")
coverage = pd.read_parquet("results-rna/regulator_coverage.parquet")
print(manifest["status"], activity.shape)
print(coverage[coverage.status != "retained"])
```

| File | Meaning |
| --- | --- |
| `regulons.json` | Exact gene signatures scored. |
| `gene_activity.parquet` | Cells × retained regulons; cell identities preserved. |
| `regulator_coverage.parquet` | Every requested TF, including missing or filtered regulators. Missing does not mean biologically inactive. |
| `motif_evidence.parquet` | Enriched motif evidence. RNA includes enriched rows before annotation pruning; `regulons.json` is the final retained set. |
| `manifest.json` | Recipe, seeds, thresholds, input fingerprints, signature source, counts, timings and artifact paths. |
| Paired outputs | Also `eregulon_edges.parquet`, `region_regulons.json`, `region_activity.parquet`, topic matrices and peak-to-gene links. |

`no_supported_regulons` means no network passed the fixed filters. Activity is then empty, with the original cell axis preserved. Inspect gene naming, database/species/build compatibility and the coverage report. Do not relax thresholds simply to recover a desired regulator. An output directory must be new or empty, protecting previous results.

## Python and reproducibility

```python
from rustscenic.validated import run

manifest = run("rna.h5ad", "results-rna", tfs="hs",
    motif_rankings="gene-rankings.feather", motif_annotations="motifs.tbl")
```

Set `RAYON_NUM_THREADS=16` before starting Python to control GRN threads. The CLI sets it from `--threads`; in Python, `threads` controls topic fitting. Both interfaces default to inference seed 777 and scoring seed 20261106. Preserve feature order as well as seeds when comparing tied ranks.

For matched HPC runs, also set `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and `OMP_NUM_THREADS=1` before starting the process, so numerical-library pools do not exceed the job's CPU allocation. Request scratch disk as well as RAM for paired runs.

Ranking files must retain authentic integer ranks across all motifs. If columns were projected from a larger database, put `{"original_rank_universe": ORIGINAL_FEATURE_COUNT}` in a same-stem JSON file. For a DataFrame, set `rankings.attrs["rank_universe_size"]`. Never renumber projected ranks. Column projection is supported; selecting only favourable motifs changes NES and is not the audited recipe.

The recipes produce regulatory associations supported by expression, motifs and, for paired data, accessibility. Existing real-data case studies support useful biological signals. Keeping separate activity channels improved cell-type classification in the [archived-cohort diagnostic](https://github.com/Ekin-Kahraman/rustscenic/blob/main/validation/f1_gap/README.md). This supports their use as downstream features; it does not establish every inferred edge as causal or guarantee recovery of every regulator on a new dataset.
