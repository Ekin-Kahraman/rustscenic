# Using RustScenic from Seurat / R

RustScenic accepts Python expression inputs: AnnData, a cells-by-genes pandas
DataFrame, or a matrix paired with gene names. A full Seurat object is not an
input. This guide exchanges one explicitly chosen RNA expression matrix and its
axis names, then returns activity scores by cell ID.

Use already log-normalized RNA expression and gene identifiers matching your
TF/regulon lists. The examples do not normalize expression, map identifiers,
select `raw.X`, or guess an AnnData layer. A layer named `data` is not proof that
its contents were normalized correctly; check how your object was prepared.

## Tested scope

The tiny example below was executed on macOS arm64 with R 4.5.2,
SeuratObject 5.4.0, Matrix 1.7-4, Python 3.12.13 and the published
RustScenic 0.5.0 wheel. Python dependencies were AnnData 0.13.4, SciPy 1.18.1,
NumPy 2.5.4 and pandas 3.1.0.

This tests real `Assay5`/SeuratObject data selection, sparse matrix exchange,
AUCell scoring and barcode-indexed `AddMetaData` return. The synthetic fixture
normalizes its counts explicitly with `log1p(10000 * counts / cell_total)`.
It does not test a full Seurat normalization/integration pipeline, SeuratDisk,
sceasy, reticulate or anndata2ri. It is an interchange check, not biological
validation of the toy gene sets.

## Export the RNA data layer

Run from a repository checkout, using
[seurat_export.R](https://github.com/Ekin-Kahraman/rustscenic/blob/main/examples/seurat_export.R):

```r
source("examples/seurat_export.R")
export_rna_data(seurat_obj, "rna_export", assay = "RNA")
```

The helper selects `seurat_obj[["RNA"]]` explicitly, regardless of the default
assay. It requires a single `data` layer and ignores `counts` and `scale.data`.
It writes:

| File | Contents |
| --- | --- |
| `expression.mtx` | Sparse, genes-by-cells RNA `data` matrix |
| `genes.txt` | One feature ID per matrix row, in order |
| `cells.txt` | One cell ID per matrix column, in order |

Names and the selected expression values are exchanged without normalization;
Matrix Market text can introduce floating-point round-off. The helper rejects
missing/split `data` layers,
nonfinite or negative values, ambiguous IDs and existing exchange files. These
checks cannot establish that normalization or gene annotation was appropriate.
The example handles an in-memory R matrix or Matrix sparse matrix; it is not an
on-disk BPCells exporter or a streaming workflow. Matrix Market text also needs
additional disk space compared with a compressed H5AD file.

If a v5 RNA assay has split normalized layers such as `data.batchA` and
`data.batchB`, inspect them first. If they were prepared consistently and
joining is appropriate for your analysis, join them deliberately:

```r
SeuratObject::Layers(seurat_obj[["RNA"]], search = NA)
seurat_obj[["RNA"]] <- SeuratObject::JoinLayers(
    seurat_obj[["RNA"]], layers = "data", new = "data"
)
export_rna_data(seurat_obj, "rna_export", assay = "RNA")
```

`JoinLayers` does not normalize data or reconcile different preprocessing.
If `data` is absent, perform your intended RNA normalization upstream instead
of exporting counts under another name. See the official
[v5 layer guidance](https://satijalab.org/seurat/articles/seurat5_essential_commands.html)
and [JoinLayers reference](https://satijalab.github.io/seurat-object/reference/SplitLayers.html).

For a saved object, the same export is executable from a shell:

```bash
Rscript examples/seurat_export.R input.rds rna_export
```

## Read and score in Python

[seurat_score.py](https://github.com/Ekin-Kahraman/rustscenic/blob/main/examples/seurat_score.py)
checks matrix dimensions and IDs, transposes **once** into cells-by-genes
orientation, and constructs a sparse AnnData with named axes. It never selects
or transforms another expression matrix.

Provide a JSON object mapping regulon names to gene lists, for example:

```json
{
  "SPI1_regulon": ["SPI1", "CD14", "LYZ", "CSF1R"],
  "PAX5_regulon": ["PAX5", "CD79A", "MS4A1"]
}
```

These are illustrative gene sets, not inferred or motif-validated regulons.
Use your analysis's own gene sets and inspect the reported gene coverage.

```bash
python examples/seurat_score.py rna_export regulons.json activity.csv --h5ad rna.h5ad
```

`activity.csv` contains cells by regulons, indexed by the original cell IDs.
The optional `rna.h5ad` has the exported log-normalized RNA expression in `X`
and can also be passed to GRN inference or the RNA pipeline. No full pipeline
or GRN accuracy claim follows from this AUCell interchange example.

Feature IDs are exported unchanged. For symbol-based TF/regulon lists, supply
matching symbols. If your features are Ensembl IDs, obtain a verified matching
symbol mapping upstream, or add it to AnnData's `var["feature_name"]` before
scoring. RustScenic can resolve that metadata when `var_names` look like Ensembl
IDs, but the text exchange does not invent or carry an additional annotation
table. Check species, identifier versions and regulon gene coverage.

## Return activity to R

```r
source("examples/seurat_export.R")
seurat_with_activity <- add_regulon_activity(seurat_obj, "activity.csv")
```

The helper matches and reorders scores by barcode before `AddMetaData`. It
preserves regulon names and numeric-looking cell IDs such as `001`, rejects
missing/extra cells, and refuses to overwrite existing metadata columns.
It requires scores for every cell in the supplied object. Seurat v5 assays can
contain different cell subsets; if RNA contains fewer cells than the whole
object, deliberately subset the object to the exported cells or handle the
unscored cells explicitly in your own metadata workflow.

Do not append or remove a barcode suffix automatically.
[`Read10X`](https://satijalab.org/seurat/reference/read10x) defaults to
`strip.suffix = FALSE`; preserve and match the actual IDs in your object.

The return step also has a command-line form:

```bash
Rscript examples/seurat_export.R --add-activity input.rds activity.csv with_activity.rds
```

## Reproduce the tiny round-trip

With the R/Python environments described above, run from a repository checkout
and use a new export directory:

```bash
Rscript examples/seurat_export.R --example tiny_rna
python examples/seurat_score.py tiny_rna tiny_rna/regulons.json tiny_rna/activity.csv --h5ad tiny_rna/rna.h5ad
Rscript examples/seurat_export.R --add-activity tiny_rna/seurat.rds tiny_rna/activity.csv tiny_rna/with_activity.rds
```

The fixture has 64 genes and 8 cells, with 333 nonzero normalized values.
Its separate `scale.data` layer has only 32 genes and contains negative values;
the export still contains the full nonnegative `data` matrix. Python reads an
8-by-64 sparse matrix and scores two gene sets, with 4/4 and 3/3 genes covered.
The activity matrix has five nonzero values. These small-fixture results are
engineering checks, not evidence of biological accuracy.

The R regression checks also cover missing/split layers, deliberate joining,
value/ID preservation and shuffled activity rows:

```bash
Rscript tests/test_seurat_interop.R . /tmp/rustscenic-seurat-check
python -m pytest tests/test_seurat_like_input.py
```

The R test requires SeuratObject and Matrix; the Python tests alone do not
exercise R or validate a third-party converter.

## If you use another converter

Check its supported object versions and its expression-selection rules rather
than assuming the converted `X` is normalized RNA:

- [SeuratDisk](https://mojaveazure.github.io/seurat-disk/reference/Convert.html)
  documents that H5Seurat-to-H5AD conversion prefers `scale.data` for `X` when
  present. In that case `raw.X` receives `data`, and `X` may contain only scaled
  features. Without `scale.data`, `X` receives `data` and `raw.X` may instead be
  counts. Neither `raw.X` nor a layer called `logcounts` is a universal fallback.
  `SaveH5Seurat` writes the object's assays; `Convert(..., assay = "RNA")`
  selects the assay converted into `X`.
- [sceasy's Seurat converter](https://github.com/cellgeni/sceasy/blob/3d0b5155752a61a5c42bd14fe60bd351400cbc51/R/functions.R#L48-L99)
  reads legacy assay slots, including `@meta.features`. Its compatibility with
  your `Assay5` object must be checked; the example here does not validate it.
- [anndata2ri](https://icb-anndata2ri.readthedocs-hosted.com/en/latest/) converts
  between AnnData and SingleCellExperiment. It is not the SeuratDisk file
  conversion API or a direct Seurat converter. A SingleCellExperiment route
  requires its own explicit assay choice and version checks.

After any conversion, inspect the actual expression values, feature coverage
and cell IDs. RustScenic's large-value normalization warning is a heuristic;
its absence does not establish that a matrix is normalized correctly.
