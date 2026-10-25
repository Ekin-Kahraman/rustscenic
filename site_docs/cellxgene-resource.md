# CELLxGENE Census-derived networks

[**Download v1.0.0 on Zenodo**](https://doi.org/10.5281/zenodo.23548341) ·
[Try the interactive example](demo/endothelial/index.html)

Precomputed exploratory gene regulatory networks for **704 human Cell Ontology
labels**, generated with the Huang Lab at Mount Sinai from **CELLxGENE Census
2025-11-08**. Reuse the results without HPC access or rerunning inference.

The collection includes **137,766,796 retained TF-to-target edges** and
**51,527 motif-supported regulons**, drawing on **565 contributing datasets**.
Each label uses up to 5,000 cells balanced by dataset and donor.

## Find and load results

1. Use the [JSON catalogue](https://zenodo.org/records/23548341/files/catalog.json?download=1)
   or [Parquet index](https://zenodo.org/records/23548341/files/resource_index.parquet?download=1)
   to find a label and its `cohort_index`.
2. Download the corresponding archive from Zenodo. Eight archives each contain
   88 cohorts, with index ranges in their filenames. For example, endothelial
   cells (`CL:0000115`, index 43) are in
   `rustscenic-cellxgene-human-networks-v1.0.0-cohorts-001-088.tar.gz`.
3. Extract the archive, enter `rustscenic-cellxgene-human-networks-v1.0.0`, then run:

```bash
python -m pip install -r requirements-reuse.txt
python reuse_example.py --cell-type-id CL:0000115
```

Use Python 3.11 or later. For the complete collection, extract all eight
archives into the same directory and run `python verify_package.py --require-complete`
inside the extracted folder. The [release README](https://zenodo.org/records/23548341/files/README.md?download=1)
includes full instructions.

Each cohort includes network tables, motif evidence, regulons, aggregate AUCell
activity summaries, metadata and source citations. Methods, checksums and the
exact development workflow are archived. Raw expression matrices, cell-level
records and input motif databases are excluded.

## Interpretation

All 704 outputs passed file-integrity and content-consistency checks.
**Biological accuracy and cross-study replication have not been established.**
Labels include broad annotations; cohorts pool tissues and disease states and
use one fixed random seed. Motif support does not experimentally confirm an
interaction. Activity scores are not directly comparable across cohorts with
different gene sets. See the [published methods](https://zenodo.org/records/23548341/files/METHODS.md?download=1).

## Citation and credits

Kahraman, E., Huangfu, L., El Moussaoui, M., & Huang, K.-L. (2026).
*RustScenic: exploratory human gene regulatory networks for 704 CELLxGENE Census
labels* (Version 1.0.0) [Data set]. Zenodo.
[doi:10.5281/zenodo.23548341](https://doi.org/10.5281/zenodo.23548341).

- **Ekin Kahraman:** software and workflow development, analysis design,
  troubleshooting, verification and packaging.
- **Longtao Huangfu:** Minerva execution and resumption of indices 9–704,
  and transfer of the 696 completed outputs.
- **Mohamed El Moussaoui:** IFB HPC access and resources for the original batch
  and supporting work.
- **Kuan-Lin Huang:** scientific supervision, resource concept and collaboration
  direction.

Additional RustScenic testing and issue reporting: **Fuaad Siyaad**.

Also cite CELLxGENE and the contributing studies listed in each cohort's
`sources.json`. See the [attribution notes](https://zenodo.org/records/23548341/files/ATTRIBUTION.md?download=1).

Derived data and resource documentation: **CC BY 4.0**. Included software:
**Apache-2.0**. See the [dataset licence](https://zenodo.org/records/23548341/files/LICENSING.txt?download=1).
This resource is independently hosted on Zenodo. CELLxGENE is the source
platform, not the host or endorser. The dataset DOI is separate from the
[RustScenic software DOI](https://doi.org/10.5281/zenodo.20246040).
