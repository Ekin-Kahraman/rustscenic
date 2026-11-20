# Reproducing the replication and the post hoc diagnosis

These are the scripts as run. The replication package in `package/` ran on the
Minerva HPC cluster, executed by a collaborator from 13 to 18 November 2026.
`audit/analyse_return.py` and the `diagnosis/` scripts ran afterwards on a
10-core macOS arm64 laptop (18 to 19 November; the published-wheel check in
`diagnosis/published-wheel-check/` on 20 November, whose scripts name the
original workspace folder, shown here as `<local working copy>`). `make_figure.py` was revised for
print legibility on 20 November and reads only files in this folder. None of
the other scripts runs unmodified from this repository: they expect the layout
and the unpublished inputs listed below.

## The replication package (`package/`)

`package/` is the code of the package that was sent for the run: the ZIP's
SHA-256, `730b0931a449a774b21fb820eb55bb9bd78bb1edbef2cf338eace4b0e5695230`, is
recorded in [`../stopping-rule/minerva/ZIP-RECEIPT.json`](../stopping-rule/minerva/ZIP-RECEIPT.json),
and every included file matches the package's own `SHA256SUMS`, which is
byte-identical to the copy inside that ZIP. `PROTOCOL.md` has SHA-256
`cd35e196ac8a52df6ab57bfdc00d7727d6a40053e6a46b2398e35edabccea55b`.

Listed in `SHA256SUMS` but not included:

- `START-HERE.txt`, `start.sh` and `retry.sh`: launch notes and two thin
  wrappers that name the site's LSF allocation account. `start.sh ACCOUNT
  [RECOVERY]` runs `orchestrate.py submit ACCOUNT RECOVERY WORK` with the
  Python of the recovery environment (`RECOVERY/work/reference/bin/python`,
  overridable with `RUSTSCENIC_PYTHON`); `retry.sh` does the same with `retry`.
- `inputs/bmmc/*`: the bone-marrow training (20,000 cells, seven donors) and
  evaluation (10,000 cells, three donors) matrices, the 5,000-gene panel with
  1,069 TF predictors and the motif subset, derived from GEO GSE194122 and the
  cisTarget v10 ranking database
  `hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather`.
- `resources/human-gene-annotations.tbl` (motif-to-gene annotations) and
  `resources/references/*.tsv.gz`, which `REFERENCES.json` derives from
  BEELINE `BEELINE-Networks.zip` (Zenodo record 19009603): first two columns as
  TF and target, upper-cased, self-links and duplicates removed. Member hashes
  are in `REFERENCES.json`.
- Two wheels, described in `resources/BUILD.json`: a local build of RustScenic
  v0.6.1 (commit `e59dc76`) plus `resources/KERNEL-PATCH.diff`, which was not
  published (SHA-256 `ff7d8766dd031d0bc48a728e5ac997296680817231a8811c08c67adb43a4db6c`),
  and the unmodified PyPI scikit-learn 1.2.2 cp311 manylinux wheel
  (`bf036ea7ef66115e0d49655f16febfa547886deba20149555a41d28f56fd6d3c`).

External inputs on the cluster, outside the package: a `rustscenic-recovery`
directory from an earlier HPC package, beside the package, providing the
reference Python environment (`work/reference`: Python 3.11.8 with
scikit-learn 1.3.2, arboreto 0.1.6 and pySCENIC 0.12.1; the same interpreter
path was used by the final Minerva run, whose package list is in
`../minerva-final/returned/jobs/*/environment.json`) and the prepared inputs
`work/prepared/atlas` (10x Genomics 1.3 million E18 mouse brain cells) and
`work/prepared/hdma` (HDMA RNA, Zenodo record 15053816), checked against
`resources/expected-atlas.json` and `resources/expected-hdma.json`.

## Local audit and diagnosis

Expected layout:

```
stopping-rule-minerva/
  package/                       published package code, plus its inputs/bmmc/ and resources data (not included)
  rustscenic-stopping-rule  ->  package     the scripts look for the package under this name
                                            (for example: ln -s package rustscenic-stopping-rule)
  audit/analyse_return.py        input: the returned archive (SEND_BACK.tar.gz, SHA-256 in
                                 audit/SEND_BACK.sha256) or its extracted folder; not published
  diagnosis/
    atlas-rebuilt/               rebuild_atlas.py output, first 500,000 atlas cells (rna.h5, tfs.txt)
    atlas-rebuilt-full/          the same for all 1,306,127 cells
    native-bins/                 refits written by queue.sh, queue-release.sh, queue-full.sh, final-timing.sh
```

The scoring scripts take the extracted return as their first argument.
`rebuild_atlas.py` also needs the final-scaling package (its `prepare_atlas.py`
and `inputs/atlas-panel.json`, not included) and the public 10x source file of
1.3 million E18 mouse brain cells (`mouse-1m.h5` in the scripts).

Interpreters. All local environments are Python 3.11.8 virtual environments
layered on the local reference environment described in
[`../stopping-rule/REPRODUCE.md`](../stopping-rule/REPRODUCE.md), each with a
different local RustScenic build installed. In this copy the shell scripts read
them from variables that replace the author-local paths; that is the only
change to the shell scripts (the only other script edit in this folder is one
docstring line in `audit/analyse_return.py`):

| Variable | Build |
| --- | --- |
| `PY_RELEASE_U8` | v0.6.1 plus the uncommitted `paired_oob` patch, 255 bins (the replication kernel) |
| `PY_EXACT_BINS` | `exp/finer-bins` prototype; `RUSTSCENIC_MAX_BINS` sets the bin budget (`queue-full.sh` used commit `095554b`) |
| `PY_RC070` | v0.7.0 release-candidate wheel built from the working tree (see `diagnosis/DIAGNOSIS.md`) |
| `FINAL_SCALING_PACKAGE`, `ATLAS_SOURCE_H5` | the final-scaling package directory and the 10x source file, for `rebuild_atlas.py` |

The released equivalents are RustScenic 0.7.0 with the default (exact bins) and
with `max_bins=255`; `diagnosis/DIAGNOSIS.md` records that the published 0.7.0
and 0.6.1 wheels reproduced the bone-marrow and 20,000-cell atlas networks bit
for bit. Run Python with `-I`.
