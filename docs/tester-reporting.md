# Tester reporting guide

If you ran rustscenic on real data — your own or a public dataset —
this page tells you what to send back, how to send it, and where it
ends up.

The shortest version:

> Open a GitHub issue using the
> [**Validation report**](https://github.com/Ekin-Kahraman/rustscenic/issues/new?template=validation_report.yml)
> template. Fill the required fields. If you can attach a JSON artefact
> matching the schema below (gist link or PR to
> `validation/community/<dataset>.json`), even better.

## Why we ask for this much

rustscenic's release notes only claim what's backed by an artefact
under `validation/`. Two real-data datasets (PBMC 3k v0.3.9 +
mouse brain E18 v0.3.10) currently anchor every "publishable
end-to-end" claim. The v0.4.x roadmap (see
[`docs/v0.4.x-benchmark-plan.md`](v0.4.x-benchmark-plan.md))
expands that to 6 datasets. Reports filed against this template
become candidate artefacts in that sweep.

The schema is decay-resistant: you can re-read a 2-year-old report
and know exactly what was run, on what data, on what hardware, with
what version, and what came out. That's the bar a published claim
needs to clear.

## Required fields (issue template)

| Field | Why we need it |
|---|---|
| Summary | One-line "what dataset, what release, what came out". |
| Dataset name + source/accession | So someone else can rerun it. URL or GEO/SRA accession. |
| Species + assembly | hg38 / mm10 / etc. — affects motif rankings and TF list. |
| Tissue / state | Determines which canonical TFs we'd expect. |
| Shapes after QC | RNA `cells × genes`; ATAC `cells × peaks` after subsetting. |
| rustscenic version | Pin to a release tag, not a commit. v0.4.0 or later. |
| Install command | Exact `pip install ...` line you used. |
| API call | The Python snippet, with all kwargs that affect the run. |
| Hardware + OS + Python | CPU, RAM, OS, Python, scanpy, anndata. |
| Headline output counts | Per-stage non-empty counts. All six stages must populate. |
| Biological sanity | Canonical TFs expected for this tissue + how many recovered. |

## Strongly preferred (optional but valuable)

- **Input file MD5s.** `md5sum` (Linux) or `md5` (macOS) of the
  fragments / RNA h5 / peaks BED. First-8MB MD5 is acceptable for
  very large fragments files. Without these, we cannot detect dataset
  drift between your run and a future re-run.
- **Per-stage wall times** (preproc, topics, GRN, cistarget,
  enhancer-link, eRegulon, AUCell). Folded into the scaling table.
- **Peak RSS in GB** (`psutil`, `/usr/bin/time -l`, etc.).
- **JSON artefact link.** A gist or PR with the JSON below.

## Canonical evidence-schema JSON

Match the schema in
[`validation/multiome_pipeline_run_v0.3.10_brain_e18.json`](../validation/multiome_pipeline_run_v0.3.10_brain_e18.json)
exactly. The shape:

```json
{
  "release": "v0.4.0",
  "smoke_type": "real_multiome_pipeline_run",
  "rustscenic_version": "0.4.0",
  "rustscenic_sha": "9ee67398689812f98bdf6856626ac57faf95be25",
  "install_command": "pip install \"rustscenic[validation] @ git+https://github.com/Ekin-Kahraman/rustscenic@v0.4.0\"",
  "api_call": "rustscenic.pipeline.run(rna=..., adata_atac=..., ...)",
  "dataset": {
    "name": "...",
    "source": "...",
    "species": "...",
    "tissue": "...",
    "rna_h5_md5": "...",
    "atac_fragments_md5_first_8mb": "...",
    "peaks_bed_md5": "..."
  },
  "shapes": {
    "rna_post_qc": [n_cells, n_genes],
    "atac_subset_to_rna_cells": [n_cells, n_peaks]
  },
  "wall_s": {"setup": 0.0, "pipeline_run_total": 0.0},
  "peak_rss_gb": 0.0,
  "outputs_non_empty": {
    "grn": true, "regulons": true, "cistarget": true,
    "enhancer_links": true, "eregulons": true, "integrated_adata": true
  },
  "headline_counts": {
    "n_grn_edges": 0, "n_regulons": 0, "n_cistarget_rows": 0,
    "n_enhancer_links": 0, "n_eregulons": 0
  },
  "biological_sanity": {
    "expected_tfs": ["..."],
    "found_in_regulons": ["..."],
    "missing_from_regulons": ["..."],
    "fraction_recovered": 0.0
  },
  "output_inventory": {
    "grn_path": {"path": "grn.parquet", "exists": true, "size_bytes": 0},
    "regulons_path": {"path": "regulons.json", "exists": true, "size_bytes": 0},
    "...": "..."
  },
  "elapsed_per_stage": {
    "preproc": 0.0, "topics": 0.0, "grn": 0.0,
    "cistarget": 0.0, "enhancer": 0.0, "eregulons": 0.0, "aucell": 0.0
  },
  "env": {
    "python": "3.13.9",
    "scanpy": "1.12.1",
    "anndata": "0.12.11",
    "os": "Darwin 25.4.0 arm64",
    "cpu": "Apple M5",
    "n_cpus": 10
  },
  "scope_notes": ["..."]
}
```

`scope_notes` is the place to record what you did differently from
the reference workflow — caller-side QC thresholds, atypical kwargs,
hardware notes, anything that affected the run.

## How to attach the JSON

Two paths, in order of preference:

1. **PR adding `validation/community/<dataset>.json`** — durable, gets a
   review, ends up in the repo. Best if your run is high-confidence and
   you're happy for the artefact to live alongside our own.
2. **Gist link** in the issue body — quickest. We may ask you to
   convert it to a PR if the run lands in the v0.4.x sweep.

Either way, the JSON itself is what we cite when we update release
notes or the benchmark plan.

## Worked example

Look at
[`validation/multiome_pipeline_run_v0.3.10_brain_e18.json`](../validation/multiome_pipeline_run_v0.3.10_brain_e18.json)
end to end. Every field in the issue template maps to a field in
that JSON, in the same order. The shape, the wording, the level of
detail — that's the bar.

If you're unsure whether a field applies, fill in what you have and
flag the rest in `scope_notes` / the "caveats" textarea.

## What happens after you file

1. We label the issue `validation` + `community-evidence`.
2. We sanity-check headline counts and biological recovery against
   what we'd expect for that tissue.
3. If the JSON is clean, we either link it from the v0.4.x sweep
   tracker or ask for a PR adding it under `validation/community/`.
4. Numbers from the run can appear in a future release-notes table
   (with attribution to your handle, unless you ask us not to). Pings
   you before publishing.

## What we do **not** want

- Screenshots of console output (the JSON is the durable record).
- Notebook outputs without a paired script. We need the *command*, not
  the *cell*.
- Claims without numbers ("it worked great" / "it was slow"). The
  schema exists so we can compare your run against the existing
  artefacts.
- Dataset uploads to the issue. Link the public source; for private
  data, paste a synthetic-data repro.

## Bug reports vs validation reports

| If | Use |
|---|---|
| The pipeline crashed, hung, or produced obviously wrong output | [Bug report](https://github.com/Ekin-Kahraman/rustscenic/issues/new?template=bug_report.yml) |
| Output disagrees with pyscenic / arboreto / pycisTopic / pycistarget | [Correctness issue](https://github.com/Ekin-Kahraman/rustscenic/issues/new?template=correctness.yml) |
| It ran end-to-end and you want to share results | This template — [Validation report](https://github.com/Ekin-Kahraman/rustscenic/issues/new?template=validation_report.yml) |

A run that finished but recovered 1/9 expected canonical TFs is a
correctness issue, not a validation report. A run that finished cleanly
with biology recovered is a validation report, even if some stage
emitted warnings.
