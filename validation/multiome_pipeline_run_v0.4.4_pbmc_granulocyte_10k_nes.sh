#!/usr/bin/env bash
# Real-data NES validation for rustscenic v0.4.4 on 10x PBMC granulocyte-sorted
# 10k human multiome. Apples-to-apples vs the v0.4.3 artefact (same data, same
# hardware, same pipeline parameters), only cistarget_nes_threshold=3.0 is added.
#
# Companion to validation/multiome_pipeline_run_v0.4.3_pbmc_granulocyte_10k.sh.
# That script established the v0.4.3 baseline: 10 of 10 canonical TFs and
# 1,578,204 enriched cistarget rows. This script adds NES filtering at the
# pyscenic / pycistarget canonical cutoff of 3.0 and records the selectivity
# of the filter alongside the same biology check.
#
# Dataset: 10x cellranger-arc 2.0.0 pbmc_granulocyte_sorted_10k
#   https://www.10xgenomics.com/datasets/pbmc-from-a-healthy-donor-granulocytes-removed-through-cell-sorting-10-k-1-standard-2-0-0

set -euo pipefail

DATA_DIR="$HOME/projects/bio/rustscenic/validation/real_multiome_pbmc_granulocyte_10k"
ARTEFACT="$HOME/projects/bio/rustscenic/validation/multiome_pipeline_run_v0.4.4_pbmc_granulocyte_10k_nes.json"
REPO="$HOME/projects/bio/rustscenic"
WORK=$(mktemp -d)
echo "work dir: $WORK"

python3 -m venv "$WORK/venv"
source "$WORK/venv/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet "${REPO}[validation]"

RUSTSCENIC_SHA=$(cd "$REPO" && git rev-parse HEAD)
RUSTSCENIC_BRANCH=$(cd "$REPO" && git rev-parse --abbrev-ref HEAD)
PYVER=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:3])))")
SCANPY_VER=$(python -c "from importlib.metadata import version; print(version('scanpy'))")
ANNDATA_VER=$(python -c "from importlib.metadata import version; print(version('anndata'))")
RUSTSCENIC_VER=$(python -c "from importlib.metadata import version; print(version('rustscenic'))")
OS=$(uname -srm)
CPU=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || cat /proc/cpuinfo 2>/dev/null | grep -m1 "model name" | sed 's/.*: //' || echo "unknown")
N_CPUS=$(python -c "import os; print(os.cpu_count())")
echo "rustscenic=${RUSTSCENIC_VER} branch=${RUSTSCENIC_BRANCH} sha=${RUSTSCENIC_SHA} cpus=${N_CPUS}"

WORK="$WORK" RUSTSCENIC_SHA="$RUSTSCENIC_SHA" RUSTSCENIC_BRANCH="$RUSTSCENIC_BRANCH" PYVER="$PYVER" \
SCANPY_VER="$SCANPY_VER" ANNDATA_VER="$ANNDATA_VER" RUSTSCENIC_VER="$RUSTSCENIC_VER" \
OS="$OS" CPU="$CPU" N_CPUS="$N_CPUS" \
DATA_DIR="$DATA_DIR" ARTEFACT="$ARTEFACT" \
python - <<'PY'
import json, os, resource, time, hashlib, signal
from pathlib import Path
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import rustscenic, rustscenic.data, rustscenic.preproc, rustscenic.pipeline

DATA = Path(os.environ["DATA_DIR"])
RNA_H5 = DATA / "pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5"
ATAC_FRAG = DATA / "pbmc_granulocyte_sorted_10k_atac_fragments.tsv.gz"
PEAKS_BED = DATA / "pbmc_granulocyte_sorted_10k_atac_peaks.bed"
WORK = Path(os.environ["WORK"]); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)

TOTAL_TIMEOUT_S = 90 * 60

class Timeout(Exception): pass
def _h(s, f): raise Timeout("pipeline.run exceeded TOTAL_TIMEOUT_S")
signal.signal(signal.SIGALRM, _h)

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

print(f"rustscenic {os.environ['RUSTSCENIC_VER']} on Python {os.environ['PYVER']}", flush=True)

t_setup = time.monotonic()
rna = sc.read_10x_h5(RNA_H5); rna.var_names_make_unique()
sc.pp.filter_cells(rna, min_genes=200)
sc.pp.filter_genes(rna, min_cells=3)
rna.var["mt"] = rna.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(rna, qc_vars=["mt"], inplace=True)
rna = rna[rna.obs["pct_counts_mt"] < 20].copy()
sc.pp.normalize_total(rna, target_sum=1e4); sc.pp.log1p(rna)
print(f"RNA shape post-QC: {rna.shape}", flush=True)

atac_full = rustscenic.preproc.fragments_to_matrix(ATAC_FRAG, PEAKS_BED)
shared = sorted(set(rna.obs_names) & set(atac_full.obs_names))
adata_atac = atac_full[shared].copy()
del atac_full
rna = rna[shared].copy()
print(f"ATAC subset: {adata_atac.shape}  shared cells: {len(shared):,}", flush=True)

motif_rankings = rustscenic.data.download_motif_rankings(species="human", verbose=False)
gene_coords = rustscenic.data.download_gene_coords(species="hs", verbose=False)
hs_tfs = rustscenic.data.tfs(species="hs")
setup_wall = time.monotonic() - t_setup
print(f"setup wall: {setup_wall:.1f}s", flush=True)

signal.alarm(TOTAL_TIMEOUT_S)
t0 = time.monotonic()
try:
    result = rustscenic.pipeline.run(
        rna=rna,
        output_dir=OUT,
        adata_atac=adata_atac,
        motif_rankings=motif_rankings,
        gene_coords=gene_coords,
        tfs=hs_tfs,
        grn_n_estimators=100,
        topics_n_topics=15,
        topics_n_passes=3,
        cistarget_top_frac=0.05,
        cistarget_auc_threshold=0.05,
        cistarget_nes_threshold=3.0,
        enhancer_max_distance=500_000,
        enhancer_min_abs_corr=0.1,
        eregulon_min_target_genes=2,
        eregulon_min_enhancer_links=1,
        seed=777,
        verbose=True,
    )
finally:
    signal.alarm(0)
pipeline_wall = time.monotonic() - t0
peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak_rss_gb = peak_rss_kb / (1024**3) if peak_rss_kb > 1e6 else peak_rss_kb / (1024**2)

cistarget_df = pd.read_parquet(result.cistarget_path)
n_post_nes = len(cistarget_df)
nes_min = float(cistarget_df["nes"].min()) if n_post_nes > 0 else None
nes_max = float(cistarget_df["nes"].max()) if n_post_nes > 0 else None
nes_median = float(cistarget_df["nes"].median()) if n_post_nes > 0 else None

import rustscenic.cistarget as rs_ct
candidate_regulons = json.loads(Path(result.candidate_regulons_path).read_text())
t1 = time.monotonic()
enrich_no_nes = rs_ct.enrich(
    motif_rankings,
    [(n, g) for n, g in candidate_regulons.items()],
    top_frac=0.05, auc_threshold=0.05,
)
no_nes_wall = time.monotonic() - t1
n_pre_nes = len(enrich_no_nes)
selectivity = (n_post_nes / n_pre_nes) if n_pre_nes else None

expected_tfs = ["SPI1","CEBPA","CEBPB","CEBPE","PAX5","GATA3","TBX21","EBF1","IRF8","FOXP3"]
regulons_json = json.loads(Path(result.regulons_path).read_text())
def _strip_suffixes(name):
    s = name
    while True:
        prev = s
        for suf in ("_regulon","_extended","_activator","_repressor"):
            if s.endswith(suf): s = s[:-len(suf)].strip()
        for paren in ("(+)","(-)"):
            if s.endswith(paren): s = s[:-3].strip()
        if s == prev: return s
regulon_tfs = {_strip_suffixes(k) for k in regulons_json.keys()}
found_tfs = sorted([t for t in expected_tfs if t in regulon_tfs])
missing_tfs = sorted([t for t in expected_tfs if t not in regulon_tfs])

n_grn_edges = int(pd.read_parquet(result.grn_path).shape[0])
n_regulons = int(getattr(result, "n_regulons", 0))
n_eregulons = int(getattr(result, "n_eregulons", 0) or 0)
n_candidate_regulons = int(getattr(result, "n_candidate_regulons", 0) or 0)
n_enhancer_rows = int(pd.read_parquet(result.enhancer_links_path).shape[0])

artefact = {
    "release": os.environ["RUSTSCENIC_VER"],
    "smoke_type": "real-data full SCENIC+ E2E via public pipeline.run with cistarget_nes_threshold=3.0 (v0.4.4 NES first real-data exercise)",
    "rustscenic_version": os.environ["RUSTSCENIC_VER"],
    "rustscenic_sha": os.environ["RUSTSCENIC_SHA"],
    "rustscenic_branch": os.environ["RUSTSCENIC_BRANCH"],
    "install_command": f"pip install \"{os.environ['HOME']}/projects/bio/rustscenic[validation]\"",
    "dataset": {
        "name": "10x pbmc_granulocyte_sorted_10k",
        "source": "cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_10k",
        "species": "Homo sapiens (hg38)",
        "rna_h5_md5": md5(RNA_H5),
        "atac_fragments_md5_first_8mb": hashlib.md5(open(ATAC_FRAG,"rb").read(8*1024*1024)).hexdigest(),
        "peaks_bed_md5": md5(PEAKS_BED),
    },
    "shapes": {
        "rna_post_qc": list(rna.shape),
        "atac_subset_to_rna_cells": list(adata_atac.shape),
    },
    "wall_s": {
        "setup": round(setup_wall, 2),
        "pipeline_run_total": round(pipeline_wall, 2),
        "no_nes_recheck": round(no_nes_wall, 2),
    },
    "peak_rss_gb": round(peak_rss_gb, 2),
    "headline_counts": {
        "n_grn_edges": n_grn_edges,
        "n_candidate_regulons": n_candidate_regulons,
        "n_regulons": n_regulons,
        "n_eregulons": n_eregulons,
        "n_cistarget_rows_pre_nes": n_pre_nes,
        "n_cistarget_rows_post_nes": n_post_nes,
        "n_enhancer_links": n_enhancer_rows,
    },
    "nes_filter_effect": {
        "auc_threshold": 0.05,
        "nes_threshold": 3.0,
        "pre_nes_rows": n_pre_nes,
        "post_nes_rows": n_post_nes,
        "selectivity_ratio": round(selectivity, 4) if selectivity else None,
        "nes_min_in_kept_set": round(nes_min, 3) if nes_min is not None else None,
        "nes_max_in_kept_set": round(nes_max, 3) if nes_max is not None else None,
        "nes_median_in_kept_set": round(nes_median, 3) if nes_median is not None else None,
    },
    "biological_sanity_under_nes": {
        "expected_pbmc_granulocyte_tfs": expected_tfs,
        "found_in_regulons": found_tfs,
        "missing_from_regulons": missing_tfs,
        "fraction_recovered": round(len(found_tfs) / len(expected_tfs), 3),
        "v043_baseline_recovery": 10,
        "delta_vs_v043": len(found_tfs) - 10,
    },
    "env": {
        "python": os.environ["PYVER"],
        "scanpy": os.environ["SCANPY_VER"],
        "anndata": os.environ["ANNDATA_VER"],
        "os": os.environ["OS"],
        "cpu": os.environ["CPU"],
        "n_cpus": int(os.environ["N_CPUS"]),
    },
}

Path(os.environ["ARTEFACT"]).write_text(json.dumps(artefact, indent=2))
print(f"artefact -> {os.environ['ARTEFACT']}", flush=True)
print(json.dumps(artefact, indent=2), flush=True)
PY

deactivate
echo "done"
