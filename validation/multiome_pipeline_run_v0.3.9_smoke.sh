#!/usr/bin/env bash
# Real-data full SCENIC+ end-to-end via the PUBLIC ORCHESTRATOR on rustscenic v0.3.9.
#
# v0.3.9 carries the alt-contig peak-name regex fix (KI270721.1, GL000220.1, ...)
# in python/rustscenic/enhancer.py — without this fix, raw 10x ATAC peaks BED
# crashed pipeline.run at the enhancer stage on real PBMC data. v0.3.8 added
# the adata_atac param (orchestrator pre-built ATAC); v0.3.9 unblocks the
# enhancer stage so the orchestrator can finish.
#
# This smoke proves the v0.4 gate-closing path: a single rustscenic.pipeline.run
# call on real PBMC multiome that emits every final SCENIC+ artefact, INCLUDING
# eRegulons, without manual stage wiring.
#
# Differs from validation/multiome_full_scenicplus_smoke.sh: that one calls
# stages individually and explicitly skips eRegulon (gene-based cistarget
# lacks peak_id at the unit level). This one delegates to pipeline.run, which
# carries the gene-cistarget → peak attribution bridge internally and feeds
# eRegulon assembly.
#
# Closes (when n_eregulons > 0):
#   - "Real-data eRegulon assembly" gate item
#   - "One public command/API proof" gate item

set -euo pipefail

DATA_DIR="$HOME/projects/bio/rustscenic/validation/real_multiome_v036"
ARTEFACT="$HOME/projects/bio/rustscenic/validation/multiome_pipeline_run_v0.3.9.json"
WORK=$(mktemp -d)
echo "work dir: $WORK"

python3 -m venv "$WORK/venv"
source "$WORK/venv/bin/activate"
pip install --quiet --upgrade pip

INSTALL_CMD='pip install --quiet --upgrade "rustscenic[validation] @ git+https://github.com/Ekin-Kahraman/rustscenic@v0.3.9"'
echo "=== install: $INSTALL_CMD ==="
eval "$INSTALL_CMD"

RUSTSCENIC_TAG="v0.3.9"
RUSTSCENIC_SHA=$(git ls-remote https://github.com/Ekin-Kahraman/rustscenic.git "refs/tags/${RUSTSCENIC_TAG}^{}" | awk '{print $1}')
PYVER=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:3])))")
SCANPY_VER=$(python -c "from importlib.metadata import version; print(version('scanpy'))")
ANNDATA_VER=$(python -c "from importlib.metadata import version; print(version('anndata'))")
RUSTSCENIC_VER=$(python -c "from importlib.metadata import version; print(version('rustscenic'))")
OS=$(uname -srm)
CPU=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || cat /proc/cpuinfo 2>/dev/null | grep -m1 "model name" | sed 's/.*: //' || echo "unknown")
N_CPUS=$(python -c "import os; print(os.cpu_count())")
echo "tag=${RUSTSCENIC_TAG} sha=${RUSTSCENIC_SHA} cpus=${N_CPUS}"

WORK="$WORK" RUSTSCENIC_SHA="$RUSTSCENIC_SHA" PYVER="$PYVER" \
SCANPY_VER="$SCANPY_VER" ANNDATA_VER="$ANNDATA_VER" RUSTSCENIC_VER="$RUSTSCENIC_VER" \
OS="$OS" CPU="$CPU" N_CPUS="$N_CPUS" \
DATA_DIR="$DATA_DIR" ARTEFACT="$ARTEFACT" \
INSTALL_CMD="$INSTALL_CMD" \
python - <<'PY'
import json, os, resource, time, hashlib, signal
from pathlib import Path
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import rustscenic, rustscenic.data, rustscenic.preproc, rustscenic.pipeline

DATA = Path(os.environ["DATA_DIR"])
RNA_H5 = DATA / "pbmc_3k_filtered_feature_bc_matrix.h5"
ATAC_FRAG = DATA / "pbmc_3k_atac_fragments.tsv.gz"
PEAKS_BED = DATA / "pbmc_3k_atac_peaks.bed"
WORK = Path(os.environ["WORK"]); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)

# Hard total timeout: pipeline.run in one block, must complete or fail in 30 min.
TOTAL_TIMEOUT_S = 30 * 60

class Timeout(Exception): pass
def _h(s, f): raise Timeout("pipeline.run exceeded TOTAL_TIMEOUT_S")
signal.signal(signal.SIGALRM, _h)

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

print(f"rustscenic {os.environ['RUSTSCENIC_VER']} (sha {os.environ['RUSTSCENIC_SHA'][:7]}) on Python {os.environ['PYVER']}", flush=True)

# --- Pre-pipeline setup: build RNA + ATAC inputs ---
t_setup = time.monotonic()
print("[setup 1/3] load + RNA QC", flush=True)
rna = sc.read_10x_h5(RNA_H5); rna.var_names_make_unique()
sc.pp.filter_cells(rna, min_genes=200)
sc.pp.filter_genes(rna, min_cells=3)
rna.var["mt"] = rna.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(rna, qc_vars=["mt"], inplace=True)
rna = rna[rna.obs["pct_counts_mt"] < 20].copy()
sc.pp.normalize_total(rna, target_sum=1e4); sc.pp.log1p(rna)
print(f"  RNA shape post-QC: {rna.shape}", flush=True)

print("[setup 2/3] build ATAC matrix and subset to RNA-QC'd cells", flush=True)
atac_full = rustscenic.preproc.fragments_to_matrix(ATAC_FRAG, PEAKS_BED)
shared = sorted(set(rna.obs_names) & set(atac_full.obs_names))
adata_atac = atac_full[shared].copy()
del atac_full
rna = rna[shared].copy()
print(f"  ATAC subset: {adata_atac.shape}", flush=True)

print("[setup 3/3] download motif rankings + gene coords (cached)", flush=True)
motif_rankings = rustscenic.data.download_motif_rankings(species="human", verbose=False)
gene_coords = rustscenic.data.download_gene_coords(species="hs", verbose=False)
print(f"  motif_rankings: {motif_rankings.shape}  gene_coords: {gene_coords.shape}", flush=True)

setup_wall = time.monotonic() - t_setup
print(f"  setup wall: {setup_wall:.1f}s", flush=True)

# --- Single pipeline.run call (the public-orchestrator proof) ---
print("\n[pipeline.run] full SCENIC+ E2E via the public API", flush=True)
print(f"  timeout: {TOTAL_TIMEOUT_S}s", flush=True)
signal.alarm(TOTAL_TIMEOUT_S)
t0 = time.monotonic()
try:
    result = rustscenic.pipeline.run(
        rna=rna,
        output_dir=OUT,
        adata_atac=adata_atac,            # the new param: skip fragments_to_matrix
        motif_rankings=motif_rankings,
        gene_coords=gene_coords,
        grn_n_estimators=100,
        topics_n_topics=10,
        topics_n_passes=3,
        cistarget_top_frac=0.05,
        cistarget_auc_threshold=0.05,
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
print(f"\n[pipeline.run done] wall={pipeline_wall:.1f}s peak_rss={peak_rss_gb:.2f}GB", flush=True)

# --- Inspect ALL output artefacts ---
def file_info(p):
    if p is None: return None
    p = Path(p)
    return {"path": p.name, "exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else None}

artefact_paths = {
    "atac_matrix_path": result.atac_matrix_path,
    "grn_path": result.grn_path,
    "regulons_path": result.regulons_path,
    "aucell_path": result.aucell_path,
    "topics_dir": result.topics_dir,
    "cistarget_path": result.cistarget_path,
    "enhancer_links_path": result.enhancer_links_path,
    "eregulons_path": result.eregulons_path,
    "integrated_adata_path": result.integrated_adata_path,
}
output_inventory = {k: file_info(v) for k, v in artefact_paths.items()}
print("\n=== output inventory ===", flush=True)
for k, v in output_inventory.items():
    print(f"  {k}: {v}", flush=True)

# --- Headline numbers ---
n_grn_edges = int(pd.read_parquet(result.grn_path).shape[0]) if result.grn_path else 0
n_regulons = int(getattr(result, "n_regulons", 0))
n_eregulons = int(getattr(result, "n_eregulons", 0) or 0)
n_cistarget_rows = int(pd.read_parquet(result.cistarget_path).shape[0]) if result.cistarget_path else 0
n_enhancer_rows = int(pd.read_parquet(result.enhancer_links_path).shape[0]) if result.enhancer_links_path else 0

print(f"\n=== headline ===", flush=True)
print(f"  GRN edges: {n_grn_edges:,}", flush=True)
print(f"  regulons: {n_regulons:,}", flush=True)
print(f"  cistarget rows: {n_cistarget_rows:,}", flush=True)
print(f"  enhancer links: {n_enhancer_rows:,}", flush=True)
print(f"  eRegulons: {n_eregulons:,}", flush=True)

# --- Hard assertion: this is the gate the smoke claims to close ---
assert n_eregulons > 0, (
    f"FAIL: pipeline.run produced 0 eRegulons on real data. "
    f"That's the gate this smoke claims to close. Investigate the gene→peak "
    f"attribution bridge or thresholds (eregulon_min_target_genes, "
    f"eregulon_min_enhancer_links, cistarget_auc_threshold)."
)

artefact = {
    "release": "v0.3.9",
    "smoke_type": "real-data full SCENIC+ E2E via public pipeline.run",
    "rustscenic_version": os.environ["RUSTSCENIC_VER"],
    "rustscenic_sha": os.environ["RUSTSCENIC_SHA"],
    "install_command": os.environ["INSTALL_CMD"],
    "api_call": (
        "rustscenic.pipeline.run(rna=adata, output_dir=..., adata_atac=adata_atac, "
        "motif_rankings=..., gene_coords=..., grn_n_estimators=100, topics_n_topics=10, "
        "topics_n_passes=3, eregulon_min_target_genes=2, eregulon_min_enhancer_links=1, seed=777)"
    ),
    "dataset": {
        "name": "10x pbmc_unsorted_3k",
        "source": "cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_unsorted_3k",
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
    },
    "peak_rss_gb": round(peak_rss_gb, 2),
    "outputs_non_empty": {
        "grn": n_grn_edges > 0,
        "regulons": n_regulons > 0,
        "cistarget": n_cistarget_rows > 0,
        "enhancer_links": n_enhancer_rows > 0,
        "eregulons": n_eregulons > 0,
        "integrated_adata": result.integrated_adata_path is not None and Path(result.integrated_adata_path).exists(),
    },
    "headline_counts": {
        "n_grn_edges": n_grn_edges,
        "n_regulons": n_regulons,
        "n_cistarget_rows": n_cistarget_rows,
        "n_enhancer_links": n_enhancer_rows,
        "n_eregulons": n_eregulons,
    },
    "output_inventory": output_inventory,
    "elapsed_per_stage": result.elapsed if hasattr(result, "elapsed") else {},
    "env": {
        "python": os.environ["PYVER"],
        "scanpy": os.environ["SCANPY_VER"],
        "anndata": os.environ["ANNDATA_VER"],
        "os": os.environ["OS"],
        "cpu": os.environ["CPU"],
        "n_cpus": int(os.environ["N_CPUS"]),
    },
    "scope_notes": [
        "Pre-pipeline setup (RNA QC, ATAC subset, motif/gene-coords download) is recorded separately from pipeline.run wall time.",
        "ATAC matrix is pre-subset to RNA-QC'd cells (the v0.3.8 adata_atac path). Raw 10x fragments has ~451k empty-droplet barcodes that must not be carried through topics — this is what the API change unblocks.",
        "v0.3.9 ships an enhancer-stage regex fix that allows alt-contig peak names (KI270721.1, GL000220.1, ...) which appear in raw 10x ATAC peaks BED. Without this fix, pipeline.run crashed at link_peaks_to_genes on this dataset.",
        "GRN n_estimators=100 (smoke speed); README claims at n_estimators=5000 — different operating point.",
        "eRegulon thresholds set permissively (min_target_genes=2, min_enhancer_links=1) for this proof-of-orchestration smoke. Production thresholds in test_pipeline_integration use 5/2 defaults.",
    ],
}

art_path = Path(os.environ["ARTEFACT"])
art_path.write_text(json.dumps(artefact, indent=2))
print(f"\nartefact → {art_path}", flush=True)
print(json.dumps(artefact, indent=2), flush=True)
PY

deactivate
echo "done"
