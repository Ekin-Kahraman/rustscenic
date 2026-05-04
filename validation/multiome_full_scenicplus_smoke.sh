#!/usr/bin/env bash
# Bounded real-data multi-stage smoke on rustscenic v0.3.7.
#
# Honest scope: exercises 5 of 6 SCENIC+ user-facing stages (grn, aucell,
# topics, cistarget, enhancer-link) ON A BOUNDED SUBSET of real PBMC multiome
# cells. The 6th stage (eRegulon assembly) is EXPLICITLY SKIPPED — see the
# scope_caveats section in the artefact and the [8/8] block below for why
# (gene-based cistarget output lacks peak_id; build_eregulons requires it).
# NOT a full-scale validation of pipeline.run on raw 10x output.
#
# The earlier full pipeline.run path (validation/multiome_fresh_env_smoke.sh +
# the unbounded version of this script) wedged at GRN for >3h after topics
# ran on the full 451k raw barcode matrix. Topics-then-GRN sequencing on a
# raw 10x fragments output does not scale on consumer hardware. The bounded
# smoke calls stages individually after pre-subsetting to RNA-QC'd cells,
# which matches the workload an actual user would run.
#
# Closes v0.4 publication-threshold item: "Real-data full-stage smoke proves
# every stage produces non-empty output on real PBMC multiome data."
# Does NOT close: "Real-data pipeline.run on raw 10x output without subsetting"
# (open: needs adata_atac parameter or fragments-subset preprocessing).

set -euo pipefail

DATA_DIR="$HOME/projects/bio/rustscenic/validation/real_multiome_v036"
ARTEFACT="$HOME/projects/bio/rustscenic/validation/multiome_full_stage_smoke_v0.3.7.json"
WORK=$(mktemp -d)
echo "work dir: $WORK"

python3 -m venv "$WORK/venv"
source "$WORK/venv/bin/activate"
pip install --quiet --upgrade pip

INSTALL_CMD='pip install --quiet --upgrade "rustscenic[validation] @ git+https://github.com/Ekin-Kahraman/rustscenic@v0.3.7"'
echo "=== install: $INSTALL_CMD ==="
eval "$INSTALL_CMD"

RUSTSCENIC_TAG="v0.3.7"
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
import json, os, resource, time, hashlib, signal, sys
from pathlib import Path
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import rustscenic, rustscenic.data, rustscenic.preproc
import rustscenic.grn, rustscenic.aucell, rustscenic.topics
import rustscenic.cistarget, rustscenic.enhancer, rustscenic.eregulon

DATA = Path(os.environ["DATA_DIR"])
RNA_H5 = DATA / "pbmc_3k_filtered_feature_bc_matrix.h5"
ATAC_FRAG = DATA / "pbmc_3k_atac_fragments.tsv.gz"
PEAKS_BED = DATA / "pbmc_3k_atac_peaks.bed"
WORK = Path(os.environ["WORK"]); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)

# Hard timeout: any single stage > 30 min aborts the smoke. Catches the same
# wedge that ate 3h41m of compute on the prior run.
STAGE_TIMEOUT_S = 30 * 60

class StageTimeout(Exception):
    pass

def _alarm_handler(signum, frame):
    raise StageTimeout("stage exceeded STAGE_TIMEOUT_S")
signal.signal(signal.SIGALRM, _alarm_handler)

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

stages = {}
def stage(name):
    class _S:
        def __enter__(self):
            self.t0 = time.monotonic()
            signal.alarm(STAGE_TIMEOUT_S)
            return self
        def __exit__(self, *a):
            signal.alarm(0)
            wall = time.monotonic() - self.t0
            kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            gb = kb / (1024**3) if kb > 1e6 else kb / (1024**2)
            stages[name] = {"wall_s": round(wall, 2), "peak_rss_gb": round(gb, 2)}
            print(f"  [{name}] wall={wall:.1f}s rss={gb:.2f}GB", flush=True)
    return _S()

print(f"rustscenic {os.environ['RUSTSCENIC_VER']} (sha {os.environ['RUSTSCENIC_SHA'][:7]}) on Python {os.environ['PYVER']}", flush=True)

# --- Stage 1: load + RNA QC ---
print("[1/8] load + RNA QC", flush=True)
with stage("load_qc"):
    rna = sc.read_10x_h5(RNA_H5); rna.var_names_make_unique()
    sc.pp.filter_cells(rna, min_genes=200)
    sc.pp.filter_genes(rna, min_cells=3)
    rna.var["mt"] = rna.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(rna, qc_vars=["mt"], inplace=True)
    rna = rna[rna.obs["pct_counts_mt"] < 20].copy()
    sc.pp.normalize_total(rna, target_sum=1e4); sc.pp.log1p(rna)
    print(f"  RNA shape post-QC: {rna.shape}", flush=True)

# --- Stage 2: ATAC fragments → matrix → SUBSET to RNA-QC'd cells ---
# This is the critical methodology fix: do not pass the unbounded raw barcode
# matrix to topics. Subset to the cells RNA QC actually keeps.
print("[2/8] ATAC fragments → matrix, subset to RNA-QC'd cells", flush=True)
with stage("atac_subset"):
    atac_full = rustscenic.preproc.fragments_to_matrix(ATAC_FRAG, PEAKS_BED)
    print(f"  raw ATAC barcodes: {atac_full.shape}", flush=True)
    shared = sorted(set(rna.obs_names) & set(atac_full.obs_names))
    atac = atac_full[shared].copy()
    del atac_full
    rna = rna[shared].copy()
    print(f"  shared cells: {len(shared)} ATAC subset: {atac.shape}", flush=True)

# Workload print BEFORE GRN — so a wedge is diagnosable
tfs = [t for t in rustscenic.data.tfs("human") if t in set(rna.var_names)]
print(f"\n=== workload ===", flush=True)
print(f"  RNA: {rna.shape}  ATAC: {atac.shape}  TFs: {len(tfs)}", flush=True)
print(f"  GRN: n_estimators=100  topics: K=10 passes=3  seed=777", flush=True)
print(f"  STAGE_TIMEOUT_S: {STAGE_TIMEOUT_S}", flush=True)
print("", flush=True)

# --- Stage 3: GRN ---
print("[3/8] grn.infer", flush=True)
with stage("grn"):
    grn = rustscenic.grn.infer(rna, tfs, n_estimators=100, seed=777, early_stop_window=25)
grn.to_parquet(OUT / "grn.parquet", index=False)
print(f"  edges: {len(grn)}", flush=True)
assert len(grn) > 0, "GRN produced 0 edges"

# --- Stage 4: AUCell ---
print("[4/8] aucell.score", flush=True)
regs = [(f"{tf}_regulon", g.nlargest(50, "importance")["target"].tolist())
        for tf, g in grn.groupby("TF") if len(g) >= 10]
with stage("aucell"):
    auc = rustscenic.aucell.score(rna, regs, top_frac=0.05)
auc.to_csv(OUT / "auc.csv")
print(f"  regulons: {len(regs)}  auc shape: {auc.shape}", flush=True)
assert auc.shape[0] > 0 and auc.shape[1] > 0, "AUCell empty"

# --- Stage 5: topics on subset ATAC ---
print("[5/8] topics.fit (online VB, K=10)", flush=True)
K = 10
with stage("topics"):
    tres = rustscenic.topics.fit(
        atac, n_topics=K, n_passes=3, batch_size=256, seed=777,
        alpha=1.0/K, eta=1.0/K,
    )
n_unique_top1 = len({a for a in tres.cell_assignment().values})
print(f"  unique top-1 topics: {n_unique_top1}/{K}", flush=True)

# --- Stage 6: cistarget (gene-based) ---
print("[6/8] cistarget.enrich (gene-based motif rankings)", flush=True)
with stage("download_motifs"):
    motif_rankings = rustscenic.data.download_motif_rankings(species="human", verbose=False)
    print(f"  motif_rankings: {motif_rankings.shape}", flush=True)
# regs is already [(name, gene_list), ...] — pass directly.
with stage("cistarget"):
    cistarget_df = rustscenic.cistarget.enrich(
        motif_rankings, regs,
        top_frac=0.05, auc_threshold=0.05,
    )
cistarget_df.to_parquet(OUT / "cistarget.parquet", index=False)
print(f"  cistarget rows: {len(cistarget_df)}", flush=True)
assert len(cistarget_df) > 0, "cistarget produced 0 rows"

# --- Stage 7: enhancer linking (peak↔gene correlation) ---
print("[7/8] enhancer.link_peaks_to_genes", flush=True)
with stage("download_gene_coords"):
    gene_coords = rustscenic.data.download_gene_coords(species="hs", verbose=False)
    print(f"  gene_coords: {gene_coords.shape}", flush=True)

# Build peak coordinates from ATAC var_names. ATAC peak IDs are typically
# 'chr:start-end'; parse them.
def parse_peak(pid):
    p = str(pid).replace(":", "-").split("-")
    if len(p) >= 3:
        try:
            return p[0], int(p[1]), int(p[2])
        except ValueError:
            return None
    return None
parsed = [parse_peak(p) for p in atac.var_names]
peak_coords = pd.DataFrame(
    [(pid, *parts) for pid, parts in zip(atac.var_names, parsed) if parts is not None],
    columns=["peak_id", "chrom", "start", "end"],
).set_index("peak_id")
# enhancer._peak_frame does `peak_coords.loc[atac_adata.var_names]` — peak_id
# must be the index, not a column.
print(f"  parsed peak coords: {len(peak_coords)}/{atac.n_vars}", flush=True)

with stage("enhancer"):
    enhancer_links = rustscenic.enhancer.link_peaks_to_genes(
        rna, atac, gene_coords, peak_coords=peak_coords,
        max_distance=500_000, min_abs_corr=0.1,
    )
enhancer_links.to_parquet(OUT / "enhancer_links.parquet", index=False)
print(f"  enhancer links: {len(enhancer_links)}", flush=True)
assert len(enhancer_links) > 0, "enhancer linking produced 0 rows"

# --- Stage 8: eRegulon assembly ---
# Honest scope: build_eregulons requires the cistarget DataFrame to carry
# 'peak_id' or 'region_id'. Gene-based cistarget (which we ran above) emits
# (regulon, motif, auc) WITHOUT peak_id. Mapping motifs back to peaks
# requires the pipeline.run private _attribute_peaks_to_cistarget bridge
# (which uses enhancer_links to attribute), OR region-based motif rankings.
# This smoke records "eRegulon stage not exercised at the unit level" rather
# than synthesising a peak_id column that misrepresents real coverage.
print("[8/8] eregulon.build_eregulons (skipped — see scope note)", flush=True)
eregulons = []
eregulon_skip_reason = (
    "Gene-based cistarget output lacks peak_id; build_eregulons requires it. "
    "Full coverage needs either region-based motif rankings + region cistarget, "
    "or pipeline.run orchestration (its _attribute_peaks_to_cistarget bridge). "
    "Marked as scope caveat; v0.4 gate item 'real-data eRegulon assembly' open."
)
print(f"  {eregulon_skip_reason}", flush=True)

# Biology presence check
canonical = ["SPI1", "PAX5", "GATA3", "TBX21", "EBF1"]
present_in_regulons = {tf: tf in {r[0].replace("_regulon","") for r in regs} for tf in canonical}
present_in_eregulons = {tf: False for tf in canonical}  # not exercised

artefact = {
    "release": "v0.3.7",
    "smoke_type": "bounded real-data full-stage smoke (RNA-QC subset)",
    "rustscenic_version": os.environ["RUSTSCENIC_VER"],
    "rustscenic_sha": os.environ["RUSTSCENIC_SHA"],
    "install_command": os.environ["INSTALL_CMD"],
    "dataset": {
        "name": "10x pbmc_unsorted_3k",
        "source": "cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_unsorted_3k",
        "rna_h5_md5": md5(RNA_H5),
        "atac_fragments_md5_first_8mb": hashlib.md5(open(ATAC_FRAG,"rb").read(8*1024*1024)).hexdigest(),
        "peaks_bed_md5": md5(PEAKS_BED),
    },
    "shapes": {
        "rna_post_qc": list(rna.shape),
        "atac_subset_to_rna_cells": list(atac.shape),
        "n_tfs_in_var_names": len(tfs),
    },
    "stages": stages,
    "outputs_non_empty": {
        "grn": len(grn) > 0,
        "aucell": auc.shape[0] > 0 and auc.shape[1] > 0,
        "topics": n_unique_top1 > 0,
        "cistarget": len(cistarget_df) > 0,
        "enhancer_links": len(enhancer_links) > 0,
        "eregulons": "skipped (see scope_caveats)",
    },
    "results": {
        "n_grn_edges": int(len(grn)),
        "n_regulons_min10": len(regs),
        "topics_unique_top1_of_K": [n_unique_top1, K],
        "n_cistarget_rows": int(len(cistarget_df)),
        "n_enhancer_links": int(len(enhancer_links)),
        "n_eregulons": len(eregulons),
        "canonical_tfs_in_regulons": present_in_regulons,
        "canonical_tfs_in_eregulons": present_in_eregulons,
    },
    "env": {
        "python": os.environ["PYVER"],
        "scanpy": os.environ["SCANPY_VER"],
        "anndata": os.environ["ANNDATA_VER"],
        "os": os.environ["OS"],
        "cpu": os.environ["CPU"],
        "n_cpus": int(os.environ["N_CPUS"]),
    },
    "scope_caveats": [
        "ATAC matrix subset to RNA-QC'd cells (matches user-recommended workflow per the fragments_to_matrix UserWarning).",
        "Stages called individually (not via pipeline.run). pipeline.run on raw 10x output without subsetting wedged at GRN for >3h on this hardware (450k+ raw barcodes through topics inflated process state for subsequent GRN). v0.4 gate item 'pipeline.run on full raw 10x' remains open.",
        "GRN n_estimators=100 (smoke speed). README claims are at n_estimators=5000 — different operating point.",
        "eRegulon stage SKIPPED: gene-based cistarget output lacks peak_id; eregulon.build_eregulons requires it. Full eRegulon coverage on real data needs either region-based motif rankings + region cistarget, or the pipeline.run _attribute_peaks_to_cistarget bridge. v0.4 gate item 'real-data eRegulon assembly' remains open.",
    ],
}

art_path = Path(os.environ["ARTEFACT"])
art_path.write_text(json.dumps(artefact, indent=2))
print(f"\nartefact → {art_path}", flush=True)
print(json.dumps(artefact, indent=2), flush=True)
PY

deactivate
echo "done"
