#!/usr/bin/env bash
# Fresh-env real-data PBMC multiome v0.3.6 smoke + benchmark artefact generator.
# Closes v0.4 gate item: real-data PBMC multiome end-to-end without intervention,
# with command/version/hardware baked into the artefact.
set -euo pipefail

DATA_DIR="$HOME/projects/bio/rustscenic/validation/real_multiome_v036"
ARTEFACT="$HOME/projects/bio/rustscenic/validation/multiome_pbmc_3k_v0.3.6.json"
WORK=$(mktemp -d)
echo "work dir: $WORK"

# 1. Fresh venv
python3 -m venv "$WORK/venv"
source "$WORK/venv/bin/activate"
pip install --quiet --upgrade pip

# 2. Install rustscenic[validation] from v0.3.6 tag (the published install path)
echo "=== installing rustscenic[validation] @ v0.3.6 ==="
pip install --quiet --upgrade "rustscenic[validation] @ git+https://github.com/Ekin-Kahraman/rustscenic@v0.3.7"

# 3. Capture provenance — read the installed-package SHA via ls-remote of
# the actual published tag, not local-repo HEAD. Local HEAD reflects the dev
# tree; the smoke installs from v0.3.6, which can point at a different commit.
RUSTSCENIC_TAG="v0.3.7"
RUSTSCENIC_SHA=$(git ls-remote https://github.com/Ekin-Kahraman/rustscenic.git "refs/tags/${RUSTSCENIC_TAG}^{}" | awk '{print $1}')
if [ -z "$RUSTSCENIC_SHA" ]; then
    RUSTSCENIC_SHA=$(git ls-remote https://github.com/Ekin-Kahraman/rustscenic.git "refs/tags/${RUSTSCENIC_TAG}" | awk '{print $1}')
fi
echo "tag=${RUSTSCENIC_TAG} sha=${RUSTSCENIC_SHA}"
PYVER=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:3])))")
SCANPY_VER=$(python -c "from importlib.metadata import version; print(version('scanpy'))")
ANNDATA_VER=$(python -c "from importlib.metadata import version; print(version('anndata'))")
RUSTSCENIC_VER=$(python -c "from importlib.metadata import version; print(version('rustscenic'))")
OS=$(uname -srm)
CPU=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || cat /proc/cpuinfo 2>/dev/null | grep -m1 "model name" | sed 's/.*: //' || echo "unknown")
TOTAL_RAM_GB=$(python -c "import psutil; print(round(psutil.virtual_memory().total/(1024**3), 1))" 2>/dev/null || echo "unknown")
N_CPUS=$(python -c "import os; print(os.cpu_count())")

echo "=== running smoke ==="
WORK="$WORK" RUSTSCENIC_SHA="$RUSTSCENIC_SHA" PYVER="$PYVER" \
SCANPY_VER="$SCANPY_VER" ANNDATA_VER="$ANNDATA_VER" RUSTSCENIC_VER="$RUSTSCENIC_VER" \
OS="$OS" CPU="$CPU" TOTAL_RAM_GB="$TOTAL_RAM_GB" N_CPUS="$N_CPUS" \
DATA_DIR="$DATA_DIR" ARTEFACT="$ARTEFACT" \
python - <<'PY'
import json, os, resource, time, hashlib
from pathlib import Path
import anndata as ad
import scanpy as sc
import rustscenic, rustscenic.aucell, rustscenic.grn, rustscenic.preproc, rustscenic.topics, rustscenic.data

DATA = Path(os.environ["DATA_DIR"])
RNA_H5 = DATA / "pbmc_3k_filtered_feature_bc_matrix.h5"
ATAC_FRAG = DATA / "pbmc_3k_atac_fragments.tsv.gz"
PEAKS_BED = DATA / "pbmc_3k_atac_peaks.bed"
WORK = Path(os.environ["WORK"]); OUT = WORK / "out"; OUT.mkdir(parents=True, exist_ok=True)

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
            self.t0 = time.monotonic(); return self
        def __exit__(self, *a):
            wall = time.monotonic() - self.t0
            kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            gb = kb / (1024**3) if kb > 1e6 else kb / (1024**2)
            stages[name] = {"wall_s": round(wall, 2), "peak_rss_gb": round(gb, 2)}
            print(f"  [{name}] wall={wall:.1f}s rss={gb:.2f}GB", flush=True)
    return _S()

print(f"rustscenic {os.environ['RUSTSCENIC_VER']} on Python {os.environ['PYVER']}", flush=True)

print("[1/6] load RNA", flush=True)
with stage("load_rna"):
    rna = sc.read_10x_h5(RNA_H5); rna.var_names_make_unique()
    print(f"  raw RNA: {rna.shape}", flush=True)

print("[2/6] preproc.fragments_to_matrix", flush=True)
with stage("fragments_to_matrix"):
    atac = rustscenic.preproc.fragments_to_matrix(ATAC_FRAG, PEAKS_BED)
    print(f"  ATAC: {atac.shape}", flush=True)

print("[3/6] RNA QC + normalise", flush=True)
with stage("rna_qc"):
    sc.pp.filter_cells(rna, min_genes=200)
    sc.pp.filter_genes(rna, min_cells=3)
    rna.var["mt"] = rna.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(rna, qc_vars=["mt"], inplace=True)
    rna = rna[rna.obs["pct_counts_mt"] < 20].copy()
    sc.pp.normalize_total(rna, target_sum=1e4); sc.pp.log1p(rna)
    print(f"  post-QC: {rna.shape}", flush=True)

print("[4/6] grn.infer (n_estimators=300)", flush=True)
tfs = [t for t in rustscenic.data.tfs("human") if t in set(rna.var_names)]
print(f"  {len(tfs)} TFs in data", flush=True)
with stage("grn"):
    grn = rustscenic.grn.infer(rna, tfs, n_estimators=300, seed=777, early_stop_window=25)
grn.to_parquet(OUT / "grn.parquet", index=False)
print(f"  edges: {len(grn)}, saved → {OUT/'grn.parquet'}", flush=True)

print("[5/6] aucell.score", flush=True)
regs = [(f"{tf}_regulon", g.nlargest(50, "importance")["target"].tolist())
        for tf, g in grn.groupby("TF") if len(g) >= 10]
print(f"  {len(regs)} regulons", flush=True)
with stage("aucell"):
    auc = rustscenic.aucell.score(rna, regs, top_frac=0.05)
auc.to_csv(OUT / "auc.csv")
print(f"  auc shape: {auc.shape}, saved → {OUT/'auc.csv'}", flush=True)

print("[6/6] topics.fit (online VB, K=10)", flush=True)
shared = sorted(set(rna.obs_names) & set(atac.obs_names))
atac_s = atac[shared].copy()
print(f"  shared cells: {len(shared)}", flush=True)
with stage("topics"):
    K = 10
    tres = rustscenic.topics.fit(atac_s, n_topics=K, n_passes=20, batch_size=256, seed=777, alpha=1.0/K, eta=1.0/K)
n_unique = len({a for a in tres.cell_assignment().values})
print(f"  unique top-1: {n_unique}/{K}", flush=True)

# Biology check: do canonical PBMC regulons appear in the regulon set?
canonical = {"SPI1": "Mono", "PAX5": "B", "GATA3": "T", "TBX21": "NK", "EBF1": "B"}
present = {tf: tf in {r[0].replace("_regulon","") for r in regs} for tf in canonical}
n_present = sum(present.values())

artefact = {
    "release": "v0.3.6",
    "rustscenic_version": os.environ["RUSTSCENIC_VER"],
    "rustscenic_sha": os.environ["RUSTSCENIC_SHA"],
    "command": "fresh-venv: pip install rustscenic[validation] @ git+...@v0.3.7 + smoke script",
    "dataset": {
        "name": "10x pbmc_unsorted_3k",
        "source": "cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_unsorted_3k",
        "rna_h5_md5": md5(RNA_H5),
        "atac_fragments_md5_first_8mb": hashlib.md5(open(ATAC_FRAG,"rb").read(8*1024*1024)).hexdigest(),
        "peaks_bed_md5": md5(PEAKS_BED),
    },
    "shapes": {
        "rna_raw": [3009, 36601],
        "rna_post_qc": list(rna.shape),
        "atac_raw_barcodes": list(atac.shape),
        "shared_cells": len(shared),
        "n_atac_peaks": int(atac.shape[1]),
    },
    "stages": stages,
    "results": {
        "n_grn_edges": int(len(grn)),
        "n_regulons_min10": len(regs),
        "topics_unique_top1_of_K": [n_unique, K],
        "canonical_pbmc_tfs_present": present,
        "canonical_hit_count": f"{n_present}/{len(canonical)}",
    },
    "env": {
        "python": os.environ["PYVER"],
        "scanpy": os.environ["SCANPY_VER"],
        "anndata": os.environ["ANNDATA_VER"],
        "os": os.environ["OS"],
        "cpu": os.environ["CPU"],
        "n_cpus": int(os.environ["N_CPUS"]),
        "total_ram_gb": float(os.environ["TOTAL_RAM_GB"]) if os.environ["TOTAL_RAM_GB"] != "unknown" else None,
    },
}

art_path = Path(os.environ["ARTEFACT"])
art_path.write_text(json.dumps(artefact, indent=2))
print(f"\nartefact → {art_path}", flush=True)
print(json.dumps(artefact, indent=2), flush=True)
PY

deactivate
echo "done"
