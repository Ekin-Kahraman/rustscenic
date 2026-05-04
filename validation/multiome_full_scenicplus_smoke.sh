#!/usr/bin/env bash
# Real-data full SCENIC+ end-to-end smoke on rustscenic v0.3.6.
# Closes v0.4 publication-threshold item:
#   "Real-data full SCENIC+ end-to-end including motif-rankings download,
#    cistarget enrichment, enhancer linking, eRegulon assembly, in fresh venv"
#
# Differs from multiome_fresh_env_smoke.sh: that one does RNA QC + GRN +
# AUCell + ATAC topics (partial). This one runs pipeline.run with motif
# rankings + gene coords so cistarget + enhancer + eRegulon all execute on
# real data.
set -euo pipefail

DATA_DIR="$HOME/projects/bio/rustscenic/validation/real_multiome_v036"
ARTEFACT="$HOME/projects/bio/rustscenic/validation/multiome_full_scenicplus_v0.3.7.json"
WORK=$(mktemp -d)
echo "work dir: $WORK"

python3 -m venv "$WORK/venv"
source "$WORK/venv/bin/activate"
pip install --quiet --upgrade pip

echo "=== installing rustscenic[validation] @ v0.3.6 ==="
pip install --quiet --upgrade "rustscenic[validation] @ git+https://github.com/Ekin-Kahraman/rustscenic@v0.3.7"

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
python - <<'PY'
import json, os, resource, time, hashlib, traceback
from pathlib import Path
import anndata as ad
import scanpy as sc
import pandas as pd
import rustscenic, rustscenic.data, rustscenic.pipeline

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

def stage(name, stages):
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

stages = {}
print(f"rustscenic {os.environ['RUSTSCENIC_VER']} on Python {os.environ['PYVER']}", flush=True)

# Pre-load + RNA QC (pipeline.run accepts an AnnData directly)
print("[setup] load + QC RNA", flush=True)
with stage("load_qc", stages):
    rna = sc.read_10x_h5(RNA_H5); rna.var_names_make_unique()
    sc.pp.filter_cells(rna, min_genes=200)
    sc.pp.filter_genes(rna, min_cells=3)
    rna.var["mt"] = rna.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(rna, qc_vars=["mt"], inplace=True)
    rna = rna[rna.obs["pct_counts_mt"] < 20].copy()
    sc.pp.normalize_total(rna, target_sum=1e4); sc.pp.log1p(rna)
    print(f"  RNA shape: {rna.shape}", flush=True)

# Download motif rankings (gene-based; auto-cached at ~/.cache/rustscenic/cistarget/)
print("[setup] download motif rankings (cached)", flush=True)
with stage("download_motifs", stages):
    motif_rankings = rustscenic.data.download_motif_rankings(species="human", verbose=False)
    print(f"  motif_rankings shape: {motif_rankings.shape}", flush=True)

# Download gene coords (GENCODE TSS, cached parquet)
print("[setup] download gene coords (cached)", flush=True)
with stage("download_gene_coords", stages):
    gene_coords = rustscenic.data.download_gene_coords(species="hs", verbose=False)
    print(f"  gene_coords shape: {gene_coords.shape}", flush=True)

# Now the full pipeline.run with all inputs present
print("[full pipeline.run] grn + aucell + topics + cistarget + enhancer + eregulon", flush=True)
with stage("pipeline_run_full", stages):
    result = rustscenic.pipeline.run(
        rna=rna,
        output_dir=OUT,
        fragments=ATAC_FRAG,
        peaks=PEAKS_BED,
        motif_rankings=motif_rankings,
        gene_coords=gene_coords,
        grn_n_estimators=300,         # speed for smoke
        topics_n_topics=10,           # smaller K for PBMC small data
        topics_n_passes=3,
        seed=777,
        verbose=True,
    )

# Inspect outputs
def file_exists_and_size(path):
    if path is None: return None
    p = Path(path)
    return {"path": str(p.name), "exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else None}

print("\n=== pipeline outputs ===", flush=True)
for attr in ["grn_path", "regulons_path", "aucell_path", "topics_dir", "cistarget_path",
             "enhancer_links_path", "eregulons_path", "integrated_adata_path"]:
    val = getattr(result, attr, None)
    print(f"  {attr}: {file_exists_and_size(val)}", flush=True)

# Pull headline numbers
n_eregulons_field = getattr(result, "n_eregulons", None)

cistarget_rows = None
if result.cistarget_path and Path(result.cistarget_path).exists():
    cistarget_rows = len(pd.read_parquet(result.cistarget_path))

enhancer_rows = None
if result.enhancer_links_path and Path(result.enhancer_links_path).exists():
    enhancer_rows = len(pd.read_parquet(result.enhancer_links_path))

eregulon_rows = None
if result.eregulons_path and Path(result.eregulons_path).exists():
    eregulon_rows = len(pd.read_parquet(result.eregulons_path))

artefact = {
    "release": "v0.3.7",
    "rustscenic_version": os.environ["RUSTSCENIC_VER"],
    "rustscenic_sha": os.environ["RUSTSCENIC_SHA"],
    "command": "fresh-venv: pip install rustscenic[validation] @ git+...@v0.3.7 + pipeline.run full SCENIC+ E2E",
    "dataset": {
        "name": "10x pbmc_unsorted_3k",
        "source": "cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_unsorted_3k",
        "rna_h5_md5": md5(RNA_H5),
        "atac_fragments_md5_first_8mb": hashlib.md5(open(ATAC_FRAG,"rb").read(8*1024*1024)).hexdigest(),
        "peaks_bed_md5": md5(PEAKS_BED),
    },
    "stages": stages,
    "outputs": {
        "n_grn_edges": pd.read_parquet(result.grn_path).shape[0] if result.grn_path else None,
        "n_regulons": int(result.n_regulons) if hasattr(result, "n_regulons") else None,
        "cistarget_rows": cistarget_rows,
        "enhancer_links_rows": enhancer_rows,
        "n_eregulons_dataframe_rows": eregulon_rows,
        "n_eregulons": n_eregulons_field,
    },
    "stage_smokes": {
        "grn": result.grn_path is not None,
        "aucell": result.aucell_path is not None,
        "topics": result.topics_dir is not None,
        "cistarget": result.cistarget_path is not None,
        "enhancer_links": result.enhancer_links_path is not None,
        "eregulons": result.eregulons_path is not None,
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

art_path = Path(os.environ["ARTEFACT"])
art_path.write_text(json.dumps(artefact, indent=2))
print(f"\nartefact → {art_path}", flush=True)
print(json.dumps(artefact, indent=2), flush=True)
PY

deactivate
echo "done"
