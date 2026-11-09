"""Replay frozen real RNA networks and exercise the installed CLI from input to output.

Usage: python check_rna.py --package PATH --returned PATH --output NEW_DIRECTORY
The package/returned paths are the existing paper-validation input and result
archives. No network downloads or inferred-network changes in replay checks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from rustscenic import validated, aucell


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--returned", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {"installed_module": str(Path(validated.__file__).resolve()),
               "scope": "Real-data orchestration compatibility, not new biological validation", "replays": [], "inputs": {}}
    for dataset in ("hESC", "mESC", "mDC"):
        folder = args.package / "rna-inputs" / dataset
        info = json.loads((folder / "input.json").read_text())
        matrix = sparse.load_npz(folder / "matrix.npz")
        rna = ad.AnnData(matrix, obs=pd.DataFrame(index=[f"{dataset}:{i}" for i in range(matrix.shape[0])]),
                         var=pd.DataFrame(index=info["genes"]))
        motifs = args.package / "rna-motifs" / info["species"]
        ranks = validated.load_rankings(motifs / "mouse-gene.feather", rna.var_names.tolist())
        annotation = validated.load_annotations(motifs / "mouse-gene-annotations.tbl")
        for path in (folder / "matrix.npz", folder / "input.json", motifs / "mouse-gene.feather",
                     motifs / "mouse-gene.json", motifs / "mouse-gene-annotations.tbl"):
            receipt["inputs"][str(path)] = digest(path)
        for seed in (777, 778, 779):
            prior = args.returned / f"rna-{dataset}-seed{seed}" / "rustscenic"
            network = pd.read_parquet(prior / "grn.parquet")
            expected = json.loads((prior / "regulons.json").read_text())
            receipt["inputs"][str(prior / "grn.parquet")] = digest(prior / "grn.parquet")
            receipt["inputs"][str(prior / "regulons.json")] = digest(prior / "regulons.json")
            start = time.perf_counter()
            with patch.object(validated.grn, "infer", return_value=network.copy()):
                result = validated._rna(rna, info["tfs"], ranks, annotation, seed, validated.SCORING_SEED)
            assert result["regulons"] == expected, (dataset, seed, "signature mismatch")
            direct = aucell.score(rna, expected, seed=validated.SCORING_SEED)
            pd.testing.assert_frame_equal(result["gene_activity"].sort_index(axis=1), direct.sort_index(axis=1), check_exact=True)
            assert np.isfinite(direct.to_numpy()).all()
            receipt["replays"].append(dict(dataset=dataset, seed=seed, cells=rna.n_obs,
                genes=rna.n_vars, regulons=len(expected), signatures_exact=True,
                seeded_scoring_exact=True, elapsed_seconds=time.perf_counter() - start))
            print(dataset, seed, len(expected), "signatures and seeded activity match", flush=True)
        if dataset == "hESC":
            rna.write_h5ad(args.output / "hESC.h5ad")
            (args.output / "tfs.txt").write_text("\n".join(info["tfs"]) + "\n")
            command = [sys.executable, "-m", "rustscenic.cli", "validated", "--rna", str(args.output / "hESC.h5ad"),
                       "--tfs", str(args.output / "tfs.txt"), "--motif-rankings", str(motifs / "mouse-gene.feather"),
                       "--motif-annotations", str(motifs / "mouse-gene-annotations.tbl"),
                       "--output", str(args.output / "hESC-cli"), "--threads", "4"]
            start = time.perf_counter()
            process = subprocess.run(command, capture_output=True, text=True, env=os.environ.copy())
            (args.output / "hESC-cli.log").write_text(process.stdout + process.stderr)
            assert process.returncode == 0, process.stderr
            manifest = json.loads((args.output / "hESC-cli/manifest.json").read_text())
            assert manifest["status"] == "complete" and manifest["n_regulons"] > 0
            scores = pd.read_parquet(args.output / "hESC-cli/gene_activity.parquet")
            assert scores.index.equals(rna.obs_names) and np.isfinite(scores.to_numpy()).all()
            receipt["fresh_cli"] = dict(command=command, elapsed_seconds=time.perf_counter() - start,
                input_cells=rna.n_obs, input_genes=rna.n_vars, n_regulons=manifest["n_regulons"],
                scores_nonzero=bool(np.any(scores.to_numpy() > 0)), inference_reused=False)
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
