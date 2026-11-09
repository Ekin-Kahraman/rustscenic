#!/usr/bin/env python3
"""Replay the paired recipe against archived real results without refitting.

Usage: python check_paired.py --inputs INPUT_DIRECTORY --saved NATIVE_RESULTS \
    --rankings RESTORED_RANKINGS.feather --output AUDIT_DIRECTORY
The paper's CSR HDF5 matrices and restored authentic rank projection are inputs.
This is a cached-stage parity audit, not a new end-to-end benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
import pandas as pd
import pyarrow.feather as feather
from scipy import sparse
from rustscenic import _audited_multiome as workflow


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 ** 2), b""):
            result.update(block)
    return result.hexdigest()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "saved", "rankings", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--cells", type=int, default=5000)
    return parser


def load_case(args, *, use_threads=True):
    """Read the same fixed cell prefix and cached fitted-stage axes in both arms."""
    def matrix(name):
        with h5py.File(args.inputs / f"{name}.h5") as handle:
            if not 0 < args.cells <= handle.attrs["shape"][0]:
                raise ValueError("cells must fit the archived input")
            pointers = handle["indptr"][:args.cells + 1]
            count = pointers[-1]
            values = sparse.csr_matrix((handle["data"][:count], handle["indices"][:count], pointers),
                                       shape=(args.cells, int(handle.attrs["shape"][1])))
            return workflow._ann(values, handle["cells"][:args.cells].astype(str), handle["features"][:].astype(str))
    ranks = feather.read_feather(args.rankings, use_threads=use_threads).set_index("motifs")
    sidecar = json.loads(args.rankings.with_suffix(".json").read_text())
    ranks.attrs["rank_universe_size"] = int(sidecar["original_rank_universe"])
    cached = {name: pd.read_parquet(args.saved / f'{name.replace("_", "-")}.parquet')
              for name in ("cell_topic", "topic_peak", "grn")}
    frames = {str(path): pd.read_csv(path, sep="\t") for path in (
        args.inputs / "gene-coordinates.tsv", args.inputs / "peaks.tsv",
        args.inputs / "motifs/screen-overlap-mapping.tsv", args.inputs / "motifs/human-gene-annotations.tbl")}
    return dict(rna=matrix("rna"), atac=matrix("atac"), ranks=ranks, cached=cached,
                frames=frames, tfs=(args.inputs / "tfs.txt").read_text().splitlines())


def updated(case, args):
    frames, inputs = case["frames"], args.inputs
    return workflow.run(case["rna"], case["atac"], case["tfs"],
        gene_coords=frames[str(inputs / "gene-coordinates.tsv")],
        region_rankings=case["ranks"], region_mapping=frames[str(inputs / "motifs/screen-overlap-mapping.tsv")],
        motif_annotations=frames[str(inputs / "motifs/human-gene-annotations.tbl")],
        output_dir=args.output, threads=4, cached_stages=case["cached"])


def main():
    args = arguments().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    case = load_case(args)
    started = time.perf_counter()
    result = updated(case, args)
    checks = {"scope": "Cached real topics and GRN replay; all downstream stages recomputed with the corrected scorer",
              "elapsed_seconds": time.perf_counter() - started, "metadata": result["metadata"]}
    for key, filename in (("regulons", "regulons.json"), ("region_regulons", "region-regulons.json")):
        old = json.loads((args.saved / filename).read_text())
        new = result[key]
        checks[key + "_exact"] = ({k: sorted(v) for k, v in old.items()} == {k: sorted(v) for k, v in new.items()})
        checks[key + "_count"] = len(new)
    for key, filename, columns in (
        ("eregulon_edges", "eregulon-edges.parquet", ["tf", "region", "target", "context"]),
        ("motif_evidence", "full-motif-evidence.parquet", ["TF", "peak_id", "motif", "auc", "nes"]),
        ("region_motif_scores", "region-motif-scores.parquet", ["region_set", "motif", "auc", "nes"]),
        ("region_to_gene", "region-to-gene.parquet", ["region", "target", "rho"]),
    ):
        old = pd.read_parquet(args.saved / filename)[columns].sort_values(columns).reset_index(drop=True)
        new = result[key][columns].sort_values(columns).reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(new, old, check_dtype=False, check_exact=True)
            checks[key + "_exact"] = True
        except AssertionError as error:
            checks[key + "_exact"] = False
            checks[key + "_difference"] = str(error)[:2000]
        checks[key + "_shape"] = list(new.shape)
    for key in ("gene_activity", "region_activity"):
        scores = result[key]
        checks[key + "_shape"] = list(scores.shape)
        checks[key + "_finite"] = bool(np.isfinite(scores).all().all())
        scores.to_parquet(args.output / f"{key}.parquet")
    checks["module_path"] = workflow.__file__
    checks["module_sha256"] = digest(workflow.__file__)
    paths = [args.inputs / "rna.h5", args.inputs / "atac.h5", args.rankings,
             *(args.saved / f"{name}.parquet" for name in ("cell-topic", "topic-peak", "grn")),
             *map(Path, case["frames"])]
    checks["input_sha256"] = {str(path): digest(path) for path in paths}
    (args.output / "replay-audit.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))
    if not all(value for key, value in checks.items() if key.endswith(("_exact", "_finite"))):
        raise SystemExit("Paired replay did not match the archived evidence")


if __name__ == "__main__":
    main()
