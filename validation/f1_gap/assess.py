"""One fixed representation diagnostic on archived networks and real matrices."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from rustscenic.aucell import score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "validation/biological_fidelity"))
from polarity import signed_gene_sets, signed_contrast


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_matrix(path):
    with h5py.File(path) as h:
        genes, cells = h["features"].asstr()[:].tolist(), h["cells"].asstr()[:].tolist()
        x = sparse.csr_matrix((h["data"][:], h["indices"][:], h["indptr"][:]), shape=(len(cells), len(genes)))
    return x, genes, cells


def directional_scores(x, genes, cells, edges, method, common):
    sets = signed_gene_sets(edges, method)
    names = [tf + suffix for tf in common for suffix in ("_activator", "_repressor")]
    available = set(genes)
    selected = {name: sets[name] for name in names if name in sets}
    coverage = {name: len(set(targets) & available) / len(targets) for name, targets in selected.items()}
    eligible = {name: targets for name, targets in selected.items() if coverage[name] >= .8}
    frame = score((x, genes), eligible, seed=20261106, top_frac=.05)
    frame.index = cells
    frame = frame.reindex(columns=names, fill_value=0.).astype(np.float32)
    return frame, {"available_directions": len(selected), "coverage": coverage,
                   "nonconstant_directions": int(np.count_nonzero(np.ptp(frame.to_numpy(), axis=0))),
                   "padded_columns": len(names)}


def run(args):
    utility_path = REPO / "docs/paper/evidence/scoring/package/utility.py"
    utility = load_module(utility_path, "frozen_utility")
    args.output.mkdir(parents=True, exist_ok=True)
    package = args.artifacts / "RustScenic-Paper-Validation-20261106/rustscenic-paper-validation"
    returned = args.artifacts / "RustScenic-Biology-Results-Audit-20261106/returned/results"
    confirmation = args.artifacts / "RustScenic-Confirmation-20261107/rustscenic-confirmation"
    inputs = confirmation / "external" if args.cohort == "external" else package / "holdout-inputs"
    x, genes, cells = read_matrix(inputs / "rna.h5")
    meta = pd.read_csv(inputs / "cell-metadata.tsv", sep="\t", dtype={"donor": str}).set_index("cell").loc[cells]
    input_info = json.loads((package / "holdout-inputs/input.json").read_text())
    for n in args.sizes:
        for seed in args.seeds:
            path = args.output / f"{args.cohort}-{n}-{seed}.json"
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite {path}")
            pair = returned / f"bmmc-train{n}-seed{seed}"
            old = json.loads((pair / "assessment.json").read_text())
            common = old["results"]["rust_common"]["signature_names"]
            if args.cohort == "external":
                external_common = json.loads((confirmation / "networks" / pair.name / "common-tfs.json").read_text())
                assert common == external_common or set(common) == set(external_common)
                common = external_common
            result = {"cohort": args.cohort, "training_cells": n, "seed": seed,
                      "common_tfs": common, "results": {}, "representation": {},
                      "source_sha256": {str(inputs / "rna.h5"): digest(inputs / "rna.h5"),
                          str(utility_path): digest(utility_path),
                          "protocol": digest(Path(__file__).with_name("PROTOCOL.md"))}}
            for method in ("rustscenic", "reference"):
                edge_path = pair / method / "eregulon-edges.parquet"
                result["source_sha256"][str(edge_path)] = digest(edge_path)
                edges = pd.read_parquet(edge_path)
                frame, description = directional_scores(x, genes, cells, edges, method, common)
                result["representation"][method] = description
                frames = {"contrast": signed_contrast(frame)[common], "directional": frame}
                for mode, values in frames.items():
                    kwargs = {}
                    if args.cohort == "development":
                        values = values.iloc[:n]
                        assert meta.loc[values.index, "split"].eq("train").all()
                    elif args.cohort == "bmmc_test":
                        values = values.iloc[list(range(n)) + list(range(20000, 30000))]
                        kwargs = {"test_donors": input_info["test_donors"], "train_donors": input_info["train_donors"]}
                    start = time.perf_counter()
                    name = f"{method}_{mode}"
                    assessed = utility.evaluate(values, meta, name, **kwargs)
                    assessed["seconds"] = time.perf_counter() - start
                    assessed["mean_macro_f1"] = float(np.mean([row["macro_f1"] for row in assessed["folds"] if row["status"] == "complete"]))
                    result["results"][name] = assessed
                    print(args.cohort, n, seed, name, round(assessed["mean_macro_f1"], 6), flush=True)
                frame.to_parquet(args.output / f"{args.cohort}-{n}-{seed}-{method}-directional.parquet")
            path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort", choices=["development", "bmmc_test", "external"], required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=[5000, 20000])
    parser.add_argument("--seeds", nargs="+", type=int, default=[777, 778, 779])
    run(parser.parse_args())
