"""Read the explicit RNA exchange and score gene sets with RustScenic >= 0.5.0.

The exchange contains genes-by-cells expression.mtx and one ID per line in
genes.txt and cells.txt. It carries expression and axis IDs, not a full Seurat
object. Values are never normalized, identifier-mapped or selected from layers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread

from rustscenic import aucell


def _read_ids(path: Path) -> list[str]:
    ids = path.read_text(encoding="utf-8").splitlines()
    if not ids or any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{path.name} must contain nonempty, unique IDs")
    return ids


def read_export(directory: str | Path) -> ad.AnnData:
    """Load named, sparse cells-by-genes expression without changing values."""
    directory = Path(directory)
    genes = _read_ids(directory / "genes.txt")
    cells = _read_ids(directory / "cells.txt")
    matrix = sparse.csr_matrix(mmread(directory / "expression.mtx"))
    if matrix.shape != (len(genes), len(cells)):
        raise ValueError(
            f"expression.mtx must be genes by cells: got {matrix.shape}, "
            f"expected {(len(genes), len(cells))} from the ID files"
        )
    if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
        raise ValueError("Expected finite, nonnegative log-normalized RNA data")
    return ad.AnnData(
        X=matrix.T.tocsr(),
        obs=pd.DataFrame(index=cells),
        var=pd.DataFrame(index=genes),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="R export directory")
    parser.add_argument("regulons", type=Path, help="JSON object mapping names to gene lists")
    parser.add_argument("output", type=Path, help="Activity CSV with barcode index")
    parser.add_argument("--h5ad", type=Path, help="Also save the prepared expression AnnData")
    args = parser.parse_args(argv)
    expression = read_export(args.directory)
    regulons = json.loads(args.regulons.read_text(encoding="utf-8"))
    activity = aucell.score(expression, regulons)
    activity.to_csv(args.output, float_format="%.17g")
    if args.h5ad is not None:
        expression.write_h5ad(args.h5ad)
    print(f"Scored {activity.shape[0]} cells and {activity.shape[1]} regulons")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
