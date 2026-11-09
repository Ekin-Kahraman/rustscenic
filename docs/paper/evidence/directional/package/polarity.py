"""Preserve TF-target direction when assessing cached multiome outputs.

The benchmark's TF-union export discards direction. Keep that historical
endpoint intact and evaluate these explicitly labelled signed signatures too.
Correlation direction is not proof of activation or repression.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def signed_gene_sets(edges: pd.DataFrame, method: str) -> dict[str, list[str]]:
    """Convert archived tf/target/context edges without merging opposite signs."""
    required = {"tf", "target", "context"}
    if not required.issubset(edges.columns):
        raise ValueError(f"Missing edge columns: {sorted(required - set(edges.columns))}")
    if edges[list(required)].isna().any().any():
        raise ValueError("Missing TF, target or context")
    context = edges["context"].astype(str)
    if method == "rustscenic":
        # Exact benchmark context format; do not infer sign from the gene name.
        positive = context.eq(edges.tf.astype(str) + "_activator; positive region-gene links")
        negative = context.eq(edges.tf.astype(str) + "_repressor; positive region-gene links")
    elif method == "reference":
        positive = context.str.contains("'positive tf2g'", regex=False)
        negative = context.str.contains("'negative tf2g'", regex=False)
    else:
        raise ValueError(f"Unknown method: {method!r}")
    if not (positive ^ negative).all():
        raise ValueError("Every edge must have exactly one explicit TF-target sign")
    result = {}
    for mask, suffix in [(positive, "_activator"), (negative, "_repressor")]:
        for tf, group in edges.loc[mask].groupby("tf", sort=True):
            result[str(tf) + suffix] = sorted(set(group.target.astype(str)))
    return dict(sorted(result.items()))


def signed_contrast(activity: pd.DataFrame) -> pd.DataFrame:
    """Return AUC(positive targets) minus AUC(negative targets), one column/TF.

    Score the two target sets separately with the same AUCell implementation.
    A missing direction contributes zero. Both absent means no TF column.
    Results can be negative and must not be described as ordinary AUCell AUCs.
    """
    if not activity.columns.is_unique or not activity.index.is_unique:
        raise ValueError("Duplicate signature or cell identifiers")
    if not np.isfinite(activity.to_numpy()).all():
        raise ValueError("Nonfinite activity values")
    groups = {}
    for name in activity.columns:
        if isinstance(name, str) and name.endswith("_activator"):
            tf, sign = name[:-10], 1
        elif isinstance(name, str) and name.endswith("_repressor"):
            tf, sign = name[:-10], -1
        else:
            raise ValueError(f"Signature lacks explicit direction: {name!r}")
        if not tf:
            raise ValueError("Empty TF identifier")
        groups.setdefault(tf, []).append((name, sign))
    names = sorted(groups)
    # Allocate the final matrix once; avoid a second full matrix in pandas.
    output = np.zeros((len(activity), len(names)), dtype=np.float64, order="F")
    for column, tf in enumerate(names):
        values = output[:, column]
        columns = groups[tf]
        for name, sign in columns:
            values += sign * activity[name].to_numpy(dtype=np.float64)
    result = pd.DataFrame(output, index=activity.index, columns=names, copy=False)
    result.attrs["score"] = "positive-target AUC minus negative-target AUC"
    return result
