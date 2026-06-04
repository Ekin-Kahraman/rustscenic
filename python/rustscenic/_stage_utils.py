"""Private helpers shared by the Python stage wrappers."""
from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
import pandas as pd

from rustscenic._rustscenic import (
    stage_prepare_regulon_indices as _stage_prepare_regulon_indices,
    stage_prepare_regulon_indices_with_coverage as _stage_prepare_regulon_indices_with_coverage,
)


def as_float32_contiguous(values) -> np.ndarray:
    """Return a C-contiguous float32 ndarray, copying only when needed."""
    arr = np.asarray(values)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return np.ascontiguousarray(arr)


def as_float32_array(values) -> np.ndarray:
    """Return a float32 ndarray without changing memory layout."""
    arr = np.asarray(values)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr


def string_list(values: Iterable) -> list[str]:
    """Return Python strings with minimal pandas boundary overhead."""
    if isinstance(values, (pd.Series, pd.Index)):
        arr = values.to_numpy(copy=False)
    elif isinstance(values, np.ndarray):
        arr = values
    else:
        out = list(values)
        if all(isinstance(value, str) for value in out):
            return out
        return [str(value) for value in out]

    out = arr.tolist()
    if not isinstance(out, list):
        return [str(out)]
    if all(isinstance(value, str) for value in out):
        return out
    return [str(value) for value in out]


def coerce_regulon(reg) -> tuple[str, list[str]]:
    """Normalise all accepted regulon-like objects to ``(name, genes)``."""
    if isinstance(reg, tuple) and len(reg) == 2:
        name, genes = reg
        return str(name), list(genes)
    if isinstance(reg, dict) and "name" in reg and "genes" in reg:
        return str(reg["name"]), list(reg["genes"])

    name = getattr(reg, "name", None) or getattr(reg, "transcription_factor", None)
    if hasattr(reg, "gene2weight"):
        genes = list(reg.gene2weight.keys())
    elif hasattr(reg, "genes"):
        genes = list(reg.genes)
    else:
        raise TypeError(f"cannot extract regulon genes from {type(reg).__name__}")
    if name is None:
        raise TypeError(f"regulon {reg!r} has no .name")
    return str(name), genes


def iter_regulon_pairs(regulons: Iterable) -> Iterator[tuple[str, list[str]]]:
    """Yield ``(name, genes)`` pairs from dicts, tuples, or Regulon objects."""
    items = regulons.items() if isinstance(regulons, dict) else regulons
    for reg in items:
        yield coerce_regulon(reg)


def prepare_regulon_indices(
    gene_names: Iterable[str],
    regulons: Iterable,
) -> tuple[list[str], list[list[int]], list[tuple[str, list[str]]], int]:
    """Map regulon genes onto a feature axis once, preserving coverage inputs."""
    gene_names_list = string_list(gene_names)
    reg_pairs = [
        (name, string_list(genes_list))
        for name, genes_list in iter_regulon_pairs(regulons)
    ]
    reg_names, reg_gene_indices, dropped_empty = _stage_prepare_regulon_indices(
        gene_names_list,
        [name for name, _ in reg_pairs],
        [genes for _, genes in reg_pairs],
    )
    return reg_names, reg_gene_indices, reg_pairs, dropped_empty


def prepare_regulon_indices_with_coverage(
    gene_names: Iterable[str],
    regulons: Iterable,
) -> tuple[
    list[str],
    list[list[int]],
    list[tuple[str, list[str]]],
    dict[str, tuple[int, int]],
    int,
]:
    """Map regulon genes and coverage in one Rust call."""
    gene_names_list = string_list(gene_names)
    reg_pairs = [
        (name, string_list(genes_list))
        for name, genes_list in iter_regulon_pairs(regulons)
    ]
    names = [name for name, _ in reg_pairs]
    reg_names, reg_gene_indices, matched, total, dropped_empty = (
        _stage_prepare_regulon_indices_with_coverage(
            gene_names_list,
            names,
            [genes for _, genes in reg_pairs],
        )
    )
    coverage = {
        name: (int(n_matched), int(n_total))
        for name, n_matched, n_total in zip(names, matched, total, strict=True)
    }
    return reg_names, reg_gene_indices, reg_pairs, coverage, dropped_empty


_REGULON_NAME_SUFFIXES = ("_regulon", "_extended", "_activator", "_repressor")
_REGULON_NAME_PARENS = ("(+)", "(-)")


def tf_from_regulon_name(name: str) -> str:
    """Reduce common SCENIC regulon-name variants to the bare TF symbol."""
    tf = str(name).strip()
    while True:
        prev = tf
        for suffix in _REGULON_NAME_SUFFIXES:
            if tf.endswith(suffix):
                tf = tf[: -len(suffix)].strip()
        for paren in _REGULON_NAME_PARENS:
            if tf.endswith(paren):
                tf = tf[:-3].strip()
        if tf == prev:
            return tf


def require_columns(df: pd.DataFrame, required: set[str], *, name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {sorted(missing)}. "
            f"Got columns: {list(df.columns)}"
        )
