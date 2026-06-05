"""End-to-end RustScenic vs SCENIC+ core benchmark.

This intentionally benchmarks the matrix-level regulatory path shared by both
tools:

  RNA + ATAC matrices + cistromes
    -> TF-to-gene
    -> region-to-gene
    -> eRegulons
    -> gene/region AUCell

It does not include raw fragment parsing, topic modelling, or pycistarget
database generation. Those need separate E2E tiers once this core pair is
stable.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def peak_rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024**3)
    return rss / (1024**2)


_PSUTIL_PROCESS: Any | None = None
_PSUTIL_CHECKED = False


def current_rss_gb() -> float:
    global _PSUTIL_CHECKED, _PSUTIL_PROCESS

    if not _PSUTIL_CHECKED:
        _PSUTIL_CHECKED = True
        try:
            import psutil

            _PSUTIL_PROCESS = psutil.Process()
        except Exception:
            _PSUTIL_PROCESS = None
    if _PSUTIL_PROCESS is not None:
        return float(_PSUTIL_PROCESS.memory_info().rss) / (1024**3)
    if sys.platform.startswith("linux"):
        statm = Path("/proc/self/statm")
        if statm.exists():
            pages = int(statm.read_text().split()[1])
            return float(pages * os.sysconf("SC_PAGE_SIZE")) / (1024**3)
    if sys.platform == "darwin":
        try:
            rss_kb = int(
                subprocess.check_output(
                    ["ps", "-o", "rss=", "-p", str(os.getpid())],
                    text=True,
                ).strip()
            )
            return float(rss_kb) / (1024**2)
        except Exception:
            pass
    return peak_rss_gb()


class PeakSampler:
    def __init__(self, interval_s: float) -> None:
        self.interval_s = max(0.05, float(interval_s))
        self.start_rss_gb = 0.0
        self.peak_rss_gb = 0.0
        self.end_rss_gb = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "PeakSampler":
        self.start_rss_gb = current_rss_gb()
        self.peak_rss_gb = self.start_rss_gb
        self.end_rss_gb = self.start_rss_gb
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 2)
        self.end_rss_gb = current_rss_gb()
        self.peak_rss_gb = max(self.peak_rss_gb, self.end_rss_gb)

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.peak_rss_gb = max(self.peak_rss_gb, current_rss_gb())

    def record(self) -> dict[str, float]:
        return {
            "start_rss_gb": round(float(self.start_rss_gb), 6),
            "peak_rss_gb": round(float(self.peak_rss_gb), 6),
            "end_rss_gb": round(float(self.end_rss_gb), 6),
        }


def synthetic_multiome(
    *,
    n_cells: int,
    n_genes: int,
    n_peaks: int,
    n_programmes: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n_programmes = max(1, min(n_programmes, n_genes, n_peaks))
    cells = [f"cell_{i:04d}" for i in range(n_cells)]
    genes = [f"G{i:04d}" for i in range(n_genes)]
    peaks = [f"peak_{i:04d}" for i in range(n_peaks)]

    clusters = np.arange(n_cells) % n_programmes
    activity = np.zeros((n_cells, n_programmes), dtype=np.float32)
    for p in range(n_programmes):
        activity[:, p] = (clusters == p).astype(np.float32)
        activity[:, p] += 0.15 * rng.normal(size=n_cells).astype(np.float32)
    np.clip(activity, 0.0, None, out=activity)

    expr = rng.gamma(shape=1.2, scale=0.2, size=(n_cells, n_genes)).astype(np.float32)
    acc = rng.gamma(shape=1.2, scale=0.2, size=(n_cells, n_peaks)).astype(np.float32)

    programme_genes: list[list[str]] = []
    programme_peaks: list[list[str]] = []
    genes_per_programme = max(3, n_genes // (n_programmes + 1))
    peaks_per_programme = max(3, n_peaks // n_programmes)
    tf_names: list[str] = []

    for p in range(n_programmes):
        g_start = p * genes_per_programme
        g_stop = min(n_genes, g_start + genes_per_programme)
        p_start = p * peaks_per_programme
        p_stop = min(n_peaks, p_start + peaks_per_programme)
        if g_start >= g_stop or p_start >= p_stop:
            continue

        for local_idx, peak_idx in enumerate(range(p_start, p_stop)):
            start = 10_000 + p * 1_500_000 + local_idx * 2_000
            peaks[peak_idx] = f"chr1:{start}-{start + 500}"
        tf_names.append(genes[g_start])
        programme_genes.append(genes[g_start:g_stop])
        programme_peaks.append(peaks[p_start:p_stop])
        expr[:, g_start:g_stop] += activity[:, [p]] * rng.uniform(
            1.0, 1.8, size=(1, g_stop - g_start)
        ).astype(np.float32)
        acc[:, p_start:p_stop] += activity[:, [p]] * rng.uniform(
            1.0, 1.8, size=(1, p_stop - p_start)
        ).astype(np.float32)

    for peak_idx, peak in enumerate(peaks):
        if peak.startswith("peak_"):
            start = 20_000_000 + peak_idx * 2_000
            peaks[peak_idx] = f"chr1:{start}-{start + 500}"

    np.clip(expr, 0.0, None, out=expr)
    np.clip(acc, 0.0, None, out=acc)
    expr = np.log1p(expr)
    acc = np.log1p(acc)

    rna = ad.AnnData(
        X=expr,
        obs=pd.DataFrame({"cluster": clusters}, index=cells),
        var=pd.DataFrame(index=genes),
    )
    atac = ad.AnnData(
        X=acc,
        obs=pd.DataFrame(index=cells),
        var=pd.DataFrame(index=peaks),
    )
    gene_coords = _gene_coords(genes, programme_genes, programme_peaks)
    search_space = _search_space(programme_genes, programme_peaks)
    cistrome = _cistrome(peaks, tf_names, programme_peaks)

    return {
        "rna": rna,
        "atac": atac,
        "expr_df": pd.DataFrame(expr, index=cells, columns=genes),
        "acc_df": pd.DataFrame(acc, index=cells, columns=peaks),
        "gene_coords": gene_coords,
        "search_space": search_space,
        "cistrome": cistrome,
        "tf_names": tf_names,
        "programme_peaks": programme_peaks,
        "settings": {
            "n_cells": n_cells,
            "n_genes": n_genes,
            "n_peaks": n_peaks,
            "n_programmes": n_programmes,
            "n_tfs": len(tf_names),
        },
    }


def real_10x_multiome(
    *,
    input_10x_h5: Path,
    dataset_name: str,
    species: str,
    n_cells: int,
    n_genes: int,
    n_peaks: int,
    n_tfs: int,
    max_distance: int,
    seed: int,
) -> dict[str, Any]:
    import scanpy as sc

    all_tfs = _load_tfs(species)
    adata = sc.read_10x_h5(input_10x_h5, gex_only=False)
    if "interval" not in adata.var.columns:
        adata.var["interval"] = _read_10x_feature_intervals(input_10x_h5)
    adata.var_names_make_unique()

    feature_types = adata.var["feature_types"].astype(str)
    gene_mask = (feature_types == "Gene Expression").to_numpy()
    peak_mask = (feature_types == "Peaks").to_numpy()

    gene_names_all = np.asarray(adata.var_names[gene_mask], dtype=object)
    peak_names_all = np.asarray(adata.var_names[peak_mask], dtype=object)
    gene_intervals = np.asarray(adata.var.loc[gene_mask, "interval"], dtype=object)
    peak_intervals = np.asarray(adata.var.loc[peak_mask, "interval"], dtype=object)

    X_gene_all = adata[:, gene_mask].X.tocsr()
    X_peak_all = adata[:, peak_mask].X.tocsr()
    cell_ok = np.asarray(X_gene_all.sum(axis=1)).ravel() > 0
    cell_ok &= np.asarray(X_peak_all.sum(axis=1)).ravel() > 0
    cells_available = np.flatnonzero(cell_ok)
    if len(cells_available) < n_cells:
        raise ValueError(
            f"requested {n_cells} cells but only {len(cells_available)} have RNA and ATAC counts"
        )
    rng = np.random.default_rng(seed)
    selected_cells = np.sort(rng.choice(cells_available, size=n_cells, replace=False))

    X_gene_sample = X_gene_all[selected_cells]
    X_peak_sample = X_peak_all[selected_cells]
    gene_score = np.asarray(X_gene_sample.sum(axis=0)).ravel()
    peak_score = np.asarray(X_peak_sample.sum(axis=0)).ravel()

    gene_coords_all = _coords_from_intervals(gene_names_all, gene_intervals, kind="gene")
    peak_coords_all = _coords_from_intervals(peak_names_all, peak_intervals, kind="peak")

    tf_candidates = [
        g for g in gene_names_all.tolist()
        if g in all_tfs and g in gene_coords_all.index and gene_score[np.where(gene_names_all == g)[0][0]] > 0
    ]
    tf_candidates.sort(key=lambda g: gene_score[np.where(gene_names_all == g)[0][0]], reverse=True)
    tf_names = tf_candidates[:n_tfs]
    if len(tf_names) < n_tfs:
        raise ValueError(f"only found {len(tf_names)} TFs with expression in the real matrix")

    tf_set = set(tf_names)
    gene_order = np.argsort(-gene_score)
    selected_genes: list[str] = list(tf_names)
    for idx in gene_order:
        gene = str(gene_names_all[idx])
        if gene in tf_set or gene not in gene_coords_all.index:
            continue
        selected_genes.append(gene)
        if len(selected_genes) >= n_genes:
            break
    if len(selected_genes) < n_genes:
        raise ValueError(f"could only select {len(selected_genes)} genes with coordinates")

    selected_gene_coords = gene_coords_all.loc[selected_genes].reset_index(drop=True)
    candidate_peak_idx = _candidate_peak_indices(
        selected_gene_coords,
        peak_coords_all,
        max_distance=max_distance,
    )
    if len(candidate_peak_idx) < n_peaks:
        raise ValueError(
            f"only {len(candidate_peak_idx)} peaks within {max_distance} bp of selected genes"
        )
    candidate_peak_idx.sort(key=lambda i: peak_score[i], reverse=True)
    selected_peak_idx = sorted(candidate_peak_idx[:n_peaks])
    selected_peaks = peak_names_all[selected_peak_idx].astype(str).tolist()
    selected_peak_coords = peak_coords_all.iloc[selected_peak_idx].reset_index(drop=True)

    gene_index = {g: i for i, g in enumerate(gene_names_all)}
    selected_gene_idx = [gene_index[g] for g in selected_genes]
    expr = _normalise_log1p(X_gene_sample[:, selected_gene_idx])
    acc = _normalise_log1p(X_peak_sample[:, selected_peak_idx])
    cells = np.asarray(adata.obs_names[selected_cells], dtype=str).tolist()

    rna = ad.AnnData(
        X=expr,
        obs=pd.DataFrame(index=cells),
        var=pd.DataFrame(index=selected_genes),
    )
    atac = ad.AnnData(
        X=acc,
        obs=pd.DataFrame(index=cells),
        var=pd.DataFrame(index=selected_peaks),
    )
    search_space = _distance_search_space(
        selected_gene_coords,
        selected_peak_coords,
        max_distance=max_distance,
    )
    if search_space.empty:
        raise ValueError("real-data search space is empty")
    cistrome = _real_cistrome(selected_peaks, tf_names, search_space)

    return {
        "rna": rna,
        "atac": atac,
        "expr_df": pd.DataFrame(expr, index=cells, columns=selected_genes),
        "acc_df": pd.DataFrame(acc, index=cells, columns=selected_peaks),
        "gene_coords": selected_gene_coords,
        "search_space": search_space,
        "cistrome": cistrome,
        "tf_names": tf_names,
        "programme_peaks": [],
        "settings": {
            "dataset": dataset_name,
            "input_10x_h5": str(input_10x_h5),
            "species": species,
            "n_cells": n_cells,
            "n_genes": len(selected_genes),
            "n_peaks": len(selected_peaks),
            "n_programmes": None,
            "n_tfs": len(tf_names),
            "search_space_pairs": int(len(search_space)),
        },
    }


def _load_tfs(species: str) -> set[str]:
    canonical = {
        "hs": "hs",
        "human": "hs",
        "homo_sapiens": "hs",
        "hg38": "hs",
        "mm": "mm",
        "mouse": "mm",
        "mus_musculus": "mm",
        "mm10": "mm",
    }.get(str(species).lower())
    if canonical is None:
        raise ValueError(
            f"unknown species {species!r}; use human/hg38/hs or mouse/mm10/mm"
        )
    filename = {"hs": "allTFs_hg38.txt", "mm": "allTFs_mm.txt"}[canonical]
    tf_path = Path(__file__).resolve().parents[2] / "python/rustscenic/data" / filename
    return {line.strip() for line in tf_path.read_text().splitlines() if line.strip()}


def _read_10x_feature_intervals(input_10x_h5: Path) -> list[str]:
    import h5py

    with h5py.File(input_10x_h5, "r") as handle:
        raw = handle["matrix/features/interval"][:]
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in raw
    ]


def _coords_from_intervals(
    names: np.ndarray,
    intervals: np.ndarray,
    *,
    kind: str,
) -> pd.DataFrame:
    rows = []
    for name, interval in zip(names, intervals):
        parsed = _parse_interval(str(interval))
        if parsed is None:
            continue
        chrom, start, end = parsed
        if kind == "gene":
            rows.append({"gene": str(name), "chrom": chrom, "tss": start})
        else:
            rows.append({"peak_id": str(name), "chrom": chrom, "start": start, "end": end})
    return pd.DataFrame(rows).set_index("gene" if kind == "gene" else "peak_id", drop=False)


def _parse_interval(interval: str) -> tuple[str, int, int] | None:
    if ":" not in interval or "-" not in interval:
        return None
    chrom, rest = interval.split(":", 1)
    start_s, end_s = rest.split("-", 1)
    try:
        return chrom, int(start_s), int(end_s)
    except ValueError:
        return None


def _candidate_peak_indices(
    gene_coords: pd.DataFrame,
    peak_coords: pd.DataFrame,
    *,
    max_distance: int,
) -> list[int]:
    selected: set[int] = set()
    peak_by_chrom = {
        chrom: sub for chrom, sub in peak_coords.reset_index(drop=True).groupby("chrom", sort=False)
    }
    for row in gene_coords.itertuples(index=False):
        sub = peak_by_chrom.get(row.chrom)
        if sub is None:
            continue
        peak_mid = ((sub["start"].to_numpy() + sub["end"].to_numpy()) // 2)
        passing = np.flatnonzero(np.abs(peak_mid - int(row.tss)) <= max_distance)
        selected.update(sub.index.to_numpy()[passing].tolist())
    return sorted(selected)


def _distance_search_space(
    gene_coords: pd.DataFrame,
    peak_coords: pd.DataFrame,
    *,
    max_distance: int,
) -> pd.DataFrame:
    rows = []
    peaks_by_chrom = {chrom: sub for chrom, sub in peak_coords.groupby("chrom", sort=False)}
    for gene in gene_coords.itertuples(index=False):
        sub = peaks_by_chrom.get(gene.chrom)
        if sub is None:
            continue
        mids = ((sub["start"].to_numpy() + sub["end"].to_numpy()) // 2)
        distances = mids - int(gene.tss)
        keep = np.abs(distances) <= max_distance
        for peak_id, dist in zip(sub.loc[keep, "peak_id"], distances[keep]):
            rows.append({"Name": peak_id, "Gene": gene.gene, "Distance": int(dist)})
    return pd.DataFrame(rows)


def _normalise_log1p(x: sp.spmatrix) -> np.ndarray:
    x = x.astype(np.float32).tocsr(copy=True)
    totals = np.asarray(x.sum(axis=1)).ravel().astype(np.float32)
    totals[totals == 0.0] = 1.0
    scale = 10_000.0 / totals
    x = sp.diags(scale).dot(x)
    x.data = np.log1p(x.data)
    return x.toarray().astype(np.float32, copy=False)


def _real_cistrome(
    peaks: list[str],
    tf_names: list[str],
    search_space: pd.DataFrame,
) -> pd.DataFrame:
    data = np.zeros((len(peaks), len(tf_names)), dtype=bool)
    peak_to_idx = {peak: i for i, peak in enumerate(peaks)}
    grouped = search_space.groupby("Gene")["Name"].apply(list).to_dict()
    all_search_peaks = sorted({p for values in grouped.values() for p in values})
    for tf_idx, tf in enumerate(tf_names):
        tf_peaks = grouped.get(tf, [])
        if not tf_peaks:
            tf_peaks = all_search_peaks[tf_idx::max(1, len(tf_names))]
        if not tf_peaks:
            tf_peaks = peaks[tf_idx::max(1, len(tf_names))]
        for peak in tf_peaks:
            if peak in peak_to_idx:
                data[peak_to_idx[peak], tf_idx] = True
    return pd.DataFrame(data, index=peaks, columns=tf_names)


def _gene_coords(
    genes: list[str],
    programme_genes: list[list[str]],
    programme_peaks: list[list[str]],
) -> pd.DataFrame:
    gene_to_tss: dict[str, int] = {}
    for genes_for_programme, peaks_for_programme in zip(programme_genes, programme_peaks):
        peak_starts = [int(p.split(":")[1].split("-")[0]) for p in peaks_for_programme]
        for i, gene in enumerate(genes_for_programme):
            gene_to_tss[gene] = peak_starts[i % len(peak_starts)] + 250
    rows = []
    for i, gene in enumerate(genes):
        rows.append((gene, "chr1", gene_to_tss.get(gene, 5_000_000 + i * 1_000)))
    return pd.DataFrame(rows, columns=["gene", "chrom", "tss"])


def _search_space(
    programme_genes: list[list[str]],
    programme_peaks: list[list[str]],
) -> pd.DataFrame:
    rows = []
    for genes_for_programme, peaks_for_programme in zip(programme_genes, programme_peaks):
        for gene in genes_for_programme:
            for peak in peaks_for_programme:
                rows.append({"Name": peak, "Gene": gene, "Distance": 0})
    return pd.DataFrame(rows)


def _cistrome(
    peaks: list[str],
    tf_names: list[str],
    programme_peaks: list[list[str]],
) -> pd.DataFrame:
    data = np.zeros((len(peaks), len(tf_names)), dtype=bool)
    peak_to_idx = {peak: i for i, peak in enumerate(peaks)}
    for tf_idx, peaks_for_programme in enumerate(programme_peaks[: len(tf_names)]):
        for peak in peaks_for_programme:
            data[peak_to_idx[peak], tf_idx] = True
    return pd.DataFrame(data, index=peaks, columns=tf_names)


def _rust_cistarget_with_peaks(cistrome: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tf in cistrome.columns:
        for peak in cistrome.index[cistrome[tf].to_numpy()]:
            rows.append(
                {
                    "regulon": f"{tf}_regulon",
                    "motif": f"M_{tf}",
                    "peak_id": peak,
                    "auc": 1.0,
                }
            )
    return pd.DataFrame(rows)


def _safe_float(value: Any, digits: int = 8) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return round(out, digits)


def _top_tf2g_edges(grn: pd.DataFrame, *, top_n: int) -> list[dict[str, Any]]:
    if grn.empty:
        return []
    ordered = grn.sort_values("importance", ascending=False).head(top_n)
    return [
        {
            "key": f"{row.TF}|{row.target}",
            "tf": str(row.TF),
            "target": str(row.target),
            "importance": _safe_float(row.importance),
        }
        for row in ordered.itertuples(index=False)
    ]


def _top_r2g_edges(
    links: pd.DataFrame,
    *,
    tool: str,
    top_n: int,
) -> list[dict[str, Any]]:
    if links.empty:
        return []
    if tool == "rustscenic":
        region_col = "peak_id"
        target_col = "gene"
        rho_col = "correlation"
        score_col = "correlation"
    else:
        region_col = "region"
        target_col = "target"
        rho_col = "rho"
        score_col = "importance"

    ranked = links.assign(_rank_score=links[rho_col].astype(float).abs())
    ranked = ranked.sort_values("_rank_score", ascending=False).head(top_n)
    records = []
    for row in ranked.itertuples(index=False):
        region = str(getattr(row, region_col))
        target = str(getattr(row, target_col))
        rho = getattr(row, rho_col)
        score = getattr(row, score_col)
        records.append(
            {
                "key": f"{region}|{target}",
                "region": region,
                "target": target,
                "rho": _safe_float(rho),
                "score": _safe_float(score),
            }
        )
    return records


def _rust_eregulon_edges(eregulons: Any) -> list[dict[str, str]]:
    if isinstance(eregulons, pd.DataFrame):
        required = {"tf", "enhancer", "target_gene"}
        missing = required - set(eregulons.columns)
        if missing:
            raise ValueError(f"rustscenic eRegulon table missing columns: {sorted(missing)}")
        rows = [
            {
                "key": f"{tf}|{region}|{target}",
                "tf": str(tf),
                "region": str(region),
                "target": str(target),
            }
            for tf, region, target in zip(
                eregulons["tf"],
                eregulons["enhancer"],
                eregulons["target_gene"],
                strict=False,
            )
        ]
        return sorted(rows, key=lambda r: r["key"])

    rows = []
    for er in eregulons:
        for target, peaks in getattr(er, "target_to_peaks", {}).items():
            for peak in peaks:
                rows.append(
                    {
                        "key": f"{er.tf}|{peak}|{target}",
                        "tf": str(er.tf),
                        "region": str(peak),
                        "target": str(target),
                    }
                )
    return sorted(rows, key=lambda r: r["key"])


def _scenicplus_eregulon_edges(eregulons: list[Any]) -> list[dict[str, str]]:
    rows = []
    for er in eregulons:
        tf = str(getattr(er, "transcription_factor"))
        for r2g in getattr(er, "regions2genes"):
            region = str(getattr(r2g, "region"))
            target = str(getattr(r2g, "target"))
            rows.append(
                {
                    "key": f"{tf}|{region}|{target}",
                    "tf": tf,
                    "region": region,
                    "target": target,
                }
            )
    return sorted(rows, key=lambda r: r["key"])


def _tf_from_signature_name(name: Any) -> str:
    text = str(name)
    if "_eregulon_" in text:
        return text.split("_eregulon_", 1)[0]
    return text.rsplit("_", 1)[0]


def _auc_by_tf(auc: pd.DataFrame) -> dict[str, list[float | None]]:
    if auc.empty:
        return {}
    grouped: dict[str, list[np.ndarray]] = {}
    for column in auc.columns:
        grouped.setdefault(_tf_from_signature_name(column), []).append(
            auc[column].to_numpy(dtype=np.float64)
        )
    out: dict[str, list[float | None]] = {}
    for tf, arrays in grouped.items():
        values = np.nanmean(np.vstack(arrays), axis=0)
        out[tf] = [_safe_float(v) for v in values]
    return out


def _output_signature(
    *,
    tool: str,
    data: dict[str, Any],
    tf2g: pd.DataFrame,
    r2g: pd.DataFrame,
    eregulons: list[Any],
    gene_auc: pd.DataFrame,
    region_auc: pd.DataFrame,
    top_n: int,
) -> dict[str, Any]:
    if tool == "rustscenic":
        eregulon_edges = _rust_eregulon_edges(eregulons)
        if isinstance(eregulons, pd.DataFrame):
            eregulon_tfs = sorted({str(tf) for tf in eregulons["tf"]})
        else:
            eregulon_tfs = sorted({str(er.tf) for er in eregulons})
    else:
        eregulon_edges = _scenicplus_eregulon_edges(eregulons)
        eregulon_tfs = sorted({str(er.transcription_factor) for er in eregulons})

    return {
        "top_n": int(top_n),
        "cells": [str(cell) for cell in data["rna"].obs_names],
        "tf_to_gene_top_edges": _top_tf2g_edges(tf2g, top_n=top_n),
        "region_to_gene_top_edges": _top_r2g_edges(r2g, tool=tool, top_n=top_n),
        "eregulon_tfs": eregulon_tfs,
        "eregulon_edges": eregulon_edges,
        "gene_auc_by_tf": _auc_by_tf(gene_auc),
        "region_auc_by_tf": _auc_by_tf(region_auc),
    }


def run_rustscenic(data: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import rustscenic.aucell
    import rustscenic.enhancer
    import rustscenic.eregulon
    import rustscenic.grn

    stage_times: dict[str, float] = {}

    t0 = time.perf_counter()
    grn = rustscenic.grn.infer(
        data["rna"],
        tf_names=data["tf_names"],
        n_estimators=args.grn_estimators,
        max_features=args.max_features,
        seed=args.seed,
        verbose=False,
    )
    stage_times["tf_to_gene"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    links = rustscenic.enhancer.link_peaks_to_genes(
        data["rna"],
        data["atac"],
        data["gene_coords"],
        max_distance=args.max_distance,
        min_abs_corr=args.min_abs_corr,
    )
    stage_times["region_to_gene"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    eregs = rustscenic.eregulon.build_eregulons_dataframe(
        grn,
        _rust_cistarget_with_peaks(data["cistrome"]),
        links,
        min_target_genes=args.min_target_genes,
        min_enhancer_links=1,
    )
    stage_times["eregulons"] = time.perf_counter() - t0
    n_eregulons = int(eregs.attrs.get("n_eregulons", 0))

    gene_regulons = rustscenic.eregulon.regulons_from_dataframe(
        eregs,
        feature_col="target_gene",
        suffix_with_index=True,
    )
    region_regulons = rustscenic.eregulon.regulons_from_dataframe(
        eregs,
        feature_col="enhancer",
        suffix_with_index=True,
    )
    t0 = time.perf_counter()
    gene_auc = rustscenic.aucell.score(data["rna"], gene_regulons, top_frac=args.top_frac)
    region_auc = rustscenic.aucell.score(
        data["atac"], region_regulons, top_frac=args.top_frac
    )
    stage_times["aucell"] = time.perf_counter() - t0

    out = {
        "stage_times": stage_times,
        "output_counts": {
            "tf_to_gene_edges": int(len(grn)),
            "region_to_gene_links": int(len(links)),
            "eregulons": n_eregulons,
            "gene_auc_shape": list(gene_auc.shape),
            "region_auc_shape": list(region_auc.shape),
        },
    }
    if args.save_signatures:
        out["output_signature"] = _output_signature(
            tool="rustscenic",
            data=data,
            tf2g=grn,
            r2g=links,
            eregulons=eregs,
            gene_auc=gene_auc,
            region_auc=region_auc,
            top_n=args.signature_top_n,
        )
    return out


def run_scenicplus(data: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    from scenicplus.TF_to_gene import calculate_TFs_to_genes_relationships
    from scenicplus.enhancer_to_gene import calculate_regions_to_genes_relationships
    from scenicplus.eregulon_enrichment import score_eRegulons
    from scenicplus.grn_builder.gsea_approach import build_grn

    stage_times: dict[str, float] = {}
    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)

        t0 = time.perf_counter()
        tf2g = calculate_TFs_to_genes_relationships(
            data["expr_df"],
            data["tf_names"],
            temp_dir=temp_dir,
            method="GBM",
            n_cpu=args.n_cpu,
            seed=args.seed,
        )
        stage_times["tf_to_gene"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        r2g = calculate_regions_to_genes_relationships(
            data["expr_df"],
            data["acc_df"],
            data["search_space"],
            temp_dir=temp_dir,
            mask_expr_dropout=False,
            importance_scoring_method="GBM",
            correlation_scoring_method="PR",
            n_cpu=args.n_cpu,
            add_distance=True,
        )
        stage_times["region_to_gene"] = time.perf_counter() - t0

        cistromes = ad.AnnData(
            X=data["cistrome"].to_numpy(dtype=bool),
            obs=pd.DataFrame(index=data["cistrome"].index),
            var=pd.DataFrame(index=data["cistrome"].columns),
        )
        t0 = time.perf_counter()
        eregs = build_grn(
            tf2g,
            r2g,
            cistromes,
            is_extended=False,
            temp_dir=str(temp_dir),
            gsea_n_perm=args.gsea_permutations,
            quantiles=(),
            top_n_regionTogenes_per_gene=(args.top_regions_per_gene,),
            top_n_regionTogenes_per_region=(),
            min_regions_per_gene=0,
            rho_dichotomize_tf2g=True,
            rho_dichotomize_r2g=False,
            rho_dichotomize_eregulon=False,
            keep_only_activating=False,
            rho_threshold=0.0,
            NES_thr=-999.0,
            adj_pval_thr=1.0,
            min_target_genes=args.min_target_genes,
            n_cpu=args.n_cpu,
            merge_eRegulons=False,
            disable_tqdm=True,
            seed=args.seed,
        )
        stage_times["eregulons"] = time.perf_counter() - t0

        metadata = _scenicplus_eregulon_metadata(eregs)
        t0 = time.perf_counter()
        auc = score_eRegulons(
            metadata,
            data["expr_df"],
            data["acc_df"],
            auc_threshold=args.top_frac,
            normalize=False,
            n_cpu=args.n_cpu,
        )
        stage_times["aucell"] = time.perf_counter() - t0

    out = {
        "stage_times": stage_times,
        "output_counts": {
            "tf_to_gene_edges": int(len(tf2g)),
            "region_to_gene_links": int(len(r2g)),
            "eregulons": int(len(eregs)),
            "gene_auc_shape": list(auc["Gene_based"].shape),
            "region_auc_shape": list(auc["Region_based"].shape),
        },
    }
    if args.save_signatures:
        out["output_signature"] = _output_signature(
            tool="scenicplus",
            data=data,
            tf2g=tf2g,
            r2g=r2g,
            eregulons=eregs,
            gene_auc=auc["Gene_based"],
            region_auc=auc["Region_based"],
            top_n=args.signature_top_n,
        )
    return out


def _scenicplus_eregulon_metadata(eregulons: list[Any]) -> pd.DataFrame:
    rows = []
    for i, er in enumerate(eregulons):
        region_name = f"{er.transcription_factor}_eregulon_{i}_region"
        gene_name = f"{er.transcription_factor}_eregulon_{i}_gene"
        for r2g in er.regions2genes:
            rows.append(
                {
                    "Region_signature_name": region_name,
                    "Gene_signature_name": gene_name,
                    "Region": getattr(r2g, "region"),
                    "Gene": getattr(r2g, "target"),
                }
            )
    if not rows:
        raise RuntimeError("SCENIC+ produced zero eRegulons")
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tool", choices=["rustscenic", "scenicplus"], required=True)
    p.add_argument("--input-10x-h5", type=Path, default=None)
    p.add_argument("--dataset-name", default="")
    p.add_argument("--species", default="hs")
    p.add_argument("--n-cells", type=int, default=150)
    p.add_argument("--n-genes", type=int, default=80)
    p.add_argument("--n-peaks", type=int, default=30)
    p.add_argument("--n-programmes", type=int, default=3)
    p.add_argument("--n-tfs", type=int, default=0)
    p.add_argument("--n-cpu", type=int, default=4)
    p.add_argument("--grn-estimators", type=int, default=5000)
    p.add_argument("--max-features", type=float, default=0.1)
    p.add_argument("--min-abs-corr", type=float, default=0.2)
    p.add_argument("--max-distance", type=int, default=500_000)
    p.add_argument("--min-target-genes", type=int, default=2)
    p.add_argument("--top-frac", type=float, default=0.05)
    p.add_argument("--top-regions-per-gene", type=int, default=5)
    p.add_argument("--gsea-permutations", type=int, default=25)
    p.add_argument("--rss-poll-interval", type=float, default=0.25)
    p.add_argument("--save-signatures", action="store_true")
    p.add_argument("--signature-top-n", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--label", default="")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    prep_t0 = time.perf_counter()
    with PeakSampler(args.rss_poll_interval) as prep_mem:
        if args.input_10x_h5 is None:
            data = synthetic_multiome(
                n_cells=args.n_cells,
                n_genes=args.n_genes,
                n_peaks=args.n_peaks,
                n_programmes=args.n_programmes,
                seed=args.seed,
            )
        else:
            data = real_10x_multiome(
                input_10x_h5=args.input_10x_h5,
                dataset_name=args.dataset_name or args.input_10x_h5.stem,
                species=args.species,
                n_cells=args.n_cells,
                n_genes=args.n_genes,
                n_peaks=args.n_peaks,
                n_tfs=args.n_tfs or args.n_programmes,
                max_distance=args.max_distance,
                seed=args.seed,
            )
    prep_wall = time.perf_counter() - prep_t0

    status = "ok"
    error = None
    compute_t0 = time.perf_counter()
    with PeakSampler(args.rss_poll_interval) as compute_mem:
        try:
            if args.tool == "rustscenic":
                result = run_rustscenic(data, args)
            else:
                result = run_scenicplus(data, args)
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            result = {"stage_times": {}, "output_counts": {}}
    wall = time.perf_counter() - compute_t0
    resource_peak = peak_rss_gb()
    sampled_peak = max(prep_mem.peak_rss_gb, compute_mem.peak_rss_gb)

    record = {
        "label": args.label,
        "tool": args.tool,
        "benchmark": "core_e2e_matrix",
        "status": status,
        "error": error,
        "wall_s": round(float(wall), 6),
        "data_prep_wall_s": round(float(prep_wall), 6),
        "total_wall_s": round(float(prep_wall + wall), 6),
        "peak_rss_gb": round(float(sampled_peak), 6),
        "start_rss_gb": round(float(compute_mem.start_rss_gb), 6),
        "resource_peak_rss_gb": round(float(resource_peak), 6),
        "memory_gb": {
            "data_prep": prep_mem.record(),
            "compute": compute_mem.record(),
            "sampled_process_peak": round(float(sampled_peak), 6),
            "resource_process_peak": round(float(resource_peak), 6),
        },
        "stage_times": {
            k: round(float(v), 6) for k, v in result["stage_times"].items()
        },
        "output_counts": result["output_counts"],
        "output_signature": result.get("output_signature"),
        "settings": {
            **data["settings"],
            "n_cpu": args.n_cpu,
            "grn_estimators": args.grn_estimators,
            "max_features": args.max_features,
            "min_abs_corr": args.min_abs_corr,
            "max_distance": args.max_distance,
            "min_target_genes": args.min_target_genes,
            "top_frac": args.top_frac,
            "top_regions_per_gene": args.top_regions_per_gene,
            "gsea_permutations": args.gsea_permutations,
            "save_signatures": args.save_signatures,
            "signature_top_n": args.signature_top_n,
            "seed": args.seed,
        },
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    printable = dict(record)
    if printable.get("output_signature") is not None:
        signature = printable["output_signature"]
        printable["output_signature"] = {
            "saved": True,
            "top_n": signature["top_n"],
            "cells": len(signature["cells"]),
            "tf_to_gene_top_edges": len(signature["tf_to_gene_top_edges"]),
            "region_to_gene_top_edges": len(signature["region_to_gene_top_edges"]),
            "eregulon_tfs": len(signature["eregulon_tfs"]),
            "eregulon_edges": len(signature["eregulon_edges"]),
            "gene_auc_by_tf": len(signature["gene_auc_by_tf"]),
            "region_auc_by_tf": len(signature["region_auc_by_tf"]),
        }
    print(json.dumps(printable, indent=2), flush=True)
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
