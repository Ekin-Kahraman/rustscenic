"""Trace regulator loss using saved topics and exact, checksummed motif ranks.

No model fitting. Sparse rank contributions avoid materialising the full
motif-by-region matrix. The independent float64 calculation is checked against
both returned implementations, including their enrichment threshold decisions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

MARKERS = ("SPI1", "GATA1", "PAX5", "EBF1", "TCF7")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def decode21(data, count):
    bits = np.unpackbits(np.frombuffer(data, np.uint8), bitorder="little")
    expanded = np.zeros((count, 32), np.uint8)
    expanded[:, :21] = bits[:count * 21].reshape(count, 21)
    return np.packbits(expanded, axis=1, bitorder="little").view("<u4").ravel()


def rank_contributions(folder):
    meta = json.loads((folder / "ranks.json").read_text())
    cutoff = round(.005 * meta["rank_universe"]) - 1
    rows, cols, weights, queries = [], [], [], []
    with (folder / "ranks.21bit").open("rb") as stream:
        for block in meta["blocks"]:
            stream.seek(block["offset"])
            values = decode21(stream.read(block["bytes"]), block["padded"])
            values = values[:block["count"]].reshape(-1, len(meta["motifs"]))
            if hashlib.sha256(values.tobytes()).hexdigest() != block["decoded_sha256"]:
                raise ValueError("Ranking transport checksum mismatch")
            query, motif = np.nonzero(values < cutoff)
            rows.append(motif)
            cols.append(query + len(queries))
            weights.append(cutoff - values[query, motif].astype(np.int64))
            queries.extend(block["columns"])
    if len(set(queries)) != len(queries):
        raise ValueError("Duplicate ranking columns")
    matrix = sparse.csr_matrix((np.concatenate(weights),
                               (np.concatenate(rows), np.concatenate(cols))),
                              shape=(len(meta["motifs"]), len(queries)))
    return matrix, meta["motifs"], queries, cutoff, len(meta["blocks"])


def enrichment(contributions, membership, cutoff):
    sizes = np.asarray(membership.sum(axis=0)).ravel()
    if np.any(sizes == 0):
        raise ValueError("Unmapped region set")
    auc = (contributions @ membership).toarray() / ((cutoff + 1) * sizes)
    sd = auc.std(axis=0)
    nes = np.divide(auc - auc.mean(axis=0), sd,
                    out=np.full_like(auc, np.nan), where=sd >= 1e-6)
    return auc, nes


def run(package, returned, output):
    output.mkdir(parents=True, exist_ok=True)
    sources, checks, sets, rosters, topics = {}, [], {}, {}, {}

    def record(path, expected=None):
        key = str(path.resolve())
        if key not in sources:
            sources[key] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        if expected is not None and sources[key]["sha256"] != expected:
            raise ValueError(f"Source hash mismatch: {path}")

    inputs = package / "holdout-inputs"
    package_hashes = {}
    for line in (package / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        package_hashes[name.lstrip("*")] = digest
    info = json.loads((inputs / "input.json").read_text())
    for name, expected in info["files"].items():
        record(inputs / name, expected)
    motifs_dir = inputs / "motifs"
    for name in ["ranks.json", "ranks.21bit", "human-gene-annotations.tbl", "screen-overlap-mapping.tsv"]:
        path = motifs_dir / name
        record(path, package_hashes[str(path.relative_to(package))])
    annotation = pd.read_csv(motifs_dir / "human-gene-annotations.tbl", sep="\t")
    eligible = annotation.description.eq("gene is directly annotated") | (
        annotation.description.str.contains("orthologous") & ~annotation.description.str.contains("similar"))
    annotation = annotation.loc[eligible]
    mapping = pd.read_csv(motifs_dir / "screen-overlap-mapping.tsv", sep="\t")
    peak_queries = mapping.groupby("Target").Query.agg(set).to_dict()
    meta = pd.read_csv(inputs / "cell-metadata.tsv", sep="\t", dtype={"donor": str}).set_index("cell")
    for n in (5000, 20000):
        for seed in (777, 778, 779):
            for method in ("rustscenic", "reference"):
                folder = returned / "results" / f"bmmc-train{n}-seed{seed}" / method
                complete = json.loads((folder / "complete.json").read_text())
                for name in ["topic-peak.parquet", "cell-topic.parquet", "region-motif-scores.parquet",
                             "grn.parquet", "regulons.json", "full-motif-evidence.parquet"]:
                    record(folder / name, complete["outputs"][name])
                key = (n, seed, method)
                peak_topic = pd.read_parquet(folder / "topic-peak.parquet")
                topics[key] = pd.read_parquet(folder / "cell-topic.parquet")
                if not meta.loc[topics[key].index, "split"].eq("train").all():
                    raise ValueError("Test donor entered topic model")
                rosters[key] = (folder, peak_topic)
                for name, values in peak_topic.iterrows():
                    peaks = values.nlargest(1000).index
                    sets[(*key, name)] = set().union(*(peak_queries.get(p, set()) for p in peaks))
    contributions, motifs, queries, cutoff, blocks = rank_contributions(motifs_dir)
    qi = {name: i for i, name in enumerate(queries)}
    set_names = list(sets)
    rr, cc = [], []
    for j, name in enumerate(set_names):
        for query in sorted(sets[name]):
            rr.append(qi[query]); cc.append(j)
    membership = sparse.csc_matrix((np.ones(len(rr), np.int64), (rr, cc)),
                                   shape=(len(queries), len(set_names)))
    auc, nes = enrichment(contributions, membership, cutoff)
    motif_index = {name: i for i, name in enumerate(motifs)}
    set_index = {name: i for i, name in enumerate(set_names)}
    marker_rows, detail_rows = [], []
    for key, (folder, peak_topic) in rosters.items():
        n, seed, method = key
        scores = pd.read_parquet(folder / "region-motif-scores.parquet")
        mi = [motif_index[x] for x in scores.motif]
        si = [set_index[(*key, x)] for x in scores.region_set]
        auc_error = np.max(np.abs(auc[mi, si] - scores.auc))
        nes_error = np.max(np.abs(nes[mi, si] - scores.nes))
        if auc_error > 1e-7 or nes_error > 1e-3:
            raise ValueError(f"Returned enrichment differs: {key}, {auc_error}, {nes_error}")
        checks.append(dict(training_cells=n, seed=seed, method=method,
                           returned_rows=len(scores), max_auc_error=auc_error, max_nes_error=nes_error))
        grn = pd.read_parquet(folder / "grn.parquet")
        evidence = pd.read_parquet(folder / "full-motif-evidence.parquet")
        regulons = json.loads((folder / "regulons.json").read_text())
        for tf in MARKERS:
            annotated = set(annotation.loc[annotation.gene_name.eq(tf), "#motif_id"]) & set(motifs)
            ix = [motif_index[x] for x in sorted(annotated)]
            js = [set_index[(*key, x)] for x in peak_topic.index]
            values = nes[np.ix_(ix, js)]
            a, b = np.unravel_index(np.nanargmax(values), values.shape)
            enriched = scores.loc[scores.motif.isin(annotated)]
            # All marker decisions, including rejected motifs, must reproduce.
            expected = {(peak_topic.index[j], motifs[ix[i]]) for i, j in zip(*np.where(values >= 3))}
            observed = set(zip(enriched.region_set, enriched.motif))
            if expected != observed:
                raise ValueError(f"Marker threshold decisions changed: {key}, {tf}")
            support_queries = contributions[ix].getnnz(axis=0) > 0
            supported = {queries[i] for i in np.flatnonzero(support_queries)}
            all_peaks = set(peak_topic.columns)
            hit_peaks = set(mapping.loc[mapping.Query.isin(supported), "Target"]) & all_peaks
            selected_queries = set().union(*(sets[(*key, t)] for t in peak_topic.index))
            marker_rows.append(dict(training_cells=n, seed=seed, method=method, TF=tf,
                database_motifs=len(annotated), grn_edges=int(grn.TF.eq(tf).sum()),
                motif_hit_peaks_in_panel=len(hit_peaks),
                motif_hit_queries_in_selected_sets=len(supported & selected_queries),
                max_nes=float(values[a, b]), best_motif=motifs[ix[a]], best_topic=peak_topic.index[b],
                passing_motif_topic_pairs=len(enriched),
                supported_peaks=int(evidence.loc[evidence.TF.eq(tf), "peak_id"].nunique()),
                final_regulon=tf in regulons))
            for i, motif in zip(ix, sorted(annotated)):
                for topic, j in zip(peak_topic.index, js):
                    detail_rows.append(dict(training_cells=n, seed=seed, method=method, TF=tf,
                        topic=topic, motif=motif, auc=auc[i, j], nes=nes[i, j], query_regions=len(sets[(*key, topic)])))
    pd.DataFrame(marker_rows).to_csv(output / "regulator-trace.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(output / "marker-motif-scores.csv", index=False)
    pd.DataFrame(checks).to_csv(output / "enrichment-reproduction.csv", index=False)
    receipt = dict(sources=sources, decoded_blocks_verified=blocks,
        motif_count=len(motifs), query_region_count=len(queries), topic_sets=len(sets),
        rank_cutoff=cutoff, rank_contribution_nonzeros=int(contributions.nnz),
        no_network_or_topic_fitting=True, no_threshold_changes=True)
    (output / "trace-verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(pd.DataFrame(marker_rows).groupby(["method", "TF"]).agg(
        min_nes=("max_nes", "min"), max_nes=("max_nes", "max"), recovered=("final_regulon", "sum")).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--returned", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.package, args.returned, args.output)
