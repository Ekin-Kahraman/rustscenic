"""Reproduce donor-level lineage effects from frozen BMMC and Kang outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import roc_auc_score

from trace import MARKERS, sha256

EXPECTED = {
    "BMMC": {"SPI1": ["CD14+ Mono", "CD16+ Mono", "cDC2"],
             "GATA1": ["Erythroblast", "Proerythroblast", "Normoblast"],
             "PAX5": ["Naive CD20+ B", "Transitional B", "B1 B"],
             "EBF1": ["Naive CD20+ B", "Transitional B", "B1 B"],
             "TCF7": ["CD4+ T naive", "CD8+ T naive"]},
    "Kang": {"SPI1": ["CD14+ Monocytes", "FCGR3A+ Monocytes"],
             "GATA1": [], "PAX5": ["B cells"], "EBF1": ["B cells"],
             "TCF7": ["CD4 T cells", "CD8 T cells"]},
}


def assess(frame, meta, cohort, n, seed, method, measurement):
    if not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError("Duplicate activity axes")
    meta = meta.loc[frame.index]
    rows = []
    for tf in MARKERS:
        for donor, group in meta.groupby("donor", sort=True):
            labels = EXPECTED[cohort][tf]
            target = group.cell_type.isin(labels).to_numpy()
            base = dict(cohort=cohort, training_cells=n, seed=seed, method=method,
                        measurement=measurement, TF=tf, donor=donor,
                        target_cells=int(target.sum()), other_cells=int((~target).sum()))
            if not labels:
                rows.append(dict(base, status="lineage_not_sampled")); continue
            if tf not in frame:
                rows.append(dict(base, status="missing_regulon")); continue
            if min(target.sum(), (~target).sum()) < 20:
                rows.append(dict(base, status="insufficient_cells")); continue
            values = frame.loc[group.index, tf].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite activity")
            auroc = float(roc_auc_score(target, values))
            rows.append(dict(base, status="assessed", auroc=auroc,
                rank_biserial=2 * auroc - 1, target_mean=float(values[target].mean()),
                other_mean=float(values[~target].mean()),
                mean_difference=float(values[target].mean() - values[~target].mean())))
    return rows


def score_sets(contributions, genes, cells, signatures):
    gi = {g: i for i, g in enumerate(genes)}
    rr, cc = [], []
    names = sorted(signatures)
    denominators = np.ones(len(names), dtype=float)
    for j, name in enumerate(names):
        targets = set(signatures[name])
        present = targets & gi.keys()
        if not targets or len(present) / len(targets) < .8:
            continue
        denominators[j] = round(.05 * len(genes)) * len(present)
        for g in sorted(present):
            rr.append(gi[g]); cc.append(j)
    membership = sparse.csc_matrix((np.ones(len(rr), np.int64), (rr, cc)), shape=(len(genes), len(names)))
    # Sum the integer rank contributions exactly, then normalise once.
    # Summing fractional per-target weights can spuriously break AUROC ties.
    values = (contributions @ membership).toarray() / denominators
    return pd.DataFrame(values, index=cells, columns=names)


def verify_rank_cache(expression, contributions):
    """Verify every cached rank against expression, including deterministic ties."""
    cutoff = round(.05 * expression.shape[1]) - 1
    order = pd.Series(np.arange(expression.shape[1])).sample(frac=1, random_state=20261106).to_numpy()
    for start in range(0, expression.shape[0], 512):
        block = expression[start:start + 512].toarray()
        top = np.argsort(-block[:, order], axis=1, kind="stable")[:, :cutoff]
        expected = sparse.csr_matrix((np.tile(np.arange(cutoff, 0, -1, dtype=float), len(block)),
            order[top].ravel(), np.arange(len(block) + 1) * cutoff), shape=block.shape)
        if (expected != contributions[start:start + len(block)]).nnz:
            raise ValueError("Rank cache does not match the frozen expression input")


def run(root, output):
    output.mkdir(parents=True, exist_ok=True)
    sources, records, matrices, checks, expected_hashes, topic_rows = {}, [], [], [], {}, []

    def tracked(path):
        sources[str(path.resolve())] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        if path in expected_hashes and sources[str(path.resolve())]["sha256"] != expected_hashes[path]:
            raise ValueError(f"Returned source checksum mismatch: {path}")
        return path

    inputs = root / "RustScenic-Paper-Validation-20261106/rustscenic-paper-validation/holdout-inputs"
    bio = root / "RustScenic-Biology-Results-Audit-20261106/returned/results"
    external = root / "RustScenic-Confirmation-Results-Audit-20261107/returned/biology"
    fixed = root / "RustScenic-Scoring-Results-Audit-20261108/returned/biology/seeded"
    kang_inputs = root / "RustScenic-Confirmation-20261107/rustscenic-confirmation/external"
    for returned_root in [bio.parent, external.parent, fixed.parent.parent]:
        manifest = json.loads(tracked(returned_root / "return-manifest.json").read_text())
        manifest = manifest.get("files", manifest)
        expected_hashes.update({returned_root / name: item["sha256"] for name, item in manifest.items()})
    for folder in [inputs, kang_inputs]:
        manifest = json.loads(tracked(folder / "input.json").read_text())
        expected_hashes.update({folder / name: digest for name, digest in manifest["files"].items()})
    metas = {cohort: pd.read_csv(tracked(folder / "cell-metadata.tsv"), sep="\t", dtype={"donor": str}).set_index("cell")
             for cohort, folder in [("BMMC", inputs), ("Kang", kang_inputs)]}
    info = json.loads(tracked(inputs / "input.json").read_text())
    bmeta = metas["BMMC"]
    if set(info["train_donors"]) & set(info["test_donors"]):
        raise ValueError("Overlapping train/test donors")
    test = bmeta.index[bmeta.split.eq("test")]
    if set(bmeta.loc[test, "donor"]) != set(info["test_donors"]):
        raise ValueError("Test donor roster changed")
    with h5py.File(tracked(inputs / "rna.h5")) as h:
        genes, cells = h["features"].asstr()[:].tolist(), h["cells"].asstr()[:].tolist()
        x = sparse.csr_matrix((h["data"][:], h["indices"][:], h["indptr"][:]), shape=(len(cells), len(genes)))
    expression = pd.DataFrame(x[:, [genes.index(tf) for tf in MARKERS]].toarray(), index=cells, columns=MARKERS)
    cache = root / "RustScenic-Biological-Fidelity-20261107/rna-rank-contributions.npz"
    contributions = sparse.load_npz(tracked(cache))
    if contributions.shape != x.shape or cells != bmeta.index.tolist():
        raise ValueError("Rank cache axes changed")
    verify_rank_cache(x, contributions)

    def add(frame, cohort, n, seed, method, measurement):
        meta = metas[cohort]
        if cohort == "BMMC":
            frame = frame.loc[test]
        records.extend(assess(frame, meta, cohort, n, seed, method, measurement))
        # Small, inspectable source data for rebuilding the donor figure.
        short = frame.reindex(columns=MARKERS).astype(float)
        short["cell"] = short.index
        for k, v in dict(cohort=cohort, training_cells=n, seed=seed, method=method, measurement=measurement).items():
            short[k] = v
        short["donor"] = meta.loc[frame.index, "donor"].values
        short["cell_type"] = meta.loc[frame.index, "cell_type"].values
        matrices.append(short.reset_index(drop=True))

    add(expression, "BMMC", 20000, 0, "expression", "TF expression")
    motif_scores = pd.read_csv(tracked(output / "marker-motif-scores.csv"))
    for n in (5000, 20000):
        for seed in (777, 778, 779):
            pair = bio / f"bmmc-train{n}-seed{seed}"
            for method in ("rustscenic", "reference"):
                topics = pd.read_parquet(tracked(pair / method / "cell-topic.parquet"))
                if not bmeta.loc[topics.index, "split"].eq("train").all():
                    raise ValueError("Test cells entered topic diagnostics")
                for tf in MARKERS:
                    target = bmeta.loc[topics.index, "cell_type"].isin(EXPECTED["BMMC"][tf])
                    motif = motif_scores[motif_scores.training_cells.eq(n) & motif_scores.seed.eq(seed)
                        & motif_scores.method.eq(method) & motif_scores.TF.eq(tf)].groupby("topic").nes.max()
                    for topic in topics:
                        topic_rows.append(dict(training_cells=n, seed=seed, method=method, TF=tf,
                            topic=topic, training_lineage_auroc=roc_auc_score(target, topics[topic]),
                            marker_max_nes=float(motif[topic]), target_cells=int(target.sum()),
                            scope="Post hoc training-cell diagnostic; all 30 topics, not held-out validation"))
                sets = json.loads(tracked(pair / method / "regulons.json").read_text())
                marker_sets = {tf: sets[tf] for tf in MARKERS if tf in sets}
                frame = pd.read_parquet(tracked(pair / f"{method}-common-scorer.parquet"))
                recalculated = score_sets(contributions, genes, cells, marker_sets)
                if marker_sets:
                    error = float(np.max(np.abs(recalculated.loc[frame.index].to_numpy() - frame[recalculated.columns].to_numpy())))
                    if error > 1e-12:
                        raise ValueError("Cached ranking does not reproduce saved scores")
                else:
                    error = 0.
                checks.append(dict(training_cells=n, seed=seed, method=method, max_abs_cached_scorer_error=error))
                add(frame, "BMMC", n, seed, method, "Gene activity")
                regions = pd.read_parquet(tracked(pair / f"{method}-regions-common-scorer.parquet"))
                add(regions, "BMMC", n, seed, method, "Observed-region activity")
                # Five fixed controls per signature/seed; preserve target counts.
                # They measure nonspecific expression information, not null biology.
                for repeat in range(5):
                    rng = np.random.default_rng(20261106 + repeat)
                    random = {tf: sorted(rng.choice([g for g in genes if g != tf], len(targets), replace=False))
                              for tf, targets in sorted(marker_sets.items())}
                    random_scores = score_sets(contributions, genes, cells, random).loc[test]
                    for row in assess(random_scores, bmeta, "BMMC", n, seed, method, "Random-target gene activity"):
                        row["repeat"] = repeat; records.append(row)
                # Remove the TF's own gene to test trivial self-expression leakage.
                no_self = {tf: [g for g in targets if g != tf] for tf, targets in marker_sets.items()}
                no_self = {tf: v for tf, v in no_self.items() if v}
                add(score_sets(contributions, genes, cells, no_self), "BMMC", n, seed, method, "Gene activity without self")
                kpair = external / f"{n}-{seed}"
                for mode, measure in [("union", "Gene activity"), ("signed", "Signed gene activity")]:
                    reference = pd.read_parquet(tracked(kpair / f"{method}-{mode}-shared.parquet"))
                    if method == "rustscenic":
                        frame = pd.read_parquet(tracked(fixed / f"{n}-{seed}-{mode}.parquet"))
                        if not frame.index.equals(reference.index) or not frame.columns.equals(reference.columns):
                            raise ValueError("Corrected native axes changed")
                        error = float(np.max(np.abs(frame.to_numpy() - reference.to_numpy())))
                        if error > 1e-7:
                            raise ValueError("Corrected native scorer mismatch")
                    else:
                        frame = reference
                    add(frame, "Kang", n, seed, method, measure)
    result = pd.DataFrame(records)
    topic_result = pd.DataFrame(topic_rows)
    topic_result.to_csv(output / "training-topic-lineage.csv", index=False)
    best = topic_result.loc[topic_result.groupby(["training_cells", "seed", "method", "TF"]).training_lineage_auroc.idxmax()]
    best.to_csv(output / "training-topic-lineage-best.csv", index=False)
    result.to_csv(output / "lineage-donor-results.csv", index=False)
    pd.concat(matrices, ignore_index=True).to_parquet(output / "lineage-activity-source.parquet", index=False)
    # Equal weight per biological donor, with algorithm seeds averaged first.
    keys = ["cohort", "training_cells", "method", "measurement", "TF"]
    assessed = result[result.status.eq("assessed")]
    donor = assessed.groupby(keys + ["donor"])[["auroc", "rank_biserial", "mean_difference"]].mean().reset_index()
    donor.to_csv(output / "lineage-donor-means.csv", index=False)
    summary = donor.groupby(keys).agg(donors=("donor", "nunique"), mean_auroc=("auroc", "mean"),
        min_donor_auroc=("auroc", "min"), max_donor_auroc=("auroc", "max"),
        mean_difference=("mean_difference", "mean")).reset_index()
    summary.to_csv(output / "lineage-summary.csv", index=False)
    (output / "case-verification.json").write_text(json.dumps(dict(sources=sources,
        scorer_reproduction=checks, records=len(result), no_model_fitting=True,
        rank_cache_verified_against_all_expression_rows=True,
        post_hoc_case_study=True, unit_of_replication="donor; seeds averaged within donors"), indent=2) + "\n")
    print(summary[(summary.training_cells == 20000) & summary.measurement.isin(
        ["Gene activity", "Observed-region activity", "Random-target gene activity", "TF expression"])].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Directory containing the existing dated return packages")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
