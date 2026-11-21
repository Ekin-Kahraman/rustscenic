"""Fixed-universe reference retrieval; missing reference links are unlabelled."""
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def expected_hits_at_k(labels, scores, k):
    """Fractional expectation at a tied cutoff, avoiding arbitrary tie ordering."""
    k = min(int(k), len(labels))
    if k <= 0:
        return 0.0
    threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
    above = scores > threshold
    tied = scores == threshold
    return float(labels[above].sum() + (k - above.sum()) * labels[tied].mean())


def reference_metrics(network, reference, genes, tfs):
    known = {(str(a), str(b)) for a, b in reference[["TF", "target"]].itertuples(index=False, name=None)
             if a in tfs and b in genes and a != b}
    supported = sorted({tf for tf, _ in known})
    if not known or not supported:
        raise ValueError("No eligible experimental reference")
    pairs = [(tf, target) for tf in supported for target in genes if tf != target]
    index = pd.MultiIndex.from_tuples(pairs, names=["TF", "target"])
    scores = network.set_index(["TF", "target"]).importance.reindex(index, fill_value=0).to_numpy(dtype=float)
    labels = np.array([p in known for p in pairs], dtype=np.int8)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite score")
    ap = float(average_precision_score(labels, scores))
    prevalence = float(labels.mean())
    per_tf = []
    hit50 = budget50 = expected50 = 0.0
    for tf in supported:
        mask = index.get_level_values("TF") == tf
        y, s = labels[mask], scores[mask]
        k = min(50, len(y))
        hits = expected_hits_at_k(y, s, k)
        hit50 += hits
        budget50 += k
        expected50 += k * float(y.mean())
        per_tf.append({"TF": tf, "eligible_reference_links": int(y.sum()), "candidate_edges": len(y),
                       "average_precision": float(average_precision_score(y, s)),
                       "reference_prevalence": float(y.mean()), "top50_expected_hits_tie_aware": hits,
                       "positive_predictions": int((s > 0).sum())})
    ep_hits = expected_hits_at_k(labels, scores, int(labels.sum()))
    return {
        "candidate_edges": len(labels), "eligible_reference_links": int(labels.sum()),
        "reference_covered_tfs": len(supported), "panel_tfs": len(tfs),
        "average_precision": ap, "reference_prevalence": prevalence,
        "ap_over_prevalence": ap / prevalence,
        "early_precision_k": int(labels.sum()), "early_precision_reference_support": ep_hits / labels.sum(),
        "early_precision_over_prevalence": ep_hits / labels.sum() / prevalence,
        "top50_expected_reference_hits_tie_aware": hit50, "top50_candidate_budget": int(budget50),
        "top50_uniform_expected_hits": expected50, "top50_fold_over_uniform": hit50 / expected50,
        "per_tf": per_tf,
    }


def positive_top_sets(network, tfs, k=50):
    result = {}
    for tf in tfs:
        group = network.loc[(network.TF == tf) & (network.importance > 0)]
        group = group.sort_values(["importance", "target"], ascending=[False, True], kind="stable")
        result[tf] = set(group.target.head(k))
    return result


def retained_support(pruned, original, reference, genes, tfs):
    """Account for pruning's reduced coverage; compare equal per-TF edge budgets."""
    known={(str(tf),str(g)) for tf,g in reference[['TF','target']].itertuples(index=False,name=None)
           if tf in tfs and g in genes and tf!=g}
    supported={tf for tf,g in known}
    predicted={(str(tf),str(g)) for tf,g,w in pruned[['TF','target','importance']].itertuples(index=False,name=None)
               if tf in supported and g in genes and tf!=g and w>0}
    hits=len(predicted&known);uniform=0.;raw_hits=0.
    for tf in sorted(supported):
        targets=[g for g in genes if g!=tf];truth=np.array([(tf,g) in known for g in targets],dtype=np.int8)
        k=sum(t==tf for t,g in predicted)
        scores=original.loc[original.TF==tf].set_index('target').importance.reindex(targets,fill_value=0).to_numpy()
        uniform+=k*float(truth.mean());raw_hits+=expected_hits_at_k(truth,scores,k)
    return {'retained_edges_for_reference_tfs':len(predicted),'retained_reference_hits':hits,
            'retained_reference_supported_fraction':hits/len(predicted) if predicted else None,
            'retained_reference_recall':hits/len(known) if known else None,
            'same_degree_uniform_expected_hits':uniform,
            'same_degree_original_grn_expected_hits_tie_aware':raw_hits,
            'retained_hits_over_uniform':hits/uniform if uniform else None,
            'retained_hits_over_matched_original':hits/raw_hits if raw_hits else None}


def seed_stability(networks, tfs):
    rows = []
    for (seed_a, a), (seed_b, b) in itertools.combinations(sorted(networks.items()), 2):
        x, y = positive_top_sets(a, tfs), positive_top_sets(b, tfs)
        values = [len(x[tf] & y[tf]) / len(x[tf] | y[tf]) for tf in tfs if x[tf] | y[tf]]
        rows.append({"seed_a": seed_a, "seed_b": seed_b, "top50_mean_jaccard": float(np.mean(values)) if values else None,
                     "tfs_evaluated": len(values), "both_empty_tfs": len(tfs) - len(values)})
    return rows
