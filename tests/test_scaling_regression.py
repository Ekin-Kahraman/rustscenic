"""Scaling regression test — catches super-linearity creep.

Measures GRN wall-time at 1k / 2k / 4k / 8k synthetic cells on every
CI run and asserts the log-log slope stays below 1.30. A clean linear
run is slope ≈ 1.0. PR #12's partition-buffer pool measured 1.05–1.17
on real data. Anything above 1.30 means super-linearity has come back
in — the test fails and the build stops.

Uses synthetic data with 50 target genes + 10 TFs + n_estimators=20
so the whole test runs in ~15 s on CI. This is not a benchmark; it's
a tripwire on the algorithmic complexity guarantee.

If this test gets flaky on CI runners with wildly varying load, bump
the threshold to 1.40 and document. But do not disable it — the whole
point is that if the slope goes up, we find out immediately.
"""
from __future__ import annotations

import time
from math import log

import numpy as np

import rustscenic.grn


# Kept small so the test runs under ~15 s total on CI.
# Four points covers one decade of cell counts (1k → 8k, 8× range),
# which is enough to fit a stable log-log slope.
CELL_COUNTS = [1_000, 2_000, 4_000, 8_000]
N_GENES = 80          # few enough to be fast; the TFs are a subset of these
N_TFS = 10            # constant across sizes so per-feature cost doesn't move
N_ESTIMATORS = 20     # reduced from the 5000 default for speed; scaling ratio holds
# Slope threshold: fail if empirical log-log slope exceeds this. 1.0 = perfectly
# linear; 1.30 is ~30% super-linear, comfortably above measurement noise but
# well below the 2.39× pre-PR-#12 regime we actually want to catch.
MAX_SLOPE = 1.30


def _synthetic_expression(n_cells: int, n_genes: int, n_tfs: int, seed: int = 0):
    """Make a deterministic log-normalised-looking expression matrix + TFs.

    Values are dense and positive — matches what scanpy produces after
    `normalize_total` + `log1p`. The exact biology is irrelevant; we're
    only measuring how GRN scales in `n_cells`.
    """
    rng = np.random.default_rng(seed)
    X = rng.gamma(shape=2.0, scale=0.5, size=(n_cells, n_genes)).astype(np.float32)
    gene_names = [f"G{i}" for i in range(n_genes)]
    tf_names = gene_names[:n_tfs]
    return X, gene_names, tf_names


def _log_log_slope(xs: list[float], ys: list[float]) -> float:
    """Ordinary-least-squares slope of log(y) vs log(x). Numpy-free."""
    lx = [log(x) for x in xs]
    ly = [log(y) for y in ys]
    mx = sum(lx) / len(lx)
    my = sum(ly) / len(ly)
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    den = sum((a - mx) ** 2 for a in lx)
    return num / den


def test_grn_scaling_is_linear():
    """GRN wall-time must scale at most 30% super-linear in n_cells.

    Regression guard on the PR #12 partition-buffer fix. If future
    changes reintroduce allocation churn or any other super-linear
    term, this test will catch it — the build breaks, the regression
    never reaches main.
    """
    times: list[float] = []
    edges: list[int] = []
    for n_cells in CELL_COUNTS:
        X, gene_names, tf_names = _synthetic_expression(n_cells, N_GENES, N_TFS)

        t0 = time.perf_counter()
        grn = rustscenic.grn.infer(
            (X, gene_names),
            tf_names=tf_names,
            n_estimators=N_ESTIMATORS,
            seed=777,
            verbose=False,
        )
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        edges.append(len(grn))

    slope = _log_log_slope(CELL_COUNTS, times)
    # Print so the actual numbers show up in CI logs — makes regressions
    # debuggable without re-running.
    print(
        f"\nscaling regression points:"
        + "".join(f"\n  n={n:>5}  t={t:.3f}s  edges={e:,}" for n, t, e in zip(CELL_COUNTS, times, edges))
        + f"\n  log-log slope: {slope:.3f}  (threshold: ≤ {MAX_SLOPE})"
    )
    assert slope <= MAX_SLOPE, (
        f"GRN scaling slope {slope:.3f} exceeds {MAX_SLOPE}. "
        f"Super-linearity has come back in. Times: {list(zip(CELL_COUNTS, times))}"
    )
