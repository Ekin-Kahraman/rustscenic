"""Head-to-head: rustscenic.topics.fit vs gensim.LdaModel.

Reproducible topic-modelling benchmark backing the K=10 / K=30 numbers
in `docs/bench-vs-references.md`.

Same cells × peaks matrix from real PBMC 3k Multiome ATAC, same seed,
2 passes. Both tools run on the same hardware.

Setup:
  pip install gensim
  # rustscenic already installed
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent.parent / "real_multiome"


def main() -> int:
    import warnings

    import rustscenic.preproc
    import rustscenic.topics

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        atac = rustscenic.preproc.fragments_to_matrix(
            str(HERE / "fragments.tsv.gz"),
            str(HERE / "peaks.bed"),
        )
        # Subset to top 3k cells by fragment count
        fpc = atac.obs["fragments_per_cell"].values
        atac = atac[np.argsort(fpc)[::-1][:3000]].copy()

    print(f"matrix: {atac.shape}, nnz={atac.X.nnz:,}\n")

    from gensim.models import LdaModel

    # gensim wants list of (token_id, count) per document
    corpus = []
    for i in range(atac.n_obs):
        row = atac.X.getrow(i)
        corpus.append(list(zip(row.indices.tolist(), row.data.astype(int).tolist())))
    id2word = {i: f"p{i}" for i in range(atac.n_vars)}

    print(f"{'K':>4s} | {'rustscenic':>12s} | {'gensim':>10s} | ratio")
    print("-" * 45)
    for K in (10, 30):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.monotonic()
            r = rustscenic.topics.fit(atac, n_topics=K, n_passes=2, seed=42)
            rs_t = time.monotonic() - t0

        t0 = time.monotonic()
        gs = LdaModel(
            corpus=corpus, num_topics=K, id2word=id2word,
            passes=2, random_state=42, alpha="auto", eta="auto",
        )
        gs_t = time.monotonic() - t0

        ratio = gs_t / rs_t
        winner = "gensim" if gs_t < rs_t else "rustscenic"
        print(f"{K:>4d} | {rs_t:>10.1f}s | {gs_t:>8.1f}s | gensim/rs={ratio:.2f}× ({winner} wins)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
