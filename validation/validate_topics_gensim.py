"""Validate rustscenic.topics against gensim's online VB LDA on binarized PBMC-3k.

gensim uses Hoffman-Blei-Bach 2010 online VB LDA — same algorithm as ours.
Validation metric: topic assignment ARI (adjusted Rand index) between the two
implementations when both given the same seed + hyperparameters.

ARI > 0.5 = strong agreement (topics are permutation-free; we use ARI not raw
label match). ARI > 0.7 = essentially identical clustering. ARI of 0 = random.
"""
import time
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from sklearn.metrics import adjusted_rand_score

import rustscenic.topics

SEED = 42
N_TOPICS = 20
N_PASSES = 10

adata = ad.read_h5ad("/Users/ekin/rustscenic/validation/reference/data/pbmc3k.h5ad")
X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
X_bin = (X > 1).astype(np.float32)  # binarize
print(f"binarized matrix: {X_bin.shape}  nonzero {(X_bin > 0).sum()}  density {(X_bin > 0).mean():.3f}")

# --- rustscenic.topics ---
print("\n--- rustscenic.topics ---")
t0 = time.monotonic()
ours = rustscenic.topics.fit(
    (sp.csr_matrix(X_bin), list(adata.obs_names), list(adata.var_names)),
    n_topics=N_TOPICS, n_passes=N_PASSES, batch_size=256, seed=SEED,
    alpha=1.0/N_TOPICS, eta=1.0/N_TOPICS,
)
ours_wall = time.monotonic() - t0
print(f"  wall: {ours_wall:.1f}s")
ours_assign = ours.cell_assignment().values

# --- gensim LDA ---
print("\n--- gensim LDA (reference) ---")
# Convert binarized matrix to gensim's corpus format: list of [(word_id, count), ...] per doc
gene_names = list(adata.var_names)
corpus = []
for i in range(X_bin.shape[0]):
    row = X_bin[i]
    nonzero = np.nonzero(row)[0]
    corpus.append([(int(w), float(row[w])) for w in nonzero])

t0 = time.monotonic()
lda = LdaModel(
    corpus=corpus,
    id2word={i: gene_names[i] for i in range(len(gene_names))},
    num_topics=N_TOPICS,
    passes=N_PASSES,
    alpha=[1.0/N_TOPICS] * N_TOPICS,
    eta=[1.0/N_TOPICS] * len(gene_names),
    random_state=SEED,
    chunksize=256,
)
gensim_wall = time.monotonic() - t0
print(f"  wall: {gensim_wall:.1f}s")
print(f"  speedup (ours vs gensim): {gensim_wall/ours_wall:.2f}x")

# gensim topic assignment per doc
gensim_assign = np.zeros(len(corpus), dtype=int)
for i, doc in enumerate(corpus):
    top = lda.get_document_topics(doc, minimum_probability=0)
    # top: list of (topic_id, prob)
    gensim_assign[i] = max(top, key=lambda x: x[1])[0]

# --- compare ---
ari = adjusted_rand_score(ours_assign.astype(int) if isinstance(ours_assign[0], (np.integer, int)) else
                          [int(x.replace("Topic_", "")) for x in ours_assign],
                          gensim_assign)
print(f"\nARI between rustscenic vs gensim topic assignments: {ari:.4f}")
print(f"  ours n-unique topics: {len(set(ours_assign))}")
print(f"  gensim n-unique topics: {len(set(gensim_assign))}")

# Count consistency by cell
from collections import Counter
print(f"  ours distribution: {Counter(ours_assign).most_common(5)}")
print(f"  gensim distribution: {Counter(gensim_assign.tolist()).most_common(5)}")
