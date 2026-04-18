"""Validate rustscenic.topics on scATAC-like synthetic data.

Generates a realistic sparse count matrix:
 - 2000 cells split evenly across 10 planted topics
 - 20000 peaks; each topic has ~400 peaks uniquely associated + shared background
 - 5% sparsity (matches real binarized scATAC)

Compares rustscenic.topics against gensim LdaModel on the same data.
Metric: ARI of topic assignments vs planted labels.
"""
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from gensim.models import LdaModel
from sklearn.metrics import adjusted_rand_score

import rustscenic.topics

N_CELLS = 2000
N_PEAKS = 20000
N_TOPICS_TRUE = 10
SEED = 42
N_PASSES = 15

# --- planted generator ---
rng = np.random.default_rng(SEED)
cells_per_topic = N_CELLS // N_TOPICS_TRUE
planted = np.repeat(np.arange(N_TOPICS_TRUE), cells_per_topic)
# assign 400 topic-specific peaks per topic (no overlap), rest are background
peaks_per_topic = N_PEAKS // (N_TOPICS_TRUE * 5)  # 400 peaks per topic → 4000 topic-specific
topic_peaks = [
    np.arange(k * peaks_per_topic, (k + 1) * peaks_per_topic)
    for k in range(N_TOPICS_TRUE)
]
background_peaks = np.arange(N_TOPICS_TRUE * peaks_per_topic, N_PEAKS)

# build sparse binarized matrix
rows, cols = [], []
for c in range(N_CELLS):
    topic = planted[c]
    # 80 peaks from this topic's set (strong signal)
    on_topic = rng.choice(topic_peaks[topic], size=80, replace=False)
    # 20 background peaks (noise)
    on_bg = rng.choice(background_peaks, size=20, replace=False)
    # small cross-contamination: 5 peaks from a different random topic
    other = rng.integers(0, N_TOPICS_TRUE)
    if other == topic: other = (other + 1) % N_TOPICS_TRUE
    on_cross = rng.choice(topic_peaks[other], size=5, replace=False)
    for p in np.concatenate([on_topic, on_bg, on_cross]):
        rows.append(c); cols.append(int(p))

data = np.ones(len(rows), dtype=np.float32)
X_bin = sp.csr_matrix((data, (rows, cols)), shape=(N_CELLS, N_PEAKS)).astype(np.float32)
print(f"synthetic scATAC-like: {X_bin.shape}  nonzero {X_bin.nnz}  density {X_bin.nnz/(N_CELLS*N_PEAKS):.4f}")
cell_names = [f"c{i}" for i in range(N_CELLS)]
peak_names = [f"p{i}" for i in range(N_PEAKS)]

# --- rustscenic.topics ---
print("\n--- rustscenic.topics ---")
t0 = time.monotonic()
ours = rustscenic.topics.fit(
    (X_bin, cell_names, peak_names),
    n_topics=N_TOPICS_TRUE, n_passes=N_PASSES, batch_size=256, seed=SEED,
    alpha=1.0/N_TOPICS_TRUE, eta=1.0/N_TOPICS_TRUE,
)
ours_wall = time.monotonic() - t0
ours_assign = np.asarray([int(s.replace("Topic_", "")) for s in ours.cell_assignment().values])
print(f"  wall: {ours_wall:.1f}s  unique topics assigned: {len(set(ours_assign))}")
ours_ari = adjusted_rand_score(planted, ours_assign)
print(f"  ARI vs planted: {ours_ari:.4f}")

# --- gensim LDA ---
print("\n--- gensim LDA ---")
corpus = []
for i in range(N_CELLS):
    r = X_bin.getrow(i).toarray()[0]
    nz = np.nonzero(r)[0]
    corpus.append([(int(w), float(r[w])) for w in nz])
t0 = time.monotonic()
lda = LdaModel(
    corpus=corpus, id2word={i: peak_names[i] for i in range(N_PEAKS)},
    num_topics=N_TOPICS_TRUE, passes=N_PASSES,
    alpha=[1.0/N_TOPICS_TRUE]*N_TOPICS_TRUE,
    eta=[1.0/N_TOPICS_TRUE]*N_PEAKS,
    random_state=SEED, chunksize=256,
)
gensim_wall = time.monotonic() - t0
gensim_assign = np.zeros(N_CELLS, dtype=int)
for i, doc in enumerate(corpus):
    top = lda.get_document_topics(doc, minimum_probability=0)
    gensim_assign[i] = max(top, key=lambda x: x[1])[0]
print(f"  wall: {gensim_wall:.1f}s  unique topics assigned: {len(set(gensim_assign))}")
gensim_ari = adjusted_rand_score(planted, gensim_assign)
print(f"  ARI vs planted: {gensim_ari:.4f}")

print(f"\nSummary:")
print(f"  rustscenic ARI {ours_ari:.3f}  wall {ours_wall:.1f}s")
print(f"  gensim     ARI {gensim_ari:.3f}  wall {gensim_wall:.1f}s")
print(f"  speedup ours vs gensim: {gensim_wall/ours_wall:.1f}x")
# cross-tool ARI
cross = adjusted_rand_score(ours_assign, gensim_assign)
print(f"  cross-tool ARI (ours vs gensim): {cross:.3f}")
