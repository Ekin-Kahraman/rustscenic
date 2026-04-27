//! Collapsed Gibbs sampling for Latent Dirichlet Allocation.
//!
//! This is the Mallet-class topic model — sampling from the true posterior
//! over per-token topic assignments rather than the variational
//! approximation in `lib.rs`. On sparse scATAC at K ≥ 30, collapsed Gibbs
//! reliably wins on topic-coherence (NPMI) and unique-topic count compared
//! to Online VB; the trade-off is convergence time (thousands of iterations
//! instead of tens of passes).
//!
//! Algorithm (Griffiths & Steyvers 2004):
//! - Each token (d, w_i) has a topic assignment z_d_i ∈ [0, K)
//! - Maintain count matrices:
//!   n_dk\[d, k\] = # tokens in doc d assigned to topic k,
//!   n_kw\[k, w\] = # tokens of word w assigned to topic k,
//!   n_k\[k\]     = Σ_w n_kw\[k, w\]
//! - Per-token Gibbs step:
//!   1. subtract token's current contribution from counts
//!   2. compute P(z = k') ∝ (n_dk + α)(n_kw + η) / (n_k + Wη) for all k'
//!   3. sample new z from that distribution
//!   4. add new token contribution to counts
//! - Repeat for `n_iters` sweeps over the corpus
//!
//! Output: posterior point estimates
//! theta\[d, k\] = (n_dk + α) / (Σ_k' n_dk' + Kα)
//!   beta[k, w]  = (n_kw + η) / (n_k + Wη)
//!
//! Two paths:
//! - `fit`: serial, bit-identical under same seed.
//! - `fit_par`: AD-LDA (Newman et al. 2009) — partition documents
//!   across Rayon workers, each thread mutates a per-thread delta on
//!   `n_kw`/`n_k` while sampling, deltas are merged at sweep
//!   boundaries. Quality matches serial at typical thread counts;
//!   wall-clock scales near-linearly. Recommended for K ≥ 30 atlas
//!   runs (50k+ cells).

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// Result of a collapsed-Gibbs LDA fit.
pub struct GibbsResult {
    /// Document-topic distribution θ_dk = (n_dk + α) / (Σ_k' n_dk' + Kα).
    /// Row-major, shape (n_docs × n_topics).
    pub theta: Vec<f32>,
    /// Topic-word distribution β_kw = (n_kw + η) / (n_k + Wη).
    /// Row-major, shape (n_topics × n_words).
    pub beta: Vec<f32>,
    /// Number of Gibbs sweeps actually run.
    pub n_iters_run: usize,
}

/// Fit collapsed-Gibbs LDA to a sparse cells × words matrix.
///
/// Each token contributes one count to the corpus; the sparse triples
/// (row_ptr, col_idx, counts) describe the document-word matrix in CSR
/// form. Counts are taken as integer multiplicities (a count of 3 spawns
/// 3 tokens of that word in that doc).
///
/// Returns posterior point estimates of θ and β after `n_iters` sweeps.
#[allow(clippy::too_many_arguments)]
pub fn fit(
    row_ptr: &[usize],
    col_idx: &[u32],
    counts: &[f32],
    n_words: usize,
    n_topics: usize,
    alpha: f32,
    eta: f32,
    n_iters: usize,
    seed: u64,
) -> GibbsResult {
    let n_docs = row_ptr.len().saturating_sub(1);
    assert!(n_topics > 0, "n_topics must be > 0");
    assert!(n_iters > 0, "n_iters must be > 0");

    // Materialise the per-token assignment array. Each non-zero (d, w_i, c_i)
    // expands to c_i tokens, each with an independent topic assignment.
    // `tokens` stores (doc, word) for each, `z` stores its current topic.
    let total_tokens: usize = counts.iter().map(|&c| c as usize).sum();
    let mut tokens: Vec<(u32, u32)> = Vec::with_capacity(total_tokens);
    for d in 0..n_docs {
        let s = row_ptr[d];
        let e = row_ptr[d + 1];
        for i in s..e {
            let w = col_idx[i];
            let c = counts[i] as usize;
            for _ in 0..c {
                tokens.push((d as u32, w));
            }
        }
    }

    let mut rng = StdRng::seed_from_u64(seed);

    // Initialise topic assignments uniformly at random.
    let mut z: Vec<u32> = (0..tokens.len())
        .map(|_| rng.gen_range(0..n_topics as u32))
        .collect();

    let mut n_dk = vec![0u32; n_docs * n_topics];
    let mut n_kw = vec![0u32; n_topics * n_words];
    let mut n_k = vec![0u32; n_topics];

    for (i, &(d, w)) in tokens.iter().enumerate() {
        let k = z[i] as usize;
        n_dk[d as usize * n_topics + k] += 1;
        n_kw[k * n_words + w as usize] += 1;
        n_k[k] += 1;
    }

    let alpha_f64 = alpha as f64;
    let eta_f64 = eta as f64;
    let w_eta = n_words as f64 * eta_f64;
    let mut p = vec![0.0_f64; n_topics];

    for _iter in 0..n_iters {
        for (i, &(d, w)) in tokens.iter().enumerate() {
            let d_off = d as usize * n_topics;
            let w_us = w as usize;
            let cur = z[i] as usize;

            // Remove current token's contribution
            n_dk[d_off + cur] -= 1;
            n_kw[cur * n_words + w_us] -= 1;
            n_k[cur] -= 1;

            // Compute unnormalised P(z = k') for each k'
            let mut total = 0.0_f64;
            for k in 0..n_topics {
                let nd = n_dk[d_off + k] as f64 + alpha_f64;
                let nw = n_kw[k * n_words + w_us] as f64 + eta_f64;
                let nk = n_k[k] as f64 + w_eta;
                let val = nd * nw / nk;
                p[k] = val;
                total += val;
            }

            // Sample from the discrete distribution
            let r: f64 = rng.gen::<f64>() * total;
            let mut acc = 0.0_f64;
            let mut new_k = n_topics - 1;
            for k in 0..n_topics {
                acc += p[k];
                if r <= acc {
                    new_k = k;
                    break;
                }
            }

            // Add new token contribution
            n_dk[d_off + new_k] += 1;
            n_kw[new_k * n_words + w_us] += 1;
            n_k[new_k] += 1;
            z[i] = new_k as u32;
        }
    }

    // Posterior point estimates of θ and β
    let mut theta = vec![0.0_f32; n_docs * n_topics];
    for d in 0..n_docs {
        let row_sum: f64 = (0..n_topics)
            .map(|k| n_dk[d * n_topics + k] as f64)
            .sum::<f64>()
            + n_topics as f64 * alpha_f64;
        if row_sum <= 0.0 {
            let uniform = 1.0_f32 / n_topics as f32;
            for k in 0..n_topics {
                theta[d * n_topics + k] = uniform;
            }
            continue;
        }
        for k in 0..n_topics {
            theta[d * n_topics + k] =
                ((n_dk[d * n_topics + k] as f64 + alpha_f64) / row_sum) as f32;
        }
    }

    let mut beta = vec![0.0_f32; n_topics * n_words];
    for k in 0..n_topics {
        let row_sum = n_k[k] as f64 + w_eta;
        if row_sum <= 0.0 {
            let uniform = 1.0_f32 / n_words as f32;
            for w in 0..n_words {
                beta[k * n_words + w] = uniform;
            }
            continue;
        }
        for w in 0..n_words {
            beta[k * n_words + w] =
                ((n_kw[k * n_words + w] as f64 + eta_f64) / row_sum) as f32;
        }
    }

    GibbsResult {
        theta,
        beta,
        n_iters_run: n_iters,
    }
}

/// Per-thread state in AD-LDA: token list, current topic assignments,
/// the slice of `n_dk` corresponding to this thread's docs, and
/// pre-allocated reusable buffers (per-sweep delta on n_kw / n_k +
/// the sampling-probabilities scratch).
struct ThreadState {
    docs: Vec<usize>,
    tokens: Vec<(u32, u32)>, // (local_doc_idx, word)
    z: Vec<u32>,
    n_dk: Vec<u32>,    // (docs.len() × n_topics), row-major
    d_kw: Vec<i32>,    // (n_topics × n_words), reused each sweep
    d_k: Vec<i32>,     // n_topics, reused each sweep
    p_scratch: Vec<f64>, // n_topics, reused each token
}

/// Greedy bin-pack docs into `n_threads` partitions balanced by token load.
fn partition_docs_by_load(
    row_ptr: &[usize],
    counts: &[f32],
    n_threads: usize,
) -> Vec<Vec<usize>> {
    let n_docs = row_ptr.len().saturating_sub(1);
    let doc_load: Vec<usize> = (0..n_docs)
        .map(|d| {
            counts[row_ptr[d]..row_ptr[d + 1]]
                .iter()
                .map(|&c| c as usize)
                .sum::<usize>()
                .max(1)
        })
        .collect();
    let mut bins: Vec<(usize, Vec<usize>)> =
        (0..n_threads).map(|_| (0, Vec::new())).collect();
    let mut doc_order: Vec<usize> = (0..n_docs).collect();
    doc_order.sort_unstable_by_key(|&d| std::cmp::Reverse(doc_load[d]));
    for d in doc_order {
        let bi = (0..n_threads).min_by_key(|&t| bins[t].0).unwrap();
        bins[bi].0 += doc_load[d];
        bins[bi].1.push(d);
    }
    for (_, docs) in bins.iter_mut() {
        docs.sort_unstable();
    }
    bins.into_iter().map(|(_, d)| d).collect()
}

/// Build per-thread token lists + initial-uniform topic assignments + the
/// pre-allocated reusable buffers (`d_kw`, `d_k`, `p_scratch`). Returns
/// the populated thread states; `n_dk` is still zero.
#[allow(clippy::too_many_arguments)]
fn init_thread_states(
    partition: Vec<Vec<usize>>,
    row_ptr: &[usize],
    col_idx: &[u32],
    counts: &[f32],
    n_topics: usize,
    n_words: usize,
    seed: u64,
) -> Vec<ThreadState> {
    let mut master_rng = StdRng::seed_from_u64(seed);
    let mut out: Vec<ThreadState> = Vec::with_capacity(partition.len());
    for docs in partition {
        let mut tokens: Vec<(u32, u32)> = Vec::new();
        for (local_d, &gd) in docs.iter().enumerate() {
            let s = row_ptr[gd];
            let e = row_ptr[gd + 1];
            for i in s..e {
                let w = col_idx[i];
                let c = counts[i] as usize;
                for _ in 0..c {
                    tokens.push((local_d as u32, w));
                }
            }
        }
        let z: Vec<u32> = (0..tokens.len())
            .map(|_| master_rng.gen_range(0..n_topics as u32))
            .collect();
        let n_dk = vec![0u32; docs.len() * n_topics];
        let d_kw = vec![0i32; n_topics * n_words];
        let d_k = vec![0i32; n_topics];
        let p_scratch = vec![0.0_f64; n_topics];
        out.push(ThreadState { docs, tokens, z, n_dk, d_kw, d_k, p_scratch });
    }
    out
}

/// Prime each thread's `n_dk` slice and the global `n_kw` / `n_k` from
/// the initial uniform assignments.
fn prime_counts(
    threads: &mut [ThreadState],
    n_topics: usize,
    n_words: usize,
) -> (Vec<u32>, Vec<u32>) {
    let mut n_kw = vec![0u32; n_topics * n_words];
    let mut n_k = vec![0u32; n_topics];
    for ts in threads.iter_mut() {
        for (i, &(local_d, w)) in ts.tokens.iter().enumerate() {
            let k = ts.z[i] as usize;
            ts.n_dk[local_d as usize * n_topics + k] += 1;
            n_kw[k * n_words + w as usize] += 1;
            n_k[k] += 1;
        }
    }
    (n_kw, n_k)
}

/// Hyperparameters held constant across a sweep — bundled to keep
/// `run_thread_sweep` under the clippy `too_many_arguments` limit.
struct SweepParams {
    n_topics: usize,
    n_words: usize,
    alpha_f64: f64,
    eta_f64: f64,
    w_eta: f64,
}

/// Run a full AD-LDA Gibbs sweep on a single thread's tokens against
/// a fixed snapshot plus the thread's own running delta. Mutates
/// `ts.z`, `ts.n_dk`, `ts.d_kw`, `ts.d_k` in place. The deltas remain
/// in the thread state so the caller can merge them after the
/// `par_iter_mut` rejoin.
fn run_thread_sweep(
    ts: &mut ThreadState,
    snap_n_kw: &[u32],
    snap_n_k: &[u32],
    params: &SweepParams,
    rng_seed: u64,
) {
    let n_topics = params.n_topics;
    let n_words = params.n_words;
    let alpha_f64 = params.alpha_f64;
    let eta_f64 = params.eta_f64;
    let w_eta = params.w_eta;

    // Zero out the persistent delta buffers from last sweep
    for v in ts.d_kw.iter_mut() {
        *v = 0;
    }
    for v in ts.d_k.iter_mut() {
        *v = 0;
    }

    let mut rng = StdRng::seed_from_u64(rng_seed);

    for i in 0..ts.tokens.len() {
        let (local_d, w) = ts.tokens[i];
        let local_d_off = local_d as usize * n_topics;
        let w_us = w as usize;
        let cur = ts.z[i] as usize;

        ts.n_dk[local_d_off + cur] -= 1;
        ts.d_kw[cur * n_words + w_us] -= 1;
        ts.d_k[cur] -= 1;

        let mut total = 0.0_f64;
        for k in 0..n_topics {
            let nd = ts.n_dk[local_d_off + k] as f64 + alpha_f64;
            let nw_global = snap_n_kw[k * n_words + w_us] as i32 + ts.d_kw[k * n_words + w_us];
            let nk_global = snap_n_k[k] as i32 + ts.d_k[k];
            let nw = nw_global.max(0) as f64 + eta_f64;
            let nk = nk_global.max(0) as f64 + w_eta;
            let val = nd * nw / nk;
            ts.p_scratch[k] = val;
            total += val;
        }

        let r: f64 = rng.gen::<f64>() * total;
        let mut acc = 0.0_f64;
        let mut new_k = n_topics - 1;
        for k in 0..n_topics {
            acc += ts.p_scratch[k];
            if r <= acc {
                new_k = k;
                break;
            }
        }

        ts.n_dk[local_d_off + new_k] += 1;
        ts.d_kw[new_k * n_words + w_us] += 1;
        ts.d_k[new_k] += 1;
        ts.z[i] = new_k as u32;
    }
}

/// Add per-thread deltas (held in each `ThreadState`) back into the global
/// `n_kw` / `n_k`. Underflows from a single thread can be negative (if a
/// topic emptied within a sweep) but are always corrected by the matching
/// positive delta in another thread — summing across all threads gives a
/// non-negative result.
///
/// Parallelised across rows of `n_kw` (i.e. topics) — each topic row of
/// size `n_words` is independent, so K-way parallelism is safe.
fn merge_deltas(
    n_kw: &mut [u32],
    n_k: &mut [u32],
    threads: &[ThreadState],
    n_topics: usize,
    n_words: usize,
) {
    n_kw
        .par_chunks_mut(n_words)
        .enumerate()
        .for_each(|(k, row)| {
            for w in 0..n_words {
                let mut total: i32 = row[w] as i32;
                for ts in threads {
                    total += ts.d_kw[k * n_words + w];
                }
                row[w] = total.max(0) as u32;
            }
        });
    for k in 0..n_topics {
        let mut total: i32 = n_k[k] as i32;
        for ts in threads {
            total += ts.d_k[k];
        }
        n_k[k] = total.max(0) as u32;
    }
}

/// Reassemble θ from per-thread n_dk slices into a global (n_docs × n_topics)
/// row-major matrix.
fn compute_theta(
    threads: &[ThreadState],
    n_docs: usize,
    n_topics: usize,
    alpha_f64: f64,
) -> Vec<f32> {
    let mut theta = vec![0.0_f32; n_docs * n_topics];
    for ts in threads {
        for (local_d, &gd) in ts.docs.iter().enumerate() {
            let row_sum: f64 = (0..n_topics)
                .map(|k| ts.n_dk[local_d * n_topics + k] as f64)
                .sum::<f64>()
                + n_topics as f64 * alpha_f64;
            if row_sum <= 0.0 {
                let uniform = 1.0_f32 / n_topics as f32;
                for k in 0..n_topics {
                    theta[gd * n_topics + k] = uniform;
                }
                continue;
            }
            for k in 0..n_topics {
                theta[gd * n_topics + k] =
                    ((ts.n_dk[local_d * n_topics + k] as f64 + alpha_f64) / row_sum) as f32;
            }
        }
    }
    theta
}

/// Parallel collapsed-Gibbs LDA via AD-LDA (Newman et al. 2009).
///
/// Documents are partitioned across `n_threads` Rayon workers. Each
/// worker:
/// - Owns a contiguous slice of doc-topic counts `n_dk` (no contention,
///   since docs are disjoint).
/// - Reads `n_kw`/`n_k` against a per-sweep starting snapshot, plus
///   the worker's own thread-local *delta* (Δn_kw, Δn_k). The
///   sampler's view at any moment is `snapshot + own_delta`.
/// - At sweep end, all worker deltas are summed back into the global
///   `n_kw` and `n_k` for the next sweep.
///
/// This is bit-identical to `fit` when `n_threads == 1`. For
/// `n_threads > 1` it diverges very slightly because cross-thread
/// updates only become visible at sweep boundaries — but Newman et
/// al. 2009 §4 show the perplexity gap is well within sampling
/// variance for typical T (4–32 threads). The trade-off is
/// `O(n_threads × n_topics × n_words × 4 bytes)` memory for the
/// thread-local deltas (e.g. 30 × 100k × 4B × 8 threads = ~96 MB).
#[allow(clippy::too_many_arguments)]
pub fn fit_par(
    row_ptr: &[usize],
    col_idx: &[u32],
    counts: &[f32],
    n_words: usize,
    n_topics: usize,
    alpha: f32,
    eta: f32,
    n_iters: usize,
    seed: u64,
    n_threads: usize,
) -> GibbsResult {
    if n_threads <= 1 {
        return fit(
            row_ptr, col_idx, counts, n_words, n_topics, alpha, eta, n_iters, seed,
        );
    }
    let n_docs = row_ptr.len().saturating_sub(1);
    assert!(n_topics > 0, "n_topics must be > 0");
    assert!(n_iters > 0, "n_iters must be > 0");

    let partition = partition_docs_by_load(row_ptr, counts, n_threads);
    let mut threads = init_thread_states(
        partition, row_ptr, col_idx, counts, n_topics, n_words, seed,
    );
    let (mut n_kw, mut n_k) = prime_counts(&mut threads, n_topics, n_words);

    let alpha_f64 = alpha as f64;
    let eta_f64 = eta as f64;
    let w_eta = n_words as f64 * eta_f64;
    let params = SweepParams {
        n_topics,
        n_words,
        alpha_f64,
        eta_f64,
        w_eta,
    };

    let mut snap_n_kw = vec![0u32; n_topics * n_words];
    let mut snap_n_k = vec![0u32; n_topics];

    for iter in 0..n_iters {
        snap_n_kw.copy_from_slice(&n_kw);
        snap_n_k.copy_from_slice(&n_k);
        threads
            .par_iter_mut()
            .enumerate()
            .for_each(|(t_idx, ts)| {
                let rng_seed = seed
                    .wrapping_add(iter as u64)
                    .wrapping_add(t_idx as u64 * 0x9E3779B9);
                run_thread_sweep(ts, &snap_n_kw, &snap_n_k, &params, rng_seed);
            });
        merge_deltas(&mut n_kw, &mut n_k, &threads, n_topics, n_words);
    }

    let theta = compute_theta(&threads, n_docs, n_topics, alpha_f64);

    let mut beta = vec![0.0_f32; n_topics * n_words];
    for k in 0..n_topics {
        let row_sum = n_k[k] as f64 + w_eta;
        if row_sum <= 0.0 {
            let uniform = 1.0_f32 / n_words as f32;
            for w in 0..n_words {
                beta[k * n_words + w] = uniform;
            }
            continue;
        }
        for w in 0..n_words {
            beta[k * n_words + w] =
                ((n_kw[k * n_words + w] as f64 + eta_f64) / row_sum) as f32;
        }
    }

    GibbsResult {
        theta,
        beta,
        n_iters_run: n_iters,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_two_topic_corpus() -> (Vec<usize>, Vec<u32>, Vec<f32>) {
        // 60 docs, 20 words. Words 0–9 belong to topic A, 10–19 to topic B.
        // Half the docs only sample topic-A words, half only topic-B words.
        let n_docs = 60;
        let mut row_ptr = vec![0usize];
        let mut col_idx = Vec::new();
        let mut counts = Vec::new();
        for d in 0..n_docs {
            let words: Vec<u32> = if d < n_docs / 2 {
                (0..10).collect()
            } else {
                (10..20).collect()
            };
            for w in &words {
                col_idx.push(*w);
                counts.push(1.0);
            }
            row_ptr.push(col_idx.len());
        }
        (row_ptr, col_idx, counts)
    }

    #[test]
    fn gibbs_recovers_two_planted_topics() {
        let (row_ptr, col_idx, counts) = synthetic_two_topic_corpus();
        let r = fit(&row_ptr, &col_idx, &counts, 20, 2, 0.1, 0.01, 200, 42);
        // theta should split docs cleanly: first 30 high on one topic,
        // next 30 high on the other.
        let n_topics = 2;
        let mut topic_per_doc = Vec::new();
        for d in 0..60 {
            let row = &r.theta[d * n_topics..(d + 1) * n_topics];
            topic_per_doc.push(if row[0] > row[1] { 0 } else { 1 });
        }
        // First half should all share one topic, second half the other.
        let first_half: u32 = topic_per_doc[0..30].iter().filter(|&&t| t == topic_per_doc[0]).count() as u32;
        let second_half_other: u32 = topic_per_doc[30..]
            .iter()
            .filter(|&&t| t != topic_per_doc[0])
            .count() as u32;
        assert!(first_half >= 28, "first half drift: {first_half}/30");
        assert!(second_half_other >= 28, "second half drift: {second_half_other}/30");
    }

    #[test]
    fn gibbs_deterministic_under_same_seed() {
        let (row_ptr, col_idx, counts) = synthetic_two_topic_corpus();
        let a = fit(&row_ptr, &col_idx, &counts, 20, 2, 0.1, 0.01, 30, 7);
        let b = fit(&row_ptr, &col_idx, &counts, 20, 2, 0.1, 0.01, 30, 7);
        assert_eq!(a.theta, b.theta);
        assert_eq!(a.beta, b.beta);
    }

    #[test]
    fn gibbs_handles_empty_doc() {
        // Doc with zero tokens — shouldn't panic, theta should be uniform-ish.
        let row_ptr = vec![0, 0, 5];
        let col_idx = vec![0, 1, 2, 3, 4];
        let counts = vec![1.0; 5];
        let r = fit(&row_ptr, &col_idx, &counts, 5, 2, 0.1, 0.01, 30, 0);
        let theta_doc0 = &r.theta[0..2];
        // Empty doc: theta should be exactly α/(Kα) = 0.5 uniform
        assert!((theta_doc0[0] - 0.5).abs() < 0.01);
        assert!((theta_doc0[1] - 0.5).abs() < 0.01);
    }

    #[test]
    fn gibbs_par_n_threads_1_matches_serial() {
        // n_threads=1 must dispatch to the serial fit and be bit-identical.
        let (row_ptr, col_idx, counts) = synthetic_two_topic_corpus();
        let serial = fit(&row_ptr, &col_idx, &counts, 20, 2, 0.1, 0.01, 30, 11);
        let par = fit_par(&row_ptr, &col_idx, &counts, 20, 2, 0.1, 0.01, 30, 11, 1);
        assert_eq!(serial.theta, par.theta);
        assert_eq!(serial.beta, par.beta);
    }

    #[test]
    fn gibbs_par_recovers_two_planted_topics() {
        // Same planted-recovery test as the serial path, with 4 threads.
        let (row_ptr, col_idx, counts) = synthetic_two_topic_corpus();
        let r = fit_par(&row_ptr, &col_idx, &counts, 20, 2, 0.1, 0.01, 200, 42, 4);
        let n_topics = 2;
        let mut topic_per_doc = Vec::new();
        for d in 0..60 {
            let row = &r.theta[d * n_topics..(d + 1) * n_topics];
            topic_per_doc.push(if row[0] > row[1] { 0 } else { 1 });
        }
        let first_half: u32 = topic_per_doc[0..30]
            .iter()
            .filter(|&&t| t == topic_per_doc[0])
            .count() as u32;
        let second_half_other: u32 = topic_per_doc[30..]
            .iter()
            .filter(|&&t| t != topic_per_doc[0])
            .count() as u32;
        // AD-LDA is approximate; allow a few more drift docs than serial.
        assert!(first_half >= 25, "first half drift: {first_half}/30");
        assert!(second_half_other >= 25, "second half drift: {second_half_other}/30");
    }

    #[test]
    fn gibbs_par_deterministic_under_same_seed_and_threads() {
        // Per-thread RNGs are seeded from (seed, iter, thread_idx) so
        // results are reproducible at fixed n_threads.
        let (row_ptr, col_idx, counts) = synthetic_two_topic_corpus();
        let a = fit_par(&row_ptr, &col_idx, &counts, 20, 3, 0.1, 0.01, 30, 7, 4);
        let b = fit_par(&row_ptr, &col_idx, &counts, 20, 3, 0.1, 0.01, 30, 7, 4);
        assert_eq!(a.theta, b.theta);
        assert_eq!(a.beta, b.beta);
    }
}
