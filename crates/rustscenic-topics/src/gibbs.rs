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
//! The implementation is single-threaded for correctness — concurrent
//! Gibbs needs sparse-LDA tricks or atomic counts to avoid races, both of
//! which lose accuracy. Atlas-scale users should rely on the Online VB
//! path; collapsed Gibbs is the high-quality K ≥ 30 path for smaller
//! samples (tens of thousands of cells).

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

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
}
