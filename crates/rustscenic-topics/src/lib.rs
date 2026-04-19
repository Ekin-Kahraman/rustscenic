// numerical inner loops read from multiple arrays by index — allow to keep readability.
#![allow(clippy::needless_range_loop)]

//! Online variational Bayes Latent Dirichlet Allocation (pycisTopic replacement).
//!
//! Implements Hoffman-Blei-Bach 2010 online VB LDA for scATAC peak-topic
//! modeling. Input is a sparse (n_cells × n_peaks) binarized accessibility
//! matrix; output is (cell × topic) + (topic × peak) probability matrices.
//!
//! Algorithmic advantages over pycisTopic's default Mallet Gibbs sampler:
//!   - Online VB converges in tens of passes vs Gibbs's thousands of iterations.
//!   - Streaming minibatch update → constant memory per cell.
//!   - Deterministic given seed (Gibbs is stochastic; topic labels permute).
//!
//! Not bit-identical to Mallet Gibbs. Topic-assignment ARI against pycisTopic
//! is the validation target (published benchmarks: ≥0.85 at convergence).
//!
//! Input format: caller passes the sparse matrix as a CSR-style triple of
//! (row_ptr, col_idx, counts). This lets callers feed both raw counts (scRNA
//! LDA) and binarized accessibility (scATAC) without modification.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use statrs::function::gamma::{digamma, ln_gamma};

/// Online VB LDA.
///
/// - `row_ptr`: length `n_docs + 1`; `row_ptr[d]..row_ptr[d+1]` indexes into `col_idx`/`counts`
/// - `col_idx`: flat array of word (peak) indices
/// - `counts`: same length as `col_idx`; per-document-word counts
/// - `n_words`: total vocabulary (n_peaks for scATAC)
/// - `n_topics`: K, the number of latent topics
/// - `alpha`: Dirichlet prior on doc-topic, per-component (default 1/K)
/// - `eta`:   Dirichlet prior on topic-word, per-component (default 1/K)
/// - `tau0`:  learning-rate delay (default 64)
/// - `kappa`: learning-rate decay (0.5 < kappa ≤ 1; default 0.7)
/// - `batch_size`: docs per update (default max(1, n_docs/100))
/// - `n_passes`: full epochs over the corpus (default 10)
/// - `seed`: RNG seed
///
/// Returns `(cell_topic, topic_word)`:
///   `cell_topic`: `[n_docs * n_topics]` row-major, each row sums to 1
///   `topic_word`: `[n_topics * n_words]` row-major, each row sums to 1
pub struct LdaResult {
    pub cell_topic: Vec<f32>,
    pub topic_word: Vec<f32>,
    pub n_docs: usize,
    pub n_words: usize,
    pub n_topics: usize,
    pub n_iters: usize,
}

#[allow(clippy::too_many_arguments)]
pub fn online_vb_lda(
    row_ptr: &[usize],
    col_idx: &[u32],
    counts: &[f32],
    n_words: usize,
    n_topics: usize,
    alpha: f32,
    eta: f32,
    tau0: f32,
    kappa: f32,
    batch_size: usize,
    n_passes: usize,
    seed: u64,
) -> LdaResult {
    assert!(row_ptr.len() >= 2);
    assert_eq!(col_idx.len(), counts.len());
    let n_docs = row_ptr.len() - 1;
    assert!(n_topics > 0);

    // Reject NaN/Inf in counts — the VB variational updates use logs and
    // digammas that propagate non-finite values silently into garbage topic
    // assignments. Fail fast with a clear message.
    if counts.iter().any(|v| !v.is_finite()) {
        panic!(
            "input counts contain NaN or Inf values — LDA is undefined on \
            non-finite counts. Cast your (cell, peak) matrix to integer counts \
            or binarize before calling topics.fit()."
        );
    }
    let bs = batch_size.max(1).min(n_docs);

    // lambda[k * W + w] = topic-word variational parameters (unnormalized Dirichlet)
    // Initialize with gamma(100, 1/100) ≈ mean 1, std 0.1. Without enough variance
    // across topics, variational updates stabilize at trivial single-topic solutions
    // (topic collapse). This init matches gensim's LdaModel default.
    let mut rng = StdRng::seed_from_u64(seed);
    let mut lambda = vec![0.0_f64; n_topics * n_words];
    for v in lambda.iter_mut() {
        // gamma(shape=100, scale=1/100) via Marsaglia-Tsang method (statrs has one,
        // but we inline a simple transformed normal approximation — sufficient for init)
        // Mean 1.0, std 0.1.
        let u1: f64 = rng.gen();
        let u2: f64 = rng.gen();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        *v = (1.0 + 0.1 * z).max(0.01);
    }

    // Precomputed E[log beta_kw] — recomputed each batch
    let mut elog_beta = vec![0.0_f64; n_topics * n_words];

    let mut t = 0usize;
    let total_batches = n_passes * n_docs.div_ceil(bs);
    let big_d = n_docs as f64;

    // Doc-topic posterior gamma_d is allocated per batch (don't keep between epochs to save RAM)
    for _pass in 0..n_passes {
        // Shuffle doc order once per pass
        let mut doc_order: Vec<usize> = (0..n_docs).collect();
        for i in (1..n_docs).rev() {
            let j = rng.gen_range(0..=i);
            doc_order.swap(i, j);
        }

        for batch in doc_order.chunks(bs) {
            // (re)compute E[log beta] from current lambda
            update_elog_beta(&lambda, &mut elog_beta, n_topics, n_words, eta as f64);

            // E-step: fit each doc's gamma (local variational parameters)
            // Accumulate sufficient stats s[k,w] = sum_d n_dw * phi_dwk
            let mut sstats = vec![0.0_f64; n_topics * n_words];

            let batch_sstats: Vec<Vec<f64>> = batch
                .par_iter()
                .map(|&d| {
                    let start = row_ptr[d];
                    let end = row_ptr[d + 1];
                    if end == start {
                        return vec![0.0_f64; n_topics * n_words];
                    }
                    let doc_cols = &col_idx[start..end];
                    let doc_counts = &counts[start..end];

                    // Init gamma_d with asymmetric noise to break topic-label symmetry.
                    // A uniform init + deterministic E-step converges to trivial symmetric
                    // posteriors on sparse data (topic collapse). Per-doc random seeds
                    // derived from doc index + global seed for reproducibility.
                    let mut doc_rng = StdRng::seed_from_u64(seed.wrapping_add(d as u64));
                    let mut gamma_d = vec![0.0_f64; n_topics];
                    for g in gamma_d.iter_mut() {
                        let u1: f64 = doc_rng.gen();
                        let u2: f64 = doc_rng.gen();
                        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                        *g = (alpha as f64 + 1.0 + 0.1 * z).max(0.01);
                    }
                    let mut elog_theta = vec![0.0_f64; n_topics];

                    for _iter in 0..50 {
                        // E[log theta_dk]
                        update_elog_theta(&gamma_d, &mut elog_theta);
                        // phi_dwk ∝ exp(E[log theta_dk] + E[log beta_kw])
                        // accumulate into new gamma without storing full phi
                        let mut gamma_new = vec![alpha as f64; n_topics];
                        for (i, &w) in doc_cols.iter().enumerate() {
                            let w = w as usize;
                            let nw = doc_counts[i] as f64;
                            // normalization
                            let mut max_log = f64::NEG_INFINITY;
                            let mut buf = vec![0.0_f64; n_topics];
                            for k in 0..n_topics {
                                let lp = elog_theta[k] + elog_beta[k * n_words + w];
                                buf[k] = lp;
                                if lp > max_log { max_log = lp; }
                            }
                            let mut s = 0.0_f64;
                            for k in 0..n_topics {
                                buf[k] = (buf[k] - max_log).exp();
                                s += buf[k];
                            }
                            if s > 0.0 {
                                let inv = 1.0 / s;
                                for k in 0..n_topics {
                                    buf[k] *= inv;
                                    gamma_new[k] += nw * buf[k];
                                }
                            }
                        }
                        let delta = gamma_new
                            .iter()
                            .zip(gamma_d.iter())
                            .map(|(a, b)| (a - b).abs())
                            .sum::<f64>() / n_topics as f64;
                        gamma_d = gamma_new;
                        if delta < 1e-3 {
                            break;
                        }
                    }

                    // Final phi → local sufficient stats s_local[k,w] += n_dw * phi_dwk
                    update_elog_theta(&gamma_d, &mut elog_theta);
                    let mut local_ss = vec![0.0_f64; n_topics * n_words];
                    for (i, &w) in doc_cols.iter().enumerate() {
                        let w = w as usize;
                        let nw = doc_counts[i] as f64;
                        let mut max_log = f64::NEG_INFINITY;
                        let mut buf = vec![0.0_f64; n_topics];
                        for k in 0..n_topics {
                            let lp = elog_theta[k] + elog_beta[k * n_words + w];
                            buf[k] = lp;
                            if lp > max_log { max_log = lp; }
                        }
                        let mut s = 0.0_f64;
                        for k in 0..n_topics {
                            buf[k] = (buf[k] - max_log).exp();
                            s += buf[k];
                        }
                        if s > 0.0 {
                            let inv = 1.0 / s;
                            for k in 0..n_topics {
                                buf[k] *= inv;
                                local_ss[k * n_words + w] += nw * buf[k];
                            }
                        }
                    }
                    local_ss
                })
                .collect();

            for local in &batch_sstats {
                for (a, b) in sstats.iter_mut().zip(local.iter()) {
                    *a += *b;
                }
            }

            // Online update rule (Hoffman 2010 eq 5):
            //   lambda_kw <- (1-rho) * lambda_kw + rho * (eta + N/|batch| * s_kw)
            let rho = (tau0 + t as f32).powf(-kappa) as f64;
            let scale = big_d / batch.len() as f64;
            for (lw, ss) in lambda.iter_mut().zip(sstats.iter()) {
                *lw = (1.0 - rho) * (*lw) + rho * (eta as f64 + scale * *ss);
            }
            t += 1;
        }
    }

    // Final compute: cell-topic theta from one more E-step over all docs, topic-word from lambda
    update_elog_beta(&lambda, &mut elog_beta, n_topics, n_words, eta as f64);

    let cell_topic: Vec<f32> = (0..n_docs)
        .into_par_iter()
        .flat_map(|d| {
            let start = row_ptr[d];
            let end = row_ptr[d + 1];
            if end == start {
                let uniform = 1.0_f32 / n_topics as f32;
                return vec![uniform; n_topics];
            }
            let doc_cols = &col_idx[start..end];
            let doc_counts = &counts[start..end];
            // Same asymmetric init for final E-step pass
            let mut doc_rng = StdRng::seed_from_u64(seed.wrapping_add(d as u64).wrapping_add(1_000_000));
            let mut gamma_d = vec![0.0_f64; n_topics];
            for g in gamma_d.iter_mut() {
                let u1: f64 = doc_rng.gen();
                let u2: f64 = doc_rng.gen();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                *g = (alpha as f64 + 1.0 + 0.1 * z).max(0.01);
            }
            let mut elog_theta = vec![0.0_f64; n_topics];
            for _iter in 0..100 {
                update_elog_theta(&gamma_d, &mut elog_theta);
                let mut gamma_new = vec![alpha as f64; n_topics];
                for (i, &w) in doc_cols.iter().enumerate() {
                    let w = w as usize;
                    let nw = doc_counts[i] as f64;
                    let mut max_log = f64::NEG_INFINITY;
                    let mut buf = vec![0.0_f64; n_topics];
                    for k in 0..n_topics {
                        let lp = elog_theta[k] + elog_beta[k * n_words + w];
                        buf[k] = lp;
                        if lp > max_log { max_log = lp; }
                    }
                    let mut s = 0.0_f64;
                    for k in 0..n_topics {
                        buf[k] = (buf[k] - max_log).exp();
                        s += buf[k];
                    }
                    if s > 0.0 {
                        let inv = 1.0 / s;
                        for k in 0..n_topics {
                            buf[k] *= inv;
                            gamma_new[k] += nw * buf[k];
                        }
                    }
                }
                let delta = gamma_new.iter().zip(gamma_d.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f64>() / n_topics as f64;
                gamma_d = gamma_new;
                if delta < 1e-5 { break; }
            }
            // normalize gamma_d
            let sum = gamma_d.iter().sum::<f64>();
            gamma_d.iter().map(|g| (g / sum) as f32).collect::<Vec<f32>>()
        })
        .collect();

    // topic_word: row-normalize lambda (each topic row sums to 1)
    let mut topic_word = vec![0.0_f32; n_topics * n_words];
    for k in 0..n_topics {
        let row_start = k * n_words;
        let sum = lambda[row_start..row_start + n_words].iter().sum::<f64>();
        if sum > 0.0 {
            for w in 0..n_words {
                topic_word[row_start + w] = (lambda[row_start + w] / sum) as f32;
            }
        }
    }

    LdaResult {
        cell_topic,
        topic_word,
        n_docs,
        n_words,
        n_topics,
        n_iters: total_batches,
    }
}

fn update_elog_theta(gamma: &[f64], out: &mut [f64]) {
    let sum: f64 = gamma.iter().sum();
    let d_sum = digamma(sum);
    for (i, g) in gamma.iter().enumerate() {
        out[i] = digamma(*g) - d_sum;
    }
}

fn update_elog_beta(lambda: &[f64], out: &mut [f64], n_topics: usize, n_words: usize, _eta: f64) {
    for k in 0..n_topics {
        let mut sum = 0.0_f64;
        for w in 0..n_words {
            sum += lambda[k * n_words + w];
        }
        let d_sum = digamma(sum);
        for w in 0..n_words {
            out[k * n_words + w] = digamma(lambda[k * n_words + w]) - d_sum;
        }
    }
}

/// Topic coherence NPMI — higher = better. Useful as an intrinsic quality metric
/// without needing a gold-standard label set. Not used in the fit; caller may
/// invoke this post-hoc on the topic_word matrix.
pub fn topic_coherence_npmi(
    topic_word: &[f32],
    n_topics: usize,
    n_words: usize,
    top_n: usize,
    row_ptr: &[usize],
    col_idx: &[u32],
) -> Vec<f32> {
    let n_docs = row_ptr.len() - 1;
    let mut out = vec![0.0_f32; n_topics];

    // Precompute doc-containment per word (is word w in doc d?)
    let mut in_doc: Vec<Vec<u32>> = (0..n_words).map(|_| Vec::new()).collect();
    for d in 0..n_docs {
        for &w in &col_idx[row_ptr[d]..row_ptr[d + 1]] {
            in_doc[w as usize].push(d as u32);
        }
    }

    for k in 0..n_topics {
        let row = &topic_word[k * n_words..(k + 1) * n_words];
        let mut topn: Vec<(f32, usize)> = row.iter().enumerate().map(|(w, &p)| (p, w)).collect();
        topn.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        topn.truncate(top_n);
        let words: Vec<usize> = topn.iter().map(|(_, w)| *w).collect();

        let mut npmi_sum = 0.0_f32;
        let mut pair_count = 0_u32;
        for i in 0..words.len() {
            for j in (i + 1)..words.len() {
                let wi = words[i];
                let wj = words[j];
                let di: std::collections::HashSet<u32> = in_doc[wi].iter().copied().collect();
                let dj = &in_doc[wj];
                let joint = dj.iter().filter(|d| di.contains(d)).count();
                let pi = in_doc[wi].len() as f64 / n_docs as f64;
                let pj = dj.len() as f64 / n_docs as f64;
                let pij = joint as f64 / n_docs as f64;
                if pij > 0.0 && pi > 0.0 && pj > 0.0 {
                    let npmi = (pij / (pi * pj)).ln() / (-pij.ln());
                    npmi_sum += npmi as f32;
                }
                pair_count += 1;
            }
        }
        if pair_count > 0 {
            out[k] = npmi_sum / pair_count as f32;
        }
    }
    out
}

// Safety: statrs' ln_gamma is private in some versions, just silence the unused-import check.
#[allow(dead_code)]
fn _ensure_ln_gamma_used(x: f64) -> f64 { ln_gamma(x) }

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn recovers_planted_topics_on_synthetic() {
        // 2 true topics, 4 words each (no overlap), 200 docs each (400 total)
        // Topic A: words {0, 1, 2, 3} each with prob 0.25
        // Topic B: words {4, 5, 6, 7} each with prob 0.25
        let n_words = 8;
        let n_docs = 400;
        let mut rng = StdRng::seed_from_u64(0);
        let mut row_ptr = vec![0_usize];
        let mut col_idx: Vec<u32> = Vec::new();
        let mut counts: Vec<f32> = Vec::new();
        for d in 0..n_docs {
            let topic = if d < 200 { 0 } else { 1 };
            let base = topic * 4;
            // each doc has the 4 words of its topic
            for i in 0..4 {
                col_idx.push((base + i) as u32);
                counts.push((rng.gen::<f32>() * 5.0 + 1.0).round().max(1.0));
            }
            row_ptr.push(col_idx.len());
        }
        let res = online_vb_lda(
            &row_ptr, &col_idx, &counts,
            n_words, 2,        // K=2
            0.5, 0.5,          // alpha, eta
            64.0, 0.7,         // tau0, kappa
            50, 30,            // batch, n_passes
            42,
        );
        // Most of each half's cell_topic mass should concentrate on one topic
        let mut top0 = 0;
        let mut top1 = 0;
        for d in 0..200 {
            let a = res.cell_topic[d * 2];
            let b = res.cell_topic[d * 2 + 1];
            if a > b { top0 += 1; } else { top1 += 1; }
        }
        // Majority agreement with some topic label
        let disagreement = top0.min(top1);
        assert!(disagreement < 40, "topic-A docs should concentrate: {} on topic 0, {} on topic 1", top0, top1);
    }

    #[test]
    fn empty_doc_gets_uniform_theta() {
        let n_words = 5;
        let row_ptr = vec![0_usize, 0, 2];
        let col_idx: Vec<u32> = vec![0, 1];
        let counts: Vec<f32> = vec![1.0, 1.0];
        let res = online_vb_lda(&row_ptr, &col_idx, &counts, n_words, 2, 0.5, 0.5, 64.0, 0.7, 1, 5, 0);
        // doc 0 is empty -> uniform
        assert_abs_diff_eq!(res.cell_topic[0], 0.5, epsilon = 1e-4);
        assert_abs_diff_eq!(res.cell_topic[1], 0.5, epsilon = 1e-4);
    }
}
