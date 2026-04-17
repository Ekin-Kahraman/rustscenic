//! Seeded RNG. Uses a splittable stream derived from the user seed so
//! per-target inference is deterministic and reproducible across thread pools.
//!
//! Not numpy-bit-identical: sklearn tree-fitting consumes RNG in a specific
//! per-split order inside Cython; matching that exactly requires reimplementing
//! sklearn's Cython RNG tape. For v0.1 we rely on statistical equivalence
//! (Spearman/Jaccard gates) rather than bit-identity.

use rand::{RngCore, SeedableRng};
use rand::rngs::StdRng;

pub struct TargetRng {
    inner: StdRng,
}

impl TargetRng {
    /// Derive a per-target RNG from the global seed + target index.
    /// Uses hash-based splitting (no global RNG mutation → parallel safe).
    pub fn new(global_seed: u64, target_idx: usize) -> Self {
        let mixed = splitmix64(global_seed ^ (target_idx as u64).wrapping_mul(0x9E3779B97F4A7C15));
        Self { inner: StdRng::seed_from_u64(mixed) }
    }

    /// Fresh RNG for a specific tree within a target's boosting loop.
    pub fn for_tree(&mut self, tree_idx: usize) -> StdRng {
        let mixed = splitmix64(self.inner.next_u64() ^ (tree_idx as u64).wrapping_mul(0xBF58476D1CE4E5B9));
        StdRng::seed_from_u64(mixed)
    }

    pub fn inner_mut(&mut self) -> &mut StdRng {
        &mut self.inner
    }
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Sample `k` distinct indices from `0..n` without replacement (Fisher-Yates head).
pub fn sample_indices(rng: &mut impl RngCore, n: usize, k: usize) -> Vec<usize> {
    let k = k.min(n);
    let mut pool: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = i + (rng.next_u64() as usize) % (n - i);
        pool.swap(i, j);
    }
    pool.truncate(k);
    pool
}

/// Bernoulli-like subsample mask: returns cell indices kept.
pub fn subsample_rows(rng: &mut impl RngCore, n: usize, frac: f32) -> Vec<usize> {
    (0..n)
        .filter(|_| {
            let u: f32 = (rng.next_u32() as f32) / (u32::MAX as f32);
            u < frac
        })
        .collect()
}

/// Fill `out` in place with bernoulli-sampled row indices.
pub fn subsample_rows_into(rng: &mut impl RngCore, n: usize, frac: f32, out: &mut Vec<usize>) {
    out.clear();
    for i in 0..n {
        let u: f32 = (rng.next_u32() as f32) / (u32::MAX as f32);
        if u < frac {
            out.push(i);
        }
    }
}

/// Module-level `rand` helper matching numpy's integer generation range style.
pub fn uniform_u32(rng: &mut impl RngCore) -> u32 {
    rng.next_u32()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn target_rng_is_deterministic() {
        let a = TargetRng::new(42, 7);
        let b = TargetRng::new(42, 7);
        let mut ra = a.inner;
        let mut rb = b.inner;
        for _ in 0..100 {
            assert_eq!(ra.next_u64(), rb.next_u64());
        }
    }

    #[test]
    fn different_targets_diverge() {
        let a = TargetRng::new(42, 0);
        let b = TargetRng::new(42, 1);
        let mut ra = a.inner;
        let mut rb = b.inner;
        let mut same = 0;
        for _ in 0..100 {
            if ra.next_u64() == rb.next_u64() {
                same += 1;
            }
        }
        assert!(same < 5, "different target seeds should not collide");
    }

    #[test]
    fn sample_indices_no_duplicates() {
        let mut rng = StdRng::seed_from_u64(0);
        let s = sample_indices(&mut rng, 100, 20);
        assert_eq!(s.len(), 20);
        let set: std::collections::HashSet<_> = s.iter().copied().collect();
        assert_eq!(set.len(), 20);
    }

    #[test]
    fn subsample_fraction_correct() {
        let mut rng = StdRng::seed_from_u64(0);
        let s = subsample_rows(&mut rng, 10_000, 0.9);
        let frac = s.len() as f32 / 10_000.0;
        assert!((frac - 0.9).abs() < 0.02, "subsample ~0.9, got {}", frac);
    }
}
