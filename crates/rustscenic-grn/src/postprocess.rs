//! Output filtering for indexed GRN edges, before Python strings are created.

use crate::IndexedAdjacency;
use std::{cmp::Ordering, collections::HashMap};

/// Return the unfiltered count and the selected edges.
///
/// Without a top-K cap, preserve inference order. With a cap, return descending
/// importance and break ties by original inference order, matching the Python
/// API's stable-sort contract.
pub fn filter_edges(
    mut edges: Vec<IndexedAdjacency>,
    min_importance: Option<f32>,
    top_targets_per_tf: Option<usize>,
) -> (usize, Vec<IndexedAdjacency>) {
    let raw_count = edges.len();
    if let Some(floor) = min_importance {
        edges.retain(|edge| edge.importance >= floor);
    }
    let Some(k) = top_targets_per_tf else {
        return (raw_count, edges);
    };
    if k == 0 || edges.is_empty() {
        return (raw_count, Vec::new());
    }

    let mut by_tf: HashMap<usize, Vec<usize>> = HashMap::new();
    for (index, edge) in edges.iter().enumerate() {
        by_tf.entry(edge.tf_idx).or_default().push(index);
    }
    let compare = |&a: &usize, &b: &usize| compare_edges(&edges, a, b);
    // A cap larger than the network must not request k * TF-count elements.
    let capacity = k.saturating_mul(by_tf.len()).min(edges.len());
    let mut order = Vec::with_capacity(capacity);
    for group in by_tf.values_mut() {
        if group.len() > k {
            // Only the best k indices are needed. The original index makes the
            // comparator total, preserving tie membership under quickselect.
            group.select_nth_unstable_by(k, compare);
            order.extend_from_slice(&group[..k]);
        } else {
            order.extend_from_slice(group);
        }
    }
    order.sort_unstable_by(compare);
    let selected = order.into_iter().map(|index| edges[index]).collect();
    (raw_count, selected)
}

fn compare_edges(edges: &[IndexedAdjacency], a: usize, b: usize) -> Ordering {
    let left = edges[a].importance;
    let right = edges[b].importance;
    match (left.is_nan(), right.is_nan()) {
        (true, true) => a.cmp(&b),
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => right
            .partial_cmp(&left)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.cmp(&b)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(
        input: Vec<IndexedAdjacency>,
        floor: Option<f32>,
        k: Option<usize>,
    ) -> (usize, Vec<IndexedAdjacency>) {
        let raw = input.len();
        let mut edges: Vec<_> = input
            .into_iter()
            .filter(|edge| floor.is_none_or(|min| edge.importance >= min))
            .collect();
        if let Some(k) = k {
            // Independent reference: full global stable sort, then a per-TF
            // count cap, rather than per-TF partitioning and a final sort.
            let mut order: Vec<_> = (0..edges.len()).collect();
            order.sort_by(|&a, &b| compare_edges(&edges, a, b));
            let mut counts = HashMap::new();
            edges = order
                .into_iter()
                .filter_map(|index| {
                    let count = counts.entry(edges[index].tf_idx).or_insert(0);
                    *count += 1;
                    (*count <= k).then_some(edges[index])
                })
                .collect();
        }
        (raw, edges)
    }

    #[test]
    fn partial_selection_preserves_full_sort_values_order_and_ties() {
        for n in [0, 1, 17, 1000] {
            let input: Vec<_> = (0..n)
                .map(|i| IndexedAdjacency {
                    tf_idx: i % 13,
                    target_idx: i,
                    importance: ((i * 37) % 17) as f32 / 16.0,
                })
                .collect();
            for floor in [
                None,
                Some(0.5),
                Some(f32::NAN),
                Some(f32::INFINITY),
                Some(f32::NEG_INFINITY),
            ] {
                for k in [None, Some(0), Some(1), Some(7), Some(100), Some(usize::MAX)] {
                    let (raw, actual) = filter_edges(input.clone(), floor, k);
                    let (expected_raw, expected) = reference(input.clone(), floor, k);
                    assert_eq!(raw, expected_raw);
                    let signature = |edges: Vec<IndexedAdjacency>| {
                        edges
                            .into_iter()
                            .map(|e| (e.tf_idx, e.target_idx, e.importance.to_bits()))
                            .collect::<Vec<_>>()
                    };
                    assert_eq!(
                        signature(actual),
                        signature(expected),
                        "n={n} floor={floor:?} k={k:?}"
                    );
                }
            }
        }
    }
}
