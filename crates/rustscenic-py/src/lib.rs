//! PyO3 bindings for rustscenic. Python package name: `rustscenic`.
use numpy::{Element, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use rayon::prelude::*;

use rustscenic_aucell::{aucell_sparse_csr, aucell_view};
use rustscenic_grn::{
    infer_indices_sparse_csc_with_overlap, infer_indices_view_with_overlap,
    infer_indices_with_overlap, GrnConfig, IndexedAdjacency, TfOverlapSummary,
};
use rustscenic_preproc::{
    build_cell_peak_matrix, build_cell_peak_matrix_for_barcodes, call_peaks_from_pseudobulks,
    fragments::barcode_qc_counts, frip as preproc_frip_fn,
    insert_size_stats as preproc_insert_size_stats_fn, read_fragments, read_peaks,
    tss_enrichment as preproc_tss_enrichment_fn, PeakCallingConfig, TssSite,
};
use rustscenic_topics::{online_vb_lda, topic_coherence_npmi_view};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

type GrnInferFilteredPyResult = (
    Py<PyList>,
    Py<PyList>,
    Py<PyArray1<f32>>,
    usize,
    usize,
    Py<PyList>,
    f32,
);
type AucellPyResult = (Py<PyArray2<f32>>, f32);
type TopicFitPyResult = (Py<PyArray2<f32>>, Py<PyArray2<f32>>);
type StageRegulonIndexPyResult = (Py<PyList>, Py<PyList>, usize);
type StageRegulonIndexCoveragePyResult = (
    Py<PyList>,
    Py<PyList>,
    Py<PyArray1<u64>>,
    Py<PyArray1<u64>>,
    usize,
);
type StageRegulonCoveragePyResult = (Py<PyArray1<u64>>, Py<PyArray1<u64>>);
type PipelineCellIndexPyResult = Py<PyArray1<u64>>;
type PipelineAttributeRowsF32PyResult = (Py<PyList>, Py<PyList>, Py<PyList>, Py<PyArray1<f32>>);
type PipelineAttributeRowsF64PyResult = (Py<PyList>, Py<PyList>, Py<PyList>, Py<PyArray1<f64>>);
type PipelineRankingColumnProjectionPyResult = (Py<PyList>, Py<PyList>);
type PreprocPeakCoordsPyResult = (
    Py<PyList>,        // peak IDs
    Py<PyList>,        // chromosome names
    Py<PyArray1<u32>>, // starts
    Py<PyArray1<u32>>, // ends
);
type FragmentsToMatrixPyResult = (
    Py<PyArray1<u32>>, // data
    Py<PyArray1<u32>>, // indices
    Py<PyArray1<u64>>, // indptr
    (usize, usize),    // shape (n_cells, n_peaks)
    Py<PyList>,        // barcode names
    Py<PyList>,        // peak names
    Py<PyArray1<u32>>, // fragments per barcode
    Py<PyArray1<u32>>, // total counts per barcode
    usize,             // total fragment records
    f32,               // median fragments per barcode, NaN when not computed
);
type InsertSizeStatsPyResult = (
    Py<PyList>,        // barcodes
    Py<PyArray1<f32>>, // mean
    Py<PyArray1<f32>>, // median
    Py<PyArray1<u32>>, // n_fragments
    Py<PyArray1<u32>>, // sub
    Py<PyArray1<u32>>, // mono
    Py<PyArray1<u32>>, // di
);
type CallPeaksPyResult = (Py<PyList>, Py<PyArray1<u32>>, Py<PyArray1<u32>>, Py<PyList>);
type GeneDuplicateSummaryPyResult = (usize, Py<PyList>);
type GeneDedupeSparsePyResult<T> = (
    Py<PyArray1<T>>,   // data
    Py<PyArray1<i32>>, // row indices
    Py<PyArray1<i64>>, // column indptr
    Py<PyList>,        // unique gene names
);
type SpecificityGroupCodesOrderPyResult = (Py<PyArray1<i32>>, Py<PyArray1<u64>>);
type SpecificityRssPyResult = Py<PyArray2<f64>>;
type EnhancerParsePeakNamesPyResult = (
    Py<PyList>,        // chromosome names
    Py<PyArray1<i64>>, // starts
    Py<PyArray1<i64>>, // ends
);
type EnhancerAlignCellIndicesPyResult = (
    Py<PyArray1<u64>>, // RNA cell indices
    Py<PyArray1<u64>>, // ATAC cell indices
);
type EnhancerChromCodesPyResult = (
    Py<PyList>,        // normalised gene chromosome names
    Py<PyList>,        // normalised peak chromosome names
    Py<PyArray1<i32>>, // gene chromosome codes
    Py<PyArray1<i32>>, // peak chromosome codes
);
type EnhancerGeneMatchPyResult = (
    Py<PyArray1<u64>>, // gene-coordinate row indices present in RNA
    Py<PyArray1<i64>>, // source RNA column indices
);
type EnhancerPeakCoordMatchPyResult = (
    Py<PyArray1<u64>>, // peak-coordinate row indices in ATAC peak order
    Py<PyList>,        // missing ATAC peak names
);
type EnhancerPearsonPyResult = (
    Py<PyArray1<u32>>, // peak indices
    Py<PyArray1<u32>>, // gene indices in the sorted gene-coordinate frame
    Py<PyArray1<i64>>, // signed peak-centre minus TSS distances
    Py<PyArray1<f32>>, // Pearson correlations
);
type ERegulonAssemblePyResult = (
    Py<PyList>,        // tf
    Py<PyList>,        // enhancer
    Py<PyList>,        // target gene
    Py<PyArray1<u32>>, // n_enhancer_links
    Py<PyArray1<f64>>, // motif_auc
    usize,             // number of input TFs after cistarget filtering
    usize,             // number of assembled eRegulons
);
type ERegulonSummaryPyResult = (
    Py<PyList>,        // tf
    Py<PyList>,        // enhancer lists
    Py<PyList>,        // target gene lists
    Py<PyArray1<u32>>, // n_enhancer_links
    Py<PyArray1<f64>>, // motif_auc
    Py<PyList>,        // target -> peak-list dictionaries
    usize,             // number of input TFs after cistarget filtering
    usize,             // number of assembled eRegulons
);
type RegionAttributionIndexPyResult = (Py<PyArray1<u64>>, Py<PyArray1<u64>>);
type RegionAttributionPeakValuePyResult = (Py<PyArray1<u64>>, Py<PyList>);
type MotifAnnotationPrunePyResult = (
    Py<PyArray1<u64>>, // enriched row indices
    Py<PyList>,        // TF names
    Py<PyList>,        // annotation TF names
);
type MotifAnnotationPruneRowsF32PyResult = (
    Py<PyList>,                // regulon
    Py<PyList>,                // motif
    Py<PyArray1<f32>>,         // auc
    Option<Py<PyArray1<f32>>>, // nes
    Py<PyList>,                // TF names
    Py<PyList>,                // annotation TF names
);
type MotifAnnotationPruneRowsF64PyResult = (
    Py<PyList>,                // regulon
    Py<PyList>,                // motif
    Py<PyArray1<f64>>,         // auc
    Option<Py<PyArray1<f64>>>, // nes
    Py<PyList>,                // TF names
    Py<PyList>,                // annotation TF names
);
type PrunedRegulonTargetsPyResult = (
    Py<PyList>, // regulon names
    Py<PyList>, // target gene lists
);
type PeakRegulonsWithFeaturesPyResult = (
    Py<PyList>, // regulon names
    Py<PyList>, // peak lists
    Py<PyList>, // sorted unique peak IDs
);
type CandidateRegulonsPyResult = (
    Py<PyList>, // regulon names
    Py<PyList>, // target gene lists
);
type CistargetEnrichmentRowsPyResult = (
    Py<PyList>,        // regulon
    Py<PyList>,        // motif
    Py<PyArray1<f32>>, // auc
    Py<PyArray1<f32>>, // nes
    Py<PyArray1<u32>>, // zero-variance regulon indices
    bool,              // motif universe too small for NES
);

struct CistargetRows {
    regulon: Vec<String>,
    motif: Vec<String>,
    auc: Vec<f32>,
    nes: Vec<f32>,
    zero_var_indices: Vec<u32>,
    too_few_motifs: bool,
}

#[derive(Clone, Copy)]
struct GenePearsonStats {
    col: usize,
    sum: f64,
    std: f64,
}

#[derive(Clone, Copy)]
struct SparseGenePearsonStats {
    start: usize,
    end: usize,
    sum: f64,
    std: f64,
}

#[derive(Clone)]
struct EnhancerPearsonLink {
    peak_idx: u32,
    gene_idx: u32,
    distance: i64,
    correlation: f32,
}

#[derive(Default)]
struct CtTfRows<'a> {
    peaks: Vec<&'a str>,
    seen_peaks: HashSet<&'a str>,
    auc_rows: Vec<(&'a str, f64)>,
}

impl<'a> CtTfRows<'a> {
    fn push(&mut self, peak: &'a str, auc: f64) {
        if self.seen_peaks.insert(peak) {
            self.peaks.push(peak);
        }
        self.auc_rows.push((peak, auc));
    }
}

struct TargetSupport<'a> {
    gene: &'a str,
    peaks: Vec<&'a str>,
    seen_peaks: HashSet<&'a str>,
}

impl<'a> TargetSupport<'a> {
    fn new(gene: &'a str) -> Self {
        Self {
            gene,
            peaks: Vec::new(),
            seen_peaks: HashSet::new(),
        }
    }

    fn add_peak(&mut self, peak: &'a str) {
        if self.seen_peaks.insert(peak) {
            self.peaks.push(peak);
        }
    }
}

struct ERegulonRows<'a> {
    tf: String,
    targets: Vec<TargetSupport<'a>>,
    n_edges: usize,
    motif_auc: f64,
}

struct PreparedRegulonIndices {
    kept_names: Vec<String>,
    kept_indices: Vec<Vec<usize>>,
    matched_counts: Vec<u64>,
    total_counts: Vec<u64>,
    dropped_empty: usize,
}

#[pyfunction]
#[pyo3(signature = (
    expression,
    gene_names,
    tf_names,
    n_estimators = 5000,
    learning_rate = 0.01,
    max_features = 0.1,
    subsample = 0.9,
    max_depth = 3,
    early_stop_window = 25,
    seed = 777,
    target_block_size = 0,
    top_targets_per_tf = None,
    min_importance = None,
))]
#[allow(clippy::too_many_arguments)]
fn grn_infer<'py>(
    py: Python<'py>,
    expression: PyReadonlyArray2<'py, f32>,
    gene_names: Vec<String>,
    tf_names: Vec<String>,
    n_estimators: usize,
    learning_rate: f32,
    max_features: f32,
    subsample: f32,
    max_depth: usize,
    early_stop_window: usize,
    seed: u64,
    target_block_size: usize,
    top_targets_per_tf: Option<usize>,
    min_importance: Option<f32>,
) -> PyResult<GrnInferFilteredPyResult> {
    let arr = expression.as_array();
    let n_cells = arr.shape()[0];
    let n_genes = arr.shape()[1];
    if gene_names.len() != n_genes {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "gene_names.len() = {} but expression has {} gene columns",
            gene_names.len(),
            n_genes
        )));
    }
    let expression_max = validate_finite_array2_max(arr, "expression matrix")?;
    let cfg = GrnConfig {
        n_estimators,
        learning_rate,
        max_features,
        subsample,
        max_depth,
        early_stop_window,
        seed,
        target_block_size,
    };

    let adjacencies = py.allow_threads(|| {
        if let Some(expr_slice) = arr.as_slice() {
            infer_indices_with_overlap(expr_slice, n_cells, &gene_names, &tf_names, &cfg)
        } else {
            infer_indices_view_with_overlap(arr, &gene_names, &tf_names, &cfg)
        }
    });
    grn_edges_to_py(
        py,
        &gene_names,
        adjacencies.0,
        min_importance,
        top_targets_per_tf,
        adjacencies.1,
        expression_max,
    )
}

#[pyfunction]
#[pyo3(signature = (
    indptr,
    indices,
    data,
    n_cells,
    n_genes,
    gene_names,
    tf_names,
    n_estimators = 5000,
    learning_rate = 0.01,
    max_features = 0.1,
    subsample = 0.9,
    max_depth = 3,
    early_stop_window = 25,
    seed = 777,
    target_block_size = 0,
    top_targets_per_tf = None,
    min_importance = None,
))]
#[allow(clippy::too_many_arguments)]
fn grn_infer_sparse_csc<'py>(
    py: Python<'py>,
    indptr: &Bound<'py, PyAny>,
    indices: PyReadonlyArray1<'py, i32>,
    data: PyReadonlyArray1<'py, f32>,
    n_cells: usize,
    n_genes: usize,
    gene_names: Vec<String>,
    tf_names: Vec<String>,
    n_estimators: usize,
    learning_rate: f32,
    max_features: f32,
    subsample: f32,
    max_depth: usize,
    early_stop_window: usize,
    seed: u64,
    target_block_size: usize,
    top_targets_per_tf: Option<usize>,
    min_importance: Option<f32>,
) -> PyResult<GrnInferFilteredPyResult> {
    if gene_names.len() != n_genes {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "gene_names.len() = {} but sparse matrix has {} gene columns",
            gene_names.len(),
            n_genes
        )));
    }
    let indices = indices
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("indices must be contiguous"))?;
    let data = data
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("data must be contiguous"))?;
    let indptr = row_ptr_any_to_usize(indptr, indices.len(), data.len())?;
    let expression_max = validate_csc_arrays(&indptr, indices, data, n_cells, n_genes)?;

    let cfg = GrnConfig {
        n_estimators,
        learning_rate,
        max_features,
        subsample,
        max_depth,
        early_stop_window,
        seed,
        target_block_size,
    };

    let adjacencies = py.allow_threads(|| {
        infer_indices_sparse_csc_with_overlap(
            &indptr,
            indices,
            data,
            n_cells,
            n_genes,
            &gene_names,
            &tf_names,
            &cfg,
        )
    });
    grn_edges_to_py(
        py,
        &gene_names,
        adjacencies.0,
        min_importance,
        top_targets_per_tf,
        adjacencies.1,
        expression_max,
    )
}

fn grn_edges_to_py(
    py: Python<'_>,
    gene_names: &[String],
    adjacencies: Vec<IndexedAdjacency>,
    min_importance: Option<f32>,
    top_targets_per_tf: Option<usize>,
    tf_overlap: TfOverlapSummary,
    expression_max: f32,
) -> PyResult<GrnInferFilteredPyResult> {
    let (raw_n_edges, adjacencies) =
        filter_grn_edges(adjacencies, min_importance, top_targets_per_tf);
    let tfs = PyList::new(
        py,
        adjacencies.iter().map(|a| gene_names[a.tf_idx].as_str()),
    )?;
    let targets = PyList::new(
        py,
        adjacencies
            .iter()
            .map(|a| gene_names[a.target_idx].as_str()),
    )?;
    let imp: Vec<f32> = adjacencies.iter().map(|a| a.importance).collect();
    let imp_arr = PyArray1::from_vec(py, imp);
    let missing_examples = PyList::new(py, tf_overlap.missing_examples.iter().map(String::as_str))?;
    Ok((
        tfs.unbind(),
        targets.unbind(),
        imp_arr.unbind(),
        raw_n_edges,
        tf_overlap.present_input_count,
        missing_examples.unbind(),
        expression_max,
    ))
}

fn validate_csc_arrays(
    indptr: &[usize],
    indices: &[i32],
    data: &[f32],
    n_cells: usize,
    n_genes: usize,
) -> PyResult<f32> {
    if indptr.len() != n_genes + 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "indptr has length {} but expected n_genes + 1 = {}",
            indptr.len(),
            n_genes + 1
        )));
    }
    if indptr.first().copied() != Some(0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "indptr[0] must be 0",
        ));
    }
    for (i, pair) in indptr.windows(2).enumerate() {
        if pair[1] < pair[0] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "indptr must be non-decreasing; indptr[{}] = {} < indptr[{}] = {}",
                i + 1,
                pair[1],
                i,
                pair[0]
            )));
        }
    }
    if indptr.last().copied() != Some(data.len()) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "indptr[-1] = {} but data has length {}",
            indptr.last().copied().unwrap_or(0),
            data.len()
        )));
    }
    if indices.len() != data.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "indices has length {} but data has length {}",
            indices.len(),
            data.len()
        )));
    }
    if let Some((i, &row)) = indices
        .iter()
        .enumerate()
        .find(|(_, &row)| row < 0 || (row as usize) >= n_cells)
    {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "indices[{i}] = {row} is negative or exceeds n_cells = {n_cells}"
        )));
    }
    let total_entries = n_cells.saturating_mul(n_genes);
    let mut max_value = if data.len() < total_entries || data.is_empty() {
        0.0_f32
    } else {
        f32::NEG_INFINITY
    };
    for (i, &value) in data.iter().enumerate() {
        if !value.is_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "sparse expression data contain NaN or Inf at position {i}: {value}"
            )));
        }
        if value < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expression matrix is sparse and contains negative values at position {i}: {value}. \
                 The sparse GRN path is for non-negative count/log-expression matrices where \
                 implicit missing entries are true zeros. Pass a dense matrix for signed \
                 residuals or centred/scaled expression."
            )));
        }
        max_value = max_value.max(value);
    }
    Ok(max_value)
}

fn validate_finite_array2_max(values: ndarray::ArrayView2<'_, f32>, label: &str) -> PyResult<f32> {
    let mut max_value = 0.0_f32;
    for (i, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{label} contains NaN or Inf at position {i}: {value}"
            )));
        }
        if i == 0 || value > max_value {
            max_value = value;
        }
    }
    Ok(max_value)
}

fn validate_finite_sparse_values_max(
    values: &[f32],
    total_entries: usize,
    label: &str,
) -> PyResult<f32> {
    let mut max_value = if values.len() < total_entries || values.is_empty() {
        0.0_f32
    } else {
        f32::NEG_INFINITY
    };
    for (i, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{label} contains NaN or Inf at position {i}: {value}"
            )));
        }
        max_value = max_value.max(value);
    }
    Ok(max_value)
}

fn filter_grn_edges(
    mut edges: Vec<IndexedAdjacency>,
    min_importance: Option<f32>,
    top_targets_per_tf: Option<usize>,
) -> (usize, Vec<IndexedAdjacency>) {
    let raw_n_edges = edges.len();
    if let Some(min_importance) = min_importance {
        edges.retain(|edge| edge.importance >= min_importance);
    }
    let Some(top_targets_per_tf) = top_targets_per_tf else {
        return (raw_n_edges, edges);
    };
    if top_targets_per_tf == 0 || edges.is_empty() {
        return (raw_n_edges, Vec::new());
    }

    let mut by_tf: HashMap<usize, Vec<usize>> = HashMap::new();
    for (idx, edge) in edges.iter().enumerate() {
        by_tf.entry(edge.tf_idx).or_default().push(idx);
    }

    let mut order = Vec::with_capacity(top_targets_per_tf.saturating_mul(by_tf.len()));
    for group in by_tf.values_mut() {
        group.sort_by(|&a, &b| {
            compare_importance_desc(
                f64::from(edges[a].importance),
                f64::from(edges[b].importance),
                a,
                b,
            )
        });
        order.extend(group.iter().take(top_targets_per_tf).copied());
    }
    order.sort_by(|&a, &b| {
        compare_importance_desc(
            f64::from(edges[a].importance),
            f64::from(edges[b].importance),
            a,
            b,
        )
    });

    let filtered = order.into_iter().map(|idx| edges[idx]).collect();
    (raw_n_edges, filtered)
}

/// Compute per-cell AUCell regulon activity.
///
/// - `expression`: (n_cells x n_genes) f32 matrix
/// - `regulon_names`: list of regulon identifiers (length == n_regulons)
/// - `regulon_gene_indices`: list of per-regulon Vec<usize> gene indices; indices
///   must be < n_genes. Enforced on the Python side.
/// - `top_frac`: fraction of top-ranked genes per cell used as AUC cutoff.
///   0.05 matches pyscenic default.
///
/// Returns (n_cells, n_regulons) f32 matrix of normalized recovery AUCs.
#[pyfunction]
#[pyo3(signature = (expression, regulon_names, regulon_gene_indices, top_frac = 0.05))]
fn aucell_score<'py>(
    py: Python<'py>,
    expression: PyReadonlyArray2<'py, f32>,
    regulon_names: Vec<String>,
    regulon_gene_indices: Vec<Vec<usize>>,
    top_frac: f32,
) -> PyResult<AucellPyResult> {
    let arr = expression.as_array();
    let n_cells = arr.shape()[0];
    let n_genes = arr.shape()[1];
    let expression_max = validate_finite_array2_max(arr, "expression matrix")?;

    if regulon_names.len() != regulon_gene_indices.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "len(regulon_names)={} mismatch len(regulon_gene_indices)={}",
            regulon_names.len(),
            regulon_gene_indices.len()
        )));
    }
    for (i, ix) in regulon_gene_indices.iter().enumerate() {
        for &g in ix {
            if g >= n_genes {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "regulon[{}] has gene index {} >= n_genes {}",
                    i, g, n_genes
                )));
            }
        }
    }

    let regulons: Vec<(String, Vec<usize>)> = regulon_names
        .into_iter()
        .zip(regulon_gene_indices)
        .collect();

    let out: Vec<f32> = py.allow_threads(|| aucell_view(arr, &regulons, top_frac));
    let n_regulons = regulons.len();

    let arr2 = ndarray::Array2::from_shape_vec((n_cells, n_regulons), out).map_err(
        |e: ndarray::ShapeError| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()),
    )?;
    Ok((
        PyArray2::from_owned_array(py, arr2).unbind(),
        expression_max,
    ))
}

/// Sparse CSR AUCell entry point. Keeps sparse AnnData inputs out of dense
/// Python chunks while preserving dense AUCell ranking semantics.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    row_ptr,
    col_idx,
    data,
    n_genes,
    regulon_names,
    regulon_gene_indices,
    top_frac = 0.05,
))]
fn aucell_score_sparse_csr<'py>(
    py: Python<'py>,
    row_ptr: &Bound<'py, PyAny>,
    col_idx: PyReadonlyArray1<'py, i32>,
    data: PyReadonlyArray1<'py, f32>,
    n_genes: usize,
    regulon_names: Vec<String>,
    regulon_gene_indices: Vec<Vec<usize>>,
    top_frac: f32,
) -> PyResult<AucellPyResult> {
    let col_idx_raw = col_idx
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("col_idx must be contiguous"))?;
    let data = data
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("data must be contiguous"))?;
    if col_idx_raw.len() != data.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "col_idx length {} differs from data length {}",
            col_idx_raw.len(),
            data.len()
        )));
    }
    if regulon_names.len() != regulon_gene_indices.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "len(regulon_names)={} mismatch len(regulon_gene_indices)={}",
            regulon_names.len(),
            regulon_gene_indices.len()
        )));
    }
    let row_ptr_usize = row_ptr_any_to_usize(row_ptr, col_idx_raw.len(), data.len())?;
    let n_cells = row_ptr_usize.len() - 1;
    let expression_max = validate_finite_sparse_values_max(
        data,
        n_cells.saturating_mul(n_genes),
        "expression data",
    )?;

    for (i, &raw) in col_idx_raw.iter().enumerate() {
        if raw < 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "col_idx[{i}] = {raw} must be non-negative"
            )));
        }
        let value = raw as usize;
        if value >= n_genes {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "col_idx[{i}] = {value} exceeds n_genes = {n_genes}"
            )));
        }
    }
    for (i, ix) in regulon_gene_indices.iter().enumerate() {
        for &g in ix {
            if g >= n_genes {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "regulon[{}] has gene index {} >= n_genes {}",
                    i, g, n_genes
                )));
            }
        }
    }

    let regulons: Vec<(String, Vec<usize>)> = regulon_names
        .into_iter()
        .zip(regulon_gene_indices)
        .collect();
    let out = py.allow_threads(|| {
        aucell_sparse_csr(
            &row_ptr_usize,
            col_idx_raw,
            data,
            n_cells,
            n_genes,
            &regulons,
            top_frac,
        )
    });
    let n_regulons = regulons.len();
    let arr2 = ndarray::Array2::from_shape_vec((n_cells, n_regulons), out).map_err(
        |e: ndarray::ShapeError| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()),
    )?;
    Ok((
        PyArray2::from_owned_array(py, arr2).unbind(),
        expression_max,
    ))
}

/// Map regulon gene names onto a feature axis.
///
/// Python keeps public regulon object coercion; Rust owns the repeated hash-map
/// lookup shared by AUCell and cistarget.
#[pyfunction]
fn stage_prepare_regulon_indices<'py>(
    py: Python<'py>,
    gene_names: Vec<String>,
    regulon_names: Vec<String>,
    regulon_genes: Vec<Vec<String>>,
) -> PyResult<StageRegulonIndexPyResult> {
    validate_regulon_name_gene_lengths(&regulon_names, &regulon_genes)?;
    let prepared = py.allow_threads(|| {
        prepare_regulon_indices_and_coverage_rs(&gene_names, &regulon_names, &regulon_genes)
    });

    let names_py = PyList::new(py, prepared.kept_names.iter().map(String::as_str))?;
    let indices_py = PyList::empty(py);
    for indices in prepared.kept_indices {
        indices_py.append(PyList::new(py, indices)?)?;
    }
    Ok((
        names_py.unbind(),
        indices_py.unbind(),
        prepared.dropped_empty,
    ))
}

/// Map regulon genes and compute coverage in a single Rust hash-map pass.
#[pyfunction]
fn stage_prepare_regulon_indices_with_coverage<'py>(
    py: Python<'py>,
    gene_names: Vec<String>,
    regulon_names: Vec<String>,
    regulon_genes: Vec<Vec<String>>,
) -> PyResult<StageRegulonIndexCoveragePyResult> {
    validate_regulon_name_gene_lengths(&regulon_names, &regulon_genes)?;
    let prepared = py.allow_threads(|| {
        prepare_regulon_indices_and_coverage_rs(&gene_names, &regulon_names, &regulon_genes)
    });

    let names_py = PyList::new(py, prepared.kept_names.iter().map(String::as_str))?;
    let indices_py = PyList::empty(py);
    for indices in prepared.kept_indices {
        indices_py.append(PyList::new(py, indices)?)?;
    }
    Ok((
        names_py.unbind(),
        indices_py.unbind(),
        PyArray1::from_vec(py, prepared.matched_counts).unbind(),
        PyArray1::from_vec(py, prepared.total_counts).unbind(),
        prepared.dropped_empty,
    ))
}

/// Count how many genes from each regulon are present in a feature axis.
#[pyfunction]
fn stage_regulon_coverage_counts<'py>(
    py: Python<'py>,
    gene_names: Vec<String>,
    regulon_genes: Vec<Vec<String>>,
) -> PyResult<StageRegulonCoveragePyResult> {
    let (matched, total) =
        py.allow_threads(|| regulon_coverage_counts_rs(&gene_names, &regulon_genes));
    Ok((
        PyArray1::from_vec(py, matched).unbind(),
        PyArray1::from_vec(py, total).unbind(),
    ))
}

fn validate_regulon_name_gene_lengths(
    regulon_names: &[String],
    regulon_genes: &[Vec<String>],
) -> PyResult<()> {
    if regulon_names.len() != regulon_genes.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "regulon arrays must have the same length; got names={} genes={}",
            regulon_names.len(),
            regulon_genes.len()
        )));
    }
    Ok(())
}

fn prepare_regulon_indices_and_coverage_rs(
    gene_names: &[String],
    regulon_names: &[String],
    regulon_genes: &[Vec<String>],
) -> PreparedRegulonIndices {
    let mut gene_to_idx: HashMap<&str, usize> = HashMap::with_capacity(gene_names.len());
    for (idx, gene) in gene_names.iter().enumerate() {
        gene_to_idx.insert(gene.as_str(), idx);
    }

    let mut kept_names = Vec::new();
    let mut kept_indices = Vec::new();
    let mut matched_counts = Vec::with_capacity(regulon_genes.len());
    let mut total_counts = Vec::with_capacity(regulon_genes.len());
    let mut dropped_empty = 0usize;
    for (name, genes) in regulon_names.iter().zip(regulon_genes) {
        let indices: Vec<usize> = genes
            .iter()
            .filter_map(|gene| gene_to_idx.get(gene.as_str()).copied())
            .collect();
        matched_counts.push(indices.len() as u64);
        total_counts.push(genes.len() as u64);
        if indices.is_empty() {
            dropped_empty += 1;
        } else {
            kept_names.push(name.clone());
            kept_indices.push(indices);
        }
    }

    PreparedRegulonIndices {
        kept_names,
        kept_indices,
        matched_counts,
        total_counts,
        dropped_empty,
    }
}

fn regulon_coverage_counts_rs(
    gene_names: &[String],
    regulon_genes: &[Vec<String>],
) -> (Vec<u64>, Vec<u64>) {
    let gene_set: HashSet<&str> = gene_names.iter().map(|gene| gene.as_str()).collect();
    let mut matched = Vec::with_capacity(regulon_genes.len());
    let mut total = Vec::with_capacity(regulon_genes.len());
    for genes in regulon_genes {
        matched.push(
            genes
                .iter()
                .filter(|gene| gene_set.contains(gene.as_str()))
                .count() as u64,
        );
        total.push(genes.len() as u64);
    }
    (matched, total)
}

macro_rules! cistarget_enrichment_pyfunction {
    ($name:ident, $rank_ty:ty, $label:literal) => {
        #[pyfunction]
        #[pyo3(signature = (rankings, motif_names, regulon_names, regulon_gene_indices, top_frac = 0.05, auc_threshold = 0.05, nes_threshold = None, nes_min_motifs = 30))]
        #[allow(clippy::too_many_arguments)]
        fn $name<'py>(
            py: Python<'py>,
            rankings: PyReadonlyArray2<'py, $rank_ty>,
            motif_names: Vec<String>,
            regulon_names: Vec<String>,
            regulon_gene_indices: Vec<Vec<usize>>,
            top_frac: f32,
            auc_threshold: f32,
            nes_threshold: Option<f32>,
            nes_min_motifs: usize,
        ) -> PyResult<CistargetEnrichmentRowsPyResult> {
            cistarget_enrichment_from_integer_rankings_py(
                py,
                rankings,
                motif_names,
                regulon_names,
                regulon_gene_indices,
                top_frac,
                auc_threshold,
                nes_threshold,
                nes_min_motifs,
                $label,
            )
        }
    };
}

cistarget_enrichment_pyfunction!(cistarget_enrichment_from_rankings_i16, i16, "int16");
cistarget_enrichment_pyfunction!(cistarget_enrichment_from_rankings_i32, i32, "int32");
cistarget_enrichment_pyfunction!(cistarget_enrichment_from_rankings_i64, i64, "int64");

macro_rules! cistarget_rankings_to_i32_pyfunction {
    ($name:ident, $value_ty:ty) => {
        #[pyfunction]
        fn $name<'py>(
            py: Python<'py>,
            rankings: PyReadonlyArray2<'py, $value_ty>,
        ) -> PyResult<Py<PyArray2<i32>>> {
            let arr = rankings.as_array();
            let shape = (arr.shape()[0], arr.shape()[1]);
            let converted = py.allow_threads(|| -> PyResult<Vec<i32>> {
                let mut out = Vec::with_capacity(arr.len());
                for value in arr.iter().copied() {
                    let value = value as f64;
                    if !value.is_finite() {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "rankings contain NaN or Inf values - motif enrichment is undefined on non-finite ranks",
                        ));
                    }
                    if value.fract() != 0.0 {
                        return Err(pyo3::exceptions::PyTypeError::new_err(
                            "rankings must contain integer rank values for cistarget enrichment",
                        ));
                    }
                    if value < i32::MIN as f64 || value > i32::MAX as f64 {
                        return Err(pyo3::exceptions::PyOverflowError::new_err(
                            "rankings contain values outside int32 range",
                        ));
                    }
                    out.push(value as i32);
                }
                Ok(out)
            })?;
            let out = ndarray::Array2::from_shape_vec(shape, converted).map_err(
                |e: ndarray::ShapeError| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()),
            )?;
            Ok(PyArray2::from_owned_array(py, out).unbind())
        }
    };
}

cistarget_rankings_to_i32_pyfunction!(cistarget_rankings_to_i32_f32, f32);
cistarget_rankings_to_i32_pyfunction!(cistarget_rankings_to_i32_f64, f64);

#[allow(clippy::too_many_arguments)]
fn cistarget_enrichment_from_integer_rankings_py<'py, T>(
    py: Python<'py>,
    rankings: PyReadonlyArray2<'py, T>,
    motif_names: Vec<String>,
    regulon_names: Vec<String>,
    regulon_gene_indices: Vec<Vec<usize>>,
    top_frac: f32,
    auc_threshold: f32,
    nes_threshold: Option<f32>,
    nes_min_motifs: usize,
    _label: &'static str,
) -> PyResult<CistargetEnrichmentRowsPyResult>
where
    T: Element + Copy + Ord + Into<i64> + Send + Sync,
{
    let arr = rankings.as_array();
    let n_motifs = arr.shape()[0];
    let n_genes = arr.shape()[1];
    validate_cistarget_ranking_inputs(
        n_motifs,
        n_genes,
        &motif_names,
        &regulon_names,
        &regulon_gene_indices,
    )?;
    let n_regulons = regulon_names.len();
    let auc = py.allow_threads(|| {
        cistarget_auc_from_integer_rankings(arr, &regulon_gene_indices, top_frac)
    });
    let rows = cistarget_enrichment_rows_impl(
        &auc,
        n_motifs,
        n_regulons,
        &motif_names,
        &regulon_names,
        auc_threshold,
        nes_threshold,
        nes_min_motifs,
    );
    cistarget_rows_to_py(py, rows)
}

fn validate_cistarget_ranking_inputs(
    n_motifs: usize,
    n_genes: usize,
    motif_names: &[String],
    regulon_names: &[String],
    regulon_gene_indices: &[Vec<usize>],
) -> PyResult<()> {
    if motif_names.len() != n_motifs {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "motif_names has length {} but rankings has {} motif rows",
            motif_names.len(),
            n_motifs
        )));
    }
    if regulon_names.len() != regulon_gene_indices.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "len(regulon_names)={} mismatch len(regulon_gene_indices)={}",
            regulon_names.len(),
            regulon_gene_indices.len()
        )));
    }
    for (i, ix) in regulon_gene_indices.iter().enumerate() {
        for &g in ix {
            if g >= n_genes {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "regulon[{}] has gene index {} >= n_genes {}",
                    i, g, n_genes
                )));
            }
        }
    }
    Ok(())
}

fn cistarget_auc_from_integer_rankings<T>(
    rankings: ndarray::ArrayView2<'_, T>,
    regulon_gene_indices: &[Vec<usize>],
    top_frac: f32,
) -> Vec<f32>
where
    T: Copy + Ord + Into<i64> + Send + Sync,
{
    let n_motifs = rankings.shape()[0];
    let n_genes = rankings.shape()[1];
    let n_regulons = regulon_gene_indices.len();
    let mut auc = vec![0.0_f32; n_motifs * n_regulons];
    if n_regulons == 0 {
        return auc;
    }
    let rank_cutoff = cistarget_rank_cutoff(top_frac, n_genes);
    if rank_cutoff == 0 {
        return auc;
    }

    auc.par_chunks_mut(n_regulons)
        .enumerate()
        .for_each(|(motif_idx, motif_auc)| {
            let row = rankings.row(motif_idx);
            score_integer_rank_row(row, regulon_gene_indices, rank_cutoff, motif_auc);
        });
    auc
}

fn cistarget_rank_cutoff(top_frac: f32, n_genes: usize) -> usize {
    let raw = (top_frac * n_genes as f32).round() as i64;
    (raw - 1).max(0) as usize
}

fn score_integer_rank_row<T>(
    row: ndarray::ArrayView1<'_, T>,
    regulon_gene_indices: &[Vec<usize>],
    rank_cutoff: usize,
    motif_auc: &mut [f32],
) where
    T: Copy + Into<i64>,
{
    for (r_idx, genes) in regulon_gene_indices.iter().enumerate() {
        let mut auc_sum = 0_u64;
        for &gene in genes {
            let rank = row[gene].into();
            if rank >= 0 {
                let rank = rank as usize;
                if rank < rank_cutoff {
                    auc_sum += (rank_cutoff - rank) as u64;
                }
            }
        }
        motif_auc[r_idx] = normalise_cistarget_auc(auc_sum, rank_cutoff, genes.len());
    }
}

fn normalise_cistarget_auc(auc_sum: u64, rank_cutoff: usize, gene_set_len: usize) -> f32 {
    if gene_set_len == 0 {
        return 0.0;
    }
    let max_auc = (rank_cutoff as u64 + 1) * gene_set_len as u64;
    if max_auc == 0 {
        0.0
    } else {
        ((auc_sum as f64) / (max_auc as f64)).clamp(0.0, 1.0) as f32
    }
}

#[allow(clippy::too_many_arguments)]
fn cistarget_enrichment_rows_impl(
    auc_slice: &[f32],
    n_motifs: usize,
    n_regulons: usize,
    motif_names: &[String],
    regulon_names: &[String],
    auc_threshold: f32,
    nes_threshold: Option<f32>,
    nes_min_motifs: usize,
) -> CistargetRows {
    let too_few_motifs = n_motifs < nes_min_motifs;
    let mut means = vec![0.0_f32; n_regulons];
    let mut stds = vec![1.0_f32; n_regulons];
    let mut zero_var = vec![false; n_regulons];
    let mut zero_var_indices = Vec::new();

    if !too_few_motifs && n_motifs > 0 {
        let n = n_motifs as f32;
        for r in 0..n_regulons {
            let mut sum = 0.0_f32;
            for m in 0..n_motifs {
                sum += auc_slice[m * n_regulons + r];
            }
            let mean = sum / n;
            let mut sumsq = 0.0_f32;
            for m in 0..n_motifs {
                let centred = auc_slice[m * n_regulons + r] - mean;
                sumsq += centred * centred;
            }
            let var = (sumsq / n).max(0.0);
            let std = var.sqrt();
            means[r] = mean;
            if std < 1e-6 {
                zero_var[r] = true;
                zero_var_indices.push(r as u32);
            } else {
                stds[r] = std;
            }
        }
    }

    let mut regulon_out = Vec::new();
    let mut motif_out = Vec::new();
    let mut auc_out = Vec::new();
    let mut nes_out = Vec::new();
    for m in 0..n_motifs {
        for r in 0..n_regulons {
            let auc_value = auc_slice[m * n_regulons + r];
            if auc_value < auc_threshold {
                continue;
            }
            let nes_value = if too_few_motifs || zero_var[r] {
                f32::NAN
            } else {
                (auc_value - means[r]) / stds[r]
            };
            if let Some(threshold) = nes_threshold {
                if !nes_value.is_finite() || nes_value < threshold {
                    continue;
                }
            }
            regulon_out.push(regulon_names[r].clone());
            motif_out.push(motif_names[m].clone());
            auc_out.push(auc_value);
            nes_out.push(nes_value);
        }
    }

    let mut order: Vec<usize> = (0..auc_out.len()).collect();
    order.sort_by(|&a, &b| {
        compare_importance_desc(f64::from(auc_out[a]), f64::from(auc_out[b]), a, b)
    });
    let regulon_out = order.iter().map(|&i| regulon_out[i].clone()).collect();
    let motif_out = order.iter().map(|&i| motif_out[i].clone()).collect();
    let auc_out = order.iter().map(|&i| auc_out[i]).collect();
    let nes_out = order.iter().map(|&i| nes_out[i]).collect();

    CistargetRows {
        regulon: regulon_out,
        motif: motif_out,
        auc: auc_out,
        nes: nes_out,
        zero_var_indices,
        too_few_motifs,
    }
}

fn cistarget_rows_to_py<'py>(
    py: Python<'py>,
    rows: CistargetRows,
) -> PyResult<CistargetEnrichmentRowsPyResult> {
    Ok((
        PyList::new(py, rows.regulon.iter().map(|s| s.as_str()))?.unbind(),
        PyList::new(py, rows.motif.iter().map(|s| s.as_str()))?.unbind(),
        PyArray1::from_vec(py, rows.auc).unbind(),
        PyArray1::from_vec(py, rows.nes).unbind(),
        PyArray1::from_vec(py, rows.zero_var_indices).unbind(),
        rows.too_few_motifs,
    ))
}

/// Fit online-VB Latent Dirichlet Allocation on a sparse (docs x words) matrix.
/// Intended for pycisTopic-style scATAC peak-topic modeling; also works for
/// plain word-topic LDA.
///
/// Returns (cell_topic, topic_word) as two numpy arrays.
#[pyfunction]
#[pyo3(signature = (
    row_ptr, col_idx, counts, n_words,
    n_topics = 50,
    alpha = 0.02,
    eta = 0.02,
    tau0 = 64.0,
    kappa = 0.7,
    batch_size = 256,
    n_passes = 10,
    seed = 42,
))]
#[allow(clippy::too_many_arguments)]
fn topics_fit<'py>(
    py: Python<'py>,
    row_ptr: &Bound<'py, PyAny>,
    col_idx: PyReadonlyArray1<'py, i32>,
    counts: PyReadonlyArray1<'py, f32>,
    n_words: usize,
    n_topics: usize,
    alpha: f32,
    eta: f32,
    tau0: f32,
    kappa: f32,
    batch_size: usize,
    n_passes: usize,
    seed: u64,
) -> PyResult<TopicFitPyResult> {
    let col_idx = col_idx
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("col_idx must be contiguous"))?;
    let counts = counts
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("counts must be contiguous"))?;
    let row_ptr = row_ptr_any_to_usize(row_ptr, col_idx.len(), counts.len())?;
    validate_topic_csr(&row_ptr, col_idx, counts, n_words, false)?;
    let result = py.allow_threads(|| {
        online_vb_lda(
            &row_ptr, col_idx, counts, n_words, n_topics, alpha, eta, tau0, kappa, batch_size,
            n_passes, seed,
        )
    });
    let n_docs = result.n_docs;
    let ct = ndarray::Array2::from_shape_vec((n_docs, n_topics), result.cell_topic).map_err(
        |e: ndarray::ShapeError| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()),
    )?;
    let tw = ndarray::Array2::from_shape_vec((n_topics, n_words), result.topic_word).map_err(
        |e: ndarray::ShapeError| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()),
    )?;
    Ok((
        PyArray2::from_owned_array(py, ct).unbind(),
        PyArray2::from_owned_array(py, tw).unbind(),
    ))
}

/// Fit collapsed-Gibbs LDA on a sparse (docs x words) matrix.
///
/// The Mallet-class topic model - better topic-coherence (NPMI) on sparse
/// scATAC at K ≥ 30 than online VB, at the cost of thousands of
/// iterations instead of tens of passes. Returns (theta, beta) as two
/// numpy arrays of shape (n_docs, n_topics) and (n_topics, n_words).
///
/// If `n_threads <= 1`, runs the bit-deterministic serial path. If
/// `n_threads > 1`, dispatches to `fit_par` (AD-LDA, Newman et al.
/// 2009) which partitions docs across threads for near-linear speedup
/// at the cost of small cross-thread staleness within a sweep.
#[pyfunction]
#[pyo3(signature = (
    row_ptr, col_idx, counts, n_words,
    n_topics = 50,
    alpha = 0.1,
    eta = 0.01,
    n_iters = 200,
    seed = 42,
    n_threads = 1,
))]
#[allow(clippy::too_many_arguments)]
fn topics_fit_gibbs<'py>(
    py: Python<'py>,
    row_ptr: &Bound<'py, PyAny>,
    col_idx: PyReadonlyArray1<'py, i32>,
    counts: PyReadonlyArray1<'py, f32>,
    n_words: usize,
    n_topics: usize,
    alpha: f32,
    eta: f32,
    n_iters: usize,
    seed: u64,
    n_threads: usize,
) -> PyResult<TopicFitPyResult> {
    let col_idx = col_idx
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("col_idx must be contiguous"))?;
    let counts = counts
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("counts must be contiguous"))?;
    let row_ptr = row_ptr_any_to_usize(row_ptr, col_idx.len(), counts.len())?;
    validate_topic_csr(&row_ptr, col_idx, counts, n_words, true)?;
    let result = py.allow_threads(|| {
        if n_threads <= 1 {
            rustscenic_topics::gibbs::fit(
                &row_ptr, col_idx, counts, n_words, n_topics, alpha, eta, n_iters, seed,
            )
        } else {
            rustscenic_topics::gibbs::fit_par(
                &row_ptr, col_idx, counts, n_words, n_topics, alpha, eta, n_iters, seed, n_threads,
            )
        }
    });
    let n_docs = row_ptr.len().saturating_sub(1);
    let theta = ndarray::Array2::from_shape_vec((n_docs, n_topics), result.theta).map_err(
        |e: ndarray::ShapeError| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()),
    )?;
    let beta = ndarray::Array2::from_shape_vec((n_topics, n_words), result.beta).map_err(
        |e: ndarray::ShapeError| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()),
    )?;
    Ok((
        PyArray2::from_owned_array(py, theta).unbind(),
        PyArray2::from_owned_array(py, beta).unbind(),
    ))
}

/// Topic-coherence NPMI per topic.
///
/// Mean pairwise NPMI over the top-N words of each topic against the
/// supplied corpus's document containment. Higher = better. The standard
/// intrinsic topic-quality metric for LDA, used to compare rustscenic's
/// online VB vs collapsed Gibbs vs Mallet on the same corpus.
#[pyfunction]
#[pyo3(signature = (topic_word, n_topics, n_words, row_ptr, col_idx, top_n = 10))]
fn topics_npmi<'py>(
    py: Python<'py>,
    topic_word: PyReadonlyArray2<'py, f32>,
    n_topics: usize,
    n_words: usize,
    row_ptr: &Bound<'py, PyAny>,
    col_idx: PyReadonlyArray1<'py, i32>,
    top_n: usize,
) -> PyResult<Py<PyArray1<f32>>> {
    let arr = topic_word.as_array();
    if arr.shape() != [n_topics, n_words] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "topic_word shape {:?} mismatch (n_topics, n_words)=({}, {})",
            arr.shape(),
            n_topics,
            n_words,
        )));
    }
    let col_idx = col_idx
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("col_idx must be contiguous"))?;
    let row_ptr = row_ptr_any_to_usize(row_ptr, col_idx.len(), col_idx.len())?;
    validate_topic_csr(&row_ptr, col_idx, &[], n_words, false)?;
    let coh = py.allow_threads(|| topic_coherence_npmi_view(arr, top_n, &row_ptr, col_idx));
    Ok(PyArray1::from_vec(py, coh).unbind())
}

/// Argmax topic assignment per cell.
///
/// Python keeps pandas labels and nullable Series construction. Rust owns the
/// row-wise scan over the dense cells × topics matrix, including empty-row
/// detection and active-topic counting.
#[pyfunction]
fn topics_cell_assignment<'py>(
    py: Python<'py>,
    cell_topic: PyReadonlyArray2<'py, f32>,
) -> PyResult<(Py<PyArray1<i64>>, usize, usize)> {
    let arr = cell_topic.as_array();
    let (assignments, active_topics, empty_cells) =
        py.allow_threads(|| topics_cell_assignment_impl(arr));
    Ok((
        PyArray1::from_vec(py, assignments).unbind(),
        active_topics,
        empty_cells,
    ))
}

fn topics_cell_assignment_impl(
    cell_topic: ndarray::ArrayView2<'_, f32>,
) -> (Vec<i64>, usize, usize) {
    let n_cells = cell_topic.shape()[0];
    let n_topics = cell_topic.shape()[1];
    let mut assignments = vec![-1_i64; n_cells];
    let mut active = vec![false; n_topics];
    let mut empty_cells = 0usize;

    for cell_idx in 0..n_cells {
        let mut row_sum = 0.0_f32;
        let mut saw_nonfinite_sum = false;
        let mut best_idx: Option<usize> = None;
        let mut best_value = f32::NEG_INFINITY;

        for topic_idx in 0..n_topics {
            let value = cell_topic[(cell_idx, topic_idx)];
            if value.is_nan() {
                continue;
            }
            row_sum += value;
            if !value.is_finite() {
                saw_nonfinite_sum = true;
                continue;
            }
            if best_idx.is_none() || value > best_value {
                best_idx = Some(topic_idx);
                best_value = value;
            }
        }

        let Some(topic_idx) = best_idx else {
            empty_cells += 1;
            continue;
        };
        if saw_nonfinite_sum || !row_sum.is_finite() || row_sum <= 0.0 {
            empty_cells += 1;
            continue;
        }
        assignments[cell_idx] = topic_idx as i64;
        active[topic_idx] = true;
    }

    let active_topics = active.into_iter().filter(|&seen| seen).count();
    (assignments, active_topics, empty_cells)
}

#[pyfunction]
fn specificity_group_codes_with_order<'py>(
    py: Python<'py>,
    labels: Vec<String>,
    missing: PyReadonlyArray1<'py, bool>,
) -> PyResult<SpecificityGroupCodesOrderPyResult> {
    let missing = missing
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("missing must be contiguous"))?;
    let (codes, first_indices) = specificity_group_codes_order_impl(&labels, missing, None)?;
    Ok((
        PyArray1::from_vec(py, codes).unbind(),
        PyArray1::from_vec(py, first_indices).unbind(),
    ))
}

#[pyfunction]
fn specificity_group_codes_with_numeric_order<'py>(
    py: Python<'py>,
    labels: Vec<String>,
    missing: PyReadonlyArray1<'py, bool>,
    numeric_values: PyReadonlyArray1<'py, f64>,
) -> PyResult<SpecificityGroupCodesOrderPyResult> {
    let missing = missing
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("missing must be contiguous"))?;
    let numeric_values = numeric_values.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("numeric_values must be contiguous")
    })?;
    let (codes, first_indices) =
        specificity_group_codes_order_impl(&labels, missing, Some(numeric_values))?;
    Ok((
        PyArray1::from_vec(py, codes).unbind(),
        PyArray1::from_vec(py, first_indices).unbind(),
    ))
}

#[derive(Clone)]
struct SpecificityGroupEntry {
    label: String,
    first_idx: usize,
    numeric_value: f64,
}

fn specificity_group_codes_order_impl(
    labels: &[String],
    missing: &[bool],
    numeric_values: Option<&[f64]>,
) -> PyResult<(Vec<i32>, Vec<u64>)> {
    if labels.len() != missing.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "labels has length {} but missing has length {}",
            labels.len(),
            missing.len()
        )));
    }
    if let Some(values) = numeric_values {
        if values.len() != labels.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "numeric_values has length {} but labels has length {}",
                values.len(),
                labels.len()
            )));
        }
    }

    let mut group_by_label: HashMap<&str, usize> = HashMap::new();
    let mut entries: Vec<SpecificityGroupEntry> = Vec::new();
    let mut old_codes = Vec::with_capacity(labels.len());

    for (idx, (label, &is_missing)) in labels.iter().zip(missing).enumerate() {
        if is_missing {
            old_codes.push(-1);
            continue;
        }
        let group_idx = if let Some(&existing) = group_by_label.get(label.as_str()) {
            existing
        } else {
            let sort_value = numeric_values.map_or(f64::NAN, |values| values[idx]);
            let group_idx = entries.len();
            entries.push(SpecificityGroupEntry {
                label: label.clone(),
                first_idx: idx,
                numeric_value: sort_value,
            });
            group_by_label.insert(label.as_str(), group_idx);
            group_idx
        };
        old_codes.push(group_idx as i32);
    }

    let mut order: Vec<usize> = (0..entries.len()).collect();
    if numeric_values.is_some() {
        order.sort_by(|&left, &right| {
            entries[left]
                .numeric_value
                .total_cmp(&entries[right].numeric_value)
                .then_with(|| entries[left].label.cmp(&entries[right].label))
                .then_with(|| entries[left].first_idx.cmp(&entries[right].first_idx))
        });
    } else {
        order.sort_by(|&left, &right| {
            entries[left]
                .label
                .cmp(&entries[right].label)
                .then_with(|| entries[left].first_idx.cmp(&entries[right].first_idx))
        });
    }

    let mut remap = vec![0_i32; entries.len()];
    let mut first_indices = Vec::with_capacity(entries.len());
    for (new_idx, &old_idx) in order.iter().enumerate() {
        remap[old_idx] = new_idx as i32;
        first_indices.push(entries[old_idx].first_idx as u64);
    }

    let codes = old_codes
        .into_iter()
        .map(|code| if code < 0 { -1 } else { remap[code as usize] })
        .collect();
    Ok((codes, first_indices))
}

/// Regulon specificity scores (RSS) for a cells × regulons AUCell matrix.
///
/// Python owns pandas label alignment and output DataFrame construction. Rust
/// owns group-code construction and the numeric Jensen-Shannon loop over
/// groups × regulons × cells.
macro_rules! specificity_rss_pyfunction {
    ($name:ident, $value_ty:ty) => {
        #[pyfunction]
        fn $name<'py>(
            py: Python<'py>,
            auc: PyReadonlyArray2<'py, $value_ty>,
            group_codes: PyReadonlyArray1<'py, i32>,
            n_groups: usize,
        ) -> PyResult<SpecificityRssPyResult> {
            let auc_arr = auc.as_array();
            let n_cells = auc_arr.shape()[0];
            let n_regulons = auc_arr.shape()[1];
            let group_codes = group_codes.as_slice().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err("group_codes must be contiguous")
            })?;
            if group_codes.len() != n_cells {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "group_codes has length {} but auc has {} cells",
                    group_codes.len(),
                    n_cells
                )));
            }

            let rss = py.allow_threads(|| {
                specificity_rss_impl(auc_arr, n_cells, n_regulons, group_codes, n_groups)
            });
            let arr = ndarray::Array2::from_shape_vec((n_groups, n_regulons), rss).map_err(
                |e: ndarray::ShapeError| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()),
            )?;
            Ok(PyArray2::from_owned_array(py, arr).unbind())
        }
    };
}

specificity_rss_pyfunction!(specificity_rss_f32, f32);
specificity_rss_pyfunction!(specificity_rss, f64);

fn specificity_rss_impl<T: Copy + Into<f64> + Sync>(
    auc: ndarray::ArrayView2<'_, T>,
    n_cells: usize,
    n_regulons: usize,
    group_codes: &[i32],
    n_groups: usize,
) -> Vec<f64> {
    let mut col_sums = vec![0.0_f64; n_regulons];
    for cell in 0..n_cells {
        for regulon in 0..n_regulons {
            col_sums[regulon] += auc[(cell, regulon)].into();
        }
    }
    for sum in &mut col_sums {
        if *sum == 0.0 {
            *sum = 1.0;
        }
    }

    let group_counts = group_counts(group_codes, n_groups);
    let mut rss = vec![0.0_f64; n_groups * n_regulons];
    rss.par_chunks_mut(n_regulons)
        .enumerate()
        .for_each(|(group_idx, group_row)| {
            let n_in = group_counts[group_idx];
            if n_in == 0 {
                return;
            }
            let p_in = 1.0 / n_in as f64;
            for r in 0..n_regulons {
                let mut kl_pm = 0.0_f64;
                let mut kl_qm = 0.0_f64;
                for cell in 0..n_cells {
                    let p = if group_codes[cell] == group_idx as i32 {
                        p_in
                    } else {
                        0.0
                    };
                    let q = auc[(cell, r)].into() / col_sums[r];
                    let m = 0.5 * (p + q);
                    if p > 0.0 && m > 0.0 {
                        kl_pm += p * (p / m).log2();
                    }
                    if q > 0.0 && m > 0.0 {
                        kl_qm += q * (q / m).log2();
                    }
                }
                let js = 0.5 * (kl_pm + kl_qm);
                group_row[r] = 1.0 - js.max(0.0).sqrt();
            }
        });
    rss
}

/// Top candidate enhancer indices per topic.
///
/// Python keeps topic/peak labels and result shape. Rust owns the per-topic
/// descending weight sort, which is the only compute-heavy part.
macro_rules! candidate_top_indices_pyfunction {
    ($name:ident, $value_ty:ty) => {
        #[pyfunction]
        fn $name<'py>(
            py: Python<'py>,
            topic_peak: PyReadonlyArray2<'py, $value_ty>,
            top_n: usize,
        ) -> PyResult<Py<PyArray2<u64>>> {
            let arr = topic_peak.as_array();
            candidate_top_indices_impl(py, arr, top_n)
        }
    };
}

candidate_top_indices_pyfunction!(specificity_candidate_top_indices_f32, f32);
candidate_top_indices_pyfunction!(specificity_candidate_top_indices, f64);

fn candidate_top_indices_impl<T: Copy + Into<f64> + Sync>(
    py: Python<'_>,
    arr: ndarray::ArrayView2<'_, T>,
    top_n: usize,
) -> PyResult<Py<PyArray2<u64>>> {
    let n_topics = arr.shape()[0];
    let n_peaks = arr.shape()[1];
    let k = top_n.min(n_peaks);
    let rows: Vec<Vec<u64>> = py.allow_threads(|| {
        (0..n_topics)
            .into_par_iter()
            .map(|topic_idx| {
                let row = arr.row(topic_idx);
                let mut order: Vec<usize> = (0..n_peaks).collect();
                order.sort_unstable_by(|&a, &b| {
                    let left: f64 = row[b].into();
                    let right: f64 = row[a].into();
                    left.total_cmp(&right).then_with(|| a.cmp(&b))
                });
                order.truncate(k);
                order.into_iter().map(|idx| idx as u64).collect()
            })
            .collect()
    });
    let mut flat = Vec::with_capacity(n_topics * k);
    for row in rows {
        flat.extend(row);
    }
    let out = ndarray::Array2::from_shape_vec((n_topics, k), flat).map_err(
        |e: ndarray::ShapeError| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()),
    )?;
    Ok(PyArray2::from_owned_array(py, out).unbind())
}

/// Parse peak identifiers such as `chr1:100-200` into coordinate arrays.
///
/// The Python API keeps DataFrame assembly and AnnData metadata handling. Rust
/// owns the linear parse/validation loop so large ATAC `var_names` do not pay
/// a Python regex cost before enhancer linking starts.
#[pyfunction]
fn enhancer_parse_peak_names<'py>(
    py: Python<'py>,
    names: Vec<String>,
) -> PyResult<Option<EnhancerParsePeakNamesPyResult>> {
    let parsed = py.allow_threads(|| parse_peak_names_impl(&names));
    let Some((chroms, starts, ends)) = parsed else {
        return Ok(None);
    };
    Ok(Some((
        PyList::new(py, chroms.iter().map(|s| s.as_str()))?.unbind(),
        PyArray1::from_vec(py, starts).unbind(),
        PyArray1::from_vec(py, ends).unbind(),
    )))
}

/// Normalise gene/peak chromosome names and assign shared integer codes.
#[pyfunction]
fn enhancer_normalise_chrom_codes<'py>(
    py: Python<'py>,
    gene_chroms: Vec<String>,
    peak_chroms: Vec<String>,
) -> PyResult<EnhancerChromCodesPyResult> {
    let (gene_norm, peak_norm, gene_codes, peak_codes) = py.allow_threads(|| {
        let gene_norm: Vec<String> = gene_chroms
            .iter()
            .map(|name| normalise_chrom_name(name))
            .collect();
        let peak_norm: Vec<String> = peak_chroms
            .iter()
            .map(|name| normalise_chrom_name(name))
            .collect();

        let mut chroms: Vec<String> = gene_norm
            .iter()
            .chain(peak_norm.iter())
            .cloned()
            .collect::<HashSet<String>>()
            .into_iter()
            .collect();
        chroms.sort_unstable();
        let code_by_chrom: HashMap<String, i32> = chroms
            .into_iter()
            .enumerate()
            .map(|(idx, chrom)| (chrom, idx as i32))
            .collect();
        let gene_codes = gene_norm
            .iter()
            .map(|chrom| code_by_chrom[chrom])
            .collect::<Vec<i32>>();
        let peak_codes = peak_norm
            .iter()
            .map(|chrom| code_by_chrom[chrom])
            .collect::<Vec<i32>>();
        (gene_norm, peak_norm, gene_codes, peak_codes)
    });

    Ok((
        PyList::new(py, gene_norm.iter().map(|s| s.as_str()))?.unbind(),
        PyList::new(py, peak_norm.iter().map(|s| s.as_str()))?.unbind(),
        PyArray1::from_vec(py, gene_codes).unbind(),
        PyArray1::from_vec(py, peak_codes).unbind(),
    ))
}

fn normalise_chrom_name(name: &str) -> String {
    let stripped = name.trim();
    let stripped = if stripped.len() >= 3 && stripped[..3].eq_ignore_ascii_case("chr") {
        &stripped[3..]
    } else {
        stripped
    };
    let upper = stripped.to_ascii_uppercase();
    if upper == "M" {
        "MT".to_owned()
    } else {
        upper
    }
}

/// Return matched RNA/ATAC cell indices in RNA barcode order.
#[pyfunction]
fn enhancer_align_cell_indices<'py>(
    py: Python<'py>,
    rna_cells: Vec<String>,
    atac_cells: Vec<String>,
) -> PyResult<EnhancerAlignCellIndicesPyResult> {
    let (rna_indices, atac_indices) = py.allow_threads(|| {
        let mut atac_index_by_cell: HashMap<&str, usize> = HashMap::with_capacity(atac_cells.len());
        for (idx, cell) in atac_cells.iter().enumerate() {
            atac_index_by_cell.entry(cell.as_str()).or_insert(idx);
        }

        let mut rna_indices = Vec::new();
        let mut atac_indices = Vec::new();
        for (idx, cell) in rna_cells.iter().enumerate() {
            if let Some(&atac_idx) = atac_index_by_cell.get(cell.as_str()) {
                rna_indices.push(idx as u64);
                atac_indices.push(atac_idx as u64);
            }
        }
        (rna_indices, atac_indices)
    });

    Ok((
        PyArray1::from_vec(py, rna_indices).unbind(),
        PyArray1::from_vec(py, atac_indices).unbind(),
    ))
}

fn parse_peak_names_impl(names: &[String]) -> Option<(Vec<String>, Vec<i64>, Vec<i64>)> {
    let mut chroms = Vec::with_capacity(names.len());
    let mut starts = Vec::with_capacity(names.len());
    let mut ends = Vec::with_capacity(names.len());

    for name in names {
        let (chrom, start, end) = parse_peak_name(name)?;
        chroms.push(chrom.to_owned());
        starts.push(start);
        ends.push(end);
    }

    Some((chroms, starts, ends))
}

fn parse_peak_name(name: &str) -> Option<(&str, i64, i64)> {
    let bytes = name.as_bytes();
    if bytes.is_empty() {
        return None;
    }

    let mut end_start = bytes.len();
    while end_start > 0 && bytes[end_start - 1].is_ascii_digit() {
        end_start -= 1;
    }
    if end_start == bytes.len() || end_start == 0 || !is_peak_separator(bytes[end_start - 1]) {
        return None;
    }

    let start_end = end_start - 1;
    let mut start_start = start_end;
    while start_start > 0 && bytes[start_start - 1].is_ascii_digit() {
        start_start -= 1;
    }
    if start_start == start_end || start_start == 0 || !is_peak_separator(bytes[start_start - 1]) {
        return None;
    }

    let chrom = &name[..start_start - 1];
    if !valid_peak_chrom(chrom) {
        return None;
    }

    let start = name[start_start..start_end].parse::<i64>().ok()?;
    let end = name[end_start..].parse::<i64>().ok()?;
    if start >= end {
        return None;
    }

    Some((chrom, start, end))
}

fn is_peak_separator(byte: u8) -> bool {
    matches!(byte, b':' | b'-' | b'_')
}

fn valid_peak_chrom(chrom: &str) -> bool {
    let mut bytes = chrom.bytes();
    let Some(first) = bytes.next() else {
        return false;
    };
    if !first.is_ascii_alphanumeric() {
        return false;
    }
    bytes.all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'_')
}

/// Match gene-coordinate rows to RNA feature columns.
///
/// Mirrors the previous Python dict behaviour: duplicate RNA gene names map
/// to the last feature column. Gene-coordinate row order is preserved.
#[pyfunction]
fn enhancer_match_gene_coords_to_rna<'py>(
    py: Python<'py>,
    rna_gene_names: Vec<String>,
    gene_coord_names: Vec<String>,
) -> PyResult<EnhancerGeneMatchPyResult> {
    let (row_indices, source_cols) = py.allow_threads(|| {
        let mut rna_col_by_gene: HashMap<&str, usize> =
            HashMap::with_capacity(rna_gene_names.len());
        for (idx, gene) in rna_gene_names.iter().enumerate() {
            rna_col_by_gene.insert(gene.as_str(), idx);
        }

        let mut row_indices = Vec::new();
        let mut source_cols = Vec::new();
        for (row_idx, gene) in gene_coord_names.iter().enumerate() {
            if let Some(&source_col) = rna_col_by_gene.get(gene.as_str()) {
                row_indices.push(row_idx as u64);
                source_cols.push(source_col as i64);
            }
        }
        (row_indices, source_cols)
    });

    Ok((
        PyArray1::from_vec(py, row_indices).unbind(),
        PyArray1::from_vec(py, source_cols).unbind(),
    ))
}

/// Match peak-coordinate rows to ATAC peak columns in ATAC peak order.
///
/// Duplicate peak-coordinate names map to the last row, matching
/// [`enhancer_match_gene_coords_to_rna`] duplicate-name semantics.
#[pyfunction]
fn enhancer_match_peak_coords_to_atac<'py>(
    py: Python<'py>,
    atac_peak_names: Vec<String>,
    peak_coord_names: Vec<String>,
) -> PyResult<EnhancerPeakCoordMatchPyResult> {
    let (row_indices, missing) = py.allow_threads(|| {
        let mut coord_row_by_peak: HashMap<&str, usize> =
            HashMap::with_capacity(peak_coord_names.len());
        for (idx, peak) in peak_coord_names.iter().enumerate() {
            coord_row_by_peak.insert(peak.as_str(), idx);
        }

        let mut row_indices = Vec::with_capacity(atac_peak_names.len());
        let mut missing = Vec::new();
        for peak in &atac_peak_names {
            if let Some(&row_idx) = coord_row_by_peak.get(peak.as_str()) {
                row_indices.push(row_idx as u64);
            } else {
                missing.push(peak.clone());
            }
        }
        (row_indices, missing)
    });

    Ok((
        PyArray1::from_vec(py, row_indices).unbind(),
        PyList::new(py, missing.iter().map(|s| s.as_str()))?.unbind(),
    ))
}

/// Prepare sorted gene coordinate arrays for enhancer linking.
///
/// Python owns pandas metadata extraction. Rust owns the scale-sensitive
/// lexicographic sort, peak/gene chromosome overlap check, and unique RNA
/// column count used by the dense-materialisation warning path.
#[pyfunction]
fn enhancer_prepare_gene_order<'py>(
    py: Python<'py>,
    peak_chrom_codes: PyReadonlyArray1<'py, i32>,
    gene_chrom_codes: PyReadonlyArray1<'py, i32>,
    gene_tss: PyReadonlyArray1<'py, i64>,
    gene_source_cols: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Py<PyArray1<u64>>, bool, usize)> {
    let peak_chrom_codes = peak_chrom_codes.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("peak_chrom_codes must be contiguous")
    })?;
    let gene_chrom_codes = gene_chrom_codes.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("gene_chrom_codes must be contiguous")
    })?;
    let gene_tss = gene_tss
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("gene_tss must be contiguous"))?;
    let gene_source_cols = gene_source_cols.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("gene_source_cols must be contiguous")
    })?;

    if gene_chrom_codes.len() != gene_tss.len() || gene_tss.len() != gene_source_cols.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "gene arrays must have the same length; got chrom_codes={}, tss={}, source_cols={}",
            gene_chrom_codes.len(),
            gene_tss.len(),
            gene_source_cols.len()
        )));
    }
    if let Some((i, &col)) = gene_source_cols
        .iter()
        .enumerate()
        .find(|(_, &col)| col < 0 || col > u32::MAX as i64)
    {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "gene_source_cols[{i}] = {col} is outside uint32 range"
        )));
    }

    let (order, has_overlap, n_unique_source_cols) = py.allow_threads(|| {
        let mut order: Vec<usize> = (0..gene_chrom_codes.len()).collect();
        order.sort_by(|&a, &b| {
            gene_chrom_codes[a]
                .cmp(&gene_chrom_codes[b])
                .then_with(|| gene_tss[a].cmp(&gene_tss[b]))
                .then_with(|| a.cmp(&b))
        });

        let gene_chrom_set: HashSet<i32> = gene_chrom_codes.iter().copied().collect();
        let has_overlap = peak_chrom_codes
            .iter()
            .any(|code| gene_chrom_set.contains(code));
        let n_unique_source_cols = gene_source_cols
            .iter()
            .copied()
            .collect::<HashSet<i64>>()
            .len();
        (order, has_overlap, n_unique_source_cols)
    });

    Ok((
        PyArray1::from_vec(py, order.into_iter().map(|idx| idx as u64).collect()).unbind(),
        has_overlap,
        n_unique_source_cols,
    ))
}

fn group_counts(group_codes: &[i32], n_groups: usize) -> Vec<usize> {
    let mut counts = vec![0usize; n_groups];
    for &code in group_codes {
        if code >= 0 {
            let code = code as usize;
            if code < n_groups {
                counts[code] += 1;
            }
        }
    }
    counts
}

/// Pearson enhancer-to-gene linking core for the sparse-ATAC path.
///
/// Python prepares AnnData/pandas metadata and passes:
/// - dense, C-contiguous RNA expression (cells x genes)
/// - ATAC CSC arrays for cells x peaks, using SciPy-native signed indices
/// - peak chromosome codes and start/end coordinates
/// - sorted gene chromosome codes, TSS positions, and RNA column indices
///
/// Returns row indices into the Python peak and sorted-gene frames, plus the
/// signed distance and correlation. This keeps the heavy per-pair loop out of
/// Python while preserving the public DataFrame contract.
#[pyfunction]
#[pyo3(signature = (
    rna,
    atac_indptr,
    atac_indices,
    atac_data,
    peak_chrom_codes,
    peak_starts,
    peak_ends,
    gene_chrom_codes,
    gene_tss,
    gene_rna_cols,
    max_distance,
    min_abs_corr,
    atac_peak_cols = None,
))]
#[allow(clippy::too_many_arguments)]
fn enhancer_link_pearson<'py>(
    py: Python<'py>,
    rna: PyReadonlyArray2<'py, f32>,
    atac_indptr: &Bound<'py, PyAny>,
    atac_indices: PyReadonlyArray1<'py, i32>,
    atac_data: PyReadonlyArray1<'py, f32>,
    peak_chrom_codes: PyReadonlyArray1<'py, i32>,
    peak_starts: PyReadonlyArray1<'py, i64>,
    peak_ends: PyReadonlyArray1<'py, i64>,
    gene_chrom_codes: PyReadonlyArray1<'py, i32>,
    gene_tss: PyReadonlyArray1<'py, i64>,
    gene_rna_cols: PyReadonlyArray1<'py, u32>,
    max_distance: i64,
    min_abs_corr: f32,
    atac_peak_cols: Option<PyReadonlyArray1<'py, u32>>,
) -> PyResult<EnhancerPearsonPyResult> {
    let rna_arr = rna.as_array();
    let n_cells = rna_arr.shape()[0];
    let n_genes = rna_arr.shape()[1];

    let indices = atac_indices
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("atac_indices must be contiguous"))?;
    let data = atac_data
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("atac_data must be contiguous"))?;
    let indptr = row_ptr_any_to_usize(atac_indptr, indices.len(), data.len())?;
    let peak_chrom_codes = peak_chrom_codes.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("peak_chrom_codes must be contiguous")
    })?;
    let peak_starts = peak_starts
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("peak_starts must be contiguous"))?;
    let peak_ends = peak_ends
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("peak_ends must be contiguous"))?;
    let gene_chrom_codes = gene_chrom_codes.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("gene_chrom_codes must be contiguous")
    })?;
    let gene_tss = gene_tss
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("gene_tss must be contiguous"))?;
    let gene_rna_cols = gene_rna_cols
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("gene_rna_cols must be contiguous"))?;
    let atac_peak_cols = match &atac_peak_cols {
        Some(cols) => Some(cols.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("atac_peak_cols must be contiguous")
        })?),
        None => None,
    };

    validate_enhancer_pearson_inputs(
        n_cells,
        n_genes,
        &indptr,
        indices,
        data,
        peak_chrom_codes,
        peak_starts,
        peak_ends,
        gene_chrom_codes,
        gene_tss,
        gene_rna_cols,
        atac_peak_cols,
    )?;

    let mut links = py.allow_threads(|| {
        if let Some(rna_slice) = rna_arr.as_slice() {
            compute_enhancer_pearson_links(
                rna_slice,
                n_cells,
                n_genes,
                &indptr,
                indices,
                data,
                peak_chrom_codes,
                peak_starts,
                peak_ends,
                gene_chrom_codes,
                gene_tss,
                gene_rna_cols,
                atac_peak_cols,
                max_distance,
                min_abs_corr,
            )
        } else {
            compute_enhancer_pearson_links_view(
                rna_arr,
                &indptr,
                indices,
                data,
                peak_chrom_codes,
                peak_starts,
                peak_ends,
                gene_chrom_codes,
                gene_tss,
                gene_rna_cols,
                atac_peak_cols,
                max_distance,
                min_abs_corr,
            )
        }
    });
    sort_enhancer_links_by_abs_corr(&mut links);

    let mut peak_ix = Vec::with_capacity(links.len());
    let mut gene_ix = Vec::with_capacity(links.len());
    let mut distances = Vec::with_capacity(links.len());
    let mut correlations = Vec::with_capacity(links.len());
    for link in links {
        peak_ix.push(link.peak_idx);
        gene_ix.push(link.gene_idx);
        distances.push(link.distance);
        correlations.push(link.correlation);
    }
    Ok((
        PyArray1::from_vec(py, peak_ix).unbind(),
        PyArray1::from_vec(py, gene_ix).unbind(),
        PyArray1::from_vec(py, distances).unbind(),
        PyArray1::from_vec(py, correlations).unbind(),
    ))
}

/// Pearson enhancer-to-gene linking for sparse ATAC and sparse RNA.
///
/// Computes the same Pearson correlation over implicit zeros as
/// [`enhancer_link_pearson`] without requiring Python to materialise dense RNA
/// blocks. RNA and ATAC are both CSC cells × features.
#[pyfunction]
#[pyo3(signature = (
    rna_indptr,
    rna_indices,
    rna_data,
    n_cells,
    n_rna_genes,
    atac_indptr,
    atac_indices,
    atac_data,
    peak_chrom_codes,
    peak_starts,
    peak_ends,
    gene_chrom_codes,
    gene_tss,
    gene_rna_cols,
    max_distance,
    min_abs_corr,
    atac_peak_cols = None,
))]
#[allow(clippy::too_many_arguments)]
fn enhancer_link_pearson_sparse_rna<'py>(
    py: Python<'py>,
    rna_indptr: &Bound<'py, PyAny>,
    rna_indices: PyReadonlyArray1<'py, i32>,
    rna_data: PyReadonlyArray1<'py, f32>,
    n_cells: usize,
    n_rna_genes: usize,
    atac_indptr: &Bound<'py, PyAny>,
    atac_indices: PyReadonlyArray1<'py, i32>,
    atac_data: PyReadonlyArray1<'py, f32>,
    peak_chrom_codes: PyReadonlyArray1<'py, i32>,
    peak_starts: PyReadonlyArray1<'py, i64>,
    peak_ends: PyReadonlyArray1<'py, i64>,
    gene_chrom_codes: PyReadonlyArray1<'py, i32>,
    gene_tss: PyReadonlyArray1<'py, i64>,
    gene_rna_cols: PyReadonlyArray1<'py, u32>,
    max_distance: i64,
    min_abs_corr: f32,
    atac_peak_cols: Option<PyReadonlyArray1<'py, u32>>,
) -> PyResult<EnhancerPearsonPyResult> {
    let rna_indices = rna_indices
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("rna_indices must be contiguous"))?;
    let rna_data = rna_data
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("rna_data must be contiguous"))?;
    let indices = atac_indices
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("atac_indices must be contiguous"))?;
    let data = atac_data
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("atac_data must be contiguous"))?;
    let rna_indptr = row_ptr_any_to_usize(rna_indptr, rna_indices.len(), rna_data.len())?;
    let indptr = row_ptr_any_to_usize(atac_indptr, indices.len(), data.len())?;
    let peak_chrom_codes = peak_chrom_codes.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("peak_chrom_codes must be contiguous")
    })?;
    let peak_starts = peak_starts
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("peak_starts must be contiguous"))?;
    let peak_ends = peak_ends
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("peak_ends must be contiguous"))?;
    let gene_chrom_codes = gene_chrom_codes.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("gene_chrom_codes must be contiguous")
    })?;
    let gene_tss = gene_tss
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("gene_tss must be contiguous"))?;
    let gene_rna_cols = gene_rna_cols
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("gene_rna_cols must be contiguous"))?;
    let atac_peak_cols = match &atac_peak_cols {
        Some(cols) => Some(cols.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("atac_peak_cols must be contiguous")
        })?),
        None => None,
    };

    validate_sparse_csc(
        "rna",
        n_cells,
        n_rna_genes,
        &rna_indptr,
        rna_indices,
        rna_data,
    )?;
    validate_enhancer_pearson_inputs(
        n_cells,
        n_rna_genes,
        &indptr,
        indices,
        data,
        peak_chrom_codes,
        peak_starts,
        peak_ends,
        gene_chrom_codes,
        gene_tss,
        gene_rna_cols,
        atac_peak_cols,
    )?;

    let mut links = py.allow_threads(|| {
        compute_enhancer_pearson_links_sparse_rna(
            n_cells,
            &rna_indptr,
            rna_indices,
            rna_data,
            &indptr,
            indices,
            data,
            peak_chrom_codes,
            peak_starts,
            peak_ends,
            gene_chrom_codes,
            gene_tss,
            gene_rna_cols,
            atac_peak_cols,
            max_distance,
            min_abs_corr,
        )
    });
    sort_enhancer_links_by_abs_corr(&mut links);

    let mut peak_ix = Vec::with_capacity(links.len());
    let mut gene_ix = Vec::with_capacity(links.len());
    let mut distances = Vec::with_capacity(links.len());
    let mut correlations = Vec::with_capacity(links.len());
    for link in links {
        peak_ix.push(link.peak_idx);
        gene_ix.push(link.gene_idx);
        distances.push(link.distance);
        correlations.push(link.correlation);
    }
    Ok((
        PyArray1::from_vec(py, peak_ix).unbind(),
        PyArray1::from_vec(py, gene_ix).unbind(),
        PyArray1::from_vec(py, distances).unbind(),
        PyArray1::from_vec(py, correlations).unbind(),
    ))
}

fn validate_sparse_csc(
    label: &str,
    n_rows: usize,
    n_cols: usize,
    indptr: &[usize],
    indices: &[i32],
    data: &[f32],
) -> PyResult<()> {
    if indptr.len() != n_cols + 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{label}_indptr has length {} but expected n_cols + 1 = {}",
            indptr.len(),
            n_cols + 1
        )));
    }
    if indices.len() != data.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{label}_indices length {} differs from {label}_data length {}",
            indices.len(),
            data.len()
        )));
    }
    let mut prev_ptr = 0usize;
    for (col, &ptr) in indptr.iter().enumerate() {
        if ptr < prev_ptr {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{label}_indptr must be non-decreasing; indptr[{col}] = {ptr} < {prev_ptr}"
            )));
        }
        if ptr > indices.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{label}_indptr[{col}] = {ptr} exceeds indices length {}",
                indices.len()
            )));
        }
        if col > 0 {
            let start = prev_ptr;
            let end = ptr;
            let mut prev_row: Option<usize> = None;
            for (offset, &raw_row) in indices[start..end].iter().enumerate() {
                let pos = start + offset;
                let row = usize::try_from(raw_row).map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "{label}_indices[{pos}] = {raw_row} must be non-negative"
                    ))
                })?;
                if row >= n_rows {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "{label}_indices[{pos}] = {row} exceeds n_rows = {n_rows}"
                    )));
                }
                if let Some(prev) = prev_row {
                    if row <= prev {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "{label} CSC columns must have strictly increasing row indices"
                        )));
                    }
                }
                prev_row = Some(row);
            }
        }
        prev_ptr = ptr;
    }
    if prev_ptr != indices.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{label}_indptr[-1] = {prev_ptr} but indices/data length is {}",
            indices.len()
        )));
    }
    if let Some((i, &value)) = data
        .iter()
        .enumerate()
        .find(|(_, &value)| !value.is_finite())
    {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{label}_data contains NaN or Inf at position {i}: {value}"
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_enhancer_pearson_inputs(
    n_cells: usize,
    n_genes: usize,
    indptr: &[usize],
    indices: &[i32],
    data: &[f32],
    peak_chrom_codes: &[i32],
    peak_starts: &[i64],
    peak_ends: &[i64],
    gene_chrom_codes: &[i32],
    gene_tss: &[i64],
    gene_rna_cols: &[u32],
    atac_peak_cols: Option<&[u32]>,
) -> PyResult<()> {
    let n_peaks = peak_chrom_codes.len();
    if peak_starts.len() != n_peaks {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "peak_starts has length {} but peak_chrom_codes has length {}",
            peak_starts.len(),
            n_peaks
        )));
    }
    if peak_ends.len() != n_peaks {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "peak_ends has length {} but peak_chrom_codes has length {}",
            peak_ends.len(),
            n_peaks
        )));
    }
    if let Some((i, (&start, &end))) = peak_starts
        .iter()
        .zip(peak_ends)
        .enumerate()
        .find(|(_, (&start, &end))| start > end)
    {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "peak_starts[{i}] = {start} exceeds peak_ends[{i}] = {end}"
        )));
    }
    if let Some(cols) = atac_peak_cols {
        if cols.len() != n_peaks {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "atac_peak_cols has length {} but peak_chrom_codes has length {}",
                cols.len(),
                n_peaks
            )));
        }
        let n_atac_cols = indptr.len().saturating_sub(1);
        if let Some((i, &col)) = cols
            .iter()
            .enumerate()
            .find(|(_, &col)| (col as usize) >= n_atac_cols)
        {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "atac_peak_cols[{i}] = {col} exceeds ATAC n_cols = {n_atac_cols}"
            )));
        }
    } else if indptr.len() != n_peaks + 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "atac_indptr has length {} but expected n_peaks + 1 = {}",
            indptr.len(),
            n_peaks + 1
        )));
    }
    if indices.len() != data.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "atac_indices length {} differs from atac_data length {}",
            indices.len(),
            data.len()
        )));
    }
    let mut prev = 0usize;
    for (i, &v) in indptr.iter().enumerate() {
        if v < prev {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "atac_indptr must be non-decreasing; indptr[{i}] = {v} < {prev}"
            )));
        }
        if v > indices.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "atac_indptr[{i}] = {v} exceeds indices length {}",
                indices.len()
            )));
        }
        prev = v;
    }
    if prev != indices.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "atac_indptr[-1] = {prev} but indices/data length is {}",
            indices.len()
        )));
    }
    for (i, &row) in indices.iter().enumerate() {
        let row = usize::try_from(row).map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "atac_indices[{i}] = {row} must be non-negative"
            ))
        })?;
        if row >= n_cells {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "atac_indices[{i}] = {row} exceeds n_cells = {n_cells}"
            )));
        }
    }
    if gene_chrom_codes.len() != gene_tss.len() || gene_tss.len() != gene_rna_cols.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "gene arrays must have the same length; got chrom_codes={}, tss={}, rna_cols={}",
            gene_chrom_codes.len(),
            gene_tss.len(),
            gene_rna_cols.len()
        )));
    }
    if let Some((i, &col)) = gene_rna_cols
        .iter()
        .enumerate()
        .find(|(_, &col)| (col as usize) >= n_genes)
    {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "gene_rna_cols[{i}] = {col} exceeds n_genes = {n_genes}"
        )));
    }
    Ok(())
}

fn peak_center(start: i64, end: i64) -> i64 {
    start.saturating_add(end.saturating_sub(start) / 2)
}

fn sort_enhancer_links_by_abs_corr(links: &mut [EnhancerPearsonLink]) {
    links.sort_unstable_by(|a, b| {
        b.correlation
            .abs()
            .total_cmp(&a.correlation.abs())
            .then_with(|| a.peak_idx.cmp(&b.peak_idx))
            .then_with(|| a.gene_idx.cmp(&b.gene_idx))
    });
}

#[allow(clippy::too_many_arguments)]
fn compute_enhancer_pearson_links(
    rna: &[f32],
    n_cells: usize,
    n_genes: usize,
    indptr: &[usize],
    indices: &[i32],
    data: &[f32],
    peak_chrom_codes: &[i32],
    peak_starts: &[i64],
    peak_ends: &[i64],
    gene_chrom_codes: &[i32],
    gene_tss: &[i64],
    gene_rna_cols: &[u32],
    atac_peak_cols: Option<&[u32]>,
    max_distance: i64,
    min_abs_corr: f32,
) -> Vec<EnhancerPearsonLink> {
    compute_enhancer_pearson_links_with(
        n_cells,
        n_genes,
        |cell, col| rna[cell * n_genes + col],
        indptr,
        indices,
        data,
        peak_chrom_codes,
        peak_starts,
        peak_ends,
        gene_chrom_codes,
        gene_tss,
        gene_rna_cols,
        atac_peak_cols,
        max_distance,
        min_abs_corr,
    )
}

#[allow(clippy::too_many_arguments)]
fn compute_enhancer_pearson_links_view(
    rna: ndarray::ArrayView2<'_, f32>,
    indptr: &[usize],
    indices: &[i32],
    data: &[f32],
    peak_chrom_codes: &[i32],
    peak_starts: &[i64],
    peak_ends: &[i64],
    gene_chrom_codes: &[i32],
    gene_tss: &[i64],
    gene_rna_cols: &[u32],
    atac_peak_cols: Option<&[u32]>,
    max_distance: i64,
    min_abs_corr: f32,
) -> Vec<EnhancerPearsonLink> {
    let n_cells = rna.shape()[0];
    let n_genes = rna.shape()[1];
    compute_enhancer_pearson_links_with(
        n_cells,
        n_genes,
        |cell, col| rna[(cell, col)],
        indptr,
        indices,
        data,
        peak_chrom_codes,
        peak_starts,
        peak_ends,
        gene_chrom_codes,
        gene_tss,
        gene_rna_cols,
        atac_peak_cols,
        max_distance,
        min_abs_corr,
    )
}

#[allow(clippy::too_many_arguments)]
fn compute_enhancer_pearson_links_with<F>(
    n_cells: usize,
    _n_genes: usize,
    rna_value: F,
    indptr: &[usize],
    indices: &[i32],
    data: &[f32],
    peak_chrom_codes: &[i32],
    peak_starts: &[i64],
    peak_ends: &[i64],
    gene_chrom_codes: &[i32],
    gene_tss: &[i64],
    gene_rna_cols: &[u32],
    atac_peak_cols: Option<&[u32]>,
    max_distance: i64,
    min_abs_corr: f32,
) -> Vec<EnhancerPearsonLink>
where
    F: Fn(usize, usize) -> f32 + Sync,
{
    if n_cells == 0 || gene_rna_cols.is_empty() || peak_chrom_codes.is_empty() {
        return Vec::new();
    }
    if max_distance < 0 {
        return Vec::new();
    }

    let n = n_cells as f64;
    let gene_stats: Vec<GenePearsonStats> = gene_rna_cols
        .par_iter()
        .map(|&raw_col| {
            let col = raw_col as usize;
            let mut sum = 0.0;
            let mut sumsq = 0.0;
            for cell in 0..n_cells {
                let value = rna_value(cell, col) as f64;
                sum += value;
                sumsq += value * value;
            }
            let mean = sum / n;
            let var = (sumsq / n - mean * mean).max(0.0);
            GenePearsonStats {
                col,
                sum,
                std: var.sqrt(),
            }
        })
        .collect();

    let chrom_ranges = contiguous_chrom_ranges(gene_chrom_codes);
    let links_by_peak: Vec<Vec<EnhancerPearsonLink>> = (0..peak_chrom_codes.len())
        .into_par_iter()
        .map(|peak_idx| {
            let Some(&(chrom_start, chrom_end)) = chrom_ranges.get(&peak_chrom_codes[peak_idx])
            else {
                return Vec::new();
            };
            let centre = peak_center(peak_starts[peak_idx], peak_ends[peak_idx]);
            let left = centre.saturating_sub(max_distance);
            let right = centre.saturating_add(max_distance).saturating_add(1);
            let lo = chrom_start + lower_bound_i64(&gene_tss[chrom_start..chrom_end], left);
            let hi = chrom_start + lower_bound_i64(&gene_tss[chrom_start..chrom_end], right);
            if lo >= hi {
                return Vec::new();
            }

            let atac_col = atac_peak_cols.map_or(peak_idx, |cols| cols[peak_idx] as usize);
            let col_start = indptr[atac_col];
            let col_end = indptr[atac_col + 1];
            let mut x_sum = 0.0;
            let mut x_sumsq = 0.0;
            for &x in &data[col_start..col_end] {
                let value = x as f64;
                x_sum += value;
                x_sumsq += value * value;
            }
            let x_mean = x_sum / n;
            let x_var = (x_sumsq / n - x_mean * x_mean).max(0.0);
            let x_std = x_var.sqrt();

            let n_candidates = hi - lo;
            let mut cross_sums = vec![0.0_f64; n_candidates];
            for k in col_start..col_end {
                let row = indices[k] as usize;
                let x_value = data[k] as f64;
                for (offset, gene_idx) in (lo..hi).enumerate() {
                    let y_col = gene_stats[gene_idx].col;
                    cross_sums[offset] += x_value * rna_value(row, y_col) as f64;
                }
            }

            let mut out = Vec::new();
            for (offset, gene_idx) in (lo..hi).enumerate() {
                let stats = gene_stats[gene_idx];
                let y_mean = stats.sum / n;
                let denom = x_std * stats.std;
                let corr = if denom > 0.0 {
                    (cross_sums[offset] / n - x_mean * y_mean) / denom
                } else {
                    0.0
                };
                let corr = corr as f32;
                if corr.abs() >= min_abs_corr {
                    out.push(EnhancerPearsonLink {
                        peak_idx: peak_idx as u32,
                        gene_idx: gene_idx as u32,
                        distance: centre.saturating_sub(gene_tss[gene_idx]),
                        correlation: corr,
                    });
                }
            }
            out
        })
        .collect();

    links_by_peak.into_iter().flatten().collect()
}

#[allow(clippy::too_many_arguments)]
fn compute_enhancer_pearson_links_sparse_rna(
    n_cells: usize,
    rna_indptr: &[usize],
    rna_indices: &[i32],
    rna_data: &[f32],
    indptr: &[usize],
    indices: &[i32],
    data: &[f32],
    peak_chrom_codes: &[i32],
    peak_starts: &[i64],
    peak_ends: &[i64],
    gene_chrom_codes: &[i32],
    gene_tss: &[i64],
    gene_rna_cols: &[u32],
    atac_peak_cols: Option<&[u32]>,
    max_distance: i64,
    min_abs_corr: f32,
) -> Vec<EnhancerPearsonLink> {
    if n_cells == 0 || gene_rna_cols.is_empty() || peak_chrom_codes.is_empty() {
        return Vec::new();
    }
    if max_distance < 0 {
        return Vec::new();
    }

    let n = n_cells as f64;
    let gene_stats: Vec<SparseGenePearsonStats> = gene_rna_cols
        .par_iter()
        .map(|&raw_col| {
            let col = raw_col as usize;
            let start = rna_indptr[col];
            let end = rna_indptr[col + 1];
            let mut sum = 0.0;
            let mut sumsq = 0.0;
            for &y in &rna_data[start..end] {
                let value = y as f64;
                sum += value;
                sumsq += value * value;
            }
            let mean = sum / n;
            let var = (sumsq / n - mean * mean).max(0.0);
            SparseGenePearsonStats {
                start,
                end,
                sum,
                std: var.sqrt(),
            }
        })
        .collect();

    let chrom_ranges = contiguous_chrom_ranges(gene_chrom_codes);
    let links_by_peak: Vec<Vec<EnhancerPearsonLink>> = (0..peak_chrom_codes.len())
        .into_par_iter()
        .map(|peak_idx| {
            let Some(&(chrom_start, chrom_end)) = chrom_ranges.get(&peak_chrom_codes[peak_idx])
            else {
                return Vec::new();
            };
            let centre = peak_center(peak_starts[peak_idx], peak_ends[peak_idx]);
            let left = centre.saturating_sub(max_distance);
            let right = centre.saturating_add(max_distance).saturating_add(1);
            let lo = chrom_start + lower_bound_i64(&gene_tss[chrom_start..chrom_end], left);
            let hi = chrom_start + lower_bound_i64(&gene_tss[chrom_start..chrom_end], right);
            if lo >= hi {
                return Vec::new();
            }

            let atac_col = atac_peak_cols.map_or(peak_idx, |cols| cols[peak_idx] as usize);
            let col_start = indptr[atac_col];
            let col_end = indptr[atac_col + 1];
            let mut x_sum = 0.0;
            let mut x_sumsq = 0.0;
            for &x in &data[col_start..col_end] {
                let value = x as f64;
                x_sum += value;
                x_sumsq += value * value;
            }
            let x_mean = x_sum / n;
            let x_var = (x_sumsq / n - x_mean * x_mean).max(0.0);
            let x_std = x_var.sqrt();

            let x_indices = &indices[col_start..col_end];
            let x_data = &data[col_start..col_end];
            let mut out = Vec::new();
            for gene_idx in lo..hi {
                let stats = gene_stats[gene_idx];
                let y_mean = stats.sum / n;
                let denom = x_std * stats.std;
                let dot = sparse_dot_sorted(
                    x_indices,
                    x_data,
                    &rna_indices[stats.start..stats.end],
                    &rna_data[stats.start..stats.end],
                );
                let corr = if denom > 0.0 {
                    (dot / n - x_mean * y_mean) / denom
                } else {
                    0.0
                };
                let corr = corr as f32;
                if corr.abs() >= min_abs_corr {
                    out.push(EnhancerPearsonLink {
                        peak_idx: peak_idx as u32,
                        gene_idx: gene_idx as u32,
                        distance: centre.saturating_sub(gene_tss[gene_idx]),
                        correlation: corr,
                    });
                }
            }
            out
        })
        .collect();

    links_by_peak.into_iter().flatten().collect()
}

fn sparse_dot_sorted(x_indices: &[i32], x_data: &[f32], y_indices: &[i32], y_data: &[f32]) -> f64 {
    let mut i = 0usize;
    let mut j = 0usize;
    let mut dot = 0.0_f64;
    while i < x_indices.len() && j < y_indices.len() {
        match x_indices[i].cmp(&y_indices[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                dot += x_data[i] as f64 * y_data[j] as f64;
                i += 1;
                j += 1;
            }
        }
    }
    dot
}

fn contiguous_chrom_ranges(codes: &[i32]) -> HashMap<i32, (usize, usize)> {
    let mut ranges = HashMap::new();
    let mut start = 0usize;
    while start < codes.len() {
        let code = codes[start];
        let mut end = start + 1;
        while end < codes.len() && codes[end] == code {
            end += 1;
        }
        ranges.insert(code, (start, end));
        start = end;
    }
    ranges
}

fn lower_bound_i64(values: &[i64], target: i64) -> usize {
    let mut lo = 0usize;
    let mut hi = values.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if values[mid] < target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Assemble eRegulon edge rows from typed columns.
///
/// This is the scale-sensitive part of `rustscenic.eregulon.build_eregulons`:
/// intersect cistarget TF->peak rows, enhancer peak->gene links, and optional
/// GRN TF->gene support without materialising pandas intermediate joins.
#[pyfunction]
#[pyo3(signature = (
    grn_tfs,
    grn_targets,
    cistarget_regulons,
    cistarget_peaks,
    cistarget_aucs,
    enhancer_peaks,
    enhancer_genes,
    enhancer_correlations,
    min_target_genes = 5,
    min_enhancer_links = 2,
    cistarget_auc_threshold = 0.05,
    use_grn_intersection = true,
))]
#[allow(clippy::too_many_arguments)]
fn eregulon_assemble<'py>(
    py: Python<'py>,
    grn_tfs: Vec<String>,
    grn_targets: Vec<String>,
    cistarget_regulons: Vec<String>,
    cistarget_peaks: Vec<String>,
    cistarget_aucs: PyReadonlyArray1<'py, f64>,
    enhancer_peaks: Vec<String>,
    enhancer_genes: Vec<String>,
    enhancer_correlations: PyReadonlyArray1<'py, f32>,
    min_target_genes: usize,
    min_enhancer_links: usize,
    cistarget_auc_threshold: f64,
    use_grn_intersection: bool,
) -> PyResult<ERegulonAssemblePyResult> {
    let cistarget_aucs = cistarget_aucs.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("cistarget_aucs must be contiguous")
    })?;
    let enhancer_correlations = enhancer_correlations.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("enhancer_correlations must be contiguous")
    })?;
    eregulon_assemble_from_slices(
        py,
        &grn_tfs,
        &grn_targets,
        &cistarget_regulons,
        &cistarget_peaks,
        cistarget_aucs,
        &enhancer_peaks,
        &enhancer_genes,
        enhancer_correlations,
        min_target_genes,
        min_enhancer_links,
        cistarget_auc_threshold,
        use_grn_intersection,
    )
}

#[pyfunction]
#[pyo3(signature = (
    grn_tfs,
    grn_targets,
    cistarget_regulons,
    cistarget_peaks,
    cistarget_aucs,
    enhancer_peaks,
    enhancer_genes,
    enhancer_correlations,
    min_target_genes = 5,
    min_enhancer_links = 2,
    cistarget_auc_threshold = 0.05,
    use_grn_intersection = true,
))]
#[allow(clippy::too_many_arguments)]
fn eregulon_assemble_f32<'py>(
    py: Python<'py>,
    grn_tfs: Vec<String>,
    grn_targets: Vec<String>,
    cistarget_regulons: Vec<String>,
    cistarget_peaks: Vec<String>,
    cistarget_aucs: PyReadonlyArray1<'py, f32>,
    enhancer_peaks: Vec<String>,
    enhancer_genes: Vec<String>,
    enhancer_correlations: PyReadonlyArray1<'py, f32>,
    min_target_genes: usize,
    min_enhancer_links: usize,
    cistarget_auc_threshold: f64,
    use_grn_intersection: bool,
) -> PyResult<ERegulonAssemblePyResult> {
    let cistarget_aucs = cistarget_aucs.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("cistarget_aucs must be contiguous")
    })?;
    let enhancer_correlations = enhancer_correlations.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("enhancer_correlations must be contiguous")
    })?;
    eregulon_assemble_from_slices(
        py,
        &grn_tfs,
        &grn_targets,
        &cistarget_regulons,
        &cistarget_peaks,
        cistarget_aucs,
        &enhancer_peaks,
        &enhancer_genes,
        enhancer_correlations,
        min_target_genes,
        min_enhancer_links,
        cistarget_auc_threshold,
        use_grn_intersection,
    )
}

#[allow(clippy::too_many_arguments)]
fn eregulon_assemble_from_slices<'py, T: Copy + Into<f64> + Sync>(
    py: Python<'py>,
    grn_tfs: &[String],
    grn_targets: &[String],
    cistarget_regulons: &[String],
    cistarget_peaks: &[String],
    cistarget_aucs: &[T],
    enhancer_peaks: &[String],
    enhancer_genes: &[String],
    enhancer_correlations: &[f32],
    min_target_genes: usize,
    min_enhancer_links: usize,
    cistarget_auc_threshold: f64,
    use_grn_intersection: bool,
) -> PyResult<ERegulonAssemblePyResult> {
    validate_eregulon_assemble_lengths(
        grn_tfs,
        grn_targets,
        cistarget_regulons,
        cistarget_peaks,
        cistarget_aucs.len(),
        enhancer_peaks,
        enhancer_genes,
        enhancer_correlations.len(),
    )?;

    let (mut eregulons, n_input_tfs) = py.allow_threads(|| {
        assemble_eregulon_groups(
            grn_tfs,
            grn_targets,
            cistarget_regulons,
            cistarget_peaks,
            cistarget_aucs,
            enhancer_peaks,
            enhancer_genes,
            enhancer_correlations,
            min_target_genes,
            min_enhancer_links,
            cistarget_auc_threshold,
            use_grn_intersection,
        )
    });
    let n_eregulons = eregulons.len();

    let n_rows = eregulons.iter().map(|er| er.n_edges).sum();
    let mut tf_col = Vec::with_capacity(n_rows);
    let mut enhancer_col = Vec::with_capacity(n_rows);
    let mut target_col = Vec::with_capacity(n_rows);
    let mut n_links_col = Vec::with_capacity(n_rows);
    let mut auc_col = Vec::with_capacity(n_rows);
    for er in &mut eregulons {
        for target in &mut er.targets {
            target.peaks.sort();
            for &peak in &target.peaks {
                tf_col.push(er.tf.as_str());
                enhancer_col.push(peak);
                target_col.push(target.gene);
                n_links_col.push(er.n_edges as u32);
                auc_col.push(er.motif_auc);
            }
        }
    }

    Ok((
        PyList::new(py, tf_col.iter().copied())?.unbind(),
        PyList::new(py, enhancer_col.iter().copied())?.unbind(),
        PyList::new(py, target_col.iter().copied())?.unbind(),
        PyArray1::from_vec(py, n_links_col).unbind(),
        PyArray1::from_vec(py, auc_col).unbind(),
        n_input_tfs,
        n_eregulons,
    ))
}

/// Assemble grouped eRegulon objects from typed columns.
///
/// This is the public `build_eregulons()` fast path: Rust does the biological
/// intersection and grouped target/peak summarisation, while Python only wraps
/// one object per eRegulon.
#[pyfunction]
#[pyo3(signature = (
    grn_tfs,
    grn_targets,
    cistarget_regulons,
    cistarget_peaks,
    cistarget_aucs,
    enhancer_peaks,
    enhancer_genes,
    enhancer_correlations,
    min_target_genes = 5,
    min_enhancer_links = 2,
    cistarget_auc_threshold = 0.05,
    use_grn_intersection = true,
))]
#[allow(clippy::too_many_arguments)]
fn eregulon_assemble_summary<'py>(
    py: Python<'py>,
    grn_tfs: Vec<String>,
    grn_targets: Vec<String>,
    cistarget_regulons: Vec<String>,
    cistarget_peaks: Vec<String>,
    cistarget_aucs: PyReadonlyArray1<'py, f64>,
    enhancer_peaks: Vec<String>,
    enhancer_genes: Vec<String>,
    enhancer_correlations: PyReadonlyArray1<'py, f32>,
    min_target_genes: usize,
    min_enhancer_links: usize,
    cistarget_auc_threshold: f64,
    use_grn_intersection: bool,
) -> PyResult<ERegulonSummaryPyResult> {
    let cistarget_aucs = cistarget_aucs.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("cistarget_aucs must be contiguous")
    })?;
    let enhancer_correlations = enhancer_correlations.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("enhancer_correlations must be contiguous")
    })?;
    eregulon_assemble_summary_from_slices(
        py,
        &grn_tfs,
        &grn_targets,
        &cistarget_regulons,
        &cistarget_peaks,
        cistarget_aucs,
        &enhancer_peaks,
        &enhancer_genes,
        enhancer_correlations,
        min_target_genes,
        min_enhancer_links,
        cistarget_auc_threshold,
        use_grn_intersection,
    )
}

#[pyfunction]
#[pyo3(signature = (
    grn_tfs,
    grn_targets,
    cistarget_regulons,
    cistarget_peaks,
    cistarget_aucs,
    enhancer_peaks,
    enhancer_genes,
    enhancer_correlations,
    min_target_genes = 5,
    min_enhancer_links = 2,
    cistarget_auc_threshold = 0.05,
    use_grn_intersection = true,
))]
#[allow(clippy::too_many_arguments)]
fn eregulon_assemble_summary_f32<'py>(
    py: Python<'py>,
    grn_tfs: Vec<String>,
    grn_targets: Vec<String>,
    cistarget_regulons: Vec<String>,
    cistarget_peaks: Vec<String>,
    cistarget_aucs: PyReadonlyArray1<'py, f32>,
    enhancer_peaks: Vec<String>,
    enhancer_genes: Vec<String>,
    enhancer_correlations: PyReadonlyArray1<'py, f32>,
    min_target_genes: usize,
    min_enhancer_links: usize,
    cistarget_auc_threshold: f64,
    use_grn_intersection: bool,
) -> PyResult<ERegulonSummaryPyResult> {
    let cistarget_aucs = cistarget_aucs.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("cistarget_aucs must be contiguous")
    })?;
    let enhancer_correlations = enhancer_correlations.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("enhancer_correlations must be contiguous")
    })?;
    eregulon_assemble_summary_from_slices(
        py,
        &grn_tfs,
        &grn_targets,
        &cistarget_regulons,
        &cistarget_peaks,
        cistarget_aucs,
        &enhancer_peaks,
        &enhancer_genes,
        enhancer_correlations,
        min_target_genes,
        min_enhancer_links,
        cistarget_auc_threshold,
        use_grn_intersection,
    )
}

#[allow(clippy::too_many_arguments)]
fn eregulon_assemble_summary_from_slices<'py, T: Copy + Into<f64> + Sync>(
    py: Python<'py>,
    grn_tfs: &[String],
    grn_targets: &[String],
    cistarget_regulons: &[String],
    cistarget_peaks: &[String],
    cistarget_aucs: &[T],
    enhancer_peaks: &[String],
    enhancer_genes: &[String],
    enhancer_correlations: &[f32],
    min_target_genes: usize,
    min_enhancer_links: usize,
    cistarget_auc_threshold: f64,
    use_grn_intersection: bool,
) -> PyResult<ERegulonSummaryPyResult> {
    validate_eregulon_assemble_lengths(
        grn_tfs,
        grn_targets,
        cistarget_regulons,
        cistarget_peaks,
        cistarget_aucs.len(),
        enhancer_peaks,
        enhancer_genes,
        enhancer_correlations.len(),
    )?;

    let (eregulons, n_input_tfs) = py.allow_threads(|| {
        assemble_eregulon_groups(
            grn_tfs,
            grn_targets,
            cistarget_regulons,
            cistarget_peaks,
            cistarget_aucs,
            enhancer_peaks,
            enhancer_genes,
            enhancer_correlations,
            min_target_genes,
            min_enhancer_links,
            cistarget_auc_threshold,
            use_grn_intersection,
        )
    });
    let n_eregulons = eregulons.len();

    let tf_col = PyList::empty(py);
    let enhancer_lists = PyList::empty(py);
    let target_lists = PyList::empty(py);
    let target_to_peaks_col = PyList::empty(py);
    let mut n_links_col = Vec::with_capacity(n_eregulons);
    let mut auc_col = Vec::with_capacity(n_eregulons);

    for mut er in eregulons {
        tf_col.append(er.tf.as_str())?;
        n_links_col.push(er.n_edges as u32);
        auc_col.push(er.motif_auc);

        let mut enhancers = Vec::new();
        let mut seen_enhancers = HashSet::new();
        for target in &mut er.targets {
            target.peaks.sort();
            for &peak in &target.peaks {
                if seen_enhancers.insert(peak) {
                    enhancers.push(peak);
                }
            }
        }
        enhancers.sort();
        enhancer_lists.append(PyList::new(py, enhancers.iter().copied())?)?;

        let mut target_names: Vec<&str> = er.targets.iter().map(|target| target.gene).collect();
        target_names.sort();
        target_lists.append(PyList::new(py, target_names.iter().copied())?)?;

        let target_to_peaks = PyDict::new(py);
        for target in &er.targets {
            target_to_peaks
                .set_item(target.gene, PyList::new(py, target.peaks.iter().copied())?)?;
        }
        target_to_peaks_col.append(target_to_peaks)?;
    }

    Ok((
        tf_col.unbind(),
        enhancer_lists.unbind(),
        target_lists.unbind(),
        PyArray1::from_vec(py, n_links_col).unbind(),
        PyArray1::from_vec(py, auc_col).unbind(),
        target_to_peaks_col.unbind(),
        n_input_tfs,
        n_eregulons,
    ))
}

#[allow(clippy::too_many_arguments)]
fn validate_eregulon_assemble_lengths(
    grn_tfs: &[String],
    grn_targets: &[String],
    cistarget_regulons: &[String],
    cistarget_peaks: &[String],
    n_cistarget_aucs: usize,
    enhancer_peaks: &[String],
    enhancer_genes: &[String],
    n_enhancer_correlations: usize,
) -> PyResult<()> {
    if cistarget_regulons.len() != cistarget_peaks.len()
        || cistarget_peaks.len() != n_cistarget_aucs
    {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "cistarget arrays must have the same length; got regulons={}, peaks={}, aucs={}",
            cistarget_regulons.len(),
            cistarget_peaks.len(),
            n_cistarget_aucs
        )));
    }
    if enhancer_peaks.len() != enhancer_genes.len()
        || enhancer_genes.len() != n_enhancer_correlations
    {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "enhancer arrays must have the same length; got peaks={}, genes={}, correlations={}",
            enhancer_peaks.len(),
            enhancer_genes.len(),
            n_enhancer_correlations
        )));
    }
    if grn_tfs.len() != grn_targets.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "grn arrays must have the same length; got TFs={}, targets={}",
            grn_tfs.len(),
            grn_targets.len()
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn assemble_eregulon_groups<'a, T: Copy + Into<f64>>(
    grn_tfs: &'a [String],
    grn_targets: &'a [String],
    cistarget_regulons: &'a [String],
    cistarget_peaks: &'a [String],
    cistarget_aucs: &'a [T],
    enhancer_peaks: &'a [String],
    enhancer_genes: &'a [String],
    enhancer_correlations: &'a [f32],
    min_target_genes: usize,
    min_enhancer_links: usize,
    cistarget_auc_threshold: f64,
    use_grn_intersection: bool,
) -> (Vec<ERegulonRows<'a>>, usize) {
    let mut ct_by_tf: HashMap<String, CtTfRows<'_>> = HashMap::new();
    for ((regulon, peak), auc) in cistarget_regulons
        .iter()
        .zip(cistarget_peaks)
        .zip(cistarget_aucs.iter().copied())
    {
        let auc = auc.into();
        if !auc.is_finite() || auc < cistarget_auc_threshold {
            continue;
        }
        let tf = tf_from_regulon_name_rs(regulon);
        ct_by_tf.entry(tf).or_default().push(peak.as_str(), auc);
    }
    let n_input_tfs = ct_by_tf.len();
    if n_input_tfs == 0 {
        return (Vec::new(), 0);
    }

    let mut links_by_peak: HashMap<&str, Vec<&str>> = HashMap::new();
    for ((peak, gene), &corr) in enhancer_peaks
        .iter()
        .zip(enhancer_genes)
        .zip(enhancer_correlations)
    {
        if corr > 0.0 {
            links_by_peak
                .entry(peak.as_str())
                .or_default()
                .push(gene.as_str());
        }
    }

    let mut grn_targets_by_tf: HashMap<&str, HashSet<&str>> = HashMap::new();
    if use_grn_intersection {
        for (tf, target) in grn_tfs.iter().zip(grn_targets) {
            grn_targets_by_tf
                .entry(tf.as_str())
                .or_default()
                .insert(target.as_str());
        }
    }

    let mut tfs: Vec<&String> = ct_by_tf.keys().collect();
    tfs.sort();

    let mut eregulons = Vec::new();
    for tf in tfs {
        let ct_rows = &ct_by_tf[tf];
        let grn_set = if use_grn_intersection {
            Some(grn_targets_by_tf.get(tf.as_str()))
        } else {
            None
        };
        if use_grn_intersection && grn_set.flatten().is_none() {
            continue;
        }

        let mut target_pos: HashMap<&str, usize> = HashMap::new();
        let mut targets: Vec<TargetSupport> = Vec::new();
        for &peak in &ct_rows.peaks {
            let Some(links) = links_by_peak.get(peak) else {
                continue;
            };
            for &gene in links {
                if let Some(Some(supported_targets)) = grn_set {
                    if !supported_targets.contains(gene) {
                        continue;
                    }
                }
                let pos = if let Some(&pos) = target_pos.get(gene) {
                    pos
                } else {
                    let pos = targets.len();
                    target_pos.insert(gene, pos);
                    targets.push(TargetSupport::new(gene));
                    pos
                };
                targets[pos].add_peak(peak);
            }
        }

        if targets.len() < min_target_genes {
            continue;
        }

        let n_edges = targets
            .iter()
            .map(|target| target.peaks.len())
            .sum::<usize>();
        if n_edges < min_enhancer_links {
            continue;
        }

        let mut supporting_peaks = HashSet::new();
        for target in &targets {
            for &peak in &target.peaks {
                supporting_peaks.insert(peak);
            }
        }
        let mut auc_sum = 0.0;
        let mut auc_count = 0usize;
        for (peak, auc) in &ct_rows.auc_rows {
            if supporting_peaks.contains(peak) {
                auc_sum += *auc;
                auc_count += 1;
            }
        }
        let motif_auc = if auc_count > 0 {
            auc_sum / auc_count as f64
        } else {
            0.0
        };

        eregulons.push(ERegulonRows {
            tf: tf.clone(),
            targets,
            n_edges,
            motif_auc,
        });
    }

    eregulons.sort_by(|a, b| {
        b.n_edges
            .cmp(&a.n_edges)
            .then_with(|| b.targets.len().cmp(&a.targets.len()))
            .then_with(|| a.tf.cmp(&b.tf))
    });
    (eregulons, n_input_tfs)
}

fn tf_from_regulon_name_rs(name: &str) -> String {
    const SUFFIXES: [&str; 4] = ["_regulon", "_extended", "_activator", "_repressor"];
    const PARENS: [&str; 2] = ["(+)", "(-)"];

    let mut tf = name.trim();
    loop {
        let prev_len = tf.len();
        for suffix in SUFFIXES.iter().chain(PARENS.iter()) {
            if let Some(stripped) = tf.strip_suffix(suffix) {
                tf = stripped.trim();
            }
        }
        if tf.len() == prev_len {
            return tf.to_string();
        }
    }
}

/// Attribute region-cistarget enriched motif rows back to supporting peak IDs.
///
/// This replaces the pandas `peak_long` x `rank_long` melt/merge path in
/// `pipeline._region_cistarget_with_peak_ids`. Rankings are lower-is-better;
/// a peak supports an enriched (regulon, motif) row when its rank is at or
/// below the caller's cutoff.
#[rustfmt::skip]
macro_rules! region_attribution_pyfunction {
    ($name:ident, $rank_ty:ty, $cutoff_ty:ty) => {
        #[pyfunction]
        #[pyo3(signature = (rankings, motif_names, peak_names, peak_regulon_names, peak_regulon_peaks, enriched_regulons, enriched_motifs, rank_cutoff))]
        #[allow(clippy::too_many_arguments)]
        fn $name<'py>(
            py: Python<'py>,
            rankings: PyReadonlyArray2<'py, $rank_ty>,
            motif_names: Vec<String>,
            peak_names: Vec<String>,
            peak_regulon_names: Vec<String>,
            peak_regulon_peaks: Vec<Vec<String>>,
            enriched_regulons: Vec<String>,
            enriched_motifs: Vec<String>,
            rank_cutoff: $cutoff_ty,
        ) -> PyResult<RegionAttributionIndexPyResult> {
            let rankings_arr = rankings.as_array();
            cistarget_region_attribution_impl(
                py,
                rankings_arr,
                motif_names,
                peak_names,
                peak_regulon_names,
                peak_regulon_peaks,
                enriched_regulons,
                enriched_motifs,
                rank_cutoff,
            )
        }
    };
}

region_attribution_pyfunction!(cistarget_region_attribution_i16, i16, i16);
region_attribution_pyfunction!(cistarget_region_attribution_i32, i32, i32);
region_attribution_pyfunction!(cistarget_region_attribution_i64, i64, i64);

#[rustfmt::skip]
macro_rules! region_attribution_peak_values_pyfunction {
    ($name:ident, $rank_ty:ty, $cutoff_ty:ty) => {
        #[pyfunction]
        #[pyo3(signature = (rankings, motif_names, peak_names, peak_regulon_names, peak_regulon_peaks, enriched_regulons, enriched_motifs, rank_cutoff))]
        #[allow(clippy::too_many_arguments)]
        fn $name<'py>(
            py: Python<'py>,
            rankings: PyReadonlyArray2<'py, $rank_ty>,
            motif_names: Vec<String>,
            peak_names: Vec<String>,
            peak_regulon_names: Vec<String>,
            peak_regulon_peaks: Vec<Vec<String>>,
            enriched_regulons: Vec<String>,
            enriched_motifs: Vec<String>,
            rank_cutoff: $cutoff_ty,
        ) -> PyResult<RegionAttributionPeakValuePyResult> {
            let rankings_arr = rankings.as_array();
            cistarget_region_attribution_peak_values_impl(
                py,
                rankings_arr,
                motif_names,
                peak_names,
                peak_regulon_names,
                peak_regulon_peaks,
                enriched_regulons,
                enriched_motifs,
                rank_cutoff,
            )
        }
    };
}

region_attribution_peak_values_pyfunction!(cistarget_region_attribution_peak_values_i16, i16, i16);
region_attribution_peak_values_pyfunction!(cistarget_region_attribution_peak_values_i32, i32, i32);
region_attribution_peak_values_pyfunction!(cistarget_region_attribution_peak_values_i64, i64, i64);

#[allow(clippy::too_many_arguments)]
fn cistarget_region_attribution_impl<'py, T: Copy + PartialOrd + Send + Sync>(
    py: Python<'py>,
    rankings: ndarray::ArrayView2<'_, T>,
    motif_names: Vec<String>,
    peak_names: Vec<String>,
    peak_regulon_names: Vec<String>,
    peak_regulon_peaks: Vec<Vec<String>>,
    enriched_regulons: Vec<String>,
    enriched_motifs: Vec<String>,
    rank_cutoff: T,
) -> PyResult<RegionAttributionIndexPyResult> {
    let n_motifs = rankings.shape()[0];
    let n_peaks = rankings.shape()[1];
    validate_region_attribution_inputs(
        n_motifs,
        n_peaks,
        &motif_names,
        &peak_names,
        &peak_regulon_names,
        &peak_regulon_peaks,
        &enriched_regulons,
        &enriched_motifs,
    )?;
    let (row_indices, peak_indices) = py.allow_threads(|| {
        assemble_region_attribution(
            rankings,
            &motif_names,
            &peak_names,
            &peak_regulon_names,
            &peak_regulon_peaks,
            &enriched_regulons,
            &enriched_motifs,
            rank_cutoff,
        )
    });
    Ok((
        PyArray1::from_vec(py, row_indices).unbind(),
        PyArray1::from_vec(py, peak_indices).unbind(),
    ))
}

#[allow(clippy::too_many_arguments)]
fn cistarget_region_attribution_peak_values_impl<'py, T: Copy + PartialOrd + Send + Sync>(
    py: Python<'py>,
    rankings: ndarray::ArrayView2<'_, T>,
    motif_names: Vec<String>,
    peak_names: Vec<String>,
    peak_regulon_names: Vec<String>,
    peak_regulon_peaks: Vec<Vec<String>>,
    enriched_regulons: Vec<String>,
    enriched_motifs: Vec<String>,
    rank_cutoff: T,
) -> PyResult<RegionAttributionPeakValuePyResult> {
    let n_motifs = rankings.shape()[0];
    let n_peaks = rankings.shape()[1];
    validate_region_attribution_inputs(
        n_motifs,
        n_peaks,
        &motif_names,
        &peak_names,
        &peak_regulon_names,
        &peak_regulon_peaks,
        &enriched_regulons,
        &enriched_motifs,
    )?;
    let (row_indices, peak_indices) = py.allow_threads(|| {
        assemble_region_attribution(
            rankings,
            &motif_names,
            &peak_names,
            &peak_regulon_names,
            &peak_regulon_peaks,
            &enriched_regulons,
            &enriched_motifs,
            rank_cutoff,
        )
    });
    let peak_values = PyList::empty(py);
    for peak_idx in peak_indices {
        let peak = peak_names
            .get(peak_idx as usize)
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("peak index out of bounds"))?;
        peak_values.append(peak)?;
    }
    Ok((
        PyArray1::from_vec(py, row_indices).unbind(),
        peak_values.unbind(),
    ))
}

#[pyfunction]
fn pipeline_expand_region_cistarget_rows_f32<'py>(
    py: Python<'py>,
    row_indices: PyReadonlyArray1<'py, u64>,
    peak_ids: Vec<String>,
    enriched_regulons: Vec<String>,
    enriched_motifs: Vec<String>,
    enriched_aucs: PyReadonlyArray1<'py, f32>,
) -> PyResult<PipelineAttributeRowsF32PyResult> {
    let row_indices = row_indices
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("row_indices must be contiguous"))?;
    let enriched_aucs = enriched_aucs
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("enriched_aucs must be contiguous"))?;
    let (regulons, motifs, peaks, aucs) = py.allow_threads(|| {
        expand_region_cistarget_rows(
            row_indices,
            &peak_ids,
            &enriched_regulons,
            &enriched_motifs,
            enriched_aucs,
        )
    })?;
    region_cistarget_rows_to_py_f32(py, regulons, motifs, peaks, aucs)
}

#[pyfunction]
fn pipeline_expand_region_cistarget_rows_f64<'py>(
    py: Python<'py>,
    row_indices: PyReadonlyArray1<'py, u64>,
    peak_ids: Vec<String>,
    enriched_regulons: Vec<String>,
    enriched_motifs: Vec<String>,
    enriched_aucs: PyReadonlyArray1<'py, f64>,
) -> PyResult<PipelineAttributeRowsF64PyResult> {
    let row_indices = row_indices
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("row_indices must be contiguous"))?;
    let enriched_aucs = enriched_aucs
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("enriched_aucs must be contiguous"))?;
    let (regulons, motifs, peaks, aucs) = py.allow_threads(|| {
        expand_region_cistarget_rows(
            row_indices,
            &peak_ids,
            &enriched_regulons,
            &enriched_motifs,
            enriched_aucs,
        )
    })?;
    region_cistarget_rows_to_py_f64(py, regulons, motifs, peaks, aucs)
}

fn region_cistarget_rows_to_py_f32(
    py: Python<'_>,
    regulons: Vec<String>,
    motifs: Vec<String>,
    peaks: Vec<String>,
    aucs: Vec<f32>,
) -> PyResult<PipelineAttributeRowsF32PyResult> {
    Ok((
        PyList::new(py, regulons.iter().map(String::as_str))?.unbind(),
        PyList::new(py, motifs.iter().map(String::as_str))?.unbind(),
        PyList::new(py, peaks.iter().map(String::as_str))?.unbind(),
        PyArray1::from_vec(py, aucs).unbind(),
    ))
}

fn region_cistarget_rows_to_py_f64(
    py: Python<'_>,
    regulons: Vec<String>,
    motifs: Vec<String>,
    peaks: Vec<String>,
    aucs: Vec<f64>,
) -> PyResult<PipelineAttributeRowsF64PyResult> {
    Ok((
        PyList::new(py, regulons.iter().map(String::as_str))?.unbind(),
        PyList::new(py, motifs.iter().map(String::as_str))?.unbind(),
        PyList::new(py, peaks.iter().map(String::as_str))?.unbind(),
        PyArray1::from_vec(py, aucs).unbind(),
    ))
}

type RegionCistargetRows<T> = (Vec<String>, Vec<String>, Vec<String>, Vec<T>);

fn expand_region_cistarget_rows<T: Copy>(
    row_indices: &[u64],
    peak_ids: &[String],
    enriched_regulons: &[String],
    enriched_motifs: &[String],
    enriched_aucs: &[T],
) -> PyResult<RegionCistargetRows<T>> {
    if row_indices.len() != peak_ids.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "row_indices length {} does not match peak_ids length {}",
            row_indices.len(),
            peak_ids.len()
        )));
    }
    if enriched_regulons.len() != enriched_motifs.len()
        || enriched_regulons.len() != enriched_aucs.len()
    {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "enriched arrays must have the same length; got regulons={} motifs={} aucs={}",
            enriched_regulons.len(),
            enriched_motifs.len(),
            enriched_aucs.len()
        )));
    }

    let mut regulons = Vec::with_capacity(row_indices.len());
    let mut motifs = Vec::with_capacity(row_indices.len());
    let mut peaks = Vec::with_capacity(row_indices.len());
    let mut aucs = Vec::with_capacity(row_indices.len());
    for (&row_idx, peak_id) in row_indices.iter().zip(peak_ids) {
        let row_idx = row_idx as usize;
        if row_idx >= enriched_regulons.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "row index {} out of bounds for {} enriched rows",
                row_idx,
                enriched_regulons.len()
            )));
        }
        regulons.push(enriched_regulons[row_idx].clone());
        motifs.push(enriched_motifs[row_idx].clone());
        peaks.push(peak_id.clone());
        aucs.push(enriched_aucs[row_idx]);
    }
    Ok((regulons, motifs, peaks, aucs))
}

#[allow(clippy::too_many_arguments)]
fn validate_region_attribution_inputs(
    n_motifs: usize,
    n_peaks: usize,
    motif_names: &[String],
    peak_names: &[String],
    peak_regulon_names: &[String],
    peak_regulon_peaks: &[Vec<String>],
    enriched_regulons: &[String],
    enriched_motifs: &[String],
) -> PyResult<()> {
    if motif_names.len() != n_motifs {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "motif_names has length {} but rankings has {} rows",
            motif_names.len(),
            n_motifs
        )));
    }
    if peak_names.len() != n_peaks {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "peak_names has length {} but rankings has {} columns",
            peak_names.len(),
            n_peaks
        )));
    }
    if peak_regulon_names.len() != peak_regulon_peaks.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "peak regulon arrays must have the same length; got names={} peaks={}",
            peak_regulon_names.len(),
            peak_regulon_peaks.len()
        )));
    }
    if enriched_regulons.len() != enriched_motifs.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "enriched arrays must have the same length; got regulons={}, motifs={}",
            enriched_regulons.len(),
            enriched_motifs.len(),
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn assemble_region_attribution<T: Copy + PartialOrd>(
    rankings: ndarray::ArrayView2<'_, T>,
    motif_names: &[String],
    peak_names: &[String],
    peak_regulon_names: &[String],
    peak_regulon_peaks: &[Vec<String>],
    enriched_regulons: &[String],
    enriched_motifs: &[String],
    rank_cutoff: T,
) -> (Vec<u64>, Vec<u64>) {
    let mut motif_rows: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, motif) in motif_names.iter().enumerate() {
        motif_rows.entry(motif.as_str()).or_default().push(i);
    }

    let mut peak_cols: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, peak) in peak_names.iter().enumerate() {
        peak_cols.entry(peak.as_str()).or_default().push(i);
    }

    let mut peaks_by_regulon: HashMap<&str, Vec<&str>> = HashMap::new();
    for (regulon, peaks) in peak_regulon_names.iter().zip(peak_regulon_peaks) {
        let entry = peaks_by_regulon.entry(regulon.as_str()).or_default();
        entry.extend(peaks.iter().map(|peak| peak.as_str()));
    }

    let mut row_indices = Vec::new();
    let mut peak_indices = Vec::new();
    for (enriched_idx, (regulon, motif)) in
        enriched_regulons.iter().zip(enriched_motifs).enumerate()
    {
        let Some(peaks) = peaks_by_regulon.get(regulon.as_str()) else {
            continue;
        };
        let Some(motif_rows_for_name) = motif_rows.get(motif.as_str()) else {
            continue;
        };
        for &peak in peaks {
            let Some(cols) = peak_cols.get(peak) else {
                continue;
            };
            for &col in cols {
                for &row in motif_rows_for_name {
                    let rank = rankings[(row, col)];
                    if rank <= rank_cutoff {
                        row_indices.push(enriched_idx as u64);
                        peak_indices.push(col as u64);
                    }
                }
            }
        }
    }
    (row_indices, peak_indices)
}

/// Attribute gene-ranking cistarget rows to enhancer-linked peak IDs.
///
/// Rust owns the memory-sensitive part: regulon-name normalisation, gene->peak
/// and TF->peak map construction with pandas-compatible first-seen
/// de-duplication, and enriched-row expansion. It returns compact source
/// indices so Python can assemble the public DataFrame without cloning strings
/// once per output row.
#[pyfunction]
fn pipeline_match_atac_cell_indices<'py>(
    py: Python<'py>,
    rna_cells: Vec<String>,
    atac_cells: Vec<String>,
) -> PyResult<PipelineCellIndexPyResult> {
    let matched = py.allow_threads(|| {
        let rna_set: HashSet<&str> = rna_cells.iter().map(|cell| cell.as_str()).collect();
        atac_cells
            .iter()
            .enumerate()
            .filter_map(|(idx, cell)| {
                if rna_set.contains(cell.as_str()) {
                    Some(idx as u64)
                } else {
                    None
                }
            })
            .collect::<Vec<u64>>()
    });
    Ok(PyArray1::from_vec(py, matched).unbind())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn pipeline_attribute_peaks_to_cistarget_rows_f32<'py>(
    py: Python<'py>,
    enriched_regulons: Vec<String>,
    enriched_motifs: Option<Vec<String>>,
    enriched_aucs: PyReadonlyArray1<'py, f32>,
    regulon_names: Vec<String>,
    regulon_targets: Vec<Vec<String>>,
    enhancer_genes: Vec<String>,
    enhancer_peaks: Vec<String>,
) -> PyResult<PipelineAttributeRowsF32PyResult> {
    let enriched_aucs = enriched_aucs
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("enriched_aucs must be contiguous"))?;
    let (regulons, motifs, peaks, aucs) = py.allow_threads(|| {
        attribute_peaks_to_cistarget_rows(
            &enriched_regulons,
            enriched_motifs.as_deref(),
            enriched_aucs,
            &regulon_names,
            &regulon_targets,
            &enhancer_genes,
            &enhancer_peaks,
        )
    })?;
    attribute_peak_rows_to_py_f32(py, regulons, motifs, peaks, aucs)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn pipeline_attribute_peaks_to_cistarget_rows_f64<'py>(
    py: Python<'py>,
    enriched_regulons: Vec<String>,
    enriched_motifs: Option<Vec<String>>,
    enriched_aucs: PyReadonlyArray1<'py, f64>,
    regulon_names: Vec<String>,
    regulon_targets: Vec<Vec<String>>,
    enhancer_genes: Vec<String>,
    enhancer_peaks: Vec<String>,
) -> PyResult<PipelineAttributeRowsF64PyResult> {
    let enriched_aucs = enriched_aucs
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("enriched_aucs must be contiguous"))?;
    let (regulons, motifs, peaks, aucs) = py.allow_threads(|| {
        attribute_peaks_to_cistarget_rows(
            &enriched_regulons,
            enriched_motifs.as_deref(),
            enriched_aucs,
            &regulon_names,
            &regulon_targets,
            &enhancer_genes,
            &enhancer_peaks,
        )
    })?;
    attribute_peak_rows_to_py_f64(py, regulons, motifs, peaks, aucs)
}

fn attribute_peak_rows_to_py_f32(
    py: Python<'_>,
    regulons: Vec<String>,
    motifs: Vec<Option<String>>,
    peaks: Vec<String>,
    aucs: Vec<f32>,
) -> PyResult<PipelineAttributeRowsF32PyResult> {
    Ok((
        PyList::new(py, regulons.iter().map(String::as_str))?.unbind(),
        option_string_list_to_py(py, motifs)?,
        PyList::new(py, peaks.iter().map(String::as_str))?.unbind(),
        PyArray1::from_vec(py, aucs).unbind(),
    ))
}

fn attribute_peak_rows_to_py_f64(
    py: Python<'_>,
    regulons: Vec<String>,
    motifs: Vec<Option<String>>,
    peaks: Vec<String>,
    aucs: Vec<f64>,
) -> PyResult<PipelineAttributeRowsF64PyResult> {
    Ok((
        PyList::new(py, regulons.iter().map(String::as_str))?.unbind(),
        option_string_list_to_py(py, motifs)?,
        PyList::new(py, peaks.iter().map(String::as_str))?.unbind(),
        PyArray1::from_vec(py, aucs).unbind(),
    ))
}

fn option_string_list_to_py(py: Python<'_>, values: Vec<Option<String>>) -> PyResult<Py<PyList>> {
    let out = PyList::empty(py);
    for value in values {
        match value {
            Some(value) => out.append(value)?,
            None => out.append(py.None())?,
        }
    }
    Ok(out.unbind())
}

type AttributePeakRows<T> = (Vec<String>, Vec<Option<String>>, Vec<String>, Vec<T>);

#[allow(clippy::too_many_arguments)]
fn attribute_peaks_to_cistarget_rows<T: Copy>(
    enriched_regulons: &[String],
    enriched_motifs: Option<&[String]>,
    enriched_aucs: &[T],
    regulon_names: &[String],
    regulon_targets: &[Vec<String>],
    enhancer_genes: &[String],
    enhancer_peaks: &[String],
) -> PyResult<AttributePeakRows<T>> {
    if enriched_regulons.len() != enriched_aucs.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "enriched arrays must have the same length; got regulons={} aucs={}",
            enriched_regulons.len(),
            enriched_aucs.len()
        )));
    }
    if let Some(motifs) = enriched_motifs {
        if motifs.len() != enriched_regulons.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "enriched motifs length {} does not match regulons length {}",
                motifs.len(),
                enriched_regulons.len()
            )));
        }
    }
    if regulon_names.len() != regulon_targets.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "regulon arrays must have the same length; got names={} targets={}",
            regulon_names.len(),
            regulon_targets.len()
        )));
    }
    if enhancer_genes.len() != enhancer_peaks.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "enhancer arrays must have the same length; got genes={} peaks={}",
            enhancer_genes.len(),
            enhancer_peaks.len()
        )));
    }

    let mut tf_target_tfs = Vec::new();
    let mut tf_target_genes = Vec::new();
    for (regulon_name, targets) in regulon_names.iter().zip(regulon_targets) {
        let tf = tf_from_regulon_name_rs(regulon_name);
        for target in targets {
            tf_target_tfs.push(tf.clone());
            tf_target_genes.push(target.clone());
        }
    }
    let enriched_tfs: Vec<String> = enriched_regulons
        .iter()
        .map(|name| tf_from_regulon_name_rs(name))
        .collect();
    let (row_indices, peak_indices) = assemble_gene_peak_attribution(
        &enriched_tfs,
        &tf_target_tfs,
        &tf_target_genes,
        enhancer_genes,
        enhancer_peaks,
    );

    let mut regulons = Vec::with_capacity(row_indices.len());
    let mut motifs = Vec::with_capacity(row_indices.len());
    let mut peaks = Vec::with_capacity(row_indices.len());
    let mut aucs = Vec::with_capacity(row_indices.len());
    for (&row_idx, &peak_idx) in row_indices.iter().zip(&peak_indices) {
        let row_idx = row_idx as usize;
        let peak_idx = peak_idx as usize;
        regulons.push(enriched_regulons[row_idx].clone());
        motifs.push(enriched_motifs.map(|values| values[row_idx].clone()));
        peaks.push(enhancer_peaks[peak_idx].clone());
        aucs.push(enriched_aucs[row_idx]);
    }
    Ok((regulons, motifs, peaks, aucs))
}

fn assemble_gene_peak_attribution(
    enriched_tfs: &[String],
    tf_target_tfs: &[String],
    tf_target_genes: &[String],
    enhancer_genes: &[String],
    enhancer_peaks: &[String],
) -> (Vec<u64>, Vec<u64>) {
    let mut gene_to_peaks: HashMap<&str, Vec<usize>> = HashMap::new();
    let mut seen_gene_peak: HashSet<(&str, &str)> = HashSet::new();
    for (i, (gene, peak)) in enhancer_genes.iter().zip(enhancer_peaks).enumerate() {
        let key = (gene.as_str(), peak.as_str());
        if seen_gene_peak.insert(key) {
            gene_to_peaks.entry(key.0).or_default().push(i);
        }
    }

    let mut tf_to_peaks: HashMap<&str, Vec<usize>> = HashMap::new();
    let mut seen_tf_peak: HashSet<(&str, &str)> = HashSet::new();
    for (tf, gene) in tf_target_tfs.iter().zip(tf_target_genes) {
        let Some(peaks) = gene_to_peaks.get(gene.as_str()) else {
            continue;
        };
        for &peak_idx in peaks {
            let peak = enhancer_peaks[peak_idx].as_str();
            let key = (tf.as_str(), peak);
            if seen_tf_peak.insert(key) {
                tf_to_peaks.entry(key.0).or_default().push(peak_idx);
            }
        }
    }

    let mut row_indices = Vec::new();
    let mut peak_indices = Vec::new();
    for (row_idx, tf) in enriched_tfs.iter().enumerate() {
        let Some(peaks) = tf_to_peaks.get(tf.as_str()) else {
            continue;
        };
        row_indices.reserve(peaks.len());
        peak_indices.reserve(peaks.len());
        for &peak_idx in peaks {
            row_indices.push(row_idx as u64);
            peak_indices.push(peak_idx as u64);
        }
    }
    (row_indices, peak_indices)
}

macro_rules! motif_annotation_prune_rows_filtered_pyfunction {
    ($fn_name:ident, $auc_ty:ty, $nes_ty:ty) => {
        #[pyfunction]
        #[allow(clippy::too_many_arguments)]
        fn $fn_name<'py>(
            py: Python<'py>,
            enriched_regulons: Vec<String>,
            enriched_motifs: Vec<String>,
            enriched_auc: PyReadonlyArray1<'py, $auc_ty>,
            enriched_nes: Option<PyReadonlyArray1<'py, $nes_ty>>,
            annotation_motifs: Vec<String>,
            annotation_tf_values: Vec<String>,
            auc_threshold: Option<f64>,
            nes_threshold: Option<f64>,
            case_sensitive: bool,
        ) -> PyResult<MotifAnnotationPrunePyResult> {
            let enriched_auc = enriched_auc.as_slice().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err("enriched_auc must be contiguous")
            })?;
            let enriched_nes_guard = enriched_nes;
            let enriched_nes = match enriched_nes_guard.as_ref() {
                Some(values) => Some(values.as_slice().map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err("enriched_nes must be contiguous")
                })?),
                None => None,
            };
            if nes_threshold.is_some() && enriched_nes.is_none() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "enriched_nes is required when nes_threshold is set",
                ));
            }
            if enriched_regulons.len() != enriched_motifs.len()
                || enriched_motifs.len() != enriched_auc.len()
            {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "enriched arrays must have the same length; got regulons={}, motifs={}, auc={}",
                    enriched_regulons.len(),
                    enriched_motifs.len(),
                    enriched_auc.len()
                )));
            }
            if let Some(nes) = enriched_nes {
                if nes.len() != enriched_regulons.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "enriched_nes has length {} but enriched_regulons has length {}",
                        nes.len(),
                        enriched_regulons.len()
                    )));
                }
            }
            if annotation_motifs.len() != annotation_tf_values.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "annotation arrays must have the same length; got motifs={} tfs={}",
                    annotation_motifs.len(),
                    annotation_tf_values.len()
                )));
            }

            let (row_indices, tf_values, annotation_tf_values) = py.allow_threads(|| {
                motif_annotation_prune_rows_filtered(
                    &enriched_regulons,
                    &enriched_motifs,
                    enriched_auc,
                    enriched_nes,
                    &annotation_motifs,
                    &annotation_tf_values,
                    auc_threshold,
                    nes_threshold,
                    case_sensitive,
                )
            });
            Ok((
                PyArray1::from_vec(py, row_indices).unbind(),
                PyList::new(py, tf_values.iter().map(String::as_str))?.unbind(),
                PyList::new(py, annotation_tf_values.iter().map(String::as_str))?.unbind(),
            ))
        }
    };
}

motif_annotation_prune_rows_filtered_pyfunction!(
    cistarget_motif_annotation_prune_rows_filtered_f32,
    f32,
    f32
);
motif_annotation_prune_rows_filtered_pyfunction!(
    cistarget_motif_annotation_prune_rows_filtered_f64,
    f64,
    f64
);

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn cistarget_motif_annotation_prune_standard_rows_f32<'py>(
    py: Python<'py>,
    enriched_regulons: Vec<String>,
    enriched_motifs: Vec<String>,
    enriched_auc: PyReadonlyArray1<'py, f32>,
    enriched_nes: Option<PyReadonlyArray1<'py, f32>>,
    annotation_motifs: Vec<String>,
    annotation_tf_values: Vec<String>,
    auc_threshold: Option<f64>,
    nes_threshold: Option<f64>,
    case_sensitive: bool,
) -> PyResult<MotifAnnotationPruneRowsF32PyResult> {
    let enriched_auc = enriched_auc
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("enriched_auc must be contiguous"))?;
    let enriched_nes_guard = enriched_nes;
    let enriched_nes = match enriched_nes_guard.as_ref() {
        Some(values) => Some(values.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("enriched_nes must be contiguous")
        })?),
        None => None,
    };
    let (regulons, motifs, aucs, nes, tf_values, annotation_tf_values) =
        py.allow_threads(|| {
            motif_annotation_prune_standard_rows(
                &enriched_regulons,
                &enriched_motifs,
                enriched_auc,
                enriched_nes,
                &annotation_motifs,
                &annotation_tf_values,
                auc_threshold,
                nes_threshold,
                case_sensitive,
            )
        })?;
    motif_annotation_rows_to_py_f32(
        py,
        regulons,
        motifs,
        aucs,
        nes,
        tf_values,
        annotation_tf_values,
    )
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn cistarget_motif_annotation_prune_standard_rows_f64<'py>(
    py: Python<'py>,
    enriched_regulons: Vec<String>,
    enriched_motifs: Vec<String>,
    enriched_auc: PyReadonlyArray1<'py, f64>,
    enriched_nes: Option<PyReadonlyArray1<'py, f64>>,
    annotation_motifs: Vec<String>,
    annotation_tf_values: Vec<String>,
    auc_threshold: Option<f64>,
    nes_threshold: Option<f64>,
    case_sensitive: bool,
) -> PyResult<MotifAnnotationPruneRowsF64PyResult> {
    let enriched_auc = enriched_auc
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("enriched_auc must be contiguous"))?;
    let enriched_nes_guard = enriched_nes;
    let enriched_nes = match enriched_nes_guard.as_ref() {
        Some(values) => Some(values.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("enriched_nes must be contiguous")
        })?),
        None => None,
    };
    let (regulons, motifs, aucs, nes, tf_values, annotation_tf_values) =
        py.allow_threads(|| {
            motif_annotation_prune_standard_rows(
                &enriched_regulons,
                &enriched_motifs,
                enriched_auc,
                enriched_nes,
                &annotation_motifs,
                &annotation_tf_values,
                auc_threshold,
                nes_threshold,
                case_sensitive,
            )
        })?;
    motif_annotation_rows_to_py_f64(
        py,
        regulons,
        motifs,
        aucs,
        nes,
        tf_values,
        annotation_tf_values,
    )
}

fn motif_annotation_rows_to_py_f32(
    py: Python<'_>,
    regulons: Vec<String>,
    motifs: Vec<String>,
    aucs: Vec<f32>,
    nes: Option<Vec<f32>>,
    tf_values: Vec<String>,
    annotation_tf_values: Vec<String>,
) -> PyResult<MotifAnnotationPruneRowsF32PyResult> {
    Ok((
        PyList::new(py, regulons.iter().map(String::as_str))?.unbind(),
        PyList::new(py, motifs.iter().map(String::as_str))?.unbind(),
        PyArray1::from_vec(py, aucs).unbind(),
        nes.map(|values| PyArray1::from_vec(py, values).unbind()),
        PyList::new(py, tf_values.iter().map(String::as_str))?.unbind(),
        PyList::new(py, annotation_tf_values.iter().map(String::as_str))?.unbind(),
    ))
}

fn motif_annotation_rows_to_py_f64(
    py: Python<'_>,
    regulons: Vec<String>,
    motifs: Vec<String>,
    aucs: Vec<f64>,
    nes: Option<Vec<f64>>,
    tf_values: Vec<String>,
    annotation_tf_values: Vec<String>,
) -> PyResult<MotifAnnotationPruneRowsF64PyResult> {
    Ok((
        PyList::new(py, regulons.iter().map(String::as_str))?.unbind(),
        PyList::new(py, motifs.iter().map(String::as_str))?.unbind(),
        PyArray1::from_vec(py, aucs).unbind(),
        nes.map(|values| PyArray1::from_vec(py, values).unbind()),
        PyList::new(py, tf_values.iter().map(String::as_str))?.unbind(),
        PyList::new(py, annotation_tf_values.iter().map(String::as_str))?.unbind(),
    ))
}

#[pyfunction]
fn pipeline_filter_cistarget_peak_rows_f32<'py>(
    py: Python<'py>,
    row_regulons: Vec<String>,
    row_motifs: Vec<String>,
    row_peaks: Vec<String>,
    row_aucs: PyReadonlyArray1<'py, f32>,
    keep_regulons: Vec<String>,
    keep_motifs: Vec<String>,
) -> PyResult<PipelineAttributeRowsF32PyResult> {
    let row_aucs = row_aucs
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("row_aucs must be contiguous"))?;
    let (regulons, motifs, peaks, aucs) = py.allow_threads(|| {
        filter_cistarget_peak_row_values(
            &row_regulons,
            &row_motifs,
            &row_peaks,
            row_aucs,
            &keep_regulons,
            &keep_motifs,
        )
    })?;
    region_cistarget_rows_to_py_f32(py, regulons, motifs, peaks, aucs)
}

#[pyfunction]
fn pipeline_filter_cistarget_peak_rows_f64<'py>(
    py: Python<'py>,
    row_regulons: Vec<String>,
    row_motifs: Vec<String>,
    row_peaks: Vec<String>,
    row_aucs: PyReadonlyArray1<'py, f64>,
    keep_regulons: Vec<String>,
    keep_motifs: Vec<String>,
) -> PyResult<PipelineAttributeRowsF64PyResult> {
    let row_aucs = row_aucs
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("row_aucs must be contiguous"))?;
    let (regulons, motifs, peaks, aucs) = py.allow_threads(|| {
        filter_cistarget_peak_row_values(
            &row_regulons,
            &row_motifs,
            &row_peaks,
            row_aucs,
            &keep_regulons,
            &keep_motifs,
        )
    })?;
    region_cistarget_rows_to_py_f64(py, regulons, motifs, peaks, aucs)
}

fn filter_cistarget_peak_row_values<T: Copy>(
    row_regulons: &[String],
    row_motifs: &[String],
    row_peaks: &[String],
    row_aucs: &[T],
    keep_regulons: &[String],
    keep_motifs: &[String],
) -> PyResult<RegionCistargetRows<T>> {
    if row_regulons.len() != row_motifs.len()
        || row_regulons.len() != row_peaks.len()
        || row_regulons.len() != row_aucs.len()
    {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "row arrays must have the same length; got regulons={} motifs={} peaks={} aucs={}",
            row_regulons.len(),
            row_motifs.len(),
            row_peaks.len(),
            row_aucs.len()
        )));
    }
    if keep_regulons.len() != keep_motifs.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "keep arrays must have the same length; got regulons={} motifs={}",
            keep_regulons.len(),
            keep_motifs.len()
        )));
    }
    let keep: HashSet<(&str, &str)> = keep_regulons
        .iter()
        .zip(keep_motifs)
        .map(|(regulon, motif)| (regulon.as_str(), motif.as_str()))
        .collect();
    let mut regulons = Vec::new();
    let mut motifs = Vec::new();
    let mut peaks = Vec::new();
    let mut aucs = Vec::new();
    for (idx, (regulon, motif)) in row_regulons.iter().zip(row_motifs).enumerate() {
        if keep.contains(&(regulon.as_str(), motif.as_str())) {
            regulons.push(regulon.clone());
            motifs.push(motif.clone());
            peaks.push(row_peaks[idx].clone());
            aucs.push(row_aucs[idx]);
        }
    }
    Ok((regulons, motifs, peaks, aucs))
}

type MotifAnnotationRows<T> = (
    Vec<String>,
    Vec<String>,
    Vec<T>,
    Option<Vec<T>>,
    Vec<String>,
    Vec<String>,
);

#[allow(clippy::too_many_arguments)]
fn motif_annotation_prune_standard_rows<T>(
    enriched_regulons: &[String],
    enriched_motifs: &[String],
    enriched_auc: &[T],
    enriched_nes: Option<&[T]>,
    annotation_motifs: &[String],
    annotation_tf_values: &[String],
    auc_threshold: Option<f64>,
    nes_threshold: Option<f64>,
    case_sensitive: bool,
) -> PyResult<MotifAnnotationRows<T>>
where
    T: Copy + Into<f64>,
{
    if enriched_regulons.len() != enriched_motifs.len()
        || enriched_motifs.len() != enriched_auc.len()
    {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "enriched arrays must have the same length; got regulons={} motifs={} auc={}",
            enriched_regulons.len(),
            enriched_motifs.len(),
            enriched_auc.len()
        )));
    }
    if let Some(nes) = enriched_nes {
        if nes.len() != enriched_regulons.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "enriched_nes has length {} but enriched_regulons has length {}",
                nes.len(),
                enriched_regulons.len()
            )));
        }
    }
    if nes_threshold.is_some() && enriched_nes.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "enriched_nes is required when nes_threshold is set",
        ));
    }
    if annotation_motifs.len() != annotation_tf_values.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "annotation arrays must have the same length; got motifs={} tfs={}",
            annotation_motifs.len(),
            annotation_tf_values.len()
        )));
    }

    let (row_indices, tf_values, annotation_tf_values) = motif_annotation_prune_rows_filtered(
        enriched_regulons,
        enriched_motifs,
        enriched_auc,
        enriched_nes,
        annotation_motifs,
        annotation_tf_values,
        auc_threshold,
        nes_threshold,
        case_sensitive,
    );

    let mut regulons = Vec::with_capacity(row_indices.len());
    let mut motifs = Vec::with_capacity(row_indices.len());
    let mut aucs = Vec::with_capacity(row_indices.len());
    let mut nes_values = enriched_nes.map(|_| Vec::with_capacity(row_indices.len()));
    for row_idx in row_indices {
        let row_idx = row_idx as usize;
        regulons.push(enriched_regulons[row_idx].clone());
        motifs.push(enriched_motifs[row_idx].clone());
        aucs.push(enriched_auc[row_idx]);
        if let (Some(source_nes), Some(out_nes)) = (enriched_nes, nes_values.as_mut()) {
            out_nes.push(source_nes[row_idx]);
        }
    }
    Ok((
        regulons,
        motifs,
        aucs,
        nes_values,
        tf_values,
        annotation_tf_values,
    ))
}

#[allow(clippy::too_many_arguments)]
fn motif_annotation_prune_rows_filtered<T, U>(
    enriched_regulons: &[String],
    enriched_motifs: &[String],
    enriched_auc: &[T],
    enriched_nes: Option<&[U]>,
    annotation_motifs: &[String],
    annotation_tf_values: &[String],
    auc_threshold: Option<f64>,
    nes_threshold: Option<f64>,
    case_sensitive: bool,
) -> (Vec<u64>, Vec<String>, Vec<String>)
where
    T: Copy + Into<f64>,
    U: Copy + Into<f64>,
{
    let annotation_rows_vec =
        normalise_motif_annotations_rs(annotation_motifs, annotation_tf_values, case_sensitive);
    let mut annotation_rows: HashMap<(&str, &str), Vec<usize>> = HashMap::new();
    for (i, row) in annotation_rows_vec.iter().enumerate() {
        annotation_rows
            .entry((row.motif.as_str(), row.tf_key.as_str()))
            .or_default()
            .push(i);
    }

    let mut row_indices = Vec::new();
    let mut tf_values = Vec::new();
    let mut annotation_tf_values = Vec::new();
    for (row_idx, (regulon, motif)) in enriched_regulons.iter().zip(enriched_motifs).enumerate() {
        if let Some(threshold) = auc_threshold {
            let auc = enriched_auc[row_idx].into();
            if auc.is_nan() || auc < threshold {
                continue;
            }
        }
        if let Some(threshold) = nes_threshold {
            let nes = enriched_nes.expect("validated enriched_nes")[row_idx].into();
            if nes.is_nan() || nes < threshold {
                continue;
            }
        }

        let tf = tf_from_regulon_name_rs(regulon);
        let tf_key = if case_sensitive {
            tf.clone()
        } else {
            tf.to_lowercase()
        };
        let Some(matches) = annotation_rows.get(&(motif.as_str(), tf_key.as_str())) else {
            continue;
        };
        row_indices.reserve(matches.len());
        tf_values.reserve(matches.len());
        annotation_tf_values.reserve(matches.len());
        for &annotation_idx in matches {
            row_indices.push(row_idx as u64);
            tf_values.push(tf.clone());
            annotation_tf_values.push(annotation_rows_vec[annotation_idx].tf.clone());
        }
    }
    (row_indices, tf_values, annotation_tf_values)
}

struct AnnotationRow {
    motif: String,
    tf_key: String,
    tf: String,
}

fn normalise_motif_annotations_rs(
    annotation_motifs: &[String],
    annotation_tf_values: &[String],
    case_sensitive: bool,
) -> Vec<AnnotationRow> {
    let mut rows = Vec::new();
    let mut seen: HashSet<(String, String, String)> = HashSet::new();
    for (motif, tf_values) in annotation_motifs.iter().zip(annotation_tf_values) {
        for tf in split_annotation_tfs_rs(tf_values) {
            let tf_value = tf.to_string();
            let tf_key = if case_sensitive {
                tf_value.clone()
            } else {
                tf_value.to_lowercase()
            };
            if seen.insert((motif.clone(), tf_key.clone(), tf_value.clone())) {
                rows.push(AnnotationRow {
                    motif: motif.clone(),
                    tf_key,
                    tf: tf_value,
                });
            }
        }
    }
    rows
}

fn split_annotation_tfs_rs(value: &str) -> impl Iterator<Item = &str> {
    value
        .split([';', ',', '|', '/'])
        .map(str::trim)
        .filter(|part| !part.is_empty())
}

macro_rules! prune_regulon_targets_pyfunction {
    ($name:ident, $rank_ty:ty) => {
        prune_regulon_targets_pyfunction!($name, $rank_ty, false);
    };
    ($name:ident, $rank_ty:ty, $validate_finite:literal) => {
        #[pyfunction]
        #[pyo3(signature = (rankings, motif_names, gene_names, candidate_names, candidate_genes, pruned_regulons, pruned_motifs, rank_cutoff, min_genes))]
        #[allow(clippy::too_many_arguments)]
        fn $name<'py>(
            py: Python<'py>,
            rankings: PyReadonlyArray2<'py, $rank_ty>,
            motif_names: Vec<String>,
            gene_names: Vec<String>,
            candidate_names: Vec<String>,
            candidate_genes: Vec<Vec<String>>,
            pruned_regulons: Vec<String>,
            pruned_motifs: Vec<String>,
            rank_cutoff: usize,
            min_genes: usize,
        ) -> PyResult<PrunedRegulonTargetsPyResult> {
            let rankings_arr = rankings.as_array();
            let (names, genes) = py.allow_threads(|| -> PyResult<_> {
                if $validate_finite {
                    for value in rankings_arr.iter().copied() {
                        if !(value as f64).is_finite() {
                            return Err(pyo3::exceptions::PyValueError::new_err(
                                "rankings contain NaN or Inf values",
                            ));
                        }
                    }
                }
                prune_regulon_targets_impl(
                    rankings_arr,
                    &motif_names,
                    &gene_names,
                    &candidate_names,
                    &candidate_genes,
                    &pruned_regulons,
                    &pruned_motifs,
                    rank_cutoff,
                    min_genes,
                )
            })?;
            let gene_lists = PyList::empty(py);
            for values in genes {
                gene_lists.append(PyList::new(py, values.iter().map(|s| s.as_str()))?)?;
            }
            Ok((
                PyList::new(py, names.iter().map(|s| s.as_str()))?.unbind(),
                gene_lists.unbind(),
            ))
        }
    };
}

prune_regulon_targets_pyfunction!(cistarget_prune_regulon_targets_i16, i16);
prune_regulon_targets_pyfunction!(cistarget_prune_regulon_targets_i32, i32);
prune_regulon_targets_pyfunction!(cistarget_prune_regulon_targets_i64, i64);
prune_regulon_targets_pyfunction!(cistarget_prune_regulon_targets_f32, f32, true);
prune_regulon_targets_pyfunction!(cistarget_prune_regulon_targets_f64, f64, true);

#[pyfunction]
fn cistarget_prune_regulon_targets_unranked<'py>(
    py: Python<'py>,
    candidate_names: Vec<String>,
    candidate_genes: Vec<Vec<String>>,
    pruned_regulons: Vec<String>,
    min_genes: usize,
) -> PyResult<PrunedRegulonTargetsPyResult> {
    let (names, genes) = py.allow_threads(|| {
        prune_regulon_targets_unranked_impl(
            &candidate_names,
            &candidate_genes,
            &pruned_regulons,
            min_genes,
        )
    })?;
    let gene_lists = PyList::empty(py);
    for values in genes {
        gene_lists.append(PyList::new(py, values.iter().map(|s| s.as_str()))?)?;
    }
    Ok((
        PyList::new(py, names.iter().map(|s| s.as_str()))?.unbind(),
        gene_lists.unbind(),
    ))
}

fn prune_regulon_targets_unranked_impl(
    candidate_names: &[String],
    candidate_genes: &[Vec<String>],
    pruned_regulons: &[String],
    min_genes: usize,
) -> PyResult<(Vec<String>, Vec<Vec<String>>)> {
    if candidate_names.len() != candidate_genes.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "candidate_names length {} does not match candidate_genes length {}",
            candidate_names.len(),
            candidate_genes.len()
        )));
    }
    let candidate_genes = deduplicate_candidate_gene_lists(candidate_genes);

    let candidate_to_idx: HashMap<&str, usize> = candidate_names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), i))
        .collect();
    let mut seen = HashSet::new();
    let mut names = Vec::new();
    let mut genes = Vec::new();
    for regulon in pruned_regulons {
        let Some(&candidate_idx) = candidate_to_idx.get(regulon.as_str()) else {
            continue;
        };
        if !seen.insert(candidate_idx) || candidate_genes[candidate_idx].len() < min_genes {
            continue;
        }
        names.push(candidate_names[candidate_idx].clone());
        genes.push(candidate_genes[candidate_idx].clone());
    }
    Ok((names, genes))
}

#[allow(clippy::too_many_arguments)]
fn prune_regulon_targets_impl<T: Copy + PartialOrd>(
    rankings: ndarray::ArrayView2<'_, T>,
    motif_names: &[String],
    gene_names: &[String],
    candidate_names: &[String],
    candidate_genes: &[Vec<String>],
    pruned_regulons: &[String],
    pruned_motifs: &[String],
    rank_cutoff: usize,
    min_genes: usize,
) -> PyResult<(Vec<String>, Vec<Vec<String>>)> {
    let n_motifs = rankings.shape()[0];
    let n_genes = rankings.shape()[1];
    if motif_names.len() != n_motifs {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "motif_names length {} does not match ranking rows {}",
            motif_names.len(),
            n_motifs
        )));
    }
    if gene_names.len() != n_genes {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "gene_names length {} does not match ranking columns {}",
            gene_names.len(),
            n_genes
        )));
    }
    if candidate_names.len() != candidate_genes.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "candidate_names length {} does not match candidate_genes length {}",
            candidate_names.len(),
            candidate_genes.len()
        )));
    }
    let candidate_genes = deduplicate_candidate_gene_lists(candidate_genes);
    if pruned_regulons.len() != pruned_motifs.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "pruned_regulons length {} does not match pruned_motifs length {}",
            pruned_regulons.len(),
            pruned_motifs.len()
        )));
    }

    let motif_to_idx: HashMap<&str, usize> = motif_names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), i))
        .collect();
    let gene_to_idx: HashMap<&str, usize> = gene_names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), i))
        .collect();
    let candidate_to_idx: HashMap<&str, usize> = candidate_names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), i))
        .collect();
    let candidate_gene_cols: Vec<Vec<Option<usize>>> = candidate_genes
        .iter()
        .map(|genes| {
            genes
                .iter()
                .map(|gene| gene_to_idx.get(gene.as_str()).copied())
                .collect()
        })
        .collect();

    let mut top_cache: HashMap<usize, HashSet<usize>> = HashMap::new();
    let mut kept: Vec<HashSet<usize>> = vec![HashSet::new(); candidate_names.len()];
    let mut seen_output = vec![false; candidate_names.len()];
    let mut output_order = Vec::new();
    for (regulon, motif) in pruned_regulons.iter().zip(pruned_motifs) {
        let Some(&candidate_idx) = candidate_to_idx.get(regulon.as_str()) else {
            continue;
        };
        let Some(&motif_idx) = motif_to_idx.get(motif.as_str()) else {
            continue;
        };
        if !seen_output[candidate_idx] {
            seen_output[candidate_idx] = true;
            output_order.push(candidate_idx);
        }
        let top_genes = top_cache
            .entry(motif_idx)
            .or_insert_with(|| top_ranked_gene_set(rankings.row(motif_idx), rank_cutoff));
        for (gene_pos, gene_col) in candidate_gene_cols[candidate_idx].iter().enumerate() {
            if gene_col.is_some_and(|col| top_genes.contains(&col)) {
                kept[candidate_idx].insert(gene_pos);
            }
        }
    }

    let mut names = Vec::new();
    let mut genes = Vec::new();
    for candidate_idx in output_order {
        if kept[candidate_idx].len() < min_genes {
            continue;
        }
        names.push(candidate_names[candidate_idx].clone());
        genes.push(
            candidate_genes[candidate_idx]
                .iter()
                .enumerate()
                .filter(|(gene_pos, _)| kept[candidate_idx].contains(gene_pos))
                .map(|(_, gene)| gene.clone())
                .collect(),
        );
    }
    Ok((names, genes))
}

fn deduplicate_candidate_gene_lists(candidate_genes: &[Vec<String>]) -> Vec<Vec<String>> {
    candidate_genes
        .iter()
        .map(|genes| {
            let mut seen = HashSet::new();
            let mut deduped = Vec::with_capacity(genes.len());
            for gene in genes {
                if seen.insert(gene.as_str()) {
                    deduped.push(gene.clone());
                }
            }
            deduped
        })
        .collect()
}

fn top_ranked_gene_set<T: Copy + PartialOrd>(
    row: ndarray::ArrayView1<'_, T>,
    rank_cutoff: usize,
) -> HashSet<usize> {
    let rank_cutoff = rank_cutoff.min(row.len());
    if rank_cutoff == 0 {
        return HashSet::new();
    }
    let mut order: Vec<usize> = (0..row.len()).collect();
    if rank_cutoff < order.len() {
        order.select_nth_unstable_by(rank_cutoff, |&a, &b| compare_rank_asc(row, a, b));
    } else {
        order.sort_by(|&a, &b| compare_rank_asc(row, a, b));
    }
    order.truncate(rank_cutoff);
    order.into_iter().collect()
}

fn compare_rank_asc<T: Copy + PartialOrd>(
    row: ndarray::ArrayView1<'_, T>,
    a: usize,
    b: usize,
) -> std::cmp::Ordering {
    match row[a]
        .partial_cmp(&row[b])
        .unwrap_or(std::cmp::Ordering::Equal)
    {
        std::cmp::Ordering::Equal => a.cmp(&b),
        other => other,
    }
}

#[pyfunction]
fn pipeline_peak_regulons_and_features_from_edges<'py>(
    py: Python<'py>,
    grn_tfs: Vec<String>,
    grn_targets: Vec<String>,
    enhancer_genes: Vec<String>,
    enhancer_peaks: Vec<String>,
) -> PyResult<PeakRegulonsWithFeaturesPyResult> {
    if grn_tfs.len() != grn_targets.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "grn_tfs length {} does not match grn_targets length {}",
            grn_tfs.len(),
            grn_targets.len()
        )));
    }
    if enhancer_genes.len() != enhancer_peaks.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "enhancer_genes length {} does not match enhancer_peaks length {}",
            enhancer_genes.len(),
            enhancer_peaks.len()
        )));
    }
    let (names, peaks, features) = py.allow_threads(|| {
        build_peak_regulons_and_features_from_edges(
            &grn_tfs,
            &grn_targets,
            &enhancer_genes,
            &enhancer_peaks,
        )
    });
    let peak_lists = PyList::empty(py);
    for values in peaks {
        peak_lists.append(PyList::new(py, values.iter().map(|s| s.as_str()))?)?;
    }
    Ok((
        PyList::new(py, names.iter().map(|s| s.as_str()))?.unbind(),
        peak_lists.unbind(),
        PyList::new(py, features.iter().map(|s| s.as_str()))?.unbind(),
    ))
}

fn build_peak_regulons_and_features_from_edges(
    grn_tfs: &[String],
    grn_targets: &[String],
    enhancer_genes: &[String],
    enhancer_peaks: &[String],
) -> (Vec<String>, Vec<Vec<String>>, Vec<String>) {
    let mut peaks_by_gene: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut seen_gene_peak: HashMap<&str, HashSet<&str>> = HashMap::new();
    for (gene, peak) in enhancer_genes.iter().zip(enhancer_peaks) {
        if seen_gene_peak
            .entry(gene.as_str())
            .or_default()
            .insert(peak.as_str())
        {
            peaks_by_gene
                .entry(gene.as_str())
                .or_default()
                .push(peak.as_str());
        }
    }

    let mut tf_order = Vec::new();
    let mut targets_by_tf: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut seen_tf_target: HashMap<&str, HashSet<&str>> = HashMap::new();
    for (tf, target) in grn_tfs.iter().zip(grn_targets) {
        let seen_targets = seen_tf_target.entry(tf.as_str()).or_default();
        if seen_targets.is_empty() {
            tf_order.push(tf.as_str());
        }
        if seen_targets.insert(target.as_str()) {
            targets_by_tf
                .entry(tf.as_str())
                .or_default()
                .push(target.as_str());
        }
    }

    let mut names = Vec::new();
    let mut peak_lists = Vec::new();
    let mut features = HashSet::new();
    for tf in tf_order {
        let mut peaks = Vec::new();
        let mut seen_peaks = HashSet::new();
        if let Some(targets) = targets_by_tf.get(tf) {
            for target in targets {
                if let Some(target_peaks) = peaks_by_gene.get(target) {
                    for &peak in target_peaks {
                        if seen_peaks.insert(peak) {
                            features.insert(peak);
                            peaks.push(peak.to_string());
                        }
                    }
                }
            }
        }
        if !peaks.is_empty() {
            names.push(format!("{tf}_regulon"));
            peak_lists.push(peaks);
        }
    }
    let mut features: Vec<String> = features.into_iter().map(str::to_string).collect();
    features.sort();
    (names, peak_lists, features)
}

#[pyfunction]
fn pipeline_project_ranking_columns<'py>(
    py: Python<'py>,
    columns: Vec<String>,
    requested_features: Vec<String>,
    motif_col: Option<String>,
) -> PyResult<PipelineRankingColumnProjectionPyResult> {
    let (keep, examples) = py.allow_threads(|| {
        build_ranking_column_projection_names(&columns, &requested_features, motif_col.as_deref())
    });
    Ok((
        PyList::new(py, keep.iter().map(String::as_str))?.unbind(),
        PyList::new(py, examples.iter().map(String::as_str))?.unbind(),
    ))
}

fn build_ranking_column_projection(
    columns: &[String],
    requested_features: &[String],
    motif_col: Option<&str>,
) -> (Vec<u64>, Vec<String>) {
    let features: HashSet<&str> = requested_features.iter().map(String::as_str).collect();
    let motif_idx = motif_col.and_then(|name| columns.iter().position(|col| col == name));

    let mut feature_indices = Vec::new();
    for (idx, col) in columns.iter().enumerate() {
        if motif_col.is_some_and(|name| col == name) {
            continue;
        }
        if features.contains(col.as_str()) {
            feature_indices.push(idx as u64);
        }
    }

    let mut examples: Vec<String> = features.into_iter().map(str::to_string).collect();
    examples.sort();
    examples.truncate(5);

    if feature_indices.is_empty() {
        return (Vec::new(), examples);
    }

    let mut indices = Vec::with_capacity(feature_indices.len() + usize::from(motif_idx.is_some()));
    if let Some(idx) = motif_idx {
        indices.push(idx as u64);
    }
    indices.extend(feature_indices);
    (indices, examples)
}

fn build_ranking_column_projection_names(
    columns: &[String],
    requested_features: &[String],
    motif_col: Option<&str>,
) -> (Vec<String>, Vec<String>) {
    let (indices, examples) =
        build_ranking_column_projection(columns, requested_features, motif_col);
    let keep = indices
        .into_iter()
        .map(|idx| columns[idx as usize].clone())
        .collect();
    (keep, examples)
}

/// Build top-target candidate regulons directly from GRN edge columns.
///
/// This mirrors pandas:
/// ``grn.sort_values("importance", ascending=False, kind="mergesort")``
/// followed by ``groupby("TF", sort=False).head(top_targets)``. Equal
/// importances keep original row order; NaNs sort last like pandas'
/// default ``na_position='last'``.
#[pyfunction]
fn pipeline_candidate_regulons_from_grn<'py>(
    py: Python<'py>,
    grn_tfs: Vec<String>,
    grn_targets: Vec<String>,
    importances: PyReadonlyArray1<'py, f64>,
    top_targets: usize,
    min_targets: usize,
) -> PyResult<CandidateRegulonsPyResult> {
    let importances = importances
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("importances must be contiguous"))?;
    if grn_tfs.len() != grn_targets.len() || grn_tfs.len() != importances.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "GRN edge arrays must have the same length; got TFs={}, targets={}, importances={}",
            grn_tfs.len(),
            grn_targets.len(),
            importances.len()
        )));
    }

    let (names, target_lists) = py.allow_threads(|| {
        build_candidate_regulons_from_grn(
            &grn_tfs,
            &grn_targets,
            importances,
            top_targets,
            min_targets,
        )
    });
    let target_lists_py = PyList::empty(py);
    for targets in target_lists {
        target_lists_py.append(PyList::new(py, targets.iter().map(String::as_str))?)?;
    }
    Ok((
        PyList::new(py, names.iter().map(String::as_str))?.unbind(),
        target_lists_py.unbind(),
    ))
}

fn build_candidate_regulons_from_grn(
    grn_tfs: &[String],
    grn_targets: &[String],
    importances: &[f64],
    top_targets: usize,
    min_targets: usize,
) -> (Vec<String>, Vec<Vec<String>>) {
    if top_targets == 0 || grn_tfs.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let mut order: Vec<usize> = (0..importances.len()).collect();
    order.sort_by(|&a, &b| compare_importance_desc(importances[a], importances[b], a, b));

    let mut counts: HashMap<&str, usize> = HashMap::new();
    let mut tf_order: Vec<&str> = Vec::new();
    let mut targets_by_tf: HashMap<&str, Vec<&str>> = HashMap::new();
    for idx in order {
        let tf = grn_tfs[idx].as_str();
        let count = counts.entry(tf).or_insert(0);
        if *count >= top_targets {
            continue;
        }
        if *count == 0 {
            tf_order.push(tf);
        }
        targets_by_tf
            .entry(tf)
            .or_default()
            .push(grn_targets[idx].as_str());
        *count += 1;
    }

    let mut names = Vec::new();
    let mut target_lists = Vec::new();
    for tf in tf_order {
        let Some(targets) = targets_by_tf.get(tf) else {
            continue;
        };
        if targets.len() < min_targets {
            continue;
        }
        names.push(format!("{tf}_regulon"));
        target_lists.push(targets.iter().map(|target| (*target).to_string()).collect());
    }
    (names, target_lists)
}

fn compare_importance_desc(a: f64, b: f64, a_idx: usize, b_idx: usize) -> std::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => a_idx.cmp(&b_idx),
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => b
            .partial_cmp(&a)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a_idx.cmp(&b_idx)),
    }
}

#[cfg(test)]
mod enhancer_tests {
    use super::*;

    fn dense_pearson(x: &[f32], y: &[f32]) -> f32 {
        let n = x.len() as f64;
        let x_sum: f64 = x.iter().map(|&v| v as f64).sum();
        let y_sum: f64 = y.iter().map(|&v| v as f64).sum();
        let x_mean = x_sum / n;
        let y_mean = y_sum / n;
        let mut num = 0.0;
        let mut x_ss = 0.0;
        let mut y_ss = 0.0;
        for (&xv, &yv) in x.iter().zip(y) {
            let xc = xv as f64 - x_mean;
            let yc = yv as f64 - y_mean;
            num += xc * yc;
            x_ss += xc * xc;
            y_ss += yc * yc;
        }
        let denom = x_ss.sqrt() * y_ss.sqrt();
        if denom > 0.0 {
            (num / denom) as f32
        } else {
            0.0
        }
    }

    #[test]
    fn enhancer_link_sort_is_descending_abs_corr_with_stable_indices() {
        let mut links = vec![
            EnhancerPearsonLink {
                peak_idx: 1,
                gene_idx: 2,
                distance: 0,
                correlation: -0.2,
            },
            EnhancerPearsonLink {
                peak_idx: 0,
                gene_idx: 3,
                distance: 0,
                correlation: 0.8,
            },
            EnhancerPearsonLink {
                peak_idx: 0,
                gene_idx: 1,
                distance: 0,
                correlation: -0.8,
            },
        ];

        sort_enhancer_links_by_abs_corr(&mut links);

        assert_eq!(
            links
                .iter()
                .map(|link| (link.peak_idx, link.gene_idx, link.correlation))
                .collect::<Vec<_>>(),
            vec![(0, 1, -0.8), (0, 3, 0.8), (1, 2, -0.2)]
        );
    }

    #[test]
    fn enhancer_pearson_sparse_matches_dense_reference() {
        // RNA is 4 cells x 2 genes, row-major.
        let rna = vec![1.0, 4.0, 2.0, 3.0, 3.0, 2.0, 4.0, 1.0];
        // One sparse peak with dense vector [0, 2, 0, 4].
        let indptr = vec![0, 2];
        let indices = vec![1, 3];
        let data = vec![2.0, 4.0];
        let links = compute_enhancer_pearson_links(
            &rna,
            4,
            2,
            &indptr,
            &indices,
            &data,
            &[1],
            &[10],
            &[20],
            &[1, 1],
            &[10, 20],
            &[0, 1],
            None,
            100,
            0.0,
        );
        assert_eq!(links.len(), 2);
        let x_dense = [0.0, 2.0, 0.0, 4.0];
        let y0 = [1.0, 2.0, 3.0, 4.0];
        let y1 = [4.0, 3.0, 2.0, 1.0];
        assert!((links[0].correlation - dense_pearson(&x_dense, &y0)).abs() < 1e-6);
        assert!((links[1].correlation - dense_pearson(&x_dense, &y1)).abs() < 1e-6);
    }

    #[test]
    fn enhancer_pearson_selected_peak_cols_use_original_csc() {
        let rna = vec![1.0, 4.0, 2.0, 3.0, 3.0, 2.0, 4.0, 1.0];
        // Two ATAC peaks in one original CSC. Select only the second peak,
        // whose dense vector is [0, 2, 0, 4].
        let indptr = vec![0, 2, 4];
        let indices = vec![0, 2, 1, 3];
        let data = vec![9.0, 9.0, 2.0, 4.0];
        let links = compute_enhancer_pearson_links(
            &rna,
            4,
            2,
            &indptr,
            &indices,
            &data,
            &[1],
            &[10],
            &[20],
            &[1, 1],
            &[10, 20],
            &[0, 1],
            Some(&[1]),
            100,
            0.0,
        );

        assert_eq!(links.len(), 2);
        assert!(links.iter().all(|link| link.peak_idx == 0));
        let x_dense = [0.0, 2.0, 0.0, 4.0];
        let y0 = [1.0, 2.0, 3.0, 4.0];
        let y1 = [4.0, 3.0, 2.0, 1.0];
        assert!((links[0].correlation - dense_pearson(&x_dense, &y0)).abs() < 1e-6);
        assert!((links[1].correlation - dense_pearson(&x_dense, &y1)).abs() < 1e-6);
    }

    #[test]
    fn enhancer_sparse_rna_matches_dense_reference() {
        let rna = vec![1.0, 4.0, 0.0, 3.0, 3.0, 0.0, 4.0, 1.0];
        let rna_indptr = vec![0, 3, 6];
        let rna_indices = vec![0, 2, 3, 0, 1, 3];
        let rna_data = vec![1.0, 3.0, 4.0, 4.0, 3.0, 1.0];
        let indptr = vec![0, 2];
        let indices = vec![1, 3];
        let data = vec![2.0, 4.0];
        let dense_links = compute_enhancer_pearson_links(
            &rna,
            4,
            2,
            &indptr,
            &indices,
            &data,
            &[1],
            &[10],
            &[20],
            &[1, 1],
            &[10, 20],
            &[0, 1],
            None,
            100,
            0.0,
        );
        let sparse_links = compute_enhancer_pearson_links_sparse_rna(
            4,
            &rna_indptr,
            &rna_indices,
            &rna_data,
            &indptr,
            &indices,
            &data,
            &[1],
            &[10],
            &[20],
            &[1, 1],
            &[10, 20],
            &[0, 1],
            None,
            100,
            0.0,
        );

        assert_eq!(sparse_links.len(), dense_links.len());
        for (sparse, dense) in sparse_links.iter().zip(&dense_links) {
            assert_eq!(sparse.peak_idx, dense.peak_idx);
            assert_eq!(sparse.gene_idx, dense.gene_idx);
            assert_eq!(sparse.distance, dense.distance);
            assert!((sparse.correlation - dense.correlation).abs() < 1e-6);
        }
    }

    #[test]
    fn enhancer_pearson_respects_tss_window_and_threshold() {
        let rna = vec![1.0, 4.0, 2.0, 3.0, 3.0, 2.0, 4.0, 1.0];
        let links = compute_enhancer_pearson_links(
            &rna,
            4,
            2,
            &[0, 2],
            &[1, 3],
            &[2.0, 4.0],
            &[1],
            &[10],
            &[20],
            &[1, 1],
            &[10, 200],
            &[0, 1],
            None,
            10,
            0.2,
        );
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].gene_idx, 0);
    }
}

fn row_ptr_any_to_usize(
    row_ptr: &Bound<'_, PyAny>,
    col_idx_len: usize,
    counts_len: usize,
) -> PyResult<Vec<usize>> {
    if let Ok(arr) = row_ptr.extract::<PyReadonlyArray1<'_, i32>>() {
        return row_ptr_values_to_usize(
            arr.as_array()
                .iter()
                .enumerate()
                .map(|(i, &raw)| (i, raw as i128)),
            col_idx_len,
            counts_len,
        );
    }
    if let Ok(arr) = row_ptr.extract::<PyReadonlyArray1<'_, i64>>() {
        return row_ptr_values_to_usize(
            arr.as_array()
                .iter()
                .enumerate()
                .map(|(i, &raw)| (i, raw as i128)),
            col_idx_len,
            counts_len,
        );
    }
    if let Ok(arr) = row_ptr.extract::<PyReadonlyArray1<'_, u64>>() {
        return row_ptr_values_to_usize(
            arr.as_array()
                .iter()
                .enumerate()
                .map(|(i, &raw)| (i, raw as i128)),
            col_idx_len,
            counts_len,
        );
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "row_ptr must be a one-dimensional int32, int64, or uint64 numpy array",
    ))
}

fn row_ptr_values_to_usize<I>(
    values: I,
    col_idx_len: usize,
    counts_len: usize,
) -> PyResult<Vec<usize>>
where
    I: IntoIterator<Item = (usize, i128)>,
{
    let mut out = Vec::new();
    let mut prev = 0usize;
    for (i, raw) in values {
        let v = usize::try_from(raw).map_err(|_| {
            pyo3::exceptions::PyOverflowError::new_err(format!(
                "row_ptr[{i}] = {raw} is negative or does not fit in usize"
            ))
        })?;
        if i == 0 && v != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "row_ptr[0] must be 0, got {v}"
            )));
        }
        if i > 0 && v < prev {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "row_ptr must be non-decreasing; row_ptr[{i}] = {v} < {prev}"
            )));
        }
        prev = v;
        out.push(v);
    }

    if out.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "row_ptr must contain at least one element",
        ));
    }

    let last = *out.last().expect("checked non-empty row_ptr");
    if last != col_idx_len || last != counts_len {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "row_ptr[-1] = {last} but col_idx has length {col_idx_len} and counts has length {counts_len}"
        )));
    }
    Ok(out)
}

fn validate_topic_csr(
    row_ptr: &[usize],
    col_idx: &[i32],
    counts: &[f32],
    n_words: usize,
    require_integer_counts: bool,
) -> PyResult<()> {
    if row_ptr.len() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "CSR matrix must contain at least one row",
        ));
    }
    if let Some((i, &w)) = col_idx
        .iter()
        .enumerate()
        .find(|(_, &w)| w < 0 || (w as usize) >= n_words)
    {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "col_idx[{i}] = {w} is negative or exceeds n_words = {n_words}"
        )));
    }
    if let Some((i, &v)) = counts.iter().enumerate().find(|(_, &v)| !v.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "input counts must be finite; counts[{i}] = {v}"
        )));
    }
    if let Some((i, &v)) = counts.iter().enumerate().find(|(_, &v)| v < 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "input counts must be non-negative; counts[{i}] = {v}"
        )));
    }
    if require_integer_counts {
        if let Some((i, &v)) = counts
            .iter()
            .enumerate()
            .find(|(_, &v)| (v - v.round()).abs() > 1e-6)
        {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Gibbs LDA counts must be integer values; counts[{i}] = {v}"
            )));
        }
    }
    Ok(())
}

/// Build a cells x peaks sparse matrix from a 10x fragments file and a peak BED.
///
/// Returns a tuple:
///   (data, indices, indptr, shape, barcodes, peaks, qc_fragments_per_cell,
///    qc_total_counts_per_cell, total_fragment_records,
///    median_fragments_per_cell_or_nan)
///
/// The first four elements can feed `scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)`
/// directly. `barcodes` is the cell-row ordering; `peaks` is the peak-column ordering
/// (matches the input BED order). QC arrays are parallel to `barcodes`.
///
/// Paths accept both `.tsv`/`.bed` plain files and `.gz` compressed.
fn median_u32(values: &[u32]) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        Some(sorted[mid] as f32)
    } else {
        Some((sorted[mid - 1] as f32 + sorted[mid] as f32) / 2.0)
    }
}

#[pyfunction]
#[pyo3(signature = (fragments_path, peaks_path, cell_barcodes=None))]
fn preproc_fragments_to_matrix<'py>(
    py: Python<'py>,
    fragments_path: PathBuf,
    peaks_path: PathBuf,
    cell_barcodes: Option<Vec<String>>,
) -> PyResult<FragmentsToMatrixPyResult> {
    let (csr, barcodes, peak_names, fpc, tcc, total_fragment_records, median_fragments) = py
        .allow_threads(|| -> anyhow::Result<_> {
            let fragments = read_fragments(&fragments_path)?;
            let qc = barcode_qc_counts(&fragments, fragments.n_barcodes() > 100_000);
            let peaks = read_peaks(&peaks_path)?;
            let (csr, bnames, pnames) = match &cell_barcodes {
                Some(requested) => {
                    build_cell_peak_matrix_for_barcodes(&fragments, &peaks, requested)
                }
                None => build_cell_peak_matrix(&fragments, &peaks),
            };
            let (fragments_per_barcode, total_counts_per_barcode, median_out) =
                if cell_barcodes.is_some() {
                    let barcode_to_idx: HashMap<&str, usize> = fragments
                        .barcode_names
                        .iter()
                        .enumerate()
                        .map(|(idx, name)| (name.as_str(), idx))
                        .collect();
                    let mut selected_fpc = Vec::with_capacity(bnames.len());
                    let mut selected_tcc = Vec::with_capacity(bnames.len());
                    for name in &bnames {
                        if let Some(&idx) = barcode_to_idx.get(name.as_str()) {
                            selected_fpc.push(qc.fragments_per_barcode[idx]);
                            selected_tcc.push(qc.total_counts_per_barcode[idx]);
                        }
                    }
                    let median = if selected_fpc.len() > 100_000 {
                        median_u32(&selected_fpc).unwrap_or(f32::NAN)
                    } else {
                        f32::NAN
                    };
                    (selected_fpc, selected_tcc, median)
                } else {
                    (
                        qc.fragments_per_barcode,
                        qc.total_counts_per_barcode,
                        qc.median_fragments_per_barcode.unwrap_or(f32::NAN),
                    )
                };
            Ok((
                csr,
                bnames,
                pnames,
                fragments_per_barcode,
                total_counts_per_barcode,
                qc.total_fragment_records,
                median_out,
            ))
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let data = PyArray1::from_vec(py, csr.data);
    let indices = PyArray1::from_vec(py, csr.indices);
    let indptr = PyArray1::from_vec(py, csr.indptr);
    let shape = (csr.n_rows, csr.n_cols);
    let barcodes_py = PyList::new(py, barcodes.iter().map(|s| s.as_str()))?;
    let peaks_py = PyList::new(py, peak_names.iter().map(|s| s.as_str()))?;
    let fpc_arr = PyArray1::from_vec(py, fpc);
    let tcc_arr = PyArray1::from_vec(py, tcc);
    Ok((
        data.unbind(),
        indices.unbind(),
        indptr.unbind(),
        shape,
        barcodes_py.unbind(),
        peaks_py.unbind(),
        fpc_arr.unbind(),
        tcc_arr.unbind(),
        total_fragment_records,
        median_fragments,
    ))
}

/// Read a peak BED and return coordinates aligned to a requested peak-name axis.
#[pyfunction]
#[pyo3(signature = (peaks_path, peak_names))]
fn preproc_peak_coords_for_names<'py>(
    py: Python<'py>,
    peaks_path: PathBuf,
    peak_names: Vec<String>,
) -> PyResult<PreprocPeakCoordsPyResult> {
    let (ids, chroms, starts, ends) = py
        .allow_threads(|| -> anyhow::Result<_> {
            let peaks = read_peaks(&peaks_path)?;
            let mut first_by_name: HashMap<&str, usize> = HashMap::with_capacity(peaks.name.len());
            for (idx, name) in peaks.name.iter().enumerate() {
                first_by_name.entry(name.as_str()).or_insert(idx);
            }

            let mut ids = Vec::new();
            let mut chroms = Vec::new();
            let mut starts = Vec::new();
            let mut ends = Vec::new();
            for requested in &peak_names {
                if let Some(&idx) = first_by_name.get(requested.as_str()) {
                    let chrom_idx = peaks.chrom_idx[idx] as usize;
                    ids.push(requested.clone());
                    chroms.push(peaks.chrom_names[chrom_idx].clone());
                    starts.push(peaks.start[idx]);
                    ends.push(peaks.end[idx]);
                }
            }
            Ok((ids, chroms, starts, ends))
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let ids_py = PyList::new(py, ids.iter().map(|s| s.as_str()))?;
    let chroms_py = PyList::new(py, chroms.iter().map(|s| s.as_str()))?;
    let starts_arr = PyArray1::from_vec(py, starts);
    let ends_arr = PyArray1::from_vec(py, ends);
    Ok((
        ids_py.unbind(),
        chroms_py.unbind(),
        starts_arr.unbind(),
        ends_arr.unbind(),
    ))
}

/// Per-barcode insert-size distribution from a fragments file.
///
/// Returns a tuple of parallel arrays indexed by barcode:
///   (barcodes, mean, median, n_fragments, sub_nucleosomal,
///    mono_nucleosomal, di_nucleosomal).
#[pyfunction]
#[pyo3(signature = (fragments_path,))]
fn preproc_insert_size_stats<'py>(
    py: Python<'py>,
    fragments_path: PathBuf,
) -> PyResult<InsertSizeStatsPyResult> {
    let (barcodes, means, medians, counts, sub, mono, di) = py
        .allow_threads(|| -> anyhow::Result<_> {
            let fragments = read_fragments(&fragments_path)?;
            let stats = preproc_insert_size_stats_fn(&fragments);
            let mut means = Vec::with_capacity(stats.len());
            let mut medians = Vec::with_capacity(stats.len());
            let mut counts = Vec::with_capacity(stats.len());
            let mut sub = Vec::with_capacity(stats.len());
            let mut mono = Vec::with_capacity(stats.len());
            let mut di = Vec::with_capacity(stats.len());
            for s in &stats {
                means.push(s.mean);
                medians.push(s.median);
                counts.push(s.n_fragments);
                sub.push(s.sub_nucleosomal);
                mono.push(s.mono_nucleosomal);
                di.push(s.di_nucleosomal);
            }
            Ok((
                fragments.barcode_names,
                means,
                medians,
                counts,
                sub,
                mono,
                di,
            ))
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok((
        PyList::new(py, barcodes.iter().map(|s| s.as_str()))?.unbind(),
        PyArray1::from_vec(py, means).unbind(),
        PyArray1::from_vec(py, medians).unbind(),
        PyArray1::from_vec(py, counts).unbind(),
        PyArray1::from_vec(py, sub).unbind(),
        PyArray1::from_vec(py, mono).unbind(),
        PyArray1::from_vec(py, di).unbind(),
    ))
}

/// Per-barcode fraction of fragments in peaks.
#[pyfunction]
#[pyo3(signature = (fragments_path, peaks_path))]
fn preproc_frip<'py>(
    py: Python<'py>,
    fragments_path: PathBuf,
    peaks_path: PathBuf,
) -> PyResult<(Py<PyList>, Py<PyArray1<f32>>)> {
    let (barcodes, scores) = py
        .allow_threads(|| -> anyhow::Result<_> {
            let fragments = read_fragments(&fragments_path)?;
            let peaks = read_peaks(&peaks_path)?;
            let scores = preproc_frip_fn(&fragments, &peaks);
            Ok((fragments.barcode_names, scores))
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok((
        PyList::new(py, barcodes.iter().map(|s| s.as_str()))?.unbind(),
        PyArray1::from_vec(py, scores).unbind(),
    ))
}

/// Per-barcode TSS enrichment score.
///
/// `tss_chroms` and `tss_positions` must be parallel lists; each pair
/// defines one TSS. The chromosome name space must match
/// `fragments.chrom_names` (after normalisation, which the Rust layer
/// handles via `normalise_chrom`).
#[pyfunction]
#[pyo3(signature = (fragments_path, tss_chroms, tss_positions))]
fn preproc_tss_enrichment<'py>(
    py: Python<'py>,
    fragments_path: PathBuf,
    tss_chroms: Vec<String>,
    tss_positions: Vec<u32>,
) -> PyResult<(Py<PyList>, Py<PyArray1<f32>>)> {
    if tss_chroms.len() != tss_positions.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "tss_chroms ({}) and tss_positions ({}) must have the same length",
            tss_chroms.len(),
            tss_positions.len()
        )));
    }
    let tss_sites: Vec<TssSite> = tss_chroms
        .into_iter()
        .zip(tss_positions)
        .map(|(chrom, position)| TssSite { chrom, position })
        .collect();
    let (barcodes, scores) = py
        .allow_threads(|| -> anyhow::Result<_> {
            let fragments = read_fragments(&fragments_path)?;
            let scores = preproc_tss_enrichment_fn(&fragments, &tss_sites);
            Ok((fragments.barcode_names, scores))
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok((
        PyList::new(py, barcodes.iter().map(|s| s.as_str()))?.unbind(),
        PyArray1::from_vec(py, scores).unbind(),
    ))
}

/// Iterative density-window consensus peak calling from pseudobulked fragments.
///
/// `cluster_per_barcode` must have length `n_barcodes` (queried via the
/// fragment file on the Rust side). Entries in `[0, n_clusters)` mark the
/// barcode's cluster; `u32::MAX` marks a barcode as unassigned.
/// When `n_clusters` is `None`, Rust derives it from the maximum assigned
/// cluster id without Python building a temporary mask.
///
/// Returns parallel arrays describing the consensus peaks:
///   (chroms, starts, ends, names).
#[pyfunction]
#[pyo3(signature = (
    fragments_path,
    cluster_per_barcode,
    n_clusters = None,
    window_size = 50,
    min_fragments_per_window = 3,
    quantile_threshold = 0.95,
    max_gap = 250,
    peak_half_width = 250,
    cluster_barcodes = None,
))]
#[allow(clippy::too_many_arguments)]
fn preproc_call_peaks<'py>(
    py: Python<'py>,
    fragments_path: PathBuf,
    cluster_per_barcode: &Bound<'py, PyAny>,
    n_clusters: Option<usize>,
    window_size: u32,
    min_fragments_per_window: u32,
    quantile_threshold: f32,
    max_gap: u32,
    peak_half_width: u32,
    cluster_barcodes: Option<Vec<String>>,
) -> PyResult<CallPeaksPyResult> {
    let cfg = PeakCallingConfig {
        window_size,
        min_fragments_per_window,
        quantile_threshold,
        max_gap,
        peak_half_width,
    };
    if let Ok(arr) = cluster_per_barcode.extract::<PyReadonlyArray1<'py, u32>>() {
        let labels = arr.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(
                "cluster_per_barcode must be a contiguous one-dimensional array",
            )
        })?;
        return preproc_call_peaks_with_labels(
            py,
            fragments_path,
            labels,
            cluster_barcodes.as_deref(),
            n_clusters,
            &cfg,
        );
    }

    if let Ok(arr) = cluster_per_barcode.extract::<PyReadonlyArray1<'py, i32>>() {
        let raw = arr.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(
                "cluster_per_barcode must be a contiguous one-dimensional array",
            )
        })?;
        let labels = cluster_labels_i32_to_u32(raw);
        return preproc_call_peaks_with_labels(
            py,
            fragments_path,
            &labels,
            cluster_barcodes.as_deref(),
            n_clusters,
            &cfg,
        );
    }

    if let Ok(arr) = cluster_per_barcode.extract::<PyReadonlyArray1<'py, i64>>() {
        let raw = arr.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(
                "cluster_per_barcode must be a contiguous one-dimensional array",
            )
        })?;
        let labels = cluster_labels_i64_to_u32(raw)?;
        return preproc_call_peaks_with_labels(
            py,
            fragments_path,
            &labels,
            cluster_barcodes.as_deref(),
            n_clusters,
            &cfg,
        );
    }

    if let Ok(arr) = cluster_per_barcode.extract::<PyReadonlyArray1<'py, u64>>() {
        let raw = arr.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(
                "cluster_per_barcode must be a contiguous one-dimensional array",
            )
        })?;
        let labels = cluster_labels_u64_to_u32(raw)?;
        return preproc_call_peaks_with_labels(
            py,
            fragments_path,
            &labels,
            cluster_barcodes.as_deref(),
            n_clusters,
            &cfg,
        );
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "cluster_per_barcode must be a one-dimensional int32, int64, uint32, or uint64 numpy array",
    ))
}

fn preproc_call_peaks_with_labels<'py>(
    py: Python<'py>,
    fragments_path: PathBuf,
    cluster_per_barcode: &[u32],
    cluster_barcodes: Option<&[String]>,
    n_clusters: Option<usize>,
    cfg: &PeakCallingConfig,
) -> PyResult<CallPeaksPyResult> {
    let (chrom_names, starts, ends, names) = py
        .allow_threads(|| -> anyhow::Result<_> {
            let fragments = read_fragments(&fragments_path)?;
            let aligned_labels;
            let labels: &[u32] = if let Some(cluster_barcodes) = cluster_barcodes {
                aligned_labels = align_cluster_labels_to_fragment_barcodes(
                    &fragments.barcode_names,
                    cluster_barcodes,
                    cluster_per_barcode,
                )?;
                &aligned_labels
            } else {
                cluster_per_barcode
            };
            if labels.len() != fragments.n_barcodes() {
                anyhow::bail!(
                    "cluster_per_barcode has length {} but the fragments file \
                     contains {} distinct barcodes",
                    labels.len(),
                    fragments.n_barcodes()
                );
            }
            let n_clusters = match n_clusters {
                Some(value) => value,
                None => labels
                    .iter()
                    .copied()
                    .filter(|&cluster| cluster != u32::MAX)
                    .max()
                    .map_or(0, |cluster| cluster as usize + 1),
            };
            let peaks = call_peaks_from_pseudobulks(&fragments, labels, n_clusters, cfg);
            let chrom_names_out: Vec<String> = peaks
                .chrom_idx
                .iter()
                .map(|&i| peaks.chrom_names[i as usize].clone())
                .collect();
            Ok((chrom_names_out, peaks.start, peaks.end, peaks.name))
        })
        .map_err(|e| {
            let message = e.to_string();
            if message.starts_with("cluster_per_barcode Series") {
                pyo3::exceptions::PyValueError::new_err(message)
            } else {
                pyo3::exceptions::PyRuntimeError::new_err(message)
            }
        })?;
    Ok((
        PyList::new(py, chrom_names.iter().map(|s| s.as_str()))?.unbind(),
        PyArray1::from_vec(py, starts).unbind(),
        PyArray1::from_vec(py, ends).unbind(),
        PyList::new(py, names.iter().map(|s| s.as_str()))?.unbind(),
    ))
}

fn align_cluster_labels_to_fragment_barcodes(
    fragment_barcodes: &[String],
    cluster_barcodes: &[String],
    cluster_labels: &[u32],
) -> anyhow::Result<Vec<u32>> {
    if cluster_barcodes.len() != cluster_labels.len() {
        anyhow::bail!(
            "cluster_per_barcode Series index has length {} but values have length {}",
            cluster_barcodes.len(),
            cluster_labels.len()
        );
    }

    let mut label_by_barcode: HashMap<&str, u32> = HashMap::with_capacity(cluster_barcodes.len());
    for (barcode, &label) in cluster_barcodes.iter().zip(cluster_labels) {
        if label_by_barcode.insert(barcode.as_str(), label).is_some() {
            anyhow::bail!("cluster_per_barcode Series has duplicate barcode index: {barcode}");
        }
    }

    let mut aligned = Vec::with_capacity(fragment_barcodes.len());
    let mut missing = Vec::new();
    for barcode in fragment_barcodes {
        if let Some(&label) = label_by_barcode.get(barcode.as_str()) {
            aligned.push(label);
        } else {
            if missing.len() < 5 {
                missing.push(barcode.as_str());
            }
            aligned.push(u32::MAX);
        }
    }

    if !missing.is_empty() {
        anyhow::bail!(
            "cluster_per_barcode Series is missing fragment barcodes: {:?}",
            missing
        );
    }

    Ok(aligned)
}

fn cluster_labels_i32_to_u32(raw: &[i32]) -> Vec<u32> {
    raw.iter()
        .map(|&cluster| {
            if cluster < 0 {
                u32::MAX
            } else {
                cluster as u32
            }
        })
        .collect()
}

fn cluster_labels_i64_to_u32(raw: &[i64]) -> PyResult<Vec<u32>> {
    raw.iter()
        .enumerate()
        .map(|(idx, &cluster)| {
            if cluster < 0 {
                Ok(u32::MAX)
            } else {
                u32::try_from(cluster).map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "cluster_per_barcode[{idx}]={cluster} exceeds uint32 range"
                    ))
                })
            }
        })
        .collect()
}

fn cluster_labels_u64_to_u32(raw: &[u64]) -> PyResult<Vec<u32>> {
    raw.iter()
        .enumerate()
        .map(|(idx, &cluster)| {
            u32::try_from(cluster).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "cluster_per_barcode[{idx}]={cluster} exceeds uint32 range"
                ))
            })
        })
        .collect()
}

macro_rules! gene_dedupe_dense_pyfunction {
    ($name:ident, $value_ty:ty) => {
        #[pyfunction]
        fn $name<'py>(
            py: Python<'py>,
            expression: PyReadonlyArray2<'py, $value_ty>,
            gene_names: Vec<String>,
        ) -> PyResult<(Py<PyArray2<$value_ty>>, Py<PyList>)> {
            let arr = expression.as_array();
            let n_cells = arr.shape()[0];
            let n_genes = arr.shape()[1];
            if gene_names.len() != n_genes {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "gene_names has length {} but expression has {} columns",
                    gene_names.len(),
                    n_genes
                )));
            }
            let (unique_names, dst_by_src) = dedupe_gene_groups(&gene_names);
            let n_unique = unique_names.len();
            let out = py.allow_threads(|| {
                let mut out = ndarray::Array2::<$value_ty>::zeros((n_cells, n_unique));
                for src in 0..n_genes {
                    let dst = dst_by_src[src];
                    for row in 0..n_cells {
                        out[(row, dst)] += arr[(row, src)];
                    }
                }
                out
            });
            Ok((
                PyArray2::from_owned_array(py, out).unbind(),
                PyList::new(py, unique_names.iter().map(|s| s.as_str()))?.unbind(),
            ))
        }
    };
}

gene_dedupe_dense_pyfunction!(gene_dedupe_dense_f32, f32);
gene_dedupe_dense_pyfunction!(gene_dedupe_dense_f64, f64);
gene_dedupe_dense_pyfunction!(gene_dedupe_dense_i32, i32);
gene_dedupe_dense_pyfunction!(gene_dedupe_dense_i64, i64);
gene_dedupe_dense_pyfunction!(gene_dedupe_dense_u32, u32);
gene_dedupe_dense_pyfunction!(gene_dedupe_dense_u64, u64);

macro_rules! gene_dedupe_sparse_csc_pyfunction {
    ($name:ident, $value_ty:ty) => {
        #[pyfunction]
        fn $name<'py>(
            py: Python<'py>,
            indptr: &Bound<'py, PyAny>,
            indices: PyReadonlyArray1<'py, i32>,
            data: PyReadonlyArray1<'py, $value_ty>,
            n_rows: usize,
            gene_names: Vec<String>,
        ) -> PyResult<GeneDedupeSparsePyResult<$value_ty>> {
            let indices = indices.as_slice().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err("indices must be contiguous")
            })?;
            let data = data
                .as_slice()
                .map_err(|_| pyo3::exceptions::PyValueError::new_err("data must be contiguous"))?;
            if indices.len() != data.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "indices length {} differs from data length {}",
                    indices.len(),
                    data.len()
                )));
            }
            let indptr = row_ptr_any_to_usize(indptr, indices.len(), data.len())?;
            if indptr.len() != gene_names.len() + 1 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "indptr has length {} but expected gene_names + 1 = {}",
                    indptr.len(),
                    gene_names.len() + 1
                )));
            }
            if let Some((i, &row)) = indices
                .iter()
                .enumerate()
                .find(|(_, &row)| row < 0 || (row as usize) >= n_rows)
            {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "indices[{i}] = {row} is negative or exceeds n_rows = {n_rows}"
                )));
            }

            let (unique_names, dst_by_src) = dedupe_gene_groups(&gene_names);
            let (out_data, out_indices, out_indptr) = py.allow_threads(|| {
                dedupe_sparse_csc_columns::<$value_ty>(
                    &indptr,
                    indices,
                    data,
                    &dst_by_src,
                    unique_names.len(),
                )
            })?;
            Ok((
                PyArray1::from_vec(py, out_data).unbind(),
                PyArray1::from_vec(py, out_indices).unbind(),
                PyArray1::from_vec(py, out_indptr).unbind(),
                PyList::new(py, unique_names.iter().map(|s| s.as_str()))?.unbind(),
            ))
        }
    };
}

gene_dedupe_sparse_csc_pyfunction!(gene_dedupe_sparse_csc_f32, f32);
gene_dedupe_sparse_csc_pyfunction!(gene_dedupe_sparse_csc_f64, f64);
gene_dedupe_sparse_csc_pyfunction!(gene_dedupe_sparse_csc_i32, i32);
gene_dedupe_sparse_csc_pyfunction!(gene_dedupe_sparse_csc_i64, i64);
gene_dedupe_sparse_csc_pyfunction!(gene_dedupe_sparse_csc_u32, u32);
gene_dedupe_sparse_csc_pyfunction!(gene_dedupe_sparse_csc_u64, u64);

#[pyfunction]
#[pyo3(signature = (gene_names, max_examples = 3))]
fn gene_duplicate_summary<'py>(
    py: Python<'py>,
    gene_names: Vec<String>,
    max_examples: usize,
) -> PyResult<GeneDuplicateSummaryPyResult> {
    let (duplicate_count, examples) = duplicate_gene_summary_rs(&gene_names, max_examples);
    Ok((
        duplicate_count,
        PyList::new(py, examples.iter().map(String::as_str))?.unbind(),
    ))
}

fn duplicate_gene_summary_rs(gene_names: &[String], max_examples: usize) -> (usize, Vec<String>) {
    let mut counts: HashMap<&str, (usize, usize)> = HashMap::with_capacity(gene_names.len());
    for (idx, name) in gene_names.iter().enumerate() {
        counts
            .entry(name.as_str())
            .and_modify(|(count, _first_idx)| *count += 1)
            .or_insert((1, idx));
    }
    let duplicate_count = gene_names.len().saturating_sub(counts.len());
    if duplicate_count == 0 || max_examples == 0 {
        return (duplicate_count, Vec::new());
    }

    let mut duplicate_rows: Vec<(&str, usize, usize)> = counts
        .iter()
        .filter_map(|(&name, &(count, first_idx))| {
            if count > 1 {
                Some((name, count, first_idx))
            } else {
                None
            }
        })
        .collect();
    duplicate_rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.2.cmp(&b.2)));
    let examples = duplicate_rows
        .into_iter()
        .take(max_examples)
        .map(|(name, _count, _first_idx)| name.to_string())
        .collect();
    (duplicate_count, examples)
}

fn dedupe_gene_groups(gene_names: &[String]) -> (Vec<String>, Vec<usize>) {
    let mut index_of: HashMap<String, usize> = HashMap::new();
    let mut unique_names = Vec::new();
    let mut dst_by_src = Vec::with_capacity(gene_names.len());
    for name in gene_names {
        let dst = if let Some(&idx) = index_of.get(name) {
            idx
        } else {
            let idx = unique_names.len();
            unique_names.push(name.clone());
            index_of.insert(name.clone(), idx);
            idx
        };
        dst_by_src.push(dst);
    }
    (unique_names, dst_by_src)
}

fn dedupe_sparse_csc_columns<T>(
    indptr: &[usize],
    indices: &[i32],
    data: &[T],
    dst_by_src: &[usize],
    n_unique: usize,
) -> PyResult<(Vec<T>, Vec<i32>, Vec<i64>)>
where
    T: Copy + Default + std::ops::AddAssign,
{
    if indptr.len() != dst_by_src.len() + 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "indptr length must equal source columns + 1",
        ));
    }

    let mut sources_by_dst = vec![Vec::<usize>::new(); n_unique];
    for (src, &dst) in dst_by_src.iter().enumerate() {
        sources_by_dst[dst].push(src);
    }

    let mut out_data = Vec::with_capacity(data.len());
    let mut out_indices = Vec::with_capacity(indices.len());
    let mut out_indptr = Vec::with_capacity(n_unique + 1);
    out_indptr.push(0_i64);

    for sources in sources_by_dst {
        if sources.len() == 1 {
            let src = sources[0];
            let start = indptr[src];
            let end = indptr[src + 1];
            out_indices.extend_from_slice(&indices[start..end]);
            out_data.extend_from_slice(&data[start..end]);
        } else {
            let mut by_row: HashMap<i32, T> = HashMap::new();
            for src in sources {
                let start = indptr[src];
                let end = indptr[src + 1];
                for offset in start..end {
                    by_row
                        .entry(indices[offset])
                        .and_modify(|value| *value += data[offset])
                        .or_insert(data[offset]);
                }
            }
            let mut rows: Vec<(i32, T)> = by_row.into_iter().collect();
            rows.sort_unstable_by_key(|(row, _)| *row);
            for (row, value) in rows {
                out_indices.push(row);
                out_data.push(value);
            }
        }
        out_indptr.push(out_data.len() as i64);
    }

    Ok((out_data, out_indices, out_indptr))
}

#[pymodule]
fn _rustscenic(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(grn_infer, m)?)?;
    m.add_function(wrap_pyfunction!(grn_infer_sparse_csc, m)?)?;
    m.add_function(wrap_pyfunction!(pipeline_candidate_regulons_from_grn, m)?)?;
    m.add_function(wrap_pyfunction!(aucell_score, m)?)?;
    m.add_function(wrap_pyfunction!(aucell_score_sparse_csr, m)?)?;
    m.add_function(wrap_pyfunction!(cistarget_enrichment_from_rankings_i16, m)?)?;
    m.add_function(wrap_pyfunction!(cistarget_enrichment_from_rankings_i32, m)?)?;
    m.add_function(wrap_pyfunction!(cistarget_enrichment_from_rankings_i64, m)?)?;
    m.add_function(wrap_pyfunction!(cistarget_rankings_to_i32_f32, m)?)?;
    m.add_function(wrap_pyfunction!(cistarget_rankings_to_i32_f64, m)?)?;
    m.add_function(wrap_pyfunction!(topics_fit, m)?)?;
    m.add_function(wrap_pyfunction!(topics_fit_gibbs, m)?)?;
    m.add_function(wrap_pyfunction!(topics_npmi, m)?)?;
    m.add_function(wrap_pyfunction!(topics_cell_assignment, m)?)?;
    m.add_function(wrap_pyfunction!(specificity_group_codes_with_order, m)?)?;
    m.add_function(wrap_pyfunction!(
        specificity_group_codes_with_numeric_order,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(specificity_rss_f32, m)?)?;
    m.add_function(wrap_pyfunction!(specificity_rss, m)?)?;
    m.add_function(wrap_pyfunction!(specificity_candidate_top_indices_f32, m)?)?;
    m.add_function(wrap_pyfunction!(specificity_candidate_top_indices, m)?)?;
    m.add_function(wrap_pyfunction!(enhancer_align_cell_indices, m)?)?;
    m.add_function(wrap_pyfunction!(enhancer_normalise_chrom_codes, m)?)?;
    m.add_function(wrap_pyfunction!(enhancer_parse_peak_names, m)?)?;
    m.add_function(wrap_pyfunction!(enhancer_match_gene_coords_to_rna, m)?)?;
    m.add_function(wrap_pyfunction!(enhancer_match_peak_coords_to_atac, m)?)?;
    m.add_function(wrap_pyfunction!(enhancer_prepare_gene_order, m)?)?;
    m.add_function(wrap_pyfunction!(enhancer_link_pearson, m)?)?;
    m.add_function(wrap_pyfunction!(enhancer_link_pearson_sparse_rna, m)?)?;
    m.add_function(wrap_pyfunction!(eregulon_assemble_f32, m)?)?;
    m.add_function(wrap_pyfunction!(eregulon_assemble, m)?)?;
    m.add_function(wrap_pyfunction!(eregulon_assemble_summary_f32, m)?)?;
    m.add_function(wrap_pyfunction!(eregulon_assemble_summary, m)?)?;
    m.add_function(wrap_pyfunction!(pipeline_match_atac_cell_indices, m)?)?;
    m.add_function(wrap_pyfunction!(
        pipeline_attribute_peaks_to_cistarget_rows_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        pipeline_attribute_peaks_to_cistarget_rows_f64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        pipeline_expand_region_cistarget_rows_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        pipeline_expand_region_cistarget_rows_f64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        cistarget_motif_annotation_prune_rows_filtered_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        cistarget_motif_annotation_prune_rows_filtered_f64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        cistarget_motif_annotation_prune_standard_rows_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        cistarget_motif_annotation_prune_standard_rows_f64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        pipeline_filter_cistarget_peak_rows_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        pipeline_filter_cistarget_peak_rows_f64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(stage_prepare_regulon_indices, m)?)?;
    m.add_function(wrap_pyfunction!(
        stage_prepare_regulon_indices_with_coverage,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(stage_regulon_coverage_counts, m)?)?;
    m.add_function(wrap_pyfunction!(cistarget_prune_regulon_targets_i32, m)?)?;
    m.add_function(wrap_pyfunction!(cistarget_prune_regulon_targets_i16, m)?)?;
    m.add_function(wrap_pyfunction!(cistarget_prune_regulon_targets_i64, m)?)?;
    m.add_function(wrap_pyfunction!(cistarget_prune_regulon_targets_f32, m)?)?;
    m.add_function(wrap_pyfunction!(cistarget_prune_regulon_targets_f64, m)?)?;
    m.add_function(wrap_pyfunction!(
        cistarget_prune_regulon_targets_unranked,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        pipeline_peak_regulons_and_features_from_edges,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(pipeline_project_ranking_columns, m)?)?;
    m.add_function(wrap_pyfunction!(cistarget_region_attribution_i16, m)?)?;
    m.add_function(wrap_pyfunction!(cistarget_region_attribution_i32, m)?)?;
    m.add_function(wrap_pyfunction!(cistarget_region_attribution_i64, m)?)?;
    m.add_function(wrap_pyfunction!(
        cistarget_region_attribution_peak_values_i16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        cistarget_region_attribution_peak_values_i32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        cistarget_region_attribution_peak_values_i64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(preproc_fragments_to_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(preproc_peak_coords_for_names, m)?)?;
    m.add_function(wrap_pyfunction!(preproc_insert_size_stats, m)?)?;
    m.add_function(wrap_pyfunction!(preproc_frip, m)?)?;
    m.add_function(wrap_pyfunction!(preproc_tss_enrichment, m)?)?;
    m.add_function(wrap_pyfunction!(preproc_call_peaks, m)?)?;
    m.add_function(wrap_pyfunction!(gene_duplicate_summary, m)?)?;
    m.add_function(wrap_pyfunction!(gene_dedupe_dense_f32, m)?)?;
    m.add_function(wrap_pyfunction!(gene_dedupe_dense_f64, m)?)?;
    m.add_function(wrap_pyfunction!(gene_dedupe_dense_i32, m)?)?;
    m.add_function(wrap_pyfunction!(gene_dedupe_dense_i64, m)?)?;
    m.add_function(wrap_pyfunction!(gene_dedupe_dense_u32, m)?)?;
    m.add_function(wrap_pyfunction!(gene_dedupe_dense_u64, m)?)?;
    m.add_function(wrap_pyfunction!(gene_dedupe_sparse_csc_f32, m)?)?;
    m.add_function(wrap_pyfunction!(gene_dedupe_sparse_csc_f64, m)?)?;
    m.add_function(wrap_pyfunction!(gene_dedupe_sparse_csc_i32, m)?)?;
    m.add_function(wrap_pyfunction!(gene_dedupe_sparse_csc_i64, m)?)?;
    m.add_function(wrap_pyfunction!(gene_dedupe_sparse_csc_u32, m)?)?;
    m.add_function(wrap_pyfunction!(gene_dedupe_sparse_csc_u64, m)?)?;
    Ok(())
}
