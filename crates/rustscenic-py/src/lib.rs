//! PyO3 bindings for rustscenic. Python package name: `rustscenic`.
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyList;

use rustscenic_aucell::aucell;
use rustscenic_grn::{infer, Adjacency, GrnConfig};
use rustscenic_preproc::{
    build_cell_peak_matrix,
    call_peaks_from_pseudobulks,
    fragments::{fragments_per_barcode, total_counts_per_barcode},
    frip as preproc_frip_fn,
    insert_size_stats as preproc_insert_size_stats_fn,
    read_fragments, read_peaks,
    tss_enrichment as preproc_tss_enrichment_fn,
    PeakCallingConfig, TssSite,
};
use rustscenic_topics::online_vb_lda;
use std::path::PathBuf;

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
) -> PyResult<(Py<PyList>, Py<PyList>, Py<PyArray1<f32>>)> {
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
    // Borrow the numpy buffer when it's already C-contiguous (the Python
    // wrapper ensures this via np.ascontiguousarray). Falls back to a copy
    // only for non-standard-layout inputs — avoids a doubled peak RSS on
    // 100k × 30k matrices, which is ~12 GB at f32.
    let owned_fallback;
    let expr_slice: &[f32] = if let Some(s) = arr.as_slice() {
        s
    } else {
        owned_fallback = arr.as_standard_layout().to_owned();
        owned_fallback
            .as_slice()
            .expect("standard_layout guarantees contiguous slice")
    };

    let cfg = GrnConfig {
        n_estimators,
        learning_rate,
        max_features,
        subsample,
        max_depth,
        early_stop_window,
        seed,
    };

    let adjacencies: Vec<Adjacency> =
        py.allow_threads(|| infer(expr_slice, n_cells, &gene_names, &tf_names, &cfg));

    let tfs = PyList::new(py, adjacencies.iter().map(|a| a.tf.as_str()))?;
    let targets = PyList::new(py, adjacencies.iter().map(|a| a.target.as_str()))?;
    let imp: Vec<f32> = adjacencies.iter().map(|a| a.importance).collect();
    let imp_arr = PyArray1::from_vec(py, imp);
    Ok((tfs.unbind(), targets.unbind(), imp_arr.unbind()))
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
) -> PyResult<Py<PyArray2<f32>>> {
    let arr = expression.as_array();
    let n_cells = arr.shape()[0];
    let n_genes = arr.shape()[1];

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

    // Borrow the numpy buffer when it's C-contiguous; fall back to a copy
    // only for non-standard-layout inputs. Saves the duplicate-of-the-input
    // allocation on large matrices.
    let owned_fallback;
    let expr_slice: &[f32] = if let Some(s) = arr.as_slice() {
        s
    } else {
        owned_fallback = arr.as_standard_layout().to_owned();
        owned_fallback
            .as_slice()
            .expect("standard_layout guarantees contiguous slice")
    };
    let regulons: Vec<(String, Vec<usize>)> =
        regulon_names.into_iter().zip(regulon_gene_indices).collect();

    let out: Vec<f32> =
        py.allow_threads(|| aucell(expr_slice, n_cells, n_genes, &regulons, top_frac));
    let n_regulons = regulons.len();

    let arr2 = ndarray::Array2::from_shape_vec((n_cells, n_regulons), out)
        .map_err(|e: ndarray::ShapeError| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, arr2).unbind())
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
    row_ptr: Vec<usize>,
    col_idx: Vec<u32>,
    counts: Vec<f32>,
    n_words: usize,
    n_topics: usize,
    alpha: f32,
    eta: f32,
    tau0: f32,
    kappa: f32,
    batch_size: usize,
    n_passes: usize,
    seed: u64,
) -> PyResult<(Py<PyArray2<f32>>, Py<PyArray2<f32>>)> {
    let result = py.allow_threads(|| {
        online_vb_lda(
            &row_ptr, &col_idx, &counts, n_words, n_topics,
            alpha, eta, tau0, kappa, batch_size, n_passes, seed,
        )
    });
    let n_docs = result.n_docs;
    let ct = ndarray::Array2::from_shape_vec((n_docs, n_topics), result.cell_topic)
        .map_err(|e: ndarray::ShapeError| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let tw = ndarray::Array2::from_shape_vec((n_topics, n_words), result.topic_word)
        .map_err(|e: ndarray::ShapeError| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok((PyArray2::from_owned_array(py, ct).unbind(), PyArray2::from_owned_array(py, tw).unbind()))
}

/// Fit collapsed-Gibbs LDA on a sparse (docs x words) matrix.
///
/// The Mallet-class topic model — better topic-coherence (NPMI) on sparse
/// scATAC at K ≥ 30 than online VB, at the cost of thousands of
/// iterations instead of tens of passes. Returns (theta, beta) as two
/// numpy arrays of shape (n_docs, n_topics) and (n_topics, n_words).
#[pyfunction]
#[pyo3(signature = (
    row_ptr, col_idx, counts, n_words,
    n_topics = 50,
    alpha = 0.1,
    eta = 0.01,
    n_iters = 200,
    seed = 42,
))]
#[allow(clippy::too_many_arguments)]
fn topics_fit_gibbs<'py>(
    py: Python<'py>,
    row_ptr: Vec<usize>,
    col_idx: Vec<u32>,
    counts: Vec<f32>,
    n_words: usize,
    n_topics: usize,
    alpha: f32,
    eta: f32,
    n_iters: usize,
    seed: u64,
) -> PyResult<(Py<PyArray2<f32>>, Py<PyArray2<f32>>)> {
    let result = py.allow_threads(|| {
        rustscenic_topics::gibbs::fit(
            &row_ptr, &col_idx, &counts, n_words, n_topics,
            alpha, eta, n_iters, seed,
        )
    });
    let n_docs = row_ptr.len().saturating_sub(1);
    let theta = ndarray::Array2::from_shape_vec((n_docs, n_topics), result.theta)
        .map_err(|e: ndarray::ShapeError| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let beta = ndarray::Array2::from_shape_vec((n_topics, n_words), result.beta)
        .map_err(|e: ndarray::ShapeError| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok((
        PyArray2::from_owned_array(py, theta).unbind(),
        PyArray2::from_owned_array(py, beta).unbind(),
    ))
}

/// Build a cells x peaks sparse matrix from a 10x fragments file and a peak BED.
///
/// Returns a tuple:
///   (data, indices, indptr, shape, barcodes, peaks, qc_fragments_per_cell,
///    qc_total_counts_per_cell)
///
/// The first four elements can feed `scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)`
/// directly. `barcodes` is the cell-row ordering; `peaks` is the peak-column ordering
/// (matches the input BED order). QC arrays are parallel to `barcodes`.
///
/// Paths accept both `.tsv`/`.bed` plain files and `.gz` compressed.
#[pyfunction]
#[pyo3(signature = (fragments_path, peaks_path))]
fn preproc_fragments_to_matrix<'py>(
    py: Python<'py>,
    fragments_path: PathBuf,
    peaks_path: PathBuf,
) -> PyResult<(
    Py<PyArray1<u32>>,   // data
    Py<PyArray1<u32>>,   // indices
    Py<PyArray1<u64>>,   // indptr
    (usize, usize),      // shape (n_cells, n_peaks)
    Py<PyList>,          // barcode names
    Py<PyList>,          // peak names
    Py<PyArray1<u32>>,   // fragments per barcode
    Py<PyArray1<u32>>,   // total counts per barcode
)> {
    let (csr, barcodes, peak_names, fpc, tcc) = py.allow_threads(|| -> anyhow::Result<_> {
        let fragments = read_fragments(&fragments_path)?;
        let fpc = fragments_per_barcode(&fragments);
        let tcc = total_counts_per_barcode(&fragments);
        let peaks = read_peaks(&peaks_path)?;
        let (csr, bnames, pnames) = build_cell_peak_matrix(&fragments, &peaks);
        Ok((csr, bnames, pnames, fpc, tcc))
    }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

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
) -> PyResult<(
    Py<PyList>,          // barcodes
    Py<PyArray1<f32>>,   // mean
    Py<PyArray1<f32>>,   // median
    Py<PyArray1<u32>>,   // n_fragments
    Py<PyArray1<u32>>,   // sub
    Py<PyArray1<u32>>,   // mono
    Py<PyArray1<u32>>,   // di
)> {
    let (barcodes, means, medians, counts, sub, mono, di) =
        py.allow_threads(|| -> anyhow::Result<_> {
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
            Ok((fragments.barcode_names, means, medians, counts, sub, mono, di))
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
///
/// Returns parallel arrays describing the consensus peaks:
///   (chroms, starts, ends, names).
#[pyfunction]
#[pyo3(signature = (
    fragments_path,
    cluster_per_barcode,
    n_clusters,
    window_size = 50,
    min_fragments_per_window = 3,
    quantile_threshold = 0.95,
    max_gap = 250,
    peak_half_width = 250,
))]
#[allow(clippy::too_many_arguments)]
fn preproc_call_peaks<'py>(
    py: Python<'py>,
    fragments_path: PathBuf,
    cluster_per_barcode: Vec<u32>,
    n_clusters: usize,
    window_size: u32,
    min_fragments_per_window: u32,
    quantile_threshold: f32,
    max_gap: u32,
    peak_half_width: u32,
) -> PyResult<(Py<PyList>, Py<PyArray1<u32>>, Py<PyArray1<u32>>, Py<PyList>)> {
    let cfg = PeakCallingConfig {
        window_size,
        min_fragments_per_window,
        quantile_threshold,
        max_gap,
        peak_half_width,
    };
    let (chrom_names, starts, ends, names) = py
        .allow_threads(|| -> anyhow::Result<_> {
            let fragments = read_fragments(&fragments_path)?;
            if cluster_per_barcode.len() != fragments.n_barcodes() {
                anyhow::bail!(
                    "cluster_per_barcode has length {} but the fragments file \
                     contains {} distinct barcodes",
                    cluster_per_barcode.len(),
                    fragments.n_barcodes()
                );
            }
            let peaks = call_peaks_from_pseudobulks(
                &fragments,
                &cluster_per_barcode,
                n_clusters,
                &cfg,
            );
            let chrom_names_out: Vec<String> = peaks
                .chrom_idx
                .iter()
                .map(|&i| peaks.chrom_names[i as usize].clone())
                .collect();
            Ok((chrom_names_out, peaks.start, peaks.end, peaks.name))
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok((
        PyList::new(py, chrom_names.iter().map(|s| s.as_str()))?.unbind(),
        PyArray1::from_vec(py, starts).unbind(),
        PyArray1::from_vec(py, ends).unbind(),
        PyList::new(py, names.iter().map(|s| s.as_str()))?.unbind(),
    ))
}

#[pymodule]
fn _rustscenic(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(grn_infer, m)?)?;
    m.add_function(wrap_pyfunction!(aucell_score, m)?)?;
    m.add_function(wrap_pyfunction!(topics_fit, m)?)?;
    m.add_function(wrap_pyfunction!(topics_fit_gibbs, m)?)?;
    m.add_function(wrap_pyfunction!(preproc_fragments_to_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(preproc_insert_size_stats, m)?)?;
    m.add_function(wrap_pyfunction!(preproc_frip, m)?)?;
    m.add_function(wrap_pyfunction!(preproc_tss_enrichment, m)?)?;
    m.add_function(wrap_pyfunction!(preproc_call_peaks, m)?)?;
    Ok(())
}
