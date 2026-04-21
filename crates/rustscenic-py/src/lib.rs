//! PyO3 bindings for rustscenic. Python package name: `rustscenic`.
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyList;

use rustscenic_aucell::aucell;
use rustscenic_grn::{infer, Adjacency, GrnConfig};
use rustscenic_preproc::{
    build_cell_peak_matrix, fragments::{fragments_per_barcode, total_counts_per_barcode},
    read_fragments, read_peaks,
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
    let expr_vec: Vec<f32> = arr.as_standard_layout().iter().copied().collect();

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
        py.allow_threads(|| infer(&expr_vec, n_cells, &gene_names, &tf_names, &cfg));

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

    let expr_vec: Vec<f32> = arr.as_standard_layout().iter().copied().collect();
    let regulons: Vec<(String, Vec<usize>)> =
        regulon_names.into_iter().zip(regulon_gene_indices).collect();

    let out: Vec<f32> = py.allow_threads(|| aucell(&expr_vec, n_cells, n_genes, &regulons, top_frac));
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

#[pymodule]
fn _rustscenic(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(grn_infer, m)?)?;
    m.add_function(wrap_pyfunction!(aucell_score, m)?)?;
    m.add_function(wrap_pyfunction!(topics_fit, m)?)?;
    m.add_function(wrap_pyfunction!(preproc_fragments_to_matrix, m)?)?;
    Ok(())
}
