//! PyO3 bindings for rustscenic. Python package name: `rustscenic`.
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyList;

use rustscenic_aucell::aucell;
use rustscenic_grn::{infer, Adjacency, GrnConfig};

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

#[pymodule]
fn _rustscenic(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(grn_infer, m)?)?;
    m.add_function(wrap_pyfunction!(aucell_score, m)?)?;
    Ok(())
}
