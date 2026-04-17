//! PyO3 bindings for rustscenic. Python package name: `rustscenic`.
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyList;

use rustscenic_grn::{infer, Adjacency, GrnConfig};

/// Infer a GRN from a dense (n_cells × n_genes) f32 expression matrix.
///
/// Returns a tuple (tfs, targets, importances) where each element is a
/// list/ndarray aligned across edges. Python wraps this into a DataFrame.
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
    let expr_vec: Vec<f32> = arr
        .as_standard_layout()
        .iter()
        .copied()
        .collect();

    let cfg = GrnConfig {
        n_estimators,
        learning_rate,
        max_features,
        subsample,
        max_depth,
        early_stop_window,
        seed,
    };

    let adjacencies: Vec<Adjacency> = py.allow_threads(|| {
        infer(&expr_vec, n_cells, &gene_names, &tf_names, &cfg)
    });

    let tfs = PyList::new(py, adjacencies.iter().map(|a| a.tf.as_str()))?;
    let targets = PyList::new(py, adjacencies.iter().map(|a| a.target.as_str()))?;
    let imp: Vec<f32> = adjacencies.iter().map(|a| a.importance).collect();
    let imp_arr = PyArray1::from_vec(py, imp);
    Ok((tfs.unbind(), targets.unbind(), imp_arr.unbind()))
}

#[pymodule]
fn _rustscenic(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(grn_infer, m)?)?;
    Ok(())
}
