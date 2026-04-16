//! PyO3 bindings for rustscenic. Python package name: `rustscenic`.
use pyo3::prelude::*;

#[pymodule]
fn _rustscenic(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
