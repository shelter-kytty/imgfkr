use pyo3::prelude::*;
use pyo3::{pymodule, types::PyModule, PyResult, Python, Bound};
use numpy::ndarray::{self, Array, ArrayView, ArrayViewD, ArrayViewMutD};
use numpy::{IntoPyArray, Ix2, PyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, ToPyArray};
use ndarray::{Dim};
use std::ops::Range;

#[pyfunction]
fn hello(s : String) -> PyResult<()> {
    Ok(println!("{}", s))
}

fn contrast_mask<'a>(greyscale: ArrayView<'a, u8, Ix2>, range: Range<u8>) -> ArrayView<'a, u8, Ix2> {
    let mut result: Array<u8, Ix2> = Array::zeros(greyscale.raw_dim());

    !unimplemented!()
}

#[pyfunction]
#[pyo3(name="contrastMask")]
fn contrast_mask_py<'py>(greyscale: &Bound<'py, PyArray<u8, Ix2>>, min: u8, max: u8) {
    let greyscale = unsafe { greyscale.as_array() };
    contrast_mask(greyscale, Range {start:min, end:max});
    !unimplemented!()
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_function(wrap_pyfunction!(contrast_mask_py, m)?)?;
    Ok(())
}
