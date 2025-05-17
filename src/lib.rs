use ndarray::{Dim, Zip};
use numpy::ndarray::{self, Array, ArrayView};
use numpy::{Ix2, PyArray, PyArrayMethods, ToPyArray};
use pyo3::prelude::*;
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};
use std::ops::Range;

fn contrast_mask<'a>(greyscale: ArrayView<'a, u8, Ix2>, range: Range<u8>) -> Array<u8, Ix2> {
    let mut result: Array<u8, Ix2> = Array::zeros(greyscale.raw_dim());

    Zip::from(greyscale).and(&mut result).for_each(|a, b| {
        *b = if range.contains(a) { u8::MAX } else { 0 };
    });

    return result;
}

#[pyfunction]
#[pyo3(name = "contrastMask")]
fn contrast_mask_py<'py>(
    py: Python<'py>,
    greyscale: &Bound<'py, PyArray<u8, Ix2>>,
    min: u8,
    max: u8,
) -> PyResult<Bound<'py, PyArray<u8, Dim<[usize; 2]>>>> {
    let greyscale = unsafe { greyscale.as_array() };
    Ok(contrast_mask(
        greyscale,
        Range {
            start: min,
            end: max + 1,
        },
    )
    .to_pyarray(py))
}

/// fn name == `lib.name` in `Cargo.toml`
/// declaractions must also exist in `name.pyi`
#[pymodule]
fn core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(contrast_mask_py, m)?)?;
    Ok(())
}
