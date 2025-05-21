use ndarray::{s, Dim, Zip};
use numpy::ndarray::{self, Array, ArrayView};
use numpy::{Ix2, Ix3, PyArray, PyArrayMethods, ToPyArray};
use pyo3::prelude::*;
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};
use std::ops::Range;

fn h_sort<'a>(colour: ArrayView<'a, u8, Ix3>, mask: ArrayView<'a, u8, Ix2>) -> Array<usize, Ix2> {
    let mut _result: Array<u8, Ix3> = Array::zeros(colour.raw_dim());

    // TODO: Put this into its own method
    let mut tallies: Array<usize, Ix2> = Array::zeros(mask.raw_dim());

    let mut i = 0;
    for mut row in tallies.rows_mut() {
        let mask_row = mask.row(i);
        i += 1;
        for mut i in 0..row.iter().count() {
            if mask_row[i] == u8::MAX {
                let idx = (&mask_row.slice(s![i..]))
                    .iter()
                    .position(|a| *a == 0)
                    .unwrap_or((&mask_row.slice(s![i..])).iter().count() + 1)
                    - 1;
                row[i] = idx;
                i += idx;

                println!("idx = {}", idx);
            }
        }
    }
    // /TODO


    return tallies;
}

#[pyfunction]
#[pyo3(name = "hSort")]
fn h_sort_py<'py>(
    py: Python<'py>,
    colour: &Bound<'py, PyArray<u8, Ix3>>,
    mask: &Bound<'py, PyArray<u8, Ix2>>,
) -> PyResult<Bound<'py, PyArray<usize, Dim<[usize; 2]>>>> {
    let colour = unsafe { colour.as_array() };
    let mask = unsafe { mask.as_array() };
    Ok(h_sort(colour, mask).to_pyarray(py))
}


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
    m.add_function(wrap_pyfunction!(h_sort_py, m)?)?;
    Ok(())
}
