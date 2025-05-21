use ndarray::{s, Dim, Zip};
use numpy::ndarray::{self, Array, ArrayView, ArrayView2, ArrayViewMut2};
use numpy::{Ix2, Ix3, PyArray, PyArrayMethods, ToPyArray};
use pyo3::prelude::*;
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};
use std::iter::zip;
use std::ops::Range;
use std::vec::Vec;

fn contrast_gradient<'a>(mask: ArrayView<'a, u8, Ix2>) -> Array<usize, Ix2> {
    let mut tallies: Array<usize, Ix2> = Array::zeros(mask.raw_dim());

    for (mut row, mask_row) in zip(tallies.rows_mut(), mask.rows()) {
        for mut i in 0..row.iter().count() {
            if mask_row[i] == u8::MAX {
                let idx = (&mask_row.slice(s![i..]))
                    .iter()
                    .position(|a| *a == 0)
                    .unwrap_or((&mask_row.slice(s![i..])).iter().count() + 1)
                    - 1;
                row[i] = idx;
                i += idx;
            }
        }
    }

    return tallies;
}

#[pyfunction]
#[pyo3(name = "contrastGradient")]
fn contrast_gradient_py<'py>(
    py: Python<'py>,
    mask: &Bound<'py, PyArray<u8, Ix2>>,
) -> PyResult<Bound<'py, PyArray<usize, Dim<[usize; 2]>>>> {
    let mask = unsafe { mask.as_array() };
    Ok(contrast_gradient(mask).to_pyarray(py))
}

fn h_edge_mask<'a>(mask: ArrayView<'a, u8, Ix2>) -> Array<usize, Ix2> {
    let mut tallies: Array<usize, Ix2> = Array::zeros(mask.raw_dim());

    for (mut row, mask_row) in zip(tallies.rows_mut(), mask.rows()) {
        let mut count = 0;
        for i in 0..row.iter().count() {
            if count > 0 {
                count -= 1;
                continue;
            }

            if mask_row[i] == u8::MAX {
                let view = &mask_row.slice(s![i..]);
                let idx = view
                    .iter()
                    .position(|a| *a == 0)
                    .unwrap_or(view.iter().count() + 1)
                    - 1;
                row[i] = idx;
                count += idx;
            }
        }
    }

    return tallies;
}

fn h_sort<'a>(colour: ArrayView<'a, u8, Ix3>, mask: ArrayView<'a, u8, Ix2>) -> Array<u8, Ix3> {
    let mut result: Array<u8, Ix3> = colour.clone().to_owned();

    let edges: Array<usize, Ix2> = h_edge_mask(mask);

    assert!(colour.dim().0 == mask.dim().0);
    assert!(colour.dim().1 == mask.dim().1);

    for (erow, (crow, mut rrow)) in zip(
        edges.rows(),
        zip(colour.outer_iter(), result.outer_iter_mut()),
    ) {
        for i in 0..erow.iter().count() {
            if erow[i] != 0 {
                let view: ArrayView2<u8> = crow.slice(s![i..i + erow[i], ..]);

                let (vector, _) = view.to_owned().into_raw_vec_and_offset();
                let mut sorted: Vec<Vec<u8>> = Vec::new();
                assert!(vector.len() % 3 == 0);
                for i in 0..vector.len() {
                    if i % 3 == 0 {
                        sorted.push(vec![vector[i], vector[i + 1], vector[i + 2]]);
                    }
                }
                // sorting by "most red" atm
                sorted.sort_by(|a, b| (a.first().unwrap()).cmp(&b.first().unwrap()));
                let sorted = Array::from_shape_vec(view.raw_dim(), sorted.concat()).unwrap();

                let mut r: ArrayViewMut2<u8> = rrow.slice_mut(s![i..i + erow[i], ..]);
                
                for (a, b) in zip(r.iter_mut(), sorted.iter()) {
                    *a = *b;
                }
            }
        }
    }

    return result;
}

#[pyfunction]
#[pyo3(name = "hSort")]
fn h_sort_py<'py>(
    py: Python<'py>,
    colour: &Bound<'py, PyArray<u8, Ix3>>,
    mask: &Bound<'py, PyArray<u8, Ix2>>,
) -> PyResult<Bound<'py, PyArray<u8, Dim<[usize; 3]>>>> {
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
    m.add_function(wrap_pyfunction!(contrast_gradient_py, m)?)?;
    Ok(())
}
