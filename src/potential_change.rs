//! Element-wise steps of the evaluation of the change of the local potential.
//!
//! phelel computes the derivative of the local potential with respect to
//! the displacement of an atom by the finite-difference method
//! (`_run_at_atom` in phelel/base/local_potential.py).  For each pair of a
//! displacement and a symmetry operation, the change of the potential is
//! evaluated on the output grid, transformed in spin space, and added to
//! the right-hand side of the normal equations.  The functions here are the
//! element-wise steps of this computation, in the order in which they are
//! called.
//!
//! Each function has a pure-Python counterpart with the same name,
//! arguments and order of the loops in phelel/base/potential_change.py,
//! and the tests of phelel compare the two.  Every function writes into its
//! first argument and allocates no array of the size of its inputs, so that
//! the steps can be chained on work arrays allocated once.

use rayon::prelude::*;

use crate::common::{cmplx_add, cmplx_conj, cmplx_mul, cmplx_neg, cmplx_scale, Cmplx, CMPLX_ZERO};

/// Number of elements given to one rayon task.
const CHUNK_SIZE: usize = 4096;

/// Multiply `phi` by `phase`, element by element.
///
/// This is the multiplication of phi_q(m) by exp(2 pi i q . (N^p)^-1 m) in
/// `FourierSeriesOnGrid.get_phi` of phelel, with both arrays flattened.
///
/// Python counterpart: `_multiply_by_phase_python`.
///
/// # Panics
///
/// If `phi` and `phase` have different lengths.
pub fn multiply_by_phase(phi: &mut [Cmplx], phase: &[Cmplx]) {
    assert_eq!(phi.len(), phase.len(), "phi and phase differ in length");
    phi.par_chunks_mut(CHUNK_SIZE)
        .zip(phase.par_chunks(CHUNK_SIZE))
        .for_each(|(phi_chunk, phase_chunk)| {
            for (phi_value, phase_value) in phi_chunk.iter_mut().zip(phase_chunk) {
                *phi_value = cmplx_mul(*phi_value, *phase_value);
            }
        });
}

/// Rotate the four spin components in spin space, element by element.
///
/// At each element, the four values [v_0, v_1, v_2, v_3] of `values` form
/// the 2x2 matrix V = [[v_0, v_1], [v_2, v_3]], which is replaced by
/// D V D^dagger with D = `spin_rotation`:
///
/// ```text
/// v'_(2 i_sp + i_s) = sum_(i_s1, i_s2) D[i_sp][i_s1] conj(D[i_s][i_s2]) v_(2 i_s1 + i_s2)
/// ```
///
/// The factors D[i_sp][i_s1] conj(D[i_s][i_s2]) are computed first, and the
/// terms are summed in this order of (i_s1, i_s2), as in the Python
/// counterpart.  The loop over the elements is outside, in parallel over
/// chunks, whereas in Python the loops over (i_sp, i_s, i_s1, i_s2) are
/// outside and the elements are handled by NumPy.
///
/// Python counterpart: `_rotate_spin_components_python`.
///
/// # Panics
///
/// If `values` does not have four slices of the same length.
pub fn rotate_spin_components(values: &mut [&mut [Cmplx]], spin_rotation: &[[Cmplx; 2]; 2]) {
    let [v_0, v_1, v_2, v_3] = values else {
        panic!("values must have four spin components");
    };
    let len = v_0.len();
    assert!(
        v_1.len() == len && v_2.len() == len && v_3.len() == len,
        "spin components differ in length"
    );

    // factors[i_sp][i_s][i_s1][i_s2] = D[i_sp][i_s1] conj(D[i_s][i_s2])
    let mut factors = [[[[CMPLX_ZERO; 2]; 2]; 2]; 2];
    for i_sp in 0..2 {
        for i_s in 0..2 {
            for i_s1 in 0..2 {
                for i_s2 in 0..2 {
                    factors[i_sp][i_s][i_s1][i_s2] = cmplx_mul(
                        spin_rotation[i_sp][i_s1],
                        cmplx_conj(spin_rotation[i_s][i_s2]),
                    );
                }
            }
        }
    }

    v_0.par_chunks_mut(CHUNK_SIZE)
        .zip(v_1.par_chunks_mut(CHUNK_SIZE))
        .zip(v_2.par_chunks_mut(CHUNK_SIZE))
        .zip(v_3.par_chunks_mut(CHUNK_SIZE))
        .for_each(|(((chunk_0, chunk_1), chunk_2), chunk_3)| {
            for i in 0..chunk_0.len() {
                let original = [chunk_0[i], chunk_1[i], chunk_2[i], chunk_3[i]];
                let mut rotated = [CMPLX_ZERO; 4];
                for i_sp in 0..2 {
                    for i_s in 0..2 {
                        let mut sum = CMPLX_ZERO;
                        for i_s1 in 0..2 {
                            for i_s2 in 0..2 {
                                let term = cmplx_mul(
                                    factors[i_sp][i_s][i_s1][i_s2],
                                    original[i_s1 * 2 + i_s2],
                                );
                                sum = cmplx_add(sum, term);
                            }
                        }
                        rotated[i_sp * 2 + i_s] = sum;
                    }
                }
                chunk_0[i] = rotated[0];
                chunk_1[i] = rotated[1];
                chunk_2[i] = rotated[2];
                chunk_3[i] = rotated[3];
            }
        });
}

/// Apply time reversal to the spin components, element by element.
///
/// Time reversal is i sigma_y K, under which the 2x2 matrix
/// V = [[v_0, v_1], [v_2, v_3]] of four spin components becomes
/// sigma_y V^* sigma_y:
///
/// ```text
/// [v_0, v_1, v_2, v_3] -> [conj(v_3), -conj(v_2), -conj(v_1), conj(v_0)]
/// ```
///
/// For two components (collinear, up and down), the two are exchanged and
/// conjugated, [v_0, v_1] -> [conj(v_1), conj(v_0)], and for one component
/// it is only conjugated.
///
/// Python counterpart: `_apply_time_reversal_to_spin_components_python`.
///
/// # Panics
///
/// If `values` does not have one, two or four slices of the same length.
pub fn apply_time_reversal_to_spin_components(values: &mut [&mut [Cmplx]]) {
    match values {
        [v_0, v_1, v_2, v_3] => {
            let len = v_0.len();
            assert!(
                v_1.len() == len && v_2.len() == len && v_3.len() == len,
                "spin components differ in length"
            );
            v_0.par_chunks_mut(CHUNK_SIZE)
                .zip(v_1.par_chunks_mut(CHUNK_SIZE))
                .zip(v_2.par_chunks_mut(CHUNK_SIZE))
                .zip(v_3.par_chunks_mut(CHUNK_SIZE))
                .for_each(|(((chunk_0, chunk_1), chunk_2), chunk_3)| {
                    for i in 0..chunk_0.len() {
                        let original = [chunk_0[i], chunk_1[i], chunk_2[i], chunk_3[i]];
                        chunk_0[i] = cmplx_conj(original[3]);
                        chunk_1[i] = cmplx_neg(cmplx_conj(original[2]));
                        chunk_2[i] = cmplx_neg(cmplx_conj(original[1]));
                        chunk_3[i] = cmplx_conj(original[0]);
                    }
                });
        }
        [v_0, v_1] => {
            assert!(v_1.len() == v_0.len(), "spin components differ in length");
            v_0.par_chunks_mut(CHUNK_SIZE)
                .zip(v_1.par_chunks_mut(CHUNK_SIZE))
                .for_each(|(chunk_0, chunk_1)| {
                    for i in 0..chunk_0.len() {
                        let original = [chunk_0[i], chunk_1[i]];
                        chunk_0[i] = cmplx_conj(original[1]);
                        chunk_1[i] = cmplx_conj(original[0]);
                    }
                });
        }
        [v_0] => {
            v_0.par_chunks_mut(CHUNK_SIZE).for_each(|chunk_0| {
                for value in chunk_0.iter_mut() {
                    *value = cmplx_conj(*value);
                }
            });
        }
        _ => panic!("values must have one, two or four spin components"),
    }
}

/// Add the outer products of `displacement` and the spin components to
/// `normal_equation_rhs`.
///
/// For each spin component c, each Cartesian direction i = 0, 1, 2 of the
/// displacement and each grid point i_grid,
///
/// ```text
/// normal_equation_rhs[c][i][i_grid] += displacement[i] * values[c][i_grid].
/// ```
///
/// In the finite-difference method of phelel (`_run_at_atom`),
/// `displacement` is a rotated displacement R u_k and `values` are the spin
/// components of the rotated change of the potential dV_kg at the grid
/// points.  One call adds one term; the calls for all pairs (k, g) give the
/// right-hand side U^T dV of the normal equations.
///
/// # Arguments
///
/// * `normal_equation_rhs` - Right-hand side of the normal equations, to
///   which the products are added.  The elements [c][i][i_grid] for the
///   spin components c, the directions i = 0, 1, 2 and the grid points
///   i_grid are flattened in this order.
/// * `displacement` - Displacement in Cartesian coordinates, R u_k in
///   `_run_at_atom`.
/// * `values` - Spin components of the potential change, one slice for each
///   component, each with one value for each grid point.
///
/// Python counterpart: `_add_to_normal_equation_rhs_python`.
///
/// # Panics
///
/// If the spin components of `values` differ in length, or if the length of
/// `normal_equation_rhs` is not 3 * `values.len()` times the length of a
/// component.
pub fn add_to_normal_equation_rhs(
    normal_equation_rhs: &mut [Cmplx],
    displacement: &[f64; 3],
    values: &[&[Cmplx]],
) {
    let num_grid_points = values.first().map_or(0, |value| value.len());
    assert!(
        values.iter().all(|value| value.len() == num_grid_points),
        "spin components differ in length"
    );
    assert_eq!(
        normal_equation_rhs.len(),
        values.len() * 3 * num_grid_points,
        "normal_equation_rhs does not have the length of the outer products"
    );
    if num_grid_points == 0 {
        return;
    }
    normal_equation_rhs
        .par_chunks_mut(num_grid_points)
        .enumerate()
        .for_each(|(row, rhs_row)| {
            let value = values[row / 3];
            let displacement_component = displacement[row % 3];
            rhs_row
                .par_chunks_mut(CHUNK_SIZE)
                .zip(value.par_chunks(CHUNK_SIZE))
                .for_each(|(rhs_chunk, value_chunk)| {
                    for (rhs_value, element) in rhs_chunk.iter_mut().zip(value_chunk) {
                        *rhs_value =
                            cmplx_add(*rhs_value, cmplx_scale(displacement_component, *element));
                    }
                });
        });
}

/// Multiply `normal_equation_rhs` by pinv(U^T U) along the direction axis.
///
/// For each spin component c and each grid point i_grid, the three values
/// `normal_equation_rhs[c][0..3][i_grid]`, one for each Cartesian direction
/// of the displacement, form a vector, which is replaced by its product with
/// `inverse_normal_matrix`:
///
/// ```text
/// normal_equation_rhs[c][i][i_grid]
///     <- sum_j inverse_normal_matrix[i][j] normal_equation_rhs[c][j][i_grid],
/// ```
///
/// with the sum in the order of j.  In the finite-difference method of
/// phelel (`_run_at_atom`), this turns the right-hand side of the normal
/// equations into dV/du.
///
/// # Arguments
///
/// * `normal_equation_rhs` - Right-hand side of the normal equations,
///   overwritten by the products (dV/du in `_run_at_atom`).  The elements
///   [c][i][i_grid] for the spin components c, the directions i = 0, 1, 2
///   and the grid points i_grid are flattened in this order.
/// * `inverse_normal_matrix` - Pseudo-inverse pinv(U^T U) of the matrix
///   U^T U of the normal equations, where the rows of U are the rotated
///   displacements.
/// * `num_grid_points` - Number of grid points.
///
/// Python counterpart: `_apply_inverse_normal_matrix_python`.
///
/// # Panics
///
/// If `num_grid_points` is zero, or if the length of `normal_equation_rhs`
/// is not a multiple of 3 * `num_grid_points`.
pub fn apply_inverse_normal_matrix(
    normal_equation_rhs: &mut [Cmplx],
    inverse_normal_matrix: &[[f64; 3]; 3],
    num_grid_points: usize,
) {
    assert!(
        num_grid_points > 0 && normal_equation_rhs.len() % (3 * num_grid_points) == 0,
        "normal_equation_rhs is not made of three vectors of length num_grid_points"
    );
    normal_equation_rhs
        .chunks_mut(3 * num_grid_points)
        .for_each(|component| {
            let (row_0, rest) = component.split_at_mut(num_grid_points);
            let (row_1, row_2) = rest.split_at_mut(num_grid_points);
            row_0
                .par_chunks_mut(CHUNK_SIZE)
                .zip(row_1.par_chunks_mut(CHUNK_SIZE))
                .zip(row_2.par_chunks_mut(CHUNK_SIZE))
                .for_each(|((chunk_0, chunk_1), chunk_2)| {
                    for i_grid in 0..chunk_0.len() {
                        let original = [chunk_0[i_grid], chunk_1[i_grid], chunk_2[i_grid]];
                        let mut product = [CMPLX_ZERO; 3];
                        for i in 0..3 {
                            for j in 0..3 {
                                product[i] = cmplx_add(
                                    product[i],
                                    cmplx_scale(inverse_normal_matrix[i][j], original[j]),
                                );
                            }
                        }
                        chunk_0[i_grid] = product[0];
                        chunk_1[i_grid] = product[1];
                        chunk_2[i_grid] = product[2];
                    }
                });
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random complex values in [-0.5, 0.5).
    fn random_values(len: usize, seed: u64) -> Vec<Cmplx> {
        let mut state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        };
        (0..len).map(|_| [next(), next()]).collect()
    }

    /// The length is not a multiple of CHUNK_SIZE, so that the last chunk is
    /// shorter than the others.
    #[test]
    fn multiply_by_phase_agrees_with_complex_product() {
        let len = 3 * CHUNK_SIZE + 17;
        let phase = random_values(len, 1);
        let original = random_values(len, 2);
        let mut phi = original.clone();
        multiply_by_phase(&mut phi, &phase);
        for i in 0..len {
            let [a, b] = original[i];
            let [c, d] = phase[i];
            assert_eq!(phi[i], [a * c - b * d, a * d + b * c]);
        }
    }

    #[test]
    #[should_panic(expected = "phi and phase differ in length")]
    fn multiply_by_phase_rejects_different_lengths() {
        let mut phi = random_values(10, 1);
        let phase = random_values(11, 2);
        multiply_by_phase(&mut phi, &phase);
    }

    /// 2x2 complex matrix multiplication, for the reference below.
    fn matmul_2x2(a: &[[Cmplx; 2]; 2], b: &[[Cmplx; 2]; 2]) -> [[Cmplx; 2]; 2] {
        let mut product = [[CMPLX_ZERO; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                product[i][j] = cmplx_add(cmplx_mul(a[i][0], b[0][j]), cmplx_mul(a[i][1], b[1][j]));
            }
        }
        product
    }

    /// D V D^dagger by 2x2 matrix products at every element.  D is not
    /// diagonal and has complex elements, so that every factor of the sum
    /// matters.  D need not be unitary for the formula to hold.
    #[test]
    fn rotate_spin_components_agrees_with_matrix_products() {
        let len = 2 * CHUNK_SIZE + 5;
        let (c, s) = (0.3f64.cos(), 0.3f64.sin());
        let (cp, sp) = (0.7f64.cos(), 0.7f64.sin());
        let spin_rotation = [[[c * cp, c * sp], [s, 0.0]], [[-s * cp, s * sp], [c, 0.0]]];
        let mut dagger = [[CMPLX_ZERO; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                dagger[i][j] = cmplx_conj(spin_rotation[j][i]);
            }
        }
        let originals: Vec<Vec<Cmplx>> = (0..4).map(|k| random_values(len, k + 10)).collect();
        let mut rotated = originals.clone();
        {
            let mut slices: Vec<&mut [Cmplx]> =
                rotated.iter_mut().map(|v| v.as_mut_slice()).collect();
            rotate_spin_components(&mut slices, &spin_rotation);
        }
        for i in 0..len {
            let matrix = [
                [originals[0][i], originals[1][i]],
                [originals[2][i], originals[3][i]],
            ];
            let expected = matmul_2x2(&matmul_2x2(&spin_rotation, &matrix), &dagger);
            for (k, value) in [
                expected[0][0],
                expected[0][1],
                expected[1][0],
                expected[1][1],
            ]
            .iter()
            .enumerate()
            {
                assert!((rotated[k][i][0] - value[0]).abs() < 1e-14);
                assert!((rotated[k][i][1] - value[1]).abs() < 1e-14);
            }
        }
    }

    #[test]
    #[should_panic(expected = "values must have four spin components")]
    fn rotate_spin_components_rejects_two_components() {
        let mut first = random_values(4, 1);
        let mut second = random_values(4, 2);
        let mut slices: Vec<&mut [Cmplx]> = vec![&mut first, &mut second];
        let identity = [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]];
        rotate_spin_components(&mut slices, &identity);
    }

    /// Apply time reversal to copies of `originals` and return them.
    fn time_reversed(originals: &[Vec<Cmplx>]) -> Vec<Vec<Cmplx>> {
        let mut reversed = originals.to_vec();
        let mut slices: Vec<&mut [Cmplx]> = reversed.iter_mut().map(|v| v.as_mut_slice()).collect();
        apply_time_reversal_to_spin_components(&mut slices);
        reversed
    }

    /// The formulas of the doc comment, for one, two and four components.
    /// The length is not a multiple of CHUNK_SIZE.
    #[test]
    fn apply_time_reversal_to_spin_components_agrees_with_formulas() {
        let len = CHUNK_SIZE + 3;
        let originals: Vec<Vec<Cmplx>> = (0..4).map(|k| random_values(len, k + 20)).collect();
        let conj = |k: usize, i: usize| [originals[k][i][0], -originals[k][i][1]];
        let minus_conj = |k: usize, i: usize| [-originals[k][i][0], originals[k][i][1]];

        let reversed = time_reversed(&originals);
        for i in 0..len {
            assert_eq!(reversed[0][i], conj(3, i));
            assert_eq!(reversed[1][i], minus_conj(2, i));
            assert_eq!(reversed[2][i], minus_conj(1, i));
            assert_eq!(reversed[3][i], conj(0, i));
        }

        let reversed = time_reversed(&originals[..2]);
        for i in 0..len {
            assert_eq!(reversed[0][i], conj(1, i));
            assert_eq!(reversed[1][i], conj(0, i));
        }

        let reversed = time_reversed(&originals[..1]);
        for i in 0..len {
            assert_eq!(reversed[0][i], conj(0, i));
        }
    }

    #[test]
    #[should_panic(expected = "values must have one, two or four spin components")]
    fn apply_time_reversal_to_spin_components_rejects_three_components() {
        let originals: Vec<Vec<Cmplx>> = (0..3).map(|k| random_values(4, k)).collect();
        time_reversed(&originals);
    }

    /// Two additions to a nonzero normal_equation_rhs, with two spin
    /// components, against element-wise sums.
    #[test]
    fn add_to_normal_equation_rhs_agrees_with_elementwise_sums() {
        let len = CHUNK_SIZE + 9;
        let values_0: Vec<Vec<Cmplx>> = (0..2).map(|k| random_values(len, k + 30)).collect();
        let values_1: Vec<Vec<Cmplx>> = (0..2).map(|k| random_values(len, k + 40)).collect();
        let displacement_0 = [0.3, -1.2, 2.5];
        let displacement_1 = [-0.7, 0.4, 1.1];
        let start = random_values(2 * 3 * len, 50);
        let mut normal_equation_rhs = start.clone();
        let additions = [(&displacement_0, &values_0), (&displacement_1, &values_1)];
        for (displacement, values) in additions {
            let slices: Vec<&[Cmplx]> = values.iter().map(|v| v.as_slice()).collect();
            add_to_normal_equation_rhs(&mut normal_equation_rhs, displacement, &slices);
        }
        for c in 0..2 {
            for i in 0..3 {
                for x in 0..len {
                    let index = (c * 3 + i) * len + x;
                    let mut expected = start[index];
                    for (displacement, values) in additions {
                        expected[0] += displacement[i] * values[c][x][0];
                        expected[1] += displacement[i] * values[c][x][1];
                    }
                    assert_eq!(normal_equation_rhs[index], expected);
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "normal_equation_rhs does not have the length of the outer products")]
    fn add_to_normal_equation_rhs_rejects_wrong_length() {
        let values = random_values(5, 1);
        let mut normal_equation_rhs = random_values(2 * 3 * 5, 2);
        add_to_normal_equation_rhs(&mut normal_equation_rhs, &[1.0, 2.0, 3.0], &[&values]);
    }

    /// Matrix-vector products for two spin components, against sums written
    /// out for each element.
    #[test]
    fn apply_inverse_normal_matrix_agrees_with_matrix_vector_products() {
        let len = CHUNK_SIZE + 9;
        let matrix = [[1.5, -0.2, 0.3], [0.7, 2.1, -1.4], [-0.6, 0.8, 0.9]];
        let original = random_values(2 * 3 * len, 60);
        let mut normal_equation_rhs = original.clone();
        apply_inverse_normal_matrix(&mut normal_equation_rhs, &matrix, len);
        for c in 0..2 {
            for i in 0..3 {
                for x in 0..len {
                    let mut expected = [0.0, 0.0];
                    for j in 0..3 {
                        let element = original[(c * 3 + j) * len + x];
                        expected[0] += matrix[i][j] * element[0];
                        expected[1] += matrix[i][j] * element[1];
                    }
                    assert_eq!(normal_equation_rhs[(c * 3 + i) * len + x], expected);
                }
            }
        }
    }

    #[test]
    #[should_panic(
        expected = "normal_equation_rhs is not made of three vectors of length num_grid_points"
    )]
    fn apply_inverse_normal_matrix_rejects_wrong_length() {
        let mut normal_equation_rhs = random_values(10, 1);
        apply_inverse_normal_matrix(&mut normal_equation_rhs, &[[1.0, 0.0, 0.0]; 3], 3);
    }
}
