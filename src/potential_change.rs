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

use crate::common::{cmplx_add, cmplx_conj, cmplx_mul, Cmplx, CMPLX_ZERO};

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
}
