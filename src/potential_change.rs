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

use crate::common::{cmplx_mul, Cmplx};

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
}
