//! Batched complex-Hermitian eigendecomposition of dynamical matrices.
//!
//! Many dynamical matrices are small (`3N x 3N`, `N` = primitive-cell
//! atoms), so the efficient parallelism for a dense q-mesh is *across*
//! matrices, not inside one.  Each matrix is diagonalized
//! single-threaded by faer (`Par::Seq`) while rayon spreads the
//! matrices over cores; `numpy.linalg.eigh` cannot do this because a
//! batched call loops over q sequentially and LAPACK's internal
//! threading is ineffective at this size.
//!
//! This module is numerically a drop-in for `numpy.linalg.eigh`: it
//! returns the raw eigenvalues in nondecreasing order and the
//! eigenvectors as columns, without applying any `sqrt`/sign/unit
//! conversion (that stays on the Python side).  The input matrices are
//! left untouched.

#![allow(dead_code)]

use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::{self_adjoint_evd, self_adjoint_evd_scratch, ComputeEigenvectors};
use faer::mat::{Mat, MatRef};
use faer::{c64, Par};
use rayon::prelude::*;

use crate::common::Cmplx;

/// Reinterpret a `[Cmplx]` (`[f64; 2]`) slice as faer `[c64]`.  `c64` is
/// `num_complex::Complex<f64>` (`repr(C) { re, im }`), layout- and
/// alignment-identical to `[f64; 2]`, so this is a zero-copy view.
#[inline]
fn cmplx_as_c64(s: &[Cmplx]) -> &[c64] {
    // SAFETY: `c64` and `[f64; 2]` have identical size/alignment/layout.
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const c64, s.len()) }
}

/// Per-worker reusable workspace for [`eigsh_one`], sized for an `n x n`
/// matrix.  Allocated once per rayon worker via `for_each_init` so the
/// faer scratch, eigenvalue diagonal, and eigenvector matrix are not
/// re-allocated per matrix.
struct EvdScratch {
    mem: MemBuffer,
    /// Eigenvalues (complex storage; imaginary parts are ~0 for a
    /// Hermitian input).
    s: Diag<c64>,
    /// Eigenvectors as columns.
    u: Mat<c64>,
}

impl EvdScratch {
    fn new(n: usize) -> Self {
        let req = self_adjoint_evd_scratch::<c64>(
            n,
            ComputeEigenvectors::Yes,
            Par::Seq,
            Default::default(),
        );
        Self {
            mem: MemBuffer::new(req),
            s: Diag::zeros(n),
            u: Mat::zeros(n, n),
        }
    }
}

/// Diagonalize one Hermitian matrix `d` (row-major `[n, n]` `Cmplx`).
///
/// Writes eigenvalues into `evals[0..n]` (nondecreasing) and
/// eigenvectors into `evecs` with layout `evecs[component * n + band]`
/// (`[component, band]` row-major), matching the orientation phonopy
/// gets from `numpy.linalg.eigh` and feeds back into
/// `reciprocal_to_normal`.  `d` is only read (and only its lower
/// triangle, as the matrix is Hermitian).
fn eigsh_one(d: &[Cmplx], n: usize, evals: &mut [f64], evecs: &mut [Cmplx], sc: &mut EvdScratch) {
    debug_assert_eq!(d.len(), n * n);
    debug_assert_eq!(evals.len(), n);
    debug_assert_eq!(evecs.len(), n * n);

    let a = MatRef::<c64>::from_row_major_slice(cmplx_as_c64(d), n, n);
    let stack = MemStack::new(&mut sc.mem);
    self_adjoint_evd(
        a,
        sc.s.as_mut(),
        Some(sc.u.as_mut()),
        Par::Seq,
        stack,
        Default::default(),
    )
    .expect("faer self_adjoint_evd failed");

    let s = sc.s.column_vector();
    for i in 0..n {
        evals[i] = s[i].re;
    }
    let u = sc.u.as_ref();
    for comp in 0..n {
        for band in 0..n {
            let z = u[(comp, band)];
            evecs[comp * n + band] = [z.re, z.im];
        }
    }
}

/// Batched Hermitian eigendecomposition, parallel over the `nq` matrices.
///
/// `dynmats` is `[nq, n, n]` row-major Hermitian, `evals` is `[nq, n]`,
/// `evecs` is `[nq, n, n]`.  Each matrix is solved single-threaded; the
/// caller should run this under `py.detach` to release the GIL.
pub fn eigsh_batch(dynmats: &[Cmplx], evals: &mut [f64], evecs: &mut [Cmplx], n: usize) {
    if n == 0 {
        return;
    }
    let m2 = n * n;
    evals
        .par_chunks_mut(n)
        .zip(evecs.par_chunks_mut(m2))
        .zip(dynmats.par_chunks(m2))
        .for_each_init(
            || EvdScratch::new(n),
            |sc, ((ev, vec), d)| eigsh_one(d, n, ev, vec, sc),
        );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Complex conjugate of a `Cmplx`.
    fn conj(a: Cmplx) -> Cmplx {
        [a[0], -a[1]]
    }

    /// `a * b` for `Cmplx` (kept local to avoid pulling in common just
    /// for the tests).
    fn mul(a: Cmplx, b: Cmplx) -> Cmplx {
        [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
    }

    /// Diagonalize a single `n x n` row-major Hermitian matrix.
    fn eigh(d: &[Cmplx], n: usize) -> (Vec<f64>, Vec<Cmplx>) {
        let mut evals = vec![0.0f64; n];
        let mut evecs = vec![[0.0f64; 2]; n * n];
        let mut sc = EvdScratch::new(n);
        eigsh_one(d, n, &mut evals, &mut evecs, &mut sc);
        (evals, evecs)
    }

    #[test]
    fn known_2x2() {
        // A = [[2, 1 - i], [1 + i, 3]]; eigenvalues 1 and 4.
        let d: [Cmplx; 4] = [[2.0, 0.0], [1.0, -1.0], [1.0, 1.0], [3.0, 0.0]];
        let (evals, evecs) = eigh(&d, 2);
        assert!((evals[0] - 1.0).abs() < 1e-12, "{:?}", evals);
        assert!((evals[1] - 4.0).abs() < 1e-12, "{:?}", evals);
        // Each column must satisfy A v = lambda v.
        check_eigenpairs(&d, 2, &evals, &evecs);
        check_orthonormal(2, &evecs);
    }

    #[test]
    fn ascending_order() {
        let d = hermitian_fixture(6);
        let (evals, _) = eigh(&d, 6);
        for w in evals.windows(2) {
            assert!(w[0] <= w[1] + 1e-12, "not ascending: {:?}", evals);
        }
    }

    #[test]
    fn reconstruct_and_unitary_n6() {
        let n = 6;
        let d = hermitian_fixture(n);
        let (evals, evecs) = eigh(&d, n);
        check_eigenpairs(&d, n, &evals, &evecs);
        check_orthonormal(n, &evecs);
    }

    #[test]
    fn batch_matches_single() {
        let n = 4;
        let a = hermitian_fixture(n);
        let b = hermitian_fixture_seed(n, 7);
        let mut dynmats = Vec::new();
        dynmats.extend_from_slice(&a);
        dynmats.extend_from_slice(&b);
        let mut evals = vec![0.0f64; 2 * n];
        let mut evecs = vec![[0.0f64; 2]; 2 * n * n];
        eigsh_batch(&dynmats, &mut evals, &mut evecs, n);

        let (ea, _) = eigh(&a, n);
        let (eb, _) = eigh(&b, n);
        for i in 0..n {
            assert!((evals[i] - ea[i]).abs() < 1e-12);
            assert!((evals[n + i] - eb[i]).abs() < 1e-12);
        }
        check_eigenpairs(&a, n, &evals[..n], &evecs[..n * n]);
        check_eigenpairs(&b, n, &evals[n..], &evecs[n * n..]);
    }

    #[test]
    fn n1() {
        let d: [Cmplx; 1] = [[3.5, 0.0]];
        let (evals, evecs) = eigh(&d, 1);
        assert!((evals[0] - 3.5).abs() < 1e-12);
        assert!((evecs[0][0].hypot(evecs[0][1]) - 1.0).abs() < 1e-12);
    }

    /// Assert `A v_k = lambda_k v_k` for every eigenpair (handles
    /// degenerate subspaces, since each returned column is still an
    /// eigenvector).
    fn check_eigenpairs(d: &[Cmplx], n: usize, evals: &[f64], evecs: &[Cmplx]) {
        for k in 0..n {
            for i in 0..n {
                // (A v)_i = sum_j A[i,j] v[j,k]
                let mut acc = [0.0f64; 2];
                for j in 0..n {
                    let aij = d[i * n + j];
                    let vjk = evecs[j * n + k];
                    let p = mul(aij, vjk);
                    acc[0] += p[0];
                    acc[1] += p[1];
                }
                let vik = evecs[i * n + k];
                let rhs = [evals[k] * vik[0], evals[k] * vik[1]];
                assert!(
                    (acc[0] - rhs[0]).abs() < 1e-9 && (acc[1] - rhs[1]).abs() < 1e-9,
                    "A v != lambda v at band {k}, comp {i}: {acc:?} vs {rhs:?}"
                );
            }
        }
    }

    /// Assert the eigenvector columns are orthonormal: `U^H U = I`.
    fn check_orthonormal(n: usize, evecs: &[Cmplx]) {
        for a in 0..n {
            for b in 0..n {
                let mut acc = [0.0f64; 2];
                for i in 0..n {
                    let p = mul(conj(evecs[i * n + a]), evecs[i * n + b]);
                    acc[0] += p[0];
                    acc[1] += p[1];
                }
                let expect = if a == b { 1.0 } else { 0.0 };
                assert!(
                    (acc[0] - expect).abs() < 1e-9 && acc[1].abs() < 1e-9,
                    "U^H U not identity at ({a},{b}): {acc:?}"
                );
            }
        }
    }

    /// Deterministic `n x n` Hermitian matrix (row-major) for tests.
    fn hermitian_fixture(n: usize) -> Vec<Cmplx> {
        hermitian_fixture_seed(n, 1)
    }

    /// Build `M = B + B^H` from a deterministic `B`, guaranteeing a
    /// Hermitian result without an RNG dependency.
    fn hermitian_fixture_seed(n: usize, seed: usize) -> Vec<Cmplx> {
        let mut b = vec![[0.0f64; 2]; n * n];
        for i in 0..n {
            for j in 0..n {
                let re = ((i * 7 + j * 3 + seed) % 11) as f64 - 5.0;
                let im = ((i * 5 + j * 2 + seed * 3) % 9) as f64 - 4.0;
                b[i * n + j] = [re, im];
            }
        }
        let mut m = vec![[0.0f64; 2]; n * n];
        for i in 0..n {
            for j in 0..n {
                let bij = b[i * n + j];
                let bji = conj(b[j * n + i]);
                m[i * n + j] = [bij[0] + bji[0], bij[1] + bji[1]];
            }
        }
        m
    }
}
