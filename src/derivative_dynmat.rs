//! Derivative of dynamical matrix with respect to q.
//!
//! Port of `ddm_*` functions from `c/derivative_dynmat.c`.  Computes
//! the analytical derivative of the dynamical matrix at a single
//! q-point, optionally including the Wang-NAC contribution.  Output
//! is a complex `[3, num_band, num_band]` tensor (one slab per
//! cartesian derivative direction); each slab is Hermitian.

use std::f64::consts::PI;

use rayon::prelude::*;

use crate::common::{Cmplx, MatD, Vec3D};

/// Wang-NAC inputs for the derivative-of-dynmat path.  When the
/// caller wants no NAC, pass `None`.
pub struct DerivativeNacParams<'a> {
    pub born: &'a [[[f64; 3]; 3]],
    pub dielectric: MatD,
    pub q_direction: Option<Vec3D>,
    pub nac_factor: f64,
}

/// `*mut T` wrapper opting into Send + Sync for rayon.  Used in the
/// `(i, j)` parallel loop of `derivative_dynmat_at_q` where each task
/// touches a disjoint 9-element output block per derivative direction.
#[derive(Clone, Copy)]
struct SyncMutPtr<T>(*mut T);
unsafe impl<T> Send for SyncMutPtr<T> {}
unsafe impl<T> Sync for SyncMutPtr<T> {}

impl<T> SyncMutPtr<T> {
    fn ptr(self) -> *mut T {
        self.0
    }
}

/// Build the derivative of the dynamical matrix at a single q-point.
/// Mirrors `ddm_get_derivative_dynmat_at_q` in `c/derivative_dynmat.c`.
///
/// `ddm` (output, in/out) has length `3 * num_band^2` packed in C
/// order with `num_band = num_patom * 3`; the leading axis is the
/// cartesian derivative direction.  `fc` first axis may be either
/// `num_satom` (full fc) or `num_patom` (compact fc); `fc_row_map[i]`
/// is the row index used into `fc`, so the caller must pass
/// `primitive.p2s_map` for full fc and `arange(num_patom)` for compact
/// fc.  `s2p_map[k]` and `fc_row_map[j]` are compared to gate the
/// inner loop, so they must come from the same atom-numbering scheme.
///
/// `lattice` is the supercell-aware basis used to build cartesian
/// position-derivative coefficients (matches the Python helper which
/// passes `pcell.cell.T`).  `reclat` is the reciprocal lattice in
/// column vectors; only used by the NAC path.
///
/// `nac` is `Some(...)` for Wang NAC, `None` for no NAC; the kernel
/// internally falls back to no-NAC at the Gamma point unless a
/// `q_direction` is set inside `nac`.
#[allow(clippy::too_many_arguments)]
pub fn derivative_dynmat_at_q(
    ddm: &mut [Cmplx],
    fc: &[f64],
    q: Vec3D,
    lattice: &MatD,
    reclat: &MatD,
    svecs: &[Vec3D],
    multi: &[[i64; 2]],
    mass: &[f64],
    s2p_map: &[i64],
    fc_row_map: &[i64],
    nac: Option<&DerivativeNacParams>,
    num_patom: usize,
    num_satom: usize,
) {
    let num_band = num_patom * 3;
    debug_assert_eq!(ddm.len(), 3 * num_band * num_band);
    debug_assert_eq!(mass.len(), num_patom);
    debug_assert_eq!(fc_row_map.len(), num_patom);
    debug_assert_eq!(s2p_map.len(), num_satom);
    debug_assert_eq!(multi.len(), num_satom * num_patom);

    // Pre-compute the NAC dipole derivatives once per call (independent
    // of the supercell-atom loop inside the per-pair kernel).
    let mut ddnac: Vec<f64> = Vec::new();
    let mut dnac: Vec<f64> = Vec::new();
    if let Some(p) = nac {
        let nac_pref = (p.nac_factor * num_patom as f64) / num_satom as f64;
        ddnac = vec![0.0; num_patom * num_patom * 27];
        dnac = vec![0.0; num_patom * num_patom * 9];
        get_derivative_nac(
            &mut ddnac,
            &mut dnac,
            num_patom,
            reclat,
            mass,
            q,
            p.born,
            &p.dielectric,
            p.q_direction,
            nac_pref,
        );
    }

    let n_pair = num_patom * num_patom;
    let ddm_ptr = SyncMutPtr(ddm.as_mut_ptr());

    (0..n_pair).into_par_iter().for_each(|ij| {
        let i = ij / num_patom;
        let j = ij % num_patom;
        // Local accumulators for this (i, j) pair, indexed
        // [derivative_direction][alpha][beta].
        let mut ddm_real = [[[0.0f64; 3]; 3]; 3];
        let mut ddm_imag = [[[0.0f64; 3]; 3]; 3];

        let mass_sqrt = (mass[i] * mass[j]).sqrt();
        for k_s in 0..num_satom {
            if s2p_map[k_s] != fc_row_map[j] {
                continue;
            }
            let i_pair = k_s * num_patom + i;
            let m_pair = multi[i_pair][0] as usize;
            let svecs_adrs = multi[i_pair][1] as usize;
            let inv_m_pair = 1.0 / m_pair as f64;

            let mut real_phase = 0.0;
            let mut imag_phase = 0.0;
            let mut real_coef = [0.0f64; 3];
            let mut imag_coef = [0.0f64; 3];
            for l in 0..m_pair {
                let svec = svecs[svecs_adrs + l];
                let phase = q[0] * svec[0] + q[1] * svec[1] + q[2] * svec[2];
                let arg = phase * 2.0 * PI;
                let s = arg.sin();
                let c = arg.cos();
                real_phase += c;
                imag_phase += s;
                let mut coef = [0.0f64; 3];
                for m in 0..3 {
                    let mut acc = 0.0;
                    for n in 0..3 {
                        acc += 2.0 * PI * lattice[m][n] * svec[n];
                    }
                    coef[m] = acc;
                }
                for m in 0..3 {
                    real_coef[m] -= coef[m] * s;
                    imag_coef[m] += coef[m] * c;
                }
            }
            real_phase *= inv_m_pair;
            imag_phase *= inv_m_pair;
            for m in 0..3 {
                real_coef[m] *= inv_m_pair;
                imag_coef[m] *= inv_m_pair;
            }

            let fc_row = fc_row_map[i] as usize;
            for l in 0..3 {
                for m in 0..3 {
                    let mut fc_elem = fc[fc_row * num_satom * 9 + k_s * 9 + l * 3 + m] / mass_sqrt;
                    if nac.is_some() {
                        fc_elem += dnac[i * 9 * num_patom + j * 9 + l * 3 + m];
                    }
                    for n in 0..3 {
                        ddm_real[n][l][m] += fc_elem * real_coef[n];
                        ddm_imag[n][l][m] += fc_elem * imag_coef[n];
                        if nac.is_some() {
                            let dd = ddnac[n * num_patom * num_patom * 9
                                + i * 9 * num_patom
                                + j * 9
                                + l * 3
                                + m];
                            ddm_real[n][l][m] += dd * real_phase;
                            ddm_imag[n][l][m] += dd * imag_phase;
                        }
                    }
                }
            }
        }

        // Disjoint write into the 3 slabs at (i, j).  The (k, l, m)
        // address is unique per (i, j) tuple so different tasks never
        // touch the same offset.
        unsafe {
            let p = ddm_ptr.ptr();
            for k_d in 0..3 {
                for l in 0..3 {
                    for m in 0..3 {
                        let adrs = k_d * num_patom * num_patom * 9
                            + (i * 3 + l) * num_patom * 3
                            + j * 3
                            + m;
                        (*p.add(adrs))[0] += ddm_real[k_d][l][m];
                        (*p.add(adrs))[1] += ddm_imag[k_d][l][m];
                    }
                }
            }
        }
    });

    // Hermitian symmetrization of the 3 slabs.  Mirrors the C loop
    // bounds verbatim (`j` starts at `i`, where `i` is the derivative
    // direction and `j` is a band index -- preserved as-is for parity).
    for d in 0..3 {
        for j in d..num_band {
            for k in 0..num_band {
                let adrs = d * num_band * num_band + j * num_band + k;
                let adrs_t = d * num_band * num_band + k * num_band + j;
                let re = (ddm[adrs][0] + ddm[adrs_t][0]) / 2.0;
                let im = (ddm[adrs][1] - ddm[adrs_t][1]) / 2.0;
                ddm[adrs][0] = re;
                ddm[adrs][1] = im;
                ddm[adrs_t][0] = re;
                ddm[adrs_t][1] = -im;
            }
        }
    }
}

/// Build the NAC dipole-derivative tensors `ddnac` and `dnac`.
/// Mirrors the file-static `get_derivative_nac` in
/// `c/derivative_dynmat.c`.
///
/// `D_nac = factor * A B / C`, `dD_nac = factor * D_nac * (A'/A + B'/B - C'/C)`.
#[allow(clippy::too_many_arguments)]
fn get_derivative_nac(
    ddnac: &mut [f64],
    dnac: &mut [f64],
    num_patom: usize,
    reclat: &MatD,
    mass: &[f64],
    q: Vec3D,
    born: &[[[f64; 3]; 3]],
    dielectric: &MatD,
    q_direction: Option<Vec3D>,
    factor: f64,
) {
    let q_for_cart = q_direction.unwrap_or(q);
    let mut q_cart = [0.0f64; 3];
    for i in 0..3 {
        for jj in 0..3 {
            q_cart[i] += reclat[i][jj] * q_for_cart[jj];
        }
    }
    let c = get_c(q_cart, dielectric);

    for i in 0..num_patom {
        for j in 0..num_patom {
            let mass_sqrt = (mass[i] * mass[j]).sqrt();
            for k in 0..3 {
                for l in 0..3 {
                    let a = get_a(i, l, q_cart, born);
                    let da = get_da(i, l, k, born);
                    for m in 0..3 {
                        let b = get_a(j, m, q_cart, born);
                        let db = get_da(j, m, k, born);
                        let dc = get_dc(l, m, k, q_cart, dielectric);
                        ddnac[k * num_patom * num_patom * 9
                            + i * 9 * num_patom
                            + j * 9
                            + l * 3
                            + m] = (da * b + db * a - a * b * dc / c) / (c * mass_sqrt) * factor;
                        if k == 0 {
                            dnac[i * 9 * num_patom + j * 9 + l * 3 + m] =
                                a * b / (c * mass_sqrt) * factor;
                        }
                    }
                }
            }
        }
    }
}

#[inline]
fn get_a(atom_i: usize, cart_i: usize, q: Vec3D, born: &[[[f64; 3]; 3]]) -> f64 {
    q[0] * born[atom_i][0][cart_i] + q[1] * born[atom_i][1][cart_i] + q[2] * born[atom_i][2][cart_i]
}

#[inline]
fn get_c(q: Vec3D, dielectric: &MatD) -> f64 {
    let mut s = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            s += q[i] * dielectric[i][j] * q[j];
        }
    }
    s
}

#[inline]
fn get_da(atom_i: usize, cart_i: usize, cart_j: usize, born: &[[[f64; 3]; 3]]) -> f64 {
    born[atom_i][cart_j][cart_i]
}

/// Cartesian derivative of `q^T eps q` with respect to the
/// `cart_k`-th component of `q`.  Mirrors the C `get_dC` switch.
#[inline]
fn get_dc(cart_i: usize, cart_j: usize, cart_k: usize, q: Vec3D, dielectric: &MatD) -> f64 {
    // `cart_i` and `cart_j` are present in the C signature for
    // notational symmetry with `get_dA`, but `dC/dq_k` is independent
    // of `(cart_i, cart_j)` because `C = q^T eps q` -- the kernel only
    // uses the value at fixed `cart_k`.
    let _ = (cart_i, cart_j);
    match cart_k {
        0 => {
            2.0 * q[0] * dielectric[0][0]
                + q[1] * (dielectric[0][1] + dielectric[1][0])
                + q[2] * (dielectric[0][2] + dielectric[2][0])
        }
        1 => {
            2.0 * q[1] * dielectric[1][1]
                + q[2] * (dielectric[1][2] + dielectric[2][1])
                + q[0] * (dielectric[0][1] + dielectric[1][0])
        }
        2 => {
            2.0 * q[2] * dielectric[2][2]
                + q[0] * (dielectric[0][2] + dielectric[2][0])
                + q[1] * (dielectric[1][2] + dielectric[2][1])
        }
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_gives_partial_derivative_of_qeq() {
        // C = q^T eps q.  Numerically check dC/dq_k against finite
        // differences on a randomly-but-fixedly-chosen dielectric
        // tensor and q.
        let dielectric: MatD = [[1.5, 0.2, -0.1], [0.2, 2.0, 0.05], [-0.1, 0.05, 1.0]];
        let q = [0.3, -0.4, 0.7];
        let h = 1e-6;
        for k in 0..3 {
            let mut q_plus = q;
            q_plus[k] += h;
            let mut q_minus = q;
            q_minus[k] -= h;
            let fd = (get_c(q_plus, &dielectric) - get_c(q_minus, &dielectric)) / (2.0 * h);
            let analytic = get_dc(0, 0, k, q, &dielectric);
            assert!(
                (fd - analytic).abs() < 1e-8,
                "k={k} fd={fd} analytic={analytic}",
            );
        }
    }

    #[test]
    fn da_recovers_a_along_axis() {
        // A(q) = sum_k q_k Z[i, k, alpha].  d/dq_k A = Z[i, k, alpha].
        // Check via the helper `get_a` and `get_da`.
        let born = vec![[[1.0, 0.5, -0.2], [0.0, 2.0, 0.1], [-0.3, 0.1, 1.5]]];
        let q = [0.7, -0.4, 0.2];
        for cart_i in 0..3 {
            // dA/dq_k for atom 0, alpha = cart_i.
            let h = 1e-6;
            for cart_k in 0..3 {
                let mut q_plus = q;
                q_plus[cart_k] += h;
                let mut q_minus = q;
                q_minus[cart_k] -= h;
                let fd = (get_a(0, cart_i, q_plus, &born) - get_a(0, cart_i, q_minus, &born))
                    / (2.0 * h);
                let analytic = get_da(0, cart_i, cart_k, &born);
                assert!(
                    (fd - analytic).abs() < 1e-8,
                    "cart_i={cart_i} cart_k={cart_k} fd={fd} analytic={analytic}",
                );
            }
        }
    }
}
