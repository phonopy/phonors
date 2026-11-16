//! Derivative of dynamical matrix with respect to q.
//!
//! Port of `ddm_*` functions from `c/derivative_dynmat.c`.  Computes
//! the analytical derivative of the dynamical matrix at a single
//! q-point, optionally including the Wang-NAC contribution.  Output
//! is a complex `[3, num_band, num_band]` tensor (one slab per
//! cartesian derivative direction); each slab is Hermitian.
//!
//! `derivative_dynmats_at_qpoints` and
//! `derivative_dynmats_at_qpoints_gonze` compute the derivatives at
//! many q-points in parallel over the q-points.

use std::f64::consts::PI;

use rayon::prelude::*;

use crate::common::{matvec_dd, Cmplx, MatD, Vec3D};
use crate::dynmat::{get_derivative_recip_dipole_dipole, make_hermitian};

/// Wang-NAC inputs for the derivative-of-dynmat path.  When the
/// caller wants no NAC, pass `None`.
pub struct DerivativeNacParams<'a> {
    pub born: &'a [[[f64; 3]; 3]],
    pub dielectric: MatD,
    pub q_direction: Option<Vec3D>,
    pub nac_factor: f64,
}

/// Gonze-Lee NAC inputs of `derivative_dynmats_at_qpoints_gonze`.
///
/// `q_direction` is in reduced coordinates and is used only at the
/// Gamma point.  `pos` are the positions of the primitive-cell atoms
/// and `g_list` the reciprocal lattice vectors to sum over, both
/// cartesian.  `nac_factor` and `lambda` are those given to
/// `get_derivative_recip_dipole_dipole`.
pub struct DerivativeGonzeParams<'a> {
    pub born: &'a [[[f64; 3]; 3]],
    pub dielectric: MatD,
    pub q_direction: Option<Vec3D>,
    pub nac_factor: f64,
    pub pos: &'a [Vec3D],
    pub g_list: &'a [Vec3D],
    pub lambda: f64,
}

/// A q-point whose cartesian norm is below this is taken as the Gamma
/// point.
const GAMMA_TOLERANCE: f64 = 1e-5;

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

    // The NAC terms are computed once per call, since they do not depend
    // on the pair of atoms.
    let nac_terms = nac.map(|p| nac_terms_at_q(q, reclat, mass, p, num_patom, num_satom));

    let n_pair = num_patom * num_patom;
    let ddm_ptr = SyncMutPtr(ddm.as_mut_ptr());

    (0..n_pair).into_par_iter().for_each(|ij| {
        let i = ij / num_patom;
        let j = ij % num_patom;
        let (ddm_real, ddm_imag) = derivative_block_of_pair(
            i,
            j,
            fc,
            q,
            lattice,
            svecs,
            multi,
            mass,
            s2p_map,
            fc_row_map,
            nac_terms.as_ref(),
            num_patom,
            num_satom,
        );

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

/// Build the derivatives of the dynamical matrices at many q-points
/// (Wang- or no-NAC path).  Parallel over q-points with rayon.
///
/// `ddms` (output) has length `n_qpoints * 3 * num_band^2` packed in C
/// order as `[q-point, derivative_direction, band, band]`.  It is
/// overwritten, so it need not be zeroed.  The other arguments are
/// those of `derivative_dynmat_at_q`.
///
/// This differs from `derivative_dynmat_at_q` in two points.  The
/// Wang-NAC terms are left out at a q-point at Gamma unless
/// `q_direction` is set in `nac`.  Every element of each slab is
/// Hermitian symmetrized.
#[allow(clippy::too_many_arguments)]
pub fn derivative_dynmats_at_qpoints(
    ddms: &mut [Cmplx],
    qpoints: &[Vec3D],
    fc: &[f64],
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
    let stride = 3 * num_band * num_band;
    assert_eq!(ddms.len(), qpoints.len() * stride);
    if stride == 0 {
        return;
    }
    ddms.par_chunks_mut(stride)
        .zip(qpoints.par_iter())
        .for_each(|(ddm, &q)| {
            let nac_at_q =
                nac.filter(|p| p.q_direction.is_some() || !is_at_gamma(matvec_dd(reclat, q)));
            fill_derivative_dynmat_at_q(
                ddm, fc, q, lattice, reclat, svecs, multi, mass, s2p_map, fc_row_map, nac_at_q,
                num_patom, num_satom,
            );
            for slab in ddm.chunks_mut(num_band * num_band) {
                make_hermitian(slab, num_band);
            }
        });
}

/// Build the derivatives of the dynamical matrices at many q-points
/// (Gonze-Lee NAC path).  Parallel over q-points with rayon.
///
/// `fc` are the short-range force constants.  At each q-point, the
/// derivative of the reciprocal dipole-dipole sum divided by
/// `sqrt(m_i m_j)` is added to the derivative of the short-range part,
/// and every element of each slab is Hermitian symmetrized.  The layout
/// of `ddms` and the other arguments are those of
/// `derivative_dynmats_at_qpoints`.
#[allow(clippy::too_many_arguments)]
pub fn derivative_dynmats_at_qpoints_gonze(
    ddms: &mut [Cmplx],
    qpoints: &[Vec3D],
    fc: &[f64],
    lattice: &MatD,
    reclat: &MatD,
    svecs: &[Vec3D],
    multi: &[[i64; 2]],
    mass: &[f64],
    s2p_map: &[i64],
    fc_row_map: &[i64],
    gonze: &DerivativeGonzeParams,
    num_patom: usize,
    num_satom: usize,
) {
    let num_band = num_patom * 3;
    let stride = 3 * num_band * num_band;
    assert_eq!(ddms.len(), qpoints.len() * stride);
    if stride == 0 {
        return;
    }
    // The direction is used in place of q + G at Gamma for G = 0, and
    // it is normalized as in phonopy.
    let q_direction_cart = gonze.q_direction.map(|direction| {
        let d = matvec_dd(reclat, direction);
        let norm = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        [d[0] / norm, d[1] / norm, d[2] / norm]
    });

    ddms.par_chunks_mut(stride)
        .zip(qpoints.par_iter())
        .for_each_init(
            || vec![[0.0f64; 2]; stride],
            |dipole_dipole, (ddm, &q)| {
                fill_derivative_dynmat_at_q(
                    ddm, fc, q, lattice, reclat, svecs, multi, mass, s2p_map, fc_row_map, None,
                    num_patom, num_satom,
                );

                let q_cart = matvec_dd(reclat, q);
                let direction = if is_at_gamma(q_cart) {
                    q_direction_cart
                } else {
                    None
                };
                get_derivative_recip_dipole_dipole(
                    dipole_dipole,
                    gonze.g_list,
                    num_patom,
                    q_cart,
                    direction,
                    gonze.born,
                    &gonze.dielectric,
                    gonze.pos,
                    gonze.nac_factor,
                    gonze.lambda,
                );

                // `dipole_dipole` is `[direction, i, alpha, j, beta]`,
                // which has the same flat addresses as a slab of `ddm`.
                for (adrs, (elem, dd)) in ddm.iter_mut().zip(dipole_dipole.iter()).enumerate() {
                    let i = (adrs / num_band) % num_band / 3;
                    let j = (adrs % num_band) / 3;
                    let mass_sqrt = (mass[i] * mass[j]).sqrt();
                    elem[0] += dd[0] / mass_sqrt;
                    elem[1] += dd[1] / mass_sqrt;
                }

                for slab in ddm.chunks_mut(num_band * num_band) {
                    make_hermitian(slab, num_band);
                }
            },
        );
}

fn is_at_gamma(q_cart: Vec3D) -> bool {
    let norm2 = q_cart[0] * q_cart[0] + q_cart[1] * q_cart[1] + q_cart[2] * q_cart[2];
    norm2 < GAMMA_TOLERANCE * GAMMA_TOLERANCE
}

/// Write the derivative of the dynamical matrix at one q-point into
/// `ddm` (`[3, num_band, num_band]`), without the Hermitian
/// symmetrization.  The pairs of atoms are computed serially.
#[allow(clippy::too_many_arguments)]
fn fill_derivative_dynmat_at_q(
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
    let nac_terms = nac.map(|p| nac_terms_at_q(q, reclat, mass, p, num_patom, num_satom));
    for i in 0..num_patom {
        for j in 0..num_patom {
            let (ddm_real, ddm_imag) = derivative_block_of_pair(
                i,
                j,
                fc,
                q,
                lattice,
                svecs,
                multi,
                mass,
                s2p_map,
                fc_row_map,
                nac_terms.as_ref(),
                num_patom,
                num_satom,
            );
            for direction in 0..3 {
                for l in 0..3 {
                    for m in 0..3 {
                        let adrs =
                            direction * num_band * num_band + (i * 3 + l) * num_band + j * 3 + m;
                        ddm[adrs] = [ddm_real[direction][l][m], ddm_imag[direction][l][m]];
                    }
                }
            }
        }
    }
}

/// Wang-NAC terms at one q-point.
///
/// `dnac` has shape `[num_patom, num_patom, 3, 3]` and is added to the
/// force constants.  `ddnac` has shape `[3, num_patom, num_patom, 3, 3]`
/// and is its derivative with respect to the cartesian components of q.
struct NacTerms {
    dnac: Vec<f64>,
    ddnac: Vec<f64>,
}

fn nac_terms_at_q(
    q: Vec3D,
    reclat: &MatD,
    mass: &[f64],
    params: &DerivativeNacParams,
    num_patom: usize,
    num_satom: usize,
) -> NacTerms {
    let nac_pref = (params.nac_factor * num_patom as f64) / num_satom as f64;
    let mut ddnac = vec![0.0; num_patom * num_patom * 27];
    let mut dnac = vec![0.0; num_patom * num_patom * 9];
    get_derivative_nac(
        &mut ddnac,
        &mut dnac,
        num_patom,
        reclat,
        mass,
        q,
        params.born,
        &params.dielectric,
        params.q_direction,
        nac_pref,
    );
    NacTerms { dnac, ddnac }
}

/// Derivative of the 3x3 block of the dynamical matrix of the pair of
/// primitive-cell atoms `(i, j)` at one q-point, before the Hermitian
/// symmetrization.
///
/// Returns the real and imaginary parts, each indexed by
/// `[derivative_direction][alpha][beta]`.
#[allow(clippy::too_many_arguments)]
fn derivative_block_of_pair(
    i: usize,
    j: usize,
    fc: &[f64],
    q: Vec3D,
    lattice: &MatD,
    svecs: &[Vec3D],
    multi: &[[i64; 2]],
    mass: &[f64],
    s2p_map: &[i64],
    fc_row_map: &[i64],
    nac_terms: Option<&NacTerms>,
    num_patom: usize,
    num_satom: usize,
) -> ([[[f64; 3]; 3]; 3], [[[f64; 3]; 3]; 3]) {
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
                if let Some(terms) = nac_terms {
                    fc_elem += terms.dnac[i * 9 * num_patom + j * 9 + l * 3 + m];
                }
                for n in 0..3 {
                    ddm_real[n][l][m] += fc_elem * real_coef[n];
                    ddm_imag[n][l][m] += fc_elem * imag_coef[n];
                    if let Some(terms) = nac_terms {
                        let dd = terms.ddnac
                            [n * num_patom * num_patom * 9 + i * 9 * num_patom + j * 9 + l * 3 + m];
                        ddm_real[n][l][m] += dd * real_phase;
                        ddm_imag[n][l][m] += dd * imag_phase;
                    }
                }
            }
        }
    }

    (ddm_real, ddm_imag)
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

    /// Deterministic pseudo-random numbers in [-0.5, 0.5).
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        }
    }

    /// Two primitive-cell atoms in a supercell of four atoms with full
    /// force constants.  The values are arbitrary: the force constants
    /// are not symmetric, so the Hermitian symmetrization changes the
    /// result, and some pairs have two shortest vectors.
    struct Fixture {
        fc: Vec<f64>,
        svecs: Vec<Vec3D>,
        multi: Vec<[i64; 2]>,
        mass: Vec<f64>,
        s2p_map: Vec<i64>,
        p2s_map: Vec<i64>,
        lattice: MatD,
        reclat: MatD,
        born: Vec<[[f64; 3]; 3]>,
        dielectric: MatD,
        pos: Vec<Vec3D>,
        g_list: Vec<Vec3D>,
    }

    const NUM_PATOM: usize = 2;
    const NUM_SATOM: usize = 4;
    const NUM_BAND: usize = NUM_PATOM * 3;
    const STRIDE: usize = 3 * NUM_BAND * NUM_BAND;
    const NAC_FACTOR: f64 = 0.7;
    const LAMBDA: f64 = 0.9;

    fn fixture() -> Fixture {
        let mut rng = Lcg(20261116);
        let fc: Vec<f64> = (0..NUM_SATOM * NUM_SATOM * 9).map(|_| rng.next()).collect();
        let mut svecs = Vec::new();
        let mut multi = Vec::new();
        for pair in 0..NUM_SATOM * NUM_PATOM {
            let m_pair = 1 + pair % 2;
            multi.push([m_pair as i64, svecs.len() as i64]);
            for _ in 0..m_pair {
                svecs.push([rng.next() * 2.0, rng.next() * 2.0, rng.next() * 2.0]);
            }
        }
        // Rows of the primitive cell are (2, 0, 0), (1, 4, 0), (0, 0, 5).
        // lattice is its transpose and reclat its inverse.
        let lattice = [[2.0, 1.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]];
        let reclat = [[0.5, 0.0, 0.0], [-0.125, 0.25, 0.0], [0.0, 0.0, 0.2]];
        let mut g_list = Vec::new();
        for a in -1..=1 {
            for b in -1..=1 {
                for c in -1..=1 {
                    g_list.push(matvec_dd(&reclat, [a as f64, b as f64, c as f64]));
                }
            }
        }
        Fixture {
            fc,
            svecs,
            multi,
            mass: vec![1.5, 3.0],
            s2p_map: vec![0, 0, 2, 2],
            p2s_map: vec![0, 2],
            lattice,
            reclat,
            born: vec![
                [[1.0, 0.2, 0.0], [0.1, 1.5, -0.1], [0.0, -0.3, 0.8]],
                [[-1.1, -0.2, 0.3], [-0.2, -1.4, 0.1], [0.05, 0.1, -0.8]],
            ],
            dielectric: [[2.5, 0.2, -0.1], [0.2, 3.0, 0.05], [-0.1, 0.05, 2.0]],
            pos: vec![[0.1, 0.2, 0.3], [1.3, 2.1, 2.6]],
            g_list,
        }
    }

    fn nac_params(fx: &Fixture, q_direction: Option<Vec3D>) -> DerivativeNacParams<'_> {
        DerivativeNacParams {
            born: &fx.born,
            dielectric: fx.dielectric,
            q_direction,
            nac_factor: NAC_FACTOR,
        }
    }

    fn gonze_params(fx: &Fixture, q_direction: Option<Vec3D>) -> DerivativeGonzeParams<'_> {
        DerivativeGonzeParams {
            born: &fx.born,
            dielectric: fx.dielectric,
            q_direction,
            nac_factor: NAC_FACTOR,
            pos: &fx.pos,
            g_list: &fx.g_list,
            lambda: LAMBDA,
        }
    }

    /// The output starts from a non-zero value to check that it is
    /// overwritten.
    fn batch(fx: &Fixture, qpoints: &[Vec3D], nac: Option<&DerivativeNacParams>) -> Vec<Cmplx> {
        let mut ddms = vec![[7.0, -7.0]; qpoints.len() * STRIDE];
        derivative_dynmats_at_qpoints(
            &mut ddms,
            qpoints,
            &fx.fc,
            &fx.lattice,
            &fx.reclat,
            &fx.svecs,
            &fx.multi,
            &fx.mass,
            &fx.s2p_map,
            &fx.p2s_map,
            nac,
            NUM_PATOM,
            NUM_SATOM,
        );
        ddms
    }

    fn batch_gonze(fx: &Fixture, qpoints: &[Vec3D], gonze: &DerivativeGonzeParams) -> Vec<Cmplx> {
        let mut ddms = vec![[7.0, -7.0]; qpoints.len() * STRIDE];
        derivative_dynmats_at_qpoints_gonze(
            &mut ddms,
            qpoints,
            &fx.fc,
            &fx.lattice,
            &fx.reclat,
            &fx.svecs,
            &fx.multi,
            &fx.mass,
            &fx.s2p_map,
            &fx.p2s_map,
            gonze,
            NUM_PATOM,
            NUM_SATOM,
        );
        ddms
    }

    /// Single q-point kernel followed by the Hermitian symmetrization of
    /// every element.
    fn single(fx: &Fixture, q: Vec3D, nac: Option<&DerivativeNacParams>) -> Vec<Cmplx> {
        let mut ddm = vec![[0.0, 0.0]; STRIDE];
        derivative_dynmat_at_q(
            &mut ddm,
            &fx.fc,
            q,
            &fx.lattice,
            &fx.reclat,
            &fx.svecs,
            &fx.multi,
            &fx.mass,
            &fx.s2p_map,
            &fx.p2s_map,
            nac,
            NUM_PATOM,
            NUM_SATOM,
        );
        for slab in ddm.chunks_mut(NUM_BAND * NUM_BAND) {
            make_hermitian(slab, NUM_BAND);
        }
        ddm
    }

    /// Central finite difference along the cartesian directions of the
    /// dynamical matrices given by `dynmats_at`.
    fn finite_difference(
        fx: &Fixture,
        q: Vec3D,
        dynmats_at: impl Fn(&[Vec3D]) -> Vec<Cmplx>,
    ) -> Vec<Cmplx> {
        let step = 1e-6;
        let mut qpoints = Vec::new();
        for direction in 0..3 {
            for sign in [1.0, -1.0] {
                // A cartesian shift e_m * step is lattice[m] * step in
                // reduced coordinates.
                let mut shifted = q;
                for n in 0..3 {
                    shifted[n] += sign * step * fx.lattice[direction][n];
                }
                qpoints.push(shifted);
            }
        }
        let dynmats = dynmats_at(&qpoints);
        let size = NUM_BAND * NUM_BAND;
        let mut ddm = vec![[0.0, 0.0]; STRIDE];
        for direction in 0..3 {
            let plus = &dynmats[2 * direction * size..(2 * direction + 1) * size];
            let minus = &dynmats[(2 * direction + 1) * size..(2 * direction + 2) * size];
            for adrs in 0..size {
                for part in 0..2 {
                    ddm[direction * size + adrs][part] =
                        (plus[adrs][part] - minus[adrs][part]) / (2.0 * step);
                }
            }
        }
        ddm
    }

    fn assert_close(actual: &[Cmplx], expected: &[Cmplx], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());
        let mut largest = 0.0f64;
        for (adrs, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            for part in 0..2 {
                assert!(
                    (a[part] - e[part]).abs() < tolerance,
                    "adrs={adrs} part={part} actual={} expected={}",
                    a[part],
                    e[part],
                );
            }
            largest = largest.max(e[0].abs()).max(e[1].abs());
        }
        // Guard against a comparison of zeros with zeros.
        assert!(largest > 1e-2, "expected values are all nearly zero");
    }

    const QPOINTS: [Vec3D; 3] = [[0.1, 0.2, 0.3], [-0.35, 0.05, 0.4], [0.5, 0.0, -0.15]];

    #[test]
    fn batch_matches_single_q_kernel() {
        let fx = fixture();
        let nacs = [
            None,
            Some(nac_params(&fx, None)),
            Some(nac_params(&fx, Some([0.3, -0.1, 0.2]))),
        ];
        for nac in &nacs {
            let ddms = batch(&fx, &QPOINTS, nac.as_ref());
            for (ddm, &q) in ddms.chunks(STRIDE).zip(QPOINTS.iter()) {
                assert_close(ddm, &single(&fx, q, nac.as_ref()), 1e-12);
            }
        }
    }

    #[test]
    fn batch_symmetrizes_elements_left_by_single_q_kernel() {
        // The single q-point kernel leaves the block of the bands 0 and
        // 1 of the third direction as computed.  With the force
        // constants of the fixture, that block is not Hermitian, and
        // the batch kernel makes it Hermitian.
        let fx = fixture();
        let q = QPOINTS[0];
        let mut unsymmetrized = vec![[0.0, 0.0]; STRIDE];
        derivative_dynmat_at_q(
            &mut unsymmetrized,
            &fx.fc,
            q,
            &fx.lattice,
            &fx.reclat,
            &fx.svecs,
            &fx.multi,
            &fx.mass,
            &fx.s2p_map,
            &fx.p2s_map,
            None,
            NUM_PATOM,
            NUM_SATOM,
        );
        let slab = 2 * NUM_BAND * NUM_BAND;
        let upper = unsymmetrized[slab + 1];
        let lower = unsymmetrized[slab + NUM_BAND];
        assert!((upper[0] - lower[0]).abs() > 1e-3);

        let ddms = batch(&fx, &[q], None);
        let upper = ddms[slab + 1];
        let lower = ddms[slab + NUM_BAND];
        assert!((upper[0] - lower[0]).abs() < 1e-14);
        assert!((upper[1] + lower[1]).abs() < 1e-14);
    }

    #[test]
    fn batch_matches_finite_difference_of_dynmats() {
        use crate::dynmat::{dynamical_matrices_at_qpoints, WangNacParams};

        let fx = fixture();
        for with_nac in [false, true] {
            let wang = WangNacParams {
                born: &fx.born,
                dielectric: fx.dielectric,
                reciprocal_lattice: fx.reclat,
                q_direction: None,
                nac_factor: NAC_FACTOR,
            };
            let nac = nac_params(&fx, None);
            let ddms = batch(&fx, &QPOINTS, with_nac.then_some(&nac));
            for (ddm, &q) in ddms.chunks(STRIDE).zip(QPOINTS.iter()) {
                let expected = finite_difference(&fx, q, |qpoints| {
                    let mut dynmats = vec![[0.0, 0.0]; qpoints.len() * NUM_BAND * NUM_BAND];
                    dynamical_matrices_at_qpoints(
                        &mut dynmats,
                        qpoints,
                        &fx.fc,
                        &fx.svecs,
                        &fx.multi,
                        &fx.mass,
                        &fx.p2s_map,
                        &fx.s2p_map,
                        NUM_PATOM,
                        NUM_SATOM,
                        with_nac.then_some(&wang),
                        true,
                    );
                    dynmats
                });
                assert_close(ddm, &expected, 1e-6);
            }
        }
    }

    #[test]
    fn batch_leaves_out_nac_at_gamma_without_q_direction() {
        let fx = fixture();
        let qpoints = [[0.0, 0.0, 0.0], QPOINTS[0]];
        let nac = nac_params(&fx, None);
        let with_nac = batch(&fx, &qpoints, Some(&nac));
        let without_nac = batch(&fx, &qpoints, None);
        // At Gamma the NAC terms are left out.
        assert_close(&with_nac[..STRIDE], &without_nac[..STRIDE], 1e-14);
        // At the other q-point they are not.
        let difference = with_nac[STRIDE..]
            .iter()
            .zip(without_nac[STRIDE..].iter())
            .map(|(a, b)| (a[0] - b[0]).abs().max((a[1] - b[1]).abs()))
            .fold(0.0, f64::max);
        assert!(difference > 1e-3);
    }

    #[test]
    fn gonze_batch_matches_finite_difference_of_dynmats() {
        use crate::dynmat::{dynamical_matrices_at_qpoints_gonze, GonzeNacParams};

        let fx = fixture();
        // The correction at q = 0 does not depend on q, so it does not
        // change the difference.
        let dd_q0 = vec![[0.0, 0.0]; NUM_PATOM * 9];
        let gonze_dynmat = GonzeNacParams {
            born: &fx.born,
            dielectric: fx.dielectric,
            reciprocal_lattice: fx.reclat,
            q_direction: None,
            nac_factor: NAC_FACTOR,
            pos: &fx.pos,
            dd_q0: &dd_q0,
            g_list: &fx.g_list,
            lambda: LAMBDA,
        };
        let ddms = batch_gonze(&fx, &QPOINTS, &gonze_params(&fx, None));
        // The dipole-dipole part must be in the result.
        let short_range = batch(&fx, &QPOINTS, None);
        let difference = ddms
            .iter()
            .zip(short_range.iter())
            .map(|(a, b)| (a[0] - b[0]).abs().max((a[1] - b[1]).abs()))
            .fold(0.0, f64::max);
        assert!(difference > 1e-3);

        for (ddm, &q) in ddms.chunks(STRIDE).zip(QPOINTS.iter()) {
            let expected = finite_difference(&fx, q, |qpoints| {
                let mut dynmats = vec![[0.0, 0.0]; qpoints.len() * NUM_BAND * NUM_BAND];
                dynamical_matrices_at_qpoints_gonze(
                    &mut dynmats,
                    qpoints,
                    &fx.fc,
                    &fx.svecs,
                    &fx.multi,
                    &fx.mass,
                    &fx.p2s_map,
                    &fx.s2p_map,
                    NUM_PATOM,
                    NUM_SATOM,
                    &gonze_dynmat,
                    true,
                );
                dynmats
            });
            assert_close(ddm, &expected, 1e-6);
        }
    }

    #[test]
    fn gonze_batch_matches_single_q_kernels() {
        // The q-direction is used at Gamma only.
        let fx = fixture();
        let q_direction = [0.3, -0.1, 0.2];
        let qpoints = [[0.0, 0.0, 0.0], QPOINTS[1]];
        let ddms = batch_gonze(&fx, &qpoints, &gonze_params(&fx, Some(q_direction)));

        let d = matvec_dd(&fx.reclat, q_direction);
        let norm = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let direction_cart = [d[0] / norm, d[1] / norm, d[2] / norm];
        for (i_q, (ddm, &q)) in ddms.chunks(STRIDE).zip(qpoints.iter()).enumerate() {
            let mut expected = single(&fx, q, None);
            let mut dipole_dipole = vec![[0.0, 0.0]; STRIDE];
            get_derivative_recip_dipole_dipole(
                &mut dipole_dipole,
                &fx.g_list,
                NUM_PATOM,
                matvec_dd(&fx.reclat, q),
                (i_q == 0).then_some(direction_cart),
                &fx.born,
                &fx.dielectric,
                &fx.pos,
                NAC_FACTOR,
                LAMBDA,
            );
            for mu in 0..3 {
                for i in 0..NUM_PATOM {
                    for alpha in 0..3 {
                        for j in 0..NUM_PATOM {
                            for beta in 0..3 {
                                let adrs = mu * NUM_BAND * NUM_BAND
                                    + (i * 3 + alpha) * NUM_BAND
                                    + j * 3
                                    + beta;
                                let mass_sqrt = (fx.mass[i] * fx.mass[j]).sqrt();
                                expected[adrs][0] += dipole_dipole[adrs][0] / mass_sqrt;
                                expected[adrs][1] += dipole_dipole[adrs][1] / mass_sqrt;
                            }
                        }
                    }
                }
            }
            for slab in expected.chunks_mut(NUM_BAND * NUM_BAND) {
                make_hermitian(slab, NUM_BAND);
            }
            assert_close(ddm, &expected, 1e-12);
        }
    }
}
