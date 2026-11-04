//! Binning of the Fourier coefficients of a supercell by q and K mod N^p.
//!
//! Kernels of the native evaluation of a Fourier series of the supercell on
//! the grid of the primitive cell in phelel (`FourierSeriesOnGrid.get_phi`).
//! The series has the coefficients c(g) at the g-points: the points g of
//! the reciprocal lattice of the supercell, given by integer coordinates in
//! the basis of the reciprocal lattice of the supercell (column vectors).
//! c(g) is periodic with the supercell FFT mesh N^s, c(g + N^s t) = c(g),
//! and is stored at the index
//!
//! ```text
//! s(g) = ((g_3 mod N^s_3) N^s_2 + (g_2 mod N^s_2)) N^s_1 + (g_1 mod N^s_1)
//! ```
//!
//! of the flattened coefficient array, called the source of g.  The series
//! is summed over the selected g-points g~: for each source, one or more
//! g-points with this source, selected by a rule such as the shortest ones.
//! The selected g-points with the same source share its coefficient with the
//! weight 1 / (their number).
//!
//! For a space-group operation acting as x^s -> W^s x^s + w^s in the
//! fractional coordinates of the supercell, the rotated function is
//!
//! ```text
//! sum_g~ c(s(g~)) weight(s(g~)) exp(-2 pi i g~' . w^s) exp(2 pi i g~' . x^s),
//! g~' = (W^s)^-T g~.
//! ```
//!
//! g~' is written in the reciprocal basis of the primitive cell as
//! h = (M^s)^-T g~' = K + q, with the supercell matrix M^s, the integer part
//! K and the fractional part q.  With N_c = |det M^s|, N_c (M^s)^-T is an
//! integer matrix, so N_c h is an integer vector and K = floor(N_c h / N_c).
//! With the Smith normal form P M^s Q = diag(d), q is labeled by
//! n_q = Q^T g~' mod d, and i_q = ((n_q)_1 d_2 + (n_q)_2) d_3 + (n_q)_3
//! (C order over d).  With Kbar = K mod N^p, the bin of g~' is
//!
//! ```text
//! i_F = ((i_q N^p_3 + Kbar_3) N^p_2 + Kbar_2) N^p_1 + Kbar_1,
//! ```
//!
//! and the term of g~ is added to F[i_F].
//!
//! The binning has two steps.  `sort_g_points_by_bin` returns g~' of all
//! selected g-points sorted by bin, and
//! `bin_sorted_supercell_fourier_coefficients` adds, for each g~' in this
//! order, the coefficient and the weight of g~ = (W^s)^T g~' with the phase
//! of g~'.  When (W^s)^-T maps the selected g-points onto themselves
//! (`g_points_closed_under_rotation`), the g~' are the selected g-points
//! themselves, so the selected g-points sorted once with the identity can be
//! passed for this rotation: their bins do not depend on the operation.
//!
//! The sums are deterministic: the g-points are sorted by (i_F, index of
//! g~ in the input) and each bin is summed in this order for any number of
//! threads.

use rayon::prelude::*;

use crate::common::{
    cmplx_add, cmplx_mul, cmplx_scale, det_i, matvec_ii, transpose_i, Cmplx, MatI, Vec3D, Vec3I,
    CMPLX_ZERO, IDENTITY,
};
use crate::snf3x3::snf3x3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FourierBinsError {
    /// The supercell matrix is singular.
    SingularP2sMatrix,
    /// The determinant of the rotation is not +1 or -1.
    NonUnimodularRotation,
    /// The Smith normal form of the supercell matrix did not converge.
    SnfNotConverged,
    /// A component of a mesh is not positive.
    NonPositiveMesh,
    /// The number of g-points or of bins exceeds 2^32.
    TooLarge,
    /// The lengths of the inputs or outputs do not agree.
    LengthMismatch,
    /// bin_offsets or source_offsets does not go from 0 to the number of
    /// g-points without decreasing, or the g-points are not in the order of
    /// source_offsets.
    InvalidOffsets,
}

/// Integer matrices that map g~ to g~' and g~' to its bin.
struct Binning {
    /// (W^s)^-T.
    w_inv_t: MatI,
    /// N_c (M^s)^-T.
    nc_m_inv_t: MatI,
    /// Q^T.
    q_t: MatI,
    num_cells: i64,
    snf_diag: Vec3I,
    fft_mesh: Vec3I,
}

impl Binning {
    fn new(p2s_matrix: MatI, fft_mesh: Vec3I, rotation: MatI) -> Result<Self, FourierBinsError> {
        let det = det_i(&p2s_matrix);
        if det == 0 {
            return Err(FourierBinsError::SingularP2sMatrix);
        }
        if fft_mesh.iter().any(|&n| n <= 0) {
            return Err(FourierBinsError::NonPositiveMesh);
        }
        let det_w = det_i(&rotation);
        if det_w.abs() != 1 {
            return Err(FourierBinsError::NonUnimodularRotation);
        }
        let snf = snf3x3(p2s_matrix).map_err(|_| FourierBinsError::SnfNotConverged)?;
        // (W^s)^-1 = adj(W^s) / det(W^s) = det(W^s) adj(W^s).
        let w_inv = adjugate(&rotation).map(|row| row.map(|x| x * det_w));
        // N_c (M^s)^-1 = N_c adj(M^s) / det(M^s) = sign(det M^s) adj(M^s).
        let nc_m_inv = adjugate(&p2s_matrix).map(|row| row.map(|x| x * det.signum()));
        Ok(Self {
            w_inv_t: transpose_i(&w_inv),
            nc_m_inv_t: transpose_i(&nc_m_inv),
            q_t: transpose_i(&snf.q),
            num_cells: det.abs(),
            snf_diag: snf.d_diag,
            fft_mesh,
        })
    }

    fn num_bins(&self) -> i64 {
        self.num_cells * self.fft_mesh[0] * self.fft_mesh[1] * self.fft_mesh[2]
    }

    /// g~' = (W^s)^-T g~.
    fn rotate(&self, g: Vec3I) -> Vec3I {
        matvec_ii(&self.w_inv_t, g)
    }

    /// Flat index i_F of the bin of g~'.
    fn i_f(&self, gp: Vec3I) -> u64 {
        let nh = matvec_ii(&self.nc_m_inv_t, gp);
        let n_q = matvec_ii(&self.q_t, gp);
        let d = self.snf_diag;
        let i_q = (n_q[0].rem_euclid(d[0]) * d[1] + n_q[1].rem_euclid(d[1])) * d[2]
            + n_q[2].rem_euclid(d[2]);
        let [n1, n2, n3] = self.fft_mesh;
        let kbar = |k: usize| {
            nh[k]
                .div_euclid(self.num_cells)
                .rem_euclid(self.fft_mesh[k])
        };
        (((i_q * n3 + kbar(2)) * n2 + kbar(1)) * n1 + kbar(0)) as u64
    }
}

/// Adjugate matrix, adj(M) M = det(M) I.
fn adjugate(m: &MatI) -> MatI {
    [
        [
            m[1][1] * m[2][2] - m[1][2] * m[2][1],
            m[2][1] * m[0][2] - m[2][2] * m[0][1],
            m[0][1] * m[1][2] - m[0][2] * m[1][1],
        ],
        [
            m[1][2] * m[2][0] - m[1][0] * m[2][2],
            m[2][2] * m[0][0] - m[2][0] * m[0][2],
            m[0][2] * m[1][0] - m[0][0] * m[1][2],
        ],
        [
            m[1][0] * m[2][1] - m[1][1] * m[2][0],
            m[2][0] * m[0][1] - m[2][1] * m[0][0],
            m[0][0] * m[1][1] - m[0][1] * m[1][0],
        ],
    ]
}

/// Source s(g) of g: the index of c(g) in the flattened coefficient array
/// of the supercell FFT mesh, g_1 fastest.
fn source_of(g: Vec3I, supercell_mesh: Vec3I) -> usize {
    let r = [0, 1, 2].map(|k| g[k].rem_euclid(supercell_mesh[k]));
    ((r[2] * supercell_mesh[1] + r[1]) * supercell_mesh[0] + r[0]) as usize
}

/// exp(-2 pi i g' . w^s) for the g-point g' and the translation w^s.
fn translation_phase(gp: Vec3I, w: Vec3D) -> Cmplx {
    let two_pi = 2.0 * std::f64::consts::PI;
    let arg = (-two_pi * gp[0] as f64) * w[0]
        + (-two_pi * gp[1] as f64) * w[1]
        + (-two_pi * gp[2] as f64) * w[2];
    let (s, c) = arg.sin_cos();
    [c, s]
}

/// Whether offsets go from 0 to len without decreasing.
fn offsets_are_valid(offsets: &[i64], len: usize) -> bool {
    offsets.first() == Some(&0)
        && offsets.windows(2).all(|w| w[0] <= w[1])
        && offsets[offsets.len() - 1] as usize == len
}

/// Split the bins 0..num_bins into one range per task, and every output at
/// the same bins, so that each task writes its own part of every output.
/// Returns the bounds of the ranges and, for each task, its parts of the
/// outputs.
fn split_bins<'a>(
    fs: &'a mut [&mut [Cmplx]],
    num_bins: usize,
) -> (Vec<usize>, Vec<Vec<&'a mut [Cmplx]>>) {
    let num_tasks = (rayon::current_num_threads() * 8).min(num_bins).max(1);
    let bin_bounds: Vec<usize> = (0..=num_tasks).map(|t| t * num_bins / num_tasks).collect();
    let mut parts: Vec<Vec<&mut [Cmplx]>> = (0..num_tasks).map(|_| Vec::new()).collect();
    for f in fs.iter_mut() {
        let mut rest: &mut [Cmplx] = f;
        for t in 0..num_tasks {
            let (part, tail) = rest.split_at_mut(bin_bounds[t + 1] - bin_bounds[t]);
            parts[t].push(part);
            rest = tail;
        }
    }
    (bin_bounds, parts)
}

/// The rotated selected g-points sorted by bin, from `sort_g_points_by_bin`.
pub struct GPointsSortedByBin {
    /// g~' = (W^s)^-T g~ of all selected g-points g~ in the order of
    /// (bin of g~', index of g~ in g).
    pub g_sorted: Vec<Vec3I>,
    /// The g~' of bin b are g_sorted[bin_offsets[b]..bin_offsets[b + 1]].
    /// Length is the number of bins N_c N^p_1 N^p_2 N^p_3 plus one.
    pub bin_offsets: Vec<i64>,
}

/// Sort g~' = (W^s)^-T g~ of the selected g-points g by the bin of g~'.
///
/// The g-points of a bin keep the order of g, so the result does not depend
/// on the number of threads.  With the identity as `rotation`, the selected
/// g-points themselves are sorted, as used for every rotation under which
/// they are closed (`g_points_closed_under_rotation`).
///
/// # Errors
///
/// `SingularP2sMatrix`, `NonPositiveMesh` for `fft_mesh`,
/// `NonUnimodularRotation`, `SnfNotConverged`, and `TooLarge` for more than
/// 2^32 g-points or bins.
pub fn sort_g_points_by_bin(
    g: &[Vec3I],
    p2s_matrix: MatI,
    fft_mesh: Vec3I,
    rotation: MatI,
) -> Result<GPointsSortedByBin, FourierBinsError> {
    let binning = Binning::new(p2s_matrix, fft_mesh, rotation)?;
    let num_bins = binning.num_bins();
    if g.len() as u64 > 1 << 32 || num_bins as u64 > 1 << 32 {
        return Err(FourierBinsError::TooLarge);
    }
    // key: i_F of g~' in the upper 32 bits, index j of g~ in the lower 32 bits.
    let mut keys: Vec<u64> = g
        .par_iter()
        .enumerate()
        .map(|(j, &gj)| (binning.i_f(binning.rotate(gj)) << 32) | j as u64)
        .collect();
    keys.par_sort_unstable();
    let g_sorted: Vec<Vec3I> = keys
        .par_iter()
        .map(|&k| binning.rotate(g[(k & 0xffff_ffff) as usize]))
        .collect();
    let mut bin_offsets = vec![0i64; num_bins as usize + 1];
    for &k in &keys {
        bin_offsets[(k >> 32) as usize + 1] += 1;
    }
    for b in 0..num_bins as usize {
        bin_offsets[b + 1] += bin_offsets[b];
    }
    Ok(GPointsSortedByBin {
        g_sorted,
        bin_offsets,
    })
}

/// Sum the Fourier coefficients by bin into `fs`.
///
/// The g~' of bin b are `g_sorted[bin_offsets[b]..bin_offsets[b + 1]]`.
/// For each g~', the coefficient of g~ = (W^s)^T g~' times its weight and
/// exp(-2 pi i g~' . w^s) is added to F[b]:
///
/// ```text
/// F[b] = sum_{g~' in bin b} c[s(g~)] source_weight[s(g~)]
///        exp(-2 pi i g~' . w^s).
/// ```
///
/// This is the binned series of the rotated function when `g_sorted` are
/// the rotated selected g-points (W^s)^-T g~ sorted by bin
/// (`sort_g_points_by_bin` with `rotation`), and also when they are the
/// selected g-points sorted once with the identity and these are closed
/// under (W^s)^-T (`g_points_closed_under_rotation`).  The terms are then
/// the same, added in another order.
///
/// `coefficients[c]` and `source_weight` have the length
/// N^s_1 N^s_2 N^s_3 of the supercell FFT mesh, and `fs[c]` the number of
/// bins `bin_offsets.len() - 1`; every element of `fs[c]` is written.
/// `source_weight[s]` is 1 / (number of selected g-points with source s).
///
/// # Errors
///
/// `NonPositiveMesh`, `NonUnimodularRotation`, `LengthMismatch` for the
/// lengths of `fs`, `coefficients` or `source_weight`, and `InvalidOffsets`
/// when `bin_offsets` does not go from 0 to `g_sorted.len()` without
/// decreasing.
#[allow(clippy::too_many_arguments)]
pub fn bin_sorted_supercell_fourier_coefficients(
    fs: &mut [&mut [Cmplx]],
    coefficients: &[&[Cmplx]],
    supercell_mesh: Vec3I,
    g_sorted: &[Vec3I],
    source_weight: &[f64],
    bin_offsets: &[i64],
    rotation: MatI,
    translation: Vec3D,
) -> Result<(), FourierBinsError> {
    use FourierBinsError as E;
    if supercell_mesh.iter().any(|&n| n <= 0) {
        return Err(E::NonPositiveMesh);
    }
    if det_i(&rotation).abs() != 1 {
        return Err(E::NonUnimodularRotation);
    }
    let num_coefs = (supercell_mesh[0] * supercell_mesh[1] * supercell_mesh[2]) as usize;
    if fs.len() != coefficients.len()
        || coefficients.iter().any(|c| c.len() != num_coefs)
        || source_weight.len() != num_coefs
    {
        return Err(E::LengthMismatch);
    }
    if !offsets_are_valid(bin_offsets, g_sorted.len()) {
        return Err(E::InvalidOffsets);
    }
    let num_bins = bin_offsets.len() - 1;
    if fs.iter().any(|f| f.len() != num_bins) {
        return Err(E::LengthMismatch);
    }

    let w_t = transpose_i(&rotation);
    let (bin_bounds, parts) = split_bins(fs, num_bins);
    parts.into_par_iter().enumerate().for_each(|(t, mut outs)| {
        let mut sums = vec![CMPLX_ZERO; coefficients.len()];
        for (local, bin) in (bin_bounds[t]..bin_bounds[t + 1]).enumerate() {
            sums.fill(CMPLX_ZERO);
            let in_bin = bin_offsets[bin] as usize..bin_offsets[bin + 1] as usize;
            for &gp in &g_sorted[in_bin] {
                let s = source_of(matvec_ii(&w_t, gp), supercell_mesh);
                let weight = source_weight[s];
                let phase = translation_phase(gp, translation);
                for (sum, c) in sums.iter_mut().zip(coefficients) {
                    let v = cmplx_mul(cmplx_scale(weight, c[s]), phase);
                    *sum = cmplx_add(*sum, v);
                }
            }
            for (out, sum) in outs.iter_mut().zip(&sums) {
                out[local] = *sum;
            }
        }
    });
    Ok(())
}

/// Whether (W^s)^-T maps the selected g-points `g` onto themselves.
///
/// `g` is in the order of the source: the selected g-points with source s
/// are `g[source_offsets[s]..source_offsets[s + 1]]`.  For each selected
/// g-point g~, g~' = (W^s)^-T g~ has to be one of the selected g-points with
/// the source of g~'.  The set is finite and the map is one-to-one, so this
/// is (W^s)^-T G = G for the set G of the selected g-points.  The check
/// stops at the first g~' that is not found.
///
/// # Errors
///
/// `NonPositiveMesh`, `NonUnimodularRotation`, `LengthMismatch` when
/// `source_offsets` does not have the length N^s_1 N^s_2 N^s_3 + 1, and
/// `InvalidOffsets` when it does not go from 0 to `g.len()` without
/// decreasing or `g` is not in its order.
pub fn g_points_closed_under_rotation(
    g: &[Vec3I],
    source_offsets: &[i64],
    supercell_mesh: Vec3I,
    rotation: MatI,
) -> Result<bool, FourierBinsError> {
    use FourierBinsError as E;
    if supercell_mesh.iter().any(|&n| n <= 0) {
        return Err(E::NonPositiveMesh);
    }
    let binning = Binning::new(IDENTITY, [1, 1, 1], rotation)?;
    let num_coefs = (supercell_mesh[0] * supercell_mesh[1] * supercell_mesh[2]) as usize;
    if source_offsets.len() != num_coefs + 1 {
        return Err(E::LengthMismatch);
    }
    let members = |s: usize| &g[source_offsets[s] as usize..source_offsets[s + 1] as usize];
    let in_order = offsets_are_valid(source_offsets, g.len())
        && g.par_iter().enumerate().all(|(k, &gt)| {
            let s = source_of(gt, supercell_mesh);
            (source_offsets[s] as usize..source_offsets[s + 1] as usize).contains(&k)
        });
    if !in_order {
        return Err(E::InvalidOffsets);
    }
    Ok(g.par_iter().all(|&gt| {
        let gp = binning.rotate(gt);
        members(source_of(gp, supercell_mesh)).contains(&gp)
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    /// diag(2, 4, 6) multiplied by unimodular matrices from both sides.  Its
    /// Smith normal form has two diagonal elements larger than one, so that
    /// C and Fortran orders of n_q differ.
    fn p2s_skewed() -> MatI {
        [[6, 4, 0], [4, 4, 0], [0, 6, 6]]
    }

    /// Conventional cell of FCC times 2 relative to the primitive cell.
    fn p2s_fcc() -> MatI {
        [[-2, 2, 2], [2, -2, 2], [2, 2, -2]]
    }

    /// Three-fold rotation (cyclic permutation of the axes).
    fn three_fold() -> MatI {
        [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    }

    /// Six-fold rotation of a hexagonal lattice in its basis.
    fn six_fold() -> MatI {
        [[1, -1, 0], [1, 0, 0], [0, 0, 1]]
    }

    /// Deterministic pseudo-random coefficients.
    fn coefficients(len: usize, seed: u64) -> Vec<Cmplx> {
        let mut x = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..len)
            .map(|_| {
                let mut next = || {
                    x = x
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((x >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
                };
                [next(), next()]
            })
            .collect()
    }

    /// Box g from the formula of the phelel docstring, in the order of the
    /// source (g_3 slowest, g_1 fastest).
    fn box_g(mesh: Vec3I) -> Vec<Vec3I> {
        let freq =
            |n: i64| -> Vec<i64> { (0..n).map(|i| if i <= n / 2 { i } else { i - n }).collect() };
        let mut g = Vec::new();
        for &g3 in &freq(mesh[2]) {
            for &g2 in &freq(mesh[1]) {
                for &g1 in &freq(mesh[0]) {
                    g.push([g1, g2, g3]);
                }
            }
        }
        g
    }

    /// Basis vectors (columns) of a hexagonal lattice, which is not
    /// orthogonal, so that W^T and W^-1 differ for its rotations.
    fn hexagonal_lattice() -> [[f64; 3]; 3] {
        let s = 3.0f64.sqrt() / 2.0;
        [[1.0, -0.5, 0.0], [0.0, s, 0.0], [0.0, 0.0, 1.6]]
    }

    /// Basis vectors (columns) of the fcc primitive cell.
    fn fcc_primitive_lattice() -> [[f64; 3]; 3] {
        [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
    }

    fn cubic_lattice() -> [[f64; 3]; 3] {
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    }

    /// Integer matrices W with elements -1, 0, 1 and W^T G W = G for the
    /// metric G = A^T A of the basis A (columns): the rotations of the
    /// lattice in fractional coordinates.
    fn lattice_rotations(lattice: [[f64; 3]; 3]) -> Vec<MatI> {
        let metric =
            |i: usize, j: usize| (0..3).map(|k| lattice[k][i] * lattice[k][j]).sum::<f64>();
        let mut rotations = Vec::new();
        for code in 0..3i64.pow(9) {
            let mut w = [[0i64; 3]; 3];
            let mut c = code;
            for row in w.iter_mut() {
                for x in row.iter_mut() {
                    *x = c % 3 - 1;
                    c /= 3;
                }
            }
            if det_i(&w).abs() != 1 {
                continue;
            }
            let keeps_metric = (0..3).all(|i| {
                (0..3).all(|j| {
                    let wgw: f64 = (0..3)
                        .flat_map(|k| (0..3).map(move |l| (k, l)))
                        .map(|(k, l)| w[k][i] as f64 * metric(k, l) * w[l][j] as f64)
                        .sum();
                    (wgw - metric(i, j)).abs() < 1e-10
                })
            });
            if keeps_metric {
                rotations.push(w);
            }
        }
        rotations
    }

    /// Selected g-points with their sources, source_offsets and source_weight.
    struct SelectedGPoints {
        g: Vec<Vec3I>,
        source: Vec<usize>,
        source_offsets: Vec<i64>,
        source_weight: Vec<f64>,
    }

    /// The box: one selected g-point per source, in the order of the source.
    fn box_g_points(mesh: Vec3I) -> SelectedGPoints {
        let g = box_g(mesh);
        let n = g.len();
        SelectedGPoints {
            source: (0..n).collect(),
            source_offsets: (0..=n as i64).collect(),
            source_weight: vec![1.0; n],
            g,
        }
    }

    /// The shortest selected g-points for the lattice A (columns), from
    /// `bz_grid_addresses` with the period lattice N^s_k b_k.
    fn shortest_g_points(mesh: Vec3I, lattice: [[f64; 3]; 3]) -> SelectedGPoints {
        let a = lattice;
        let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
        // Reciprocal basis b_k (columns of A^-T), from the cofactors of A.
        let cof = |i: usize, j: usize| {
            let (i1, i2) = ((i + 1) % 3, (i + 2) % 3);
            let (j1, j2) = ((j + 1) % 3, (j + 2) % 3);
            a[i1][j1] * a[i2][j2] - a[i1][j2] * a[i2][j1]
        };
        let mut period = [[0.0; 3]; 3];
        for i in 0..3 {
            for k in 0..3 {
                period[i][k] = cof(i, k) / det * mesh[k] as f64;
            }
        }
        let bz = crate::bzgrid::bz_grid_addresses(mesh, IDENTITY, [0, 0, 0], period, 2).unwrap();
        let source: Vec<usize> = bz.bzg2grg.iter().map(|&s| s as usize).collect();
        let source_weight = bz
            .bz_map
            .windows(2)
            .map(|w| 1.0 / (w[1] - w[0]) as f64)
            .collect();
        SelectedGPoints {
            g: bz.addresses,
            source,
            source_offsets: bz.bz_map,
            source_weight,
        }
    }

    /// The binned series summed term by term in the order of the selected
    /// g-points, without sorting: the term of g~ is added to the bin of
    /// g~' = (W^s)^-T g~.
    fn reference(
        selected: &SelectedGPoints,
        coefs: &[Cmplx],
        p2s: MatI,
        fft_mesh: Vec3I,
        rotation: MatI,
        translation: Vec3D,
    ) -> Vec<Cmplx> {
        let binning = Binning::new(p2s, fft_mesh, rotation).unwrap();
        let mut f = vec![[0.0; 2]; binning.num_bins() as usize];
        for (j, &gj) in selected.g.iter().enumerate() {
            let gp = binning.rotate(gj);
            let s = selected.source[j];
            let c = coefs[s];
            let weight = selected.source_weight[s];
            let w = translation;
            let two_pi = 2.0 * std::f64::consts::PI;
            let arg = (-two_pi * gp[0] as f64) * w[0]
                + (-two_pi * gp[1] as f64) * w[1]
                + (-two_pi * gp[2] as f64) * w[2];
            let (sin, cos) = arg.sin_cos();
            let v = cmplx_mul([c[0] * weight, c[1] * weight], [cos, sin]);
            let i = binning.i_f(gp) as usize;
            f[i][0] += v[0];
            f[i][1] += v[1];
        }
        f
    }

    /// Sort with `sort_rotation`, then sum with `rotation`, into outputs
    /// filled with NaN.
    #[allow(clippy::too_many_arguments)]
    fn run(
        coefs: &[Vec<Cmplx>],
        mesh: Vec3I,
        selected: &SelectedGPoints,
        p2s: MatI,
        fft_mesh: Vec3I,
        sort_rotation: MatI,
        rotation: MatI,
        translation: Vec3D,
    ) -> Vec<Vec<Cmplx>> {
        let sorted = sort_g_points_by_bin(&selected.g, p2s, fft_mesh, sort_rotation).unwrap();
        let n = sorted.bin_offsets.len() - 1;
        let mut fs: Vec<Vec<Cmplx>> = coefs.iter().map(|_| vec![[f64::NAN; 2]; n]).collect();
        let mut views: Vec<&mut [Cmplx]> = fs.iter_mut().map(|f| f.as_mut_slice()).collect();
        let cs: Vec<&[Cmplx]> = coefs.iter().map(|c| c.as_slice()).collect();
        bin_sorted_supercell_fourier_coefficients(
            &mut views,
            &cs,
            mesh,
            &sorted.g_sorted,
            &selected.source_weight,
            &sorted.bin_offsets,
            rotation,
            translation,
        )
        .unwrap();
        fs
    }

    /// Closure checked with a hash set of all selected g-points.
    fn closed_by_set(selected: &SelectedGPoints, rotation: MatI) -> bool {
        let binning = Binning::new(IDENTITY, [1, 1, 1], rotation).unwrap();
        let set: HashSet<Vec3I> = selected.g.iter().copied().collect();
        selected.g.iter().all(|&g| set.contains(&binning.rotate(g)))
    }

    /// Lattices, meshes and supercell matrices of the tests.  Some meshes are
    /// not kept by some rotations of the lattice, so that the selected
    /// g-points are not closed under them.
    fn cases() -> Vec<([[f64; 3]; 3], Vec3I, MatI, Vec3I)> {
        vec![
            (
                hexagonal_lattice(),
                [6, 6, 4],
                [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
                [2, 3, 2],
            ),
            (
                hexagonal_lattice(),
                [6, 5, 4],
                [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
                [2, 3, 2],
            ),
            (fcc_primitive_lattice(), [5, 5, 5], p2s_fcc(), [2, 2, 3]),
            (fcc_primitive_lattice(), [4, 4, 6], p2s_skewed(), [2, 3, 2]),
        ]
    }

    #[test]
    fn source_of_box_follows_phelel_order() {
        let mesh = [3, 2, 4];
        let g = box_g(mesh);
        for (j, &gj) in g.iter().enumerate() {
            assert_eq!(source_of(gj, mesh), j);
            assert_eq!(source_of([gj[0] + 3, gj[1] - 4, gj[2] + 8], mesh), j);
        }
        assert_eq!(
            &g[..6],
            &[
                [0, 0, 0],
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [-1, 1, 0]
            ]
        );
    }

    /// g' = (W^s)^-T g, checked by (W^s)^T g' = g without the inverse.
    #[test]
    fn rotate_applies_inverse_transpose() {
        // Six-fold rotation of a hexagonal lattice, which is not orthogonal,
        // and the same times the inversion, with determinant -1.
        let rotoinversion = six_fold().map(|row| row.map(|x| -x));
        for w in [three_fold(), six_fold(), rotoinversion] {
            let binning = Binning::new(IDENTITY, [1, 1, 1], w).unwrap();
            for g in box_g([5, 4, 3]) {
                assert_eq!(matvec_ii(&transpose_i(&w), binning.rotate(g)), g);
            }
        }
    }

    /// N_c h = N_c (M^s)^-T g' splits into N_c K + N_c q, where
    /// q = P^T diag(d)^-1 n_q mod 1 with n_q unraveled from i_q in C order,
    /// and Kbar = K mod N^p.
    #[test]
    fn index_splits_h_into_k_and_q() {
        let fft_mesh = [3, 4, 5];
        for p2s in [p2s_skewed(), p2s_fcc()] {
            let snf = snf3x3(p2s).unwrap();
            let d = snf.d_diag;
            assert!(d.iter().filter(|&&x| x > 1).count() >= 2);
            let binning = Binning::new(p2s, fft_mesh, three_fold()).unwrap();
            let nc = binning.num_cells;
            // M^s N_c (M^s)^-1 = N_c I checks N_c (M^s)^-1 independently of
            // the adjugate.
            let nc_m_inv = transpose_i(&binning.nc_m_inv_t);
            assert_eq!(
                crate::common::matmul_i(&p2s, &nc_m_inv),
                [[nc, 0, 0], [0, nc, 0], [0, 0, nc]]
            );
            for g in box_g([7, 6, 5]) {
                let gp = binning.rotate(g);
                let mut i = binning.i_f(gp) as i64;
                let kbar1 = i % fft_mesh[0];
                i /= fft_mesh[0];
                let kbar2 = i % fft_mesh[1];
                i /= fft_mesh[1];
                let kbar3 = i % fft_mesh[2];
                let i_q = i / fft_mesh[2];
                let n_q = [i_q / (d[1] * d[2]), (i_q / d[2]) % d[1], i_q % d[2]];
                // N_c q in row vectors: (n_q * (N_c / d)) P mod N_c.
                let scaled = [0, 1, 2].map(|k| n_q[k] * (nc / d[k]));
                let nq = matvec_ii(&transpose_i(&snf.p), scaled).map(|x| x.rem_euclid(nc));
                let nh = matvec_ii(&binning.nc_m_inv_t, gp);
                for k in 0..3 {
                    assert_eq!(nh[k].rem_euclid(nc), nq[k]);
                    let kk = (nh[k] - nq[k]) / nc;
                    assert_eq!(kk.rem_euclid(fft_mesh[k]), [kbar1, kbar2, kbar3][k]);
                }
            }
        }
    }

    #[test]
    fn test_lattices_have_expected_rotations() {
        assert_eq!(lattice_rotations(hexagonal_lattice()).len(), 24);
        assert_eq!(lattice_rotations(fcc_primitive_lattice()).len(), 48);
    }

    /// g_sorted are the rotated selected g-points, each in its bin, and the
    /// g-points of a bin keep the order of the selected g-points.
    #[test]
    fn sort_g_points_by_bin_sorts_rotated_g_points() {
        let mesh = [6, 6, 4];
        let selected = shortest_g_points(mesh, hexagonal_lattice());
        assert!(selected.g.len() > (mesh[0] * mesh[1] * mesh[2]) as usize);
        let p2s = [[2, 0, 0], [0, 2, 0], [0, 0, 1]];
        let fft_mesh = [2, 3, 2];
        let index_of: HashMap<Vec3I, usize> = selected
            .g
            .iter()
            .enumerate()
            .map(|(j, &g)| (g, j))
            .collect();
        for rotation in [IDENTITY, six_fold()] {
            let sorted = sort_g_points_by_bin(&selected.g, p2s, fft_mesh, rotation).unwrap();
            let binning = Binning::new(p2s, fft_mesh, rotation).unwrap();
            assert_eq!(sorted.bin_offsets.len() as i64, binning.num_bins() + 1);
            assert!(offsets_are_valid(&sorted.bin_offsets, selected.g.len()));
            let w_t = transpose_i(&rotation);
            let mut seen = vec![false; selected.g.len()];
            for b in 0..sorted.bin_offsets.len() - 1 {
                let range = sorted.bin_offsets[b] as usize..sorted.bin_offsets[b + 1] as usize;
                let indices: Vec<usize> = sorted.g_sorted[range]
                    .iter()
                    .map(|&gp| {
                        assert_eq!(binning.i_f(gp) as usize, b);
                        index_of[&matvec_ii(&w_t, gp)]
                    })
                    .collect();
                assert!(indices.windows(2).all(|w| w[0] < w[1]));
                for j in indices {
                    assert!(!seen[j]);
                    seen[j] = true;
                }
            }
            assert!(seen.iter().all(|&x| x));
        }
    }

    /// Sorting the rotated selected g-points for each rotation gives the
    /// term-by-term sum exactly, for box and shortest selected g-points and
    /// for every rotation of the lattice, also those under which the
    /// selected g-points are not closed.  The terms are added in the same
    /// order, so the results are equal bit by bit.
    #[test]
    fn g_points_sorted_for_each_rotation_equal_reference() {
        let mut num_not_closed = 0;
        for (lattice, mesh, p2s, fft_mesh) in cases() {
            let num_coefs = (mesh[0] * mesh[1] * mesh[2]) as usize;
            let coefs = vec![coefficients(num_coefs, 1), coefficients(num_coefs, 2)];
            for selected in [box_g_points(mesh), shortest_g_points(mesh, lattice)] {
                for rotation in lattice_rotations(lattice) {
                    if !closed_by_set(&selected, rotation) {
                        num_not_closed += 1;
                    }
                    for translation in [[0.0; 3], [0.1, -0.3, 0.7]] {
                        let fs = run(
                            &coefs,
                            mesh,
                            &selected,
                            p2s,
                            fft_mesh,
                            rotation,
                            rotation,
                            translation,
                        );
                        for (c, f) in coefs.iter().zip(&fs) {
                            let r = reference(&selected, c, p2s, fft_mesh, rotation, translation);
                            assert_eq!(f, &r);
                        }
                    }
                }
            }
        }
        assert!(num_not_closed > 0);
    }

    /// For a rotation under which the selected g-points are closed, the
    /// selected g-points sorted once with the identity give the term-by-term
    /// sum up to rounding (the terms are added in another order).
    #[test]
    fn g_points_sorted_once_equal_reference_when_closed() {
        let mut num_closed = 0;
        for (lattice, mesh, p2s, fft_mesh) in cases() {
            let num_coefs = (mesh[0] * mesh[1] * mesh[2]) as usize;
            let coefs = vec![coefficients(num_coefs, 3)];
            for selected in [box_g_points(mesh), shortest_g_points(mesh, lattice)] {
                for rotation in lattice_rotations(lattice) {
                    if !closed_by_set(&selected, rotation) {
                        continue;
                    }
                    num_closed += 1;
                    let translation = [0.1, -0.3, 0.7];
                    let fs = run(
                        &coefs,
                        mesh,
                        &selected,
                        p2s,
                        fft_mesh,
                        IDENTITY,
                        rotation,
                        translation,
                    );
                    let r = reference(&selected, &coefs[0], p2s, fft_mesh, rotation, translation);
                    let scale = r
                        .iter()
                        .map(|v| v[0].abs().max(v[1].abs()))
                        .fold(0.0, f64::max);
                    assert!(scale > 0.1);
                    for (a, b) in fs[0].iter().zip(&r) {
                        assert!((a[0] - b[0]).abs() < 1e-13 * scale);
                        assert!((a[1] - b[1]).abs() < 1e-13 * scale);
                    }
                }
            }
        }
        assert!(num_closed > 0);
    }

    /// g_points_closed_under_rotation agrees with the check by a hash set.  The
    /// mesh (4, 4, 6) with the axes 1 and 3 exchanged: (0, 0, 3) is a
    /// shortest selected g-point and (3, 0, 0) is not, so it is not closed.
    #[test]
    fn g_points_closed_under_rotation_equals_set_check() {
        let mut counts = [0usize; 2];
        for (lattice, mesh, _, _) in cases() {
            for selected in [box_g_points(mesh), shortest_g_points(mesh, lattice)] {
                for rotation in lattice_rotations(lattice) {
                    let closed = g_points_closed_under_rotation(
                        &selected.g,
                        &selected.source_offsets,
                        mesh,
                        rotation,
                    )
                    .unwrap();
                    assert_eq!(closed, closed_by_set(&selected, rotation));
                    counts[closed as usize] += 1;
                }
            }
        }
        assert!(counts[0] > 0 && counts[1] > 0);
        let mesh = [4, 4, 6];
        let selected = shortest_g_points(mesh, cubic_lattice());
        assert!(selected.g.contains(&[0, 0, 3]) && !selected.g.contains(&[3, 0, 0]));
        let swap13 = [[0, 0, 1], [0, 1, 0], [1, 0, 0]];
        assert_eq!(
            g_points_closed_under_rotation(&selected.g, &selected.source_offsets, mesh, swap13),
            Ok(false)
        );
        assert_eq!(
            g_points_closed_under_rotation(&selected.g, &selected.source_offsets, mesh, IDENTITY),
            Ok(true)
        );
    }

    /// Every element of the outputs is written, also the bins without
    /// g-points.
    #[test]
    fn every_bin_is_written() {
        let mesh = [4, 4, 4];
        let selected = shortest_g_points(mesh, fcc_primitive_lattice());
        let p2s = p2s_fcc();
        let fft_mesh = [3, 3, 3];
        let sorted = sort_g_points_by_bin(&selected.g, p2s, fft_mesh, IDENTITY).unwrap();
        let num_empty = sorted
            .bin_offsets
            .windows(2)
            .filter(|w| w[0] == w[1])
            .count();
        assert!(num_empty > 0);
        let coefs = vec![coefficients(64, 8)];
        let fs = run(
            &coefs,
            mesh,
            &selected,
            p2s,
            fft_mesh,
            IDENTITY,
            three_fold(),
            [0.25, 0.5, 0.125],
        );
        assert!(fs[0].iter().all(|v| v[0].is_finite() && v[1].is_finite()));
        for (b, w) in sorted.bin_offsets.windows(2).enumerate() {
            if w[0] == w[1] {
                assert_eq!(fs[0][b], [0.0, 0.0]);
            }
        }
    }

    #[test]
    fn independent_of_number_of_threads() {
        let mesh = [6, 6, 4];
        let selected = shortest_g_points(mesh, hexagonal_lattice());
        let coefs = vec![coefficients(144, 9)];
        let results: Vec<Vec<Vec<Cmplx>>> = [1, 3, 8]
            .iter()
            .map(|&n| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build()
                    .unwrap()
                    .install(|| {
                        run(
                            &coefs,
                            mesh,
                            &selected,
                            [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
                            [2, 3, 2],
                            six_fold(),
                            six_fold(),
                            [0.3, 0.1, 0.2],
                        )
                    })
            })
            .collect();
        assert_eq!(results[0], results[1]);
        assert_eq!(results[0], results[2]);
    }

    #[test]
    fn errors() {
        use FourierBinsError as E;
        let g = [[0, 0, 0], [1, 0, 0]];
        let singular = [[1, 0, 0], [0, 1, 0], [1, 1, 0]];
        let twice = [[2, 0, 0], [0, 1, 0], [0, 0, 1]];
        assert_eq!(
            sort_g_points_by_bin(&g, singular, [1, 1, 1], IDENTITY).err(),
            Some(E::SingularP2sMatrix)
        );
        assert_eq!(
            sort_g_points_by_bin(&g, IDENTITY, [1, 0, 1], IDENTITY).err(),
            Some(E::NonPositiveMesh)
        );
        assert_eq!(
            sort_g_points_by_bin(&g, IDENTITY, [1, 1, 1], twice).err(),
            Some(E::NonUnimodularRotation)
        );

        let mesh = [2, 2, 2];
        let c = coefficients(8, 5);
        let weight = vec![1.0; 8];
        let call = |mesh: Vec3I,
                    source_weight: &[f64],
                    bin_offsets: &[i64],
                    out_len: usize,
                    rotation: MatI| {
            let mut f = vec![[0.0; 2]; out_len];
            bin_sorted_supercell_fourier_coefficients(
                &mut [&mut f],
                &[&c],
                mesh,
                &g,
                source_weight,
                bin_offsets,
                rotation,
                [0.0; 3],
            )
        };
        assert_eq!(call(mesh, &weight, &[0, 1, 2], 2, IDENTITY), Ok(()));
        assert_eq!(
            call(mesh, &weight, &[0, 1, 2], 2, twice),
            Err(E::NonUnimodularRotation)
        );
        assert_eq!(
            call([2, 0, 2], &weight, &[0, 1, 2], 2, IDENTITY),
            Err(E::NonPositiveMesh)
        );
        assert_eq!(
            call(mesh, &weight[1..], &[0, 1, 2], 2, IDENTITY),
            Err(E::LengthMismatch)
        );
        assert_eq!(
            call(mesh, &weight, &[0, 1, 2], 3, IDENTITY),
            Err(E::LengthMismatch)
        );
        for offsets in [&[][..], &[1, 1, 2], &[0, 2, 1], &[0, 1, 1]] {
            assert_eq!(
                call(
                    mesh,
                    &weight,
                    offsets,
                    offsets.len().saturating_sub(1),
                    IDENTITY
                ),
                Err(E::InvalidOffsets)
            );
        }

        let selected = box_g_points(mesh);
        assert_eq!(
            g_points_closed_under_rotation(&selected.g, &selected.source_offsets, mesh, IDENTITY),
            Ok(true)
        );
        assert_eq!(
            g_points_closed_under_rotation(
                &selected.g,
                &selected.source_offsets[1..],
                mesh,
                IDENTITY
            ),
            Err(E::LengthMismatch)
        );
        let mut reversed = selected.g.clone();
        reversed.reverse();
        assert_eq!(
            g_points_closed_under_rotation(&reversed, &selected.source_offsets, mesh, IDENTITY),
            Err(E::InvalidOffsets)
        );
        assert_eq!(
            g_points_closed_under_rotation(&selected.g, &selected.source_offsets, mesh, twice),
            Err(E::NonUnimodularRotation)
        );
    }
}
