//! Binning of the Fourier coefficients of a supercell by q and K mod N^p.
//!
//! Kernel of the native evaluation of a Fourier series of the supercell on
//! the grid of the primitive cell in phelel (`FourierSeriesOnGrid.get_phi`).
//! The series has the coefficients c(g) at the integer coordinates g of the
//! reciprocal lattice vectors of the supercell (column vectors).  For a
//! space-group operation acting as x^s -> W^s x^s + w^s in the fractional
//! coordinates of the supercell,
//!
//! ```text
//! sum_g c(g) exp(2 pi i g . (W^s)^-1 (x^s - w^s))
//!     = sum_g c(g) exp(-2 pi i g' . w^s) exp(2 pi i g' . x^s),
//! g' = (W^s)^-T g.
//! ```
//!
//! g' is written in the reciprocal basis of the primitive cell as
//!
//! ```text
//! h = (M^s)^-T g' = K + q,
//! ```
//!
//! with the supercell matrix M^s, the integer part K and the fractional part
//! q.  With N_c = |det M^s|, N_c (M^s)^-T is an integer matrix, so N_c h is
//! an integer vector and K = floor(N_c h / N_c).  With the Smith normal form
//! P M^s Q = diag(d), q is labeled by n_q = Q^T g' mod d, and
//! i_q = ((n_q)_1 d_2 + (n_q)_2) d_3 + (n_q)_3 (C order over d).  With
//! Kbar = K mod N^p, the bin of g is
//!
//! ```text
//! i_F = ((i_q N^p_3 + Kbar_3) N^p_2 + Kbar_2) N^p_1 + Kbar_1,
//! ```
//!
//! and c(g) exp(-2 pi i g' . w^s) is added to F[i_F].
//!
//! The sums are deterministic: the rows are sorted by (i_F, row) and each
//! bin is summed in the order of the rows, as `np.bincount` does, for any
//! number of threads.

use rayon::prelude::*;

use crate::common::{
    cmplx_mul, det_i, matvec_ii, transpose_i, Cmplx, MatI, Vec3D, Vec3I, CMPLX_ZERO,
};
use crate::snf3x3::snf3x3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinSupercellFourierCoefficientsError {
    /// The supercell matrix is singular.
    SingularP2sMatrix,
    /// The determinant of the rotation is not +1 or -1.
    NonUnimodularRotation,
    /// The Smith normal form of the supercell matrix did not converge.
    SnfNotConverged,
    /// A component of the supercell FFT mesh or of fft_mesh is not positive.
    NonPositiveMesh,
    /// The number of rows or of bins exceeds 2^32.
    TooLarge,
    /// The lengths of the inputs or outputs do not agree.
    LengthMismatch,
    /// An element of source is outside [0, N^s_1 N^s_2 N^s_3).
    SourceOutOfRange,
    /// The supercell FFT mesh is not invariant under the rotation.
    MeshNotInvariant,
    /// bin_offsets does not go from 0 to the number of rows without
    /// decreasing.
    InvalidBinOffsets,
}

/// Explicitly given g with the coefficient c[source[j]] * weight[j] of row j.
pub struct Representatives<'a> {
    pub g: &'a [Vec3I],
    pub source: &'a [i64],
    pub weight: &'a [f64],
}

/// Integer matrices that map g to g' and g' to its bin.
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
    fn new(
        p2s_matrix: MatI,
        fft_mesh: Vec3I,
        rotation: MatI,
    ) -> Result<Self, BinSupercellFourierCoefficientsError> {
        let det = det_i(&p2s_matrix);
        if det == 0 {
            return Err(BinSupercellFourierCoefficientsError::SingularP2sMatrix);
        }
        if fft_mesh.iter().any(|&n| n <= 0) {
            return Err(BinSupercellFourierCoefficientsError::NonPositiveMesh);
        }
        let det_w = det_i(&rotation);
        if det_w.abs() != 1 {
            return Err(BinSupercellFourierCoefficientsError::NonUnimodularRotation);
        }
        let snf = snf3x3(p2s_matrix)
            .map_err(|_| BinSupercellFourierCoefficientsError::SnfNotConverged)?;
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

    /// g' = (W^s)^-T g.
    fn rotate(&self, g: Vec3I) -> Vec3I {
        matvec_ii(&self.w_inv_t, g)
    }

    /// Flat index i_F of the bin of g'.
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

/// The rows of the series: the box of the supercell FFT mesh or given g.
enum Rows<'a> {
    Box(Vec3I),
    Given(Representatives<'a>),
}

impl Rows<'_> {
    fn len(&self) -> usize {
        match self {
            Rows::Box(mesh) => (mesh[0] * mesh[1] * mesh[2]) as usize,
            Rows::Given(r) => r.g.len(),
        }
    }

    /// g of row j.  For the box, row j = (i_3 N^s_2 + i_2) N^s_1 + i_1 is
    /// g_k = i_k for i_k <= N^s_k / 2 and i_k - N^s_k otherwise.
    fn g(&self, j: usize) -> Vec3I {
        match self {
            Rows::Box(mesh) => {
                let j = j as i64;
                let i = [
                    j % mesh[0],
                    (j / mesh[0]) % mesh[1],
                    j / (mesh[0] * mesh[1]),
                ];
                [0, 1, 2].map(|k| {
                    if i[k] <= mesh[k] / 2 {
                        i[k]
                    } else {
                        i[k] - mesh[k]
                    }
                })
            }
            Rows::Given(r) => r.g[j],
        }
    }

    fn coefficient(&self, c: &[Cmplx], j: usize) -> Cmplx {
        match self {
            Rows::Box(_) => c[j],
            Rows::Given(r) => {
                let v = c[r.source[j] as usize];
                [v[0] * r.weight[j], v[1] * r.weight[j]]
            }
        }
    }
}

/// Number of bins N_c N^p_1 N^p_2 N^p_3, the length of each output of
/// `bin_supercell_fourier_coefficients`.
pub fn num_bins(
    p2s_matrix: MatI,
    fft_mesh: Vec3I,
) -> Result<usize, BinSupercellFourierCoefficientsError> {
    let binning = Binning::new(p2s_matrix, fft_mesh, crate::common::IDENTITY)?;
    Ok(binning.num_bins() as usize)
}

/// Sum the Fourier coefficients by bin into `fs`.
///
/// `coefficients[c]` has the length N^s_1 N^s_2 N^s_3 of the supercell FFT
/// mesh `mesh` (flattened with g_1 fastest) and `fs[c]` the length
/// `num_bins`; every element of `fs[c]` is written.  `rotation` is W^s of
/// the operation, and the rows g are binned by g' = (W^s)^-T g.  The phase
/// exp(-2 pi i g' . w^s) is applied when `translation` is w^s.  With
/// `representatives` None, the rows are the box of `mesh`.
pub fn bin_supercell_fourier_coefficients(
    fs: &mut [&mut [Cmplx]],
    coefficients: &[&[Cmplx]],
    mesh: Vec3I,
    p2s_matrix: MatI,
    fft_mesh: Vec3I,
    rotation: MatI,
    translation: Option<Vec3D>,
    representatives: Option<Representatives>,
) -> Result<(), BinSupercellFourierCoefficientsError> {
    use BinSupercellFourierCoefficientsError as E;
    if mesh.iter().any(|&n| n <= 0) {
        return Err(E::NonPositiveMesh);
    }
    let binning = Binning::new(p2s_matrix, fft_mesh, rotation)?;
    let num_bins = binning.num_bins();
    let num_coefs = (mesh[0] * mesh[1] * mesh[2]) as usize;
    if fs.len() != coefficients.len()
        || coefficients.iter().any(|c| c.len() != num_coefs)
        || fs.iter().any(|f| f.len() as i64 != num_bins)
    {
        return Err(E::LengthMismatch);
    }
    if let Some(r) = &representatives {
        if r.source.len() != r.g.len() || r.weight.len() != r.g.len() {
            return Err(E::LengthMismatch);
        }
        if r.source
            .par_iter()
            .any(|&s| s < 0 || s as usize >= num_coefs)
        {
            return Err(E::SourceOutOfRange);
        }
    }
    let rows = match representatives {
        None => Rows::Box(mesh),
        Some(r) => Rows::Given(r),
    };
    let num_rows = rows.len();
    if num_rows as u64 > 1 << 32 || num_bins as u64 > 1 << 32 {
        return Err(E::TooLarge);
    }

    // Sort the rows by (bin, row).  The keys are distinct, so the order does
    // not depend on the number of threads.
    let mut keys: Vec<u64> = (0..num_rows)
        .into_par_iter()
        .map(|j| (binning.i_f(binning.rotate(rows.g(j))) << 32) | j as u64)
        .collect();
    keys.par_sort_unstable();

    let phase = |gp: Vec3I| -> Option<Cmplx> { translation.map(|w| translation_phase(gp, w)) };

    // Split the sorted keys at the bins of the tasks.
    let (bin_bounds, parts) = split_bins(fs, num_bins as usize);
    let key_bounds: Vec<usize> = bin_bounds
        .iter()
        .map(|&b| keys.partition_point(|&k| ((k >> 32) as usize) < b))
        .collect();

    parts.into_par_iter().enumerate().for_each(|(t, mut outs)| {
        let mut sums = vec![[0.0f64; 2]; coefficients.len()];
        let mut pos = key_bounds[t];
        for (local, bin) in (bin_bounds[t]..bin_bounds[t + 1]).enumerate() {
            sums.fill([0.0, 0.0]);
            while pos < key_bounds[t + 1] && (keys[pos] >> 32) as usize == bin {
                let j = (keys[pos] & 0xffff_ffff) as usize;
                let ph = phase(binning.rotate(rows.g(j)));
                for (sum, c) in sums.iter_mut().zip(coefficients) {
                    let mut v = rows.coefficient(c, j);
                    if let Some(ph) = ph {
                        v = cmplx_mul(v, ph);
                    }
                    sum[0] += v[0];
                    sum[1] += v[1];
                }
                pos += 1;
            }
            for (out, sum) in outs.iter_mut().zip(&sums) {
                out[local] = *sum;
            }
        }
    });
    Ok(())
}

/// exp(-2 pi i g' . w^s) for the integer vector g' and the translation w^s.
fn translation_phase(gp: Vec3I, w: Vec3D) -> Cmplx {
    let two_pi = 2.0 * std::f64::consts::PI;
    let arg = (-two_pi * gp[0] as f64) * w[0]
        + (-two_pi * gp[1] as f64) * w[1]
        + (-two_pi * gp[2] as f64) * w[2];
    let (s, c) = arg.sin_cos();
    [c, s]
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

/// Whether the supercell FFT mesh N^s is invariant under the rotation W^s.
///
/// This is the integrality of N^s W^s (N^s)^-1, whose elements are
/// N_i W_ij / N_j: W^s maps the mesh point x^s = (N^s)^-1 m to the point
/// with N^s W^s (N^s)^-1 m.  It is equivalent to the integrality of
/// (N^s)^-1 (W^s)^-T N^s, the transpose of its inverse, under which the
/// classes g mod N^s in reciprocal space go to classes.  phonopy's `BZGrid`
/// also checks its grid by the integrality of a matrix, but its grid
/// D^-1 a of points in one reciprocal cell needs D r^-T D^-1, and that
/// matrix with D = N^s, r = W^s is a different condition here.
fn rotation_keeps_mesh(mesh: Vec3I, rotation: &MatI) -> bool {
    (0..3).all(|i| (0..3).all(|j| (mesh[i] * rotation[i][j]) % mesh[j] == 0))
}

/// Rows sorted by bin, the result of `sort_rows_by_bin`.
pub struct RowsSortedByBin {
    /// Rows in the order of (bin, row).
    pub order: Vec<i64>,
    /// The rows of bin b are order[bin_offsets[b]..bin_offsets[b + 1]].
    /// Length is the number of bins plus one.
    pub bin_offsets: Vec<i64>,
}

/// Sort the rows g by the bin i_F(g) of g itself (no rotation).
///
/// For the shortest representatives, the rows sorted once by this function
/// are used by `bin_sorted_supercell_fourier_coefficients` for every
/// operation that keeps the supercell FFT mesh.  The rows of a bin are in
/// increasing order, so the order does not depend on the number of
/// threads.
pub fn sort_rows_by_bin(
    g: &[Vec3I],
    p2s_matrix: MatI,
    fft_mesh: Vec3I,
) -> Result<RowsSortedByBin, BinSupercellFourierCoefficientsError> {
    let binning = Binning::new(p2s_matrix, fft_mesh, crate::common::IDENTITY)?;
    let num_bins = binning.num_bins();
    if g.len() as u64 > 1 << 32 || num_bins as u64 > 1 << 32 {
        return Err(BinSupercellFourierCoefficientsError::TooLarge);
    }
    let mut keys: Vec<u64> = g
        .par_iter()
        .enumerate()
        .map(|(j, &gj)| (binning.i_f(gj) << 32) | j as u64)
        .collect();
    keys.par_sort_unstable();
    let order: Vec<i64> = keys.par_iter().map(|&k| (k & 0xffff_ffff) as i64).collect();
    let mut bin_offsets = vec![0i64; num_bins as usize + 1];
    for &k in &keys {
        bin_offsets[(k >> 32) as usize + 1] += 1;
    }
    for b in 0..num_bins as usize {
        bin_offsets[b + 1] += bin_offsets[b];
    }
    Ok(RowsSortedByBin { order, bin_offsets })
}

/// Sum the Fourier coefficients by bin into `fs`, with the rows sorted by
/// bin once (Eq. fold-rotated-presorted of fft-instead-of-nufft-en.tex).
///
/// The rows g~' of `g_sorted` are the shortest representatives (the union
/// G_min of the sets [g]_min) sorted by `sort_rows_by_bin`, and the rows of
/// bin b are `g_sorted[bin_offsets[b]..bin_offsets[b + 1]]`.  For each row
/// g~', g~ = (W^s)^T g~' is computed with its class index
/// s = flat(g~ mod N^s), with g~_1 fastest, and
/// c[s] * class_weight[s] * exp(-2 pi i g~' . w^s) is added to its bin.
/// This equals `bin_supercell_fourier_coefficients` with the rows G_min up
/// to rounding (the terms are added in another order), when (W^s)^-T maps
/// G_min onto itself.  That holds when the mesh `mesh` is invariant under
/// `rotation`, which is checked; the possible failure of it from comparing
/// lengths with a tolerance in the search of G_min is not checked.
///
/// `coefficients[c]` and `class_weight` have the length N^s_1 N^s_2 N^s_3,
/// and `fs[c]` the number of bins `bin_offsets.len() - 1`; every element of
/// `fs[c]` is written.  `class_weight[s]` is 1 / |[g]_min| of class s.
///
/// # Errors
///
/// `NonPositiveMesh`, `NonUnimodularRotation`, `MeshNotInvariant` when the
/// mesh is not invariant under `rotation`, `LengthMismatch` for lengths of
/// `fs`, `coefficients` or `class_weight`, and `InvalidBinOffsets` when
/// `bin_offsets` does not go from 0 to `g_sorted.len()` without
/// decreasing.
#[allow(clippy::too_many_arguments)]
pub fn bin_sorted_supercell_fourier_coefficients(
    fs: &mut [&mut [Cmplx]],
    coefficients: &[&[Cmplx]],
    mesh: Vec3I,
    g_sorted: &[Vec3I],
    class_weight: &[f64],
    bin_offsets: &[i64],
    rotation: MatI,
    translation: Option<Vec3D>,
) -> Result<(), BinSupercellFourierCoefficientsError> {
    use BinSupercellFourierCoefficientsError as E;
    if mesh.iter().any(|&n| n <= 0) {
        return Err(E::NonPositiveMesh);
    }
    if det_i(&rotation).abs() != 1 {
        return Err(E::NonUnimodularRotation);
    }
    if !rotation_keeps_mesh(mesh, &rotation) {
        return Err(E::MeshNotInvariant);
    }
    let num_coefs = (mesh[0] * mesh[1] * mesh[2]) as usize;
    if fs.len() != coefficients.len()
        || coefficients.iter().any(|c| c.len() != num_coefs)
        || class_weight.len() != num_coefs
    {
        return Err(E::LengthMismatch);
    }
    if bin_offsets.first() != Some(&0)
        || bin_offsets.windows(2).any(|w| w[1] < w[0])
        || bin_offsets[bin_offsets.len() - 1] as usize != g_sorted.len()
    {
        return Err(E::InvalidBinOffsets);
    }
    let num_bins = bin_offsets.len() - 1;
    if fs.iter().any(|f| f.len() != num_bins) {
        return Err(E::LengthMismatch);
    }

    let w_t = transpose_i(&rotation);
    let class_index = |g: Vec3I| -> usize {
        let r = [0, 1, 2].map(|k| g[k].rem_euclid(mesh[k]));
        ((r[2] * mesh[1] + r[1]) * mesh[0] + r[0]) as usize
    };
    let (bin_bounds, parts) = split_bins(fs, num_bins);
    parts.into_par_iter().enumerate().for_each(|(t, mut outs)| {
        let mut sums = vec![CMPLX_ZERO; coefficients.len()];
        for (local, bin) in (bin_bounds[t]..bin_bounds[t + 1]).enumerate() {
            sums.fill(CMPLX_ZERO);
            let rows = bin_offsets[bin] as usize..bin_offsets[bin + 1] as usize;
            for &gp in &g_sorted[rows] {
                let s = class_index(matvec_ii(&w_t, gp));
                let weight = class_weight[s];
                let ph = translation.map(|w| translation_phase(gp, w));
                for (sum, c) in sums.iter_mut().zip(coefficients) {
                    let mut v = [c[s][0] * weight, c[s][1] * weight];
                    if let Some(ph) = ph {
                        v = cmplx_mul(v, ph);
                    }
                    sum[0] += v[0];
                    sum[1] += v[1];
                }
            }
            for (out, sum) in outs.iter_mut().zip(&sums) {
                out[local] = *sum;
            }
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::IDENTITY;

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

    /// Box g from the formula of the phelel docstring, built independently
    /// of `Rows::g`.
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

    /// Sequential scatter-add in the order of the rows, as np.bincount.
    fn reference(
        g: &[Vec3I],
        coef_of_row: impl Fn(usize) -> Cmplx,
        p2s: MatI,
        fft_mesh: Vec3I,
        rotation: MatI,
        translation: Option<Vec3D>,
    ) -> Vec<Cmplx> {
        let binning = Binning::new(p2s, fft_mesh, rotation).unwrap();
        let mut f = vec![[0.0; 2]; binning.num_bins() as usize];
        for (j, &gj) in g.iter().enumerate() {
            let gp = binning.rotate(gj);
            let mut v = coef_of_row(j);
            if let Some(w) = translation {
                let two_pi = 2.0 * std::f64::consts::PI;
                let arg = (-two_pi * gp[0] as f64) * w[0]
                    + (-two_pi * gp[1] as f64) * w[1]
                    + (-two_pi * gp[2] as f64) * w[2];
                let (s, cs) = arg.sin_cos();
                v = cmplx_mul(v, [cs, s]);
            }
            let i = binning.i_f(gp) as usize;
            f[i][0] += v[0];
            f[i][1] += v[1];
        }
        f
    }

    fn run(
        coefs: &[Vec<Cmplx>],
        mesh: Vec3I,
        p2s: MatI,
        fft_mesh: Vec3I,
        rotation: MatI,
        translation: Option<Vec3D>,
        representatives: Option<Representatives>,
    ) -> Vec<Vec<Cmplx>> {
        let n = num_bins(p2s, fft_mesh).unwrap();
        let mut fs: Vec<Vec<Cmplx>> = coefs.iter().map(|_| vec![[f64::NAN; 2]; n]).collect();
        let mut views: Vec<&mut [Cmplx]> = fs.iter_mut().map(|f| f.as_mut_slice()).collect();
        let cs: Vec<&[Cmplx]> = coefs.iter().map(|c| c.as_slice()).collect();
        bin_supercell_fourier_coefficients(
            &mut views,
            &cs,
            mesh,
            p2s,
            fft_mesh,
            rotation,
            translation,
            representatives,
        )
        .unwrap();
        fs
    }

    #[test]
    fn box_rows_follow_phelel_order() {
        let mesh = [3, 2, 4];
        let rows = Rows::Box(mesh);
        let g: Vec<Vec3I> = (0..rows.len()).map(|j| rows.g(j)).collect();
        assert_eq!(g, box_g(mesh));
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
        let six_fold = [[1, -1, 0], [1, 0, 0], [0, 0, 1]];
        let rotoinversion = six_fold.map(|row| row.map(|x| -x));
        for w in [three_fold(), six_fold, rotoinversion] {
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
    fn box_matches_sequential_sum() {
        let mesh = [7, 6, 5];
        let fft_mesh = [3, 4, 2];
        let g = box_g(mesh);
        let coefs = vec![coefficients(210, 1), coefficients(210, 2)];
        for (p2s, rotation, translation) in [
            (p2s_skewed(), IDENTITY, None),
            (p2s_fcc(), three_fold(), Some([0.25, 0.5, 0.125])),
        ] {
            let fs = run(&coefs, mesh, p2s, fft_mesh, rotation, translation, None);
            for (c, f) in coefs.iter().zip(&fs) {
                let r = reference(&g, |j| c[j], p2s, fft_mesh, rotation, translation);
                assert_eq!(f, &r);
            }
        }
    }

    #[test]
    fn representatives_match_sequential_sum() {
        let mesh = [4, 3, 2];
        let fft_mesh = [2, 3, 2];
        let base = box_g(mesh);
        // Two images for every third class, with weight 1/2 each.
        let mut g = Vec::new();
        let mut source = Vec::new();
        let mut weight = Vec::new();
        for (j, &gj) in base.iter().enumerate() {
            if j % 3 == 0 {
                g.push(gj);
                g.push([gj[0] - mesh[0], gj[1], gj[2] + mesh[2]]);
                source.extend([j as i64, j as i64]);
                weight.extend([0.5, 0.5]);
            } else {
                g.push(gj);
                source.push(j as i64);
                weight.push(1.0);
            }
        }
        let coefs = vec![coefficients(24, 3)];
        let translation = Some([0.1, -0.3, 0.7]);
        let reps = Representatives {
            g: &g,
            source: &source,
            weight: &weight,
        };
        let fs = run(
            &coefs,
            mesh,
            p2s_fcc(),
            fft_mesh,
            three_fold(),
            translation,
            Some(reps),
        );
        let c = &coefs[0];
        let r = reference(
            &g,
            |j| {
                let v = c[source[j] as usize];
                [v[0] * weight[j], v[1] * weight[j]]
            },
            p2s_fcc(),
            fft_mesh,
            three_fold(),
            translation,
        );
        assert_eq!(fs[0], r);
    }

    #[test]
    fn independent_of_number_of_threads() {
        let mesh = [9, 8, 7];
        let coefs = vec![coefficients(504, 4)];
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
                            p2s_fcc(),
                            [2, 2, 3],
                            three_fold(),
                            Some([0.3, 0.1, 0.2]),
                            None,
                        )
                    })
            })
            .collect();
        assert_eq!(results[0], results[1]);
        assert_eq!(results[0], results[2]);
    }

    #[test]
    fn errors() {
        use BinSupercellFourierCoefficientsError as E;
        let c = coefficients(8, 5);
        let call = |mesh: Vec3I,
                    p2s: MatI,
                    fft_mesh: Vec3I,
                    out_len: usize,
                    reps: Option<Representatives>| {
            let mut f = vec![[0.0; 2]; out_len];
            bin_supercell_fourier_coefficients(
                &mut [&mut f],
                &[&c],
                mesh,
                p2s,
                fft_mesh,
                IDENTITY,
                None,
                reps,
            )
        };
        let singular = [[1, 0, 0], [0, 1, 0], [1, 1, 0]];
        assert_eq!(
            call([2, 2, 2], singular, [1, 1, 1], 0, None),
            Err(E::SingularP2sMatrix)
        );
        assert_eq!(
            call([2, 2, 2], IDENTITY, [1, 0, 1], 0, None),
            Err(E::NonPositiveMesh)
        );
        let mut f = vec![[0.0; 2]; 1];
        assert_eq!(
            bin_supercell_fourier_coefficients(
                &mut [&mut f],
                &[&c],
                [2, 2, 2],
                IDENTITY,
                [1, 1, 1],
                [[2, 0, 0], [0, 1, 0], [0, 0, 1]],
                None,
                None,
            ),
            Err(E::NonUnimodularRotation)
        );
        assert_eq!(
            call([2, -2, 2], IDENTITY, [1, 1, 1], 1, None),
            Err(E::NonPositiveMesh)
        );
        assert_eq!(
            call([2, 2, 1], IDENTITY, [1, 1, 1], 1, None),
            Err(E::LengthMismatch)
        );
        assert_eq!(
            call([2, 2, 2], IDENTITY, [1, 1, 1], 2, None),
            Err(E::LengthMismatch)
        );
        let g = [[0, 0, 0]];
        let reps = |source: &'static [i64], weight: &'static [f64]| Representatives {
            g: &g,
            source,
            weight,
        };
        assert_eq!(
            call([2, 2, 2], IDENTITY, [1, 1, 1], 1, Some(reps(&[8], &[1.0]))),
            Err(E::SourceOutOfRange)
        );
        assert_eq!(
            call([2, 2, 2], IDENTITY, [1, 1, 1], 1, Some(reps(&[-1], &[1.0]))),
            Err(E::SourceOutOfRange)
        );
        assert_eq!(
            call([2, 2, 2], IDENTITY, [1, 1, 1], 1, Some(reps(&[0], &[]))),
            Err(E::LengthMismatch)
        );
        assert_eq!(
            call([2, 2, 2], IDENTITY, [1, 1, 1], 1, Some(reps(&[0], &[1.0]))),
            Ok(())
        );
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

    /// The shortest representatives G_min of the classes g mod N^s for the
    /// lattice A (columns), from `bz_grid_addresses` with the period
    /// lattice N^s_k b_k, and the weight 1 / |[g]_min| of each class.
    fn shortest_rows(mesh: Vec3I, lattice: [[f64; 3]; 3]) -> (Vec<Vec3I>, Vec<i64>, Vec<f64>) {
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
        let counts: Vec<i64> = bz.bz_map.windows(2).map(|w| w[1] - w[0]).collect();
        let class_weight = counts.iter().map(|&n| 1.0 / n as f64).collect();
        (bz.addresses, bz.bzg2grg, class_weight)
    }

    /// Run `bin_sorted_supercell_fourier_coefficients` on outputs filled
    /// with NaN.
    #[allow(clippy::too_many_arguments)]
    fn run_sorted(
        coefs: &[Vec<Cmplx>],
        mesh: Vec3I,
        g: &[Vec3I],
        class_weight: &[f64],
        p2s: MatI,
        fft_mesh: Vec3I,
        rotation: MatI,
        translation: Option<Vec3D>,
    ) -> Vec<Vec<Cmplx>> {
        let sorted = sort_rows_by_bin(g, p2s, fft_mesh).unwrap();
        let g_sorted: Vec<Vec3I> = sorted.order.iter().map(|&j| g[j as usize]).collect();
        let n = sorted.bin_offsets.len() - 1;
        let mut fs: Vec<Vec<Cmplx>> = coefs.iter().map(|_| vec![[f64::NAN; 2]; n]).collect();
        let mut views: Vec<&mut [Cmplx]> = fs.iter_mut().map(|f| f.as_mut_slice()).collect();
        let cs: Vec<&[Cmplx]> = coefs.iter().map(|c| c.as_slice()).collect();
        bin_sorted_supercell_fourier_coefficients(
            &mut views,
            &cs,
            mesh,
            &g_sorted,
            class_weight,
            &sorted.bin_offsets,
            rotation,
            translation,
        )
        .unwrap();
        fs
    }

    #[test]
    fn lattice_rotations_of_test_lattices() {
        assert_eq!(lattice_rotations(hexagonal_lattice()).len(), 24);
        assert_eq!(lattice_rotations(fcc_primitive_lattice()).len(), 48);
    }

    #[test]
    fn rotation_keeps_mesh_is_integrality() {
        let six_fold = [[1, -1, 0], [1, 0, 0], [0, 0, 1]];
        assert!(rotation_keeps_mesh([6, 6, 4], &six_fold));
        assert!(!rotation_keeps_mesh([6, 5, 4], &six_fold));
        let swap13 = [[0, 0, 1], [0, 1, 0], [1, 0, 0]];
        assert!(!rotation_keeps_mesh([4, 4, 6], &swap13));
        assert!(rotation_keeps_mesh([4, 4, 6], &IDENTITY));
        for w in lattice_rotations(fcc_primitive_lattice()) {
            assert!(rotation_keeps_mesh([5, 5, 5], &w));
        }
        // N W N^-1 integral for W, and (N^-1 W^-T N) integral for W^-1 = W^T
        // of an orthogonal W, checked against the elements directly.
        let mesh = [2, 4, 4];
        for w in lattice_rotations([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) {
            let integral = (0..3).all(|i| (0..3).all(|j| (mesh[i] * w[i][j]) % mesh[j] == 0));
            assert_eq!(rotation_keeps_mesh(mesh, &w), integral);
        }
    }

    #[test]
    fn sort_rows_by_bin_partitions_rows() {
        let mesh = [6, 6, 4];
        let (g, _, _) = shortest_rows(mesh, hexagonal_lattice());
        assert!(g.len() > (mesh[0] * mesh[1] * mesh[2]) as usize);
        let p2s = [[2, 0, 0], [0, 2, 0], [0, 0, 1]];
        let fft_mesh = [2, 3, 2];
        let sorted = sort_rows_by_bin(&g, p2s, fft_mesh).unwrap();
        let binning = Binning::new(p2s, fft_mesh, IDENTITY).unwrap();
        assert_eq!(sorted.bin_offsets.len() as i64, binning.num_bins() + 1);
        assert_eq!(sorted.bin_offsets[0], 0);
        assert_eq!(*sorted.bin_offsets.last().unwrap() as usize, g.len());
        let mut seen = vec![false; g.len()];
        for b in 0..sorted.bin_offsets.len() - 1 {
            let rows =
                &sorted.order[sorted.bin_offsets[b] as usize..sorted.bin_offsets[b + 1] as usize];
            assert!(rows.windows(2).all(|w| w[0] < w[1]));
            for &j in rows {
                assert_eq!(binning.i_f(g[j as usize]) as usize, b);
                assert!(!seen[j as usize]);
                seen[j as usize] = true;
            }
        }
        assert!(seen.iter().all(|&x| x));
    }

    /// Eq. fold-rotated-presorted equals Eq. fold-rotated-shortest (the
    /// present kernel with the rows G_min) for every rotation of the
    /// lattice, with a mesh invariant under all of them.
    #[test]
    fn sorted_matches_present_kernel() {
        for (lattice, mesh, p2s, fft_mesh) in [
            (
                hexagonal_lattice(),
                [6, 6, 4],
                [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
                [2, 3, 2],
            ),
            (fcc_primitive_lattice(), [5, 5, 5], p2s_fcc(), [2, 2, 3]),
        ] {
            let (g, source, class_weight) = shortest_rows(mesh, lattice);
            let weight: Vec<f64> = source.iter().map(|&s| class_weight[s as usize]).collect();
            let num_coefs = (mesh[0] * mesh[1] * mesh[2]) as usize;
            let coefs = vec![coefficients(num_coefs, 6), coefficients(num_coefs, 7)];
            for rotation in lattice_rotations(lattice) {
                for translation in [None, Some([0.1, -0.3, 0.7])] {
                    let present = run(
                        &coefs,
                        mesh,
                        p2s,
                        fft_mesh,
                        rotation,
                        translation,
                        Some(Representatives {
                            g: &g,
                            source: &source,
                            weight: &weight,
                        }),
                    );
                    let sorted = run_sorted(
                        &coefs,
                        mesh,
                        &g,
                        &class_weight,
                        p2s,
                        fft_mesh,
                        rotation,
                        translation,
                    );
                    for (f, r) in sorted.iter().zip(&present) {
                        let scale = r
                            .iter()
                            .map(|v| v[0].abs().max(v[1].abs()))
                            .fold(0.0, f64::max);
                        assert!(scale > 0.1);
                        for (a, b) in f.iter().zip(r) {
                            assert!((a[0] - b[0]).abs() < 1e-13 * scale);
                            assert!((a[1] - b[1]).abs() < 1e-13 * scale);
                        }
                    }
                }
            }
        }
    }

    /// Every element of the outputs is written, also the bins without rows.
    #[test]
    fn sorted_writes_every_bin() {
        let mesh = [4, 4, 4];
        let (g, _, class_weight) = shortest_rows(mesh, fcc_primitive_lattice());
        let p2s = p2s_fcc();
        let fft_mesh = [3, 3, 3];
        let sorted = sort_rows_by_bin(&g, p2s, fft_mesh).unwrap();
        let num_empty = sorted
            .bin_offsets
            .windows(2)
            .filter(|w| w[0] == w[1])
            .count();
        assert!(num_empty > 0);
        let coefs = vec![coefficients(64, 8)];
        let fs = run_sorted(
            &coefs,
            mesh,
            &g,
            &class_weight,
            p2s,
            fft_mesh,
            three_fold(),
            Some([0.25, 0.5, 0.125]),
        );
        assert!(fs[0].iter().all(|v| v[0].is_finite() && v[1].is_finite()));
        for (b, w) in sorted.bin_offsets.windows(2).enumerate() {
            if w[0] == w[1] {
                assert_eq!(fs[0][b], [0.0, 0.0]);
            }
        }
    }

    #[test]
    fn sorted_independent_of_number_of_threads() {
        let mesh = [6, 6, 4];
        let (g, _, class_weight) = shortest_rows(mesh, hexagonal_lattice());
        let coefs = vec![coefficients(144, 9)];
        let six_fold = [[1, -1, 0], [1, 0, 0], [0, 0, 1]];
        let results: Vec<Vec<Vec<Cmplx>>> = [1, 3, 8]
            .iter()
            .map(|&n| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build()
                    .unwrap()
                    .install(|| {
                        run_sorted(
                            &coefs,
                            mesh,
                            &g,
                            &class_weight,
                            [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
                            [2, 3, 2],
                            six_fold,
                            Some([0.3, 0.1, 0.2]),
                        )
                    })
            })
            .collect();
        assert_eq!(results[0], results[1]);
        assert_eq!(results[0], results[2]);
    }

    #[test]
    fn sorted_errors() {
        use BinSupercellFourierCoefficientsError as E;
        let mesh = [4, 4, 6];
        let c = coefficients(96, 10);
        let weight = vec![1.0; 96];
        let g = [[0, 0, 0], [1, 0, 0]];
        let call = |mesh: Vec3I,
                    class_weight: &[f64],
                    bin_offsets: &[i64],
                    out_len: usize,
                    rotation: MatI| {
            let mut f = vec![[0.0; 2]; out_len];
            bin_sorted_supercell_fourier_coefficients(
                &mut [&mut f],
                &[&c],
                mesh,
                &g,
                class_weight,
                bin_offsets,
                rotation,
                None,
            )
        };
        assert_eq!(call(mesh, &weight, &[0, 1, 2], 2, IDENTITY), Ok(()));
        let swap13 = [[0, 0, 1], [0, 1, 0], [1, 0, 0]];
        assert_eq!(
            call(mesh, &weight, &[0, 1, 2], 2, swap13),
            Err(E::MeshNotInvariant)
        );
        assert_eq!(
            call(
                mesh,
                &weight,
                &[0, 1, 2],
                2,
                [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
            ),
            Err(E::NonUnimodularRotation)
        );
        assert_eq!(
            call([4, 0, 6], &weight, &[0, 1, 2], 2, IDENTITY),
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
                Err(E::InvalidBinOffsets)
            );
        }
    }
}
