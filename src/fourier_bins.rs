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

use crate::common::{cmplx_mul, det_i, matvec_ii, transpose_i, Cmplx, MatI, Vec3D, Vec3I};
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

    let phase = |gp: Vec3I| -> Option<Cmplx> {
        translation.map(|w| {
            let two_pi = 2.0 * std::f64::consts::PI;
            let arg = (-two_pi * gp[0] as f64) * w[0]
                + (-two_pi * gp[1] as f64) * w[1]
                + (-two_pi * gp[2] as f64) * w[2];
            let (s, c) = arg.sin_cos();
            [c, s]
        })
    };

    // Split the bins into ranges and the sorted keys at the same bins, so
    // that each task writes its own part of every output.
    let num_bins = num_bins as usize;
    let num_tasks = (rayon::current_num_threads() * 8).min(num_bins).max(1);
    let bin_bounds: Vec<usize> = (0..=num_tasks).map(|t| t * num_bins / num_tasks).collect();
    let key_bounds: Vec<usize> = bin_bounds
        .iter()
        .map(|&b| keys.partition_point(|&k| ((k >> 32) as usize) < b))
        .collect();
    let mut parts: Vec<Vec<&mut [Cmplx]>> = (0..num_tasks).map(|_| Vec::new()).collect();
    for f in fs.iter_mut() {
        let mut rest: &mut [Cmplx] = f;
        for t in 0..num_tasks {
            let (part, tail) = rest.split_at_mut(bin_bounds[t + 1] - bin_bounds[t]);
            parts[t].push(part);
            rest = tail;
        }
    }

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
}
