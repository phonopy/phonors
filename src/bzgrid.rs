//! Brillouin-zone grid address construction.
//!
//! Port of `bzg_get_bz_grid_addresses` and friends in `c/bzgrid.c`.

use rayon::prelude::*;

use crate::common::{det_d, matvec_dd, matvec_id, matvec_ii, nint, MatD, MatI, Vec3D, Vec3I};
use crate::grgrid::{double_grid_address, grid_address_from_index, grid_index_from_address};

const GRID_TOLERANCE_FACTOR: f64 = 0.01;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BzGridAddressesError {
    /// `Q` is not unimodular (|det(Q)| != 1).
    NotUnimodularQ,
    /// Unsupported `bz_grid_type` (must be 1 or 2).
    BadGridType,
    /// The reciprocal basis vectors are linearly dependent.
    SingularReciprocalLattice,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotateBzGridError {
    /// Unsupported `bz_grid_type` (must be 1 or 2).
    BadGridType,
}

/// Output of `bz_grid_addresses`.
pub struct BzGridAddresses {
    /// Addresses in BZ layout, length = `num_gp`.
    pub addresses: Vec<Vec3I>,
    /// Map from GR-grid mesh (type 2) or 2x mesh (type 1) to BZ index.
    pub bz_map: Vec<i64>,
    /// Map from BZ index back to GR grid index.
    pub bzg2grg: Vec<i64>,
}

fn inverse_unimodular(m: &MatI) -> Option<MatI> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() != 1 {
        return None;
    }
    let c = [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det,
            (m[2][1] * m[0][2] - m[2][2] * m[0][1]) / det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det,
            (m[2][2] * m[0][0] - m[2][0] * m[0][2]) / det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det,
            (m[2][0] * m[0][1] - m[2][1] * m[0][0]) / det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det,
        ],
    ];
    Some(c)
}

/// Tolerance for BZ reduction, in units of squared length.
pub(crate) fn bz_tolerance(rec_lattice: &MatD, d_diag: Vec3I) -> f64 {
    let mut max_len = 0.0f64;
    for i in 0..3 {
        let mut len = 0.0;
        for j in 0..3 {
            len += rec_lattice[j][i] * rec_lattice[j][i];
        }
        len /= (d_diag[i] * d_diag[i]) as f64;
        if len > max_len {
            max_len = len;
        }
    }
    max_len * GRID_TOLERANCE_FACTOR
}

/// Margin against rounding added to the range of each coefficient
/// `m_i` of the search.
const BZ_SEARCH_MARGIN: f64 = 1e-8;

/// Bound of `|m_i|` of the candidates that the search visits first.
/// `bz_search_space` in `c/bzgrid.c` searches only the 125 candidates
/// with `|m_i| <= 2`.
const BZ_SEARCH_FIRST: i64 = 2;

/// Integers from `lo` to `hi` in the order of the frequencies of an
/// FFT: the non-negative ones in increasing order, and then the negative
/// ones in increasing order, e.g., 0, 1, 2, -2, -1.
fn fft_order(lo: i64, hi: i64) -> impl Iterator<Item = i64> + Clone {
    (lo.max(0)..=hi).chain(lo..=hi.min(-1))
}

/// Call `f` with each candidate `m` with `lo[i] <= m_i <= hi[i]`.
///
/// The candidates with all `|m_i| <= BZ_SEARCH_FIRST` come first, in
/// lexicographic order of `(m_0, m_1, m_2)` with each `m_i` in the order
/// of `fft_order`.  This is the order of `bz_search_space` in
/// `c/bzgrid.c`, so the order of the images is that of the C code
/// whenever they are all among these candidates.  The candidates with
/// some `|m_i| > BZ_SEARCH_FIRST` follow, in lexicographic order of
/// `(m_0, m_1, m_2)` with each `m_i` in increasing order.
fn for_each_candidate(lo: Vec3I, hi: Vec3I, mut f: impl FnMut(Vec3I)) {
    let first_lo = lo.map(|x| x.max(-BZ_SEARCH_FIRST));
    let first_hi = hi.map(|x| x.min(BZ_SEARCH_FIRST));
    for m0 in fft_order(first_lo[0], first_hi[0]) {
        for m1 in fft_order(first_lo[1], first_hi[1]) {
            for m2 in fft_order(first_lo[2], first_hi[2]) {
                f([m0, m1, m2]);
            }
        }
    }
    if lo == first_lo && hi == first_hi {
        return;
    }
    for m0 in lo[0]..=hi[0] {
        for m1 in lo[1]..=hi[1] {
            for m2 in lo[2]..=hi[2] {
                let m = [m0, m1, m2];
                if m.iter().any(|x| x.abs() > BZ_SEARCH_FIRST) {
                    f(m);
                }
            }
        }
    }
}

/// Lengths of the dual vectors of the columns of `rec_lattice`, i.e.,
/// of the rows of its inverse.  `None` if `rec_lattice` is singular.
fn dual_lengths(rec_lattice: &MatD) -> Option<Vec3D> {
    let col = |j: usize| [rec_lattice[0][j], rec_lattice[1][j], rec_lattice[2][j]];
    let cross = |a: Vec3D, b: Vec3D| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let norm = |a: Vec3D| (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    let det = det_d(rec_lattice);
    if det == 0.0 || !det.is_finite() {
        return None;
    }
    Some([
        norm(cross(col(1), col(2))) / det.abs(),
        norm(cross(col(2), col(0))) / det.abs(),
        norm(cross(col(0), col(1))) / det.abs(),
    ])
}

fn set_bz_address(
    m: Vec3I,
    grid_address: Vec3I,
    d_diag: Vec3I,
    nint_v: Vec3I,
    q_inv: &MatI,
) -> Vec3I {
    let delta_g = [m[0] - nint_v[0], m[1] - nint_v[1], m[2] - nint_v[2]];
    let delta_g = matvec_ii(q_inv, delta_g);
    [
        grid_address[0] + delta_g[0] * d_diag[0],
        grid_address[1] + delta_g[1] * d_diag[1],
        grid_address[2] + delta_g[2] * d_diag[2],
    ]
}

/// Number of GR-grid points handled by one rayon task when filling
/// the BZ grid.
const BZ_GRID_BLOCK_SIZE: usize = 1024;

/// Parameters of the search for the BZ images of GR-grid points.
struct BzSearch {
    d_diag: Vec3I,
    ps: Vec3I,
    q: MatI,
    q_inv: MatI,
    rec_lattice: MatD,
    tolerance: f64,
    /// Lengths of the dual vectors of the columns of `rec_lattice`.
    dual_lengths: Vec3D,
}

impl BzSearch {
    /// Split the reduced coordinates `q_red` of GR-grid point
    /// `grid_address` in the basis `rec_lattice` into
    /// `q_red = s0 + nint_v` with `|s0_i| <= 1/2` and an integer vector
    /// `nint_v`.  Returns `(s0, nint_v)`.
    fn split_reduced_coordinates(&self, grid_address: Vec3I) -> (Vec3D, Vec3I) {
        let d_diag = self.d_diag;
        let dadrs = double_grid_address(grid_address, self.ps);
        let mut q_red = [
            dadrs[0] as f64 / (2.0 * d_diag[0] as f64),
            dadrs[1] as f64 / (2.0 * d_diag[1] as f64),
            dadrs[2] as f64 / (2.0 * d_diag[2] as f64),
        ];
        q_red = matvec_id(&self.q, q_red);
        let mut nint_v = [0i64; 3];
        for i in 0..3 {
            nint_v[i] = nint(q_red[i]);
            q_red[i] -= nint_v[i] as f64;
        }
        (q_red, nint_v)
    }

    /// Squared length of `rec_lattice (s0 + m)`.
    fn distance(&self, s0: Vec3D, m: Vec3I) -> f64 {
        let v = [
            s0[0] + m[0] as f64,
            s0[1] + m[1] as f64,
            s0[2] + m[2] as f64,
        ];
        let vc = matvec_dd(&self.rec_lattice, v);
        vc[0] * vc[0] + vc[1] * vc[1] + vc[2] * vc[2]
    }

    /// Range `lo[i] <= m_i <= hi[i]` that contains every candidate that
    /// can lower the minimum or be kept in `for_each_image`.
    ///
    /// The minimum starts at `|y0|^2` with `y0 = rec_lattice s0` and only
    /// decreases, so such a candidate `x = rec_lattice (s0 + m)` has
    /// `|x|^2 < |y0|^2 + tolerance = R^2`.  With the dual vectors `w*_i`
    /// (rows of `rec_lattice^-1`), `s0_i + m_i = w*_i . x`, so
    /// `|s0_i + m_i| <= R |w*_i|`.  This holds for any basis; a reduced
    /// basis only makes the range smaller.
    fn search_range(&self, s0: Vec3D) -> (Vec3I, Vec3I) {
        let r = (self.distance(s0, [0, 0, 0]) + self.tolerance).sqrt();
        let mut lo = [0i64; 3];
        let mut hi = [0i64; 3];
        for i in 0..3 {
            let half_width = r * self.dual_lengths[i] + BZ_SEARCH_MARGIN;
            lo[i] = (-s0[i] - half_width).ceil() as i64;
            hi[i] = (-s0[i] + half_width).floor() as i64;
        }
        (lo, hi)
    }

    /// Call `f` with the BZ address of each image of GR-grid point
    /// `gr_index`, in the order of `for_each_candidate`.
    ///
    /// The minimum is lowered only by a candidate shorter than it by more
    /// than `tolerance`, so that the first of nearly equal candidates
    /// sets it, and the candidates within `tolerance` of it are kept.
    fn for_each_image(&self, gr_index: i64, mut f: impl FnMut(Vec3I)) {
        let gr_adrs = grid_address_from_index(gr_index, self.d_diag);
        let (s0, nint_v) = self.split_reduced_coordinates(gr_adrs);
        let (lo, hi) = self.search_range(s0);
        let mut min_distance = self.distance(s0, [0, 0, 0]);
        for_each_candidate(lo, hi, |m| {
            let distance = self.distance(s0, m);
            if distance < min_distance - self.tolerance {
                min_distance = distance;
            }
        });
        for_each_candidate(lo, hi, |m| {
            if self.distance(s0, m) < min_distance + self.tolerance {
                f(set_bz_address(m, gr_adrs, self.d_diag, nint_v, &self.q_inv));
            }
        });
    }

    fn num_images(&self, gr_index: i64) -> i64 {
        let mut count = 0;
        self.for_each_image(gr_index, |_| count += 1);
        count
    }
}

/// Build BZ grid addresses, map, and GR mapping.
///
/// `bz_grid_type` mirrors the C API: `1` = sparse layout, `2` = dense layout.
///
/// The images of each GR-grid point are counted in parallel, the
/// counts are turned into offsets by a prefix sum, and the images are
/// then written in parallel at those offsets.  The result does not
/// depend on the number of threads.
///
/// The search finds all images for any basis `rec_lattice`, but the
/// tolerance of the ties, `bz_tolerance`, is computed from the lengths
/// of its columns.  Pass a reduced basis, such as the Niggli-reduced
/// basis that phonopy passes, whose longest column is as short as
/// possible.  With a strongly skewed basis, the tolerance can be larger
/// than the difference between the squared lengths of two images, and
/// then images that are not the shortest are also kept.
pub fn bz_grid_addresses(
    d_diag: Vec3I,
    q: MatI,
    ps: Vec3I,
    rec_lattice: MatD,
    bz_grid_type: i64,
) -> Result<BzGridAddresses, BzGridAddressesError> {
    debug_assert!(d_diag.iter().all(|&d| d >= 1));
    if bz_grid_type != 1 && bz_grid_type != 2 {
        return Err(BzGridAddressesError::BadGridType);
    }
    let q_inv = inverse_unimodular(&q).ok_or(BzGridAddressesError::NotUnimodularQ)?;
    let dual_lengths =
        dual_lengths(&rec_lattice).ok_or(BzGridAddressesError::SingularReciprocalLattice)?;
    let search = BzSearch {
        d_diag,
        ps,
        q,
        q_inv,
        rec_lattice,
        tolerance: bz_tolerance(&rec_lattice, d_diag),
        dual_lengths,
    };
    if bz_grid_type == 2 {
        Ok(bz_grid_addresses_dense(&search))
    } else {
        Ok(bz_grid_addresses_sparse(&search))
    }
}

/// Ranges `start..end` of GR-grid points, each handled by one rayon
/// task.
fn bz_grid_blocks(total_num_gp: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..total_num_gp)
        .step_by(BZ_GRID_BLOCK_SIZE)
        .map(move |start| (start, (start + BZ_GRID_BLOCK_SIZE).min(total_num_gp)))
}

/// Dense layout: addresses appear in order, gp_map holds the starting
/// BZ index for each GR-grid point (length = total+1).
fn bz_grid_addresses_dense(search: &BzSearch) -> BzGridAddresses {
    let d_diag = search.d_diag;
    let total_num_gp = (d_diag[0] * d_diag[1] * d_diag[2]) as usize;

    let mut bz_map = vec![0i64; total_num_gp + 1];
    bz_map[1..]
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, c)| *c = search.num_images(i as i64));
    for i in 0..total_num_gp {
        bz_map[i + 1] += bz_map[i];
    }

    let size = bz_map[total_num_gp] as usize;
    let mut addresses: Vec<Vec3I> = vec![[0; 3]; size];
    let mut bzg2grg: Vec<i64> = vec![0; size];

    // Split the outputs into the disjoint ranges of the blocks.
    let mut tasks = Vec::with_capacity(total_num_gp.div_ceil(BZ_GRID_BLOCK_SIZE));
    let mut adrs_rest = &mut addresses[..];
    let mut grg_rest = &mut bzg2grg[..];
    for (start, end) in bz_grid_blocks(total_num_gp) {
        let len = (bz_map[end] - bz_map[start]) as usize;
        let (adrs, rest) = std::mem::take(&mut adrs_rest).split_at_mut(len);
        adrs_rest = rest;
        let (grg, rest) = std::mem::take(&mut grg_rest).split_at_mut(len);
        grg_rest = rest;
        tasks.push((start, end, adrs, grg));
    }

    tasks.into_par_iter().for_each(|(start, end, adrs, grg)| {
        let mut k = 0;
        for i in start..end {
            search.for_each_image(i as i64, |a| {
                adrs[k] = a;
                grg[k] = i as i64;
                k += 1;
            });
        }
    });
    BzGridAddresses {
        addresses,
        bz_map,
        bzg2grg,
    }
}

/// Sparse layout, converted from the dense layout: the first image of
/// GR-grid point `i` is stored at `i`, and the other images are
/// appended after the `total_num_gp` first images in the order of the
/// grid points.
fn bz_grid_addresses_sparse(search: &BzSearch) -> BzGridAddresses {
    let dense = bz_grid_addresses_dense(search);
    let d_diag = search.d_diag;
    let total_num_gp = (d_diag[0] * d_diag[1] * d_diag[2]) as usize;
    let bzmesh: Vec3I = [d_diag[0] * 2, d_diag[1] * 2, d_diag[2] * 2];
    let num_bzmesh = (bzmesh[0] * bzmesh[1] * bzmesh[2]) as usize;
    let images = |i: usize| dense.bz_map[i] as usize..dense.bz_map[i + 1] as usize;

    let mut addresses = Vec::with_capacity(dense.addresses.len());
    let mut bzg2grg = Vec::with_capacity(dense.addresses.len());
    for i in 0..total_num_gp {
        addresses.push(dense.addresses[images(i).start]);
        bzg2grg.push(i as i64);
    }
    for i in 0..total_num_gp {
        for j in images(i).skip(1) {
            addresses.push(dense.addresses[j]);
            bzg2grg.push(i as i64);
        }
    }

    // bz_map[bzgp] is the index of the image at `bzgp` on the 2x mesh.
    // Equivalent to C's grg_get_double_grid_index on the 2x mesh:
    // `(adrs*2 + ps - ps)/2` collapses to `adrs`, after which
    // `grid_index_from_address` applies the Euclidean modulo against
    // `bzmesh`.  Images of different GR-grid points never share `bzgp`,
    // and the images of one point are visited in the serial order, so
    // the last one wins on a collision as in the serial code.
    let mut bz_map = vec![num_bzmesh as i64; num_bzmesh + total_num_gp + 1];
    for (gp, &a) in addresses.iter().enumerate() {
        bz_map[grid_index_from_address(a, bzmesh) as usize] = gp as i64;
    }
    // bz_map[num_bzmesh + i] is the number of extra images of the grid
    // points before `i`.
    for i in 0..=total_num_gp {
        bz_map[num_bzmesh + i] = dense.bz_map[i] - i as i64;
    }
    BzGridAddresses {
        addresses,
        bz_map,
        bzg2grg,
    }
}

/// Rotate a BZ-grid point and return the rotated point's BZ index.
///
/// Port of `bzg_rotate_grid_index` in `c/bzgrid.c`.  Searches the
/// translationally-equivalent BZ images listed in `bz_map` for the
/// one whose address exactly matches the rotated address.  Falls
/// back to the first image (the C code's "ill-defined bzgrid"
/// branch) when no match is found.
pub fn rotate_bz_grid_index(
    bz_grid_index: i64,
    rotation: MatI,
    bz_grid_addresses: &[Vec3I],
    bz_map: &[i64],
    d_diag: Vec3I,
    ps: Vec3I,
    bz_grid_type: i64,
) -> Result<i64, RotateBzGridError> {
    if bz_grid_type != 1 && bz_grid_type != 2 {
        return Err(RotateBzGridError::BadGridType);
    }

    let start_adrs = bz_grid_addresses[bz_grid_index as usize];
    let dadrs = double_grid_address(start_adrs, ps);
    let dadrs_rot = matvec_ii(&rotation, dadrs);
    // `(dadrs_rot - PS) / 2` uses truncation toward zero, matching C.
    let adrs_rot = [
        (dadrs_rot[0] - ps[0]) / 2,
        (dadrs_rot[1] - ps[1]) / 2,
        (dadrs_rot[2] - ps[2]) / 2,
    ];
    let gp = grid_index_from_address(adrs_rot, d_diag);

    if bz_grid_type == 1 {
        if bz_grid_addresses[gp as usize] == adrs_rot {
            return Ok(gp);
        }
        let num_grgp = d_diag[0] * d_diag[1] * d_diag[2];
        let num_bzgp = num_grgp * 8;
        let start = bz_map[(num_bzgp + gp) as usize] + num_grgp;
        let end = bz_map[(num_bzgp + gp + 1) as usize] + num_grgp;
        for i in start..end {
            if bz_grid_addresses[i as usize] == adrs_rot {
                return Ok(i);
            }
        }
        // Ill-defined bzgrid fallback.
        Ok(bz_map[gp as usize])
    } else {
        let start = bz_map[gp as usize];
        let end = bz_map[(gp + 1) as usize];
        for i in start..end {
            if bz_grid_addresses[i as usize] == adrs_rot {
                return Ok(i);
            }
        }
        Ok(bz_map[gp as usize])
    }
}

/// Borrowed view onto a BZ grid: matches the fields read by C's
/// `RecgridConstBZGrid` from `tpi_*` callsites.
pub struct BzGridView<'a> {
    pub d_diag: Vec3I,
    pub addresses: &'a [Vec3I],
    pub gp_map: &'a [i64],
    /// 1 = sparse (gp_map indexes 2x mesh), 2 = dense (gp_map[g]..gp_map[g+1]).
    pub bz_grid_type: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BzGridError {
    /// `bz_grid_type` not 1 or 2.
    BadGridType,
    /// `tp_type` not 2, 3, or 4.
    BadTpType,
}

/// Public: BZ-grid neighbours of many `grid_points` along a list of
/// relative grid addresses.  Mirrors `ph3py_get_neighboring_gird_points`.  `out`
/// is `num_grid_points * relative_grid_address.len()` long in row-major
/// order: chunk `i` holds the neighbours of `grid_points[i]`.
///
/// Caller must ensure `out.len() == grid_points.len() * relative_grid_address.len()`.
pub fn neighboring_grid_points_many(
    out: &mut [i64],
    grid_points: &[i64],
    relative_grid_address: &[Vec3I],
    bzgrid: &BzGridView,
) -> Result<(), BzGridError> {
    let num_rga = relative_grid_address.len();
    out.par_chunks_mut(num_rga)
        .zip(grid_points.par_iter())
        .try_for_each(|(chunk, &gp)| {
            fill_neighboring_grid_points(chunk, gp, relative_grid_address, bzgrid)
        })
}

pub(crate) fn fill_neighboring_grid_points(
    out: &mut [i64],
    grid_point: i64,
    relative_grid_address: &[Vec3I],
    bzgrid: &BzGridView,
) -> Result<(), BzGridError> {
    match bzgrid.bz_grid_type {
        1 => fill_neighboring_grid_points_type1(out, grid_point, relative_grid_address, bzgrid),
        2 => fill_neighboring_grid_points_type2(out, grid_point, relative_grid_address, bzgrid),
        _ => Err(BzGridError::BadGridType),
    }
}

fn fill_neighboring_grid_points_type1(
    out: &mut [i64],
    grid_point: i64,
    relative_grid_address: &[Vec3I],
    bzgrid: &BzGridView,
) -> Result<(), BzGridError> {
    let bzmesh: Vec3I = [
        bzgrid.d_diag[0] * 2,
        bzgrid.d_diag[1] * 2,
        bzgrid.d_diag[2] * 2,
    ];
    let prod_bz_mesh = bzmesh[0] * bzmesh[1] * bzmesh[2];
    let base = bzgrid.addresses[grid_point as usize];

    for (i, rel) in relative_grid_address.iter().enumerate() {
        let bz_address: Vec3I = [base[0] + rel[0], base[1] + rel[1], base[2] + rel[2]];
        let bz_gp = bzgrid.gp_map[grid_index_from_address(bz_address, bzmesh) as usize];
        if bz_gp == prod_bz_mesh {
            out[i] = grid_index_from_address(bz_address, bzgrid.d_diag);
        } else {
            out[i] = bz_gp;
        }
    }
    Ok(())
}

fn fill_neighboring_grid_points_type2(
    out: &mut [i64],
    grid_point: i64,
    relative_grid_address: &[Vec3I],
    bzgrid: &BzGridView,
) -> Result<(), BzGridError> {
    let base = bzgrid.addresses[grid_point as usize];
    for (i, rel) in relative_grid_address.iter().enumerate() {
        let bz_address: Vec3I = [base[0] + rel[0], base[1] + rel[1], base[2] + rel[2]];
        let gp = grid_index_from_address(bz_address, bzgrid.d_diag);
        let lo = bzgrid.gp_map[gp as usize];
        let hi = bzgrid.gp_map[(gp + 1) as usize];
        out[i] = lo;
        if hi - lo > 1 {
            for j in lo..hi {
                let a = bzgrid.addresses[j as usize];
                if a == bz_address {
                    out[i] = j;
                    break;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{matmul_d, transpose_i};

    const IDENTITY: MatI = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    const ZERO: Vec3I = [0, 0, 0];

    /// Unimodular U that skews a basis B into B U.  Some images need
    /// |m_i| > 2 in the basis B U, so the 125 candidates of the C code are
    /// not enough.  The tolerance computed from B U stays below a quarter
    /// of the differences of the squared lengths for the lattices and
    /// meshes of the tests.  A stronger skew makes the tolerance, which
    /// grows with the lengths of the basis vectors, larger than these
    /// differences, and then the images depend on the basis.
    fn skew() -> MatI {
        [[1, 3, 0], [0, 1, 2], [0, 0, 1]]
    }

    /// Rotation by `angle` about the unit vector along `axis`.
    fn rotation(axis: Vec3D, angle: f64) -> MatD {
        let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        let [x, y, z] = [axis[0] / n, axis[1] / n, axis[2] / n];
        let (s, c) = angle.sin_cos();
        let t = 1.0 - c;
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    }

    fn scaled(m: &MatD, factor: f64) -> MatD {
        m.map(|row| row.map(|x| x * factor))
    }

    fn cubic() -> MatD {
        scaled(&mat_i_to_d(&IDENTITY), 1.0 / 10.7)
    }

    /// Reciprocal basis (columns) of the hexagonal lattice with a = 3
    /// and c = 4.8.
    fn hexagonal() -> MatD {
        let r = 1.0 / (3.0 * 3.0f64.sqrt());
        [
            [1.0 / 3.0, 0.0, 0.0],
            [r, 2.0 * r, 0.0],
            [0.0, 0.0, 1.0 / 4.8],
        ]
    }

    /// Reciprocal basis (columns) of the FCC lattice with a = 4.05.
    fn fcc() -> MatD {
        scaled(
            &[[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0]],
            1.0 / 4.05,
        )
    }

    fn mat_i_to_d(m: &MatI) -> MatD {
        m.map(|row| row.map(|x| x as f64))
    }

    fn dense(d_diag: Vec3I, q: MatI, rec_lattice: MatD) -> BzGridAddresses {
        bz_grid_addresses(d_diag, q, ZERO, rec_lattice, 2).unwrap()
    }

    fn images(bzg: &BzGridAddresses, gp: usize) -> &[Vec3I] {
        &bzg.addresses[bzg.bz_map[gp] as usize..bzg.bz_map[gp + 1] as usize]
    }

    fn num_gp(d_diag: Vec3I) -> usize {
        (d_diag[0] * d_diag[1] * d_diag[2]) as usize
    }

    /// Cartesian coordinates of the point at `address` with PS = 0.
    fn cartesian(rec_lattice: &MatD, q: &MatI, d_diag: Vec3I, address: Vec3I) -> Vec3D {
        let s = [
            address[0] as f64 / d_diag[0] as f64,
            address[1] as f64 / d_diag[1] as f64,
            address[2] as f64 / d_diag[2] as f64,
        ];
        matvec_dd(rec_lattice, matvec_id(q, s))
    }

    /// With the obtuse superbase of the path type w_1 = (-1, 1, 0),
    /// w_2 = (1, 0, -1), w_3 = (0, 0, 1) of the cubic lattice, the point
    /// (w_1 + w_3) / 2 is the center of a cube and its eight images
    /// (+-1/2, +-1/2, +-1/2) are ties.  One of them needs the coefficient
    /// 2 of w_3.  The basis is rotated so that the ties are equal only up
    /// to rounding.
    #[test]
    fn path_superbase_ties() {
        let path: MatI = [[-1, 1, 0], [1, 0, -1], [0, 0, 1]];
        let path = transpose_i(&path);
        let rec = matmul_d(
            &rotation([1.0, 2.0, 3.0], 0.7),
            &scaled(&mat_i_to_d(&path), 0.37),
        );
        let d_diag = [4, 4, 4];
        let bzg = dense(d_diag, IDENTITY, rec);
        let gp = grid_index_from_address([2, 0, 2], d_diag) as usize;
        // In the cubic lattice, 4 x = path a, and x = (+-1/2, +-1/2, +-1/2).
        let mut corners: Vec<Vec3I> = images(&bzg, gp)
            .iter()
            .map(|&a| matvec_ii(&path, a))
            .collect();
        corners.sort();
        let mut expected: Vec<Vec3I> = Vec::new();
        for x in [-2, 2] {
            for y in [-2, 2] {
                for z in [-2, 2] {
                    expected.push([x, y, z]);
                }
            }
        }
        assert_eq!(corners, expected);
    }

    /// The images are the same when the basis is replaced by a skewed one,
    /// rec_lattice U with Q = U^-1, which describes the same grid.
    #[test]
    fn images_do_not_depend_on_basis() {
        let u = skew();
        let u_inv = inverse_unimodular(&u).unwrap();
        for (rec, d_diag) in [
            (cubic(), [6, 6, 6]),
            (hexagonal(), [6, 6, 4]),
            (fcc(), [5, 6, 4]),
        ] {
            let bzg = dense(d_diag, IDENTITY, rec);
            let bzg_skewed = dense(d_diag, u_inv, matmul_d(&rec, &mat_i_to_d(&u)));
            let mut num_ties = 0;
            for gp in 0..num_gp(d_diag) {
                let mut expected = images(&bzg, gp).to_vec();
                let mut found = images(&bzg_skewed, gp).to_vec();
                expected.sort();
                found.sort();
                assert_eq!(found, expected, "grid point {gp}");
                num_ties += (expected.len() > 1) as usize;
            }
            assert!(num_ties > 0);
        }
    }

    /// The images for a skewed basis equal those of a search over all
    /// |m_i| <= 8.
    #[test]
    fn images_equal_brute_force_for_skewed_basis() {
        const M: i64 = 8;
        let u = skew();
        let rec = matmul_d(&cubic(), &mat_i_to_d(&u));
        let d_diag = [4, 5, 6];
        let tolerance = bz_tolerance(&rec, d_diag);
        let bzg = dense(d_diag, IDENTITY, rec);
        let mut max_m = 0;
        for gp in 0..num_gp(d_diag) {
            let address = grid_address_from_index(gp as i64, d_diag);
            let s: Vec3D = [0, 1, 2].map(|i| address[i] as f64 / d_diag[i] as f64);
            let mut candidates = Vec::new();
            for m0 in -M..=M {
                for m1 in -M..=M {
                    for m2 in -M..=M {
                        let x =
                            matvec_dd(&rec, [s[0] + m0 as f64, s[1] + m1 as f64, s[2] + m2 as f64]);
                        candidates.push(([m0, m1, m2], x, x[0] * x[0] + x[1] * x[1] + x[2] * x[2]));
                    }
                }
            }
            let min = candidates.iter().map(|c| c.2).fold(f64::INFINITY, f64::min);
            let expected: Vec<_> = candidates
                .iter()
                .filter(|c| c.2 < min + tolerance)
                .collect();
            for c in &expected {
                max_m = max_m.max(c.0.iter().map(|x| x.abs()).max().unwrap());
            }
            let found = images(&bzg, gp);
            assert_eq!(found.len(), expected.len(), "grid point {gp}");
            for &a in found {
                let x = cartesian(&rec, &IDENTITY, d_diag, a);
                assert!(
                    expected
                        .iter()
                        .any(|c| (0..3).all(|i| (c.1[i] - x[i]).abs() < 1e-9)),
                    "grid point {gp}"
                );
            }
        }
        // The images are inside the searched box, and some are outside
        // |m_i| <= 2.
        assert!(max_m < M);
        assert!(max_m > 2);
    }

    /// The candidates with all |m_i| <= 2 are visited in the order of
    /// `bz_search_space` in `c/bzgrid.c`.  The expected entries are taken
    /// from that array at the given positions.
    #[test]
    fn first_candidates_follow_c_order() {
        let mut visited = Vec::new();
        for_each_candidate([-2, -2, -2], [2, 2, 2], |m| visited.push(m));
        assert_eq!(visited.len(), 125);
        for (i, m) in [
            (0, [0, 0, 0]),
            (1, [0, 0, 1]),
            (2, [0, 0, 2]),
            (3, [0, 0, -2]),
            (4, [0, 0, -1]),
            (5, [0, 1, 0]),
            (6, [0, 1, 1]),
            (24, [0, -1, -1]),
            (25, [1, 0, 0]),
            (49, [1, -1, -1]),
            (50, [2, 0, 0]),
            (62, [2, 2, 2]),
            (63, [2, 2, -2]),
            (75, [-2, 0, 0]),
            (100, [-1, 0, 0]),
            (124, [-1, -1, -1]),
        ] {
            assert_eq!(visited[i], m, "position {i}");
        }
        // With a larger range, the same 125 come first, and the others
        // follow.
        let mut visited_large = Vec::new();
        for_each_candidate([-3, -2, -4], [2, 3, 2], |m| visited_large.push(m));
        assert_eq!(visited_large.len(), 6 * 6 * 7);
        assert_eq!(visited_large[..125], visited[..]);
        assert!(visited_large[125..]
            .iter()
            .all(|m| m.iter().any(|x| x.abs() > 2)));
    }

    /// The images and their order agree with the C code for bases where
    /// its 125 candidates are enough.
    #[test]
    fn order_of_images_is_kept() {
        let cases: [(MatD, Vec3I, Vec<(Vec3I, Vec<Vec3I>)>); 2] = [
            (
                fcc(),
                [4, 4, 4],
                vec![
                    ([2, 0, 0], vec![[-2, 0, 0], [2, 0, 0]]),
                    (
                        [3, 2, 1],
                        vec![[-1, -2, 1], [-1, -2, -3], [-1, 2, 1], [3, 2, 1]],
                    ),
                ],
            ),
            (
                hexagonal(),
                [6, 6, 4],
                vec![
                    ([3, 0, 0], vec![[-3, 0, 0], [3, 0, 0]]),
                    ([2, 2, 0], vec![[2, 2, 0], [2, -4, 0], [-4, 2, 0]]),
                    (
                        [3, 0, 2],
                        vec![[-3, 0, -2], [-3, 0, 2], [3, 0, -2], [3, 0, 2]],
                    ),
                    (
                        [2, 2, 2],
                        vec![
                            [2, 2, -2],
                            [2, 2, 2],
                            [2, -4, -2],
                            [2, -4, 2],
                            [-4, 2, -2],
                            [-4, 2, 2],
                        ],
                    ),
                ],
            ),
        ];
        for (rec, d_diag, expected) in cases {
            let bzg = dense(d_diag, IDENTITY, rec);
            let sparse = bz_grid_addresses(d_diag, IDENTITY, ZERO, rec, 1).unwrap();
            for (address, images_expected) in expected {
                let gp = grid_index_from_address(address, d_diag) as usize;
                assert_eq!(images(&bzg, gp), &images_expected[..]);
                assert_eq!(sparse.addresses[gp], images_expected[0]);
            }
        }
    }

    #[test]
    fn singular_reciprocal_lattice_is_an_error() {
        let rec = [[1.0, 2.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]];
        assert_eq!(
            bz_grid_addresses([2, 2, 2], IDENTITY, ZERO, rec, 2).err(),
            Some(BzGridAddressesError::SingularReciprocalLattice)
        );
    }
}
