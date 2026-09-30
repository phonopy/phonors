//! Triplet integration-weight kernels.
//!
//! Port of `c/triplet_iw.c`.  These are the per-triplet workers that
//! compute the tetrahedron-method or Gaussian-smeared integration
//! weights consumed by `tpl_get_integration_weight*` (in
//! `c/triplet.c`).

#![allow(dead_code)]

use rayon::prelude::*;

use crate::bzgrid::{fill_neighboring_grid_points, BzGridError, BzGridView};
use crate::common::Vec3I;
use crate::tetrahedron_method::{integration_weight, WeightFunction};

/// `*mut T` wrapper opting into Send + Sync for rayon.  Used only inside
/// parallel kernels where the Rust author has manually verified that
/// the offsets touched by each task are disjoint.
#[derive(Clone, Copy)]
struct SyncMutPtr<T>(*mut T);
unsafe impl<T> Send for SyncMutPtr<T> {}
unsafe impl<T> Sync for SyncMutPtr<T> {}

impl<T> SyncMutPtr<T> {
    /// Method-style accessor for the raw pointer.  Calling this method
    /// (rather than touching the `.0` field) is what forces the 2021
    /// edition's disjoint-capture analysis to capture the whole
    /// wrapper into a closure, preserving its Send + Sync impls.
    fn ptr(self) -> *mut T {
        self.0
    }
}

const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

/// Per-channel relative grid addresses: 2 sign channels (q2, q3),
/// 24 * n tetrahedra, 4 vertices, 3 spatial components.  Built by
/// `triplet::set_relative_grid_address`.
pub type TpRelativeGridAddress = [Vec<[Vec3I; 4]>; 2];

/// BZ-grid indices of the tetrahedron vertices of a triplet, per
/// channel: `[2][num_tetra][4]`.
type TpVertices = [Vec<[i64; 4]>; 2];

/// Frequencies at the tetrahedron vertices of one band pair, per
/// channel: `[3][num_tetra][4]`.
type FreqVertices = [Vec<[f64; 4]>; 3];

fn new_freq_vertices(num_tetra: usize) -> FreqVertices {
    [
        vec![[0.0f64; 4]; num_tetra],
        vec![[0.0f64; 4]; num_tetra],
        vec![[0.0f64; 4]; num_tetra],
    ]
}

/// Triplet integration-weight type.
///
/// * `Type2`: q1+q2+q3=G, ph-ph lifetime — outputs `(g[2], g[0]-g[1])`.
/// * `Type3`: q1+q2+q3=G, collision matrix — outputs
///   `(g[2], g[0]-g[1], g[0]+g[1]+g[2])`.
/// * `Type4`: q+k_i-k_f=G, el-ph phonon decay — outputs `(g[0])`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TpType {
    Type2,
    Type3,
    Type4,
}

impl TpType {
    /// Map the C-side integer encoding (`tp_type`) onto the enum.
    pub fn try_from_i64(tp_type: i64) -> Result<Self, BzGridError> {
        match tp_type {
            2 => Ok(TpType::Type2),
            3 => Ok(TpType::Type3),
            4 => Ok(TpType::Type4),
            _ => Err(BzGridError::BadTpType),
        }
    }

    /// Number of g channels written into `iw` (== `g.shape[0]` on the
    /// Python side for Type2/Type3, 1 for Type4).
    pub fn num_channels(self) -> usize {
        match self {
            TpType::Type2 => 2,
            TpType::Type3 => 3,
            TpType::Type4 => 1,
        }
    }
}

/// Public: tetrahedron-method integration weights for a single triplet.
///
/// Mirrors `tpi_get_integration_weight`.  `iw_ch` is a slice of
/// per-channel output blocks for this triplet; its length must equal
/// `tp_type.num_channels()` and each inner slice covers
/// `num_band0 * num_band1 * num_band2` elements.  `iw_zero` is the
/// (single-channel) flag block of the same per-block size.  Disjoint
/// slices across triplets make this safe to call from rayon.
pub fn integration_weight_per_triplet(
    iw_ch: &mut [&mut [f64]],
    iw_zero: &mut [i8],
    frequency_points: &[f64],
    num_band0: i64,
    tp_relative_grid_address: &TpRelativeGridAddress,
    triplet: [i64; 3],
    bzgrid: &BzGridView,
    frequencies1: &[f64],
    num_band1: i64,
    frequencies2: &[f64],
    num_band2: i64,
    tp_type: TpType,
) -> Result<(), BzGridError> {
    debug_assert_eq!(iw_ch.len(), tp_type.num_channels());
    let nbb = (num_band1 * num_band2) as usize;
    let nb0 = num_band0 as usize;

    let vertices = triplet_tetrahedra_vertices(tp_relative_grid_address, triplet, bzgrid)?;
    let mut freq_vertices = new_freq_vertices(vertices[0].len());

    let max_i = max_tetra_channels(tp_type);
    for b12 in 0..nbb {
        let b1 = (b12 as i64) / num_band2;
        let b2 = (b12 as i64) % num_band2;
        build_freq_vertices(
            &mut freq_vertices,
            &vertices,
            frequencies1,
            frequencies2,
            num_band1,
            num_band2,
            b1,
            b2,
            tp_type,
        );
        let bboxes = freq_vertices_bboxes(&freq_vertices, max_i);
        for j in 0..nb0 {
            let adrs = j * nbb + b12;
            let f0 = frequency_points[j];
            let (ch, iwz) = compute_tetra_channels(f0, &freq_vertices, &bboxes, tp_type);
            iw_zero[adrs] = iwz;
            for (k, iw_ch_k) in iw_ch.iter_mut().enumerate() {
                iw_ch_k[adrs] = ch[k];
            }
        }
    }
    Ok(())
}

/// Public: tetrahedron-method integration weights for a single triplet,
/// with the inner `b12` loop parallelised over rayon.  Use this variant
/// when the outer triplet loop is too small to feed all threads.  The
/// per-channel slice contract is identical to the serial version.
pub fn integration_weight_per_triplet_inner_par(
    iw_ch: &mut [&mut [f64]],
    iw_zero: &mut [i8],
    frequency_points: &[f64],
    num_band0: i64,
    tp_relative_grid_address: &TpRelativeGridAddress,
    triplet: [i64; 3],
    bzgrid: &BzGridView,
    frequencies1: &[f64],
    num_band1: i64,
    frequencies2: &[f64],
    num_band2: i64,
    tp_type: TpType,
) -> Result<(), BzGridError> {
    debug_assert_eq!(iw_ch.len(), tp_type.num_channels());
    let nbb = (num_band1 * num_band2) as usize;
    let nb0 = num_band0 as usize;

    let vertices = triplet_tetrahedra_vertices(tp_relative_grid_address, triplet, bzgrid)?;

    let ch_ptrs: Vec<SyncMutPtr<f64>> = iw_ch
        .iter_mut()
        .map(|s| SyncMutPtr(s.as_mut_ptr()))
        .collect();
    let iwz_ptr = SyncMutPtr(iw_zero.as_mut_ptr());

    let max_i = max_tetra_channels(tp_type);
    // SAFETY: for each b12 in 0..nbb the inner j loop writes to offsets
    // { j * nbb + b12 : j in 0..nb0 } in every per-channel slice and in
    // iw_zero.  These index sets are pairwise disjoint across different
    // b12 values, so concurrent writes from different rayon tasks do not
    // race.  Slice lengths are nb0 * nbb so all offsets are in-bounds.
    let num_tetra = vertices[0].len();
    (0..nbb).into_par_iter().for_each_init(
        || new_freq_vertices(num_tetra),
        |freq_vertices, b12| {
            let b1 = (b12 as i64) / num_band2;
            let b2 = (b12 as i64) % num_band2;
            build_freq_vertices(
                freq_vertices,
                &vertices,
                frequencies1,
                frequencies2,
                num_band1,
                num_band2,
                b1,
                b2,
                tp_type,
            );
            let bboxes = freq_vertices_bboxes(freq_vertices, max_i);
            for j in 0..nb0 {
                let adrs = j * nbb + b12;
                let f0 = frequency_points[j];
                let (ch, iwz) = compute_tetra_channels(f0, freq_vertices, &bboxes, tp_type);
                unsafe {
                    *iwz_ptr.ptr().add(adrs) = iwz;
                    for (k, ch_ptr) in ch_ptrs.iter().enumerate() {
                        *ch_ptr.ptr().add(adrs) = ch[k];
                    }
                }
            }
        },
    );

    Ok(())
}

/// Public: Gaussian-smeared integration weights for a single triplet.
///
/// Mirrors `tpi_get_integration_weight_with_sigma`.  `iw_ch` is a slice
/// of per-channel output blocks (length = `tp_type.num_channels()`,
/// each `num_band0 * num_band * num_band` long).  `cutoff <= 0`
/// disables the cutoff-skip optimisation (matches C semantics).
pub fn integration_weight_with_sigma_per_triplet(
    iw_ch: &mut [&mut [f64]],
    iw_zero: &mut [i8],
    sigma: f64,
    cutoff: f64,
    frequency_points: &[f64],
    num_band0: i64,
    triplet: [i64; 3],
    frequencies: &[f64],
    num_band: i64,
    tp_type: TpType,
) {
    debug_assert_eq!(iw_ch.len(), tp_type.num_channels());
    let nbb = (num_band * num_band) as usize;
    let nb0 = num_band0 as usize;

    for b12 in 0..nbb {
        let b1 = (b12 as i64) / num_band;
        let b2 = (b12 as i64) % num_band;
        let f1 = frequencies[(triplet[1] * num_band + b1) as usize];
        let f2 = frequencies[(triplet[2] * num_band + b2) as usize];
        for j in 0..nb0 {
            let adrs = j * nbb + b12;
            let f0 = frequency_points[j];
            let (ch, iwz) = compute_sigma_channels(f0, f1, f2, sigma, cutoff, tp_type);
            iw_zero[adrs] = iwz;
            for (k, iw_ch_k) in iw_ch.iter_mut().enumerate() {
                iw_ch_k[adrs] = ch[k];
            }
        }
    }
}

/// Public: Gaussian-smeared integration weights for a single triplet,
/// with the inner `b12` loop parallelised over rayon.  See the
/// non-`_inner_par` variant for the slice contract.
pub fn integration_weight_with_sigma_per_triplet_inner_par(
    iw_ch: &mut [&mut [f64]],
    iw_zero: &mut [i8],
    sigma: f64,
    cutoff: f64,
    frequency_points: &[f64],
    num_band0: i64,
    triplet: [i64; 3],
    frequencies: &[f64],
    num_band: i64,
    tp_type: TpType,
) {
    debug_assert_eq!(iw_ch.len(), tp_type.num_channels());
    let nbb = (num_band * num_band) as usize;
    let nb0 = num_band0 as usize;

    let ch_ptrs: Vec<SyncMutPtr<f64>> = iw_ch
        .iter_mut()
        .map(|s| SyncMutPtr(s.as_mut_ptr()))
        .collect();
    let iwz_ptr = SyncMutPtr(iw_zero.as_mut_ptr());

    // SAFETY: see the tetrahedron inner_par variant above; the same
    // disjointedness argument holds (offsets j * nbb + b12 are pairwise
    // disjoint across b12).
    (0..nbb).into_par_iter().for_each(|b12| {
        let b1 = (b12 as i64) / num_band;
        let b2 = (b12 as i64) % num_band;
        let f1 = frequencies[(triplet[1] * num_band + b1) as usize];
        let f2 = frequencies[(triplet[2] * num_band + b2) as usize];
        for j in 0..nb0 {
            let adrs = j * nbb + b12;
            let f0 = frequency_points[j];
            let (ch, iwz) = compute_sigma_channels(f0, f1, f2, sigma, cutoff, tp_type);
            unsafe {
                *iwz_ptr.ptr().add(adrs) = iwz;
                for (k, ch_ptr) in ch_ptrs.iter().enumerate() {
                    *ch_ptr.ptr().add(adrs) = ch[k];
                }
            }
        }
    });
}

/// Build the `[2][num_tetra][4]` per-channel BZ-grid vertex indices for
/// a triplet.  Mirrors `get_triplet_tetrahedra_vertices`.
fn triplet_tetrahedra_vertices(
    tp_relative_grid_address: &TpRelativeGridAddress,
    triplet: [i64; 3],
    bzgrid: &BzGridView,
) -> Result<TpVertices, BzGridError> {
    let num_tetra = tp_relative_grid_address[0].len();
    let mut vertices: TpVertices = [vec![[0i64; 4]; num_tetra], vec![[0i64; 4]; num_tetra]];
    for i in 0..2 {
        for j in 0..num_tetra {
            fill_neighboring_grid_points(
                &mut vertices[i][j],
                triplet[i + 1],
                &tp_relative_grid_address[i][j],
                bzgrid,
            )?;
        }
    }
    Ok(vertices)
}

/// Fill the `[3][num_tetra][4]` per-channel frequency vertices for a
/// single `(b1, b2)` band pair.  Mirrors `set_freq_vertices`.
///
/// For Type2/Type3 the three channels are `-f1+f2`, `f1-f2`,
/// `f1+f2` (negative input frequencies are clamped to 0).
/// For Type4 only channel 0 (`-f1+f2`) is populated.
fn build_freq_vertices(
    out: &mut FreqVertices,
    vertices: &TpVertices,
    frequencies1: &[f64],
    frequencies2: &[f64],
    num_band1: i64,
    num_band2: i64,
    b1: i64,
    b2: i64,
    tp_type: TpType,
) {
    for i in 0..vertices[0].len() {
        for j in 0..4 {
            let mut f1 = frequencies1[(vertices[0][i][j] * num_band1 + b1) as usize];
            let mut f2 = frequencies2[(vertices[1][i][j] * num_band2 + b2) as usize];
            match tp_type {
                TpType::Type2 | TpType::Type3 => {
                    if f1 < 0.0 {
                        f1 = 0.0;
                    }
                    if f2 < 0.0 {
                        f2 = 0.0;
                    }
                    out[0][i][j] = -f1 + f2;
                    out[1][i][j] = f1 - f2;
                    out[2][i][j] = f1 + f2;
                }
                TpType::Type4 => {
                    out[0][i][j] = -f1 + f2;
                }
            }
        }
    }
}

/// Number of tetrahedron channels to compute for a given `tp_type`.
fn max_tetra_channels(tp_type: TpType) -> usize {
    match tp_type {
        TpType::Type2 | TpType::Type3 => 3,
        TpType::Type4 => 1,
    }
}

/// Per-channel (fmin, fmax) bounding boxes across the tetrahedra's
/// 4 vertices.  Only the first `max_i` entries are populated.  Hoisted
/// out of the `f0` loop so the per-`f0` in-tetrahedron test reduces
/// from a min/max scan over all vertices to a pair of comparisons.
fn freq_vertices_bboxes(freq_vertices: &FreqVertices, max_i: usize) -> [(f64, f64); 3] {
    let mut out = [(0.0f64, 0.0f64); 3];
    for i in 0..max_i {
        let mut fmin = freq_vertices[i][0][0];
        let mut fmax = freq_vertices[i][0][0];
        for j in 0..freq_vertices[i].len() {
            for k in 0..4 {
                let v = freq_vertices[i][j][k];
                if fmin > v {
                    fmin = v;
                }
                if fmax < v {
                    fmax = v;
                }
            }
        }
        out[i] = (fmin, fmax);
    }
    out
}

/// Compute g[0..max_i] and the iw_zero flag for one (f0, freq_vertices).
/// Mirrors `set_g`.  `bboxes[i]` must hold the (fmin, fmax) of
/// `freq_vertices[i]` for every `i in 0..max_i`.  Returns `(g, iw_zero)`
/// where `iw_zero == 1` means every populated `g[i]` is exactly zero.
fn compute_g(
    f0: f64,
    freq_vertices: &FreqVertices,
    bboxes: &[(f64, f64); 3],
    max_i: usize,
) -> ([f64; 3], i8) {
    let mut g = [0.0f64; 3];
    let mut iw_zero: i8 = 1;
    for i in 0..max_i {
        let (fmin, fmax) = bboxes[i];
        if fmin <= f0 && f0 <= fmax {
            g[i] = integration_weight(f0, &freq_vertices[i], WeightFunction::I);
            iw_zero = 0;
        } else {
            g[i] = 0.0;
        }
    }
    (g, iw_zero)
}

/// Pure helper: combine tetrahedron g-values into the per-channel output
/// slots for a single `(f0, freq_vertices)`.  The first
/// `tp_type.num_channels()` entries of the returned array are meaningful;
/// trailing entries are 0.0 padding.  No side effects.
fn compute_tetra_channels(
    f0: f64,
    freq_vertices: &FreqVertices,
    bboxes: &[(f64, f64); 3],
    tp_type: TpType,
) -> ([f64; 3], i8) {
    let max_i = max_tetra_channels(tp_type);
    let (g, iwz) = compute_g(f0, freq_vertices, bboxes, max_i);
    let ch = match tp_type {
        TpType::Type2 => [g[2], g[0] - g[1], 0.0],
        TpType::Type3 => [g[2], g[0] - g[1], g[0] + g[1] + g[2]],
        TpType::Type4 => [g[0], 0.0, 0.0],
    };
    (ch, iwz)
}

/// Pure helper: compute the per-channel Gaussian-smeared values and the
/// iw_zero flag for a single `(f0, f1, f2)`.  `cutoff <= 0` disables the
/// skip optimisation (matches the C convention).  Only the first
/// `tp_type.num_channels()` entries of the returned array are meaningful.
fn compute_sigma_channels(
    f0: f64,
    f1: f64,
    f2: f64,
    sigma: f64,
    cutoff: f64,
    tp_type: TpType,
) -> ([f64; 3], i8) {
    match tp_type {
        TpType::Type2 | TpType::Type3 => {
            if cutoff > 0.0
                && (f0 + f1 - f2).abs() > cutoff
                && (f0 - f1 + f2).abs() > cutoff
                && (f0 - f1 - f2).abs() > cutoff
            {
                return ([0.0, 0.0, 0.0], 1);
            }
            let g0 = gaussian(f0 + f1 - f2, sigma);
            let g1 = gaussian(f0 - f1 + f2, sigma);
            let g2 = gaussian(f0 - f1 - f2, sigma);
            let ch = match tp_type {
                TpType::Type2 => [g2, g0 - g1, 0.0],
                TpType::Type3 => [g2, g0 - g1, g0 + g1 + g2],
                TpType::Type4 => unreachable!(),
            };
            (ch, 0)
        }
        TpType::Type4 => {
            if cutoff > 0.0 && (f0 + f1 - f2).abs() > cutoff {
                ([0.0, 0.0, 0.0], 1)
            } else {
                ([gaussian(f0 + f1 - f2, sigma), 0.0, 0.0], 0)
            }
        }
    }
}

/// Mirrors `funcs_gaussian` from `c/funcs.c`.
fn gaussian(x: f64, sigma: f64) -> f64 {
    INV_SQRT_2PI / sigma * (-x * x / 2.0 / sigma / sigma).exp()
}

/// `(first band, number of bands)` of each degenerate set with more than
/// one band.  `degenerate_ids[b]` is the smallest band index of the set of
/// band `b`, and the bands of a set are consecutive.
fn degenerate_runs(degenerate_ids: &[i64]) -> Vec<(usize, usize)> {
    let num_band = degenerate_ids.len();
    let starts: Vec<usize> = (0..num_band)
        .filter(|&b| degenerate_ids[b] == b as i64)
        .collect();
    starts
        .iter()
        .zip(starts.iter().skip(1).chain([&num_band]))
        .map(|(&s, &e)| (s, e - s))
        .filter(|&(_, n)| n > 1)
        .collect()
}

/// Average the integration weights of one triplet over the degenerate
/// bands at q' and at q'', in place.
///
/// Mirrors `_average_weights_over_degenerate_sets` in phono3py's
/// `triplets.py`.  `iw` holds the channels one after another, each of
/// shape `(num_band0, num_band, num_band)` with the band at q' before the
/// band at q''; `iw_zero` has the shape of one channel.  The weights are
/// averaged over the sets at q' first and at q'' second.  An element that
/// has a nonzero weight in any channel after the average is unmarked in
/// `iw_zero`; marks are never added.
///
/// `degenerate_ids1` and `degenerate_ids2` (`num_band` each) give the
/// smallest band index of the degenerate set of each band at q' and q''.
pub fn average_weights_over_degenerate_sets(
    iw: &mut [f64],
    iw_zero: &mut [i8],
    degenerate_ids1: &[i64],
    degenerate_ids2: &[i64],
    num_band0: usize,
    num_band: usize,
) {
    let runs1 = degenerate_runs(degenerate_ids1);
    let runs2 = degenerate_runs(degenerate_ids2);
    if runs1.is_empty() && runs2.is_empty() {
        return;
    }
    let nbb = num_band * num_band;
    let num_band_prod = num_band0 * nbb;
    let num_channels = iw.len() / num_band_prod;
    for block in iw.chunks_mut(nbb) {
        for &(start, count) in &runs1 {
            for b2 in 0..num_band {
                let sum: f64 = (start..start + count)
                    .map(|b1| block[b1 * num_band + b2])
                    .sum();
                let mean = sum / count as f64;
                for b1 in start..start + count {
                    block[b1 * num_band + b2] = mean;
                }
            }
        }
        for &(start, count) in &runs2 {
            for b1 in 0..num_band {
                let row = &mut block[b1 * num_band..(b1 + 1) * num_band];
                let sum: f64 = row[start..start + count].iter().sum();
                let mean = sum / count as f64;
                row[start..start + count].fill(mean);
            }
        }
    }
    for (k, z) in iw_zero.iter_mut().enumerate() {
        if *z != 0 && (0..num_channels).any(|c| iw[c * num_band_prod + k] != 0.0) {
            *z = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn degenerate_runs_skip_single_bands() {
        assert_eq!(
            degenerate_runs(&[0, 1, 1, 3, 3, 3, 6]),
            vec![(1, 2), (3, 3)]
        );
        assert!(degenerate_runs(&[0, 1, 2]).is_empty());
        assert_eq!(degenerate_runs(&[0, 0, 0]), vec![(0, 3)]);
    }

    #[test]
    fn average_weights_over_degenerate_sets_keeps_block_sums() {
        let num_band0 = 2;
        let num_band = 4;
        let nbb = num_band * num_band;
        let num_band_prod = num_band0 * nbb;
        let ids1 = [0i64, 1, 1, 3];
        let ids2 = [0i64, 0, 0, 3];
        // Two channels of distinct values; one element of the (1..3, 0..3)
        // block of channel-pair (j=0) is zero in both channels and marked.
        let mut iw: Vec<f64> = (0..2 * num_band_prod)
            .map(|i| ((i * 7919) % 97) as f64 - 40.0)
            .collect();
        let mut iw_zero = vec![0i8; num_band_prod];
        let k = 2 * num_band + 1;
        iw[k] = 0.0;
        iw[num_band_prod + k] = 0.0;
        iw_zero[k] = 1;
        let orig = iw.clone();

        average_weights_over_degenerate_sets(
            &mut iw,
            &mut iw_zero,
            &ids1,
            &ids2,
            num_band0,
            num_band,
        );

        let block_sum = |v: &[f64], off: usize| -> f64 {
            (1..3)
                .flat_map(|b1| (0..3).map(move |b2| (b1, b2)))
                .map(|(b1, b2)| v[off + b1 * num_band + b2])
                .sum()
        };
        for off in (0..2 * num_band_prod).step_by(nbb) {
            assert!((block_sum(&iw, off) - block_sum(&orig, off)).abs() < 1e-12);
            let first = iw[off + num_band];
            for b1 in 1..3 {
                for b2 in 0..3 {
                    assert!((iw[off + b1 * num_band + b2] - first).abs() < 1e-12);
                }
            }
            // Band 0 at q' and band 3 at q'' are not degenerate.
            assert_eq!(iw[off + 3 * num_band + 3], orig[off + 3 * num_band + 3]);
        }
        assert_eq!(iw_zero[k], 0);
    }

    #[test]
    fn gaussian_at_zero_is_peak() {
        let sigma = 0.1;
        let g0 = gaussian(0.0, sigma);
        let expected = INV_SQRT_2PI / sigma;
        assert!((g0 - expected).abs() < 1e-12);
    }

    #[test]
    fn gaussian_one_sigma_drops_to_e_minus_half() {
        let sigma = 0.5;
        let g = gaussian(sigma, sigma);
        let expected = INV_SQRT_2PI / sigma * (-0.5_f64).exp();
        assert!((g - expected).abs() < 1e-12);
    }

    #[test]
    fn gaussian_integration_weight_with_sigma_type4_no_cutoff() {
        let mut iw = vec![0.0f64; 2];
        let mut iw_zero = vec![0i8; 2];
        let frequency_points = [1.0, 2.0];
        let frequencies = [0.0, 1.0];
        // num_band = 1, num_band0 = 2, triplet = (_, 0, 1) so f1=0, f2=1.
        let mut chs: [&mut [f64]; 1] = [iw.as_mut_slice()];
        integration_weight_with_sigma_per_triplet(
            &mut chs,
            &mut iw_zero,
            0.5,
            -1.0, // cutoff disabled
            &frequency_points,
            2,
            [0, 0, 1],
            &frequencies,
            1,
            TpType::Type4,
        );
        // For j=0: f0=1, f1=0, f2=1, x=f0+f1-f2=0
        // For j=1: f0=2, f1=0, f2=1, x=1
        let expected0 = gaussian(0.0, 0.5);
        let expected1 = gaussian(1.0, 0.5);
        assert!((iw[0] - expected0).abs() < 1e-12);
        assert!((iw[1] - expected1).abs() < 1e-12);
        assert_eq!(iw_zero, vec![0, 0]);
    }

    #[test]
    fn gaussian_integration_weight_with_sigma_type4_cutoff_zeros() {
        let mut iw = vec![0.0f64; 1];
        let mut iw_zero = vec![0i8; 1];
        let mut chs: [&mut [f64]; 1] = [iw.as_mut_slice()];
        // cutoff small enough to discard everything (|x|=1 > 0.5)
        integration_weight_with_sigma_per_triplet(
            &mut chs,
            &mut iw_zero,
            0.5,
            0.5,
            &[2.0],
            1,
            [0, 0, 1],
            &[0.0, 1.0],
            1,
            TpType::Type4,
        );
        assert_eq!(iw, vec![0.0]);
        assert_eq!(iw_zero[0], 1);
    }

    #[test]
    fn tetra_channels_of_repeated_set_are_unchanged() {
        // Two copies of the same 24 tetrahedra give the channels of one.
        let mut single = new_freq_vertices(24);
        for (i, channel) in single.iter_mut().enumerate() {
            for (j, tetra) in channel.iter_mut().enumerate() {
                let x = (i * 24 + j) as f64;
                *tetra = [x.sin(), 1.0 + x.cos(), 0.5 * (2.0 * x).sin(), 2.0 - x.cos()];
            }
        }
        let doubled: FreqVertices =
            std::array::from_fn(|i| single[i].iter().chain(single[i].iter()).copied().collect());
        let bb_single = freq_vertices_bboxes(&single, 3);
        let bb_doubled = freq_vertices_bboxes(&doubled, 3);
        assert_eq!(bb_single, bb_doubled);
        for f0 in [-1.0, 0.2, 0.9, 1.6, 3.5] {
            for tp in [TpType::Type2, TpType::Type3, TpType::Type4] {
                let (ch1, z1) = compute_tetra_channels(f0, &single, &bb_single, tp);
                let (ch2, z2) = compute_tetra_channels(f0, &doubled, &bb_doubled, tp);
                assert_eq!(z1, z2);
                for k in 0..3 {
                    assert!((ch1[k] - ch2[k]).abs() < 1e-14, "{f0} {tp:?} {k}");
                }
            }
        }
    }
}
