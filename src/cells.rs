//! Crystal-cell helpers (atomic position permutation matching).
//!
//! Mirrors the cells routines in `c/phonopy.c` that operate on
//! scaled positions and a lattice.  The fc2/fc3 helpers live in their
//! own modules; this module is the home for cell-level utilities.

/// Find the permutation that maps `pos` onto `rot_pos` (modulo the
/// lattice).  Mirrors `phpy_compute_permutation` in `c/phonopy.c`.
///
/// `pos` and `rot_pos` are scaled positions, shape `[num_pos, 3]` packed
/// in C order.  `lat` is the lattice basis as row vectors of a 3x3 matrix
/// (i.e. `lat[i]` is the i-th basis vector); the C signature passes
/// `double lat[3][3]` and computes `lat[k][l] * diff[l]`, so we
/// preserve that orientation.
///
/// `rot_atom[i]` is set to the index `i_orig` such that `pos[i_orig]`
/// (modulo the lattice) equals `rot_pos[i]` within `symprec`, or kept
/// at -1 when no match is found.
///
/// Returns `true` if every position was matched, otherwise `false` (in
/// which case some entries of `rot_atom` remain `-1`).
pub(crate) fn compute_permutation(
    rot_atom: &mut [i64],
    lat: &[f64; 9],
    pos: &[f64],
    rot_pos: &[f64],
    num_pos: usize,
    symprec: f64,
) -> bool {
    for slot in rot_atom.iter_mut().take(num_pos) {
        *slot = -1;
    }

    // Optimization mirroring the C version: iterate primarily by `pos`
    // (find where index 0 belongs in `rot_atom`, then 1, ...).  Keep
    // the running unassigned index `search_start` to skip over
    // already-filled slots.  Fast when the permutation is close to the
    // identity.
    let mut search_start = 0usize;
    for i in 0..num_pos {
        while search_start < num_pos && rot_atom[search_start] >= 0 {
            search_start += 1;
        }
        for j in search_start..num_pos {
            if rot_atom[j] >= 0 {
                continue;
            }

            let mut diff = [0.0f64; 3];
            for k in 0..3 {
                let d = pos[i * 3 + k] - rot_pos[j * 3 + k];
                diff[k] = d - d.round();
            }
            let mut distance2 = 0.0f64;
            for k in 0..3 {
                let mut diff_cart = 0.0f64;
                for l in 0..3 {
                    diff_cart += lat[k * 3 + l] * diff[l];
                }
                distance2 += diff_cart * diff_cart;
            }

            if distance2.sqrt() < symprec {
                rot_atom[j] = i as i64;
                break;
            }
        }
    }

    rot_atom.iter().take(num_pos).all(|&v| v >= 0)
}

/// Mirror of `phpy_set_smallest_vectors_sparse` in `c/phonopy.c`.
///
/// For each pair `(pos_to[i], pos_from[j])`, find the lattice translations
/// whose images yield the shortest distance.  Output:
///
/// - `smallest_vectors[(i * num_pos_from + j) * 27 + count]` for
///   `count = 0..multiplicity-1` holds the equidistant shortest vectors,
///   transformed back to supercell coordinates by `trans_mat`.
/// - `multiplicity[i * num_pos_from + j]` is the number of equidistant
///   minima.  Multiplicities greater than 27 are clipped (matching the
///   storage capacity of `smallest_vectors`); the C version printed a
///   warning and broke out of the loop, but we silently clip here -- the
///   caller is expected to size `lattice_points` so the cap is not hit.
///
/// `reduced_basis` is a 3x3 matrix in row-major order; the inner loop
/// computes `reduced_basis[l] dot vec` mirroring the C indexing
/// `reduced_basis[l][0..2]`.  `trans_mat` is similarly row-major i64.
pub(crate) fn set_smallest_vectors_sparse(
    smallest_vectors: &mut [f64],
    multiplicity: &mut [i64],
    pos_to: &[f64],
    num_pos_to: usize,
    pos_from: &[f64],
    num_pos_from: usize,
    lattice_points: &[i64],
    num_lattice_points: usize,
    reduced_basis: &[f64; 9],
    trans_mat: &[i64; 9],
    symprec: f64,
) {
    let mut length = vec![0.0f64; num_lattice_points];
    let mut vec_buf = vec![[0.0f64; 3]; num_lattice_points];

    for i in 0..num_pos_to {
        for j in 0..num_pos_from {
            for k in 0..num_lattice_points {
                for l in 0..3 {
                    vec_buf[k][l] =
                        pos_to[i * 3 + l] - pos_from[j * 3 + l] + lattice_points[k * 3 + l] as f64;
                }
                let mut len_sq = 0.0f64;
                for l in 0..3 {
                    let comp = reduced_basis[l * 3] * vec_buf[k][0]
                        + reduced_basis[l * 3 + 1] * vec_buf[k][1]
                        + reduced_basis[l * 3 + 2] * vec_buf[k][2];
                    len_sq += comp * comp;
                }
                length[k] = len_sq.sqrt();
            }

            let minimum = length.iter().copied().fold(f64::INFINITY, f64::min);

            let mut count: usize = 0;
            for k in 0..num_lattice_points {
                if length[k] - minimum < symprec {
                    if count < 27 {
                        for l in 0..3 {
                            let vec_xyz = trans_mat[l * 3] as f64 * vec_buf[k][0]
                                + trans_mat[l * 3 + 1] as f64 * vec_buf[k][1]
                                + trans_mat[l * 3 + 2] as f64 * vec_buf[k][2];
                            let idx = ((i * num_pos_from + j) * 27 + count) * 3 + l;
                            smallest_vectors[idx] = vec_xyz;
                        }
                    }
                    count += 1;
                }
            }
            multiplicity[i * num_pos_from + j] = count.min(27) as i64;
        }
    }
}

/// Mirror of `phpy_set_smallest_vectors_dense` in `c/phonopy.c`.
///
/// Two-pass interface preserved for parity with the C kernel:
///
/// - `initialize == true`: only fills `multiplicity[(i,j)] = [count, addr]`.
///   `addr` is the running offset into the dense `smallest_vectors` array.
/// - `initialize == false`: writes the actual shortest vectors at the
///   addresses computed by the previous pass.
pub(crate) fn set_smallest_vectors_dense(
    smallest_vectors: &mut [f64],
    multiplicity: &mut [i64],
    pos_to: &[f64],
    num_pos_to: usize,
    pos_from: &[f64],
    num_pos_from: usize,
    lattice_points: &[i64],
    num_lattice_points: usize,
    reduced_basis: &[f64; 9],
    trans_mat: &[i64; 9],
    initialize: bool,
    symprec: f64,
) {
    let mut length = vec![0.0f64; num_lattice_points];
    let mut vec_buf = vec![[0.0f64; 3]; num_lattice_points];

    let mut adrs: i64 = 0;
    for i in 0..num_pos_to {
        for j in 0..num_pos_from {
            for k in 0..num_lattice_points {
                for l in 0..3 {
                    vec_buf[k][l] =
                        pos_to[i * 3 + l] - pos_from[j * 3 + l] + lattice_points[k * 3 + l] as f64;
                }
                let mut len_sq = 0.0f64;
                for l in 0..3 {
                    let comp = reduced_basis[l * 3] * vec_buf[k][0]
                        + reduced_basis[l * 3 + 1] * vec_buf[k][1]
                        + reduced_basis[l * 3 + 2] * vec_buf[k][2];
                    len_sq += comp * comp;
                }
                length[k] = len_sq.sqrt();
            }

            let minimum = length.iter().copied().fold(f64::INFINITY, f64::min);

            let mut count: i64 = 0;
            for k in 0..num_lattice_points {
                if length[k] - minimum < symprec {
                    if !initialize {
                        for l in 0..3 {
                            let vec_xyz = trans_mat[l * 3] as f64 * vec_buf[k][0]
                                + trans_mat[l * 3 + 1] as f64 * vec_buf[k][1]
                                + trans_mat[l * 3 + 2] as f64 * vec_buf[k][2];
                            let idx = ((adrs + count) as usize) * 3 + l;
                            smallest_vectors[idx] = vec_xyz;
                        }
                    }
                    count += 1;
                }
            }
            if initialize {
                let m = (i * num_pos_from + j) * 2;
                multiplicity[m] = count;
                multiplicity[m + 1] = adrs;
            }
            adrs += count;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_lat() -> [f64; 9] {
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    }

    #[test]
    fn identity_permutation() {
        let pos = [0.0, 0.0, 0.0, 0.5, 0.5, 0.5];
        let rot_pos = pos;
        let mut rot_atom = [0i64; 2];
        let ok = compute_permutation(&mut rot_atom, &identity_lat(), &pos, &rot_pos, 2, 1e-5);
        assert!(ok);
        assert_eq!(rot_atom, [0, 1]);
    }

    #[test]
    fn swap_permutation() {
        let pos = [0.0, 0.0, 0.0, 0.5, 0.5, 0.5];
        // rot_pos has the two atoms swapped.
        let rot_pos = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0];
        let mut rot_atom = [0i64; 2];
        let ok = compute_permutation(&mut rot_atom, &identity_lat(), &pos, &rot_pos, 2, 1e-5);
        assert!(ok);
        assert_eq!(rot_atom, [1, 0]);
    }

    #[test]
    fn lattice_wrap() {
        // pos[0] + (1, 0, 0) lattice translation == rot_pos[0]; should match.
        let pos = [0.1, 0.2, 0.3];
        let rot_pos = [1.1, 0.2, 0.3];
        let mut rot_atom = [0i64; 1];
        let ok = compute_permutation(&mut rot_atom, &identity_lat(), &pos, &rot_pos, 1, 1e-5);
        assert!(ok);
        assert_eq!(rot_atom, [0]);
    }

    #[test]
    fn no_match_returns_false() {
        let pos = [0.0, 0.0, 0.0];
        let rot_pos = [0.5, 0.0, 0.0];
        let mut rot_atom = [0i64; 1];
        let ok = compute_permutation(&mut rot_atom, &identity_lat(), &pos, &rot_pos, 1, 1e-3);
        assert!(!ok);
        assert_eq!(rot_atom, [-1]);
    }

    fn cubic_lattice_points() -> Vec<i64> {
        // 27 lattice translations (-1, 0, +1) along each axis
        let mut pts: Vec<i64> = Vec::with_capacity(27 * 3);
        for x in -1..=1i64 {
            for y in -1..=1i64 {
                for z in -1..=1i64 {
                    pts.extend_from_slice(&[x, y, z]);
                }
            }
        }
        pts
    }

    #[test]
    fn sparse_simple_cubic_pair() {
        // pos_to[0] = (0.6, 0, 0), pos_from[0] = (0, 0, 0).  Closest under
        // unit cubic lattice is the (-1, 0, 0) translation -> (-0.4, 0, 0)
        // in supercell coords (trans_mat = identity).
        let pos_to = [0.6, 0.0, 0.0];
        let pos_from = [0.0, 0.0, 0.0];
        let lattice_points = cubic_lattice_points();
        let reduced_basis = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let trans_mat = [1, 0, 0, 0, 1, 0, 0, 0, 1];

        let mut svecs = vec![0.0f64; 27 * 3];
        let mut mult = vec![0i64; 1];
        set_smallest_vectors_sparse(
            &mut svecs,
            &mut mult,
            &pos_to,
            1,
            &pos_from,
            1,
            &lattice_points,
            27,
            &reduced_basis,
            &trans_mat,
            1e-5,
        );
        assert_eq!(mult[0], 1);
        assert!((svecs[0] - (-0.4)).abs() < 1e-12);
        assert!(svecs[1].abs() < 1e-12);
        assert!(svecs[2].abs() < 1e-12);
    }

    #[test]
    fn sparse_cubic_corner_8_equidistant() {
        // pos_to = (0.5, 0.5, 0.5) from origin under cubic lattice has 8
        // equidistant nearest images: (+/-0.5, +/-0.5, +/-0.5).
        let pos_to = [0.5, 0.5, 0.5];
        let pos_from = [0.0, 0.0, 0.0];
        let lattice_points = cubic_lattice_points();
        let reduced_basis = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let trans_mat = [1, 0, 0, 0, 1, 0, 0, 0, 1];

        let mut svecs = vec![0.0f64; 27 * 3];
        let mut mult = vec![0i64; 1];
        set_smallest_vectors_sparse(
            &mut svecs,
            &mut mult,
            &pos_to,
            1,
            &pos_from,
            1,
            &lattice_points,
            27,
            &reduced_basis,
            &trans_mat,
            1e-5,
        );
        assert_eq!(mult[0], 8);
    }

    #[test]
    fn dense_two_pass_matches_sparse() {
        // Drive the dense kernel through the same scenario as the sparse
        // 8-equidistant test and verify total count + values agree.
        let pos_to = [0.5, 0.5, 0.5];
        let pos_from = [0.0, 0.0, 0.0];
        let lattice_points = cubic_lattice_points();
        let reduced_basis = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let trans_mat = [1, 0, 0, 0, 1, 0, 0, 0, 1];

        let mut multi = vec![0i64; 2];
        let mut dummy_svecs = vec![0.0f64; 1];
        set_smallest_vectors_dense(
            &mut dummy_svecs,
            &mut multi,
            &pos_to,
            1,
            &pos_from,
            1,
            &lattice_points,
            27,
            &reduced_basis,
            &trans_mat,
            true,
            1e-5,
        );
        assert_eq!(multi, vec![8, 0]);

        let mut svecs = vec![0.0f64; 8 * 3];
        set_smallest_vectors_dense(
            &mut svecs,
            &mut multi,
            &pos_to,
            1,
            &pos_from,
            1,
            &lattice_points,
            27,
            &reduced_basis,
            &trans_mat,
            false,
            1e-5,
        );
        for chunk in svecs.chunks(3) {
            assert!((chunk[0].abs() - 0.5).abs() < 1e-12);
            assert!((chunk[1].abs() - 0.5).abs() < 1e-12);
            assert!((chunk[2].abs() - 0.5).abs() < 1e-12);
        }
    }
}
