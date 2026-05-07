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
}
