//! Second-order force constant (fc2) helpers.
//!
//! Mirrors the fc2 routines in `c/phonopy.c`.  `fc` (full FC2) has
//! layout `[n_satom, n_satom, 3, 3]` packed in C order, so each atom
//! pair occupies a contiguous block of 9 doubles indexed as
//! `i * n_satom * 9 + j * 9 + k * 3 + l`.

/// Average `fc[i, j, k, l]` with `fc[j, i, l, k]` over every atom pair
/// and write the mean back to both positions, enforcing the index-
/// permutation symmetry of full fc2.  Mirrors the static helper
/// `set_index_permutation_symmetry_fc` in `c/phonopy.c`.
fn set_index_permutation_symmetry_fc(fc: &mut [f64], natom: usize) {
    let stride_i = natom * 9;
    for i in 0..natom {
        // Off-diagonal pairs.
        for j in (i + 1)..natom {
            for k in 0..3 {
                for l in 0..3 {
                    let m = i * stride_i + j * 9 + k * 3 + l;
                    let n = j * stride_i + i * 9 + l * 3 + k;
                    let avg = (fc[m] + fc[n]) / 2.0;
                    fc[m] = avg;
                    fc[n] = avg;
                }
            }
        }
        // Diagonal block: average fc[i, i, k, l] with fc[i, i, l, k].
        for k in 0..2 {
            for l in (k + 1)..3 {
                let m = i * stride_i + i * 9 + k * 3 + l;
                let n = i * stride_i + i * 9 + l * 3 + k;
                let avg = (fc[m] + fc[n]) / 2.0;
                fc[m] = avg;
                fc[n] = avg;
            }
        }
    }
}

/// Replace each on-site `fc[i, i, k, l]` by `-(S_kl + S_lk) / 2` where
/// `S_kl = sum_{j != i} fc[i, j, k, l]`.  Enforces the acoustic sum
/// rule on full fc2.  Mirrors the static helper
/// `set_translational_symmetry_fc` in `c/phonopy.c`.
fn set_translational_symmetry_fc(fc: &mut [f64], natom: usize) {
    let stride_i = natom * 9;
    for i in 0..natom {
        let mut sums = [[0.0f64; 3]; 3];
        for k in 0..3 {
            for l in 0..3 {
                let mut s = 0.0;
                for j in 0..natom {
                    if i != j {
                        s += fc[i * stride_i + j * 9 + k * 3 + l];
                    }
                }
                sums[k][l] = s;
            }
        }
        for k in 0..3 {
            for l in 0..3 {
                fc[i * stride_i + i * 9 + k * 3 + l] = -(sums[k][l] + sums[l][k]) / 2.0;
            }
        }
    }
}

/// Symmetrize full fc2 by repeatedly removing column/row drift and
/// applying the index permutation, then enforce the acoustic sum rule
/// once at the end.  Mirrors `phpy_perm_trans_symmetrize_fc` in
/// `c/phonopy.c`.
///
/// `fc` has shape `[n_satom, n_satom, 3, 3]` and is modified in place.
/// `level` is the number of drift-removal + permutation iterations.
pub(crate) fn perm_trans_symmetrize_fc(fc: &mut [f64], n_satom: usize, level: usize) {
    let stride_i = n_satom * 9;
    let inv_n = 1.0 / (n_satom as f64);

    for _ in 0..level {
        // Subtract drift along the column axis (sum over i for fixed j).
        for j in 0..n_satom {
            for k in 0..3 {
                for l in 0..3 {
                    let mut sum = 0.0;
                    for i in 0..n_satom {
                        sum += fc[i * stride_i + j * 9 + k * 3 + l];
                    }
                    sum *= inv_n;
                    for i in 0..n_satom {
                        fc[i * stride_i + j * 9 + k * 3 + l] -= sum;
                    }
                }
            }
        }
        // Subtract drift along the row axis (sum over j for fixed i).
        for i in 0..n_satom {
            for k in 0..3 {
                for l in 0..3 {
                    let mut sum = 0.0;
                    for j in 0..n_satom {
                        sum += fc[i * stride_i + j * 9 + k * 3 + l];
                    }
                    sum *= inv_n;
                    for j in 0..n_satom {
                        fc[i * stride_i + j * 9 + k * 3 + l] -= sum;
                    }
                }
            }
        }
        set_index_permutation_symmetry_fc(fc, n_satom);
    }
    set_translational_symmetry_fc(fc, n_satom);
}

/// Apply the index-permutation operation on compact fc2 (shape
/// `[n_patom, n_satom, 3, 3]`) in place.  Mirrors
/// `phpy_set_index_permutation_symmetry_compact_fc` in `c/phonopy.c`.
///
/// For each unordered pair of supercell atoms reachable from a
/// primitive-cell anchor, the kernel relates the (i_p, j) block to its
/// counterpart (j_p, i_trans) under the symmetry that maps `j` to its
/// primitive-cell representative `j'`.  When `transpose` is `true` the
/// two blocks are swapped (with a `(k, l)` -> `(l, k)` index swap);
/// when `transpose` is `false` they are averaged into the symmetric
/// part.  A scratch `done` flag array of length `n_patom * n_satom`
/// keeps the iteration over (i_p, j) from touching the same pair
/// twice.
///
/// Inputs:
///   * `fc`: in/out, length `n_patom * n_satom * 9`
///   * `p2s`: primitive -> supercell index map, length `n_patom`
///   * `s2pp`: supercell -> primitive index map, length `n_satom`
///   * `nsym_list`: per-supercell-atom symmetry index, length `n_satom`
///   * `perms`: atomic-permutation table, shape `[n_sym, n_satom]`
///     packed C order; only `perms[nsym_list[j] * n_satom + i]` is
///     read, so `n_sym` is implicit
pub(crate) fn set_index_permutation_symmetry_compact_fc(
    fc: &mut [f64],
    p2s: &[i64],
    s2pp: &[i64],
    nsym_list: &[i64],
    perms: &[i64],
    n_satom: usize,
    n_patom: usize,
    transpose: bool,
) {
    let stride_ip = n_satom * 9;
    let mut done = vec![false; n_patom * n_satom];

    for j in 0..n_satom {
        let j_p = s2pp[j] as usize;
        for i_p in 0..n_patom {
            let i = p2s[i_p] as usize;
            if i == j {
                // Diagonal block: handle the (k, l) <-> (l, k) swap on
                // the i_p-th on-site element (only the upper triangle
                // l > k is iterated, mirroring the C source).
                for k in 0..3 {
                    for l in (k + 1)..3 {
                        let m = i_p * stride_ip + i * 9 + k * 3 + l;
                        let n = i_p * stride_ip + i * 9 + l * 3 + k;
                        if transpose {
                            fc.swap(m, n);
                        } else {
                            let avg = (fc[m] + fc[n]) / 2.0;
                            fc[m] = avg;
                            fc[n] = avg;
                        }
                    }
                }
            }
            if !done[i_p * n_satom + j] {
                // (j, i) -- nsym_list[j] --> (j', i')
                // i' = perms[nsym_list[j] * n_satom + i]
                let i_trans = perms[(nsym_list[j] as usize) * n_satom + i] as usize;
                done[i_p * n_satom + j] = true;
                done[j_p * n_satom + i_trans] = true;
                for k in 0..3 {
                    for l in 0..3 {
                        let m = i_p * stride_ip + j * 9 + k * 3 + l;
                        let n = j_p * stride_ip + i_trans * 9 + l * 3 + k;
                        if transpose {
                            fc.swap(m, n);
                        } else {
                            let avg = (fc[m] + fc[n]) / 2.0;
                            fc[m] = avg;
                            fc[n] = avg;
                        }
                    }
                }
            }
        }
    }
}

/// On-site (`fc[i_p, p2s[i_p], :, :]`) acoustic-sum-rule enforcement
/// for compact fc2.  Mirrors the static helper
/// `set_translational_symmetry_compact_fc` in `c/phonopy.c`.
fn set_translational_symmetry_compact_fc(
    fc: &mut [f64],
    p2s: &[i64],
    n_satom: usize,
    n_patom: usize,
) {
    let stride_ip = n_satom * 9;
    for i_p in 0..n_patom {
        let mut sums = [[0.0f64; 3]; 3];
        for k in 0..3 {
            for l in 0..3 {
                let mut s = 0.0;
                for j in 0..n_satom {
                    if p2s[i_p] as usize != j {
                        s += fc[i_p * stride_ip + j * 9 + k * 3 + l];
                    }
                }
                sums[k][l] = s;
            }
        }
        let p_idx = p2s[i_p] as usize;
        for k in 0..3 {
            for l in 0..3 {
                fc[i_p * stride_ip + p_idx * 9 + k * 3 + l] = -(sums[k][l] + sums[l][k]) / 2.0;
            }
        }
    }
}

/// Symmetrize compact fc2 (shape `[n_patom, n_satom, 3, 3]`) by
/// repeated row-drift removal interleaved with permutation symmetry,
/// then enforce the acoustic sum rule on the on-site blocks.  Mirrors
/// `phpy_perm_trans_symmetrize_compact_fc` in `c/phonopy.c`.
///
/// At each `level` iteration the kernel runs the inner cycle twice:
///
///   transpose -> subtract row drift
///
/// then closes with the symmetric (averaging) variant of
/// `set_index_permutation_symmetry_compact_fc`.  Finally it writes the
/// translational-symmetry on-site element.
pub(crate) fn perm_trans_symmetrize_compact_fc(
    fc: &mut [f64],
    p2s: &[i64],
    s2pp: &[i64],
    nsym_list: &[i64],
    perms: &[i64],
    n_satom: usize,
    n_patom: usize,
    level: usize,
) {
    let stride_ip = n_satom * 9;
    let inv_n = 1.0 / (n_satom as f64);
    for _ in 0..level {
        for _n in 0..2 {
            // Transpose only.
            set_index_permutation_symmetry_compact_fc(
                fc, p2s, s2pp, nsym_list, perms, n_satom, n_patom, true,
            );
            // Subtract row drift on each (i_p, k, l) row of length n_satom.
            for i_p in 0..n_patom {
                for k in 0..3 {
                    for l in 0..3 {
                        let mut sum = 0.0;
                        for j in 0..n_satom {
                            sum += fc[i_p * stride_ip + j * 9 + k * 3 + l];
                        }
                        sum *= inv_n;
                        for j in 0..n_satom {
                            fc[i_p * stride_ip + j * 9 + k * 3 + l] -= sum;
                        }
                    }
                }
            }
        }
        // Symmetric variant (average), not transpose.
        set_index_permutation_symmetry_compact_fc(
            fc, p2s, s2pp, nsym_list, perms, n_satom, n_patom, false,
        );
    }
    set_translational_symmetry_compact_fc(fc, p2s, n_satom, n_patom);
}

/// Distribute fc2 onto symmetry-related atom-pair blocks.  Mirrors
/// the file-static `distribute_fc2` in `c/phonopy.c`.
///
/// `fc2` has shape `[fc_dim0, num_pos, 3, 3]` packed in C order.  For
/// every target atom in `atom_list` whose `map_atoms` entry differs
/// (i.e. it has a "done" sibling reachable via symmetry), the block
/// is filled by rotating the corresponding block at the done sibling
/// using the cartesian rotation `r_carts[map_syms[atom_todo]]`.
///
/// Inputs:
///   * `fc2`: in/out, length `fc_dim0 * num_pos * 9`
///   * `atom_list`: target atom indices, length `len_atom_list`
///   * `fc_indices_of_atom_list`: first-axis index for each target,
///     length `len_atom_list`; values must be `< fc_dim0`
///   * `r_carts`: cartesian rotation matrices, length `num_rot * 9`
///   * `permutations`: atomic permutations, shape `[num_rot, num_pos]`
///   * `map_atoms`: target -> done atom map, length `num_pos`
///   * `map_syms`: target -> symmetry index map, length `num_pos`
pub(crate) fn distribute_fc2(
    fc2: &mut [f64],
    atom_list: &[i64],
    fc_indices_of_atom_list: &[i64],
    r_carts: &[f64],
    permutations: &[i64],
    map_atoms: &[i64],
    map_syms: &[i64],
    num_pos: usize,
) {
    let len_atom_list = atom_list.len();
    let stride_first = num_pos * 9;

    // atom_list_reverse[atom_done] = i such that atom_list[i] == atom_done.
    // Only entries for which atom_done == atom_list[i] (i.e., atoms in the
    // done set) are defined; others are left as the placeholder -1.
    let mut atom_list_reverse = vec![-1i64; num_pos];
    for i in 0..len_atom_list {
        let atom_done = map_atoms[atom_list[i] as usize];
        if atom_done == atom_list[i] {
            atom_list_reverse[atom_done as usize] = i as i64;
        }
    }

    for i in 0..len_atom_list {
        let atom_todo = atom_list[i] as usize;
        let atom_done = map_atoms[atom_todo] as usize;
        let sym_index = map_syms[atom_todo] as usize;

        // Skip atoms that map to themselves -- those are the "done" atoms,
        // already filled by the caller.
        if atom_todo == atom_done {
            continue;
        }

        let r_cart_base = sym_index * 9;
        let perm_base = sym_index * num_pos;
        let done_first = fc_indices_of_atom_list[atom_list_reverse[atom_done] as usize] as usize;
        let todo_first = fc_indices_of_atom_list[i] as usize;

        for atom_other in 0..num_pos {
            let perm_other = permutations[perm_base + atom_other] as usize;
            let done_offset = done_first * stride_first + perm_other * 9;
            let todo_offset = todo_first * stride_first + atom_other * 9;
            // P' = R^T P R, with each 3x3 block evaluated independently.
            // Read the source 9 doubles into a local register-sized buffer
            // first so the borrow checker accepts the disjoint mutable
            // write below (source/target offsets cannot collide because
            // `atom_done != atom_todo`).
            let mut src = [0.0f64; 9];
            src.copy_from_slice(&fc2[done_offset..done_offset + 9]);
            for j in 0..3 {
                for k in 0..3 {
                    let mut acc = 0.0f64;
                    for l in 0..3 {
                        for m in 0..3 {
                            acc += r_carts[r_cart_base + l * 3 + j]
                                * r_carts[r_cart_base + m * 3 + k]
                                * src[l * 3 + m];
                        }
                    }
                    fc2[todo_offset + j * 3 + k] += acc;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny fc2 with arbitrary, asymmetric entries.
    fn make_fc(natom: usize) -> Vec<f64> {
        let mut fc = vec![0.0f64; natom * natom * 9];
        for i in 0..natom {
            for j in 0..natom {
                for k in 0..3 {
                    for l in 0..3 {
                        let v = ((i * 7 + j * 13 + k * 5 + l) as f64) * 0.01
                            - if i == j { 0.5 } else { 0.0 };
                        fc[i * natom * 9 + j * 9 + k * 3 + l] = v;
                    }
                }
            }
        }
        fc
    }

    #[test]
    fn permutation_symmetry_makes_fc_kl_symmetric() {
        let natom = 4;
        let mut fc = make_fc(natom);
        set_index_permutation_symmetry_fc(&mut fc, natom);
        for i in 0..natom {
            for j in 0..natom {
                for k in 0..3 {
                    for l in 0..3 {
                        let a = fc[i * natom * 9 + j * 9 + k * 3 + l];
                        let b = fc[j * natom * 9 + i * 9 + l * 3 + k];
                        assert!((a - b).abs() < 1e-15, "i={i} j={j} k={k} l={l} a={a} b={b}",);
                    }
                }
            }
        }
    }

    #[test]
    fn translational_symmetry_zeros_row_sums() {
        let natom = 4;
        let mut fc = make_fc(natom);
        set_translational_symmetry_fc(&mut fc, natom);
        // After enforcement: sum_j fc[i, j, k, l] over j (including j=i)
        // should equal -(sums_kl + sums_lk)/2 + sums_kl which is in
        // general non-zero because the helper only enforces the symmetric
        // part.  But the symmetric combination must sum to zero:
        //   sum_j ( fc[i, j, k, l] + fc[i, j, l, k] ) / 2 == 0
        for i in 0..natom {
            for k in 0..3 {
                for l in 0..3 {
                    let mut s = 0.0;
                    for j in 0..natom {
                        s += 0.5
                            * (fc[i * natom * 9 + j * 9 + k * 3 + l]
                                + fc[i * natom * 9 + j * 9 + l * 3 + k]);
                    }
                    assert!(s.abs() < 1e-13, "i={i} k={k} l={l} sum={s}");
                }
            }
        }
    }

    #[test]
    fn distribute_fc2_identity_does_nothing() {
        // 2 atoms, identity rotation, atom_list = [0, 1].  With identity
        // map_atoms = [0, 1] (each atom maps to itself), the function
        // should skip every target -- no fc2 element should change.
        let num_pos = 2;
        let mut fc2 = vec![0.0f64; num_pos * num_pos * 9];
        for (idx, slot) in fc2.iter_mut().enumerate() {
            *slot = (idx as f64) * 0.5 - 1.5;
        }
        let original = fc2.clone();
        let atom_list = [0i64, 1];
        let fc_indices = [0i64, 1];
        let r_carts = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let permutations = [0i64, 1];
        let map_atoms = [0i64, 1];
        let map_syms = [0i64, 0];
        distribute_fc2(
            &mut fc2,
            &atom_list,
            &fc_indices,
            &r_carts,
            &permutations,
            &map_atoms,
            &map_syms,
            num_pos,
        );
        assert_eq!(fc2, original);
    }

    #[test]
    fn distribute_fc2_inversion_negates_off_diagonal() {
        // 2 atoms, inversion symmetry maps atom 1 to atom 0 (so they are
        // siblings of each other).  We seed fc2[0, *] with values, leave
        // fc2[1, *] zero, and ask the kernel to fill atom 1's row from
        // atom 0's via the inversion (R = -I).
        // Under R = -I, the rotation of any 3x3 block by R^T P R reduces
        // to (-I)^T P (-I) = P (no sign change).  So fc2[1, perm[j]]
        // should equal fc2[0, j] after the kernel runs.
        let num_pos = 2;
        // fc2[0, 0]: identity 3x3
        // fc2[0, 1]: arbitrary
        let mut fc2 = vec![0.0f64; num_pos * num_pos * 9];
        let p00 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let p01 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        fc2[..9].copy_from_slice(&p00);
        fc2[9..18].copy_from_slice(&p01);
        let atom_list = [0i64, 1];
        let fc_indices = [0i64, 1];
        let r_carts = [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0];
        // Inversion permutes atom 0 <-> 1.
        let permutations = [1i64, 0];
        // map_atoms[0] = 0 (atom 0 is "done"); map_atoms[1] = 0 (atom 1
        // gets filled from atom 0).  map_syms[1] = 0 (use the only
        // rotation in r_carts).
        let map_atoms = [0i64, 0];
        let map_syms = [0i64, 0];
        distribute_fc2(
            &mut fc2,
            &atom_list,
            &fc_indices,
            &r_carts,
            &permutations,
            &map_atoms,
            &map_syms,
            num_pos,
        );
        // Expect fc2[1, 0] += fc2[0, 1] (since perm sends atom_other=0
        // to permutation[0]=1) and fc2[1, 1] += fc2[0, 0].
        let block_10 = &fc2[18..27];
        let block_11 = &fc2[27..36];
        assert_eq!(block_10, p01.as_slice());
        assert_eq!(block_11, p00.as_slice());
    }

    #[test]
    fn full_routine_idempotent_at_level_one() {
        let natom = 5;
        let mut fc1 = make_fc(natom);
        perm_trans_symmetrize_fc(&mut fc1, natom, 1);
        let mut fc2 = fc1.clone();
        perm_trans_symmetrize_fc(&mut fc2, natom, 1);
        for (a, b) in fc1.iter().zip(fc2.iter()) {
            assert!((a - b).abs() < 1e-12, "a={a} b={b}");
        }
    }
}
