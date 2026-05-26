//! Symmetry operations on 4th-order force constants (fc4).
//!
//! Companion to [`crate::fc3`] one rank higher. fc4 is invariant under any
//! simultaneous permutation of its four `(atom, Cartesian)` legs, so the
//! symmetrized fc4 is the mean over the 24 leg permutations.
//!
//! Both the full layout `(n_satom, n_satom, n_satom, n_satom, 3, 3, 3, 3)`
//! and the compact layout `(n_patom, n_satom, n_satom, n_satom, 3, 3, 3, 3)`
//! are supported. The compact routine maps supercell indices to primitive
//! slabs via `s2pp`, `p2s`, `nsym_list`, and `perms`, exactly as
//! [`crate::fc3::set_permutation_symmetry_compact_fc3`] does for fc3.
//!
//! The symmetrization routines are serial because the `done`
//! orbit-bookkeeping array is shared across orbits; `rotate_delta_fc3s`
//! parallelises over the output atom triples.

use std::f64::consts::PI;

use rayon::prelude::*;

use crate::common::{cmplx_mul, Cmplx, Vec3D};

/// The 24 permutations of the four leg positions `(0, 1, 2, 3)`.
const PERMS24: [[usize; 4]; 24] = [
    [0, 1, 2, 3],
    [0, 1, 3, 2],
    [0, 2, 1, 3],
    [0, 2, 3, 1],
    [0, 3, 1, 2],
    [0, 3, 2, 1],
    [1, 0, 2, 3],
    [1, 0, 3, 2],
    [1, 2, 0, 3],
    [1, 2, 3, 0],
    [1, 3, 0, 2],
    [1, 3, 2, 0],
    [2, 0, 1, 3],
    [2, 0, 3, 1],
    [2, 1, 0, 3],
    [2, 1, 3, 0],
    [2, 3, 0, 1],
    [2, 3, 1, 0],
    [3, 0, 1, 2],
    [3, 0, 2, 1],
    [3, 1, 0, 2],
    [3, 1, 2, 0],
    [3, 2, 0, 1],
    [3, 2, 1, 0],
];

/// Average a single permutation orbit over the 24 leg permutations and write
/// the mean back to all of them, in place.
///
/// `blocks[s]` is the flat atom-block index (each block holds 81 Cartesian
/// elements) reached by permutation `PERMS24[s]`. For permutation `sigma`, the
/// element at that block carries the base Cartesian indices reordered by the
/// same `sigma`. When atoms coincide, some `blocks` entries coincide; the
/// repeated reads/writes are consistent, so no special-casing is needed.
fn symmetrize_orbit(fc4: &mut [f64], blocks: &[usize; 24]) {
    let mut elem = [0.0f64; 81];
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                for d in 0..3 {
                    let cpos = [a, b, c, d];
                    let mut sum = 0.0;
                    for (s, sig) in PERMS24.iter().enumerate() {
                        let co =
                            cpos[sig[0]] * 27 + cpos[sig[1]] * 9 + cpos[sig[2]] * 3 + cpos[sig[3]];
                        sum += fc4[blocks[s] * 81 + co];
                    }
                    elem[a * 27 + b * 9 + c * 3 + d] = sum / 24.0;
                }
            }
        }
    }
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                for d in 0..3 {
                    let v = elem[a * 27 + b * 9 + c * 3 + d];
                    let cpos = [a, b, c, d];
                    for (s, sig) in PERMS24.iter().enumerate() {
                        let co =
                            cpos[sig[0]] * 27 + cpos[sig[1]] * 9 + cpos[sig[2]] * 3 + cpos[sig[3]];
                        fc4[blocks[s] * 81 + co] = v;
                    }
                }
            }
        }
    }
}

/// Enforce permutation symmetry on full fc4 in place.
///
/// `fc4` is flat with layout `(num_atom, num_atom, num_atom, num_atom, 3, 3,
/// 3, 3)`, i.e. `num_atom^4` blocks of 81.
pub(crate) fn set_permutation_symmetry_fc4(fc4: &mut [f64], num_atom: usize) {
    let n1 = num_atom;
    let n2 = n1 * n1;
    let n3 = n2 * n1;
    let mut done = vec![false; n1 * n3];

    for i in 0..n1 {
        for j in 0..n1 {
            for k in 0..n1 {
                for l in 0..n1 {
                    let atoms = [i, j, k, l];
                    let mut blocks = [0usize; 24];
                    for (s, sig) in PERMS24.iter().enumerate() {
                        blocks[s] = atoms[sig[0]] * n3
                            + atoms[sig[1]] * n2
                            + atoms[sig[2]] * n1
                            + atoms[sig[3]];
                    }
                    if blocks.iter().any(|&b| done[b]) {
                        continue;
                    }
                    for &b in &blocks {
                        done[b] = true;
                    }
                    symmetrize_orbit(fc4, &blocks);
                }
            }
        }
    }
}

/// Enforce permutation symmetry on compact fc4 in place.
///
/// `fc4` is flat with layout `(n_patom, n_satom, n_satom, n_satom, 3, 3, 3,
/// 3)`. For each permutation, the leading atom is mapped to its primitive
/// slab `s2pp[w]` and the other three atoms are translated by the operation
/// `nsym_list[w]` (via `perms`), so the orbit stays inside the compact array.
pub(crate) fn set_permutation_symmetry_compact_fc4(
    fc4: &mut [f64],
    p2s: &[i64],
    s2pp: &[i64],
    nsym_list: &[i64],
    perms: &[i64],
    n_satom: usize,
    n_patom: usize,
) {
    let n1 = n_satom;
    let n2 = n1 * n1;
    let n3 = n2 * n1;
    let mut done = vec![false; n_patom * n3];

    for i_p in 0..n_patom {
        let i = p2s[i_p] as usize;
        for j in 0..n_satom {
            for k in 0..n_satom {
                for l in 0..n_satom {
                    let atoms = [i, j, k, l];
                    let mut blocks = [0usize; 24];
                    for (s, sig) in PERMS24.iter().enumerate() {
                        let w = atoms[sig[0]];
                        let off = nsym_list[w] as usize * n_satom;
                        blocks[s] = s2pp[w] as usize * n3
                            + perms[off + atoms[sig[1]]] as usize * n2
                            + perms[off + atoms[sig[2]]] as usize * n1
                            + perms[off + atoms[sig[3]]] as usize;
                    }
                    if blocks.iter().any(|&b| done[b]) {
                        continue;
                    }
                    for &b in &blocks {
                        done[b] = true;
                    }
                    symmetrize_orbit(fc4, &blocks);
                }
            }
        }
    }
}

// Swap two 81-blocks with the Cartesian permutation (a, b, c, d) <-> (b, a,
// c, d), the leg-0/leg-1 transpose used by the dim-0/dim-1 swap.
fn swap_blocks_ab(fc4: &mut [f64], adrs: usize, adrs_t: usize) {
    let mut src = [0.0f64; 81];
    let mut dst = [0.0f64; 81];
    src.copy_from_slice(&fc4[adrs..adrs + 81]);
    dst.copy_from_slice(&fc4[adrs_t..adrs_t + 81]);
    if adrs != adrs_t {
        for a in 0..3 {
            for b in 0..3 {
                for c in 0..3 {
                    for d in 0..3 {
                        // fc4[adrs][a, b, c, d] = fc4[adrs_t][b, a, c, d]
                        fc4[adrs + a * 27 + b * 9 + c * 3 + d] = dst[b * 27 + a * 9 + c * 3 + d];
                    }
                }
            }
        }
    }
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                for d in 0..3 {
                    // fc4[adrs_t][b, a, c, d] = elem_src[a, b, c, d]
                    fc4[adrs_t + b * 27 + a * 9 + c * 3 + d] = src[a * 27 + b * 9 + c * 3 + d];
                }
            }
        }
    }
}

/// Transpose dims 0 and 1 of a compact fc4 in place.
///
/// `fc4` has layout `(n_patom, n_satom, n_satom, n_satom, 3, 3, 3, 3)`. This
/// is the only transpose needed by the compact acoustic-sum-rule recipe
/// (which then applies mean-subtraction directly on dims 1, 2, 3). `t_type`
/// mirrors [`crate::fc3::transpose_compact_fc3`]; only `t_type == 0` (swap dim
/// 0 <-> dim 1) is implemented, other values are no-ops. The swap crosses
/// primitive slabs via `s2pp`/`p2s`/`nsym_list`/`perms` and is serialised by a
/// `done` table.
pub(crate) fn transpose_compact_fc4(
    fc4: &mut [f64],
    p2s: &[i64],
    s2pp: &[i64],
    nsym_list: &[i64],
    perms: &[i64],
    n_satom: usize,
    n_patom: usize,
    t_type: i64,
) {
    if t_type != 0 {
        return;
    }
    let n1 = n_satom;
    let n2 = n1 * n1;
    let n3 = n2 * n1;
    let mut done = vec![false; n_satom * n_patom];
    for i_p in 0..n_patom {
        let i = p2s[i_p] as usize;
        for j in 0..n_satom {
            if done[i_p * n_satom + j] {
                continue;
            }
            let j_p = s2pp[j] as usize;
            let off = nsym_list[j] as usize * n_satom;
            let i_trans = perms[off + i] as usize;
            done[i_p * n_satom + j] = true;
            done[j_p * n_satom + i_trans] = true;
            for k in 0..n_satom {
                let k_trans = perms[off + k] as usize;
                for l in 0..n_satom {
                    let l_trans = perms[off + l] as usize;
                    let adrs = (i_p * n3 + j * n2 + k * n1 + l) * 81;
                    let adrs_t = (j_p * n3 + i_trans * n2 + k_trans * n1 + l_trans) * 81;
                    swap_blocks_ab(fc4, adrs, adrs_t);
                }
            }
        }
    }
}

/// Rotate one 81-element rank-4 Cartesian block: `out[a,b,c,d] = R[a,i] R[b,j]
/// R[c,k] R[d,l] tensor[i,j,k,l]` (row-major 3x3 `rot_cart`).
fn tensor4_rotation(out: &mut [f64; 81], tensor: &[f64], rot_cart: &[f64]) {
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                for d in 0..3 {
                    let mut sum = 0.0;
                    for i in 0..3 {
                        let ra = rot_cart[a * 3 + i];
                        for j in 0..3 {
                            let rab = ra * rot_cart[b * 3 + j];
                            for k in 0..3 {
                                let rabc = rab * rot_cart[c * 3 + k];
                                for l in 0..3 {
                                    sum += rabc
                                        * rot_cart[d * 3 + l]
                                        * tensor[i * 27 + j * 9 + k * 3 + l];
                                }
                            }
                        }
                    }
                    out[a * 27 + b * 9 + c * 3 + d] = sum;
                }
            }
        }
    }
}

/// Distribute fc4 from the `source` first-index row to the `target` row.
///
/// For each atom triple `(i, j, k)` the 81-element block at
/// `(target, i, j, k)` is written from the rotated block at
/// `(source, atom_mapping[i], atom_mapping[j], atom_mapping[k])`. `fc4` is flat
/// with layout `(rows, num_atom, num_atom, num_atom, 3, 3, 3, 3)`; `target` and
/// `source` index the first axis (supercell index for full fc4, compact slab
/// index for compact fc4). Rank-4 analogue of
/// [`crate::fc3::distribute_fc3`].
pub(crate) fn distribute_fc4(
    fc4: &mut [f64],
    target: usize,
    source: usize,
    atom_mapping: &[i64],
    num_atom: usize,
    rot_cart: &[f64],
) {
    let stride_k = 81;
    let stride_j = num_atom * stride_k;
    let stride_i = num_atom * stride_j;
    let stride_first = num_atom * stride_i;
    let src_base = source * stride_first;
    let tgt_base = target * stride_first;
    for i in 0..num_atom {
        let mi = atom_mapping[i] as usize;
        for j in 0..num_atom {
            let mj = atom_mapping[j] as usize;
            for k in 0..num_atom {
                let mk = atom_mapping[k] as usize;
                let adrs_out = tgt_base + i * stride_i + j * stride_j + k * stride_k;
                let adrs_in = src_base + mi * stride_i + mj * stride_j + mk * stride_k;
                let mut buf = [0.0f64; 81];
                tensor4_rotation(&mut buf, &fc4[adrs_in..adrs_in + 81], rot_cart);
                fc4[adrs_out..adrs_out + 81].copy_from_slice(&buf);
            }
        }
    }
}

/// Rotate displacement-difference fc3 tensors and project them onto fc4.
///
/// Rank-4 analogue of [`crate::fc3::rotate_delta_fc2s`]. Shapes:
/// * `fc4`: `(num_atom, num_atom, num_atom, 3, 3, 3, 3)` flat (one first-atom
///   slice; `num_atom^3` blocks of 81).
/// * `delta_fc3s`: `(num_disp, num_atom, num_atom, num_atom, 3, 3, 3)` flat.
/// * `inv_u`: `(3, num_disp * num_site_sym)`.
/// * `site_sym_cart`: `(num_site_sym, 3, 3)` row-major rotations.
/// * `rot_map_syms`: `(num_site_sym, num_atom)`.
///
/// For each atom triple `(i, j, k)`, the delta fc3 blocks at the
/// site-symmetry-mapped triples are rotated (rank-3 Cartesian rotation) and
/// stacked over `(disp, site_sym)`, then contracted with `inv_u` so that the
/// leading Cartesian index of fc4 carries the displacement-direction
/// derivative. Parallelised over the output triple since each writes its own
/// 81-block.
pub(crate) fn rotate_delta_fc3s(
    fc4: &mut [f64],
    delta_fc3s: &[f64],
    inv_u: &[f64],
    site_sym_cart: &[f64],
    rot_map_syms: &[i64],
    num_atom: usize,
    num_site_sym: usize,
    num_disp: usize,
) {
    let total = num_disp * num_site_sym;
    fc4.par_chunks_mut(81)
        .enumerate()
        .for_each(|(i_atoms, fc4_block)| {
            let i_a = i_atoms / (num_atom * num_atom);
            let j_a = (i_atoms / num_atom) % num_atom;
            let k_a = i_atoms % num_atom;
            let mut rot_delta = vec![0.0f64; total * 27];
            for i_d in 0..num_disp {
                for j_s in 0..num_site_sym {
                    let src_i = rot_map_syms[j_s * num_atom + i_a] as usize;
                    let src_j = rot_map_syms[j_s * num_atom + j_a] as usize;
                    let src_k = rot_map_syms[j_s * num_atom + k_a] as usize;
                    let src_base =
                        (((i_d * num_atom + src_i) * num_atom + src_j) * num_atom + src_k) * 27;
                    let dst_base = (i_d * num_site_sym + j_s) * 27;
                    let r = &site_sym_cart[j_s * 9..j_s * 9 + 9];
                    // tensor3_rotation: out[a,b,c] = sum r[a,i] r[b,j] r[c,k] T[i,j,k]
                    for a in 0..3 {
                        for b in 0..3 {
                            for c in 0..3 {
                                let mut s = 0.0;
                                for i in 0..3 {
                                    let ra = r[a * 3 + i];
                                    for j in 0..3 {
                                        let rab = ra * r[b * 3 + j];
                                        for k in 0..3 {
                                            s += rab
                                                * r[c * 3 + k]
                                                * delta_fc3s[src_base + i * 9 + j * 3 + k];
                                        }
                                    }
                                }
                                rot_delta[dst_base + a * 9 + b * 3 + c] = s;
                            }
                        }
                    }
                }
            }
            // fc4[i,j,k][lead, bcd] = sum_n inv_u[lead, n] * rot_delta[n, bcd]
            for lead in 0..3 {
                for bcd in 0..27 {
                    let mut s = 0.0;
                    for n in 0..total {
                        s += inv_u[lead * total + n] * rot_delta[n * 27 + bcd];
                    }
                    fc4_block[lead * 27 + bcd] = s;
                }
            }
        });
}

/// Summed phase factor `(1/count) sum_v exp(2 pi i q . v)` over the shortest
/// vectors of one (satom, patom) pair (dense svecs).
fn phase_factor_fc4(q: Vec3D, svecs: &[Vec3D], multi: [i64; 2]) -> Cmplx {
    let count = multi[0] as usize;
    let base = multi[1] as usize;
    let mut re = 0.0;
    let mut im = 0.0;
    for i in 0..count {
        let s = svecs[base + i];
        let (sn, cs) = ((q[0] * s[0] + q[1] * s[1] + q[2] * s[2]) * 2.0 * PI).sin_cos();
        re += cs;
        im += sn;
    }
    let inv = 1.0 / count as f64;
    [re * inv, im * inv]
}

/// Fourier transform fc4 to reciprocal space at a q-point quartet.
///
/// Output `fc4_reciprocal` is atom-first `(num_patom, num_patom, num_patom,
/// num_patom, 3, 3, 3, 3)` flat (num_patom^4 blocks of 81). `q4` is the quartet
/// of fractional q-points; only the last three legs enter the phase factors
/// (the first leg is the reference), matching the 2015 phono4py convention (no
/// pre-phase). `fc4` is full `(n_satom, ...)` or compact `(n_patom, ...)` per
/// `is_compact_fc`. Phase factors for all (patom, satom, leg) are precomputed
/// once; the per-block accumulation is parallelised over the output quartet.
#[allow(clippy::too_many_arguments)]
pub(crate) fn real_to_reciprocal_fc4(
    fc4_reciprocal: &mut [Cmplx],
    q4: &[Vec3D; 4],
    fc4: &[f64],
    is_compact_fc: bool,
    svecs: &[Vec3D],
    multiplicity: &[[i64; 2]],
    p2s_map: &[i64],
    s2p_map: &[i64],
    num_satom: usize,
    num_patom: usize,
) {
    let mut p1 = vec![[0.0f64; 2]; num_patom * num_satom];
    let mut p2 = vec![[0.0f64; 2]; num_patom * num_satom];
    let mut p3 = vec![[0.0f64; 2]; num_patom * num_satom];
    for i in 0..num_patom {
        for j in 0..num_satom {
            let multi = multiplicity[j * num_patom + i];
            p1[i * num_satom + j] = phase_factor_fc4(q4[1], svecs, multi);
            p2[i * num_satom + j] = phase_factor_fc4(q4[2], svecs, multi);
            p3[i * num_satom + j] = phase_factor_fc4(q4[3], svecs, multi);
        }
    }

    let ns2 = num_satom * num_satom;
    let ns3 = ns2 * num_satom;
    let np = num_patom;
    fc4_reciprocal
        .par_chunks_mut(81)
        .enumerate()
        .for_each(|(blk, out)| {
            let i = blk / (np * np * np);
            let jp = (blk / (np * np)) % np;
            let kp = (blk / np) % np;
            let lp = blk % np;
            let row_i = if is_compact_fc {
                i
            } else {
                p2s_map[i] as usize
            };
            let s2p_jp = p2s_map[jp];
            let s2p_kp = p2s_map[kp];
            let s2p_lp = p2s_map[lp];
            let mut re = [0.0f64; 81];
            let mut im = [0.0f64; 81];
            for j in 0..num_satom {
                if s2p_map[j] != s2p_jp {
                    continue;
                }
                let ph_j = p1[i * num_satom + j];
                for k in 0..num_satom {
                    if s2p_map[k] != s2p_kp {
                        continue;
                    }
                    let ph_jk = cmplx_mul(ph_j, p2[i * num_satom + k]);
                    for l in 0..num_satom {
                        if s2p_map[l] != s2p_lp {
                            continue;
                        }
                        let ph = cmplx_mul(ph_jk, p3[i * num_satom + l]);
                        let addr = (row_i * ns3 + j * ns2 + k * num_satom + l) * 81;
                        let block = &fc4[addr..addr + 81];
                        for m in 0..81 {
                            re[m] += ph[0] * block[m];
                            im[m] += ph[1] * block[m];
                        }
                    }
                }
            }
            for m in 0..81 {
                out[m] = [re[m], im[m]];
            }
        });
}

/// Contract a reciprocal-space fc4 with eigenvectors to normal coordinates.
///
/// Computes, for every band `s'` at the second grid point,
/// `fc4_normal[s'] = (1/(f1 f2[s'])) sum_{ijkl,mnpq} a[i,m] b[j,n] c[k,p]
/// d[l,q] fc4_reciprocal[i,j,k,l,m,n,p,q]`, where `a = e1*.conj/sqrt(m)`,
/// `b = e1/sqrt(m)`, `c = e2/sqrt(m)`, `d = e2*.conj/sqrt(m)` (the 2015
/// `_sum_in_atoms` convention). The first two legs (the fixed `e1` band) are
/// contracted once into `U[k,l,p,q]`, then each `e2` band is contracted with
/// `U`, so all output bands are produced in one call.
///
/// Shapes (flat): `fc4_normal` (num_band,); `fc4_reciprocal`
/// (num_patom^4 * 81); `e1` (num_band,); `e2` (num_band, num_band) row-major
/// with band index last (`e2[comp*num_band + band]`); `f2` (num_band,);
/// `inv_sqrt_masses` (num_patom,).
#[allow(clippy::too_many_arguments)]
pub(crate) fn reciprocal_to_normal_fc4(
    fc4_normal: &mut [Cmplx],
    fc4_reciprocal: &[Cmplx],
    e1: &[Cmplx],
    e2: &[Cmplx],
    f1: f64,
    f2: &[f64],
    inv_sqrt_masses: &[f64],
    cutoff_frequency: f64,
) {
    let n = inv_sqrt_masses.len();
    let num_band = n * 3;
    for v in fc4_normal.iter_mut() {
        *v = [0.0, 0.0];
    }
    if f1 <= cutoff_frequency {
        return;
    }

    // a[i,m] = conj(e1[i,m]) / sqrt(m_i); b[j,n] = e1[j,n] / sqrt(m_j).
    let mut a = vec![[0.0f64; 2]; num_band];
    let mut b = vec![[0.0f64; 2]; num_band];
    for i in 0..n {
        let w = inv_sqrt_masses[i];
        for m in 0..3 {
            let e = e1[i * 3 + m];
            a[i * 3 + m] = [e[0] * w, -e[1] * w];
            b[i * 3 + m] = [e[0] * w, e[1] * w];
        }
    }

    // U[k,l,p,q] = sum_{i,m,j,n} a[i,m] b[j,n] fc4_reciprocal[i,j,k,l,m,n,p,q].
    let mut u = vec![[0.0f64; 2]; n * n * 9];
    for i in 0..n {
        for m in 0..3 {
            let aim = a[i * 3 + m];
            for j in 0..n {
                for nn in 0..3 {
                    let ab = cmplx_mul(aim, b[j * 3 + nn]);
                    for k in 0..n {
                        for l in 0..n {
                            let base = (((i * n + j) * n + k) * n + l) * 81 + m * 27 + nn * 9;
                            let ubase = (k * n + l) * 9;
                            for pq in 0..9 {
                                let prod = cmplx_mul(ab, fc4_reciprocal[base + pq]);
                                u[ubase + pq][0] += prod[0];
                                u[ubase + pq][1] += prod[1];
                            }
                        }
                    }
                }
            }
        }
    }

    // Per second-grid-point band: contract U with c (e2) and d (e2*).
    for band in 0..num_band {
        if f2[band] <= cutoff_frequency {
            continue;
        }
        let mut acc = [0.0f64; 2];
        for k in 0..n {
            let wk = inv_sqrt_masses[k];
            for p in 0..3 {
                let ce = e2[(k * 3 + p) * num_band + band];
                let ckp = [ce[0] * wk, ce[1] * wk];
                for l in 0..n {
                    let wl = inv_sqrt_masses[l];
                    for q in 0..3 {
                        let de = e2[(l * 3 + q) * num_band + band];
                        let dlq = [de[0] * wl, -de[1] * wl];
                        let cd = cmplx_mul(ckp, dlq);
                        let t = cmplx_mul(cd, u[(k * n + l) * 9 + p * 3 + q]);
                        acc[0] += t[0];
                        acc[1] += t[1];
                    }
                }
            }
        }
        let inv = 1.0 / (f1 * f2[band]);
        fc4_normal[band] = [acc[0] * inv, acc[1] * inv];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seeded(size: usize, mult: usize, modulo: usize, shift: f64) -> Vec<f64> {
        (0..size)
            .map(|i| ((i * mult) % modulo) as f64 - shift)
            .collect()
    }

    #[test]
    fn set_permutation_symmetry_fc4_idempotent() {
        let n = 2;
        let mut fc4 = seeded(n * n * n * n * 81, 17, 101, 50.0);
        set_permutation_symmetry_fc4(&mut fc4, n);
        let snapshot = fc4.clone();
        set_permutation_symmetry_fc4(&mut fc4, n);
        for (a, b) in snapshot.iter().zip(fc4.iter()) {
            assert!((a - b).abs() < 1e-13);
        }
    }

    #[test]
    fn set_permutation_symmetry_fc4_enforces_all_permutations() {
        let n = 2;
        let n1 = n;
        let n2 = n1 * n1;
        let n3 = n2 * n1;
        let mut fc4 = seeded(n * n * n * n * 81, 13, 97, 48.0);
        set_permutation_symmetry_fc4(&mut fc4, n);
        // For every quadruple and Cartesian set, all 24 permuted entries equal.
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    for l in 0..n {
                        let atoms = [i, j, k, l];
                        for a in 0..3 {
                            for b in 0..3 {
                                for c in 0..3 {
                                    for d in 0..3 {
                                        let cpos = [a, b, c, d];
                                        let base = (i * n3 + j * n2 + k * n1 + l) * 81
                                            + a * 27
                                            + b * 9
                                            + c * 3
                                            + d;
                                        let ref_val = fc4[base];
                                        for sig in PERMS24.iter() {
                                            let blk = atoms[sig[0]] * n3
                                                + atoms[sig[1]] * n2
                                                + atoms[sig[2]] * n1
                                                + atoms[sig[3]];
                                            let co = cpos[sig[0]] * 27
                                                + cpos[sig[1]] * 9
                                                + cpos[sig[2]] * 3
                                                + cpos[sig[3]];
                                            assert!((fc4[blk * 81 + co] - ref_val).abs() < 1e-13);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn set_permutation_symmetry_compact_fc4_idempotent() {
        // Trivial symmetry (n_patom == n_satom, identity perms).
        let n = 3;
        let p2s: Vec<i64> = (0..n as i64).collect();
        let s2pp = p2s.clone();
        let nsym_list = vec![0i64; n];
        let perms: Vec<i64> = (0..n as i64).collect();
        let mut fc4 = seeded(n * n * n * n * 81, 19, 89, 44.0);
        set_permutation_symmetry_compact_fc4(&mut fc4, &p2s, &s2pp, &nsym_list, &perms, n, n);
        let snapshot = fc4.clone();
        set_permutation_symmetry_compact_fc4(&mut fc4, &p2s, &s2pp, &nsym_list, &perms, n, n);
        for (a, b) in snapshot.iter().zip(fc4.iter()) {
            assert!((a - b).abs() < 1e-13);
        }
    }

    #[test]
    fn set_permutation_symmetry_compact_fc4_matches_full_under_trivial_symmetry() {
        let n = 3;
        let p2s: Vec<i64> = (0..n as i64).collect();
        let s2pp = p2s.clone();
        let nsym_list = vec![0i64; n];
        let perms: Vec<i64> = (0..n as i64).collect();

        let base = seeded(n * n * n * n * 81, 23, 97, 48.0);
        let mut fc4_full = base.clone();
        let mut fc4_compact = base.clone();
        set_permutation_symmetry_fc4(&mut fc4_full, n);
        set_permutation_symmetry_compact_fc4(
            &mut fc4_compact,
            &p2s,
            &s2pp,
            &nsym_list,
            &perms,
            n,
            n,
        );
        for (a, b) in fc4_full.iter().zip(fc4_compact.iter()) {
            assert!((a - b).abs() < 1e-13);
        }
    }

    #[test]
    fn transpose_compact_fc4_dim01_is_involution() {
        // Trivial symmetry (n_patom == n_satom, identity perms): applying the
        // dim-0 <-> dim-1 transpose twice returns the original array.
        let n = 3;
        let p2s: Vec<i64> = (0..n as i64).collect();
        let s2pp = p2s.clone();
        let nsym_list = vec![0i64; n];
        let perms: Vec<i64> = (0..n as i64).collect();
        let mut fc4 = seeded(n * n * n * n * 81, 11, 73, 36.0);
        let original = fc4.clone();
        transpose_compact_fc4(&mut fc4, &p2s, &s2pp, &nsym_list, &perms, n, n, 0);
        transpose_compact_fc4(&mut fc4, &p2s, &s2pp, &nsym_list, &perms, n, n, 0);
        for (a, b) in original.iter().zip(fc4.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn transpose_compact_fc4_dim01_matches_full_swap_under_trivial_symmetry() {
        // Under trivial symmetry the compact transpose must equal the explicit
        // full dim-0 <-> dim-1 swap: new[i,j,k,l][a,b,c,d] = old[j,i,k,l][b,a,c,d].
        let n = 3;
        let n1 = n;
        let n2 = n1 * n1;
        let n3 = n2 * n1;
        let p2s: Vec<i64> = (0..n as i64).collect();
        let s2pp = p2s.clone();
        let nsym_list = vec![0i64; n];
        let perms: Vec<i64> = (0..n as i64).collect();

        let original = seeded(n * n * n * n * 81, 29, 83, 41.0);
        let mut expected = vec![0.0f64; original.len()];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    for l in 0..n {
                        for a in 0..3 {
                            for b in 0..3 {
                                for c in 0..3 {
                                    for d in 0..3 {
                                        expected[(i * n3 + j * n2 + k * n1 + l) * 81
                                            + a * 27
                                            + b * 9
                                            + c * 3
                                            + d] = original[(j * n3 + i * n2 + k * n1 + l) * 81
                                            + b * 27
                                            + a * 9
                                            + c * 3
                                            + d];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let mut fc4 = original.clone();
        transpose_compact_fc4(&mut fc4, &p2s, &s2pp, &nsym_list, &perms, n, n, 0);
        for (a, b) in expected.iter().zip(fc4.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn distribute_fc4_identity_copies_with_mapping() {
        // Identity rotation + identity mapping: target row becomes a copy of
        // the source row.
        let n = 3;
        let mut fc4 = seeded(n * n * n * n * 81, 7, 67, 33.0);
        let mapping: Vec<i64> = (0..n as i64).collect();
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        distribute_fc4(&mut fc4, 1, 0, &mapping, n, &identity);
        let row = n * n * n * 81;
        for off in 0..row {
            assert!((fc4[row + off] - fc4[off]).abs() < 1e-15);
        }
    }

    #[test]
    fn distribute_fc4_rotation_matches_manual() {
        // A 90-degree rotation about z: x->y, y->-x, z->z.
        let n = 1;
        let mut fc4 = seeded(81, 5, 47, 23.0);
        let source = fc4[0..81].to_vec();
        let rot = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let mapping = [0i64];
        distribute_fc4(&mut fc4, 0, 0, &mapping, n, &rot);
        // Manual rank-4 rotation of the source block.
        let mut expected = [0.0f64; 81];
        for a in 0..3 {
            for b in 0..3 {
                for c in 0..3 {
                    for d in 0..3 {
                        let mut sum = 0.0;
                        for i in 0..3 {
                            for j in 0..3 {
                                for k in 0..3 {
                                    for l in 0..3 {
                                        sum += rot[a * 3 + i]
                                            * rot[b * 3 + j]
                                            * rot[c * 3 + k]
                                            * rot[d * 3 + l]
                                            * source[i * 27 + j * 9 + k * 3 + l];
                                    }
                                }
                            }
                        }
                        expected[a * 27 + b * 9 + c * 3 + d] = sum;
                    }
                }
            }
        }
        for off in 0..81 {
            assert!((fc4[off] - expected[off]).abs() < 1e-12);
        }
    }

    #[test]
    fn rotate_delta_fc3s_identity_symmetry_is_projection() {
        // num_site_sym = 1 (identity rotation), num_disp = 1: rot_delta equals
        // the delta fc3 block, and fc4[i,j,k][lead,bcd] = inv_u[lead,0] *
        // delta_fc3s[0,i,j,k][bcd].
        let num_atom = 2;
        let num_site_sym = 1;
        let num_disp = 1;
        let total = num_disp * num_site_sym;
        let delta_fc3s = seeded(num_disp * num_atom * num_atom * num_atom * 27, 3, 53, 26.0);
        let inv_u = [0.25f64, -0.5, 2.0]; // (3, total=1)
        let site_sym_cart = [1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let rot_map_syms: Vec<i64> = (0..num_atom as i64).collect();
        let mut fc4 = vec![0.0f64; num_atom * num_atom * num_atom * 81];
        rotate_delta_fc3s(
            &mut fc4,
            &delta_fc3s,
            &inv_u,
            &site_sym_cart,
            &rot_map_syms,
            num_atom,
            num_site_sym,
            num_disp,
        );
        for i in 0..num_atom {
            for j in 0..num_atom {
                for k in 0..num_atom {
                    let blk = ((i * num_atom + j) * num_atom + k) * 81;
                    // delta_fc3s displacement 0: ((i*n + j)*n + k) block of 27.
                    let src = ((i * num_atom + j) * num_atom + k) * 27;
                    for lead in 0..3 {
                        for bcd in 0..27 {
                            let expected = inv_u[lead * total] * delta_fc3s[src + bcd];
                            assert!((fc4[blk + lead * 27 + bcd] - expected).abs() < 1e-13);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn reciprocal_to_normal_fc4_matches_bruteforce() {
        let n = 2;
        let num_band = n * 3;
        let len = n * n * n * n * 81;
        let re = seeded(len, 7, 101, 50.0);
        let im = seeded(len, 5, 97, 48.0);
        let fc4r: Vec<Cmplx> = (0..len).map(|i| [re[i], im[i]]).collect();
        let e1re = seeded(num_band, 3, 29, 14.0);
        let e1im = seeded(num_band, 11, 31, 15.0);
        let e1: Vec<Cmplx> = (0..num_band).map(|i| [e1re[i], e1im[i]]).collect();
        let e2re = seeded(num_band * num_band, 13, 41, 20.0);
        let e2im = seeded(num_band * num_band, 17, 43, 21.0);
        let e2: Vec<Cmplx> = (0..num_band * num_band)
            .map(|i| [e2re[i], e2im[i]])
            .collect();
        let f1 = 2.0;
        let f2 = vec![1.5; num_band];
        let inv_sqrt_m = [0.5f64, 0.8];
        let mut out = vec![[0.0f64; 2]; num_band];
        reciprocal_to_normal_fc4(&mut out, &fc4r, &e1, &e2, f1, &f2, &inv_sqrt_m, 1e-4);

        for band in 0..num_band {
            let mut acc = [0.0f64; 2];
            for i in 0..n {
                for m in 0..3 {
                    let aim = [
                        e1[i * 3 + m][0] * inv_sqrt_m[i],
                        -e1[i * 3 + m][1] * inv_sqrt_m[i],
                    ];
                    for j in 0..n {
                        for nn in 0..3 {
                            let bjn = [
                                e1[j * 3 + nn][0] * inv_sqrt_m[j],
                                e1[j * 3 + nn][1] * inv_sqrt_m[j],
                            ];
                            for k in 0..n {
                                for p in 0..3 {
                                    let ce = e2[(k * 3 + p) * num_band + band];
                                    let ckp = [ce[0] * inv_sqrt_m[k], ce[1] * inv_sqrt_m[k]];
                                    for l in 0..n {
                                        for q in 0..3 {
                                            let de = e2[(l * 3 + q) * num_band + band];
                                            let dlq =
                                                [de[0] * inv_sqrt_m[l], -de[1] * inv_sqrt_m[l]];
                                            let base = (((i * n + j) * n + k) * n + l) * 81
                                                + m * 27
                                                + nn * 9
                                                + p * 3
                                                + q;
                                            let t = cmplx_mul(
                                                cmplx_mul(cmplx_mul(aim, bjn), cmplx_mul(ckp, dlq)),
                                                fc4r[base],
                                            );
                                            acc[0] += t[0];
                                            acc[1] += t[1];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            let inv = 1.0 / (f1 * f2[band]);
            let r0 = acc[0] * inv;
            let r1 = acc[1] * inv;
            // Relative tolerance: the factorized kernel sums in a different
            // order than this brute force, and the magnitudes here are large.
            assert!((out[band][0] - r0).abs() < 1e-9 * (r0.abs() + 1.0));
            assert!((out[band][1] - r1).abs() < 1e-9 * (r1.abs() + 1.0));
        }
    }

    #[test]
    fn real_to_reciprocal_fc4_single_atom_is_identity() {
        // 1-atom cell: the only shortest vector is the origin, so every phase
        // factor is 1 at any q and fc4_reciprocal == fc4 (real).
        let fc4: Vec<f64> = (0..81).map(|v| v as f64 - 40.0).collect();
        let svecs = [[0.0f64, 0.0, 0.0]];
        let multi = [[1i64, 0]];
        let p2s = [0i64];
        let s2p = [0i64];
        let q4 = [
            [0.1, 0.2, 0.3],
            [0.4, 0.0, 0.1],
            [0.0, 0.3, 0.2],
            [0.2, 0.1, 0.0],
        ];
        let mut rec = vec![[0.0f64; 2]; 81];
        real_to_reciprocal_fc4(&mut rec, &q4, &fc4, false, &svecs, &multi, &p2s, &s2p, 1, 1);
        for m in 0..81 {
            assert!((rec[m][0] - fc4[m]).abs() < 1e-13);
            assert!(rec[m][1].abs() < 1e-13);
        }
    }
}
