# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- `grid_index_from_address`, which took a single address, is replaced by
  `grid_indices_from_addresses`, which takes addresses of shape `(n, 3)` and
  returns their GR grid-point indices.

## [0.4.0] - 2026-09-25

### Changed
- `integration_weights_at_grid_points`, `triplets_integration_weights`,
  `pp_collision`, `collision_at_grid_point`, and
  `collision_at_grid_points_batched` now accept `relative_grid_address` of
  shape `(24 * n, 4, 3)`, i.e., `n` concatenated sets of 24 tetrahedra, and
  average the tetrahedron-method weights over the sets. The previous
  `(24, 4, 3)` input is the `n = 1` case and behaves as before.

## [0.3.0] - 2026-06-27

### Added
- `eigvalsh_batch` kernel for diagonalizing a batch of Hermitian dynamical
  matrices. Each matrix is solved single-threaded while the batch is
  parallelized across cores, giving a large speedup over `numpy.linalg.eigh`
  for dense meshes of small matrices. Numerically a drop-in for
  `numpy.linalg.eigh`: eigenvalues are returned in ascending order,
  eigenvectors as columns, and the input matrices are left unmodified.
- `eigvalsh_values_batch` kernel: same as `eigvalsh_batch` but computes only
  the eigenvalues, skipping the eigenvectors and their output buffer. A
  drop-in for `numpy.linalg.eigvalsh`.

## [0.2.1] - 2026-06-11

### Added
- `fc4` (4th-order force constants) kernels

### Fixed
- Fix nonzero-Gamma point issue in `derivative_recip_dipole_dipole`

## [0.2.0] - 2026-05-25

### Added
- `derivative_recip_dipole_dipole` kernel for the q-derivative of the
  dipole-dipole part of the dynamical matrix in the Gonze-Lee formulation

## [0.1.2] - 2026-05-20

### Fixed
- Fix `RefCell` double-borrow panic in `pp_collision::with_scratch` under rayon
  work-stealing when `inner_par` is true; use take-and-put-back so the borrow
  is released before the closure runs.

## [0.1.1] - 2026-05-18

### Changed
- Set `gil_used = false` to release the GIL during Rust kernel execution

## [0.1.0] - 2026-05-10

Initial public release.  Provides the Rust kernel set used by
[phonopy](https://github.com/phonopy/phonopy) and
[phono3py](https://github.com/phonopy/phono3py).  Distributed as
`abi3-py310` wheels for Linux x86_64 / aarch64, macOS x86_64 /
arm64, and Windows x86_64.

[Unreleased]: https://github.com/phonopy/phonors/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/phonopy/phonors/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/phonopy/phonors/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/phonopy/phonors/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/phonopy/phonors/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/phonopy/phonors/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/phonopy/phonors/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/phonopy/phonors/releases/tag/v0.1.0
