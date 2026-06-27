# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `eigvalsh_batch` kernel for diagonalizing a batch of Hermitian dynamical
  matrices. Each matrix is solved single-threaded while the batch is
  parallelized across cores, giving a large speedup over `numpy.linalg.eigh`
  for dense meshes of small matrices. Numerically a drop-in for
  `numpy.linalg.eigh`: eigenvalues are returned in ascending order,
  eigenvectors as columns, and the input matrices are left unmodified.

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

[Unreleased]: https://github.com/phonopy/phonors/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/phonopy/phonors/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/phonopy/phonors/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/phonopy/phonors/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/phonopy/phonors/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/phonopy/phonors/releases/tag/v0.1.0
