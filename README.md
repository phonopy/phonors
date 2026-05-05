# phonors

Rust kernels for phonopy and phono3py.

`phonors` is a Rust extension module providing the heavy numerical
kernels used by [phono3py](https://github.com/phonopy/phono3py)
(via the `phono3py[rust]` extra) and intended to also back
[phonopy](https://github.com/phonopy/phonopy) as its C kernels are
ported.  It is built with [maturin](https://www.maturin.rs/) and
[PyO3](https://pyo3.rs/), distributed as `abi3-py310` wheels
(Python 3.10+).

## Installation

```bash
maturin develop --release
```

### Optional: native CPU tuning

By default, `maturin develop --release` builds with the Rust baseline
target (x86-64 v1 on x86_64, Armv8.0 on aarch64), so the resulting
module runs on any CPU of that architecture.  For a local build that
will only run on the current machine, enabling the host CPU's full
instruction set can recover a few percent of wall-clock:

```bash
RUSTFLAGS='-C target-cpu=native' maturin develop --release
```

## License

BSD-3-Clause.  See [LICENSE](LICENSE).
