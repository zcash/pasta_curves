# SageMath derivations of the crate's constants

The scripts in this directory recompute, from the curve definitions
alone, the magic constants that appear in the crate's source, printing
them in the exact shape of the Rust code so the two can be diffed:

- `glv_constants.sage` — the GLV short-basis and Babai-rounding
  constants in the `GlvParams` implementations in `src/glv.rs`
  (`V1A`, `V1B_NEG`, `V2A`, `V2B`, `G1`, `G2`, for both curves).
- `glv_boundary_scalars.sage` — the `*_BOUNDARY_SCALAR` witnesses used
  by the `babai_boundary_*` and `native_vs_glv_boundary_*` regression
  tests in `src/glv.rs`.
- `divstep_lower_bounds.sage` — the `slow_inversions` known answers in
  `src/test_fields.rs` and the witnesses of
  `lean/PastaCurves/Inversion/LowerBound.lean`: inputs on which the
  constant-time inversion's half-delta divsteps run long (531 and 554
  steps for Pallas, 530 and 555 for Vesta), found by a seeded search.
  Each witness is a lower bound on the worst case for its modulus; the
  upper bound, 588, is the hull certificate's.

The scripts use exact integer/rational arithmetic and hand-rolled
lattice reduction only, so their output is deterministic and does not
depend on the SageMath version.

## Running

Sage is pinned via [uv](https://docs.astral.sh/uv/) using
[passagemath](https://pypi.org/project/passagemath-standard/), the
pip-installable distribution of SageMath (see `pyproject.toml` /
`uv.lock`). From this directory:

```console
$ uv run sage glv_constants.sage
$ uv run sage glv_boundary_scalars.sage
$ uv run sage divstep_lower_bounds.sage
```

Any reasonably recent standalone SageMath installation works too:
`sage glv_constants.sage`.
