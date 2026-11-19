# SageMath derivations of the crate's constants

The scripts in this directory recompute, from the curve definitions
alone, the magic constants that appear in the crate's source, printing
them in the exact shape of the Rust code so the two can be diffed:

- `glv_constants.sage` — the GLV short-basis and Babai-rounding
  constants in the `GlvConstants` implementations in `src/glv.rs`
  (`BASIS_1_A`, `BASIS_1_B_ABS`, `BASIS_2_A`, `BASIS_2_B`,
  `ROUNDING_COEFF_1`, `ROUNDING_COEFF_2`, for both curves).
- `glv_boundary_scalars.sage` — the `*_BOUNDARY_SCALAR` witnesses used
  by the `babai_boundary_*` and `native_vs_glv_boundary_*` regression
  tests in `src/glv.rs`.
- `glv_test_vectors.sage` - known-answer vectors for both recodings:
  `k * G` on Pallas and Vesta for a fixed set of scalars, printed as the
  `PALLAS_VECTORS` and `VESTA_VECTORS` tables in the `vectors` test
  module of `src/glv.rs`, which `src/glv_eisenstein.rs` reuses.
- `glv_eisenstein.sage`: the Eisenstein-integer recoding in
  `src/glv_eisenstein.rs`: the ring facts it rests on (2 inert in
  `Z[w]`, unit group `mu_6`), the free action of `mu_6` on the 48 odd
  residue classes of `Z[w]/8` that reduces the table to 8 orbit
  representatives, the `REPS` constant, the seven-addition chain that
  builds them and the proof that it has no exceptional case in affine
  coordinates, the `MAX_DIGITS` bound, the recoder itself, checked
  end to end by running the resulting ladder on Pallas and Vesta, and the
  norm bound showing that the batch-affine ladder has no exceptional case
  but the identity.
- `msm_window_thresholds.sage`: the `DEFAULT_MULTIEXP_WINDOW_THRESHOLDS`
  table in `src/glv/msm.rs`, the floors of `e^k` that encode the generic
  MSM's `ceil(ln(terms))` window schedule. It computes them with interval
  arithmetic and checks each floor is exact.

The scripts use exact integer/rational arithmetic, hand-rolled lattice
reduction and verified interval bounds. Their output is deterministic.

## Running

Sage is pinned via [uv](https://docs.astral.sh/uv/) using
[passagemath](https://pypi.org/project/passagemath-standard/), the
pip-installable distribution of SageMath (see `pyproject.toml` /
`uv.lock`). From this directory:

```console
$ uv run sage glv_constants.sage
$ uv run sage glv_boundary_scalars.sage
$ uv run sage glv_eisenstein.sage
$ uv run sage msm_window_thresholds.sage
```

Any reasonably recent standalone SageMath installation works too:
`sage glv_constants.sage`.
