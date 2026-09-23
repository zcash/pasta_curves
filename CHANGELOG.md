# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to Rust's notion of
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- `zeroize` feature flag, which enables `impl zeroize::DefaultIsZeroes` for
  `Fp`, `Fq`, `Ep`, `EpAffine`, `Eq` and `EqAffine`. Zeroizing a field element
  sets it to zero; zeroizing a point sets it to the identity.
- `pasta_curves::glv_eisenstein` module, behind the new `glv-eisenstein`
  feature flag. This is an alternative recoding for the GLV split that
  `pasta_curves::glv` performs: instead of two independent width-4 wNAF digit
  strings it recodes the pair `(k1, k2)` as a single width-3 NAF over the
  Eisenstein integers `Z[w] = Z[X]/(X^2 + X + 1)`, whose unit group `mu_6` is
  exactly the six automorphisms of a `j`-invariant-0 curve. The 48 odd residue
  classes mod 8 fall into 8 free `mu_6`-orbits, so eight stored points reach
  every digit. That cuts the ladder from ~51.2 point additions to ~38.4, with a
  table costing 7 additions instead of 4, and guarantees at most one addition
  per ladder column. Like `glv`, it is variable-time in the scalar and so is
  for use where scalars are not secret. Verified against the curves by
  `sage/glv_eisenstein.sage`.

  Over a batch, `Table::batch` runs the seven-addition chain that builds the
  eight orbit representatives in affine coordinates too, with one field
  inversion shared across the batch per step, rather than projectively
  followed by a normalization of all `8 * n` entries. That makes the
  Eisenstein table build cheaper than the split-wNAF one (76us against 85us
  for 50 Pallas points) despite needing three more additions.

  The module also provides `batch_mul`, which multiplies many points by one
  shared scalar. Because the scalar is shared, every lane executes the same
  ladder column at the same time, so the accumulators can be kept in affine
  coordinates with a single field inversion shared across the whole batch per
  column, and a column's doubling and addition fuse into one
  Eisentrager-Lauter-Montgomery step. Results come back affine, which is what
  a key-agreement KDF needs anyway. Whether that beats the projective ladder
  depends on the cost of a field inversion relative to a multiplication, so
  the shared inversion is `VartimeField::invert_vartime`, the safegcd one.
  With it the measured crossover is a batch of about 96, which is what
  `BATCH_AFFINE_THRESHOLD` records and what `batch_mul` dispatches on; below
  it `batch_mul` uses the projective ladder, so it is never slower. Measured
  against the split-wNAF GLV ladder on Pallas, end to end, `batch_mul` is 9%
  faster at a batch of 16, 11% at 64, 13% at 128, 15% at 256 and 16% at 512.
  Against the constant-time Fermat inversion the affine ladder would not
  break even until a batch of about 420.
- `pasta_curves::{EpAffine, EqAffine}::from_xy_unchecked`, a `const`
  constructor that builds an affine point from coordinates without checking
  that it lies on the curve. It is intended for protocol constants and
  precomputed tables, which can now be written as `const` or `static` items
  instead of paying for `CurveAffine::from_xy` on every use.
- `pasta_curves::arithmetic`:
  - `VartimeField`, an extension trait for `ff::Field` that exposes
    variable-time operations. All trait methods have default impls that fall
    back on the constant-time implementations, but can be overriden for
    additional performance.
  - `VartimeBatchInvert`, a variable-time equivalent of `ff::BatchInvert`.
  - `impl VartimeField for pasta_curves::{Fp, Fq}`.

### Changed
- MSRV is now 1.85.0.
- Migrated to `ff 0.14`, `group 0.14`, `rand 0.10`.
- `pasta_curves::arithmetic`:
  - The `Base` and `ScalarExt` associated types of `CurveExt` and `CurveAffine`
    now have an additional `VartimeField` bound, enabling downstream generic
    code to use variable-time operations.
  - Changes to `CurveExt` trait:
    - Added `CurveExt::to_affine_vartime`
    - Added `CurveExt::batch_normalize_vartime`

## [0.5.2] - 2026-07-23
### Added
- `pasta_curves::deferred` module, behind the new `deferred` feature flag. This
  provides the `DeferredField` trait and a wide `Product` accumulator, which
  together allow summing many field multiplications (e.g. an inner product)
  with a single Montgomery reduction at the end.
- `pasta_curves::glv` module, behind the new `glv` feature flag. This provides
  variable-time scalar multiplication for Pallas and Vesta via their cube-root
  endomorphism, for use where scalars are not secret (e.g. in verifiers); its
  precomputations can be reused across multiplications that share a point or a
  scalar.

### Changed
- MSRV is now 1.63.0.

## [0.5.1] - 2023-03-02
### Fixed
- Fix a bug on 32-bit platforms that could cause the square root implementation
  to return an incorrect result.
- The `sqrt-table` feature now works without `std` and only requires `alloc`.

## [0.5.0] - 2022-12-06
### Added
- `serde` feature flag, which enables Serde compatibility to the crate types.
  Field elements and points are serialized to their canonical byte encoding
  (encoded as hexadecimal if the data format is human readable).

### Changed
- Migrated to `ff 0.13`, `group 0.13`, `ec-gpu 0.2`.
- `pasta_curves::arithmetic`:
  - `FieldExt` bounds on associated types of `CurveExt` and `CurveAffine` have
    been replaced by bounds on `ff::WithSmallOrderMulGroup<3>` (and `Ord` in the
    case of `CurveExt`).
- `pasta_curves::hashtocurve`:
  - `FieldExt` bounds on the module functions have been replaced by equivalent
    `ff` trait bounds.

### Removed
- `pasta_curves::arithmetic`:
  - `FieldExt` (use `ff::PrimeField` or `ff::WithSmallOrderMulGroup` instead).
  - `Group`
  - `SqrtRatio` (use `ff::Field::{sqrt_ratio, sqrt_alt}` instead).
  - `SqrtTables` (from public API, as it isn't suitable for generic usage).

## [0.4.1] - 2022-10-13
### Added
- `uninline-portable` feature flag, which disables inlining of some functions.
  This is useful for tiny microchips (such as ARM Cortex-M0), where inlining
  can hurt performance and blow up binary size.

## [0.4.0] - 2022-05-05
### Changed
- MSRV is now 1.56.0.
- Migrated to `ff 0.12`, `group 0.12`.

## [0.3.1] - 2022-04-20
### Added
- `gpu` feature flag, which exposes implementations of the `GpuField` trait from
  the `ec-gpu` crate for `pasta_curves::{Fp, Fq}`. This flag will eventually
  control all GPU functionality.
- `repr-c` feature flag, which helps to facilitate usage of this crate's types
  across FFI by conditionally adding `repr(C)` attribute to point structures.
- `pasta_curves::arithmetic::Coordinates::from_xy`

### Changed
- `pasta_curves::{Fp, Fq}` are now declared as `repr(transparent)`, to enable
  their use across FFI. They remain opaque structs in Rust code.

## [0.3.0] - 2022-01-03
### Added
- Support for `no-std` builds, via two new (default-enabled) feature flags:
  - `alloc` enables the `pasta_curves::arithmetic::{CurveAffine, CurveExt}`
    traits, as well as implementations of traits like `group::WnafGroup`.
  - `sqrt-table` depends on `alloc`, and enables the large precomputed tables
    (stored on the heap) that speed up square root computation.
- `pasta_curves::arithmetic::SqrtRatio` trait, extending `ff::PrimeField` with
  square roots of ratios. This trait is likely to be moved into the `ff` crate
  in a future release (once we're satisfied with it).

### Removed
- `pasta_curves::arithmetic`:
  - `Field` re-export (`pasta_curves::group::ff::Field` is equivalent).
  - `FieldExt::ROOT_OF_UNITY` (use `ff::PrimeField::root_of_unity` instead).
  - `FieldExt::{T_MINUS1_OVER2, pow_by_t_minus1_over2, get_lower_32, sqrt_alt,`
    `sqrt_ratio}` (moved to `SqrtRatio` trait).
  - `FieldExt::{RESCUE_ALPHA, RESCUE_INVALPHA}`
  - `FieldExt::from_u64` (use `From<u64> for ff::PrimeField` instead).
  - `FieldExt::{from_bytes, read, to_bytes, write}`
    (use `ff::PrimeField::{from_repr, to_repr}` instead).
  - `FieldExt::rand` (use `ff::Field::random` instead).
  - `CurveAffine::{read, write}`
    (use `group::GroupEncoding::{from_bytes, to_bytes}` instead).

## [0.2.1] - 2021-09-17
### Changed
- The crate is now licensed as `MIT OR Apache-2.0`.

## [0.2.0] - 2021-09-02
### Changed
- Migrated to `ff 0.11`, `group 0.11`.

## [0.1.2] - 2021-08-06
### Added
- Implementation of `group::WnafGroup` for Pallas and Vesta, enabling them to be
  used with `group::Wnaf` for targeted performance improvements.

## [0.1.1] - 2021-06-04
### Added
- Implementations of `group::cofactor::{CofactorCurve, CofactorCurveAffine}` for
  Pallas and Vesta, enabling them to be used in cofactor-aware protocols that
  also want to leverage the affine point representation.

## [0.1.0] - 2021-06-01
Initial release!
