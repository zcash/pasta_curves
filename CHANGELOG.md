# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to Rust's notion of
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- `pasta_curves::glv_eisenstein` module, behind the new `glv-eisenstein`
  feature flag. An alternative recoding for the same GLV split that
  `pasta_curves::glv` performs: the pair `(k1, k2)` becomes a single width-3
  NAF over the Eisenstein integers `Z[w] = Z[X]/(X^2 + X + 1)` rather than two
  independent width-4 wNAFs. The unit group `mu_6` is exactly the six
  automorphisms of a `j`-invariant-0 curve and acts freely on the 48 odd
  residue classes mod 8, so eight stored points reach every digit. The ladder
  drops from ~51.2 point additions to ~38.4, at most one per column, for a
  table costing 7 additions instead of 4. Like `glv` it is variable-time in
  the scalar, so only for scalars that are not secret. Verified against the
  curves by `sage/glv_eisenstein.sage`.

  `Table::batch`, and `batch_mul` for many points against one shared scalar,
  both work in affine coordinates with a single field inversion shared per
  step instead of normalizing afterwards; `batch_mul` additionally fuses each
  column's doubling and addition into one Eisentrager-Lauter-Montgomery step
  and returns affine results. The inversion is
  `VartimeField::invert_vartime`, and `batch_mul` falls back to the
  projective ladder below `BATCH_AFFINE_THRESHOLD`, so it is never slower;
  `Table::batch_mul_affine` takes the affine ladder whatever the size. A
  scalar recoded once is a `Recoded`, which may be reused across points.
- One free function per call shape, in both `pasta_curves::glv` and
  `pasta_curves::glv_eisenstein`, so neither module asks the caller to
  assemble the precomputation:
  - `mul`, one point against one scalar.
  - `batch_mul`, many points against one shared scalar, building the
    per-point tables with a single field inversion.
  - `mul_scalars`, one point against many scalars, building the table once
    and sharing one inversion back to affine.
  - `mul_pairs`, many points against many scalars, sharing one inversion to
    build the tables and another back to affine.

### Changed
- `pasta_curves::arithmetic`:
  - Changes to `CurveExt` trait:
    - The `AffineExt` associated type now additionally requires
      `Base = <Self as CurveExt>::Base`, so generic code can pass coordinates
      between a curve's projective and affine forms. Every real curve already
      satisfies this; it was simply never stated.
  - Changes to `CurveAffine` trait:
    - Added `CurveAffine::from_xy_unchecked`, the trait counterpart of the
      inherent constructor of the same name, alongside the existing
      `CurveAffine::from_xy`. Generic code building points from coordinates it
      has already validated cannot afford `from_xy`: the on-curve check costs
      about 85ns against 4ns, which over a windowed ladder's digit lookups is
      larger than the saving the window buys.

## [0.6.0] - 2026-09-25
### Added
- `zeroize` feature flag, which enables `impl zeroize::DefaultIsZeroes` for
  `Fp`, `Fq`, `Ep`, `EpAffine`, `Eq` and `EqAffine`. Zeroizing a field element
  sets it to zero; zeroizing a point sets it to the identity.
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
