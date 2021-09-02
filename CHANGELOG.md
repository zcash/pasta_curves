# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to Rust's notion of
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
