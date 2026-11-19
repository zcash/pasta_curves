//! The calibration given at build time by `PASTA_CURVES_CALIBRATION`, with
//! `--cfg pasta_curves_calibration`. The variable lists comma-separated
//! `field=value` pairs, optionally starting with `base=<name>` (default:
//! `default`); every field not listed keeps the base's value:
//!
//! ```text
//! base=default,min_glv_multiexp_terms=64,glv_weights=5/6/5
//! ```
//!
//! The fields are those of [`Calibration`]; the weights are written
//! `point_visit/bucket_addition/doubling`. Parsing runs at compile time, so a
//! malformed or out-of-range value fails the build instead of being ignored.

use super::{Calibration, DEFAULT};
use crate::config::{eq, number, split_once};
use crate::glv::msm::plan::PlannerWeights;

/// Widest window a calibration may give the bucket method directly. Each window
/// has `2^(w - 1)` buckets, so the bound keeps the scratch reasonable.
const MAX_WINDOW_BITS: usize = 16;

/// Most window bits the GLV bucket method may add to halo2's default width.
const MAX_GLV_EXTRA_WINDOW_BITS: usize = 4;

/// The calibrations `base=` can name.
const fn named(name: &[u8]) -> Calibration {
    if eq(name, b"default") {
        DEFAULT
    } else {
        panic!("PASTA_CURVES_CALIBRATION: unknown base calibration")
    }
}

/// Parses `spec`, the value of `PASTA_CURVES_CALIBRATION`, and checks the
/// result. Panics, which fails the build in a constant, on any error.
pub(in crate::glv::msm) const fn parse(spec: &str) -> Calibration {
    let mut rest = spec.as_bytes();
    let mut c = DEFAULT;
    let mut first = true;
    while !rest.is_empty() {
        let (item, tail) = split_once(rest, b',');
        rest = tail;
        let (key, value) = split_once(item, b'=');
        if key.is_empty() || value.is_empty() {
            panic!("PASTA_CURVES_CALIBRATION: expected `field=value`");
        }
        if eq(key, b"base") {
            if !first {
                panic!("PASTA_CURVES_CALIBRATION: `base` must come first");
            }
            c = named(value);
        } else {
            set(&mut c, key, value);
        }
        first = false;
    }
    validate(&c);
    c
}

/// Sets the field `key` of `c` to `value`.
const fn set(c: &mut Calibration, key: &[u8], value: &[u8]) {
    if eq(key, b"min_glv_multiexp_terms") {
        c.min_glv_multiexp_terms = number(value);
    } else if eq(key, b"serial_straus_max_terms") {
        c.serial_straus_max_terms = number(value);
    } else if eq(key, b"straus_leaf_terms") {
        c.straus_leaf_terms = number(value);
    } else if eq(key, b"min_parallel_straus_terms") {
        #[cfg(feature = "multicore")]
        {
            c.min_parallel_straus_terms = number(value);
        }
        #[cfg(not(feature = "multicore"))]
        {
            let _ = value;
            panic!("PASTA_CURVES_CALIBRATION: min_parallel_straus_terms needs `multicore`");
        }
    } else if eq(key, b"max_parallel_straus_terms") {
        #[cfg(feature = "multicore")]
        {
            c.max_parallel_straus_terms = number(value);
        }
        #[cfg(not(feature = "multicore"))]
        {
            let _ = value;
            panic!("PASTA_CURVES_CALIBRATION: max_parallel_straus_terms needs `multicore`");
        }
    } else if eq(key, b"serial_narrow_window_bits") {
        c.serial_narrow_window_bits = number(value);
    } else if eq(key, b"serial_wide_window_bits") {
        c.serial_wide_window_bits = number(value);
    } else if eq(key, b"serial_wide_window_min_terms") {
        c.serial_wide_window_min_terms = number(value);
    } else if eq(key, b"glv_weights") {
        c.glv_weights = weights(value);
    } else if eq(key, b"generic_weights") {
        c.generic_weights = weights(value);
    } else if eq(key, b"glv_extra_window_bits") {
        c.glv_extra_window_bits = number(value);
    } else if eq(key, b"min_window_block_terms") {
        c.min_window_block_terms = number(value);
    } else if eq(key, b"window_block_terms_per_bucket") {
        c.window_block_terms_per_bucket = number(value);
    } else {
        panic!("PASTA_CURVES_CALIBRATION: unknown field");
    }
}

/// Rejects the values the planner and the bucket method do not support.
const fn validate(c: &Calibration) {
    if c.straus_leaf_terms == 0 {
        panic!("PASTA_CURVES_CALIBRATION: straus_leaf_terms must be at least 1");
    }
    #[cfg(feature = "multicore")]
    if c.min_parallel_straus_terms == 0 || c.min_parallel_straus_terms > c.max_parallel_straus_terms
    {
        panic!(
            "PASTA_CURVES_CALIBRATION: need 1 <= min_parallel_straus_terms <= max_parallel_straus_terms"
        );
    }
    if c.serial_narrow_window_bits == 0
        || c.serial_narrow_window_bits > MAX_WINDOW_BITS
        || c.serial_wide_window_bits == 0
        || c.serial_wide_window_bits > MAX_WINDOW_BITS
    {
        panic!("PASTA_CURVES_CALIBRATION: serial window widths must be in 1..=16");
    }
    if c.glv_extra_window_bits > MAX_GLV_EXTRA_WINDOW_BITS {
        panic!("PASTA_CURVES_CALIBRATION: glv_extra_window_bits must be at most 4");
    }
    if c.min_window_block_terms == 0 || c.window_block_terms_per_bucket == 0 {
        panic!("PASTA_CURVES_CALIBRATION: window block sizes must be at least 1");
    }
    let (g, n) = (c.glv_weights, c.generic_weights);
    if g.point_visit == 0
        || g.bucket_addition == 0
        || g.doubling == 0
        || n.point_visit == 0
        || n.bucket_addition == 0
        || n.doubling == 0
    {
        panic!("PASTA_CURVES_CALIBRATION: weights must be at least 1");
    }
}

/// Parses `point_visit/bucket_addition/doubling`.
const fn weights(value: &[u8]) -> PlannerWeights {
    let (point_visit, rest) = split_once(value, b'/');
    let (bucket_addition, doubling) = split_once(rest, b'/');
    PlannerWeights {
        point_visit: number(point_visit),
        bucket_addition: number(bucket_addition),
        doubling: number(doubling),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_is_default() {
        assert_eq!(parse(""), DEFAULT);
        assert_eq!(parse("base=default"), DEFAULT);
    }

    #[test]
    fn overrides_fields() {
        let c = parse(
            "base=default,min_glv_multiexp_terms=64,serial_straus_max_terms=32,\
             glv_weights=5/6/5,glv_extra_window_bits=1",
        );
        assert_eq!(
            c,
            Calibration {
                min_glv_multiexp_terms: 64,
                serial_straus_max_terms: 32,
                glv_weights: PlannerWeights {
                    point_visit: 5,
                    bucket_addition: 6,
                    doubling: 5,
                },
                glv_extra_window_bits: 1,
                ..DEFAULT
            }
        );
    }

    #[test]
    fn every_field_parses() {
        #[cfg(feature = "multicore")]
        let c = parse(
            "min_glv_multiexp_terms=1,serial_straus_max_terms=2,straus_leaf_terms=3,\
             min_parallel_straus_terms=4,max_parallel_straus_terms=5,\
             serial_narrow_window_bits=6,serial_wide_window_bits=7,\
             serial_wide_window_min_terms=8,glv_weights=9/10/11,generic_weights=12/13/14,\
             glv_extra_window_bits=3,min_window_block_terms=16,window_block_terms_per_bucket=17",
        );
        #[cfg(not(feature = "multicore"))]
        let c = parse(
            "min_glv_multiexp_terms=1,serial_straus_max_terms=2,straus_leaf_terms=3,\
             serial_narrow_window_bits=6,serial_wide_window_bits=7,\
             serial_wide_window_min_terms=8,glv_weights=9/10/11,generic_weights=12/13/14,\
             glv_extra_window_bits=3,min_window_block_terms=16,window_block_terms_per_bucket=17",
        );
        assert_eq!(
            (
                c.min_glv_multiexp_terms,
                c.serial_straus_max_terms,
                c.straus_leaf_terms
            ),
            (1, 2, 3)
        );
        #[cfg(feature = "multicore")]
        assert_eq!(
            (c.min_parallel_straus_terms, c.max_parallel_straus_terms),
            (4, 5)
        );
        assert_eq!(
            (
                c.serial_narrow_window_bits,
                c.serial_wide_window_bits,
                c.serial_wide_window_min_terms
            ),
            (6, 7, 8)
        );
        assert_eq!(
            (
                c.glv_weights.point_visit,
                c.glv_weights.bucket_addition,
                c.glv_weights.doubling
            ),
            (9, 10, 11)
        );
        assert_eq!(
            (
                c.generic_weights.point_visit,
                c.generic_weights.bucket_addition,
                c.generic_weights.doubling
            ),
            (12, 13, 14)
        );
        assert_eq!(
            (
                c.glv_extra_window_bits,
                c.min_window_block_terms,
                c.window_block_terms_per_bucket
            ),
            (3, 16, 17)
        );
    }

    #[test]
    #[should_panic(expected = "unknown field")]
    fn rejects_unknown_field() {
        parse("min_glv_terms=64");
    }

    #[test]
    #[should_panic(expected = "unknown base")]
    fn rejects_unknown_base() {
        parse("base=fast");
    }

    #[test]
    #[should_panic(expected = "must come first")]
    fn rejects_late_base() {
        parse("straus_leaf_terms=8,base=default");
    }

    #[test]
    #[should_panic(expected = "expected a number")]
    fn rejects_non_number() {
        parse("straus_leaf_terms=8x");
    }

    #[test]
    #[should_panic(expected = "expected `field=value`")]
    fn rejects_missing_value() {
        parse("straus_leaf_terms");
    }

    #[test]
    #[should_panic(expected = "number too large")]
    fn rejects_overflow() {
        parse("straus_leaf_terms=99999999999999999999999");
    }

    #[test]
    #[should_panic(expected = "expected a number")]
    fn rejects_short_weights() {
        parse("glv_weights=5/6");
    }

    #[test]
    #[should_panic(expected = "serial window widths")]
    fn rejects_zero_width() {
        parse("serial_narrow_window_bits=0");
    }

    #[test]
    #[should_panic(expected = "glv_extra_window_bits")]
    fn rejects_wide_extra() {
        parse("glv_extra_window_bits=5");
    }

    #[test]
    #[should_panic(expected = "weights must be at least 1")]
    fn rejects_zero_weight() {
        parse("generic_weights=0/5/5");
    }

    #[cfg(not(feature = "multicore"))]
    #[test]
    #[should_panic(expected = "needs `multicore`")]
    fn rejects_parallel_fields_without_multicore() {
        parse("min_parallel_straus_terms=16");
    }

    #[cfg(feature = "multicore")]
    #[test]
    #[should_panic(expected = "min_parallel_straus_terms")]
    fn rejects_inverted_parallel_range() {
        parse("min_parallel_straus_terms=50,max_parallel_straus_terms=48");
    }
}
