//! The calibration given at build time by `PASTA_CURVES_BATCH_MUL_SCALARS_CALIBRATION`,
//! with `--cfg pasta_curves_batch_mul_scalars_calibration`. The variable lists
//! comma-separated `field=value` pairs, optionally starting with
//! `base=<name>` (default: `default`); every field not listed keeps the
//! base's value:
//!
//! ```text
//! base=default,min_block_products=16,max_threads=10
//! ```
//!
//! The fields are those of [`Calibration`]. Parsing runs at compile time, so a
//! malformed or out-of-range value fails the build instead of being ignored.

use super::{Calibration, DEFAULT};
use crate::config::{eq, number, split_once};

/// The calibrations `base=` can name.
const fn named(name: &[u8]) -> Calibration {
    if eq(name, b"default") {
        DEFAULT
    } else {
        panic!("PASTA_CURVES_BATCH_MUL_SCALARS_CALIBRATION: unknown base calibration")
    }
}

/// Parses `spec`, the value of `PASTA_CURVES_BATCH_MUL_SCALARS_CALIBRATION`, and checks the
/// result. Panics, which fails the build in a constant, on any error.
pub(super) const fn parse(spec: &str) -> Calibration {
    let mut rest = spec.as_bytes();
    let mut c = DEFAULT;
    let mut first = true;
    while !rest.is_empty() {
        let (item, tail) = split_once(rest, b',');
        rest = tail;
        let (key, value) = split_once(item, b'=');
        if key.is_empty() || value.is_empty() {
            panic!("PASTA_CURVES_BATCH_MUL_SCALARS_CALIBRATION: expected `field=value`");
        }
        if eq(key, b"base") {
            if !first {
                panic!("PASTA_CURVES_BATCH_MUL_SCALARS_CALIBRATION: `base` must come first");
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
    if eq(key, b"glv_min_products") {
        c.glv_min_products = number(value);
    } else if eq(key, b"eisenstein_min_products") {
        #[cfg(feature = "glv-eisenstein")]
        {
            c.eisenstein_min_products = number(value);
        }
        #[cfg(not(feature = "glv-eisenstein"))]
        {
            let _ = value;
            panic!(
                "PASTA_CURVES_BATCH_MUL_SCALARS_CALIBRATION: eisenstein_min_products needs `glv-eisenstein`"
            );
        }
    } else if eq(key, b"min_block_products")
        || eq(key, b"min_balanced_block_lanes")
        || eq(key, b"balanced_blocks_per_thread")
        || eq(key, b"max_threads")
    {
        set_threads(c, key, value);
    } else {
        panic!("PASTA_CURVES_BATCH_MUL_SCALARS_CALIBRATION: unknown field");
    }
}

/// Sets one of the fields that split a grid between threads.
#[cfg(feature = "multicore")]
const fn set_threads(c: &mut Calibration, key: &[u8], value: &[u8]) {
    let value = number(value);
    if eq(key, b"min_block_products") {
        c.min_block_products = value;
    } else if eq(key, b"min_balanced_block_lanes") {
        c.min_balanced_block_lanes = value;
    } else if eq(key, b"balanced_blocks_per_thread") {
        c.balanced_blocks_per_thread = value;
    } else {
        c.max_threads = value;
    }
}

/// Without `multicore` a grid is never split, so these fields do not exist.
#[cfg(not(feature = "multicore"))]
const fn set_threads(_: &mut Calibration, _: &[u8], _: &[u8]) {
    panic!("PASTA_CURVES_BATCH_MUL_SCALARS_CALIBRATION: the thread fields need `multicore`");
}

/// Rejects the values the grid evaluation does not support.
const fn validate(c: &Calibration) {
    #[cfg(feature = "multicore")]
    if c.min_block_products == 0
        || c.min_balanced_block_lanes == 0
        || c.balanced_blocks_per_thread == 0
    {
        panic!(
            "PASTA_CURVES_BATCH_MUL_SCALARS_CALIBRATION: block sizes and counts must be at least 1"
        );
    }
    let _ = c;
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
        let mut expected = DEFAULT;
        expected.glv_min_products = 24;
        assert_eq!(parse("base=default,glv_min_products=24"), expected);
    }

    #[cfg(feature = "glv-eisenstein")]
    #[test]
    fn eisenstein_threshold_parses() {
        assert_eq!(
            parse("eisenstein_min_products=32").eisenstein_min_products,
            32
        );
    }

    #[cfg(not(feature = "glv-eisenstein"))]
    #[test]
    #[should_panic(expected = "needs `glv-eisenstein`")]
    fn rejects_eisenstein_threshold_without_the_feature() {
        parse("eisenstein_min_products=32");
    }

    #[cfg(feature = "multicore")]
    #[test]
    fn thread_fields_parse() {
        let c = parse(
            "min_block_products=4,min_balanced_block_lanes=32,\
             balanced_blocks_per_thread=3,max_threads=10",
        );
        assert_eq!(
            (
                c.min_block_products,
                c.min_balanced_block_lanes,
                c.balanced_blocks_per_thread,
                c.max_threads
            ),
            (4, 32, 3, 10)
        );
    }

    #[cfg(feature = "multicore")]
    #[test]
    #[should_panic(expected = "at least 1")]
    fn rejects_empty_blocks() {
        parse("min_block_products=0");
    }

    #[cfg(feature = "multicore")]
    #[test]
    #[should_panic(expected = "at least 1")]
    fn rejects_no_blocks_per_thread() {
        parse("balanced_blocks_per_thread=0");
    }

    #[cfg(not(feature = "multicore"))]
    #[test]
    #[should_panic(expected = "need `multicore`")]
    fn rejects_thread_fields_without_multicore() {
        parse("max_threads=4");
    }

    #[test]
    #[should_panic(expected = "unknown field")]
    fn rejects_unknown_field() {
        parse("min_products=16");
    }

    #[test]
    #[should_panic(expected = "unknown base")]
    fn rejects_unknown_base() {
        parse("base=phone");
    }

    #[test]
    #[should_panic(expected = "must come first")]
    fn rejects_late_base() {
        parse("glv_min_products=8,base=default");
    }

    #[test]
    #[should_panic(expected = "expected `field=value`")]
    fn rejects_missing_value() {
        parse("glv_min_products");
    }

    #[test]
    #[should_panic(expected = "expected a number")]
    fn rejects_non_number() {
        parse("glv_min_products=8x");
    }
}
