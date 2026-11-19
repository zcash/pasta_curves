//! Parsing configuration given at build time, such as a calibration read from
//! an environment variable with `option_env!`: the byte-level steps of
//! reading a `field=value` list, which a `const fn` must spell out since the
//! standard library's parsing is not available in constants. Each
//! configuration's own parser names its fields and checks their values.
//!
//! A malformed value panics, which fails the build in a constant.

/// Parses a decimal `usize`.
pub(crate) const fn number(digits: &[u8]) -> usize {
    if digits.is_empty() {
        panic!("configuration: expected a number");
    }
    let mut n: usize = 0;
    let mut i = 0;
    while i < digits.len() {
        let d = digits[i];
        if !d.is_ascii_digit() {
            panic!("configuration: expected a number");
        }
        n = match n.checked_mul(10) {
            Some(n) => match n.checked_add((d - b'0') as usize) {
                Some(n) => n,
                None => panic!("configuration: number too large"),
            },
            None => panic!("configuration: number too large"),
        };
        i += 1;
    }
    n
}

/// Splits at the first `sep`, dropping it; the tail is empty if there is none.
pub(crate) const fn split_once(bytes: &[u8], sep: u8) -> (&[u8], &[u8]) {
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == sep {
            let (head, tail) = bytes.split_at(i);
            return (head, tail.split_at(1).1);
        }
        i += 1;
    }
    (bytes, &[])
}

/// Byte-slice equality, usable in a constant.
pub(crate) const fn eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers() {
        assert_eq!(number(b"0"), 0);
        assert_eq!(number(b"0042"), 42);
        assert_eq!(number(b"18446744073709551615"), u64::MAX as usize);
    }

    #[test]
    #[should_panic(expected = "expected a number")]
    fn rejects_empty_number() {
        number(b"");
    }

    #[test]
    #[should_panic(expected = "expected a number")]
    fn rejects_non_digit() {
        number(b"1x");
    }

    #[test]
    #[should_panic(expected = "number too large")]
    fn rejects_overflow() {
        number(b"99999999999999999999999");
    }

    #[test]
    fn splits_at_the_first_separator() {
        assert_eq!(split_once(b"a=b=c", b'='), (&b"a"[..], &b"b=c"[..]));
        assert_eq!(split_once(b"a=", b'='), (&b"a"[..], &b""[..]));
        assert_eq!(split_once(b"abc", b'='), (&b"abc"[..], &b""[..]));
        assert_eq!(split_once(b"", b'='), (&b""[..], &b""[..]));
    }

    #[test]
    fn compares_bytes() {
        assert!(eq(b"", b""));
        assert!(eq(b"base", b"base"));
        assert!(!eq(b"base", b"bas"));
        assert!(!eq(b"base", b"bash"));
    }
}
