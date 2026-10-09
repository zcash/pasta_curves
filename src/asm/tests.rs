//! Tests of the entry points, for both Pasta fields: known answers, the reference
//! vectors, and a differential test against the portable arithmetic.
//!
//! The constants and known answers are those of `crate::test_fields`. Every
//! multiplication, squaring, and conversion case is also among the reference
//! vectors recorded from the assembly on Apple M-series hardware, and agrees with
//! it. The vectors have no addition, subtraction, or repeated-squaring cases. The
//! differential test compares every entry point with the portable arithmetic of the
//! field types, on edge and pseudo-random operands, so it covers those routines
//! too. The inversion's blocks and driver are tested in `crate::inversion`.

use super::{Limbs, add, from_mont, sub};

use super::{mul, sqr_n_mul, square};
use crate::test_fields::{FIELDS, FP, FQ, Field, ONE, ZERO, p_minus_1};

/// The entry points agree with the portable arithmetic of the field types, on edge operands and on
/// pseudo-random ones. The portable arithmetic is the field types' inherent `const fn`s, which
/// never use the backend, applied to the same Montgomery residues. The portable multiplication is
/// exact whenever the product is below `2^256 p`, which holds whenever one operand is canonical, so
/// it is the reference under both of the multiplication's contracts, and for `from_mont`, which
/// multiplies by the residue `1`. The operands are canonical where a routine's contract asks for
/// that; elsewhere they range over every four-limb value. Like the rest of this module's tests, the
/// test is compiled only where the module has a backend, so it always compares the assembly with
/// the portable code. The multiplication also gets 200,000 random canonical operand pairs, and the
/// cases of its second contract where its accumulator comes nearest to wrapping.
#[test]
fn entry_points_match_the_portable_arithmetic() {
    use rand::{Rng, SeedableRng};
    use rand_xorshift::XorShiftRng;

    macro_rules! check {
        ($field:ident, $f:expr, $forced_r2:expr, $forced_r3:expr) => {{
            use crate::fields::$field;
            let f = $f;
            let portable_mul = |x: &Limbs, y: &Limbs| $field::mul(&$field(*x), &$field(*y)).0;
            let portable_square = |x: &Limbs| $field::square(&$field(*x)).0;
            let mut rng = XorShiftRng::from_seed([0x5a; 16]);
            let random_any: [Limbs; 32] =
                core::array::from_fn(|_| core::array::from_fn(|_| rng.next_u64()));
            // The Montgomery forms of elements reduced from 512 random bits, close to uniform.
            let random_canonical: [Limbs; 32] = core::array::from_fn(|_| {
                $field::from_u512(core::array::from_fn(|_| rng.next_u64())).0
            });

            let high = [u64::MAX, u64::MAX, u64::MAX, (1 << 62) - 1];
            let edge_canonical = [ZERO, ONE, f.r, p_minus_1(f), f.pm2, high];
            let edge_any = [[u64::MAX; 4], f.modulus, [0, 0, 0, u64::MAX]];
            let canonical = || edge_canonical.iter().chain(&random_canonical);
            let any = || canonical().chain(&edge_any).chain(&random_any);

            for a in canonical() {
                assert_eq!(square(a, &f.modulus, f.inv), portable_square(a), "{a:x?}");
                for b in canonical() {
                    let sum = $field::add(&$field(*a), &$field(*b)).0;
                    let difference = $field::sub(&$field(*a), &$field(*b)).0;
                    assert_eq!(add(a, b, &f.modulus), sum, "{a:x?} {b:x?}");
                    assert_eq!(sub(a, b, &f.modulus), difference, "{a:x?} {b:x?}");
                }
                for b in any() {
                    let product = portable_mul(a, b);
                    assert_eq!(mul(a, b, &f.modulus, f.inv), product, "{a:x?} {b:x?}");
                    // The other contract: any left operand, with a canonical right operand
                    // whose limbs 1 to 3 are at most `2^64 - 3`.
                    if crate::montgomery::mul_contract(b, a, &f.modulus) {
                        assert_eq!(mul(b, a, &f.modulus, f.inv), product, "{b:x?} {a:x?}");
                    }
                    let mut power = *a;
                    for count in 0..4 {
                        let expected = portable_mul(&power, b);
                        let actual = sqr_n_mul(a, count, b, &f.modulus, f.inv);
                        assert_eq!(actual, expected, "{a:x?} {count} {b:x?}");
                        power = portable_square(&power);
                    }
                }
            }
            for a in any() {
                let expected = portable_mul(a, &[1, 0, 0, 0]);
                assert_eq!(from_mont(a, &f.modulus, f.inv), expected, "{a:x?}");
            }

            // Many more random canonical operands for the multiplication.
            let mut rng = XorShiftRng::from_seed([0x42; 16]);
            for _ in 0..200_000u32 {
                let a = $field::from_raw(core::array::from_fn(|_| rng.next_u64())).0;
                let b = $field::from_raw(core::array::from_fn(|_| rng.next_u64())).0;
                assert_eq!(mul(&a, &b, &f.modulus, f.inv), portable_mul(&a, &b), "{a:x?} {b:x?}");
            }

            // The multiplication's second contract, where its five-limb accumulator comes nearest
            // to wrapping: an unreduced left operand, and a canonical right operand whose limbs 1
            // to 3 are at most `2^64 - 3`. Each case asserts that it is within the contract.
            let second_contract = |lhs: Limbs, rhs: Limbs| {
                assert!(crate::montgomery::mul_contract(&lhs, &rhs, &f.modulus), "{lhs:x?} {rhs:x?}");
                let product = portable_mul(&lhs, &rhs);
                assert_eq!(mul(&lhs, &rhs, &f.modulus, f.inv), product, "{lhs:x?} {rhs:x?}");
            };
            // The right operands are `R2` and `R3`, which `from_u512` multiplies by, and two
            // canonical values near the modulus.
            let mut dense = [u64::MAX - 2; 4];
            dense[3] = f.modulus[3] - 1;
            for rhs in [f.r2, f.r3, p_minus_1(f), dense] {
                second_contract([u64::MAX; 4], rhs);
            }
            // The low limbs of these left operands bring the first two Montgomery quotients to
            // (or near) their maximum, and their top limbs are all ones.
            second_contract($forced_r2, f.r2);
            second_contract($forced_r3, f.r3);
            let mut rng = XorShiftRng::from_seed([0x9d; 16]);
            for i in 0..20_000u32 {
                let mut lhs: Limbs = core::array::from_fn(|_| rng.next_u64());
                if i % 2 == 0 {
                    lhs[2] = u64::MAX;
                    lhs[3] = u64::MAX;
                }
                second_contract(lhs, f.r2);
                second_contract(lhs, f.r3);
            }
            // Left operands with the top bit set, against right operands just below the modulus,
            // where the bound `T < 2p` on the final reduction's input is tightest.
            let mut rng = XorShiftRng::from_seed([0x17; 16]);
            let mut n = 0u32;
            while n < 100_000 {
                let mut lhs: Limbs = core::array::from_fn(|_| rng.next_u64());
                lhs[3] |= 1 << 63;
                let mut rhs = f.modulus;
                rhs[0] = rhs[0].wrapping_sub(rng.next_u64() >> (rng.next_u32() % 64));
                if rng.next_u32() & 1 == 1 {
                    rhs[1] = rhs[1].wrapping_sub(rng.next_u64() >> 60);
                }
                if crate::montgomery::mul_contract(&lhs, &rhs, &f.modulus) {
                    n += 1;
                    second_contract(lhs, rhs);
                }
            }
        }};
    }
    check!(
        Fp,
        &FP,
        [0x3cc9961eeeeeeeef, 0x907f42c685cc8a31, u64::MAX, u64::MAX],
        [0x032c286da5f9b149, 0x3f747fab2d936552, u64::MAX, u64::MAX]
    );
    check!(
        Fq,
        &FQ,
        [0xf3bfcadeeeeeeeef, 0x27fa6352b2545d71, u64::MAX, u64::MAX],
        [0x0000000000000d24, 0x00000000000007c2, u64::MAX, u64::MAX]
    );
}

#[test]
fn add_known_answers() {
    for f in FIELDS {
        assert_eq!(add(&f.r, &f.r, &f.modulus), f.two_r);
        assert_eq!(add(&f.r, &f.two_r, &f.modulus), f.three_r);
        assert_eq!(add(&f.two_r, &f.r, &f.modulus), f.three_r);
        let pm1 = p_minus_1(f);
        assert_eq!(add(&pm1, &pm1, &f.modulus), f.pm2);
        assert_eq!(add(&ZERO, &pm1, &f.modulus), pm1);
        assert_eq!(add(&pm1, &ZERO, &f.modulus), pm1);
        assert_eq!(add(&pm1, &ONE, &f.modulus), ZERO);
    }
}

#[test]
fn sub_known_answers() {
    for f in FIELDS {
        assert_eq!(sub(&f.r, &f.r, &f.modulus), ZERO);
        assert_eq!(sub(&f.two_r, &f.r, &f.modulus), f.r);
        assert_eq!(sub(&f.three_r, &f.r, &f.modulus), f.two_r);
        assert_eq!(sub(&f.three_r, &f.two_r, &f.modulus), f.r);
        let pm1 = p_minus_1(f);
        assert_eq!(sub(&pm1, &pm1, &f.modulus), ZERO);
        assert_eq!(sub(&pm1, &f.pm2, &f.modulus), ONE);
        assert_eq!(sub(&ZERO, &pm1, &f.modulus), ONE);
        assert_eq!(sub(&ZERO, &ONE, &f.modulus), pm1);
    }
}

#[test]
fn mul_known_answers() {
    for f in FIELDS {
        assert_eq!(mul(&f.r, &f.r, &f.modulus, f.inv), f.r);
        assert_eq!(mul(&f.r, &f.r2, &f.modulus, f.inv), f.r2);
        assert_eq!(mul(&f.r2, &f.r3, &f.modulus, f.inv), f.r4);
        assert_eq!(mul(&f.r3, &f.r2, &f.modulus, f.inv), f.r4);
        let pm1 = p_minus_1(f);
        assert_eq!(mul(&pm1, &pm1, &f.modulus, f.inv), f.pm1_sq);
        assert_eq!(mul(&ZERO, &pm1, &f.modulus, f.inv), ZERO);
    }
}

#[test]
fn square_known_answers() {
    for f in FIELDS {
        assert_eq!(square(&f.r, &f.modulus, f.inv), f.r);
        assert_eq!(square(&f.r2, &f.modulus, f.inv), f.r3);
        let pm1 = p_minus_1(f);
        assert_eq!(square(&pm1, &f.modulus, f.inv), f.pm1_sq);
        assert_eq!(square(&ZERO, &f.modulus, f.inv), ZERO);
    }
}

#[test]
fn sqr_n_mul_known_answers() {
    for f in FIELDS {
        assert_eq!(sqr_n_mul(&f.r2, 0, &f.r3, &f.modulus, f.inv), f.r4);
        assert_eq!(sqr_n_mul(&f.r, 1, &f.r2, &f.modulus, f.inv), f.r2);
        assert_eq!(sqr_n_mul(&f.r2, 1, &f.r, &f.modulus, f.inv), f.r3);
        assert_eq!(sqr_n_mul(&f.r2, 2, &f.r3, &f.modulus, f.inv), f.r7);
    }
}

#[test]
fn from_mont_known_answers() {
    for f in FIELDS {
        assert_eq!(from_mont(&f.r, &f.modulus, f.inv), ONE);
        assert_eq!(from_mont(&f.r2, &f.modulus, f.inv), f.r);
        assert_eq!(from_mont(&ZERO, &f.modulus, f.inv), ZERO);
        // `from_mont` accepts any four-limb value: the all-ones input is the
        // largest one it accepts.
        assert_eq!(from_mont(&[u64::MAX; 4], &f.modulus, f.inv), f.from_mont_ones);
    }
}

/// The reference vectors: outputs of Semolina's `mul_mont_pasta`, `sqr_mont_pasta`, and
/// `from_mont_pasta` as vendored by pasta_curves at `8ad85e9fab7929f6236960e472f432a4bd9ccd74`,
/// recorded on an Apple M-series machine by the test in `test-vectors/dump-asm-vectors.patch`
/// (`test-vectors/README.md` describes the sampling). One vector per line: the routine (`MUL`,
/// `SQR`, `FROM`), the field (`Fp`, `Fq`), the operands, and the output, each 256-bit value as
/// 64 big-endian hex digits.
const VECTORS: &str = include_str!("../../test-vectors/pasta_mul-armv8-vectors.txt");

/// A 256-bit value written as 64 big-endian hex digits, as little-endian limbs.
fn parse_limbs(hex: &str) -> Limbs {
    assert_eq!(hex.len(), 64);
    let limb = |i: usize| u64::from_str_radix(&hex[16 * (3 - i)..16 * (4 - i)], 16).unwrap();
    [limb(0), limb(1), limb(2), limb(3)]
}

/// A vector line: the routine, the field, the first operand, the second operand of a
/// multiplication, and the recorded output.
fn parse_vector(line: &str) -> (&str, &'static Field, Limbs, Option<Limbs>, Limbs) {
    let mut words = line.split_whitespace();
    let op = words.next().unwrap();
    let f = match words.next().unwrap() {
        "Fp" => &FP,
        "Fq" => &FQ,
        key => panic!("unknown field {key}"),
    };
    let first = parse_limbs(words.next().unwrap());
    let second = parse_limbs(words.next().unwrap());
    let third = words.next().map(parse_limbs);
    assert!(words.next().is_none(), "{line}");
    match third {
        Some(expected) => (op, f, first, Some(second), expected),
        None => (op, f, first, None, second),
    }
}

/// Whether a vector's operands are inside its routine's contract: for the multiplication, a
/// canonical left operand, or a canonical right operand whose limbs 1 to 3 are at most
/// `2^64 - 3` (the contract that the proofs establish); for the squaring, the addition, and
/// the subtraction, canonical inputs; for the conversion, any input.
fn in_contract(op: &str, f: &Field, first: &Limbs, second: Option<&Limbs>) -> bool {
    match op {
        "MUL" => {
            let rhs = second.unwrap();
            crate::limbs::is_canonical(first, &f.modulus)
                || (crate::limbs::is_canonical(rhs, &f.modulus)
                    && rhs[1..].iter().all(|&limb| limb <= u64::MAX - 2))
        }
        "SQR" => crate::limbs::is_canonical(first, &f.modulus),
        "ADD" | "SUB" => {
            crate::limbs::is_canonical(first, &f.modulus)
                && crate::limbs::is_canonical(second.unwrap(), &f.modulus)
        }
        "FROM" => true,
        _ => panic!("unknown routine {op}"),
    }
}

/// Runs the routine that a vector names on its operands.
fn run(op: &str, f: &Field, first: &Limbs, second: Option<&Limbs>) -> Limbs {
    match op {
        "MUL" => mul(first, second.unwrap(), &f.modulus, f.inv),
        "SQR" => square(first, &f.modulus, f.inv),
        "FROM" => from_mont(first, &f.modulus, f.inv),
        "ADD" => add(first, second.unwrap(), &f.modulus),
        "SUB" => sub(first, second.unwrap(), &f.modulus),
        _ => panic!("unknown routine {op}"),
    }
}

/// Every reference vector whose operands are inside its routine's contract is reproduced by
/// the crate's routines, which transcribe the routines that produced the vectors. The 180
/// vectors outside the contracts, multiplications with unreduced operands, are not run: the
/// block drops the fifth limb of its final candidate, which can change the result there (it
/// agrees with the routine on 136 of them and differs on 44, all with both operands
/// unreduced). In a debug build the test checks instead that the assertion of the routine's
/// contract fires on each of them. Where a panic cannot be caught, it skips them, with one
/// warning. The file holds no addition or subtraction vectors; the counts below say so, and
/// the code handles them so that a file that gains some needs no other change.
#[test]
fn hardware_vectors_match() {
    // Indexed by routine: MUL, SQR, FROM, ADD, SUB.
    let mut checked = [0usize; 5];
    let mut outside = [0usize; 5];
    for line in VECTORS.lines().filter(|line| !line.is_empty()) {
        let (op, f, first, second, expected) = parse_vector(line);
        let index = match op {
            "MUL" => 0,
            "SQR" => 1,
            "FROM" => 2,
            "ADD" => 3,
            "SUB" => 4,
            _ => panic!("unknown routine {op}"),
        };
        if !in_contract(op, f, &first, second.as_ref()) {
            outside[index] += 1;
            // The assertion itself needs no std. Catching its panic needs both std and a
            // panic strategy that unwinds.
            #[cfg(all(debug_assertions, panic = "unwind"))]
            {
                let panic = std::panic::catch_unwind(|| run(op, f, &first, second.as_ref()))
                    .expect_err("the debug assertion of the routine's contract did not fire");
                let message = panic
                    .downcast_ref::<&str>()
                    .expect("the assertion's message is a string literal");
                let expected_message = match op {
                    "MUL" => "requires a canonical lhs",
                    "SQR" => "requires a canonical input",
                    _ => "requires a canonical",
                };
                assert!(message.contains(expected_message), "{line}: {message}");
            }
            continue;
        }
        assert_eq!(run(op, f, &first, second.as_ref()), expected, "{line}");
        checked[index] += 1;
    }
    assert_eq!(checked, [806, 34, 34, 0, 0]);
    assert_eq!(outside, [180, 0, 0, 0, 0]);
    #[cfg(all(debug_assertions, not(panic = "unwind")))]
    std::eprintln!(
        "warning: the assertions of the routines' contracts were not checked to fire on the \
         {} vectors outside them, since panics cannot be caught here",
        outside.iter().sum::<usize>()
    );
}
