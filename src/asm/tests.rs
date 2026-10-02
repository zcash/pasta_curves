//! Known-answer tests of the six entry points, for both Pasta fields.
//!
//! The expected values were computed independently with big-integer Montgomery
//! arithmetic (`a * b * 2^-256 mod p`) in Python, and the ones that are also
//! among the reference vectors recorded from the assembly on Apple M-series
//! hardware agree with those. The differential tests against the portable
//! arithmetic are `asm_matches_portable_arithmetic` and the `mul` tests beside
//! it in `src/fields/fp.rs` and `src/fields/fq.rs`.

use super::{Limbs, add, from_mont, sub};

use super::{mul, sqr_n_mul, square};

/// The borrow chain of `is_canonical` decides `value < modulus` at the limb boundaries.
#[test]
fn is_canonical_borrow_chain() {
    let m = [5, 0, 0, 7];
    assert!(super::is_canonical(&[4, 0, 0, 7], &m));
    assert!(!super::is_canonical(&m, &m));
    assert!(!super::is_canonical(&[6, 0, 0, 7], &m));
    // A borrow out of the low limbs is absorbed by a larger top limb, and forced by a smaller one.
    assert!(super::is_canonical(&[u64::MAX, u64::MAX, u64::MAX, 6], &m));
    assert!(!super::is_canonical(&[0, 0, 0, 8], &m));
    // A middle limb decides when the top limbs agree.
    assert!(!super::is_canonical(&[0, 1, 0, 7], &m));
    assert!(super::is_canonical(&[u64::MAX, 0, 0, 6], &m));
    assert!(super::is_canonical(&[0, 0, 0, 0], &m));
}

/// One field's constants and known answers.
struct Field {
    modulus: Limbs,
    /// `-modulus[0]^-1 mod 2^64`.
    inv: u64,
    /// `R = 2^256 mod p`, the Montgomery form of `1`.
    r: Limbs,
    /// `2R mod p`.
    two_r: Limbs,
    /// `3R mod p`.
    three_r: Limbs,
    /// `R^2 mod p`.
    r2: Limbs,
    /// `R^3 mod p`.
    r3: Limbs,
    /// `mul(R2, R3) = R^4 mod p`.
    r4: Limbs,
    /// `sqr_n_mul(R2, 2, R3) = R^7 mod p`.
    r7: Limbs,
    /// `mul(p - 1, p - 1)`.
    pm1_sq: Limbs,
    /// `p - 2`.
    pm2: Limbs,
}

/// The Pallas base field (`pasta_curves::Fp`).
const FP: Field = Field {
    modulus: [
        0x992d30ed00000001,
        0x224698fc094cf91b,
        0x0000000000000000,
        0x4000000000000000,
    ],
    inv: 0x992d30ecffffffff,
    r: [
        0x34786d38fffffffd,
        0x992c350be41914ad,
        0xffffffffffffffff,
        0x3fffffffffffffff,
    ],
    two_r: [
        0xcfc3a984fffffff9,
        0x1011d11bbee5303e,
        0xffffffffffffffff,
        0x3fffffffffffffff,
    ],
    three_r: [
        0x6b0ee5d0fffffff5,
        0x86f76d2b99b14bd0,
        0xfffffffffffffffe,
        0x3fffffffffffffff,
    ],
    r2: [
        0x8c78ecb30000000f,
        0xd7d30dbd8b0de0e7,
        0x7797a99bc3c95d18,
        0x096d41af7b9cb714,
    ],
    r3: [
        0xf185a5993a9e10f9,
        0xf6a68f3b6ac5b1d1,
        0xdf8d1014353fd42c,
        0x2ae309222d2d9910,
    ],
    r4: [
        0x1dfc65f6ad0492ae,
        0x84379b4cc10e927b,
        0x710d6cd04c692c97,
        0x21cce888a6cab566,
    ],
    r7: [
        0x7a3a1c29d1d1bd45,
        0x17023e5920bb6157,
        0x9004eaaf35c21e06,
        0x007efbf9151076fc,
    ],
    pm1_sq: [
        0xcf3f8e8753a769a9,
        0xac9fba6a4077fc57,
        0x70cb2996efc89a65,
        0x21f1c4ff1e2278d5,
    ],
    pm2: [
        0x992d30ecffffffff,
        0x224698fc094cf91b,
        0x0000000000000000,
        0x4000000000000000,
    ],
};

/// The Vesta base field (`pasta_curves::Fq`).
const FQ: Field = Field {
    modulus: [
        0x8c46eb2100000001,
        0x224698fc0994a8dd,
        0x0000000000000000,
        0x4000000000000000,
    ],
    inv: 0x8c46eb20ffffffff,
    r: [
        0x5b2b3e9cfffffffd,
        0x992c350be3420567,
        0xffffffffffffffff,
        0x3fffffffffffffff,
    ],
    two_r: [
        0x2a0f9218fffffff9,
        0x1011d11bbcef61f1,
        0xffffffffffffffff,
        0x3fffffffffffffff,
    ],
    three_r: [
        0xf8f3e594fffffff5,
        0x86f76d2b969cbe7a,
        0xfffffffffffffffe,
        0x3fffffffffffffff,
    ],
    r2: [
        0xfc9678ff0000000f,
        0x67bb433d891a16e3,
        0x7fae231004ccf590,
        0x096d41af7ccfdaa9,
    ],
    r3: [
        0x008b421c249dae4c,
        0xe13bda50dba41326,
        0x88fececb8e15cb63,
        0x07dd97a06e6792c8,
    ],
    r4: [
        0x569bba29179df5c1,
        0xf7abe57547cfa14c,
        0x8d0f36071632bdab,
        0x2c37a71489ba6088,
    ],
    r7: [
        0x56244c6793e8be1f,
        0xbe518f1c2a1b26e6,
        0xb6c73001c86b2b65,
        0x1b7d3aff1b7fd420,
    ],
    pm1_sq: [
        0x6119a3dd8e1a6f7f,
        0xc68de1279dc601eb,
        0x5790be58c050df13,
        0x1f7a89dd17647953,
    ],
    pm2: [
        0x8c46eb20ffffffff,
        0x224698fc0994a8dd,
        0x0000000000000000,
        0x4000000000000000,
    ],
};

const FIELDS: [&Field; 2] = [&FP, &FQ];

const ONE: Limbs = [1, 0, 0, 0];
const ZERO: Limbs = [0, 0, 0, 0];

/// `p - 1`; `modulus[0]` is odd, so the subtraction does not borrow.
fn p_minus_1(f: &Field) -> Limbs {
    let mut limbs = f.modulus;
    limbs[0] -= 1;
    limbs
}

/// The constants above are copies of the field types' own, so that a known answer here says
/// something about the fields the crate computes in: check that they agree.
#[test]
fn constants_match_the_field_types() {
    use crate::fields::{fp, fq};

    assert_eq!(FP.modulus, fp::MODULUS.0);
    assert_eq!(FP.inv, fp::INV);
    assert_eq!(FP.r, fp::R.0);
    assert_eq!(FP.r2, fp::R2.0);
    assert_eq!(FP.r3, fp::R3.0);

    assert_eq!(FQ.modulus, fq::MODULUS.0);
    assert_eq!(FQ.inv, fq::INV);
    assert_eq!(FQ.r, fq::R.0);
    assert_eq!(FQ.r2, fq::R2.0);
    assert_eq!(FQ.r3, fq::R3.0);
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
    }
    // `from_mont` accepts any four-limb value: the all-ones input is the
    // extreme case of that contract, where the candidate is largest.
    assert_eq!(
        from_mont(&[u64::MAX; 4], &FP.modulus, FP.inv),
        [
            0xc9eda265ac589659,
            0x75a6de91c8d4fcc3,
            0x8f34d6691037659a,
            0x1e0e3b00e1dd872a,
        ]
    );
    assert_eq!(
        from_mont(&[u64::MAX; 4], &FQ.modulus, FQ.inv),
        [
            0x2b2d474371e59083,
            0x5bb8b7d46bcea6f2,
            0xa86f41a73faf20ec,
            0x20857622e89b86ac,
        ]
    );
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
            super::is_canonical(first, &f.modulus)
                || (super::is_canonical(rhs, &f.modulus)
                    && rhs[1..].iter().all(|&limb| limb <= u64::MAX - 2))
        }
        "SQR" => super::is_canonical(first, &f.modulus),
        "ADD" | "SUB" => {
            super::is_canonical(first, &f.modulus)
                && super::is_canonical(second.unwrap(), &f.modulus)
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
