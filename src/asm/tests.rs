//! Known-answer tests of the four entry points, for both Pasta fields.
//!
//! The expected values were computed independently with big-integer Montgomery
//! arithmetic (`a * b * 2^-256 mod p`) in Python, and the ones that are also among
//! the reference vectors recorded from the assembly on Apple M-series hardware
//! agree with those. The differential tests against the portable arithmetic are
//! those of zcash/pasta_curves#100: `aarch64_asm_matches_portable_arithmetic` and
//! the `mul` tests beside it in `src/fields/fp.rs` and `src/fields/fq.rs`. The
//! field types do not use the backend yet.

use super::{Limbs, add, from_mont, mul, sqr_n_mul, square, sub};

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
}
