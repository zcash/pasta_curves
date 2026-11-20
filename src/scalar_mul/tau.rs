//! Constant-time regular width-three τ-adic scalar multiplication.
//!
//! Here τ = 1 − ω, τ² = −3ω, and the input GLV halves are written as
//! `(k1 + k2) − k2 τ`. The fixed 54-round recoding and the digit table are
//! derived and checked in `sage/tau_recoding.sage`.

use ff::{Field, WithSmallOrderMulGroup};
use group::Group;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

use super::{FormulaField, Homogeneous};
use crate::glv::{GlvConstants, decompose_signed};
use crate::{Fp, Fq};

// A signed two's-complement integer in three little-endian limbs. Sage proves
// |a|, |b| < 2^129, including every recoding state, and the largest numerator
// in the exact divisions fits in signed 192 bits. No i128 intermediate is used.
#[derive(Clone, Copy, Debug)]
struct Signed192([u64; 3]);

impl Signed192 {
    const ZERO: Self = Self([0; 3]);

    fn from_i128(x: i128) -> Self {
        Self([x as u64, (x >> 64) as u64, (x >> 127) as u64])
    }

    fn from_i8(x: i8) -> Self {
        Self::from_i128(i128::from(x))
    }

    fn add(self, rhs: Self) -> Self {
        let (lo, carry) = self.0[0].overflowing_add(rhs.0[0]);
        let (mid0, carry0) = self.0[1].overflowing_add(rhs.0[1]);
        let (mid, carry1) = mid0.overflowing_add(u64::from(carry));
        let hi = self.0[2]
            .wrapping_add(rhs.0[2])
            .wrapping_add(u64::from(carry0 | carry1));
        Self([lo, mid, hi])
    }

    fn neg(self) -> Self {
        let (lo, carry) = (!self.0[0]).overflowing_add(1);
        let (mid, carry) = (!self.0[1]).overflowing_add(u64::from(carry));
        let (hi, _) = (!self.0[2]).overflowing_add(u64::from(carry));
        Self([lo, mid, hi])
    }

    fn sub(self, rhs: Self) -> Self {
        self.add(rhs.neg())
    }

    fn twice(self) -> Self {
        self.add(self)
    }

    fn thrice(self) -> Self {
        self.twice().add(self)
    }

    #[cfg(test)]
    fn magnitude(self) -> ([u64; 3], Choice) {
        let negative = Choice::from((self.0[2] >> 63) as u8);
        let mask = 0u64.wrapping_sub(u64::from(negative.unwrap_u8()));
        let (lo, carry) = (self.0[0] ^ mask).overflowing_add(u64::from(negative.unwrap_u8()));
        let (mid, carry) = (self.0[1] ^ mask).overflowing_add(u64::from(carry));
        let (hi, _) = (self.0[2] ^ mask).overflowing_add(u64::from(carry));
        ([lo, mid, hi], negative)
    }

    // Long division by the public constants 3 and 9, using 16-bit chunks. The
    // tests use it as the reference for mod9 and div_exact.
    // Every 16-bit dividend is < 9*2^16; the 32-bit reciprocal formula
    // floor((x * 0x38e38e39)/2^33) computes x/9 exactly (and similarly
    // 0xaaaaaaab for /3). Thus no target-dependent integer division helper
    // sees secret values, including on 32-bit targets.
    #[cfg(test)]
    fn quotient_remainder(self, divisor: u8) -> (Self, u8) {
        let (magnitude, negative) = self.magnitude();
        let reciprocal = if divisor == 9 {
            0x38e3_8e39u64
        } else {
            0xaaaa_aaabu64
        };
        let mut quotient = [0u64; 3];
        let mut remainder = 0u32;
        for i in (0..3).rev() {
            for j in (0..4).rev() {
                let dividend = (remainder << 16) | ((magnitude[i] >> (j * 16)) as u32 & 0xffff);
                let q = ((u64::from(dividend) * reciprocal) >> 33) as u32;
                remainder = dividend - q * u32::from(divisor);
                quotient[i] |= u64::from(q) << (j * 16);
            }
        }
        let positive = Self(quotient);
        let remainder = remainder as u8;
        let adjust = negative & !remainder.ct_eq(&0);
        // With a positive Euclidean remainder, a negative input needs the
        // quotient -(|x|/d)-1 when |x| is not an exact multiple of d.
        let negative_quotient = positive.neg().sub(Self::from_i8(adjust.unwrap_u8() as i8));
        let signed_quotient = Self::conditional_select(&positive, &negative_quotient, negative);
        let inverse = divisor - remainder;
        let signed_remainder = u8::conditional_select(&remainder, &inverse, adjust);
        (signed_quotient, signed_remainder)
    }

    // The product modulo 2^192, which is the two's-complement product of the
    // signed values.
    fn wrapping_mul(self, rhs: [u64; 3]) -> Self {
        let [x0, x1, x2] = self.0;
        let [c0, c1, c2] = rhs;
        let p00 = u128::from(x0) * u128::from(c0);
        let p01 = u128::from(x0) * u128::from(c1);
        let p10 = u128::from(x1) * u128::from(c0);
        let s1 = (p00 >> 64) + u128::from(p01 as u64) + u128::from(p10 as u64);
        let r2 = ((s1 >> 64) as u64)
            .wrapping_add((p01 >> 64) as u64)
            .wrapping_add((p10 >> 64) as u64)
            .wrapping_add(x0.wrapping_mul(c2))
            .wrapping_add(x1.wrapping_mul(c1))
            .wrapping_add(x2.wrapping_mul(c0));
        Self([p00 as u64, s1 as u64, r2])
    }

    // Division by 3 or 9 when it is exact, as it is in the recoding by the
    // verified residue table: multiplying by the divisor's inverse modulo
    // 2^192 (derived in Sage) then gives the quotient exactly, without a
    // long division (Granlund and Montgomery, "Division by invariant integers
    // using multiplication", PLDI 1994).
    fn div_exact(self, divisor: u8) -> Self {
        const INVERSE_3: [u64; 3] = [
            0xaaaa_aaaa_aaaa_aaab,
            0xaaaa_aaaa_aaaa_aaaa,
            0xaaaa_aaaa_aaaa_aaaa,
        ];
        const INVERSE_9: [u64; 3] = [
            0x8e38_e38e_38e3_8e39,
            0x38e3_8e38_e38e_38e3,
            0xe38e_38e3_8e38_e38e,
        ];
        self.wrapping_mul(if divisor == 9 { INVERSE_9 } else { INVERSE_3 })
    }

    // The Euclidean remainder modulo 9, without a long division. As 2^6 = 1
    // (mod 9), so are 2^12, 2^48 and 2^192. The limbs, read as an unsigned
    // U = x + 2^192 s with s the sign bit, give x = U - s = U + 8s (mod 9).
    // Summing U's four 48-bit chunks and 8s gives s1 < 2^51; summing s1's
    // five 12-bit chunks gives s2 < 2^15 < 9 * 2^16, in the range where the
    // reciprocal formula of quotient_remainder is exact (Sage checks these
    // bounds).
    fn mod9(self) -> u8 {
        const MASK48: u64 = (1 << 48) - 1;
        const MASK12: u64 = (1 << 12) - 1;
        let [l0, l1, l2] = self.0;
        let s1 = (l0 & MASK48)
            + ((l0 >> 48) | ((l1 << 16) & MASK48))
            + ((l1 >> 32) | ((l2 << 32) & MASK48))
            + (l2 >> 16)
            + 8 * (l2 >> 63);
        let s2 = (s1 & MASK12)
            + ((s1 >> 12) & MASK12)
            + ((s1 >> 24) & MASK12)
            + ((s1 >> 36) & MASK12)
            + (s1 >> 48);
        let q = (s2 * 0x38e3_8e39) >> 33;
        (s2 - 9 * q) as u8
    }

    #[cfg(test)]
    fn to_i128(self) -> i128 {
        assert_eq!(self.0[2], if self.0[1] >> 63 != 0 { u64::MAX } else { 0 });
        (u128::from(self.0[0]) | (u128::from(self.0[1]) << 64)) as i128
    }
}

impl ConditionallySelectable for Signed192 {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        Self(core::array::from_fn(|i| {
            u64::conditional_select(&a.0[i], &b.0[i], choice)
        }))
    }
}

// (a, b) represents a+bτ. Every one of the 54 residues with a mod 3 != 0
// occurs once in the Sage-derived table. A digit's metadata gives its base
// representative, ω rotation, sign, and integer coefficients.
#[derive(Clone, Copy)]
struct Digit {
    index: u8,
    rotation: u8,
    negative: u8,
    a: i8,
    b: i8,
}

const BASES: [(i8, i8); 6] = [(1, 0), (2, 0), (1, 1), (1, -2), (2, 1), (1, 2)];
// Rows are a mod 9 = 1,2,4,5,7,8, columns are b mod 9 = 0..8.
const METADATA: [[(u8, u8, u8); 9]; 6] = [
    [
        (5, 2, 0),
        (3, 0, 0),
        (0, 1, 0),
        (0, 0, 0),
        (3, 0, 0),
        (0, 1, 0),
        (0, 0, 0),
        (2, 0, 0),
        (5, 0, 0),
    ],
    [
        (4, 2, 0),
        (1, 1, 0),
        (0, 2, 1),
        (1, 0, 0),
        (1, 1, 0),
        (0, 2, 1),
        (1, 0, 0),
        (4, 0, 0),
        (5, 1, 1),
    ],
    [
        (3, 1, 0),
        (1, 2, 1),
        (3, 2, 0),
        (2, 1, 0),
        (1, 2, 1),
        (2, 2, 0),
        (3, 1, 0),
        (4, 1, 1),
        (2, 2, 0),
    ],
    [
        (2, 1, 1),
        (2, 2, 1),
        (4, 1, 0),
        (3, 1, 1),
        (3, 2, 1),
        (1, 2, 0),
        (2, 1, 1),
        (3, 2, 1),
        (1, 2, 0),
    ],
    [
        (4, 2, 1),
        (5, 1, 0),
        (4, 0, 1),
        (1, 0, 1),
        (0, 2, 0),
        (1, 1, 1),
        (1, 0, 1),
        (0, 2, 0),
        (1, 1, 1),
    ],
    [
        (5, 2, 1),
        (5, 0, 1),
        (2, 0, 1),
        (0, 0, 1),
        (0, 1, 1),
        (2, 0, 1),
        (0, 0, 1),
        (0, 1, 1),
        (3, 0, 1),
    ],
];
const RESIDUES: [u8; 6] = [1, 2, 4, 5, 7, 8];

const fn make_digit(metadata: (u8, u8, u8)) -> Digit {
    let (index, rotation, negative) = metadata;
    let (mut a, mut b) = BASES[index as usize];
    let mut i = 0;
    while i < rotation {
        let next_a = a + 3 * b;
        b = -a - 2 * b;
        a = next_a;
        i += 1;
    }
    if negative != 0 {
        a = -a;
        b = -b;
    }
    Digit {
        index,
        rotation,
        negative,
        a,
        b,
    }
}

const fn digits() -> [[Digit; 9]; 6] {
    let mut out = [[make_digit((0, 0, 0)); 9]; 6];
    let mut row = 0;
    while row < 6 {
        let mut column = 0;
        while column < 9 {
            out[row][column] = make_digit(METADATA[row][column]);
            column += 1;
        }
        row += 1;
    }
    out
}
const DIGITS: [[Digit; 9]; 6] = digits();

// A digit packed into one word, so that the constant-time table scan makes
// one masked selection per entry instead of one per field. Bits 0..8 and
// 8..16 hold a and b (two's complement), bits 16..19 the representative
// index, bits 19..21 the rotation and bit 21 the sign.
const fn pack(digit: Digit) -> u32 {
    (digit.a as u8 as u32)
        | ((digit.b as u8 as u32) << 8)
        | ((digit.index as u32) << 16)
        | ((digit.rotation as u32) << 19)
        | ((digit.negative as u32) << 21)
}

fn unpack(word: u32) -> Digit {
    Digit {
        index: ((word >> 16) & 7) as u8,
        rotation: ((word >> 19) & 3) as u8,
        negative: ((word >> 21) & 1) as u8,
        a: word as u8 as i8,
        b: (word >> 8) as u8 as i8,
    }
}

// The selection key of a residue pair (a mod 9, b mod 9).
const fn key(a9: u8, b9: u8) -> u8 {
    (a9 << 4) | b9
}

// The digit table flattened, each entry with its selection key.
const fn packed_digits() -> [(u8, u32); 54] {
    let mut out = [(0, 0); 54];
    let mut row = 0;
    while row < 6 {
        let mut column = 0;
        while column < 9 {
            out[9 * row + column] = (key(RESIDUES[row], column as u8), pack(DIGITS[row][column]));
            column += 1;
        }
        row += 1;
    }
    out
}
const PACKED_DIGITS: [(u8, u32); 54] = packed_digits();

fn choose_digit(a9: u8, b9: u8) -> Digit {
    let key = key(a9, b9);
    let mut result = pack(make_digit((0, 0, 0)));
    for (entry_key, entry) in PACKED_DIGITS.iter() {
        result = u32::conditional_select(&result, entry, key.ct_eq(entry_key));
    }
    unpack(result)
}

struct Recode {
    digits: [Digit; 54],
    top_rotation: u8,
    top_negative: u8,
    correction: Choice,
}

fn recode<C: Group + GlvConstants>(scalar: &C::Scalar) -> Recode {
    let (k1, k2) = decompose_signed::<C>(scalar);
    let mut a = Signed192::from_i128(k1).add(Signed192::from_i128(k2));
    let mut b = Signed192::from_i128(k2).neg();
    let residue = a.mod9();
    let correction = residue.ct_eq(&0) | residue.ct_eq(&3) | residue.ct_eq(&6);
    a = a.add(Signed192::conditional_select(
        &Signed192::ZERO,
        &Signed192::from_i8(1),
        correction,
    ));
    let mut out = [make_digit((0, 0, 0)); 54];
    for digit in &mut out {
        let d = choose_digit(a.mod9(), b.mod9());
        let x = a.sub(Signed192::from_i8(d.a));
        let y = b.sub(Signed192::from_i8(d.b));
        // (x+yτ)/τ³ = ((x+2y)/3) − ((2x+3y)/9)τ.
        a = x.add(y.twice()).div_exact(3);
        b = x.twice().add(y.thrice()).div_exact(9).neg();
        *digit = d;
    }
    // Sage's norm bound proves that the final nondivisible state is a unit.
    // Scan all six units, including when the scalar is zero.
    let mut top_rotation = 0u8;
    let mut top_negative = 0u8;
    for rotation in 0..3u8 {
        for negative in 0..2u8 {
            let unit = make_digit((0, rotation, negative));
            let match_unit = a.0[0].ct_eq(&(unit.a as i64 as u64))
                & a.0[1].ct_eq(&(if unit.a < 0 { u64::MAX } else { 0 }))
                & a.0[2].ct_eq(&(if unit.a < 0 { u64::MAX } else { 0 }))
                & b.0[0].ct_eq(&(unit.b as i64 as u64))
                & b.0[1].ct_eq(&(if unit.b < 0 { u64::MAX } else { 0 }))
                & b.0[2].ct_eq(&(if unit.b < 0 { u64::MAX } else { 0 }));
            top_rotation = u8::conditional_select(&top_rotation, &rotation, match_unit);
            top_negative = u8::conditional_select(&top_negative, &negative, match_unit);
        }
    }
    Recode {
        digits: out,
        top_rotation,
        top_negative,
        correction,
    }
}

// A table point with its X coordinate's first two omega rotations, X and
// zeta*X. The third, zeta^2*X = -X - zeta*X, is formed after the selection,
// so the scan selects four field elements per entry instead of five.
#[derive(Clone, Copy)]
struct Rotated<F: Field> {
    x: [F; 2],
    y: F,
    z: F,
}

impl<F: Field + WithSmallOrderMulGroup<3>> Rotated<F> {
    fn new(p: Homogeneous<F>) -> Self {
        Self {
            x: [p.x, p.x * F::ZETA],
            y: p.y,
            z: p.z,
        }
    }
}

// The base-field constants 1 - beta and (1 - beta)^3 of the tau
// endomorphism, with beta = ZETA, evaluated at compile time from ZETA with
// the fields' const arithmetic.
pub(crate) trait TauField: FormulaField + WithSmallOrderMulGroup<3> {
    const ONE_MINUS_ZETA: Self;
    const ONE_MINUS_ZETA_CUBED: Self;
}

macro_rules! tau_field {
    ($field:ident) => {
        impl TauField for $field {
            const ONE_MINUS_ZETA: Self = $field::one().sub(&$field::ZETA);
            const ONE_MINUS_ZETA_CUBED: Self =
                Self::ONE_MINUS_ZETA.square().mul(&Self::ONE_MINUS_ZETA);
        }
    };
}
tau_field!(Fp);
tau_field!(Fq);

fn tau<F: TauField>(p: Homogeneous<F>) -> Homogeneous<F> {
    // The complete isogeny ψ : E_5 → E_-135 followed by the isomorphism
    // (X:Y:Z) ↦ ((1−β)X:Y:(1−β)^3 Z) is 1−ω on the native curve.
    let y2 = p.y.square();
    let z2 = p.z.square();
    let (m5, m15, m45) = (z2.mul_small(5), z2.mul_small(15), z2.mul_small(45));
    Homogeneous {
        x: p.x * (y2 + m15) * F::ONE_MINUS_ZETA,
        y: p.y * (y2 - m45),
        z: p.z * (y2 - m5) * F::ONE_MINUS_ZETA_CUBED,
    }
}

fn representatives<F: FormulaField>(p: Homogeneous<F>, t: Homogeneous<F>) -> [Homogeneous<F>; 6] {
    let two_p = p.double();
    let p_plus_t = p.add(&t);
    let two_t = t.double();
    [
        p,
        two_p,
        p_plus_t,
        p.add(&Homogeneous {
            y: -two_t.y,
            ..two_t
        }),
        two_p.add(&t),
        p.add(&two_t),
    ]
}

// (rotation + offset) mod 3 for rotation in 0..3 and offset in 0..2, in
// bit arithmetic: a remainder operation on the secret rotation can compile
// to a division instruction (wasm32 emits i32.rem_u for it). The sum is in
// 0..=3, and it is 3 exactly when both of its low bits are set.
fn add_rotation(rotation: u8, offset: u8) -> u8 {
    let sum = rotation + offset;
    sum - 3 * (sum & (sum >> 1) & 1)
}

fn rotated_select<F: Field>(
    points: &[Rotated<F>; 6],
    index: u8,
    rotation: u8,
    sign: u8,
) -> Homogeneous<F> {
    let mut x = [F::ZERO; 2];
    let mut y = F::ONE;
    let mut z = F::ZERO;
    for (i, point) in points.iter().enumerate() {
        let choice = index.ct_eq(&(i as u8));
        for (out, &candidate) in x.iter_mut().zip(point.x.iter()) {
            *out = F::conditional_select(out, &candidate, choice);
        }
        y = F::conditional_select(&y, &point.y, choice);
        z = F::conditional_select(&z, &point.z, choice);
    }
    // 1 + zeta + zeta^2 = 0, including for the identity's zero X.
    let x = [x[0], x[1], -x[0] - x[1]];
    let mut selected_x = F::ZERO;
    for (i, candidate) in x.iter().enumerate() {
        selected_x = F::conditional_select(&selected_x, candidate, rotation.ct_eq(&(i as u8)));
    }
    let negative_y = -y;
    Homogeneous {
        x: selected_x,
        y: F::conditional_select(&y, &negative_y, Choice::from(sign)),
        z,
    }
}

impl<F: TauField> Homogeneous<F> {
    /// Internal regular τ-adic multiplication for either Pasta scalar field.
    /// Caller supplies the curve whose scalar field's ZETA is paired with this curve's β.
    pub(crate) fn multiply_tau<C: Group + GlvConstants>(&self, scalar: &C::Scalar) -> Self {
        let recoding = recode::<C>(scalar);
        let tau_p = tau(*self);
        let base = representatives(*self, tau_p);
        // tau is a group endomorphism, so the representatives built from
        // tau(P) and tau^2(P) are the tau-images of those built from P and
        // tau(P). One tau per entry costs less than the doublings and
        // additions of a second `representatives`, and the first entry is
        // tau(P) itself.
        let tau_base = [
            tau_p,
            tau(base[1]),
            tau(base[2]),
            tau(base[3]),
            tau(base[4]),
            tau(base[5]),
        ];
        let table = [base.map(Rotated::new), tau_base.map(Rotated::new)];

        // τ^(3j) = 3^floor(3j/2) (-ω)^floor(3j/2) τ^(j mod 2).
        // The top unit is at τ^162, hence exponent 81 for (-ω).
        let mut acc = rotated_select(
            &table[0],
            0,
            recoding.top_rotation,
            recoding.top_negative ^ 1,
        ); // (-ω)^81 = -1.
        let mut previous = 81usize;
        for j in (0..54).rev() {
            let exponent = 3 * j / 2;
            for _ in exponent..previous {
                acc = acc.triple();
            }
            let digit = recoding.digits[j];
            // exponent mod 3 is j mod 2: floor(3j/2) is 3(j/2) for even j
            // and 3(j-1)/2 + 1 for odd j.
            let rotation = add_rotation(digit.rotation, (j & 1) as u8);
            let sign = digit.negative ^ (exponent as u8 & 1);
            let addend = rotated_select(&table[j & 1], digit.index, rotation, sign);
            acc = acc.add(&addend);
            previous = exponent;
        }
        let minus_p = Self {
            y: -self.y,
            ..*self
        };
        let correction = Self::conditional_select(&Self::identity(), &minus_p, recoding.correction);
        acc.add(&correction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Fp, Fq, group::CurveAffine, pallas, vesta};
    use ff::PrimeField;
    use group::{Curve, Group};
    use proptest::prelude::*;

    fn signed<F: PrimeField>(x: i128) -> F {
        let positive = F::from_u128(x.unsigned_abs());
        if x < 0 { -positive } else { positive }
    }

    fn check_recode<C>(scalar: C::Scalar)
    where
        C: Group + GlvConstants,
        C::Scalar: PrimeField + WithSmallOrderMulGroup<3> + core::fmt::Debug,
    {
        let (k1, k2) = decompose_signed::<C>(&scalar);
        let r = recode::<C>(&scalar);
        let unit = make_digit((0, r.top_rotation, r.top_negative));
        let mut a = i128::from(unit.a);
        let mut b = i128::from(unit.b);
        // Reconstruct modulo the scalar order (i128 cannot hold the input).
        let tau = C::Scalar::ONE - C::Scalar::ZETA;
        let mut reconstructed = signed::<C::Scalar>(a) + signed::<C::Scalar>(b) * tau;
        for digit in r.digits.iter().rev() {
            reconstructed = reconstructed * (tau.square() * tau)
                + signed::<C::Scalar>(i128::from(digit.a))
                + signed::<C::Scalar>(i128::from(digit.b)) * tau;
        }
        reconstructed -= C::Scalar::from(u64::from(r.correction.unwrap_u8()));
        assert_eq!(reconstructed, scalar);
        assert_eq!(
            signed::<C::Scalar>(k1) + signed::<C::Scalar>(k2) * C::Scalar::ZETA,
            scalar
        );
        a = i128::from(unit.a);
        b = i128::from(unit.b);
        assert_eq!(a * a + 3 * a * b + 3 * b * b, 1);
    }

    #[test]
    fn signed192_arithmetic_and_exact_divisions() {
        for x in [
            -i128::MAX,
            i128::MIN,
            -999,
            -9,
            -1,
            0,
            1,
            3,
            9,
            999,
            i128::MAX,
        ] {
            let w = Signed192::from_i128(x);
            assert_eq!(w.0[2], if x < 0 { u64::MAX } else { 0 });
            assert_eq!(w.mod9(), x.rem_euclid(9) as u8);
            let (q3, r3) = w.quotient_remainder(3);
            let (q9, r9) = w.quotient_remainder(9);
            assert_eq!(q3.to_i128(), x.div_euclid(3));
            assert_eq!(q9.to_i128(), x.div_euclid(9));
            assert_eq!(r3, x.rem_euclid(3) as u8);
            assert_eq!(r9, x.rem_euclid(9) as u8);
            assert_eq!(w.neg().add(w).0, [0; 3]);
        }
        // Exercise the actual 130-bit state range, not just i128 inputs.
        for w in [
            Signed192([u64::MAX, u64::MAX, 1]),
            Signed192([0, 0, 2]),
            Signed192([u64::MAX, u64::MAX, 3]),
        ] {
            for n in [w, w.neg()] {
                for divisor in [3, 9] {
                    let (q, r) = n.quotient_remainder(divisor);
                    assert!(r < divisor);
                    let multiple = if divisor == 3 {
                        q.thrice()
                    } else {
                        q.thrice().thrice()
                    };
                    let remainder = Signed192::from_i8(r as i8);
                    assert_eq!(multiple.add(remainder).0, n.0);
                }
            }
        }
    }

    #[test]
    fn sage_lut_all_residues() {
        for (row, a) in RESIDUES.iter().enumerate() {
            for b in 0..9u8 {
                let d = choose_digit(*a, b);
                assert_eq!(
                    (d.index, d.rotation, d.negative),
                    METADATA[row][usize::from(b)]
                );
                assert_eq!(i16::from(d.a).rem_euclid(9), i16::from(*a));
                assert_eq!(i16::from(d.b).rem_euclid(3), i16::from(b % 3));
                assert_ne!(
                    (((i16::from(*a) - i16::from(d.a)) + 2 * (i16::from(b) - i16::from(d.b))) / 3)
                        .rem_euclid(3),
                    0
                );
            }
        }
    }

    #[test]
    fn packed_digit_table() {
        for (row, entries) in DIGITS.iter().enumerate() {
            for (column, d) in entries.iter().enumerate() {
                let u = unpack(pack(*d));
                assert_eq!(
                    (u.index, u.rotation, u.negative, u.a, u.b),
                    (d.index, d.rotation, d.negative, d.a, d.b)
                );
                let (k, word) = PACKED_DIGITS[9 * row + column];
                assert_eq!((k, word), (key(RESIDUES[row], column as u8), pack(*d)));
            }
        }
        // Keys are distinct over the whole 9x9 residue square, so a scan
        // selects at most one entry, and exactly one for a valid key.
        for a9 in 0..9u8 {
            for b9 in 0..9u8 {
                let matches = PACKED_DIGITS
                    .iter()
                    .filter(|(k, _)| *k == key(a9, b9))
                    .count();
                assert_eq!(matches, usize::from(a9 % 3 != 0));
            }
        }
    }

    fn check_tau_constants<F: TauField + core::fmt::Debug>() {
        let t = F::ONE - F::ZETA;
        assert_eq!(F::ONE_MINUS_ZETA, t);
        assert_eq!(F::ONE_MINUS_ZETA_CUBED, t.square() * t);
        // (1 - beta)^3 = -3 - 6 beta, as beta^2 + beta + 1 = 0.
        let three = F::ONE + F::ONE + F::ONE;
        assert_eq!(F::ONE_MINUS_ZETA_CUBED, -three - (three + three) * F::ZETA);
    }

    #[test]
    fn rotation_offsets() {
        for j in 0..54usize {
            assert_eq!((3 * j / 2) % 3, j & 1);
        }
        for rotation in 0..3u8 {
            for offset in 0..2u8 {
                assert_eq!(add_rotation(rotation, offset), (rotation + offset) % 3);
            }
        }
    }

    #[test]
    fn tau_constants() {
        check_tau_constants::<Fp>();
        check_tau_constants::<Fq>();
    }

    #[test]
    fn scalar_boundaries() {
        for s in [
            Fp::ZERO,
            Fp::ONE,
            -Fp::ONE,
            Fp::ZETA,
            Fp::from(2u64),
            Fp::from_u128((1u128 << 127) - 1),
        ] {
            check_recode::<vesta::Point>(s);
        }
        for s in [
            Fq::ZERO,
            Fq::ONE,
            -Fq::ONE,
            Fq::ZETA,
            Fq::from(2u64),
            Fq::from_u128((1u128 << 127) - 1),
        ] {
            check_recode::<pallas::Point>(s);
        }
    }

    #[test]
    fn mod9_boundaries() {
        // The extremes of each 48-bit and 12-bit chunk, and of the sign.
        for limbs in [
            [0, 0, 0],
            [u64::MAX; 3],
            [u64::MAX, u64::MAX, u64::MAX >> 1],
            [0, 0, 1 << 63],
            [1, 0, 1 << 63],
            [(1 << 48) - 1, 0, 0],
            [u64::MAX, u64::MAX, 0],
            [0, 0, u64::MAX >> 16],
        ] {
            let w = Signed192(limbs);
            assert_eq!(w.mod9(), w.quotient_remainder(9).1, "{limbs:x?}");
        }
    }

    proptest! {
        #[test]
        fn div_exact_matches_long_division(limbs in any::<[u64; 3]>()) {
            // Any signed quotient below 2^186 in magnitude, multiplied back
            // up, is an exact multiple of 3 and of 9 within signed 192 bits.
            let q = Signed192([limbs[0], limbs[1], ((limbs[2] as i64) >> 5) as u64]);
            let n3 = q.thrice();
            let n9 = n3.thrice();
            prop_assert_eq!(n3.div_exact(3).0, q.0);
            prop_assert_eq!(n9.div_exact(9).0, q.0);
            let (long_quotient, remainder) = n9.quotient_remainder(9);
            prop_assert_eq!((long_quotient.0, remainder), (q.0, 0));
        }

        #[test]
        fn mod9_matches_long_division(limbs in any::<[u64; 3]>()) {
            let w = Signed192(limbs);
            prop_assert_eq!(w.mod9(), w.quotient_remainder(9).1);
        }

        #[test]
        fn full_width_fp(words in any::<[u64;4]>()) {
            let scalar = Fp::from_raw(words);
            check_recode::<vesta::Point>(scalar);
            assert_vesta(1, scalar);
        }
        #[test]
        fn full_width_fq(words in any::<[u64;4]>()) {
            let scalar = Fq::from_raw(words);
            check_recode::<pallas::Point>(scalar);
            assert_pallas(1, scalar);
        }
    }

    fn equal<F: Field>(a: Homogeneous<F>, b: Homogeneous<F>) {
        assert_eq!(a.z.is_zero().unwrap_u8(), b.z.is_zero().unwrap_u8());
        assert_eq!(a.x * b.z, b.x * a.z);
        assert_eq!(a.y * b.z, b.y * a.z);
    }

    fn binary<P: Group + Copy>(point: P, bytes: &[u8; 32]) -> P {
        let mut acc = P::identity();
        for bit in bytes
            .iter()
            .rev()
            .flat_map(|byte| (0..8).rev().map(move |i| (byte >> i) & 1))
        {
            acc = acc.double();
            if bit != 0 {
                acc += point;
            }
        }
        acc
    }

    fn assert_pallas(scale: u64, scalar: Fq) {
        let generator = pallas::Point::generator();
        let point = binary(generator, &Fq::from(scale).to_repr());
        let point_affine = point.to_affine();
        let (px, py) = point_affine.raw_xy_for_tests();
        let p = Homogeneous::<Fp>::from_affine(px, py, point_affine.is_identity());
        let repr = scalar.to_repr();
        let expected = binary(point, &repr);
        let actual = p.multiply_tau::<pallas::Point>(&scalar);
        equal(actual, p.multiply(&repr));
        let expected_affine = expected.to_affine();
        let (ex, ey) = expected_affine.raw_xy_for_tests();
        equal(
            actual,
            Homogeneous::from_affine(ex, ey, expected_affine.is_identity()),
        );
        let omega = Homogeneous {
            x: p.x * Fp::ZETA,
            ..p
        };
        equal(
            tau(p),
            p.add(&Homogeneous {
                y: -omega.y,
                ..omega
            }),
        );
    }

    fn assert_vesta(scale: u64, scalar: Fp) {
        let generator = vesta::Point::generator();
        let point = binary(generator, &Fp::from(scale).to_repr());
        let point_affine = point.to_affine();
        let (px, py) = point_affine.raw_xy_for_tests();
        let p = Homogeneous::<Fq>::from_affine(px, py, point_affine.is_identity());
        let repr = scalar.to_repr();
        let expected = binary(point, &repr);
        let actual = p.multiply_tau::<vesta::Point>(&scalar);
        equal(actual, p.multiply(&repr));
        let expected_affine = expected.to_affine();
        let (ex, ey) = expected_affine.raw_xy_for_tests();
        equal(
            actual,
            Homogeneous::from_affine(ex, ey, expected_affine.is_identity()),
        );
        let omega = Homogeneous {
            x: p.x * Fq::ZETA,
            ..p
        };
        equal(
            tau(p),
            p.add(&Homogeneous {
                y: -omega.y,
                ..omega
            }),
        );
    }
    #[test]
    fn multiplication_matches_binary_both_curves() {
        for i in 0..20u64 {
            let ps = Fq::from(i).square() + Fq::ZETA;
            let vs = Fp::from(i).square() + Fp::ZETA;
            assert_pallas(i, ps);
            assert_vesta(i, vs);
        }
        for s in [Fq::ZERO, Fq::ONE, -Fq::ONE, Fq::ZETA] {
            assert_pallas(1, s);
        }
        for s in [Fp::ZERO, Fp::ONE, -Fp::ONE, Fp::ZETA] {
            assert_vesta(1, s);
        }
        assert_pallas(0, -Fq::ONE);
        assert_vesta(0, -Fp::ONE);
    }
}
