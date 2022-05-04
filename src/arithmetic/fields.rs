//! This module contains the `Field` abstraction that allows us to write
//! code that generalizes over a pair of fields.

use core::mem::size_of;

use static_assertions::const_assert;
use subtle::{Choice, ConditionallySelectable, CtOption};

use super::Group;

use core::assert;

#[cfg(feature = "sqrt-table")]
use alloc::{boxed::Box, vec::Vec};
#[cfg(feature = "sqrt-table")]
use core::marker::PhantomData;

const_assert!(size_of::<usize>() >= 4);

/// A trait that exposes additional operations related to calculating square roots of
/// prime-order finite fields.
pub trait SqrtRatio: ff::PrimeField {
    /// The value $(T-1)/2$ such that $2^S \cdot T = p - 1$ with $T$ odd.
    const T_MINUS1_OVER2: [u64; 4];

    /// Raise this field element to the power [`Self::T_MINUS1_OVER2`].
    ///
    /// Field implementations may override this to use an efficient addition chain.
    fn pow_by_t_minus1_over2(&self) -> Self {
        ff::Field::pow_vartime(self, &Self::T_MINUS1_OVER2)
    }

    /// Gets the lower 32 bits of this field element when expressed
    /// canonically.
    fn get_lower_32(&self) -> u32;

    /// Computes:
    ///
    /// - $(\textsf{true}, \sqrt{\textsf{num}/\textsf{div}})$, if $\textsf{num}$ and
    ///   $\textsf{div}$ are nonzero and $\textsf{num}/\textsf{div}$ is a square in the
    ///   field;
    /// - $(\textsf{true}, 0)$, if $\textsf{num}$ is zero;
    /// - $(\textsf{false}, 0)$, if $\textsf{num}$ is nonzero and $\textsf{div}$ is zero;
    /// - $(\textsf{false}, \sqrt{G_S \cdot \textsf{num}/\textsf{div}})$, if
    ///   $\textsf{num}$ and $\textsf{div}$ are nonzero and $\textsf{num}/\textsf{div}$ is
    ///   a nonsquare in the field;
    ///
    /// where $G_S$ is a non-square.
    ///
    /// For `pasta_curves`, $G_S$ is currently [`ff::PrimeField::root_of_unity`], a
    /// generator of the order $2^S$ subgroup. Users of this crate should not rely on this
    /// generator being fixed; it may be changed in future crate versions to simplify the
    /// implementation of the SSWU hash-to-curve algorithm.
    ///
    /// The choice of root from sqrt is unspecified.
    fn sqrt_ratio(num: &Self, div: &Self) -> (Choice, Self) {
        // General implementation:
        //
        // a = num * inv0(div)
        //   = {    0    if div is zero
        //     { num/div otherwise
        //
        // b = G_S * a
        //   = {      0      if div is zero
        //     { G_S*num/div otherwise
        //
        // Since G_S is non-square, a and b are either both zero (and both square), or
        // only one of them is square. We can therefore choose the square root to return
        // based on whether a is square, but for the boolean output we need to handle the
        // num != 0 && div == 0 case specifically.

        let a = div.invert().unwrap_or_else(Self::zero) * num;
        let b = a * Self::root_of_unity();
        let sqrt_a = a.sqrt();
        let sqrt_b = b.sqrt();

        let num_is_zero = num.is_zero();
        let div_is_zero = div.is_zero();
        let is_square = sqrt_a.is_some();
        let is_nonsquare = sqrt_b.is_some();
        assert!(bool::from(
            num_is_zero | div_is_zero | (is_square ^ is_nonsquare)
        ));

        (
            is_square & !(!num_is_zero & div_is_zero),
            CtOption::conditional_select(&sqrt_b, &sqrt_a, is_square).unwrap(),
        )
    }

    /// Equivalent to `Self::sqrt_ratio(self, one())`.
    fn sqrt_alt(&self) -> (Choice, Self) {
        Self::sqrt_ratio(self, &Self::one())
    }
}

/// This trait is a common interface for dealing with elements of a finite
/// field.
pub trait FieldExt: SqrtRatio + From<bool> + Ord + Group<Scalar = Self> {
    /// Modulus of the field written as a string for display purposes
    const MODULUS: &'static str;

    /// Inverse of `PrimeField::root_of_unity()`
    const ROOT_OF_UNITY_INV: Self;

    /// Generator of the $t-order$ multiplicative subgroup
    const DELTA: Self;

    /// Inverse of $2$ in the field.
    const TWO_INV: Self;

    /// Element of multiplicative order $3$.
    const ZETA: Self;

    /// Obtains a field element congruent to the integer `v`.
    fn from_u128(v: u128) -> Self;

    /// Obtains a field element that is congruent to the provided little endian
    /// byte representation of an integer.
    fn from_bytes_wide(bytes: &[u8; 64]) -> Self;

    /// Exponentiates `self` by `by`, where `by` is a little-endian order
    /// integer exponent.
    fn pow(&self, by: &[u64; 4]) -> Self {
        let mut res = Self::one();
        for e in by.iter().rev() {
            for i in (0..64).rev() {
                res = res.square();
                let mut tmp = res;
                tmp *= self;
                res.conditional_assign(&tmp, (((*e >> i) & 0x1) as u8).into());
            }
        }
        res
    }

    /// Gets the lower 128 bits of this field element when expressed
    /// canonically.
    fn get_lower_128(&self) -> u128;
}

/// Tonelli–Shanks' square-root algorithm for `p mod 16 = 1`.
///
/// https://eprint.iacr.org/2012/685.pdf (page 12, algorithm 5)
///
/// `tm1d2` should be set to `(t - 1) // 2`, where `t = (modulus - 1) >> F::S`.
#[cfg(not(feature = "sqrt-table"))]
#[cfg_attr(docsrs, doc(cfg(not(feature = "sqrt-table"))))]
pub(crate) fn sqrt_tonelli_shanks<F: ff::PrimeField, S: AsRef<[u64]>>(
    f: &F,
    tm1d2: S,
) -> CtOption<F> {
    use subtle::ConstantTimeEq;

    // w = self^((t - 1) // 2)
    let w = f.pow_vartime(tm1d2);

    let mut v = F::S;
    let mut x = w * f;
    let mut b = x * w;

    // Initialize z as the 2^S root of unity.
    let mut z = F::root_of_unity();

    for max_v in (1..=F::S).rev() {
        let mut k = 1;
        let mut tmp = b.square();
        let mut j_less_than_v: Choice = 1.into();

        for j in 2..max_v {
            let tmp_is_one = tmp.ct_eq(&F::one());
            let squared = F::conditional_select(&tmp, &z, tmp_is_one).square();
            tmp = F::conditional_select(&squared, &tmp, tmp_is_one);
            let new_z = F::conditional_select(&z, &squared, tmp_is_one);
            j_less_than_v &= !j.ct_eq(&v);
            k = u32::conditional_select(&j, &k, tmp_is_one);
            z = F::conditional_select(&z, &new_z, j_less_than_v);
        }

        let result = x * z;
        x = F::conditional_select(&result, &x, b.ct_eq(&F::one()));
        z = z.square();
        b *= z;
        v = k;
    }

    CtOption::new(
        x,
        (x * x).ct_eq(f), // Only return Some if it's the square root.
    )
}

/// Parameters for a perfect hash function used in square root computation.
#[cfg(feature = "sqrt-table")]
#[cfg_attr(docsrs, doc(cfg(feature = "sqrt-table")))]
#[derive(Debug)]
struct SqrtHasher<F: FieldExt> {
    hash_xor: u32,
    hash_mod: usize,
    marker: PhantomData<F>,
}

#[cfg(feature = "sqrt-table")]
impl<F: FieldExt> SqrtHasher<F> {
    /// Returns a perfect hash of x for use with SqrtTables::inv.
    fn hash(&self, x: &F) -> usize {
        // This is just the simplest constant-time perfect hash construction that could
        // possibly work. The 32 low-order bits are unique within the 2^S order subgroup,
        // then the xor acts as "salt" to injectively randomize the output when taken modulo
        // `hash_mod`. Since the table is small, we do not need anything more complicated.
        ((x.get_lower_32() ^ self.hash_xor) as usize) % self.hash_mod
    }
}

/// Tables used for square root computation.
#[cfg(feature = "sqrt-table")]
#[cfg_attr(docsrs, doc(cfg(feature = "sqrt-table")))]
#[derive(Debug)]
pub struct SqrtTables<F: FieldExt> {
    hasher: SqrtHasher<F>,
    inv: Vec<u8>,
    g0: Box<[F; 256]>,
    g1: Box<[F; 256]>,
    g2: Box<[F; 256]>,
    g3: Box<[F; 129]>,
}

#[cfg(feature = "sqrt-table")]
impl<F: FieldExt> SqrtTables<F> {
    /// Build tables given parameters for the perfect hash.
    pub fn new(hash_xor: u32, hash_mod: usize) -> Self {
        use alloc::vec;

        let hasher = SqrtHasher {
            hash_xor,
            hash_mod,
            marker: PhantomData,
        };

        let mut gtab = (0..4).scan(F::root_of_unity(), |gi, _| {
            // gi == ROOT_OF_UNITY^(256^i)
            let gtab_i: Vec<F> = (0..256)
                .scan(F::one(), |acc, _| {
                    let res = *acc;
                    *acc *= *gi;
                    Some(res)
                })
                .collect();
            *gi = gtab_i[255] * *gi;
            Some(gtab_i)
        });
        let gtab_0 = gtab.next().unwrap();
        let gtab_1 = gtab.next().unwrap();
        let gtab_2 = gtab.next().unwrap();
        let mut gtab_3 = gtab.next().unwrap();
        assert_eq!(gtab.next(), None);

        // Now invert gtab[3].
        let mut inv: Vec<u8> = vec![1; hash_mod];
        for (j, gtab_3_j) in gtab_3.iter().enumerate() {
            let hash = hasher.hash(gtab_3_j);
            // 1 is the last value to be assigned, so this ensures there are no collisions.
            assert!(inv[hash] == 1);
            inv[hash] = ((256 - j) & 0xFF) as u8;
        }

        gtab_3.truncate(129);

        SqrtTables::<F> {
            hasher,
            inv,
            g0: gtab_0.into_boxed_slice().try_into().unwrap(),
            g1: gtab_1.into_boxed_slice().try_into().unwrap(),
            g2: gtab_2.into_boxed_slice().try_into().unwrap(),
            g3: gtab_3.into_boxed_slice().try_into().unwrap(),
        }
    }

    /// Computes:
    ///
    /// * (true,  sqrt(num/div)),                 if num and div are nonzero and num/div is a square in the field;
    /// * (true,  0),                             if num is zero;
    /// * (false, 0),                             if num is nonzero and div is zero;
    /// * (false, sqrt(ROOT_OF_UNITY * num/div)), if num and div are nonzero and num/div is a nonsquare in the field;
    ///
    /// where ROOT_OF_UNITY is a generator of the order 2^n subgroup (and therefore a nonsquare).
    ///
    /// The choice of root from sqrt is unspecified.
    pub fn sqrt_ratio(&self, num: &F, div: &F) -> (Choice, F) {
        // Based on:
        // * [Sarkar2020](https://eprint.iacr.org/2020/1407)
        // * [BDLSY2012](https://cr.yp.to/papers.html#ed25519)
        //
        // We need to calculate uv and v, where v = u^((T-1)/2), u = num/div, and p-1 = T * 2^S.
        // We can rewrite as follows:
        //
        //      v = (num/div)^((T-1)/2)
        //        = num^((T-1)/2) * div^(p-1 - (T-1)/2)    [Fermat's Little Theorem]
        //        =       "       * div^(T * 2^S - (T-1)/2)
        //        =       "       * div^((2^(S+1) - 1)*(T-1)/2 + 2^S)
        //        = (num * div^(2^(S+1) - 1))^((T-1)/2) * div^(2^S)
        //
        // Let  w = (num * div^(2^(S+1) - 1))^((T-1)/2) * div^(2^S - 1).
        // Then v = w * div, and uv = num * v / div = num * w.
        //
        // We calculate:
        //
        //      s = div^(2^S - 1) using an addition chain
        //      t = div^(2^(S+1) - 1) = s^2 * div
        //      w = (num * t)^((T-1)/2) * s using another addition chain
        //
        // then u and uv as above. The addition chains are given in
        // https://github.com/zcash/pasta/blob/master/addchain_sqrt.py .
        // The overall cost of this part is similar to a single full-width exponentiation,
        // regardless of S.

        let sqr = |x: F, i: u32| (0..i).fold(x, |x, _| x.square());

        // s = div^(2^S - 1)
        let s = (0..5).fold(*div, |d: F, i| sqr(d, 1 << i) * d);

        // t == div^(2^(S+1) - 1)
        let t = s.square() * div;

        // w = (num * t)^((T-1)/2) * s
        let w = (t * num).pow_by_t_minus1_over2() * s;

        // v == u^((T-1)/2)
        let v = w * div;

        // uv = u * v
        let uv = w * num;

        let res = self.sqrt_common(&uv, &v);

        let sqdiv = res.square() * div;
        let is_square = (sqdiv - num).is_zero();
        let is_nonsquare = (sqdiv - F::root_of_unity() * num).is_zero();
        assert!(bool::from(
            num.is_zero() | div.is_zero() | (is_square ^ is_nonsquare)
        ));

        (is_square, res)
    }

    /// Same as sqrt_ratio(u, one()) but more efficient.
    pub fn sqrt_alt(&self, u: &F) -> (Choice, F) {
        let v = u.pow_by_t_minus1_over2();
        let uv = *u * v;

        let res = self.sqrt_common(&uv, &v);

        let sq = res.square();
        let is_square = (sq - u).is_zero();
        let is_nonsquare = (sq - F::root_of_unity() * u).is_zero();
        assert!(bool::from(u.is_zero() | (is_square ^ is_nonsquare)));

        (is_square, res)
    }

    /// Common part of sqrt_ratio and sqrt_alt: return their result given v = u^((T-1)/2) and uv = u * v.
    fn sqrt_common(&self, uv: &F, v: &F) -> F {
        let sqr = |x: F, i: u32| (0..i).fold(x, |x, _| x.square());
        let inv = |x: F| self.inv[self.hasher.hash(&x)] as usize;

        let x3 = *uv * v;
        let x2 = sqr(x3, 8);
        let x1 = sqr(x2, 8);
        let x0 = sqr(x1, 8);

        // i = 0, 1
        let mut t_ = inv(x0); // = t >> 16
                              // 1 == x0 * ROOT_OF_UNITY^(t_ << 24)
        assert!(t_ < 0x100);
        let alpha = x1 * self.g2[t_];

        // i = 2
        t_ += inv(alpha) << 8; // = t >> 8
                               // 1 == x1 * ROOT_OF_UNITY^(t_ << 16)
        assert!(t_ < 0x10000);
        let alpha = x2 * self.g1[t_ & 0xFF] * self.g2[t_ >> 8];

        // i = 3
        t_ += inv(alpha) << 16; // = t
                                // 1 == x2 * ROOT_OF_UNITY^(t_ << 8)
        assert!(t_ < 0x1000000);
        let alpha = x3 * self.g0[t_ & 0xFF] * self.g1[(t_ >> 8) & 0xFF] * self.g2[t_ >> 16];

        t_ += inv(alpha) << 24; // = t << 1
                                // 1 == x3 * ROOT_OF_UNITY^t_
        t_ = (t_ + 1) >> 1;
        assert!(t_ <= 0x80000000);

        *uv * self.g0[t_ & 0xFF]
            * self.g1[(t_ >> 8) & 0xFF]
            * self.g2[(t_ >> 16) & 0xFF]
            * self.g3[t_ >> 24]
    }
}

/// Compute a + b + carry, returning the result and the new carry over.
#[inline(always)]
pub(crate) const fn adc(a: u64, b: u64, carry: u64) -> (u64, u64) {
    let ret = (a as u128) + (b as u128) + (carry as u128);
    (ret as u64, (ret >> 64) as u64)
}

/// Compute a - (b + borrow), returning the result and the new borrow.
#[inline(always)]
pub(crate) const fn sbb(a: u64, b: u64, borrow: u64) -> (u64, u64) {
    let ret = (a as u128).wrapping_sub((b as u128) + ((borrow >> 63) as u128));
    (ret as u64, (ret >> 64) as u64)
}

/// Compute a + (b * c) + carry, returning the result and the new carry over.
#[inline(always)]
pub(crate) const fn mac(a: u64, b: u64, c: u64, carry: u64) -> (u64, u64) {
    let ret = (a as u128) + ((b as u128) * (c as u128)) + (carry as u128);
    (ret as u64, (ret >> 64) as u64)
}
