//! Four-way AVX2 arithmetic for the square-root exponentiation chain.
//!
//! [`crate::pallas::Affine::batch_from_bytes_vartime`] spends nearly all of its time in
//! one square root per point, a chain of about 250 squarings. A scalar chain
//! already saturates the x86 multiplier, so this module runs four chains in one
//! set of AVX2 registers instead.
//!
//! # Availability
//!
//! This module requires `asm` and `sqrt-table`, and respects `pasta_curves_noasm`.
//! Entry points check AVX2, XSAVE/OSXSAVE and enabled XMM/YMM state at runtime,
//! including in no_std builds. Scalar interleaved decoding remains the fallback;
//! BMI2/ADX availability alone is deliberately not treated as an AVX2 guarantee.
//!
//! # Arithmetic
//!
//! Elements are held in radix 2^29 (nine limbs), one element per 64-bit lane,
//! in Montgomery form with `R' = 2^261`. Both Pasta moduli are `1 mod 2^32`,
//! so the Montgomery factor of each reduction round is a negation, and in
//! radix 2^29 they have limb 0 equal to 1, limbs 5 to 7 zero, and limb 8 equal
//! to 2^22 (checked by [`pasta_limbs29`]).
//!
//! Bounds:
//! - Inputs below `2p` give a product below `4p^2`, and the Montgomery result
//!   `(t + Q p) / R'` is below `4p^2 / R' + p < 1.04 p`, so values stay below
//!   `2p` with no conditional subtractions; callers canonicalize once at the
//!   end.
//! - Output limbs 0 to 7 are below 2^29 and limb 8 is below 2^23, so every
//!   input limb fits the 32-bit operands of `vpmuludq` (doubled limbs in `sqr`
//!   are below 2^30).
//! - A column accumulates at most nine products below 2^60 (doubled cross
//!   products), five reduction terms below 2^58, and a carry below 2^35, so it
//!   stays below 2^64.
//!
//! The value of each lane is identical to the scalar Montgomery arithmetic's
//! after conversion; `batch_from_bytes_vartime` is tested against `from_bytes`.
#![allow(unsafe_code)]
// Target-feature operations and calls require explicit unsafe blocks on the MSRV.
#![allow(unused_unsafe)]

use core::arch::x86_64::*;
use core::sync::atomic::{AtomicU8, Ordering};

static AVX2: AtomicU8 = AtomicU8::new(0);

/// Whether this CPU and operating system permit AVX2 instructions.
#[inline]
pub(crate) fn available() -> bool {
    match AVX2.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => detect(),
    }
}

#[cold]
fn detect() -> bool {
    // SAFETY: CPUID is baseline on x86-64. XGETBV is called only when XSAVE,
    // OSXSAVE and AVX are supported; XCR0 must enable both XMM and YMM state.
    #[allow(unused_unsafe)]
    let supported = unsafe {
        let max = __cpuid(0).eax;
        max >= 7 && {
            let state = __cpuid(1).ecx;
            let required = (1 << 26) | (1 << 27) | (1 << 28);
            state & required == required
                && _xgetbv(0) & 6 == 6
                && __cpuid_count(7, 0).ebx & (1 << 5) != 0
        }
    };
    AVX2.store(if supported { 2 } else { 1 }, Ordering::Relaxed);
    supported
}

use super::fields::SqrtTableHelpers;

const L: usize = 9;
const BITS: i32 = 29;
const MASK: u64 = (1 << BITS) - 1;

/// Four field elements, limb-major.
#[derive(Clone, Copy)]
struct V([__m256i; L]);

/// Little-endian 256-bit integers, four per vector, `G` vectors.
pub(crate) type Lanes<const G: usize> = [[[u64; 4]; 4]; G];

/// Converts a little-endian 256-bit integer to radix 2^29.
const fn to29(x: &[u64; 4]) -> [u64; L] {
    let mut out = [0u64; L];
    let mut i = 0;
    while i < L {
        let bit = i * BITS as usize;
        let (w, s) = (bit / 64, bit % 64);
        let mut v = x[w] >> s;
        if s + BITS as usize > 64 && w + 1 < 4 {
            v |= x[w + 1] << (64 - s);
        }
        out[i] = v & MASK;
        i += 1;
    }
    out
}

/// Converts a Pasta modulus to radix 2^29, checking the sparse shape that `mul` and `sqr`
/// assume.
const fn pasta_limbs29(p: &[u64; 4]) -> [u64; L] {
    let l = to29(p);
    assert!(l[0] == 1 && l[5] == 0 && l[6] == 0 && l[7] == 0 && l[8] == 1 << 22);
    l
}

/// Lane-wise Montgomery product.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn mul<const P0: u64, const P1: u64, const P2: u64, const P3: u64>(a: &V, b: &V) -> V {
    // SAFETY: callers establish AVX2 availability; lane bounds are documented above.
    unsafe {
        let p = const { pasta_limbs29(&[P0, P1, P2, P3]) };
        {
            let mask = _mm256_set1_epi64x(MASK as i64);
            let zero = _mm256_setzero_si256();
            let p1 = _mm256_set1_epi64x(p[1] as i64);
            let p2 = _mm256_set1_epi64x(p[2] as i64);
            let p3 = _mm256_set1_epi64x(p[3] as i64);
            let p4 = _mm256_set1_epi64x(p[4] as i64);
            let a = &a.0;
            let b = &b.0;
            let mut q = [zero; L];
            let mut out = [zero; L];
            let mut carry = zero;
            let col = _mm256_add_epi64(_mm256_mul_epu32(a[0], b[0]), carry);
            q[0] = _mm256_and_si256(_mm256_sub_epi64(zero, col), mask);
            carry = _mm256_srli_epi64::<BITS>(_mm256_add_epi64(col, q[0]));
            let col = _mm256_add_epi64(
                _mm256_add_epi64(_mm256_mul_epu32(a[0], b[1]), _mm256_mul_epu32(a[1], b[0])),
                _mm256_add_epi64(_mm256_mul_epu32(q[0], p1), carry),
            );
            q[1] = _mm256_and_si256(_mm256_sub_epi64(zero, col), mask);
            carry = _mm256_srli_epi64::<BITS>(_mm256_add_epi64(col, q[1]));
            let col = _mm256_add_epi64(
                _mm256_add_epi64(
                    _mm256_add_epi64(_mm256_mul_epu32(a[0], b[2]), _mm256_mul_epu32(a[1], b[1])),
                    _mm256_add_epi64(_mm256_mul_epu32(a[2], b[0]), _mm256_mul_epu32(q[0], p2)),
                ),
                _mm256_add_epi64(_mm256_mul_epu32(q[1], p1), carry),
            );
            q[2] = _mm256_and_si256(_mm256_sub_epi64(zero, col), mask);
            carry = _mm256_srli_epi64::<BITS>(_mm256_add_epi64(col, q[2]));
            let col = _mm256_add_epi64(
                _mm256_add_epi64(
                    _mm256_add_epi64(_mm256_mul_epu32(a[0], b[3]), _mm256_mul_epu32(a[1], b[2])),
                    _mm256_add_epi64(_mm256_mul_epu32(a[2], b[1]), _mm256_mul_epu32(a[3], b[0])),
                ),
                _mm256_add_epi64(
                    _mm256_add_epi64(_mm256_mul_epu32(q[0], p3), _mm256_mul_epu32(q[1], p2)),
                    _mm256_add_epi64(_mm256_mul_epu32(q[2], p1), carry),
                ),
            );
            q[3] = _mm256_and_si256(_mm256_sub_epi64(zero, col), mask);
            carry = _mm256_srli_epi64::<BITS>(_mm256_add_epi64(col, q[3]));
            let col = _mm256_add_epi64(
                _mm256_add_epi64(
                    _mm256_add_epi64(
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[0], b[4]),
                            _mm256_mul_epu32(a[1], b[3]),
                        ),
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[2], b[2]),
                            _mm256_mul_epu32(a[3], b[1]),
                        ),
                    ),
                    _mm256_add_epi64(
                        _mm256_add_epi64(_mm256_mul_epu32(a[4], b[0]), _mm256_mul_epu32(q[0], p4)),
                        _mm256_add_epi64(_mm256_mul_epu32(q[1], p3), _mm256_mul_epu32(q[2], p2)),
                    ),
                ),
                _mm256_add_epi64(_mm256_mul_epu32(q[3], p1), carry),
            );
            q[4] = _mm256_and_si256(_mm256_sub_epi64(zero, col), mask);
            carry = _mm256_srli_epi64::<BITS>(_mm256_add_epi64(col, q[4]));
            let col = _mm256_add_epi64(
                _mm256_add_epi64(
                    _mm256_add_epi64(
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[0], b[5]),
                            _mm256_mul_epu32(a[1], b[4]),
                        ),
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[2], b[3]),
                            _mm256_mul_epu32(a[3], b[2]),
                        ),
                    ),
                    _mm256_add_epi64(
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[4], b[1]),
                            _mm256_mul_epu32(a[5], b[0]),
                        ),
                        _mm256_add_epi64(_mm256_mul_epu32(q[1], p4), _mm256_mul_epu32(q[2], p3)),
                    ),
                ),
                _mm256_add_epi64(
                    _mm256_add_epi64(_mm256_mul_epu32(q[3], p2), _mm256_mul_epu32(q[4], p1)),
                    carry,
                ),
            );
            q[5] = _mm256_and_si256(_mm256_sub_epi64(zero, col), mask);
            carry = _mm256_srli_epi64::<BITS>(_mm256_add_epi64(col, q[5]));
            let col = _mm256_add_epi64(
                _mm256_add_epi64(
                    _mm256_add_epi64(
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[0], b[6]),
                            _mm256_mul_epu32(a[1], b[5]),
                        ),
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[2], b[4]),
                            _mm256_mul_epu32(a[3], b[3]),
                        ),
                    ),
                    _mm256_add_epi64(
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[4], b[2]),
                            _mm256_mul_epu32(a[5], b[1]),
                        ),
                        _mm256_add_epi64(_mm256_mul_epu32(a[6], b[0]), _mm256_mul_epu32(q[2], p4)),
                    ),
                ),
                _mm256_add_epi64(
                    _mm256_add_epi64(_mm256_mul_epu32(q[3], p3), _mm256_mul_epu32(q[4], p2)),
                    _mm256_add_epi64(_mm256_mul_epu32(q[5], p1), carry),
                ),
            );
            q[6] = _mm256_and_si256(_mm256_sub_epi64(zero, col), mask);
            carry = _mm256_srli_epi64::<BITS>(_mm256_add_epi64(col, q[6]));
            let col = _mm256_add_epi64(
                _mm256_add_epi64(
                    _mm256_add_epi64(
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[0], b[7]),
                            _mm256_mul_epu32(a[1], b[6]),
                        ),
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[2], b[5]),
                            _mm256_mul_epu32(a[3], b[4]),
                        ),
                    ),
                    _mm256_add_epi64(
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[4], b[3]),
                            _mm256_mul_epu32(a[5], b[2]),
                        ),
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[6], b[1]),
                            _mm256_mul_epu32(a[7], b[0]),
                        ),
                    ),
                ),
                _mm256_add_epi64(
                    _mm256_add_epi64(
                        _mm256_add_epi64(_mm256_mul_epu32(q[3], p4), _mm256_mul_epu32(q[4], p3)),
                        _mm256_add_epi64(_mm256_mul_epu32(q[5], p2), _mm256_mul_epu32(q[6], p1)),
                    ),
                    carry,
                ),
            );
            q[7] = _mm256_and_si256(_mm256_sub_epi64(zero, col), mask);
            carry = _mm256_srli_epi64::<BITS>(_mm256_add_epi64(col, q[7]));
            let col = _mm256_add_epi64(
                _mm256_add_epi64(
                    _mm256_add_epi64(
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[0], b[8]),
                            _mm256_mul_epu32(a[1], b[7]),
                        ),
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[2], b[6]),
                            _mm256_mul_epu32(a[3], b[5]),
                        ),
                    ),
                    _mm256_add_epi64(
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[4], b[4]),
                            _mm256_mul_epu32(a[5], b[3]),
                        ),
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[6], b[2]),
                            _mm256_mul_epu32(a[7], b[1]),
                        ),
                    ),
                ),
                _mm256_add_epi64(
                    _mm256_add_epi64(
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[8], b[0]),
                            _mm256_slli_epi64::<22>(q[0]),
                        ),
                        _mm256_add_epi64(_mm256_mul_epu32(q[4], p4), _mm256_mul_epu32(q[5], p3)),
                    ),
                    _mm256_add_epi64(
                        _mm256_add_epi64(_mm256_mul_epu32(q[6], p2), _mm256_mul_epu32(q[7], p1)),
                        carry,
                    ),
                ),
            );
            q[8] = _mm256_and_si256(_mm256_sub_epi64(zero, col), mask);
            carry = _mm256_srli_epi64::<BITS>(_mm256_add_epi64(col, q[8]));
            let col = _mm256_add_epi64(
                _mm256_add_epi64(
                    _mm256_add_epi64(
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[1], b[8]),
                            _mm256_mul_epu32(a[2], b[7]),
                        ),
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[3], b[6]),
                            _mm256_mul_epu32(a[4], b[5]),
                        ),
                    ),
                    _mm256_add_epi64(
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[5], b[4]),
                            _mm256_mul_epu32(a[6], b[3]),
                        ),
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[7], b[2]),
                            _mm256_mul_epu32(a[8], b[1]),
                        ),
                    ),
                ),
                _mm256_add_epi64(
                    _mm256_add_epi64(
                        _mm256_add_epi64(_mm256_slli_epi64::<22>(q[1]), _mm256_mul_epu32(q[5], p4)),
                        _mm256_add_epi64(_mm256_mul_epu32(q[6], p3), _mm256_mul_epu32(q[7], p2)),
                    ),
                    _mm256_add_epi64(_mm256_mul_epu32(q[8], p1), carry),
                ),
            );
            out[0] = _mm256_and_si256(col, mask);
            carry = _mm256_srli_epi64::<BITS>(col);
            let col = _mm256_add_epi64(
                _mm256_add_epi64(
                    _mm256_add_epi64(
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[2], b[8]),
                            _mm256_mul_epu32(a[3], b[7]),
                        ),
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[4], b[6]),
                            _mm256_mul_epu32(a[5], b[5]),
                        ),
                    ),
                    _mm256_add_epi64(
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[6], b[4]),
                            _mm256_mul_epu32(a[7], b[3]),
                        ),
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[8], b[2]),
                            _mm256_slli_epi64::<22>(q[2]),
                        ),
                    ),
                ),
                _mm256_add_epi64(
                    _mm256_add_epi64(_mm256_mul_epu32(q[6], p4), _mm256_mul_epu32(q[7], p3)),
                    _mm256_add_epi64(_mm256_mul_epu32(q[8], p2), carry),
                ),
            );
            out[1] = _mm256_and_si256(col, mask);
            carry = _mm256_srli_epi64::<BITS>(col);
            let col = _mm256_add_epi64(
                _mm256_add_epi64(
                    _mm256_add_epi64(
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[3], b[8]),
                            _mm256_mul_epu32(a[4], b[7]),
                        ),
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[5], b[6]),
                            _mm256_mul_epu32(a[6], b[5]),
                        ),
                    ),
                    _mm256_add_epi64(
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[7], b[4]),
                            _mm256_mul_epu32(a[8], b[3]),
                        ),
                        _mm256_add_epi64(_mm256_slli_epi64::<22>(q[3]), _mm256_mul_epu32(q[7], p4)),
                    ),
                ),
                _mm256_add_epi64(_mm256_mul_epu32(q[8], p3), carry),
            );
            out[2] = _mm256_and_si256(col, mask);
            carry = _mm256_srli_epi64::<BITS>(col);
            let col = _mm256_add_epi64(
                _mm256_add_epi64(
                    _mm256_add_epi64(_mm256_mul_epu32(a[4], b[8]), _mm256_mul_epu32(a[5], b[7])),
                    _mm256_add_epi64(_mm256_mul_epu32(a[6], b[6]), _mm256_mul_epu32(a[7], b[5])),
                ),
                _mm256_add_epi64(
                    _mm256_add_epi64(_mm256_mul_epu32(a[8], b[4]), _mm256_slli_epi64::<22>(q[4])),
                    _mm256_add_epi64(_mm256_mul_epu32(q[8], p4), carry),
                ),
            );
            out[3] = _mm256_and_si256(col, mask);
            carry = _mm256_srli_epi64::<BITS>(col);
            let col = _mm256_add_epi64(
                _mm256_add_epi64(
                    _mm256_add_epi64(_mm256_mul_epu32(a[5], b[8]), _mm256_mul_epu32(a[6], b[7])),
                    _mm256_add_epi64(_mm256_mul_epu32(a[7], b[6]), _mm256_mul_epu32(a[8], b[5])),
                ),
                _mm256_add_epi64(_mm256_slli_epi64::<22>(q[5]), carry),
            );
            out[4] = _mm256_and_si256(col, mask);
            carry = _mm256_srli_epi64::<BITS>(col);
            let col = _mm256_add_epi64(
                _mm256_add_epi64(
                    _mm256_add_epi64(_mm256_mul_epu32(a[6], b[8]), _mm256_mul_epu32(a[7], b[7])),
                    _mm256_add_epi64(_mm256_mul_epu32(a[8], b[6]), _mm256_slli_epi64::<22>(q[6])),
                ),
                carry,
            );
            out[5] = _mm256_and_si256(col, mask);
            carry = _mm256_srli_epi64::<BITS>(col);
            let col = _mm256_add_epi64(
                _mm256_add_epi64(_mm256_mul_epu32(a[7], b[8]), _mm256_mul_epu32(a[8], b[7])),
                _mm256_add_epi64(_mm256_slli_epi64::<22>(q[7]), carry),
            );
            out[6] = _mm256_and_si256(col, mask);
            carry = _mm256_srli_epi64::<BITS>(col);
            let col = _mm256_add_epi64(
                _mm256_add_epi64(_mm256_mul_epu32(a[8], b[8]), _mm256_slli_epi64::<22>(q[8])),
                carry,
            );
            out[7] = _mm256_and_si256(col, mask);
            carry = _mm256_srli_epi64::<BITS>(col);
            out[8] = carry;
            V(out)
        }
    }
}

/// Lane-wise Montgomery square.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn sqr<const P0: u64, const P1: u64, const P2: u64, const P3: u64>(a: &V) -> V {
    // SAFETY: callers establish AVX2 availability; lane bounds are documented above.
    unsafe {
        let p = const { pasta_limbs29(&[P0, P1, P2, P3]) };
        {
            let mask = _mm256_set1_epi64x(MASK as i64);
            let zero = _mm256_setzero_si256();
            let p1 = _mm256_set1_epi64x(p[1] as i64);
            let p2 = _mm256_set1_epi64x(p[2] as i64);
            let p3 = _mm256_set1_epi64x(p[3] as i64);
            let p4 = _mm256_set1_epi64x(p[4] as i64);
            let a = &a.0;
            let d: [__m256i; L] = core::array::from_fn(|i| _mm256_add_epi64(a[i], a[i]));
            let mut q = [zero; L];
            let mut out = [zero; L];
            let mut carry = zero;
            let col = _mm256_add_epi64(_mm256_mul_epu32(a[0], a[0]), carry);
            q[0] = _mm256_and_si256(_mm256_sub_epi64(zero, col), mask);
            carry = _mm256_srli_epi64::<BITS>(_mm256_add_epi64(col, q[0]));
            let col = _mm256_add_epi64(
                _mm256_add_epi64(_mm256_mul_epu32(a[0], d[1]), _mm256_mul_epu32(q[0], p1)),
                carry,
            );
            q[1] = _mm256_and_si256(_mm256_sub_epi64(zero, col), mask);
            carry = _mm256_srli_epi64::<BITS>(_mm256_add_epi64(col, q[1]));
            let col = _mm256_add_epi64(
                _mm256_add_epi64(
                    _mm256_add_epi64(_mm256_mul_epu32(a[0], d[2]), _mm256_mul_epu32(a[1], a[1])),
                    _mm256_add_epi64(_mm256_mul_epu32(q[0], p2), _mm256_mul_epu32(q[1], p1)),
                ),
                carry,
            );
            q[2] = _mm256_and_si256(_mm256_sub_epi64(zero, col), mask);
            carry = _mm256_srli_epi64::<BITS>(_mm256_add_epi64(col, q[2]));
            let col = _mm256_add_epi64(
                _mm256_add_epi64(
                    _mm256_add_epi64(_mm256_mul_epu32(a[0], d[3]), _mm256_mul_epu32(a[1], d[2])),
                    _mm256_add_epi64(_mm256_mul_epu32(q[0], p3), _mm256_mul_epu32(q[1], p2)),
                ),
                _mm256_add_epi64(_mm256_mul_epu32(q[2], p1), carry),
            );
            q[3] = _mm256_and_si256(_mm256_sub_epi64(zero, col), mask);
            carry = _mm256_srli_epi64::<BITS>(_mm256_add_epi64(col, q[3]));
            let col = _mm256_add_epi64(
                _mm256_add_epi64(
                    _mm256_add_epi64(_mm256_mul_epu32(a[0], d[4]), _mm256_mul_epu32(a[1], d[3])),
                    _mm256_add_epi64(_mm256_mul_epu32(a[2], a[2]), _mm256_mul_epu32(q[0], p4)),
                ),
                _mm256_add_epi64(
                    _mm256_add_epi64(_mm256_mul_epu32(q[1], p3), _mm256_mul_epu32(q[2], p2)),
                    _mm256_add_epi64(_mm256_mul_epu32(q[3], p1), carry),
                ),
            );
            q[4] = _mm256_and_si256(_mm256_sub_epi64(zero, col), mask);
            carry = _mm256_srli_epi64::<BITS>(_mm256_add_epi64(col, q[4]));
            let col = _mm256_add_epi64(
                _mm256_add_epi64(
                    _mm256_add_epi64(_mm256_mul_epu32(a[0], d[5]), _mm256_mul_epu32(a[1], d[4])),
                    _mm256_add_epi64(_mm256_mul_epu32(a[2], d[3]), _mm256_mul_epu32(q[1], p4)),
                ),
                _mm256_add_epi64(
                    _mm256_add_epi64(_mm256_mul_epu32(q[2], p3), _mm256_mul_epu32(q[3], p2)),
                    _mm256_add_epi64(_mm256_mul_epu32(q[4], p1), carry),
                ),
            );
            q[5] = _mm256_and_si256(_mm256_sub_epi64(zero, col), mask);
            carry = _mm256_srli_epi64::<BITS>(_mm256_add_epi64(col, q[5]));
            let col = _mm256_add_epi64(
                _mm256_add_epi64(
                    _mm256_add_epi64(
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[0], d[6]),
                            _mm256_mul_epu32(a[1], d[5]),
                        ),
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[2], d[4]),
                            _mm256_mul_epu32(a[3], a[3]),
                        ),
                    ),
                    _mm256_add_epi64(
                        _mm256_add_epi64(_mm256_mul_epu32(q[2], p4), _mm256_mul_epu32(q[3], p3)),
                        _mm256_add_epi64(_mm256_mul_epu32(q[4], p2), _mm256_mul_epu32(q[5], p1)),
                    ),
                ),
                carry,
            );
            q[6] = _mm256_and_si256(_mm256_sub_epi64(zero, col), mask);
            carry = _mm256_srli_epi64::<BITS>(_mm256_add_epi64(col, q[6]));
            let col = _mm256_add_epi64(
                _mm256_add_epi64(
                    _mm256_add_epi64(
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[0], d[7]),
                            _mm256_mul_epu32(a[1], d[6]),
                        ),
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[2], d[5]),
                            _mm256_mul_epu32(a[3], d[4]),
                        ),
                    ),
                    _mm256_add_epi64(
                        _mm256_add_epi64(_mm256_mul_epu32(q[3], p4), _mm256_mul_epu32(q[4], p3)),
                        _mm256_add_epi64(_mm256_mul_epu32(q[5], p2), _mm256_mul_epu32(q[6], p1)),
                    ),
                ),
                carry,
            );
            q[7] = _mm256_and_si256(_mm256_sub_epi64(zero, col), mask);
            carry = _mm256_srli_epi64::<BITS>(_mm256_add_epi64(col, q[7]));
            let col = _mm256_add_epi64(
                _mm256_add_epi64(
                    _mm256_add_epi64(
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[0], d[8]),
                            _mm256_mul_epu32(a[1], d[7]),
                        ),
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[2], d[6]),
                            _mm256_mul_epu32(a[3], d[5]),
                        ),
                    ),
                    _mm256_add_epi64(
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[4], a[4]),
                            _mm256_slli_epi64::<22>(q[0]),
                        ),
                        _mm256_add_epi64(_mm256_mul_epu32(q[4], p4), _mm256_mul_epu32(q[5], p3)),
                    ),
                ),
                _mm256_add_epi64(
                    _mm256_add_epi64(_mm256_mul_epu32(q[6], p2), _mm256_mul_epu32(q[7], p1)),
                    carry,
                ),
            );
            q[8] = _mm256_and_si256(_mm256_sub_epi64(zero, col), mask);
            carry = _mm256_srli_epi64::<BITS>(_mm256_add_epi64(col, q[8]));
            let col = _mm256_add_epi64(
                _mm256_add_epi64(
                    _mm256_add_epi64(
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[1], d[8]),
                            _mm256_mul_epu32(a[2], d[7]),
                        ),
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[3], d[6]),
                            _mm256_mul_epu32(a[4], d[5]),
                        ),
                    ),
                    _mm256_add_epi64(
                        _mm256_add_epi64(_mm256_slli_epi64::<22>(q[1]), _mm256_mul_epu32(q[5], p4)),
                        _mm256_add_epi64(_mm256_mul_epu32(q[6], p3), _mm256_mul_epu32(q[7], p2)),
                    ),
                ),
                _mm256_add_epi64(_mm256_mul_epu32(q[8], p1), carry),
            );
            out[0] = _mm256_and_si256(col, mask);
            carry = _mm256_srli_epi64::<BITS>(col);
            let col = _mm256_add_epi64(
                _mm256_add_epi64(
                    _mm256_add_epi64(
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[2], d[8]),
                            _mm256_mul_epu32(a[3], d[7]),
                        ),
                        _mm256_add_epi64(
                            _mm256_mul_epu32(a[4], d[6]),
                            _mm256_mul_epu32(a[5], a[5]),
                        ),
                    ),
                    _mm256_add_epi64(
                        _mm256_add_epi64(_mm256_slli_epi64::<22>(q[2]), _mm256_mul_epu32(q[6], p4)),
                        _mm256_add_epi64(_mm256_mul_epu32(q[7], p3), _mm256_mul_epu32(q[8], p2)),
                    ),
                ),
                carry,
            );
            out[1] = _mm256_and_si256(col, mask);
            carry = _mm256_srli_epi64::<BITS>(col);
            let col = _mm256_add_epi64(
                _mm256_add_epi64(
                    _mm256_add_epi64(_mm256_mul_epu32(a[3], d[8]), _mm256_mul_epu32(a[4], d[7])),
                    _mm256_add_epi64(_mm256_mul_epu32(a[5], d[6]), _mm256_slli_epi64::<22>(q[3])),
                ),
                _mm256_add_epi64(
                    _mm256_add_epi64(_mm256_mul_epu32(q[7], p4), _mm256_mul_epu32(q[8], p3)),
                    carry,
                ),
            );
            out[2] = _mm256_and_si256(col, mask);
            carry = _mm256_srli_epi64::<BITS>(col);
            let col = _mm256_add_epi64(
                _mm256_add_epi64(
                    _mm256_add_epi64(_mm256_mul_epu32(a[4], d[8]), _mm256_mul_epu32(a[5], d[7])),
                    _mm256_add_epi64(_mm256_mul_epu32(a[6], a[6]), _mm256_slli_epi64::<22>(q[4])),
                ),
                _mm256_add_epi64(_mm256_mul_epu32(q[8], p4), carry),
            );
            out[3] = _mm256_and_si256(col, mask);
            carry = _mm256_srli_epi64::<BITS>(col);
            let col = _mm256_add_epi64(
                _mm256_add_epi64(_mm256_mul_epu32(a[5], d[8]), _mm256_mul_epu32(a[6], d[7])),
                _mm256_add_epi64(_mm256_slli_epi64::<22>(q[5]), carry),
            );
            out[4] = _mm256_and_si256(col, mask);
            carry = _mm256_srli_epi64::<BITS>(col);
            let col = _mm256_add_epi64(
                _mm256_add_epi64(_mm256_mul_epu32(a[6], d[8]), _mm256_mul_epu32(a[7], a[7])),
                _mm256_add_epi64(_mm256_slli_epi64::<22>(q[6]), carry),
            );
            out[5] = _mm256_and_si256(col, mask);
            carry = _mm256_srli_epi64::<BITS>(col);
            let col = _mm256_add_epi64(
                _mm256_add_epi64(_mm256_mul_epu32(a[7], d[8]), _mm256_slli_epi64::<22>(q[7])),
                carry,
            );
            out[6] = _mm256_and_si256(col, mask);
            carry = _mm256_srli_epi64::<BITS>(col);
            let col = _mm256_add_epi64(
                _mm256_add_epi64(_mm256_mul_epu32(a[8], a[8]), _mm256_slli_epi64::<22>(q[8])),
                carry,
            );
            out[7] = _mm256_and_si256(col, mask);
            carry = _mm256_srli_epi64::<BITS>(col);
            out[8] = carry;
            V(out)
        }
    }
}

/// Packs four little-endian 256-bit integers (each below 2^261).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn pack(x: &[[u64; 4]; 4]) -> V {
    // SAFETY: callers establish AVX2 availability; lane bounds are documented above.
    unsafe {
        let l = [to29(&x[0]), to29(&x[1]), to29(&x[2]), to29(&x[3])];
        let mut v = [_mm256_setzero_si256(); L];
        for (i, limb) in v.iter_mut().enumerate() {
            *limb = _mm256_set_epi64x(
                l[3][i] as i64,
                l[2][i] as i64,
                l[1][i] as i64,
                l[0][i] as i64,
            );
        }
        V(v)
    }
}

/// Unpacks four integers; limbs 0 to 7 must be below 2^29 (as `mul` and `sqr`
/// leave them).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn unpack(v: &V) -> [[u64; 4]; 4] {
    // SAFETY: callers establish AVX2 availability; lane bounds are documented above.
    unsafe {
        let mut limbs = [[0u64; 4]; L];
        for (dst, src) in limbs.iter_mut().zip(&v.0) {
            // SAFETY: stores exactly the 32 bytes of the `[u64; 4]` behind `dst`.
            unsafe { _mm256_storeu_si256(dst.as_mut_ptr() as *mut __m256i, *src) };
        }
        let mut out = [[0u64; 4]; 4];
        for (lane, out) in out.iter_mut().enumerate() {
            for (i, limb) in limbs.iter().enumerate() {
                let bit = i * BITS as usize;
                let (w, s) = (bit / 64, bit % 64);
                let x = limb[lane];
                out[w] |= x << s;
                if s + BITS as usize > 64 && w + 1 < 4 {
                    out[w + 1] |= x >> (64 - s);
                }
            }
        }
        out
    }
}

/// Lane-wise `a^(2^n)` over `G` vectors.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn sqr_n<const P0: u64, const P1: u64, const P2: u64, const P3: u64, const G: usize>(
    a: &[V; G],
    n: u32,
) -> [V; G] {
    // SAFETY: callers establish AVX2 availability; lane bounds are documented above.
    unsafe {
        let mut t = *a;
        for _ in 0..n {
            for v in t.iter_mut() {
                *v = sqr::<P0, P1, P2, P3>(v);
            }
        }
        t
    }
}

/// Lane-wise products over `G` vectors.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn mul_g<const P0: u64, const P1: u64, const P2: u64, const P3: u64, const G: usize>(
    a: &[V; G],
    b: &[V; G],
) -> [V; G] {
    // SAFETY: callers establish AVX2 availability; lane bounds are documented above.
    unsafe {
        let mut t = *a;
        for (t, b) in t.iter_mut().zip(b) {
            *t = mul::<P0, P1, P2, P3>(t, b);
        }
        t
    }
}

/// The square-root exponentiation of `4G` field elements.
///
/// `u` holds canonical Montgomery limbs (`a * 2^256 mod p`). `steps` and
/// `final_squarings` describe the field's addition chain for `(t - 1) / 2`:
/// starting from `u`, each `(n, e)` squares `n` times and multiplies by
/// `u^e`, for `e` in {1, 3, 7, 9, 11, 13}. `c_in` is `2^266 mod p` and
/// `c_out` is `2^256 mod p`, as integers; they move values into and out of the
/// `R'` domain.
///
/// Returns, as Montgomery limbs below `2p`, `[uv, x0, x1, x2, x3]` where
/// `v = u^((t-1)/2)`, `x3 = uv * v`, and `x2`, `x1`, `x0` are `x3` squared 8,
/// 16 and 24 times: the inputs of the table-driven square-root finish.
///
/// # Safety
///
/// The CPU and operating system must support AVX2, including enabled XMM/YMM
/// state. Callers must first check [`available`].
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn sqrt_chain<
    const P0: u64,
    const P1: u64,
    const P2: u64,
    const P3: u64,
    const G: usize,
>(
    u: &Lanes<G>,
    c_in: &[u64; 4],
    c_out: &[u64; 4],
    steps: &[(u32, u32)],
    final_squarings: u32,
) -> [Lanes<G>; 5] {
    // SAFETY: callers establish AVX2 availability; lane bounds are documented above.
    unsafe {
        let c_in = pack(&[*c_in; 4]);
        let c_out = pack(&[*c_out; 4]);
        let mut x = [c_in; G];
        for (x, u) in x.iter_mut().zip(u) {
            *x = mul::<P0, P1, P2, P3>(&pack(u), &c_in);
        }

        let x2 = sqr_n::<P0, P1, P2, P3, G>(&x, 1);
        let x3 = mul_g::<P0, P1, P2, P3, G>(&x2, &x);
        let x7 = mul_g::<P0, P1, P2, P3, G>(&sqr_n::<P0, P1, P2, P3, G>(&x3, 1), &x);
        let x9 = mul_g::<P0, P1, P2, P3, G>(&x7, &x2);
        let x11 = mul_g::<P0, P1, P2, P3, G>(&x9, &x2);
        let x13 = mul_g::<P0, P1, P2, P3, G>(&x11, &x2);
        let mut acc = x;
        for &(n, e) in steps {
            let by = match e {
                1 => &x,
                3 => &x3,
                7 => &x7,
                9 => &x9,
                11 => &x11,
                13 => &x13,
                _ => unreachable!("addition-chain step multiplies by an unsupported power"),
            };
            acc = mul_g::<P0, P1, P2, P3, G>(&sqr_n::<P0, P1, P2, P3, G>(&acc, n), by);
        }
        let v = sqr_n::<P0, P1, P2, P3, G>(&acc, final_squarings);

        let uv = mul_g::<P0, P1, P2, P3, G>(&x, &v);
        let x3 = mul_g::<P0, P1, P2, P3, G>(&uv, &v);
        let x2 = sqr_n::<P0, P1, P2, P3, G>(&x3, 8);
        let x1 = sqr_n::<P0, P1, P2, P3, G>(&x2, 8);
        let x0 = sqr_n::<P0, P1, P2, P3, G>(&x1, 8);

        let mut out = [[[[0u64; 4]; 4]; G]; 5];
        for (out, values) in out.iter_mut().zip([&uv, &x0, &x1, &x2, &x3]) {
            for (out, value) in out.iter_mut().zip(values) {
                *out = unpack(&mul::<P0, P1, P2, P3>(value, &c_out));
            }
        }
        out
    }
}

/// A Pasta field whose square-root exponentiation can run in [`sqrt_chain`].
pub(crate) trait Avx2Chain: SqrtTableHelpers {
    /// Canonical Montgomery limbs (`a * 2^256 mod p`).
    fn limbs(&self) -> [u64; 4];
    /// Builds an element from Montgomery limbs below `2p`.
    fn from_limbs_lazy(limbs: [u64; 4]) -> Self;
    /// [`sqrt_chain`] for this field's modulus and addition chain.
    fn sqrt_chain<const G: usize>(u: &Lanes<G>) -> [Lanes<G>; 5];
}

/// Exercises the AVX2 arithmetic at the limits of its documented bounds:
/// inputs anywhere below `2p`, including the limb pattern that maximizes every
/// column sum, which random inputs essentially never produce.
#[cfg(test)]
mod tests;
