//! The module's entry points, re-exported at its root.

use core::hint::black_box;

/// Four little-endian 64-bit limbs, least significant first: a field element
/// (in Montgomery form, or canonical after [`from_mont`]) or a modulus.
pub type Limbs = [u64; 4];

/// `1` when `value < modulus` as little-endian 256-bit integers, else `0`: the borrow out of
/// the four-limb subtraction `value - modulus`, computed limb by limb. The debug assertions
/// must leave the routines' timing as it is, so the check has no data-dependent branch, and
/// each borrow passes through [`black_box`], the barrier that `subtle` uses, which keeps the
/// optimizer from turning the chain or its callers' combinations back into branches: an
/// opaque word can only be combined arithmetically.
#[inline(always)]
fn is_canonical_word(value: &Limbs, modulus: &Limbs) -> u64 {
    let mut borrow = 0;
    for (v, m) in value.iter().zip(modulus) {
        let (difference, underflow) = v.overflowing_sub(*m);
        let (_, borrow_underflow) = difference.overflowing_sub(borrow);
        borrow = black_box(u64::from(underflow | borrow_underflow));
    }
    borrow
}

/// Whether `value < modulus` as little-endian 256-bit integers; see [`is_canonical_word`].
#[inline(always)]
pub(crate) fn is_canonical(value: &Limbs, modulus: &Limbs) -> bool {
    is_canonical_word(value, modulus) == 1
}

/// The condition that `mul` asserts: a canonical `lhs`, or a canonical `rhs` whose limbs 1 to 3
/// are at most `2^64 - 3`, combined as opaque words (see [`is_canonical_word`]) so that the
/// check runs the same instructions whatever the operands.
#[inline(always)]
pub(crate) fn mul_contract(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs) -> bool {
    let rhs_limbs_ok = black_box(u64::from(rhs[1] <= u64::MAX - 2))
        & black_box(u64::from(rhs[2] <= u64::MAX - 2))
        & black_box(u64::from(rhs[3] <= u64::MAX - 2));
    (is_canonical_word(lhs, modulus) | (is_canonical_word(rhs, modulus) & rhs_limbs_ok)) == 1
}

/// Adds two residues for a Pasta modulus and conditionally subtracts the modulus.
///
/// Outputs are canonical.
///
/// # Safety
///
/// Both inputs must be canonical. This is debug-asserted, and under that precondition the
/// machine-checked proofs in `lean/` establish the result (`add_entry_spec`).
///
/// `modulus` must be either the Pallas or Vesta field modulus. Any other values will
/// cause undefined results.
#[inline(always)]
pub fn add(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs) -> Limbs {
    debug_assert!(
        is_canonical(lhs, modulus),
        "pasta_curves::asm::add requires a canonical lhs"
    );
    debug_assert!(
        is_canonical(rhs, modulus),
        "pasta_curves::asm::add requires a canonical rhs"
    );

    #[cfg(target_arch = "aarch64")]
    {
        super::aarch64::add(lhs, rhs, modulus)
    }

    #[cfg(target_arch = "x86_64")]
    {
        super::x86_64::add(lhs, rhs, modulus)
    }
}

/// Subtracts two residues for a Pasta modulus, adding the modulus back on underflow.
///
/// Outputs are canonical.
///
/// # Safety
///
/// Both inputs must be canonical. This is debug-asserted, and under that precondition the
/// machine-checked proofs in `lean/` establish the result (`sub_entry_spec`).
///
/// `modulus` must be either the Pallas or Vesta field modulus. Any other values will
/// cause undefined results.
#[inline(always)]
pub fn sub(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs) -> Limbs {
    debug_assert!(
        is_canonical(lhs, modulus),
        "pasta_curves::asm::sub requires a canonical lhs"
    );
    debug_assert!(
        is_canonical(rhs, modulus),
        "pasta_curves::asm::sub requires a canonical rhs"
    );

    #[cfg(target_arch = "aarch64")]
    {
        super::aarch64::sub(lhs, rhs, modulus)
    }

    #[cfg(target_arch = "x86_64")]
    {
        super::x86_64::sub(lhs, rhs, modulus)
    }
}

/// Multiplies two Montgomery residues for a Pasta modulus.
///
/// # Safety
///
/// Either `lhs` is canonical (below the modulus) and `rhs` is any four-limb value, or
/// `rhs` is canonical with each of its limbs 1 to 3 at most `2^64 - 3` and `lhs` is any
/// four-limb value. This is debug-asserted, and under that precondition the machine-checked
/// proofs in `lean/` establish the result (`mul_entry_spec`, from `mulMont_spec_of_lhs_lt`
/// and `mulMont_spec_of_rhs_lt`).
///
/// `modulus` must be either the Pallas or Vesta field modulus, and `inv` must be
/// correctly derived from it. Any other values will cause undefined results.
#[inline(always)]
pub fn mul(lhs: &Limbs, rhs: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
    debug_assert!(
        mul_contract(lhs, rhs, modulus),
        "pasta_curves::asm::mul requires a canonical lhs, or a canonical rhs with limbs 1 to 3 \
         at most 2^64 - 3"
    );

    #[cfg(target_arch = "aarch64")]
    {
        super::aarch64::mul(lhs, rhs, modulus, inv)
    }

    #[cfg(target_arch = "x86_64")]
    {
        super::x86_64::mul(lhs, rhs, modulus, inv)
    }
}

/// Squares a canonical Montgomery residue for a Pasta modulus.
///
/// Outputs are canonical.
///
/// # Safety
///
/// The input of `square` must be canonical. This is debug-asserted, and under that
/// precondition the machine-checked proofs in `lean/` establish the result
/// (`square_entry_spec`).
///
/// `modulus` must be either the Pallas or Vesta field modulus, and `inv` must be
/// correctly derived from it. Any other values will cause undefined results.
#[inline(always)]
pub fn square(value: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
    debug_assert!(
        is_canonical(value, modulus),
        "pasta_curves::asm::square requires a canonical input"
    );

    #[cfg(target_arch = "aarch64")]
    {
        super::aarch64::square(value, modulus, inv)
    }

    #[cfg(target_arch = "x86_64")]
    {
        super::x86_64::square(value, modulus, inv)
    }
}

/// Squares a canonical Montgomery residue `count` times, then multiplies the
/// result by `rhs`.
///
/// A `count` of zero is just the multiplication. The squarings keep the value canonical, so the
/// multiplication is under its contract with a canonical `lhs`, and any four-limb `rhs` is
/// accepted. The accumulator stays in registers across the squarings: on AArch64 each step is
/// an inline block that the compiler inlines, and on x86-64 the squaring blocks are always
/// inlined into one loop, with the multiplication called once at the end.
///
/// # Safety
///
/// `value` must be canonical. This is debug-asserted, and under that precondition the
/// machine-checked proofs in `lean/` establish the result (`sqrNMul_entry_spec`).
///
/// `modulus` must be either the Pallas or Vesta field modulus, and `inv` must be
/// correctly derived from it. Any other values will cause undefined results.
#[inline]
pub fn sqr_n_mul(value: &Limbs, count: usize, rhs: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
    // On aarch64, `square` and `mul` can be inlined and optimised by Rust.
    #[cfg(target_arch = "aarch64")]
    {
        let mut acc = *value;
        for _ in 0..count {
            acc = square(&acc, modulus, inv);
        }
        mul(&acc, rhs, modulus, inv)
    }

    // On x86_64, `square` and `mul` can't be inlined due to register pressure, so the backend
    // has its own loop over the always-inlined squaring blocks.
    #[cfg(target_arch = "x86_64")]
    {
        super::x86_64::sqr_n_mul(value, count, rhs, modulus, inv)
    }
}

/// Converts a Montgomery residue into its canonical integer, `value * 2^-256 mod p`: a
/// Montgomery multiplication by one.
///
/// Any four-limb `value` is accepted, and the machine-checked proofs in `lean/` establish the
/// result (`fromMont_entry_spec`). On AArch64 the conversion is the multiplication block with
/// `1` as its right operand, which is canonical with limbs 1 to 3 zero and so inside the
/// multiplication's contract for any left operand. On x86-64 it is a dedicated block.
///
/// # Safety
///
/// `modulus` must be either the Pallas or Vesta field modulus, and `inv` must be
/// correctly derived from it. Any other values will cause undefined results.
#[inline]
pub fn from_mont(value: &Limbs, modulus: &Limbs, inv: u64) -> Limbs {
    // On aarch64, `mul` can be inlined and optimised by Rust.
    #[cfg(target_arch = "aarch64")]
    {
        mul(value, &[1, 0, 0, 0], modulus, inv)
    }

    // On x86_64, `mul` can't be inlined due to register pressure, so we use a dedicated
    // register-only assembly implementation instead.
    #[cfg(target_arch = "x86_64")]
    {
        super::x86_64::from_mont(value, modulus, inv)
    }
}

/// Inverts a canonical Montgomery residue for a Pasta modulus, in constant time.
///
/// Returns the canonical `z` with `x * z ≡ 2^512 (mod p)`: for `x` the Montgomery form of a nonzero
/// residue `X`, `z` is the Montgomery form of `X^-1`; for `x = 0` it is `0`, so a caller that needs
/// an optional inverse checks for zero separately. The algorithm is the serial variant of
/// Bernstein, Chen, Harrison, Huang, Maxwell, Wang, Wuille, and Yang, "Accelerating and verifying
/// constant-time modular inversion" (EUROCRYPT 2026), as in s2n-bignum's `bignum_montinv_p256`: 590
/// half-delta divsteps in ten rounds of 59, computed on packed words, with the coefficients reduced
/// by one Montgomery word per round. It runs a fixed sequence of register-only blocks, so its
/// timing does not depend on `x`. The design and the correctness argument are in
/// `book/src/design/inversion.md`.
///
/// Outputs are canonical.
///
/// # Safety
///
/// `x` must be canonical. This is debug-asserted. Under that precondition the machine-checked
/// proofs in `lean/` establish the result (`invert_entry_spec`, from `montInv_spec` on words and
/// the six block proofs).
///
/// `modulus` must be either the Pallas or Vesta field modulus, `inv` must be correctly derived
/// from it, and `v0` must be `2^562 mod p`, the starting value of the coefficient `v`, which
/// compensates the ten one-word Montgomery reductions (`2^562 = 2^(512 + 5 * 10)`). Any other
/// values will cause undefined results.
///
/// The inversion is provided on AArch64.
#[cfg(any(target_arch = "aarch64", doc))]
#[cfg_attr(docsrs, doc(cfg(target_arch = "aarch64")))]
#[inline]
pub fn invert(x: &Limbs, modulus: &Limbs, inv: u64, v0: &Limbs) -> Limbs {
    debug_assert!(
        is_canonical(x, modulus),
        "pasta_curves::asm::invert requires a canonical input"
    );
    super::aarch64::invert(x, modulus, inv, v0)
}
