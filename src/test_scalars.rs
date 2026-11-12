//! Scalars for the unit tests, drawn from proptest's integer strategies.

use ff::FromUniformBytes;

/// Reduces `N` little-endian `u64` limbs, zero-extended to 64 bytes, with
/// `from_uniform_bytes`.
///
/// With eight uniform limbs the result is within statistical distance
/// `p / 2^512` of uniform. With fewer it is not uniform, but it still reaches
/// every residue once `64 N` exceeds the modulus's bit length.
pub(crate) fn from_uniform_limbs<F, const N: usize>(limbs: [u64; N]) -> F
where
    F: FromUniformBytes<64>,
{
    const { assert!(N <= 8) };
    let mut bytes = [0u8; 64];
    for (chunk, limb) in bytes.chunks_exact_mut(8).zip(limbs) {
        chunk.copy_from_slice(&limb.to_le_bytes());
    }
    F::from_uniform_bytes(&bytes)
}
