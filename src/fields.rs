//! This module contains implementations for the two finite fields of the Pallas
//! and Vesta curves.

mod fp;
mod fq;

pub use fp::*;
pub use fq::*;

#[cfg(feature = "gpu")]
fn u64_to_u32(limbs: &[u64]) -> alloc::vec::Vec<u32> {
    limbs
        .iter()
        .flat_map(|limb| alloc::vec![(limb & 0xFFFF_FFFF) as u32, (limb >> 32) as u32])
        .collect()
}
