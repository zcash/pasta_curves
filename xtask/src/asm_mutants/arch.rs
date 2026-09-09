#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
mod aarch64_apple;
#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
pub(super) use aarch64_apple::AArch64Apple;
