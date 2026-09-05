#[cfg(feature = "aarch64-asm")]
use std::env;

fn main() {
    println!("cargo:rerun-if-changed=src/asm/pasta_mul-armv8.S");

    #[cfg(feature = "aarch64-asm")]
    build_aarch64_asm();
}

#[cfg(feature = "aarch64-asm")]
fn build_aarch64_asm() {
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let target_endian = env::var("CARGO_CFG_TARGET_ENDIAN").unwrap();
    let target_family = env::var("CARGO_CFG_TARGET_FAMILY").unwrap();
    let target_pointer_width = env::var("CARGO_CFG_TARGET_POINTER_WIDTH").unwrap();

    if target_arch == "aarch64"
        && target_endian == "little"
        && target_family == "unix"
        && target_pointer_width == "64"
    {
        cc::Build::new()
            .file("src/asm/pasta_mul-armv8.S")
            .compile("pasta_curves_aarch64");
    }
}
