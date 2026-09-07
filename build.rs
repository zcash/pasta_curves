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
    let target_family = env::var("CARGO_CFG_TARGET_FAMILY").ok();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_pointer_width = env::var("CARGO_CFG_TARGET_POINTER_WIDTH").unwrap();
    let target_is_unix = target_family
        .as_deref()
        .is_some_and(|families| families.split(',').any(|family| family == "unix"));

    if target_arch == "aarch64"
        && target_endian == "little"
        && (target_is_unix || target_os == "none")
        && target_pointer_width == "64"
    {
        cc::Build::new()
            .file("src/asm/pasta_mul-armv8.S")
            .compile("pasta_curves_aarch64");
    }
}
