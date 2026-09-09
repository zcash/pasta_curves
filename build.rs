#[cfg(feature = "aarch64-asm")]
use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=src/asm/pasta_mul-armv8.S");
    println!("cargo:rerun-if-env-changed=PASTA_CURVES_ARMV8_ASM_SOURCE");

    #[cfg(feature = "aarch64-asm")]
    build_aarch64_asm();
}

#[cfg(feature = "aarch64-asm")]
fn build_aarch64_asm() {
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let target_vendor = env::var("CARGO_CFG_TARGET_VENDOR").unwrap();

    if target_arch == "aarch64" && target_vendor == "apple" {
        let asm_path = env::var_os("PASTA_CURVES_ARMV8_ASM_SOURCE")
            .map_or_else(|| PathBuf::from("src/asm/pasta_mul-armv8.S"), PathBuf::from);

        cc::Build::new()
            .file(asm_path)
            .compile("pasta_curves_aarch64");
    }
}
