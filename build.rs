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
    let target_family = env::var("CARGO_CFG_TARGET_FAMILY").unwrap();

    if target_arch == "aarch64" && target_family == "unix" {
        cc::Build::new()
            .file("src/asm/pasta_mul-armv8.S")
            .compile("pasta_curves_aarch64");
    }
}
