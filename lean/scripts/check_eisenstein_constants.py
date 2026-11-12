#!/usr/bin/env python3
"""Check that the Eisenstein modules state the crate's own constants.

`PastaCurves/Eisenstein/Classes.lean` states `REPS` and `PastaCurves/Eisenstein/Ladder.lean`
states each curve's `ZETA` and the short-basis constants of `GlvParams`, by hand. This script
reads the same constants out of the Rust sources and fails if any differs, so that a change to
`src/glv_eisenstein.rs`, `src/glv.rs`, or the fields' `ZETA` cannot leave the Lean statements
about a stale value.

Usage, from anywhere: lean/scripts/check_eisenstein_constants.py
Exits 1 on a mismatch or on a constant it cannot find.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEAN = ROOT / "lean" / "PastaCurves" / "Eisenstein"


def fail(msg):
    print("error: " + msg, file=sys.stderr)
    sys.exit(1)


def one(pattern, text, what):
    m = re.search(pattern, text, re.DOTALL)
    if not m:
        fail("cannot find " + what)
    return m


def rust_reps():
    src = (ROOT / "src" / "glv_eisenstein.rs").read_text()
    body = one(r"const REPS: \[\(i8, i8\); REP_COUNT\] = \[(.*?)\];", src, "REPS in Rust").group(1)
    return [(int(a), int(b)) for a, b in re.findall(r"\((-?\d+),\s*(-?\d+)\)", body)]


def lean_reps():
    src = (LEAN / "Classes.lean").read_text()
    body = one(r"def reps : List \(E ℤ\) :=\s*\[(.*?)\]", src, "reps in Classes.lean").group(1)
    return [(int(a), int(b)) for a, b in re.findall(r"⟨(-?\d+), (-?\d+)⟩", body)]


def rust_glv(curve):
    src = (ROOT / "src" / "glv.rs").read_text()
    body = one(
        rf"impl GlvParams for {curve}::Point \{{(.*?)\n\}}", src, f"GlvParams for {curve}"
    ).group(1)
    return {
        name: int(
            one(rf"const {name}: u128 = (0x[0-9a-f]+);", body, f"{name} for {curve}").group(1), 16
        )
        for name in ("V1A", "V1B_NEG", "V2A", "V2B")
    }


def rust_zeta(field):
    src = (ROOT / "src" / "fields" / f"{field.lower()}.rs").read_text()
    body = one(
        rf"impl WithSmallOrderMulGroup<3> for {field} \{{\s*const ZETA: Self = "
        rf"{field}::from_raw\(\[(.*?)\]\)",
        src,
        f"{field}::ZETA",
    ).group(1)
    return [int(x, 16) for x in re.findall(r"0x[0-9a-f]+", body)]


def lean_curve(name):
    src = (LEAN / "Ladder.lean").read_text()
    body = one(rf"def {name} : LadderParams where(.*?)\n\n", src, f"{name} in Ladder.lean").group(1)
    n = one(r"n := \(([a-zA-Z]+)\.modulus\.toNat", body, f"n of {name}").group(1)
    lam = one(r"lam := ofLimbs ((?:0x[0-9a-f]+ ?){4})", body, f"lam of {name}").group(1)
    b1 = one(r"B₁ := \((0x[0-9a-f]+) \+ (0x[0-9a-f]+)\) / 2 \+ 1", body, f"B1 of {name}")
    b2 = one(r"B₂ := \((0x[0-9a-f]+) \+ (0x[0-9a-f]+)\) / 2 \+ 1", body, f"B2 of {name}")
    return (
        n,
        [int(x, 16) for x in lam.split()],
        [int(x, 16) for x in b1.groups()],
        [int(x, 16) for x in b2.groups()],
    )


def main():
    if rust_reps() != lean_reps():
        fail(f"Classes.lean's reps {lean_reps()} differ from Rust's REPS {rust_reps()}")

    # Pallas' scalars live in Fq, its group order is Vesta's base modulus, and the reverse.
    for curve, field, base in (("pallas", "Fq", "vestaBase"), ("vesta", "Fp", "pallasBase")):
        n, lam, b1, b2 = lean_curve(curve)
        glv = rust_glv(curve)
        if n != base:
            fail(f"{curve}'s group order should be {base}, not {n}")
        if lam != rust_zeta(field):
            fail(f"{curve}'s lam differs from {field}::ZETA")
        if b1 != [glv["V1A"], glv["V2A"]]:
            fail(f"{curve}'s B1 is not built from V1A and V2A")
        if b2 != [glv["V1B_NEG"], glv["V2B"]]:
            fail(f"{curve}'s B2 is not built from V1B_NEG and V2B")

    print("Eisenstein constants: match the crate.")


if __name__ == "__main__":
    main()
