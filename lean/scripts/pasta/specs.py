"""The proof files and the routines each one proves."""

# Existing proof files only. None selects every generated routine of that architecture.
SPEC_MANIFEST = {
    "lean/PastaCurves/AArch64/Spec/Add.lean": ("AArch64", ("addMod",)),
    "lean/PastaCurves/AArch64/Spec/Sub.lean": ("AArch64", ("subMod",)),
    "lean/PastaCurves/AArch64/Spec/Mul.lean": ("AArch64", ("mulMont", "mulMontRound")),
    "lean/PastaCurves/AArch64/Spec/Square.lean": ("AArch64", ("sqrMont",)),
    "lean/PastaCurves/AArch64/Spec/CondSub.lean": ("AArch64", ("condSubBlock",)),
    "lean/PastaCurves/AArch64/Spec/Amontred.lean": ("AArch64", ("amontredBlock",)),
    "lean/PastaCurves/AArch64/Spec/SignMag.lean": ("AArch64", ("signMagBlock",)),
    "lean/PastaCurves/AArch64/Spec/DeRow.lean": ("AArch64", ("deRowBlock",)),
    "lean/PastaCurves/AArch64/Spec/FgRow.lean": ("AArch64", ("fgRowBlock",)),
    "lean/PastaCurves/AArch64/Spec/Divstep.lean": ("AArch64", ("divstepRound", "divstepLast")),
    "lean/PastaCurves/AArch64/Spec/Divstep59.lean": ("AArch64", ("divstep59Block",)),
    "lean/PastaCurves/X86_64/Spec/Add.lean": ("X86_64", ("addMod",)),
    "lean/PastaCurves/X86_64/Spec/Sub.lean": ("X86_64", ("subMod",)),
    "lean/PastaCurves/X86_64/Spec/FromMont.lean": ("X86_64", ("fromMont",)),
    "lean/PastaCurves/X86_64/Spec/Mul.lean": ("X86_64", ("mulMont", "mulMontRound")),
    "lean/PastaCurves/X86_64/Spec/Square.lean": ("X86_64", ("squareLo", "squareHi")),
    "lean/PastaCurves/X86_64/Spec/CondSub.lean": ("X86_64", ("condSubBlock",)),
    "lean/PastaCurves/X86_64/Spec/SignMag.lean": ("X86_64", ("signMagBlock",)),
    "lean/PastaCurves/X86_64/Spec/Divstep59.lean": ("X86_64", ("divstep59Block",)),
    "lean/PastaCurves/X86_64/Spec/Divstep.lean": ("X86_64", ("divstepRound",)),
    "lean/PastaCurves/X86_64/Spec/DeRow.lean": ("X86_64", ("deRowBlock",)),
    "lean/PastaCurves/X86_64/Spec/FgRow.lean": ("X86_64", ("fgRowBlock",)),
    "lean/PastaCurves/X86_64/Spec/Amontred.lean": ("X86_64", ("amontredBlock",)),
}

# Missing proofs are tracked by routine, not by hypothetical files; a block transcribed ahead of
# its proof is listed here under its architecture. None at present.
UNPROVED_ROUTINES: dict[str, tuple[str, ...]] = {}
