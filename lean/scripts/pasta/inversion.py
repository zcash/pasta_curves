"""The constant-time inversion's blocks, which are the same functions on every architecture.

Every backend that implements the inversion transcribes these blocks under these names and
signatures, so that one record of the blocks (`InvertBlocks`) and one proof of the composition
serve every architecture. An architecture's block list adds them after its own.
"""

from asm2lean.block import Block

# The fields of each argument and result kind, in order: the words of a multi-word value, or
# `None` for a single word. The structures are declared in `PastaCurves/Semantics.lean`, shared by
# the backends.
KINDS: dict[str, tuple[str, ...] | None] = {
    "Limbs": ("l0", "l1", "l2", "l3"),
    "Signed5": ("l0", "l1", "l2", "l3", "l4"),
    "Divstep59Result": ("two_delta", "u", "v", "q", "r"),
    "SignMag": ("u", "v", "q", "r", "su", "sv", "sq", "sr"),
    "Nat": None,
}


BLOCKS: tuple[Block, ...] = (
    Block(
        "divstep59",
        "divstep59Block",
        (
            "The inline `asm!` block of `divstep59`: 59 half-delta divsteps on the low words `f0` and "
            "`g0` of `f` and `g` at `two_delta`, as three packed batches of 20, 20, and 19 steps in which "
            "the matrix coefficients ride in the upper bits of the two words, with the two matrix "
            "products between and after the batches. It is s2n-bignum's `divstep59` macro on named "
            "registers. The result is the new `two_delta` and the 59-step matrix, as two's-complement words."
        ),
        (("two_delta", "Nat"), ("f0", "Nat"), ("g0", "Nat")),
        result="Divstep59Result",
    ),
    Block(
        "sign_mag",
        "signMagBlock",
        (
            "The inline `asm!` block of `sign_mag`: the sign-magnitude form of the four entries of a "
            "transition matrix, each entry's magnitude and its sign as a mask (all ones when negative, "
            "else zero), which the row blocks take."
        ),
        (("u", "Nat"), ("v", "Nat"), ("q", "Nat"), ("r", "Nat")),
        result="SignMag",
    ),
    Block(
        "fg_row",
        "fgRowBlock",
        (
            "The inline `asm!` block of `fg_row`: one row of the update of `f` and `g`, "
            "`(m0 f + m1 g) / 2^59` on five-word signed values, from the row's magnitudes `m0`, `m1` "
            "and sign masks `s0`, `s1`, as a 320-bit accumulation shifted right by 59."
        ),
        (
            ("f", "Signed5"),
            ("g", "Signed5"),
            ("m0", "Nat"),
            ("m1", "Nat"),
            ("s0", "Nat"),
            ("s1", "Nat"),
        ),
        result="Signed5",
    ),
    Block(
        "de_row",
        "deRowBlock",
        (
            "The inline `asm!` block of `de_row`: one row of the combination of `d` and `e`, "
            "`m0 d + m1 e` as a five-word signed value, from the row's magnitudes `m0`, `m1` and "
            "sign masks `s0`, `s1`. It is the accumulation of `fgRowBlock` without the shift."
        ),
        (
            ("d", "Limbs"),
            ("e", "Limbs"),
            ("m0", "Nat"),
            ("m1", "Nat"),
            ("s0", "Nat"),
            ("s1", "Nat"),
        ),
        result="Signed5",
    ),
    Block(
        "amontred",
        "amontredBlock",
        (
            "The inline `asm!` block of `amontred`: the almost-Montgomery reduction of a five-word "
            "signed value by one word, `(s + w p) / 2^64` for `s = t + 2^61 p` and "
            "`w = s inv mod 2^64`, as four words. For the Pasta shape no top carry is captured and "
            "no conditional subtraction follows."
        ),
        (("t", "Signed5"), ("modulus", "Limbs"), ("inv", "Nat")),
    ),
    Block(
        "cond_sub",
        "condSubBlock",
        (
            "The inline `asm!` block of `cond_sub`: the subtraction of the modulus from `value`, kept "
            "unless it borrows. For `value < 2p` the result is `value mod p`."
        ),
        (("value", "Limbs"), ("modulus", "Limbs")),
    ),
)
