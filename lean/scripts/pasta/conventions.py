"""How the crate's Lean proofs name things, which the generated skeletons must match."""

from asm2lean.ir import Conventions

CONVENTIONS = Conventions(
    # The hypothesis bounding each argument, as the correctness theorems name it: the `Bounded`
    # hypothesis of a structured argument, the bound of a word argument. Any other argument `a`
    # has `ha`.
    hypotheses={
        "t": "ht",
        "modulus": "hm",
        "lhs": "hlhs",
        "rhs": "hrhs",
        "value": "hv",
        "product": "hproduct",
        "acc": "hacc",
        "inv": "hinv_lt",
    },
    # `lsl62_lsr2_split` (`PastaCurves/Spec.lean`): the 62-bit shift of the high limb `2^62`.
    shift_splits={62: "lsl62_lsr2_split"},
)
