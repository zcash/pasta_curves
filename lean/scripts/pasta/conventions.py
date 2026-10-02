"""How the crate's Lean proofs name things, which the generated skeletons must match."""

from asm2lean.ir import Conventions

LIMBS = ["l0", "l1", "l2", "l3"]

CONVENTIONS = Conventions(
    # The fields of the structured arguments, by argument name, for the `Bounded` projections.
    arg_fields={
        "t": LIMBS,
        "lhs": LIMBS,
        "rhs": LIMBS,
        "value": LIMBS,
        "modulus": LIMBS,
        "product": [f"l{i}" for i in range(8)],
    },
    # The `Bounded` hypothesis of each argument; any other argument `a` has `ha`.
    bound_hyps={
        "t": "ht",
        "modulus": "hm",
        "lhs": "hlhs",
        "rhs": "hrhs",
        "value": "hv",
        "product": "hproduct",
        "acc": "hacc",
    },
    inv_bound="hinv_lt",
)
