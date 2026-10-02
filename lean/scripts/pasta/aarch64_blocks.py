"""The crate's AArch64 blocks: what to transcribe from `src/asm/aarch64.rs`, and how."""

import textwrap

from asm2lean import lean, stream
from asm2lean.aarch64 import Block, Target, transcribe
from asm2lean.aarch64 import stream as block_stream
from asm2lean.reroll import MacroRound, PrologueLoop, Role

from .conventions import CONVENTIONS
from .paths import ROOT

SOURCE = ROOT / "src/asm/aarch64.rs"
OUTPUT = ROOT / "lean/PastaCurves/AArch64/Transcription.lean"
PROGRAMS_OUTPUT = ROOT / "lean/PastaCurves/AArch64/Programs.lean"

HEADER = """/-
Copyright Supranational LLC (the routines, transcribed from Semolina v0.1.4).
Copyright (c) 2026 the pasta_curves contributors (the transcription).
-/
"""

# The instruction-comment column of the transcription. It is fixed, so that a changed or added
# block does not re-align the comments of the others; code that reaches it (the round calls, and the
# longer expressions of the inversion blocks) takes its comment two spaces after the code.
COMMENT_COLUMN = 32

# The Lean type of each argument or result kind and, for a structure, its fields in order. The
# structures are declared in `PastaCurves/Semantics.lean`, shared by the backends.
KINDS = {
    "Limbs": ["l0", "l1", "l2", "l3"],
    "Signed5": ["l0", "l1", "l2", "l3", "l4"],
    "Divstep59Result": ["d", "m00", "m01", "m10", "m11"],
    "SignMag": ["m00", "m01", "m10", "m11", "s00", "s01", "s10", "s11"],
    "Nat": None,
}

# The inline `asm!` blocks of the crate. The block's template lines are the instruction stream;
# its `in` and `inout` operands bind the arguments (`lhs[0]` is `lhs.l0`, a word argument such
# as `inv` is itself, and a `let mut a0 = value[0];` before the block makes the `inout` operand
# `a0` the limb `value.l0`), and its named `out` and its `inout` operands are the result words.
BLOCKS = [
    Block(
        "mul",
        "mulMont",
        (
            "The inline `asm!` block of `mul`: Montgomery multiplication, `lhs * rhs * 2^-256 mod p`, "
            "with the result in the block's output operands. Its rounds are those of Semolina's "
            "`mul_mont_pasta`; its epilogue keeps four limbs of the final candidate."
        ),
        (("lhs", "Limbs"), ("rhs", "Limbs"), ("modulus", "Limbs"), ("inv", "Nat")),
    ),
    Block(
        "square",
        "sqrMont",
        (
            "The inline `asm!` block of `square`: Montgomery squaring, `value^2 * 2^-256 mod p`, the "
            "squaring loop body of Semolina's `sqr_n_mul_mont_pasta` followed by a conditional "
            "subtraction, with the result in the block's `inout` operands."
        ),
        (("value", "Limbs"), ("modulus", "Limbs"), ("inv", "Nat")),
    ),
    Block(
        "add",
        "addMod",
        (
            "The inline `asm!` block of `add`: modular addition, `lhs + rhs mod p`, as a full-width "
            "addition, a subtraction of the modulus, and the selection of the reduced sum when that "
            "subtraction did not borrow, with the result in the block's `inout` operands."
        ),
        (("lhs", "Limbs"), ("rhs", "Limbs"), ("modulus", "Limbs")),
    ),
    Block(
        "sub",
        "subMod",
        (
            "The inline `asm!` block of `sub`: modular subtraction, `lhs - rhs mod p`, as a full-width "
            "subtraction and the addition of the modulus when it borrowed, with the result in the "
            "block's `inout` operands."
        ),
        (("lhs", "Limbs"), ("rhs", "Limbs"), ("modulus", "Limbs")),
    ),
    Block(
        "divstep59",
        "divstep59Block",
        (
            "The inline `asm!` block of `divstep59`: 59 half-delta divsteps on the low words `f0` and "
            "`g0` of `f` and `g` at `d`, as three packed batches of 20, 20, and 19 steps in which "
            "the matrix coefficients ride in the upper bits of the two words, with the two matrix "
            "products between and after the batches. It is s2n-bignum's `divstep59` macro on named "
            "registers. The result is the new `d` and the 59-step matrix, as two's-complement words."
        ),
        (("d", "Nat"), ("f0", "Nat"), ("g0", "Nat")),
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
        (("m00", "Nat"), ("m01", "Nat"), ("m10", "Nat"), ("m11", "Nat")),
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
        "uv_row",
        "uvRowBlock",
        (
            "The inline `asm!` block of `uv_row`: one row of the combination of `u` and `v`, "
            "`m0 u + m1 v` as a five-word signed value, from the row's magnitudes `m0`, `m1` and "
            "sign masks `s0`, `s1`. It is the accumulation of `fgRowBlock` without the shift."
        ),
        (
            ("u", "Limbs"),
            ("v", "Limbs"),
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
]


# `mul` was generated by Semolina's `pasta_mul-armv8.pl` as a prologue, one round body repeated
# three times, and an epilogue; the rounds differ only in the register holding the round's `rhs`
# limb.
MUL_ROUNDS = PrologueLoop(
    end="lsl t3,q,#62",
    count=3,
    operand="b{i}",
    round="mulMontRound",
    state="MulMontAcc",
    param="b",
    arg="acc",
    value=("r0", "r1", "r2", "r3", "r4"),
    roles=(
        Role("r0", "r0", "accumulator limb 0"),
        Role("r1", "r1", "accumulator limb 1"),
        Role("r2", "r2", "accumulator limb 2"),
        Role("r3", "r3", "accumulator limb 3"),
        Role("r4", "r4", "accumulator limb 4"),
        Role("q", "q", "the round's Montgomery quotient `q`"),
        Role("t1", "t1", "`low(p1 * q)`, the first term of the round's reduction"),
        Role("t3", "t3", "`low(q * 2^62)`, the third term of the round's reduction"),
    ),
)

# The macro arms (see `rust.parse_macros`) transcribed as round definitions, by (macro, arm).
MACRO_ROUNDS = {
    ("divstep", ""): MacroRound(
        round="divstepRound",
        doc=(
            "one packed half-delta divstep on the words `f` and `g` at `d`, from the flags of the "
            "parity test of `g`, ending with the parity test of the new `g` for the next step"
        ),
        state="DivstepState",
        emit_struct=True,
        arg="st",
        call="step",
        roles=(
            Role("d", "d", "the doubled half-delta `d`, as a two's-complement word"),
            Role("pf", "f", "the packed `f` word"),
            Role("pg", "g", "the packed `g` word"),
            Role("fl", "fl", "the flags, set by the parity test of the packed `g` word"),
        ),
    ),
    ("divstep", "last"): MacroRound(
        round="divstepLast",
        doc="the last divstep of a batch: `divstepRound` without the parity test at its end",
        state="DivstepState",
        emit_struct=False,
        arg="st",
        call="step",
        roles=(
            Role("d", "d", None),
            Role("pf", "f", None),
            Role("pg", "g", None),
            Role("fl", "fl", None),
        ),
    ),
}

TARGET = Target(KINDS, CONVENTIONS, loops={"mulMont": MUL_ROUNDS}, macro_rounds=MACRO_ROUNDS)

MODULE_DOC = (
    "GENERATED by `lean/scripts/gen.py` from the `asm!` blocks of BLOCKS in `src/asm/aarch64.rs`; do "
    "not edit by hand. Each definition follows its block instruction by instruction (the "
    "instruction is the trailing comment; the two lines that unpack an instruction's (result, "
    "carry) pair are marked as its continuation), over the semantics of "
    "`PastaCurves.AArch64.Semantics`. Registers are rebound by the instructions that write them, `c` "
    "is the carry flag, `fl` the four flags, `s` is the (result, carry) pair of the instruction "
    "that last set both, argument limbs are read where the block's operands bind them, and the "
    "output words are bound where the block's output operands hold them. A block whose template "
    "invokes a macro for a repeated step calls that step's definition once per run of consecutive "
    "invocations, iterated over the run. "
    "Bindings that nothing reads are left as comments. See the generator's docstring for what it "
    "checks."
)


def block_list():
    """The blocks' Rust names as an English list, for the generated module's docstring."""
    names = [f"`{block.rust_name}`" for block in BLOCKS]
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def programs(source=None):
    """Every program of the transcription, in order."""
    source = SOURCE.read_text() if source is None else source
    return [program for block in BLOCKS for program in transcribe(source, block, TARGET)]


def text(source=None):
    """The transcription module."""
    doc = textwrap.fill(
        MODULE_DOC.replace("BLOCKS", block_list()),
        width=100,
        break_long_words=False,
        break_on_hyphens=False,
    )
    definitions = "\n".join(lean.definition(p, COMMENT_COLUMN) for p in programs(source))
    return (
        HEADER
        + "import PastaCurves.AArch64.Semantics\n"
        + f"\n/-!\n# The crate's inline Pasta field blocks, transcribed\n\n{doc}\n-/\n\n"
        + "namespace PastaCurves.AArch64\n\n"
        + definitions
        + "\nend PastaCurves.AArch64\n"
    )


# The column of the instruction comments of the instruction streams.
PROGRAMS_COMMENT_COLUMN = 64

PROGRAMS_DOC = (
    "GENERATED by `lean/scripts/gen.py` from the `asm!` blocks of BLOCKS in `src/asm/aarch64.rs`; do "
    "not edit by hand. Each definition is a block's instruction stream as syntax, one "
    "`PastaCurves.AArch64.Instr` per instruction in the block's order, with the instruction as the "
    "trailing comment: the input of the constant-time proofs, which classify each instruction by "
    "the leakage policy of `PastaCurves.AArch64.Leakage`. The front end that reads the blocks is "
    "the transcription's, and every stream is of a block that lifts."
)


def programs_text(source=None):
    """The instruction streams module."""
    source = SOURCE.read_text() if source is None else source
    doc = textwrap.fill(
        PROGRAMS_DOC.replace("BLOCKS", block_list()),
        width=100,
        break_long_words=False,
        break_on_hyphens=False,
    )
    definitions = "\n".join(
        stream.definition(*block_stream(source, block, TARGET), PROGRAMS_COMMENT_COLUMN)
        for block in BLOCKS
    )
    return (
        HEADER
        + "import PastaCurves.AArch64.Leakage\n"
        + f"\n/-!\n# The crate's inline Pasta field blocks, as instruction streams\n\n{doc}\n-/\n\n"
        + "namespace PastaCurves.AArch64\n\n"
        + definitions
        + "\nend PastaCurves.AArch64\n"
    )
