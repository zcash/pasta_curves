#!/usr/bin/env python3
"""AArch64 backend for the unified Lean transcription generator.

The shared fail-closed Rust/``asm!`` parser feeds this architecture-specific
instruction emitter. It produces

- `lean/PastaCurves/AArch64/Transcription.lean`: the AArch64 blocks as Lean definitions; and
- `lean/PastaCurves/AArch64/Vectors.lean`: kernel-checked AArch64 reference vectors.

Invoke it through ``python3 lean/scripts/gen.py``.

The transcription is deliberately mechanical. A block is read from its template lines, with
the operand placeholders as register names, rebound by each instruction that writes them:
the `in` and `inout` operands bind argument limbs and word arguments, the named `out` and the
`inout` operands are the result words, and the block ends as a routine does. The compiler's
allocation of registers to the operands is not modelled; the script checks that every
register the block reads was written by the block or bound by an operand.

A template line may also be an invocation of a `macro_rules!` macro of the source file whose
arms expand to instruction lines (see `asm_source.parse_macros`). An arm listed in
`MACRO_ROUNDS` is transcribed once, as a definition over the registers it carries, and each
invocation becomes a call of it; any other arm is expanded in place.

A binding that nothing later reads is not emitted. For an operand, or for an output of a
round call, this records that the block binds a value it never uses, and the dropped binding
is left as a comment; for a flag it is an ordinary unread flag write. A computed register that
is never read would be dead code in the block and is reported as an error, since none is
expected.

Run from the repository root:

    python3 lean/scripts/gen.py

`lean/scripts/check.sh` regenerates and fails if the output differs from the committed
files.

The script also generates the mechanical part of each block's correctness proof in
`lean/PastaCurves/AArch64/Spec.lean`: `--skeleton AArch64:NAME` prints it (see `skeleton`), and
`--check-spec FILE` checks that FILE contains every block's skeleton verbatim once its
`-- BEGIN ... -- END` annotation blocks are removed; the check script runs that too.
Python 3.9+; stdlib only.
"""

import dataclasses
import re
import textwrap
from pathlib import Path

import asm_source
import gen

INLINE = Path("src/asm/aarch64.rs")
OUT_PROGRAM = Path("lean/PastaCurves/AArch64/Transcription.lean")

# Names the transcription uses for its own bindings; an operand may not take them.
RESERVED_OPERAND_NAMES = {"s", "c", "fl"}

HEADER = """/-
Copyright Supranational LLC (the routines, transcribed from Semolina v0.1.4).
Copyright (c) 2026 the pasta_curves contributors (the transcription).
-/
"""

# Code longer than this does not set the instruction-comment column (see `Routine.text`): the
# round calls, and the longer expressions of the inversion blocks, take their comment two spaces
# after the code, so that a new block does not re-align the comments of the existing ones.
COMMENT_COLUMN_MAX = 30

# Blocks whose instruction stream Semolina's generator (`pasta_mul-armv8.pl`) emitted as a
# prologue, a loop body repeated a fixed number of times, and an epilogue. The body is
# transcribed once, as a function of the registers that cross its boundary; the block calls it
# once per round. `end` is the instruction that ends the prologue and each body. The rounds
# differ in one place only, which is checked: `operand`, the register holding the round's `rhs`
# limb (`{i}` is the round number), which the round takes as its parameter `param`. `roles`
# lists the registers carried from one round to the next as (register, field, description); the
# structure `state` has those fields, `value` names the ones that make up the accumulator, and
# `emit_struct` says whether this block's transcription declares the structure.
LOOPS = {
    "mulMont": {
        "end": "lsl t3,q,#62",
        "count": 3,
        "operand": "b{i}",
        "round": "mulMontRound",
        "state": "MulMontAcc",
        "emit_struct": True,
        "param": "b",
        "arg": "acc",
        "value": ["r0", "r1", "r2", "r3", "r4"],
        "roles": [
            ("r0", "r0", "accumulator limb 0"),
            ("r1", "r1", "accumulator limb 1"),
            ("r2", "r2", "accumulator limb 2"),
            ("r3", "r3", "accumulator limb 3"),
            ("r4", "r4", "accumulator limb 4"),
            ("q", "q", "the round's Montgomery quotient `q`"),
            ("t1", "t1", "`low(p1 * q)`, the first term of the round's reduction"),
            ("t3", "t3", "`low(q * 2^62)`, the third term of the round's reduction"),
        ],
    },
}

# The Lean type of each argument kind and, for a structure, its fields in order.
KIND_FIELDS = {
    "Limbs": ["l0", "l1", "l2", "l3"],
    "Signed5": ["l0", "l1", "l2", "l3", "l4"],
    "Nat": None,
}


@dataclasses.dataclass(frozen=True)
class RoutineConfig:
    """An inline `asm!` block of the crate to transcribe: the Rust function whose block to read,
    the Lean name, the docstring, the arguments as (name, kind) in signature order, and the
    result's Lean type. A result type other than `Limbs` or `Signed5` is a structure declared by
    the transcription, with `result_fields` as its fields in the source's result order and
    `result_doc` as its docstring."""

    rust_name: str
    lean_name: str
    doc: str
    args: tuple
    result: str = "Limbs"
    result_fields: tuple = ("l0", "l1", "l2", "l3")
    result_doc: str = ""

    @property
    def arg_names(self):
        return [name for name, _ in self.args]

    def fields(self, arg):
        """The fields of argument `arg`, or `None` for a word."""
        return KIND_FIELDS[dict(self.args)[arg]]


# The inline `asm!` blocks of the crate. The block's template lines are the instruction stream;
# its `in` and `inout` operands bind the arguments (`lhs[0]` is `lhs.l0`, a word argument such
# as `inv` is itself, and a `let mut a0 = value[0];` before the block makes the `inout` operand
# `a0` the limb `value.l0`), and its named `out` and its `inout` operands are the result words.
INLINE_ROUTINES = [
    RoutineConfig(
        "mul",
        "mulMont",
        (
            "The inline `asm!` block of `mul`: Montgomery multiplication, `lhs * rhs * 2^-256 mod p`, "
            "with the result in the block's output operands. Its rounds are those of Semolina's "
            "`mul_mont_pasta`; its epilogue keeps four limbs of the final candidate."
        ),
        (("lhs", "Limbs"), ("rhs", "Limbs"), ("modulus", "Limbs"), ("inv", "Nat")),
    ),
    RoutineConfig(
        "square",
        "sqrMont",
        (
            "The inline `asm!` block of `square`: Montgomery squaring, `value^2 * 2^-256 mod p`, the "
            "squaring loop body of Semolina's `sqr_n_mul_mont_pasta` followed by a conditional "
            "subtraction, with the result in the block's `inout` operands."
        ),
        (("value", "Limbs"), ("modulus", "Limbs"), ("inv", "Nat")),
    ),
    RoutineConfig(
        "add",
        "addMod",
        (
            "The inline `asm!` block of `add`: modular addition, `lhs + rhs mod p`, as a full-width "
            "addition, a subtraction of the modulus, and the selection of the reduced sum when that "
            "subtraction did not borrow, with the result in the block's `inout` operands."
        ),
        (("lhs", "Limbs"), ("rhs", "Limbs"), ("modulus", "Limbs")),
    ),
    RoutineConfig(
        "sub",
        "subMod",
        (
            "The inline `asm!` block of `sub`: modular subtraction, `lhs - rhs mod p`, as a full-width "
            "subtraction and the addition of the modulus when it borrowed, with the result in the "
            "block's `inout` operands."
        ),
        (("lhs", "Limbs"), ("rhs", "Limbs"), ("modulus", "Limbs")),
    ),
    RoutineConfig(
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
        result_fields=("d", "m00", "m01", "m10", "m11"),
        result_doc=(
            "The result of `divstep59Block`: the new `d` and the entries of the 59-step matrix, each a "
            "two's-complement word."
        ),
    ),
    RoutineConfig(
        "sign_mag",
        "signMagBlock",
        (
            "The inline `asm!` block of `sign_mag`: the sign-magnitude form of the four entries of a "
            "transition matrix, each entry's magnitude and its sign as a mask (all ones when negative, "
            "else zero), which the row blocks take."
        ),
        (("m00", "Nat"), ("m01", "Nat"), ("m10", "Nat"), ("m11", "Nat")),
        result="SignMag",
        result_fields=("m00", "m01", "m10", "m11", "s00", "s01", "s10", "s11"),
        result_doc=(
            "The result of `signMagBlock`: the magnitudes `m00` to `m11` of the matrix entries and "
            "their sign masks `s00` to `s11`."
        ),
    ),
    RoutineConfig(
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
        result_fields=("l0", "l1", "l2", "l3", "l4"),
    ),
    RoutineConfig(
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
        result_fields=("l0", "l1", "l2", "l3", "l4"),
    ),
    RoutineConfig(
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
]

# The macro arms (see `asm_source.parse_macros`) that are transcribed as round definitions,
# keyed by (macro, arm): the definition's name, the structure of the registers it carries in
# and out, `roles` as (register, field, description) for that structure, and whether this arm's
# transcription declares the structure. The field `fl` holds the flags; every other field is a
# word. A round's body may read only carried registers.
MACRO_ROUNDS = {
    ("divstep", ""): {
        "round": "divstepRound",
        "doc": (
            "one packed half-delta divstep on the words `f` and `g` at `d`, from the flags of the "
            "parity test of `g`, ending with the parity test of the new `g` for the next step"
        ),
        "state": "DivstepState",
        "emit_struct": True,
        "arg": "st",
        "call": "step",
        "roles": [
            ("d", "d", "the doubled half-delta `d`, as a two's-complement word"),
            ("pf", "f", "the packed `f` word"),
            ("pg", "g", "the packed `g` word"),
            ("fl", "fl", "the flags, set by the parity test of the packed `g` word"),
        ],
    },
    ("divstep", "last"): {
        "round": "divstepLast",
        "doc": "the last divstep of a batch: `divstepRound` without the parity test at its end",
        "state": "DivstepState",
        "emit_struct": False,
        "arg": "st",
        "call": "step",
        "roles": [
            ("d", "d", None),
            ("pf", "f", None),
            ("pg", "g", None),
            ("fl", "fl", None),
        ],
    },
}


def tokenize(rest):
    return re.findall(r"\[[^\]]*\]!?|[^,\s]+", rest)


def imm(tok):
    """An immediate operand such as `#62` or `8*1`, as an integer."""
    s = tok.lstrip("#")
    if re.fullmatch(r"0x[0-9a-fA-F]+", s):
        return int(s, 16)
    if not re.fullmatch(r"-?[0-9]+(\*[0-9]+)?", s):
        raise ValueError(f"unexpected immediate {tok}")
    return eval(s)


class Emitter(gen.Emitter):
    """AArch64 instruction decoder backed by the shared binding and liveness IR."""

    def __init__(self, ins, directions=None):
        super().__init__()
        self.ins = ins
        self.directions = directions or {}
        # Which flag binding the last flag-setting instruction produced: `c` (a carry chain) or
        # `fl` (the four flags), so that a condition reads the flags that are current.
        self.flags = None

    def read(self, tok):
        if tok == "xzr":
            return "0"
        if tok.startswith("#"):
            return str(imm(tok))
        if tok in ("c", "fl") and self.flags != tok:
            raise ValueError(f"{tok} read while the flags are not in that form")
        if tok not in self.known:
            raise ValueError(f"{tok} read before being written")
        self.cur_reads.add(tok)
        return tok

    def bind(self, name, expr, comment, reads=None, load=False, fact=None, note=None):
        if name in ("c", "fl"):
            self.flags = name
        if name != "xzr":
            super().bind(name, expr, comment, reads=reads, load=load, fact=fact, note=note)

    def operand(self, toks, text):
        """A second source operand: a register or immediate, optionally shifted left by an
        immediate (`b, lsl #k`), as an expression."""
        if len(toks) == 1:
            return self.read(toks[0])
        if len(toks) == 3 and toks[1] == "lsl":
            k = imm(toks[2])
            if toks[0].startswith("#"):
                return str(imm(toks[0]) * 2**k)
            return f"(lsl {self.read(toks[0])} {k})"
        raise ValueError(f"unsupported operand: {text}")

    def step(self, op, t, text):
        # The number of operand tokens, or the numbers a shifted second source allows.
        arity = {
            "mov": (2,),
            "mul": (3,),
            "umulh": (3,),
            "madd": (4,),
            "msub": (4,),
            "mneg": (3,),
            "lsl": (3,),
            "lsr": (3,),
            "asr": (3,),
            "sbfx": (4,),
            "extr": (4,),
            "adds": (3,),
            "adcs": (3,),
            "adc": (3,),
            "subs": (3,),
            "sbcs": (3,),
            "add": (3, 5),
            "sub": (3,),
            "neg": (2,),
            "and": (3,),
            "orr": (3,),
            "eor": (3,),
            "tst": (2,),
            "cmp": (2,),
            "ccmp": (4,),
            "csel": (4,),
            "cneg": (3,),
            "csetm": (2,),
        }
        if op not in arity:
            raise ValueError(f"unhandled instruction: {text}")
        if len(t) not in arity[op]:
            expected = " or ".join(str(n) for n in arity[op])
            raise ValueError(f"{op} expects {expected} operands, got {len(t)}: {text}")
        if op not in ("tst", "cmp", "ccmp") and t[0] != "xzr":
            if self.directions.get(t[0]) == "in":
                raise ValueError(f"input-only register {t[0]} cannot be written: {text}")
            if t[0] not in self.directions:
                raise ValueError(f"undeclared destination {t[0]}: {text}")
        self.cur_reads = set()
        if op == "mov":
            a = self.read(t[1])
            self.bind(t[0], a, text, fact=("mov", a))
        elif op == "mul":
            a, b = self.read(t[1]), self.read(t[2])
            self.bind(t[0], f"mulLo {a} {b}", text, fact=("mul", a, b))
        elif op == "umulh":
            a, b = self.read(t[1]), self.read(t[2])
            self.bind(t[0], f"umulh {a} {b}", text, fact=("umulh", a, b))
        elif op in ("madd", "msub"):
            a, b, c = self.read(t[1]), self.read(t[2]), self.read(t[3])
            self.bind(t[0], f"{op} {a} {b} {c}", text, fact=(op, a, b, c))
        elif op == "mneg":
            a, b = self.read(t[1]), self.read(t[2])
            self.bind(t[0], f"mneg {a} {b}", text, fact=("mneg", a, b))
        elif op in ("lsl", "lsr", "asr"):
            a, k = self.read(t[1]), imm(t[2])
            self.bind(t[0], f"{op} {a} {k}", text, fact=(op, a, k))
        elif op == "sbfx":
            a, lsb, w = self.read(t[1]), imm(t[2]), imm(t[3])
            self.bind(t[0], f"sbfx {a} {lsb} {w}", text, fact=("sbfx", a, lsb, w))
        elif op == "extr":
            hi, lo, k = self.read(t[1]), self.read(t[2]), imm(t[3])
            self.bind(t[0], f"extr {hi} {lo} {k}", text, fact=("extr", hi, lo, k))
        elif op in ("adds", "adcs", "adc"):
            cin = "0" if op == "adds" else self.read("c")
            a, b = self.read(t[1]), self.read(t[2])
            expr = f"addc {a} {b} {cin}"
            if op == "adc":
                self.bind(t[0], f"({expr}).1", text, fact=("adc", a, b, cin))
            else:
                self.bind("s", expr, text, fact=("adds", a, b, cin))
                self.bind(t[0], "s.1", text, reads=("s",), fact=("fst",), note=f"  `-> {t[0]}")
                self.bind("c", "s.2", text, reads=("s",), fact=("snd",), note="  `-> carry")
        elif op in ("subs", "sbcs"):
            cin = "1" if op == "subs" else self.read("c")
            a, b = self.read(t[1]), self.read(t[2])
            expr = f"subc {a} {b} {cin}"
            if t[0] == "xzr":
                self.bind("c", f"({expr}).2", text, fact=("subs_carry", a, b, cin))
            else:
                self.bind("s", expr, text, fact=("subs", a, b, cin))
                self.bind(t[0], "s.1", text, reads=("s",), fact=("fst",), note=f"  `-> {t[0]}")
                self.bind("c", "s.2", text, reads=("s",), fact=("snd",), note="  `-> carry")
        elif op == "add":
            a, b = self.read(t[1]), self.operand(t[2:], text)
            self.bind(t[0], f"addw {a} {b}", text, fact=("addw", a, b))
        elif op == "sub":
            a, b = self.read(t[1]), self.read(t[2])
            self.bind(t[0], f"subw {a} {b}", text, fact=("subw", a, b))
        elif op == "neg":
            a = self.read(t[1])
            self.bind(t[0], f"negw {a}", text, fact=("negw", a))
        elif op in ("and", "orr", "eor"):
            a, b = self.read(t[1]), self.read(t[2])
            self.bind(t[0], f"{op}w {a} {b}", text, fact=(op, a, b))
        elif op == "tst":
            a, b = self.read(t[0]), self.read(t[1])
            self.bind("fl", f"tstFlags (andw {a} {b})", text, fact=("tst", a, b))
        elif op == "cmp":
            a, b = self.read(t[0]), self.read(t[1])
            self.bind("fl", f"cmpFlags {a} {b}", text, fact=("cmp", a, b))
        elif op == "ccmp":
            if t[3] != "ne":
                raise ValueError(f"unexpected condition: {text}")
            fl, a, b, nzcv = self.read("fl"), self.read(t[0]), self.read(t[1]), imm(t[2])
            self.bind("fl", f"ccmpNe {fl} {a} {b} {nzcv}", text, fact=("ccmp_ne", fl, a, b, nzcv))
        elif op == "csel":
            if t[3] in ("lo", "cc"):
                c, a, b = self.read("c"), self.read(t[1]), self.read(t[2])
                self.bind(t[0], f"cselLo {c} {a} {b}", text, fact=("select", c, a, b))
            elif t[3] in ("cs", "hs"):
                c, a, b = self.read("c"), self.read(t[1]), self.read(t[2])
                self.bind(t[0], f"cselCs {c} {a} {b}", text, fact=("select", c, b, a))
            elif t[3] in ("ne", "ge"):
                fl, a, b = self.read("fl"), self.read(t[1]), self.read(t[2])
                fn = f"csel{t[3].capitalize()}"
                self.bind(t[0], f"{fn} {fl} {a} {b}", text, fact=(f"csel_{t[3]}", fl, a, b))
            else:
                raise ValueError(f"unexpected condition: {text}")
        elif op == "cneg":
            if t[2] not in ("ge", "mi"):
                raise ValueError(f"unexpected condition: {text}")
            fl, a = self.read("fl"), self.read(t[1])
            fn = f"cneg{t[2].capitalize()}"
            self.bind(t[0], f"{fn} {fl} {a}", text, fact=(f"cneg_{t[2]}", fl, a))
        elif op == "csetm":
            if t[1] != "mi":
                raise ValueError(f"unexpected condition: {text}")
            fl = self.read("fl")
            self.bind(t[0], f"csetmMi {fl}", text, fact=("csetm_mi", fl))
        else:
            raise ValueError(f"unhandled instruction: {text}")

    def run(self, start):
        pc = start
        while True:
            op, tokens, text = self.ins[pc]
            if op == "ret":
                self.end_pc = pc
                return
            self.pc = pc
            self.step(op, tokens, text)
            pc += 1


# Fields of the argument structures, by argument name, for the skeleton's bound hypotheses.
LIMB_FIELDS = ["l0", "l1", "l2", "l3"]
ARG_FIELDS = {arg: LIMB_FIELDS for arg in ("t", "lhs", "rhs", "value", "modulus")}


def loop_routines(e, ins, name, doc, args, result, cfg):
    """Split the transcription of a routine that the assembly's generator emitted as prologue,
    repeated body, and epilogue (see `LOOPS`): the body becomes its own definition over the
    registers that cross its boundary, and the routine calls it once per round."""
    first_pc = min(en["pc"] for en in e.entries if en["pc"] is not None)
    ends = [p for p in range(first_pc, e.end_pc) if ins[p][2] == cfg["end"]]
    if len(ends) != cfg["count"] + 1:
        raise ValueError(f"{name}: expected {cfg['count'] + 1} `{cfg['end']}`, found {len(ends)}")
    count = cfg["count"]
    template = cfg["operand"]
    varying = [None] + [template.format(i=k) for k in range(1, count + 1)]
    bodies = []
    for k in range(1, count + 1):
        texts = [ins[p][2] for p in range(ends[k - 1] + 1, ends[k] + 1)]
        texts = [re.sub(rf"\b{re.escape(varying[k])}\b", varying[1], t) for t in texts]
        bodies.append(texts)
    if any(b != bodies[0] for b in bodies):
        raise ValueError(f"{name}: the round bodies differ beyond `{varying[1]}`")

    def part(en):
        pc = en["pc"]
        if pc is None or pc <= ends[0]:
            return "prologue"
        for k in range(1, count + 1):
            if ends[k - 1] < pc <= ends[k]:
                return ("body", k)
        return "epilogue"

    prologue = [en for en in e.entries if part(en) == "prologue"]
    body = [en for en in e.entries if part(en) == ("body", 1)]
    after = [en for en in e.entries if part(en) not in ("prologue", ("body", 1))]
    # The register holding the round's `rhs` limb: the operand.
    loaded = varying[1]
    body_entries = body
    # Registers the body reads before writing (its inputs), and registers it writes that a
    # later instruction reads before they are written again (its outputs).
    bound, live_in = set(), []
    for en in body_entries:
        for r in sorted(en["reads"]):
            if r not in bound and r not in live_in and r != loaded:
                live_in.append(r)
        bound.add(en["name"])
    rebound, live_out = set(), []
    for en in after:
        for r in sorted(en["reads"]):
            if r in bound and r not in rebound and r not in live_out:
                live_out.append(r)
        rebound.add(en["name"])
    regs = [r for r, _, _ in cfg["roles"]]
    fields = [f for _, f, _ in cfg["roles"]]
    if sorted(live_out) != sorted(regs):
        raise ValueError(
            f"{name}: registers carried between rounds are {sorted(live_out)}, "
            f"`roles` lists {sorted(regs)}"
        )
    # An input is an invariant argument if its value on entry is an argument limb or `inv`;
    # otherwise it is carried state.
    defs = {}
    for en in prologue:
        defs[en["name"]] = en
    invariant, carried = [], []
    for r in live_in:
        d = defs[r]
        if d["fact"][0] in ("load", "inv"):
            invariant.append((r, d))
        else:
            carried.append(r)
    if sorted(carried) != sorted(regs):
        raise ValueError(
            f"{name}: registers read from the previous round are {sorted(carried)}, "
            f"`roles` lists {sorted(regs)}"
        )
    limb_args = [
        arg
        for arg in args
        if any(d["fact"][0] == "load" and d["fact"][1] == arg for _, d in invariant)
    ]
    uses_inv = any(d["fact"][0] == "inv" for _, d in invariant)
    param, sarg = cfg["param"], cfg["arg"]
    # The round: its inputs bound as arguments, then the body.
    re_ = Emitter(ins)
    order = {arg: i for i, arg in enumerate(limb_args)}

    def key(rd):
        d = rd[1]
        return (0, order[d["fact"][1]], d["fact"][2]) if d["fact"][0] == "load" else (1, 0, "")

    for r, d in sorted(invariant, key=key):
        re_.bind(r, d["expr"], "argument", reads=(), load=(d["fact"][0] == "load"), fact=d["fact"])
    re_.bind(loaded, param, "argument", reads=(), fact=("param", param))
    for r, f in zip(regs, fields):
        re_.bind(r, f"{sarg}.{f}", "argument", reads=(), load=True, fact=("load", sarg, f))
    re_.entries += [dict(en) for en in body_entries]
    re_.cur_reads = set()
    round_result = [re_.read(r) for r in regs]
    sig = (
        f"def {cfg['round']} ({' '.join(limb_args)} : Limbs) "
        f"({'inv ' if uses_inv else ''}{param} : Nat) "
        f"({sarg} : {cfg['state']}) : {cfg['state']} :="
    )
    where = f"the instructions between one `{cfg['end']}` and the next"
    round_doc = (
        f"One round of `{name}`: {where} in each of its {count} rounds, on the "
        f"round's `rhs` limb `{param}` and the registers `{sarg}` carried from the "
        "previous round."
    )
    struct = None
    if cfg["emit_struct"]:
        value = cfg["value"]
        lines = [
            f"/-- The registers that `{name}` carries from one round to the next. -/",
            f"structure {cfg['state']} where",
        ]
        for _, f, desc in cfg["roles"]:
            lines += [f"  /-- {desc} -/", f"  {f} : Nat"]
        lines += [
            "  deriving DecidableEq, Repr",
            "",
            f"namespace {cfg['state']}",
            "",
            "/-- Every field is below `2^64`. -/",
            f"def Bounded (s : {cfg['state']}) : Prop :=",
        ]
        lines += wrap_tactic(
            "",
            [f"s.{f} < 2^64" + (" ∧" if i < len(fields) - 1 else "") for i, f in enumerate(fields)],
            "",
            indent="  ",
        )
        lines += [
            "",
            (
                f"/-- The accumulator's value: limbs `{'`, `'.join(value)}` with weights "
                f"`2^0` to `2^{64 * (len(value) - 1)}`. -/"
            ),
            f"def toNat (s : {cfg['state']}) : Nat :=",
        ]
        terms = [f"{f'2^{64 * i} * ' if i else ''}s.{f}" for i, f in enumerate(value)]
        lines += wrap_tactic(
            "",
            [t + (" +" if i < len(value) - 1 else "") for i, t in enumerate(terms)],
            "",
            indent="  ",
        )
        lines += ["", f"end {cfg['state']}", ""]
        struct = "\n".join(lines)
    rnd = Routine(
        round_doc,
        sig,
        re_.render(round_result),
        f"  ⟨{', '.join(round_result)}⟩",
        cfg["round"],
        re_,
        round_result,
        struct=struct,
        arg_fields={sarg: fields},
    )
    # The block: prologue, then per round the call and the outputs.
    me = Emitter(ins)
    me.entries = [dict(en) for en in prologue]
    inv_regs = [r for r, _ in invariant if defs[r]["fact"][0] == "inv"]
    n_scalar = len(inv_regs) + 1
    fmt = (
        f"{cfg['round']} {' '.join(limb_args)} "
        + " ".join(f"{{{i}}}" for i in range(n_scalar))
        + " ⟨"
        + ", ".join(f"{{{n_scalar + i}}}" for i in range(len(regs)))
        + "⟩"
    )
    for k in range(1, count + 1):
        call_regs = inv_regs + [varying[k]] + regs
        rname = f"round{k}"
        me.entries.append(
            {
                "name": rname,
                "expr": fmt.format(*call_regs),
                "comment": f"round {k}",
                "reads": set(call_regs),
                "load": False,
                "fact": ("call", fmt, call_regs),
                "pc": None,
            }
        )
        for r, f in zip(regs, fields):
            me.entries.append(
                {
                    "name": r,
                    "expr": f"{rname}.{f}",
                    "comment": f"round {k} output",
                    "reads": {rname},
                    "load": False,
                    "fact": ("callout", rname, f),
                    "pc": None,
                }
            )
    me.entries += [dict(en) for en in e.entries if part(en) == "epilogue"]
    main = Routine(
        doc,
        f"def {name} ({' '.join(args)} : Limbs) (inv : Nat) : Limbs :=",
        me.render(result),
        f"  ⟨{', '.join(result)}⟩",
        name,
        me,
        result,
    )
    return [rnd, main]


def parse_inline(path, config):
    """Adapt the shared Rust asm parser to the AArch64 emitter's instruction representation."""
    fn = config.rust_name
    parsed = asm_source.parse_function(
        path.read_text(),
        fn,
        config.arg_names,
        len(config.result_fields),
        reserved_names=RESERVED_OPERAND_NAMES,
        allowed_options={"pure", "nomem", "nostack"},
        required_options={"pure", "nomem", "nostack"},
    )
    ins = []
    for template in parsed.instructions:
        text = re.sub(r"\{(\w+)\}", r"\1", template)
        text = re.sub(r"\s+", " ", text.replace(", ", ",")).strip()
        match = re.match(r"(\S+)\s*(.*)", text)
        ins.append((match.group(1), tokenize(match.group(2)), text))
    ins.append(("ret", [], "ret"))
    asm_source.declaration_directions(parsed, fn)
    outputs = asm_source.output_bindings(parsed, fn)
    returned = asm_source.returned_registers(parsed, fn)
    decls = [(decl.name, decl.kind, decl.value, decl.output) for decl in parsed.declarations]
    return ins, decls, parsed.locals, outputs, returned, parsed.origins


def signature(name, args, result):
    """`def name (a b : K) (c : K') : result :=`, grouping consecutive arguments of one kind."""
    groups = []
    for arg, kind in args:
        if groups and groups[-1][1] == kind:
            groups[-1][0].append(arg)
        else:
            groups.append(([arg], kind))
    params = " ".join(f"({' '.join(names)} : {kind})" for names, kind in groups)
    return f"def {name} {params} : {result} :="


def result_struct(config):
    """The declaration of a block's result structure, when the result is not a shared type."""
    if config.result in KIND_FIELDS:
        return None
    lines = [gen.docstring(config.result_doc), f"structure {config.result} where"]
    lines += [f"  {f} : Nat" for f in config.result_fields]
    lines += ["  deriving DecidableEq, Repr", ""]
    return "\n".join(lines)


def emit_inline(config):
    name, doc, args = config.lean_name, config.doc, config.arg_names
    ins, decls, lets, named_outputs, returned, origins = parse_inline(INLINE, config)
    e = Emitter(ins, {n: kind for n, kind, _, _ in decls})
    outs = []
    for n, kind, v, out in decls:
        if kind in ("in", "inout"):
            m = re.fullmatch(r"(\w+)\[(\d)\]", v)
            if m:
                arg, i = m.group(1), int(m.group(2))
            elif v in lets:
                arg, i = lets[v]
            elif v in args and config.fields(v) is None:
                fact = ("inv",) if v == "inv" else ("param", v)
                e.bind(n, v, "argument", reads=(), fact=fact)
                arg = None
            else:
                raise ValueError(f"{name}: unexpected input operand {n} = {v}")
            if arg is not None:
                if arg not in args:
                    raise ValueError(f"{name}: operand {n} reads {v}, not an argument")
                fields = config.fields(arg)
                if fields is None or i >= len(fields):
                    raise ValueError(f"{name}: operand {n} reads {v}, which {arg} does not have")
                e.bind(
                    n, f"{arg}.l{i}", "argument", reads=(), load=True, fact=("load", arg, f"l{i}")
                )
            # An `inout` operand whose output is discarded (`=> _`) is an input only.
            if kind == "inout" and out != "_":
                if named_outputs.get(out) != n:
                    raise ValueError(f"{name}: output binding {out} does not name operand {n}")
                outs.append((out, n))
        elif kind == "out":
            if v != "_":
                if named_outputs.get(v) != n:
                    raise ValueError(f"{name}: output binding {v} does not name operand {n}")
                outs.append((v, n))
        else:
            raise ValueError(f"{name}: unsupported operand direction {kind}")
    ordered_outputs = tuple(reg for _, reg in sorted(outs))
    if ordered_outputs != returned:
        raise ValueError(
            f"{name}: source output order {returned} differs from the order of the output "
            f"names {ordered_outputs}"
        )
    e.run(0)
    e.cur_reads = set()
    result = [e.read(reg) for reg in ordered_outputs]
    if len(result) != len(config.result_fields):
        raise ValueError(f"{name}: {len(result)} output operands")
    arg_fields = {arg: fields for arg, fields in ((a, config.fields(a)) for a in args) if fields}
    if name in LOOPS:
        limb_args = [arg for arg in args if config.fields(arg)]
        return loop_routines(e, ins, name, doc, limb_args, result, LOOPS[name])
    if any(origins):
        return macro_routines(e, ins, origins, config, result, arg_fields)
    return [
        Routine(
            doc,
            signature(name, config.args, config.result),
            e.render(result),
            f"  ⟨{', '.join(result)}⟩",
            name,
            e,
            result,
            struct=result_struct(config),
            arg_fields=arg_fields,
        )
    ]


def state_struct(cfg):
    """The declaration of a round's carried-register structure (see `MACRO_ROUNDS`)."""
    lines = [
        f"/-- The registers that `{cfg['round']}` carries in and out. -/",
        f"structure {cfg['state']} where",
    ]
    for _, f, desc in cfg["roles"]:
        lines += [f"  /-- {desc} -/", f"  {f} : {'Flags' if f == 'fl' else 'Nat'}"]
    words = [f for _, f, _ in cfg["roles"] if f != "fl"]
    lines += [
        "  deriving DecidableEq, Repr",
        "",
        f"namespace {cfg['state']}",
        "",
        "/-- Every word is below `2^64`. -/",
        f"def Bounded (s : {cfg['state']}) : Prop :=",
    ]
    lines += wrap_tactic(
        "",
        [f"s.{f} < 2^64" + (" ∧" if i < len(words) - 1 else "") for i, f in enumerate(words)],
        "",
        indent="  ",
    )
    lines += ["", f"end {cfg['state']}", ""]
    return "\n".join(lines)


def macro_routines(e, ins, origins, config, result, arg_fields):
    """Split the transcription of a routine whose template invokes macros: each arm listed in
    `MACRO_ROUNDS` becomes a definition over the registers it carries (`roles`), and each of its
    invocations a call; other arms stay expanded in place."""
    name, doc = config.lean_name, config.doc
    origin_of = {pc: origin for pc, origin in enumerate(origins)}

    def origin(en):
        return origin_of.get(en["pc"]) if en["pc"] is not None else None

    def cfg_of(o):
        return MACRO_ROUNDS.get((o.macro, o.arm)) if o is not None else None

    # The entries in segments: consecutive entries of one round invocation form a segment.
    segments, current, current_site = [], [], object()
    for en in e.entries:
        o = origin(en)
        site = o.site if cfg_of(o) else None
        if site != current_site:
            if current:
                segments.append((current_site, current))
            current, current_site = [], site
        current.append(en)
    if current:
        segments.append((current_site, current))

    # The round definitions, one per arm, from each arm's first invocation.
    rounds = {}  # (macro, arm) -> Routine
    round_of_site = {}
    for site, entries in segments:
        if site is None:
            continue
        o = origin(entries[0])
        key = (o.macro, o.arm)
        cfg = MACRO_ROUNDS[key]
        round_of_site[site] = cfg
        regs = [r for r, _, _ in cfg["roles"]]
        fields = [f for _, f, _ in cfg["roles"]]
        # The body's inputs: registers read before the body writes them.
        bound, live_in = set(), []
        for en in entries:
            for r in sorted(en["reads"]):
                if r not in bound and r not in live_in:
                    live_in.append(r)
            bound.add(en["name"])
        if key in rounds:
            continue
        stray = sorted(set(live_in) - set(regs))
        if stray:
            raise ValueError(f"{name}: {cfg['round']} reads {stray}, which `roles` does not carry")
        unused = sorted(set(regs) - set(live_in) - bound)
        if unused:
            raise ValueError(f"{name}: {cfg['round']} neither reads nor writes {unused}")
        re_ = Emitter(ins)
        sarg = cfg["arg"]
        for r, f in zip(regs, fields):
            re_.bind(r, f"{sarg}.{f}", "argument", reads=(), load=True, fact=("load", sarg, f))
        re_.entries += [dict(en) for en in entries]
        re_.flags = "fl" if "fl" in bound or "fl" in live_in else None
        re_.cur_reads = set()
        round_result = [re_.read(r) for r in regs]
        rounds[key] = Routine(
            f"`{o.macro}!({o.arm})`, {cfg['doc']}",
            f"def {cfg['round']} ({sarg} : {cfg['state']}) : {cfg['state']} :=",
            re_.render(round_result),
            f"  ⟨{', '.join(round_result)}⟩",
            cfg["round"],
            re_,
            round_result,
            struct=state_struct(cfg) if cfg["emit_struct"] else None,
            arg_fields={sarg: [f for f in fields if f != "fl"]},
        )
    # The block: each round segment becomes a call and the outputs it carries.
    me = Emitter(ins)
    for site, entries in segments:
        if site is None:
            me.entries += [dict(en) for en in entries]
            continue
        cfg = round_of_site[site]
        regs = [r for r, _, _ in cfg["roles"]]
        fields = [f for _, f, _ in cfg["roles"]]
        rname = f"{cfg['call']}{site + 1}"
        o = origin(entries[0])
        arm = f"{o.macro}!({o.arm})"
        fmt = f"{cfg['round']} ⟨" + ", ".join(f"{{{i}}}" for i in range(len(regs))) + "⟩"
        me.entries.append(
            {
                "name": rname,
                "expr": fmt.format(*regs),
                "comment": f"{arm}, invocation {site + 1}",
                "reads": set(regs),
                "load": False,
                "fact": ("call", fmt, regs),
                "pc": None,
            }
        )
        for r, f in zip(regs, fields):
            me.entries.append(
                {
                    "name": r,
                    "expr": f"{rname}.{f}",
                    "comment": f"{arm}, invocation {site + 1} output",
                    "reads": {rname},
                    "load": False,
                    "optional": True,
                    "fact": ("callout", rname, f),
                    "pc": None,
                }
            )
    main = Routine(
        doc,
        signature(name, config.args, config.result),
        me.render(result),
        f"  ⟨{', '.join(result)}⟩",
        name,
        me,
        result,
        struct=result_struct(config),
        arg_fields=arg_fields,
    )
    return list(rounds.values()) + [main]


class SkeletonBackend(gen.SkeletonBackend):
    """AArch64 grouping and proof facts used by the shared skeleton traversal."""

    def prepare(self, emitter, entries):
        entries = [dict(entry) for entry in entries]
        names = gen.ssa_names(entries)
        i = 0
        while i < len(entries):
            kind = entries[i]["fact"][0]
            if kind not in ("adds", "subs"):
                i += 1
                continue
            # An `adds`/`subs` whose carry nothing reads has no carry binding (the inline
            # block's last shift), so its group is the pair and the carry is a ghost, as for
            # `adc`.
            dead_carry = i + 2 >= len(entries) or entries[i + 2]["fact"] != ("snd",)
            group_count = 2 if dead_carry else 3
            entries[i]["group"] = list(
                zip(entries[i : i + group_count], names[i : i + group_count])
            )
            entries[i]["group_label"] = names[i + 1]
            entries[i]["dead_carry"] = dead_carry
            i += group_count
        return gen.SkeletonPreparation(entries, names)

    def fact(self, kind, ops, context):
        if kind in ("adds", "subs"):
            a, b, cin = ops
            i, entries = context.index, context.entries
            xn = context.group_names[1]
            dead_carry = context.entry["dead_carry"]
            cn = f"k_{xn}" if dead_carry else context.group_names[2]
            if kind == "adds":
                val = f"({a} + {b} + {cin})"
                lin = f"{xn} + 2^64 * {cn} = {a} + {b} + {cin}"
                lin_proof = "Nat.mod_add_div _ _"
                carry_proof = f"addc_carry_le_one {a} {b} {cin} {context.lt64(a)} {context.lt64(b)} {context.le1(cin)}"
            else:
                val = f"({a} + 2^64 - {b} - (1 - {cin}))"
                lin = f"{xn} + 2^64 * {cn} + {b} + 1 = {a} + 2^64 + {cin}"
                lin_proof = f"subc_lin {a} {b} {cin} {context.lt64(b)} {context.le1(cin)}"
                carry_proof = f"subc_carry_le_one {a} {b} {cin} {context.lt64(a)}"
            context.eq(xn, f"{val} % 2^64")
            if dead_carry:
                context.lines.append(
                    f"  have b_{xn} : {xn} < 2^64 := by rw [e_{xn}]; "
                    "exact Nat.mod_lt _ (Nat.two_pow_pos _)"
                )
                context.lines.append(f"  obtain ⟨{cn}, b_{cn}, l_{xn}⟩ :")
                context.lines.append(f"      ∃ k, k ≤ 1 ∧ {lin.replace(cn, 'k')} :=")
                context.lines.append(f"    ⟨{val} / 2^64, {carry_proof},")
                context.lines.append(f"      by rw [e_{xn}]; exact {lin_proof}⟩")
                context.lines.append(f"  clear e_{xn}")
                context.ren[entries[i + 1]["name"]] = xn
                context.bnd[xn] = f"b_{xn}"
                context.consumed = 2
            else:
                context.eq(cn, f"{val} / 2^64")
                context.lines.append(f"  have l_{xn} : {lin} := by")
                context.lines.append(f"    rw [e_{xn}, e_{cn}]; exact {lin_proof}")
                context.lines.append(
                    f"  have b_{xn} : {xn} < 2^64 := by rw [e_{xn}]; exact Nat.mod_lt _ (Nat.two_pow_pos _)"
                )
                context.lines.append(f"  have b_{cn} : {cn} ≤ 1 := by")
                context.lines.append(f"    rw [e_{cn}]; exact {carry_proof}")
                context.lines.append(f"  clear e_{xn} e_{cn}")
                context.ren[entries[i + 1]["name"]] = xn
                context.ren[entries[i + 2]["name"]] = cn
                context.bnd[xn], context.bnd[cn] = f"b_{xn}", f"b_{cn}"
                context.unit_bound.add(cn)
                context.consumed = 3
            return True
        if kind == "subs_carry":
            a, b, cin = ops
            nm = context.name
            context.eq(nm, f"({a} + 2^64 - {b} - (1 - {cin})) / 2^64")
            context.lines.append(
                f"  have b_{nm} : {nm} ≤ 1 := by rw [e_{nm}]; exact subc_carry_le_one {a} {b} {cin} {context.lt64(a)}"
            )
            context.lines.append(
                f"  have l_{nm} : ({nm} = 1 ∧ {b} + 1 ≤ {a} + {cin}) ∨ ({nm} = 0 ∧ {a} + {cin} < {b} + 1) :="
            )
            context.lines.append(
                f"    subc_carry_cases {a} {b} {cin} _ e_{nm} {context.lt64(a)} {context.lt64(b)} {context.le1(cin)}"
            )
            context.lines.append(f"  clear e_{nm}")
            context.bnd[nm] = f"b_{nm}"
            context.unit_bound.add(nm)
            return True
        return False


SKELETON_BACKEND = SkeletonBackend()


class Routine(gen.Routine):
    architecture = "AArch64"
    skeleton_backend = SKELETON_BACKEND


wrap_tactic = gen.wrap_tactic


MODULE_DOC = (
    "GENERATED by `lean/scripts/gen.py` from the `asm!` blocks of BLOCKS in `src/asm/aarch64.rs`; do "
    "not edit by hand. Each definition follows its block instruction by instruction (the "
    "instruction is the trailing comment; the two lines that unpack an instruction's (result, "
    "carry) pair are marked as its continuation), over the semantics of "
    "`PastaCurves.AArch64.Semantics`. Registers are rebound by the instructions that write them, `c` "
    "is the carry flag, `fl` the four flags, `s` is the (result, carry) pair of the instruction "
    "that last set both, argument limbs are read where the block's operands bind them, and the "
    "output words are bound where the block's output operands hold them. A block whose template "
    "invokes a macro for a repeated step calls that step's definition once per invocation. "
    "Bindings that nothing reads are left as comments. See the generator's docstring for what it "
    "checks."
)


def gen_program():
    doc = textwrap.fill(
        MODULE_DOC.replace("BLOCKS", block_list()),
        width=100,
        break_long_words=False,
        break_on_hyphens=False,
    )
    parts = [
        HEADER,
        "import PastaCurves.AArch64.Semantics\n",
        f"""
/-!
# The crate's inline Pasta field blocks, transcribed

{doc}
-/

namespace PastaCurves.AArch64

""",
    ]
    routines = all_routines()
    # One comment column for the whole file: two spaces past the widest ordinary `let`. Lines
    # longer than COMMENT_COLUMN_MAX are outliers (the round calls) and do not set the column.
    column = 2 + max(
        len(code)
        for r in routines
        for code, _ in r.lines
        if code is not None and len(code) <= COMMENT_COLUMN_MAX
    )
    parts.append("\n".join(r.text(column) for r in routines))
    parts.append("\nend PastaCurves.AArch64\n")
    return "".join(parts)


# Proof skeleton construction and checking are shared in gen.py.


def block_list():
    """The blocks' Rust names as an English list, for the generated module's docstring."""
    names = [f"`{config.rust_name}`" for config in INLINE_ROUTINES]
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def all_routines():
    routines = []
    for config in INLINE_ROUTINES:
        routines += emit_inline(config)
    return routines


def generated_outputs():
    """Return the AArch64 generated paths and contents without writing files."""
    return [(OUT_PROGRAM, gen_program())]
