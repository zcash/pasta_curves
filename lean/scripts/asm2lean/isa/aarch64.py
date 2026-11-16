"""The AArch64 instructions that the blocks use, lifted to the IR.

The flags take one of two forms, named after the binding that holds them: `c`, the carry of a
carry chain (`adds`/`adcs`/`subs`/`sbcs`, read by `adc`, `adcs`, `sbcs`, and `csel` on `lo`/`cs`),
or `fl`, the four flags of a test (`tst`, `cmp`, `ccmp`, read by `csel` on `ne`/`ge`, `cneg`,
`csetm`, `ccmp`). A condition reads the form that the last flag-setting instruction produced, and
reading the other is an error. `xzr` reads as `0`; a write to it is discarded. An immediate keeps
the radix of the source, so that a hex constant reads in the transcription as in the assembly.
After a subtraction the carry is set when no borrow occurred (`PastaCurves.AArch64.Semantics`).
"""

from __future__ import annotations

import dataclasses
import enum
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING

from .. import isa
from ..ir import (
    MOD_LT,
    App,
    Asr,
    Component,
    Dead,
    FlagState,
    Get,
    Let,
    Lit,
    Load,
    Mov,
    MulHi,
    MulLo,
    Node,
    Operand,
    PairOp,
    Reg,
    Select,
    Shl,
    Shr,
    Term,
    Word,
    apply,
    bounded,
    dst,
    lit,
    pair_names,
    rename_term,
    show,
    src,
)
from ..rust import Direction, GenerationError
from . import Instruction

if TYPE_CHECKING:
    from ..skeleton import Step

ZERO = "xzr"
CARRY = "c"  # the carry of a carry chain
FLAGS = "fl"  # the four flags of a test
PAIR = "s"  # the (result, carry) pair of the last `addc`/`subc`


class Mnemonic(str, enum.Enum):
    MOV = "mov"
    MUL = "mul"
    UMULH = "umulh"
    MADD = "madd"
    MSUB = "msub"
    MNEG = "mneg"
    LSL = "lsl"
    LSR = "lsr"
    ASR = "asr"
    SBFX = "sbfx"
    EXTR = "extr"
    ADDS = "adds"
    ADCS = "adcs"
    ADC = "adc"
    SUBS = "subs"
    SBCS = "sbcs"
    ADD = "add"
    SUB = "sub"
    NEG = "neg"
    AND = "and"
    ORR = "orr"
    EOR = "eor"
    TST = "tst"
    CMP = "cmp"
    CCMP = "ccmp"
    CSEL = "csel"
    CNEG = "cneg"
    CSETM = "csetm"


# The number of operand tokens of each instruction, or the numbers a shifted source allows.
ARITY = {
    Mnemonic.MOV: (2,),
    Mnemonic.MUL: (3,),
    Mnemonic.UMULH: (3,),
    Mnemonic.MADD: (4,),
    Mnemonic.MSUB: (4,),
    Mnemonic.MNEG: (3,),
    Mnemonic.LSL: (3,),
    Mnemonic.LSR: (3,),
    Mnemonic.ASR: (3,),
    Mnemonic.SBFX: (4,),
    Mnemonic.EXTR: (4,),
    Mnemonic.ADDS: (3,),
    Mnemonic.ADCS: (3,),
    Mnemonic.ADC: (3,),
    Mnemonic.SUBS: (3,),
    Mnemonic.SBCS: (3,),
    Mnemonic.ADD: (3, 5),
    Mnemonic.SUB: (3,),
    Mnemonic.NEG: (2,),
    Mnemonic.AND: (3,),
    Mnemonic.ORR: (3,),
    Mnemonic.EOR: (3,),
    Mnemonic.TST: (2,),
    Mnemonic.CMP: (2,),
    Mnemonic.CCMP: (4,),
    Mnemonic.CSEL: (4,),
    Mnemonic.CNEG: (3,),
    Mnemonic.CSETM: (2,),
}

# Instructions that write only the flags, not a destination register.
FLAG_ONLY = {Mnemonic.TST, Mnemonic.CMP, Mnemonic.CCMP}


class Condition(str, enum.Enum):
    LO = "lo"  # carry clear
    CC = "cc"  # carry clear
    CS = "cs"  # carry set
    HS = "hs"  # carry set
    NE = "ne"
    GE = "ge"
    MI = "mi"


def parse_instruction(text: str) -> Instruction:
    """An instruction of a template line. Its operands keep a placeholder as `{name}`, distinct
    from a bare word (`xzr`, a condition, `lsl`), which a name of the same spelling must not be
    confused with; its text, the transcription's comment, drops the braces and is normalized
    (`, ` to `,`, runs of spaces to one)."""
    normalized = re.sub(r"\s+", " ", text.replace(", ", ",")).strip()
    match = re.match(r"(\S+)\s*(.*)", normalized)
    if match is None:
        raise GenerationError(f"cannot parse instruction {text!r}")
    plain = re.sub(r"\{(\w+)\}", r"\1", normalized)
    return Instruction(match.group(1), tuple(tokenize(match.group(2))), plain)


def tokenize(rest: str) -> list[str]:
    return re.findall(r"\[[^\]]*\]!?|[^,\s]+", rest)


def immediate(token: str) -> int:
    """An immediate operand such as `#62`, `#0x100`, or `8*1`, as an integer."""
    s = token.lstrip("#")
    if re.fullmatch(r"0x[0-9a-fA-F]+", s):
        return int(s, 16)
    m = re.fullmatch(r"(-?[0-9]+)(?:\*([0-9]+))?", s)
    if m is None:
        raise GenerationError(f"unexpected immediate {token}")
    return int(m.group(1)) * int(m.group(2) or 1)


def literal(token: str, shift: int = 0) -> str:
    """An immediate as a Lean literal, in the radix of the source; `shift` is the `lsl` amount of
    a shifted immediate."""
    value = immediate(token) << shift
    if value < 0:
        raise GenerationError(f"negative immediate {token}: write the word it encodes")
    if value >= 2**64:
        raise GenerationError(f"immediate {token} does not fit a word")
    return hex(value) if token.lstrip("#").startswith("0x") else str(value)


def bits(token: str, low: int, high: int, what: str) -> int:
    """An immediate that must lie in `[low, high]`, such as a shift amount."""
    value = immediate(token)
    if not low <= value <= high:
        raise GenerationError(f"{what} {value} is outside [{low}, {high}]")
    return value


# -- the nodes of AArch64's semantics ---------------------------------------------------------


@dataclasses.dataclass(frozen=True, kw_only=True)
class CarryPair(PairOp):
    """`adds`/`adcs` (`addc`) and `subs`/`sbcs` (`subc`): the pair, its result in `dest`, and its
    carry in `c`. A carry that nothing reads is not bound, and the proof obtains it as a ghost."""

    pair = PAIR
    dest: str = dst()
    a: Term = src()
    b: Term = src()
    cin: Term = src()

    def pair_args(self) -> tuple[Operand, ...]:
        return (self.a, self.b, self.cin)

    def components(self) -> list[Component]:
        # The lifter rejects `xzr` as the destination: the proof step needs the value.
        return [
            Component("1", self.dest, f"  `-> {self.source_name('dest')}"),
            Component("2", CARRY, "  `-> carry", Dead.DROP),
        ]

    def label(self, names: list[str]) -> str:
        return names[1]

    def formulas(
        self, a: str, b: str, cin: str, xn: str, cn: str, step: Step
    ) -> tuple[str, str, str, str]:
        raise NotImplementedError

    def prove(self, step: Step) -> None:
        a, b, cin = step.r(self.a), step.r(self.b), step.r(self.cin)
        xn = step.names[1]
        dead_carry = len(step.names) == 2
        cn = f"k_{xn}" if dead_carry else step.names[2]
        val, lin, lin_proof, carry_proof = self.formulas(a, b, cin, xn, cn, step)
        step.eq(xn, f"{val} % 2^64")
        step.bound(xn, MOD_LT)
        if dead_carry:
            step.line(f"  obtain ⟨{cn}, b_{cn}, l_{xn}⟩ :")
            step.line(f"      ∃ k, k ≤ 1 ∧ {lin.replace(cn, 'k')} :=")
            step.line(f"    ⟨{val} / 2^64, {carry_proof},")
            step.line(f"      by rw [e_{xn}]; exact {lin_proof}⟩")
            step.line(f"  clear e_{xn}")
            step.define(1)
        else:
            step.eq(cn, f"{val} / 2^64")
            step.line(f"  have l_{xn} : {lin} := by")
            step.line(f"    rw [e_{xn}, e_{cn}]; exact {lin_proof}")
            step.line(f"  have b_{cn} : {cn} ≤ 1 := by")
            step.line(f"    rw [e_{cn}]; exact {carry_proof}")
            step.line(f"  clear e_{xn} e_{cn}")
            step.define(1)
            step.define(2)
            step.unit(cn)


@dataclasses.dataclass(frozen=True, kw_only=True)
class AddCarry(CarryPair):
    function = "addc"

    def formulas(
        self, a: str, b: str, cin: str, xn: str, cn: str, step: Step
    ) -> tuple[str, str, str, str]:
        return (
            f"({a} + {b} + {cin})",
            f"{xn} + 2^64 * {cn} = {a} + {b} + {cin}",
            "Nat.mod_add_div _ _",
            f"addc_carry_le_one {a} {b} {cin} {step.lt64(self.a)} {step.lt64(self.b)} {step.le1(self.cin)}",
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class SubBorrow(CarryPair):
    function = "subc"

    def formulas(
        self, a: str, b: str, cin: str, xn: str, cn: str, step: Step
    ) -> tuple[str, str, str, str]:
        return (
            f"({a} + 2^64 - {b} - (1 - {cin}))",
            f"{xn} + 2^64 * {cn} + {b} + 1 = {a} + 2^64 + {cin}",
            f"subc_lin {a} {b} {cin} {step.lt64(self.b)} {step.le1(self.cin)}",
            f"subc_carry_le_one {a} {b} {cin} {step.lt64(self.a)}",
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class SubFlag(Node):
    """`subs xzr, a, b`: only the no-borrow flag of the difference."""

    a: Term = src()
    b: Term = src()
    cin: Term = src()

    def lets(self) -> list[Let]:
        term = Get(apply("subc", self.a, self.b, self.cin), "2")
        return [self._let(CARRY, term, dead=Dead.DROP)]

    def prove(self, step: Step) -> None:
        a, b, cin, nm = step.r(self.a), step.r(self.b), step.r(self.cin), step.name
        step.eq(nm, f"({a} + 2^64 - {b} - (1 - {cin})) / 2^64")
        step.line(
            f"  have b_{nm} : {nm} ≤ 1 := by rw [e_{nm}]; exact subc_carry_le_one {a} {b} {cin} {step.lt64(self.a)}"
        )
        step.line(
            f"  have l_{nm} : ({nm} = 1 ∧ {b} + 1 ≤ {a} + {cin}) ∨ ({nm} = 0 ∧ {a} + {cin} < {b} + 1) :="
        )
        step.line(
            f"    subc_carry_cases {a} {b} {cin} _ e_{nm} {step.lt64(self.a)} {step.lt64(self.b)} {step.le1(self.cin)}"
        )
        step.line(f"  clear e_{nm}")
        step.unit(nm)


@dataclasses.dataclass(frozen=True, kw_only=True)
class AddTrunc(Node):
    """`adc`: the sum's low word; the carry-out is obtained as a ghost."""

    dest: str = dst()
    a: Term = src()
    b: Term = src()
    cin: Term = src()

    def lets(self) -> list[Let]:
        return [self._let(self.dest, Get(apply("addc", self.a, self.b, self.cin), "1"))]

    def prove(self, step: Step) -> None:
        a, b, cin, nm = step.r(self.a), step.r(self.b), step.r(self.cin), step.name
        step.eq(nm, f"({a} + {b} + {cin}) % 2^64")
        step.bound(nm, MOD_LT)
        step.line(f"  obtain ⟨k_{nm}, b_k_{nm}, l_{nm}⟩ :")
        step.line(f"      ∃ k, k ≤ 1 ∧ {nm} + 2^64 * k = {a} + {b} + {cin} :=")
        step.line(
            f"    ⟨({a} + {b} + {cin}) / 2^64, addc_carry_le_one {a} {b} {cin} {step.lt64(self.a)} {step.lt64(self.b)} {step.le1(self.cin)},"
        )
        step.line(f"      by rw [e_{nm}]; exact Nat.mod_add_div _ _⟩")
        step.line(f"  clear e_{nm}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class AddShifted(Node):
    """`add d, a, b, lsl #k`: `addw a (lsl b k)`."""

    dest: str = dst()
    a: Term = src()
    b: Term = src()
    k: int

    def lets(self) -> list[Let]:
        return [self._let(self.dest, apply("addw", self.a, apply("lsl", self.b, self.k)))]

    def prove(self, step: Step) -> None:
        a, b = step.r(self.a), step.r(self.b)
        expr = f"addw {a} (lsl {b} {self.k})"
        bounded(step, expr, f"addw_lt {a} (lsl {b} {self.k})")


@dataclasses.dataclass(frozen=True, kw_only=True)
class Sbfx(Node):
    dest: str = dst()
    a: Term = src()
    lsb: int
    width: int

    def lets(self) -> list[Let]:
        return [self._let(self.dest, apply("sbfx", self.a, self.lsb, self.width))]

    def prove(self, step: Step) -> None:
        a, lsb, w = step.r(self.a), self.lsb, self.width
        bounded(step, f"sbfx {a} {lsb} {w}", f"sbfx_lt {a} {lsb} {w} {step.lt64(self.a)}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class SetFlags(Node):
    """A test that sets the four flags: `tst` (`tstFlags (andw a b)`), `cmp` (`cmpFlags a b`),
    and `ccmp` on `ne` (`ccmpNe fl a b nzcv`). Only its equation is recorded."""

    test: str
    args: tuple[Term, ...] = src()

    def term(self) -> App:
        if self.test == "tst":
            return apply("tstFlags", apply("andw", *self.args))
        if self.test == "cmp":
            return apply("cmpFlags", *self.args)
        return apply("ccmpNe", *self.args)

    def lets(self) -> list[Let]:
        return [self._let(FLAGS, self.term(), dead=Dead.DROP)]

    def prove(self, step: Step) -> None:
        step.eq(step.name, show(rename_term(self.term(), step.r)))


@dataclasses.dataclass(frozen=True, kw_only=True)
class LoadFlags(Load):
    """The flags carried into a round: only the equation, since flags have no bound."""

    def prove(self, step: Step) -> None:
        step.eq(step.name, f"{self.arg}.{self.field}")


# -- lifting ------------------------------------------------------------------------------------


class Lifter(isa.Lifter):
    """Lifts a block's AArch64 instructions to nodes, checking that every register is written
    before it is read, that only `out` and `inout` operands are written, and that a condition reads
    the flags in the form that the last flag-setting instruction produced."""

    parse_instruction = staticmethod(parse_instruction)

    def __init__(self, directions: Mapping[str, Direction] | None = None) -> None:
        super().__init__(directions)
        self.flags: str | None = None  # the form of the current flags: `c`, `fl`, or `None`

    def read(self, token: str) -> Term:
        """A source operand of the template: an operand `{x}`, `xzr` (zero), or an immediate."""
        if token == ZERO:
            return Lit("0")
        if token.startswith("#"):
            return Lit(literal(token))
        return Reg(self.require_written(self.placeholder(token)))

    def flag(self, form: str) -> Reg:
        """The flags, in the form `form` (`c` or `fl`) that the instruction reads."""
        if self.flags != form:
            raise GenerationError(f"{form} read while the flags are not in that form")
        return Reg(form)

    def dest(self, token: str) -> str:
        """A destination of the template: an operand `{x}` that may be written, or `xzr`."""
        if token == ZERO:
            return ZERO
        name = self.placeholder(token)
        self.check_writable(name)
        return name

    @staticmethod
    def placeholder(token: str) -> str:
        """The operand a `{name}` token names; any other bare word is a register the block does
        not declare, which the transcription cannot bind."""
        match = re.fullmatch(r"\{(\w+)\}", token)
        if match is None:
            raise GenerationError(f"bare register {token}: name it through an operand")
        return match.group(1)

    zero_register = ZERO

    def emitted(self, node: Node) -> None:
        """A node writing the flags sets their form."""
        for name in node.writes():
            if name in (CARRY, FLAGS):
                self.flags = name

    def second(self, tokens: tuple[str, ...], text: str) -> tuple[Term, int | None]:
        """A second source operand: a register or immediate, optionally shifted left by an
        immediate (`b, lsl #k`): `(value, shift)`, the shift `None` unless it is a register's."""
        if len(tokens) == 1:
            return self.read(tokens[0]), None
        if len(tokens) == 3 and tokens[1] == "lsl":
            k = bits(tokens[2], 0, 63, "shift")
            if tokens[0].startswith("#"):
                return Lit(literal(tokens[0], k)), None
            return self.read(tokens[0]), k
        raise GenerationError(f"unsupported operand: {text}")

    def lift_instruction(self, ins: Instruction) -> None:
        try:
            op = Mnemonic(ins.op)
        except ValueError:
            raise GenerationError(f"unhandled instruction: {ins.text}") from None
        t, text = ins.operands, ins.text
        if len(t) not in ARITY[op]:
            expected = " or ".join(str(n) for n in ARITY[op])
            raise GenerationError(f"{op.value} expects {expected} operands, got {len(t)}: {text}")
        d = self.dest(t[0]) if op not in FLAG_ONLY else ""
        match op:
            case Mnemonic.MOV:
                self.emit(Mov(dest=d, a=self.read(t[1])))
            case Mnemonic.MUL:
                self.emit(MulLo(dest=d, a=self.read(t[1]), b=self.read(t[2])))
            case Mnemonic.UMULH:
                self.emit(MulHi(dest=d, function="umulh", a=self.read(t[1]), b=self.read(t[2])))
            case Mnemonic.MADD | Mnemonic.MSUB:
                args = (self.read(t[1]), self.read(t[2]), self.read(t[3]))
                self.emit(Word(dest=d, function=op.value, args=args))
            case Mnemonic.MNEG:
                self.emit(Word(dest=d, function="mneg", args=(self.read(t[1]), self.read(t[2]))))
            case Mnemonic.LSL:
                self.emit(Shl(dest=d, a=self.read(t[1]), k=bits(t[2], 1, 63, "shift")))
            case Mnemonic.LSR:
                self.emit(Shr(dest=d, a=self.read(t[1]), k=bits(t[2], 1, 63, "shift")))
            case Mnemonic.ASR:
                self.emit(Asr(dest=d, a=self.read(t[1]), k=bits(t[2], 1, 63, "shift")))
            case Mnemonic.SBFX:
                a = self.read(t[1])
                lsb = bits(t[2], 0, 63, "sbfx lsb")
                width = bits(t[3], 1, 64 - lsb, "sbfx width")
                self.emit(Sbfx(dest=d, a=a, lsb=lsb, width=width))
            case Mnemonic.EXTR:
                hi, lo = self.read(t[1]), self.read(t[2])
                k = bits(t[3], 0, 63, "extr lsb")
                self.emit(Word(dest=d, function="extr", args=(hi, lo, lit(k))))
            case Mnemonic.ADDS | Mnemonic.ADCS | Mnemonic.ADC:
                if d == ZERO and op is not Mnemonic.ADC:
                    # Only `subs xzr` (a comparison) has a node of its own, `SubFlag`.
                    raise GenerationError(f"{op.value} to xzr is unsupported: {text}")
                cin: Term = Lit("0") if op is Mnemonic.ADDS else self.flag(CARRY)
                a, b = self.read(t[1]), self.read(t[2])
                node = AddTrunc if op is Mnemonic.ADC else AddCarry
                self.emit(node(dest=d, a=a, b=b, cin=cin))
            case Mnemonic.SUBS | Mnemonic.SBCS:
                cin = Lit("1") if op is Mnemonic.SUBS else self.flag(CARRY)
                a, b = self.read(t[1]), self.read(t[2])
                if d == ZERO:
                    self.emit(SubFlag(a=a, b=b, cin=cin))
                else:
                    self.emit(SubBorrow(dest=d, a=a, b=b, cin=cin))
            case Mnemonic.ADD:
                a = self.read(t[1])
                b, shift = self.second(t[2:], text)
                if shift is None:
                    self.emit(Word(dest=d, function="addw", args=(a, b)))
                else:
                    self.emit(AddShifted(dest=d, a=a, b=b, k=shift))
            case Mnemonic.SUB:
                self.emit(Word(dest=d, function="subw", args=(self.read(t[1]), self.read(t[2]))))
            case Mnemonic.NEG:
                self.emit(Word(dest=d, function="negw", args=(self.read(t[1]),)))
            case Mnemonic.AND | Mnemonic.ORR | Mnemonic.EOR:
                a, b = self.read(t[1]), self.read(t[2])
                self.emit(Word(dest=d, function=f"{op.value}w", args=(a, b), bounds=(0, 1)))
            case Mnemonic.TST | Mnemonic.CMP:
                self.emit(SetFlags(test=op.value, args=(self.read(t[0]), self.read(t[1]))))
            case Mnemonic.CCMP:
                if self._condition(t[3], text) is not Condition.NE:
                    raise GenerationError(f"unexpected condition: {text}")
                bits(t[2], 0, 15, "ccmp nzcv")  # the mask of the four flags
                compared = (
                    self.flag(FLAGS),
                    self.read(t[0]),
                    self.read(t[1]),
                    Lit(literal(t[2])),
                )
                self.emit(SetFlags(test="ccmp", args=compared))
            case Mnemonic.CSEL:
                self._csel(d, t, text)
            case Mnemonic.CNEG:
                condition = self._condition(t[2], text)
                if condition not in (Condition.GE, Condition.MI):
                    raise GenerationError(f"unexpected condition: {text}")
                fl, a = self.flag(FLAGS), self.read(t[1])
                function = f"cneg{condition.value.capitalize()}"
                self.emit(Word(dest=d, function=function, args=(fl, a), bounds=(1,)))
            case Mnemonic.CSETM:
                if self._condition(t[1], text) is not Condition.MI:
                    raise GenerationError(f"unexpected condition: {text}")
                self.emit(Word(dest=d, function="csetmMi", args=(self.flag(FLAGS),)))

    @staticmethod
    def _condition(token: str, text: str) -> Condition:
        try:
            return Condition(token)
        except ValueError:
            raise GenerationError(f"unexpected condition: {text}") from None

    def _csel(self, d: str, t: tuple[str, ...], text: str) -> None:
        condition = self._condition(t[3], text)
        if condition in (Condition.LO, Condition.CC, Condition.CS, Condition.HS):
            c, a, b = self.flag(CARRY), self.read(t[1]), self.read(t[2])
            clear = condition in (Condition.LO, Condition.CC)
            self.emit(
                Select(
                    dest=d,
                    function="cselLo" if clear else "cselCs",
                    flag=c,
                    x=a,
                    y=b,
                    x_when=FlagState.CLEAR if clear else FlagState.SET,
                )
            )
        elif condition in (Condition.NE, Condition.GE):
            fl, a, b = self.flag(FLAGS), self.read(t[1]), self.read(t[2])
            function = f"csel{condition.value.capitalize()}"
            self.emit(Word(dest=d, function=function, args=(fl, a, b), bounds=(1, 2)))
        else:
            raise GenerationError(f"unexpected condition: {text}")


# -- the instruction set -------------------------------------------------------------------------


class AArch64(isa.Isa):
    """AArch64: the flags in their two forms, `xzr`, and every argument bound as `argument`. Its
    blocks touch no memory."""

    name = "AArch64"
    lifter = Lifter
    # Its own bindings, and `xzr`: an operand of that name would read as the zero register.
    reserved_names = frozenset({CARRY, FLAGS, ZERO}) | pair_names(AddCarry, SubBorrow)
    allowed_options = frozenset({"pure", "nomem", "nostack"})
    required_options = frozenset({"pure", "nomem", "nostack"})

    def state_load(self, kind: str) -> type[Load]:
        """The flags carried into a round have no bound: `LoadFlags`."""
        return LoadFlags if kind == "Flags" else Load


AARCH64 = AArch64()
