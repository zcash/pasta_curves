"""The AArch64 instructions that the blocks use, lifted to the IR.

The flags take one of two forms, named after the binding that holds them: `c`, the carry of a
carry chain (`adds`/`adcs`/`subs`/`sbcs`, read by `adc`, `adcs`, `sbcs`, and `csel` on `lo`/`cs`),
or `fl`, the four flags of a test (`tst`, `cmp`, `ccmp`, read by `csel` on `ne`/`ge`, `cneg`,
`csetm`, `ccmp`). A condition reads the form that the last flag-setting instruction produced, and
reading the other is an error. `xzr` reads as `0`; a write to it is discarded. An immediate keeps
the radix of the source, so that a hex constant reads in the transcription as in the assembly.
After a subtraction the carry is set when no borrow occurred (`PastaCurves.AArch64.Semantics`).
"""

import dataclasses
import enum
import re

from . import rust
from .ir import (
    MOD_LT,
    Dead,
    FlagState,
    Load,
    Mov,
    MulHi,
    MulLo,
    Node,
    Scalar,
    Select,
    Shl,
    Shr,
    dst,
    src,
)

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


@dataclasses.dataclass(frozen=True)
class Instruction:
    """One instruction: its mnemonic as written, its operand tokens, and its normalized text."""

    op: str
    operands: tuple
    text: str


def parse_instruction(template):
    """An instruction of a template line: the placeholders become register names."""
    text = re.sub(r"\{(\w+)\}", r"\1", template)
    text = re.sub(r"\s+", " ", text.replace(", ", ",")).strip()
    match = re.match(r"(\S+)\s*(.*)", text)
    return Instruction(match.group(1), tuple(tokenize(match.group(2))), text)


def tokenize(rest):
    return re.findall(r"\[[^\]]*\]!?|[^,\s]+", rest)


def immediate(token):
    """An immediate operand such as `#62`, `#0x100`, or `8*1`, as an integer."""
    s = token.lstrip("#")
    if re.fullmatch(r"0x[0-9a-fA-F]+", s):
        return int(s, 16)
    m = re.fullmatch(r"(-?[0-9]+)(?:\*([0-9]+))?", s)
    if m is None:
        raise ValueError(f"unexpected immediate {token}")
    return int(m.group(1)) * int(m.group(2) or 1)


def literal(token, shift=0):
    """An immediate as a Lean literal, in the radix of the source; `shift` is the `lsl` amount of
    a shifted immediate."""
    value = immediate(token) << shift
    return hex(value) if token.lstrip("#").startswith("0x") else str(value)


def _bounded(step, value, proof):
    step.eq(step.name, value)
    step.bound(step.name, proof)


# -- the nodes of AArch64's semantics ---------------------------------------------------------


@dataclasses.dataclass(frozen=True, kw_only=True)
class CarryPair(Node):
    """`adds`/`adcs` (`addc`) and `subs`/`sbcs` (`subc`): the pair, its result in `dest`, and its
    carry in `c`. A carry that nothing reads is not bound, and the proof obtains it as a ghost."""

    function = ""
    dest: str = dst()
    a: str = src()
    b: str = src()
    cin: str = src()

    def lets(self):
        pair = self._let(
            PAIR, f"{self.function} {self.a} {self.b} {self.cin}", self.a, self.b, self.cin
        )
        result = self._let(self.dest, f"{PAIR}.1", PAIR, note=f"  `-> {self.source_name('dest')}")
        carry = self._let(CARRY, f"{PAIR}.2", PAIR, note="  `-> carry", dead=Dead.DROP)
        return [pair, carry] if self.dest == ZERO else [pair, result, carry]

    def label(self, names):
        return names[1]

    def formulas(self, a, b, cin, xn, cn, step):
        raise NotImplementedError

    def prove(self, step):
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

    def formulas(self, a, b, cin, xn, cn, step):
        return (
            f"({a} + {b} + {cin})",
            f"{xn} + 2^64 * {cn} = {a} + {b} + {cin}",
            "Nat.mod_add_div _ _",
            f"addc_carry_le_one {a} {b} {cin} {step.lt64(a)} {step.lt64(b)} {step.le1(cin)}",
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class SubBorrow(CarryPair):
    function = "subc"

    def formulas(self, a, b, cin, xn, cn, step):
        return (
            f"({a} + 2^64 - {b} - (1 - {cin}))",
            f"{xn} + 2^64 * {cn} + {b} + 1 = {a} + 2^64 + {cin}",
            f"subc_lin {a} {b} {cin} {step.lt64(b)} {step.le1(cin)}",
            f"subc_carry_le_one {a} {b} {cin} {step.lt64(a)}",
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class SubFlag(Node):
    """`subs xzr, a, b`: only the no-borrow flag of the difference."""

    a: str = src()
    b: str = src()
    cin: str = src()

    def lets(self):
        expr = f"(subc {self.a} {self.b} {self.cin}).2"
        return [self._let(CARRY, expr, self.a, self.b, self.cin, dead=Dead.DROP)]

    def prove(self, step):
        a, b, cin, nm = step.r(self.a), step.r(self.b), step.r(self.cin), step.name
        step.eq(nm, f"({a} + 2^64 - {b} - (1 - {cin})) / 2^64")
        step.line(
            f"  have b_{nm} : {nm} ≤ 1 := by rw [e_{nm}]; exact subc_carry_le_one {a} {b} {cin} {step.lt64(a)}"
        )
        step.line(
            f"  have l_{nm} : ({nm} = 1 ∧ {b} + 1 ≤ {a} + {cin}) ∨ ({nm} = 0 ∧ {a} + {cin} < {b} + 1) :="
        )
        step.line(
            f"    subc_carry_cases {a} {b} {cin} _ e_{nm} {step.lt64(a)} {step.lt64(b)} {step.le1(cin)}"
        )
        step.line(f"  clear e_{nm}")
        step.unit(nm)


@dataclasses.dataclass(frozen=True, kw_only=True)
class AddTrunc(Node):
    """`adc`: the sum's low word; the carry-out is obtained as a ghost."""

    dest: str = dst()
    a: str = src()
    b: str = src()
    cin: str = src()

    def lets(self):
        return [
            self._let(self.dest, f"(addc {self.a} {self.b} {self.cin}).1", self.a, self.b, self.cin)
        ]

    def prove(self, step):
        a, b, cin, nm = step.r(self.a), step.r(self.b), step.r(self.cin), step.name
        step.eq(nm, f"({a} + {b} + {cin}) % 2^64")
        step.bound(nm, MOD_LT)
        step.line(f"  obtain ⟨k_{nm}, b_k_{nm}, l_{nm}⟩ :")
        step.line(f"      ∃ k, k ≤ 1 ∧ {nm} + 2^64 * k = {a} + {b} + {cin} :=")
        step.line(
            f"    ⟨({a} + {b} + {cin}) / 2^64, addc_carry_le_one {a} {b} {cin} {step.lt64(a)} {step.lt64(b)} {step.le1(cin)},"
        )
        step.line(f"      by rw [e_{nm}]; exact Nat.mod_add_div _ _⟩")
        step.line(f"  clear e_{nm}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class Lsl(Shl):
    """`lsl` by an immediate: by 62 as every instruction set's `Shl`, else bounded on the spot."""

    def prove(self, step):
        if self.k == 62:
            return super().prove(step)
        _bounded(step, f"{step.r(self.a)} * 2^{self.k} % 2^64", MOD_LT)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Word(Node):
    """A word operation whose bound is the instance `<function>_lt` of a lemma of
    `AArch64/Spec/Words.lean`: `addw`, `subw`, `negw`, `madd`, `msub`, `mneg`, `extr`."""

    dest: str = dst()
    function: str
    args: tuple = src()

    def lets(self):
        return [
            self._let(self.dest, f"{self.function} {' '.join(map(str, self.args))}", *self.args)
        ]

    def prove(self, step):
        args = " ".join(str(step.r(a)) if isinstance(a, str) else str(a) for a in self.args)
        _bounded(step, f"{self.function} {args}", f"{self.function}_lt {args}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class AddShifted(Node):
    """`add d, a, b, lsl #k`: `addw a (lsl b k)`."""

    dest: str = dst()
    a: str = src()
    b: str = src()
    k: int

    def lets(self):
        return [self._let(self.dest, f"addw {self.a} (lsl {self.b} {self.k})", self.a, self.b)]

    def prove(self, step):
        a, b = step.r(self.a), step.r(self.b)
        expr = f"addw {a} (lsl {b} {self.k})"
        _bounded(step, expr, f"addw_lt {a} (lsl {b} {self.k})")


@dataclasses.dataclass(frozen=True, kw_only=True)
class Logic(Node):
    """`and`, `orr`, `eor`: `andw`, `orrw`, `eorw`, bounded by both operands' bounds."""

    dest: str = dst()
    function: str
    a: str = src()
    b: str = src()

    def lets(self):
        return [self._let(self.dest, f"{self.function} {self.a} {self.b}", self.a, self.b)]

    def prove(self, step):
        a, b, fn = step.r(self.a), step.r(self.b), self.function
        _bounded(step, f"{fn} {a} {b}", f"{fn}_lt {a} {b} {step.lt64(a)} {step.lt64(b)}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class Asr(Node):
    dest: str = dst()
    a: str = src()
    k: int

    def lets(self):
        return [self._let(self.dest, f"asr {self.a} {self.k}", self.a)]

    def prove(self, step):
        a = step.r(self.a)
        _bounded(step, f"asr {a} {self.k}", f"asr_lt {a} {self.k} {step.lt64(a)}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class Sbfx(Node):
    dest: str = dst()
    a: str = src()
    lsb: int
    width: int

    def lets(self):
        return [self._let(self.dest, f"sbfx {self.a} {self.lsb} {self.width}", self.a)]

    def prove(self, step):
        a, lsb, w = step.r(self.a), self.lsb, self.width
        _bounded(step, f"sbfx {a} {lsb} {w}", f"sbfx_lt {a} {lsb} {w} {step.lt64(a)}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class FlagSelect(Node):
    """`csel` on `ne` or `ge`: `cselNe`/`cselGe` on the four flags."""

    dest: str = dst()
    function: str
    flags: str = src()
    a: str = src()
    b: str = src()

    def lets(self):
        expr = f"{self.function} {self.flags} {self.a} {self.b}"
        return [self._let(self.dest, expr, self.flags, self.a, self.b)]

    def prove(self, step):
        fl, a, b, fn = step.r(self.flags), step.r(self.a), step.r(self.b), self.function
        _bounded(step, f"{fn} {fl} {a} {b}", f"{fn}_lt {fl} {a} {b} {step.lt64(a)} {step.lt64(b)}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class CondNeg(Node):
    """`cneg` on `ge` or `mi`: `cnegGe`/`cnegMi`."""

    dest: str = dst()
    function: str
    flags: str = src()
    a: str = src()

    def lets(self):
        return [self._let(self.dest, f"{self.function} {self.flags} {self.a}", self.flags, self.a)]

    def prove(self, step):
        fl, a, fn = step.r(self.flags), step.r(self.a), self.function
        _bounded(step, f"{fn} {fl} {a}", f"{fn}_lt {fl} {a} {step.lt64(a)}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class CondSetMask(Node):
    """`csetm d, mi`: `csetmMi`."""

    dest: str = dst()
    flags: str = src()

    def lets(self):
        return [self._let(self.dest, f"csetmMi {self.flags}", self.flags)]

    def prove(self, step):
        fl = step.r(self.flags)
        _bounded(step, f"csetmMi {fl}", f"csetmMi_lt {fl}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class SetFlags(Node):
    """A test that sets the four flags: `tst` (`tstFlags (andw a b)`), `cmp` (`cmpFlags a b`),
    and `ccmp` on `ne` (`ccmpNe fl a b nzcv`). Only its equation is recorded."""

    test: str
    args: tuple = src()

    def expr(self, rename=lambda r: r):
        args = [rename(a) if isinstance(a, str) else a for a in self.args]
        if self.test == "tst":
            return f"tstFlags (andw {args[0]} {args[1]})"
        if self.test == "cmp":
            return f"cmpFlags {args[0]} {args[1]}"
        return f"ccmpNe {' '.join(map(str, args))}"

    def lets(self):
        return [self._let(FLAGS, self.expr(), *self.args, dead=Dead.DROP)]

    def prove(self, step):
        step.eq(step.name, self.expr(step.r))


@dataclasses.dataclass(frozen=True, kw_only=True)
class LoadFlags(Load):
    """The flags carried into a round: only the equation, since flags have no bound."""

    def prove(self, step):
        step.eq(step.name, f"{self.arg}.{self.field}")


# -- lifting ------------------------------------------------------------------------------------


class Lifter:
    """Lifts a block's instructions to nodes, checking that every register is written before it
    is read, that only `out` and `inout` operands are written, and that a condition reads the
    flags in the form that the last flag-setting instruction produced."""

    def __init__(self, directions=None):
        self.directions = directions or {}
        self.nodes = []
        self.known = set()  # registers holding a value
        self.flags = None  # the form of the current flags: `c`, `fl`, or `None`
        self.pc, self.text = None, None

    def read(self, token):
        if token == ZERO:
            return "0"
        if token.startswith("#"):
            return literal(token)
        if token in (CARRY, FLAGS) and self.flags != token:
            raise ValueError(f"{token} read while the flags are not in that form")
        if token not in self.known:
            raise ValueError(f"{token} read before being written")
        return token

    def emit(self, node_type, **fields):
        node = node_type(comment=self.text, pc=self.pc, **fields)
        names = [let.name for let in node.lets()]
        if names == [ZERO]:
            return None  # a write to `xzr` is discarded
        self.nodes.append(node)
        for name in names:
            if name in (CARRY, FLAGS):
                self.flags = name
            self.known.add(name)
        return node

    def bind(self, node):
        """Record an argument binding."""
        self.nodes.append(node)
        self.known.update(let.name for let in node.lets())

    def second(self, tokens, text):
        """A second source operand: a register or immediate, optionally shifted left by an
        immediate (`b, lsl #k`): `(value, shift)`, the shift `None` unless it is a register's."""
        if len(tokens) == 1:
            return self.read(tokens[0]), None
        if len(tokens) == 3 and tokens[1] == "lsl":
            k = immediate(tokens[2])
            if tokens[0].startswith("#"):
                return literal(tokens[0], k), None
            return self.read(tokens[0]), k
        raise ValueError(f"unsupported operand: {text}")

    def lift(self, ins):
        try:
            op = Mnemonic(ins.op)
        except ValueError:
            raise ValueError(f"unhandled instruction: {ins.text}") from None
        t, text = ins.operands, ins.text
        if len(t) not in ARITY[op]:
            expected = " or ".join(str(n) for n in ARITY[op])
            raise ValueError(f"{op.value} expects {expected} operands, got {len(t)}: {text}")
        if op not in FLAG_ONLY and t[0] != ZERO:
            if self.directions.get(t[0]) == "in":
                raise ValueError(f"input-only register {t[0]} cannot be written: {text}")
            if t[0] not in self.directions:
                raise ValueError(f"undeclared destination {t[0]}: {text}")
        self.text = text
        match op:
            case Mnemonic.MOV:
                self.emit(Mov, dest=t[0], a=self.read(t[1]))
            case Mnemonic.MUL:
                self.emit(MulLo, dest=t[0], a=self.read(t[1]), b=self.read(t[2]))
            case Mnemonic.UMULH:
                self.emit(MulHi, dest=t[0], a=self.read(t[1]), b=self.read(t[2]))
            case Mnemonic.MADD | Mnemonic.MSUB:
                args = (self.read(t[1]), self.read(t[2]), self.read(t[3]))
                self.emit(Word, dest=t[0], function=op.value, args=args)
            case Mnemonic.MNEG:
                self.emit(Word, dest=t[0], function="mneg", args=(self.read(t[1]), self.read(t[2])))
            case Mnemonic.LSL:
                self.emit(Lsl, dest=t[0], a=self.read(t[1]), k=immediate(t[2]))
            case Mnemonic.LSR:
                self.emit(Shr, dest=t[0], a=self.read(t[1]), k=immediate(t[2]))
            case Mnemonic.ASR:
                self.emit(Asr, dest=t[0], a=self.read(t[1]), k=immediate(t[2]))
            case Mnemonic.SBFX:
                a = self.read(t[1])
                self.emit(Sbfx, dest=t[0], a=a, lsb=immediate(t[2]), width=immediate(t[3]))
            case Mnemonic.EXTR:
                hi, lo = self.read(t[1]), self.read(t[2])
                self.emit(Word, dest=t[0], function="extr", args=(hi, lo, immediate(t[3])))
            case Mnemonic.ADDS | Mnemonic.ADCS | Mnemonic.ADC:
                cin = "0" if op is Mnemonic.ADDS else self.read(CARRY)
                a, b = self.read(t[1]), self.read(t[2])
                node = AddTrunc if op is Mnemonic.ADC else AddCarry
                self.emit(node, dest=t[0], a=a, b=b, cin=cin)
            case Mnemonic.SUBS | Mnemonic.SBCS:
                cin = "1" if op is Mnemonic.SUBS else self.read(CARRY)
                a, b = self.read(t[1]), self.read(t[2])
                if t[0] == ZERO:
                    self.emit(SubFlag, a=a, b=b, cin=cin)
                else:
                    self.emit(SubBorrow, dest=t[0], a=a, b=b, cin=cin)
            case Mnemonic.ADD:
                a = self.read(t[1])
                b, shift = self.second(t[2:], text)
                if shift is None:
                    self.emit(Word, dest=t[0], function="addw", args=(a, b))
                else:
                    self.emit(AddShifted, dest=t[0], a=a, b=b, k=shift)
            case Mnemonic.SUB:
                self.emit(Word, dest=t[0], function="subw", args=(self.read(t[1]), self.read(t[2])))
            case Mnemonic.NEG:
                self.emit(Word, dest=t[0], function="negw", args=(self.read(t[1]),))
            case Mnemonic.AND | Mnemonic.ORR | Mnemonic.EOR:
                a, b = self.read(t[1]), self.read(t[2])
                self.emit(Logic, dest=t[0], function=f"{op.value}w", a=a, b=b)
            case Mnemonic.TST | Mnemonic.CMP:
                self.emit(SetFlags, test=op.value, args=(self.read(t[0]), self.read(t[1])))
            case Mnemonic.CCMP:
                if self._condition(t[3], text) is not Condition.NE:
                    raise ValueError(f"unexpected condition: {text}")
                args = (self.read(FLAGS), self.read(t[0]), self.read(t[1]), immediate(t[2]))
                self.emit(SetFlags, test="ccmp", args=args)
            case Mnemonic.CSEL:
                self._csel(t, text)
            case Mnemonic.CNEG:
                condition = self._condition(t[2], text)
                if condition not in (Condition.GE, Condition.MI):
                    raise ValueError(f"unexpected condition: {text}")
                fl, a = self.read(FLAGS), self.read(t[1])
                function = f"cneg{condition.value.capitalize()}"
                self.emit(CondNeg, dest=t[0], function=function, flags=fl, a=a)
            case Mnemonic.CSETM:
                if self._condition(t[1], text) is not Condition.MI:
                    raise ValueError(f"unexpected condition: {text}")
                self.emit(CondSetMask, dest=t[0], flags=self.read(FLAGS))

    @staticmethod
    def _condition(token, text):
        try:
            return Condition(token)
        except ValueError:
            raise ValueError(f"unexpected condition: {text}") from None

    def _csel(self, t, text):
        condition = self._condition(t[3], text)
        if condition in (Condition.LO, Condition.CC, Condition.CS, Condition.HS):
            c, a, b = self.read(CARRY), self.read(t[1]), self.read(t[2])
            clear = condition in (Condition.LO, Condition.CC)
            self.emit(
                Select,
                dest=t[0],
                function="cselLo" if clear else "cselCs",
                flag=c,
                x=a,
                y=b,
                x_when=FlagState.CLEAR if clear else FlagState.SET,
            )
        elif condition in (Condition.NE, Condition.GE):
            fl, a, b = self.read(FLAGS), self.read(t[1]), self.read(t[2])
            function = f"csel{condition.value.capitalize()}"
            self.emit(FlagSelect, dest=t[0], function=function, flags=fl, a=a, b=b)
        else:
            raise ValueError(f"unexpected condition: {text}")


# -- the instruction stream, for the leakage model -------------------------------------------------

# The instructions whose last operand is a condition code.
CONDITIONAL = {Mnemonic.CSEL, Mnemonic.CNEG, Mnemonic.CSETM, Mnemonic.CCMP}


def instruction_term(ins):
    """An instruction as a term of `PastaCurves.AArch64.Instr`, the syntax of the leakage model:
    its mnemonic and its operands as written. A register is named as in the block, `xzr` is the
    zero register, an immediate keeps the radix of the source, a shifted operand's `lsl #k` is one
    operand, and the last operand of a conditional instruction is its condition. A memory operand
    is an error: the blocks are `nomem`, and the leakage model has no address for one."""
    try:
        op = Mnemonic(ins.op)
    except ValueError:
        raise ValueError(f"unhandled instruction: {ins.text}") from None
    tokens = list(ins.operands)
    condition = None
    if op in CONDITIONAL:
        condition = Lifter._condition(tokens.pop(), ins.text)
    operands = []
    while tokens:
        token = tokens.pop(0)
        if token == "lsl" and tokens and tokens[0].startswith("#"):
            operands.append(f".lsl {literal(tokens.pop(0))}")
        elif token == ZERO:
            operands.append(".zero")
        elif token.startswith("#"):
            if immediate(token) < 0:
                raise ValueError(f"negative immediate: {ins.text}")
            operands.append(f".imm {literal(token)}")
        elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token):
            operands.append(f'.reg "{token}"')
        else:
            raise ValueError(f"unsupported operand {token}: {ins.text}")
    if condition is not None:
        operands.append(f".cond .{condition.value}")
    return f"⟨.{op.value}, [{', '.join(operands)}]⟩"


# -- a block ------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Block:
    """An inline `asm!` block to transcribe: the Rust function whose block to read, the Lean name,
    the docstring, the arguments as (name, kind) in signature order, and the result's kind, a
    structure whose fields are the result words in the source's result order."""

    rust_name: str
    lean_name: str
    doc: str
    args: tuple
    result: str = "Limbs"

    @property
    def arg_names(self):
        return [name for name, _ in self.args]


@dataclasses.dataclass(frozen=True)
class Parsed:
    """A block's instructions (and their macro origins), declarations, and outputs."""

    instructions: tuple
    origins: tuple
    declarations: tuple
    locals: dict
    outputs: dict
    returned: tuple


def parse_block(source, block, kinds, reserved_names):
    """Read `block`'s function from the Rust `source` through the front end."""
    fn = block.rust_name
    parsed = rust.parse_function(
        source,
        fn,
        block.arg_names,
        len(kinds[block.result]),
        reserved_names=reserved_names,
        allowed_options={"pure", "nomem", "nostack"},
        required_options={"pure", "nomem", "nostack"},
    )
    rust.declaration_directions(parsed, fn)
    return Parsed(
        tuple(parse_instruction(t) for t in parsed.instructions),
        parsed.origins,
        parsed.declarations,
        parsed.locals,
        rust.output_bindings(parsed, fn),
        rust.returned_registers(parsed, fn),
    )


def lift_block(parsed, block, kinds):
    """Bind the operands, lift the instructions, and return the lifter and the result registers.
    An `in` or `inout` operand reads an argument limb (`lhs[0]`, or a local copied from one before
    the block) or a word argument; the named `out` and the `inout` operands are the results."""
    name = block.lean_name
    fields = {arg: kinds[kind] for arg, kind in block.args}
    lifter = Lifter({d.name: d.kind for d in parsed.declarations})
    outs = []
    for d in parsed.declarations:
        if d.kind in ("in", "inout"):
            m = re.fullmatch(r"(\w+)\[(\d)\]", d.value)
            if m:
                arg, i = m.group(1), int(m.group(2))
            elif d.value in parsed.locals:
                arg, i = parsed.locals[d.value]
            elif d.value in fields and fields[d.value] is None:
                lifter.bind(Scalar(comment="argument", dest=d.name, name=d.value))
                arg = None
            else:
                raise ValueError(f"{name}: unexpected input operand {d.name} = {d.value}")
            if arg is not None:
                if arg not in fields:
                    raise ValueError(f"{name}: operand {d.name} reads {d.value}, not an argument")
                if fields[arg] is None or i >= len(fields[arg]):
                    raise ValueError(
                        f"{name}: operand {d.name} reads {d.value}, which {arg} does not have"
                    )
                lifter.bind(Load(comment="argument", dest=d.name, arg=arg, field=f"l{i}"))
            # An `inout` operand whose output is discarded (`=> _`) is an input only.
            if d.kind == "inout" and d.output != "_":
                if parsed.outputs.get(d.output) != d.name:
                    raise ValueError(
                        f"{name}: output binding {d.output} does not name operand {d.name}"
                    )
                outs.append((d.output, d.name))
        elif d.kind == "out":
            if d.value != "_":
                if parsed.outputs.get(d.value) != d.name:
                    raise ValueError(
                        f"{name}: output binding {d.value} does not name operand {d.name}"
                    )
                outs.append((d.value, d.name))
        else:
            raise ValueError(f"{name}: unsupported operand direction {d.kind}")
    ordered = tuple(reg for _, reg in sorted(outs))
    if ordered != parsed.returned:
        raise ValueError(
            f"{name}: source output order {parsed.returned} differs from the order of the output "
            f"names {ordered}"
        )
    for pc, ins in enumerate(parsed.instructions):
        lifter.pc = pc
        lifter.lift(ins)
    results = [lifter.read(reg) for reg in ordered]
    if len(results) != len(kinds[block.result]):
        raise ValueError(f"{name}: {len(results)} output operands")
    return lifter, results


@dataclasses.dataclass(frozen=True)
class Target:
    """What the AArch64 transcription needs to know about the crate: the fields of each argument
    or result kind (`None` for a word), the proof conventions, the blocks rerolled as a prologue
    loop (`loops`, by Lean name), and the macro arms rerolled as rounds (`macro_rounds`)."""

    kinds: dict
    conventions: object
    loops: dict = dataclasses.field(default_factory=dict)
    macro_rounds: dict = dataclasses.field(default_factory=dict)
    reserved_names: frozenset = frozenset({PAIR, CARRY, FLAGS})


def stream(source, block, target):
    """The instruction stream of one block, for the leakage model: its name, its docstring, and
    each instruction as (term, text), in order, macro invocations expanded."""
    parsed = parse_block(source, block, target.kinds, target.reserved_names)
    lift_block(parsed, block, target.kinds)  # the stream is of a block that lifts
    doc = (
        f"The instruction stream of the inline `asm!` block of `{block.rust_name}`, as written, "
        "its macro invocations expanded."
    )
    terms = [(instruction_term(ins), ins.text) for ins in parsed.instructions]
    return f"{block.lean_name}Program", doc, terms


def transcribe(source, block, target):
    """The programs of one block: the block itself, preceded by its round definitions when its
    instruction stream repeats (see `reroll.py`)."""
    from . import lean, reroll

    parsed = parse_block(source, block, target.kinds, target.reserved_names)
    lifter, results = lift_block(parsed, block, target.kinds)

    def make(name, doc, signature, nodes, results, **kw):
        return target.conventions.program("AArch64", name, doc, signature, nodes, results, **kw)

    arg_fields = {arg: target.kinds[kind] for arg, kind in block.args if target.kinds[kind]}
    signature = lean.signature(block.lean_name, block.args, block.result)
    loop = target.loops.get(block.lean_name)
    if loop is not None:
        limb_args = [arg for arg, kind in block.args if target.kinds[kind]]
        return loop.reroll(
            block.lean_name, block.doc, lifter, parsed.instructions, limb_args, results, make
        )
    if any(parsed.origins):
        rounds = reroll.MacroRounds(target.macro_rounds, LoadFlags)
        return rounds.reroll(block, lifter, parsed.origins, results, make, signature, arg_fields)
    return [
        make(block.lean_name, block.doc, signature, lifter.nodes, results, arg_fields=arg_fields)
    ]
