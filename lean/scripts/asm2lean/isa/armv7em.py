"""ARMv7E-M (Thumb-2): the instruction set of the Cortex-M4 and Cortex-M7 class backend.

Its words are 32 bits, where the shared nodes of `ir.py` and the `word_step` bound are 64-bit: so
every node here states its own 32-bit bound as a line after its step (`have b_x : x < 2^32`), and
the step itself records only the defining equation. The semantics are those of
`PastaCurves/ARMv7EM/Semantics.lean`: `addc32` and `subc32` return a (value, carry) pair, `umaal32`
a (low, high) pair, and `c` (and, in a loop, `z`) model the carry and zero flags.

The lifter accepts the small subset of Thumb-2 the backend uses, and checks what the transcription
relies on: every register is declared, read only once it holds a value, and written only if its
operand may be; the carry flag is read only while valid; immediates fit a word and shifts are
below 32; `umaal`'s destinations are distinct and do not overlap its inputs; memory is read only
at the initialized words of a declared pointer, at word offsets 0 to 28; and a loop is a label
`2:` before every instruction and one `bne 2b` reading a valid zero flag.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, ClassVar

from ..ir import (
    WIDTH32,
    App,
    Arg,
    CarryChain,
    Component,
    Const,
    Dead,
    Get,
    Infix,
    Ite,
    Let,
    Lit,
    Mov,
    Node,
    Reg,
    Scalar,
    Term,
    apply,
    dst,
    src,
)
from ..lean import wrap_words
from ..rust import Direction, GenerationError
from . import Instruction, Isa, Lifter

if TYPE_CHECKING:
    from ..block import Block
    from ..skeleton import Step

WORD_MOD = Const("wordMod")
MOD_LT = "Nat.mod_lt _ (Nat.two_pow_pos _)"


def bound32(step: Step, name: str, value: str, proof: str = MOD_LT) -> None:
    """`name := value`, with its bound below `2^32` by `proof`, on one line after the step."""
    step.eq(name, value)
    step.line(f"  have b_{name} : {name} < 2^32 := by rw [e_{name}]; exact {proof}")
    step.state.bnd[name] = f"b_{name}"


def unit_flag(step: Step, name: str) -> None:
    """`name` is a 0-or-1 flag, by the case split of its `if`."""
    step.line(f"  have b_{name} : {name} ≤ 1 := by rw [e_{name}]; split <;> decide")
    step.state.bnd[name] = f"b_{name}"
    step.state.unit_bound.add(name)


def head(step: Step) -> str:
    return "word_step" if step.state.program.clear_values else "word_step -clear"


# -- arguments and memory --------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, kw_only=True)
class Field(Node):
    """A register holding a field of a structure argument on entry, `dest := arg.field`, bounded
    by the hypothesis on that field, `h<arg>.<field>` (a loop round's carried registers)."""

    dest: str = dst()
    arg: str
    field: str

    def lets(self) -> list[Let]:
        return [self._let(self.dest, Get(Arg(self.arg), self.field))]

    def prove(self, step: Step) -> None:
        value = f"{self.arg}.{self.field}"
        bound32(step, step.name, value, f"h{value}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class WordLoad(Node):
    """A 32-bit word of a pointer operand's argument: the low (`% wordMod`) or high
    (`/ wordMod`) half of the 64-bit limb `limb`."""

    dest: str = dst()
    arg: str
    limb: str
    high: bool

    def lets(self) -> list[Let]:
        term = Infix("/" if self.high else "%", Get(Arg(self.arg), self.limb), WORD_MOD)
        return [self._let(self.dest, term, dead=Dead.COMMENT)]

    def prove(self, step: Step) -> None:
        from ..skeleton import projection

        nm = step.name
        program = step.state.program
        step.eq(nm, f"{self.arg}.{self.limb} {'/' if self.high else '%'} 2^32")
        if self.high:
            limbs = projection(program.arg_fields[self.arg], self.limb)
            proof = (
                f"Nat.div_lt_of_lt_mul (lt_of_lt_of_eq "
                f"{program.bound_hyp(self.arg)}.{limbs} (by norm_num))"
            )
        else:
            proof = MOD_LT
        step.line(f"  have b_{nm} : {nm} < 2^32 := by")
        step.line(f"    rw [e_{nm}]; exact {proof}")
        step.state.bnd[nm] = f"b_{nm}"


@dataclasses.dataclass(frozen=True, kw_only=True)
class Zero(Node):
    """A word of zero-initialized memory."""

    dest: str = dst()

    def lets(self) -> list[Let]:
        return [self._let(self.dest, Lit("0"), dead=Dead.COMMENT)]

    def prove(self, step: Step) -> None:
        nm = step.name
        step.eq(nm, "0")
        step.line(f"  have b_{nm} : {nm} < 2^32 := by rw [e_{nm}]; decide")
        step.state.bnd[nm] = f"b_{nm}"


# -- instructions ----------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, kw_only=True)
class Carry(CarryChain):
    """An instruction writing the carry flag, a link of the shared carry chain: each instruction
    is a subclass, fixing its function and its carry-in (`cin`, a literal or the flag `c`).

    The pair's name depends on the block (`temp`), a carry nothing reads is left out rather than
    obtained as a ghost, and the step is labelled by the value only when it names the pair, the
    value, and the carry."""

    ghost_carry = False
    carry_in: ClassVar[Term]

    temp: str

    def pair_name(self) -> str:
        return self.temp

    def label(self, names: list[str]) -> str:
        return names[1] if len(names) == 3 else names[0]


@dataclasses.dataclass(frozen=True, kw_only=True)
class Adds(Carry):
    function = "addc32"
    carry_in = Lit("0")


@dataclasses.dataclass(frozen=True, kw_only=True)
class Adcs(Carry):
    function = "addc32"
    carry_in = Reg("c")


@dataclasses.dataclass(frozen=True, kw_only=True)
class Subs(Carry):
    """`subs`; in a loop, it also sets the zero flag `z` that the loop's branch reads."""

    function = "subc32"
    subtraction = True
    carry_in = Lit("1")
    zero: bool = False

    def lets(self) -> list[Let]:
        lets = super().lets()
        if self.zero:
            lets.append(Let("z", Ite(Reg(self.dest), Lit("1"), Lit("0")), None, "  `-> zero"))
        return lets

    def prove_rest(self, step: Step, let: Let, name: str) -> None:
        if let.name != "z":
            super().prove_rest(step, let, name)
        # The zero flag's own step, after the pair's facts, on the value they named.
        ren = step.state.ren
        step.line(f"  -- {name}: {let.comment}")
        zero = f"(if {ren[self.dest]} = 0 then 1 else 0)"
        step.lines += wrap_words(head(step), [f"{name} := {zero}"], "")
        unit_flag(step, name)
        ren[let.name] = name


@dataclasses.dataclass(frozen=True, kw_only=True)
class Rsbs(Carry):
    """`rsbs dest, x, #0`: `0 - x`, whose value may go unread when only its borrow is wanted."""

    function = "subc32"
    subtraction = True
    carry_in = Lit("1")

    def components(self) -> list[Component]:
        _, carry = super().components()
        return [Component("1", self.dest, None, Dead.COMMENT), carry]

    def prove(self, step: Step) -> None:
        names, lets = step.names, step.lets
        if len(names) != 2 or lets[1].name != "c":
            super().prove(step)
            return
        # Only the borrow is read: its bound, without the value's.
        a, b, cin = step.r(self.a), step.r(self.b), step.r(self.cin)
        flag = names[1]
        step.eq(flag, f"({a} + 2^32 - {b} - (1 - {cin})) / 2^32")
        step.line(f"  have b_{flag} : {flag} ≤ 1 := by")
        step.line(f"    rw [e_{flag}]; exact subc32_carry_le_one {a} {b} {cin} {step.lt64(self.a)}")
        step.line(f"  clear e_{flag}")
        step.define(1)
        step.unit(flag)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Sbcs(Carry):
    function = "subc32"
    subtraction = True
    carry_in = Reg("c")


@dataclasses.dataclass(frozen=True, kw_only=True)
class CarryIn(Node):
    """An instruction reading the carry flag without writing it: `dest := (function left right
    c).1`. Each instruction is a subclass."""

    function: ClassVar[str]

    dest: str = dst()
    left: Term = src()
    right: Term = src()

    def lets(self) -> list[Let]:
        term = Get(App(self.function, (self.left, self.right, Reg("c"))), "1")
        return [self._let(self.dest, term)]


@dataclasses.dataclass(frozen=True, kw_only=True)
class Adc(CarryIn):
    """`adc`, whose carry-out the step names as a ghost bounded by `1`."""

    function = "addc32"

    def prove(self, step: Step) -> None:
        nm = step.name
        a, b, carry = step.r(self.left), step.r(self.right), step.r(Reg("c"))
        total = f"({a} + {b} + {carry})"
        bound32(step, nm, f"{total} % 2^32")
        step.line(f"  obtain ⟨k_{nm}, b_k_{nm}, l_{nm}⟩ :")
        step.line(f"      ∃ k, k ≤ 1 ∧ {nm} + 2^32 * k = {a} + {b} + {carry} :=")
        step.line(
            f"    ⟨{total} / 2^32, addc32_carry_le_one {a} {b} {carry} "
            f"{step.lt64(self.left)} {step.lt64(self.right)} {step.le1(Reg('c'))},"
        )
        step.line(f"      by rw [e_{nm}]; exact Nat.mod_add_div _ _⟩")
        step.line(f"  clear e_{nm}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class Sbc(CarryIn):
    function = "subc32"

    def prove(self, step: Step) -> None:
        a, b, carry = step.r(self.left), step.r(self.right), step.r(Reg("c"))
        bound32(step, step.name, f"({a} + 2^32 - {b} - (1 - {carry})) % 2^32")


@dataclasses.dataclass(frozen=True, kw_only=True)
class Binary(Node):
    """An instruction of two operands writing one register, without flags: each instruction is a
    subclass, giving its term and its step."""

    dest: str = dst()
    left: Term = src()
    right: Term = src()

    def term(self) -> Term:
        raise NotImplementedError

    def lets(self) -> list[Let]:
        return [self._let(self.dest, self.term())]


@dataclasses.dataclass(frozen=True, kw_only=True)
class Add(Binary):
    """`add`, modulo the word."""

    def term(self) -> Term:
        return Infix("%", Infix("+", self.left, self.right), WORD_MOD)

    def prove(self, step: Step) -> None:
        a, b = step.r(self.left), step.r(self.right)
        bound32(step, step.name, f"({a} + {b}) % 2^32")


@dataclasses.dataclass(frozen=True, kw_only=True)
class Rsb(Binary):
    """`rsb dest, left, right`: `right - left`, modulo the word."""

    def term(self) -> Term:
        return Infix("%", Infix("-", Infix("+", self.right, WORD_MOD), self.left), WORD_MOD)

    def prove(self, step: Step) -> None:
        a, b = step.r(self.left), step.r(self.right)
        bound32(step, step.name, f"({b} + 2^32 - {a}) % 2^32")


@dataclasses.dataclass(frozen=True, kw_only=True)
class Eor(Binary):
    def term(self) -> Term:
        return App("eor32", (self.left, self.right))

    def prove(self, step: Step) -> None:
        nm, a, b = step.name, step.r(self.left), step.r(self.right)
        bound32(step, nm, f"({a} ^^^ {b}) % 2^32")
        step.line(
            f"  have w_{nm} : ({a} ^^^ {b}) < 2^32 := Nat.xor_lt_two_pow "
            f"{step.lt64(self.left)} {step.lt64(self.right)}"
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class And(Binary):
    def term(self) -> Term:
        return App("andw", (self.left, self.right))

    def prove(self, step: Step) -> None:
        nm, a, b = step.name, step.r(self.left), step.r(self.right)
        step.eq(nm, f"andw {a} {b}")
        step.line(f"  have b_{nm} : {nm} < 2^32 := by")
        step.line(f"    rw [e_{nm}]; exact Nat.and_lt_two_pow {a} {step.lt64(self.right)}")
        step.state.bnd[nm] = f"b_{nm}"


@dataclasses.dataclass(frozen=True, kw_only=True)
class Shift(Node):
    """A shift right by an immediate below 32: each instruction is a subclass."""

    function: ClassVar[str]

    dest: str = dst()
    a: Term = src()
    k: int

    def lets(self) -> list[Let]:
        return [self._let(self.dest, apply(self.function, self.a, self.k))]


@dataclasses.dataclass(frozen=True, kw_only=True)
class Lsr(Shift):
    function = "lsr32"

    def prove(self, step: Step) -> None:
        nm, a = step.name, step.r(self.a)
        step.eq(nm, f"{a} / 2^{self.k}")
        step.line(f"  have b_{nm} : {nm} < 2^32 := by")
        step.line(
            f"    rw [e_{nm}]; exact lt_of_le_of_lt (Nat.div_le_self _ _) {step.lt64(self.a)}"
        )
        step.state.bnd[nm] = f"b_{nm}"


@dataclasses.dataclass(frozen=True, kw_only=True)
class Asr(Shift):
    function = "asr32"

    def prove(self, step: Step) -> None:
        nm, a = step.name, step.r(self.a)
        step.eq(nm, f"asr32 {a} {self.k}")
        step.line(f"  have b_{nm} : {nm} < 2^32 := by")
        if self.k == 1:
            step.line(f"    rw [e_{nm}]; exact asr32_one_lt {a} {step.lt64(self.a)}")
        else:
            step.line(f"    rw [e_{nm}, asr32]; split <;> norm_num [wordMod] at * <;> omega")
        step.state.bnd[nm] = f"b_{nm}"


@dataclasses.dataclass(frozen=True, kw_only=True)
class Orr(Node):
    """`orr` with its second operand shifted left: `orr32 left (lsl32 right k)`."""

    dest: str = dst()
    left: Term = src()
    right: Term = src()
    k: int

    def lets(self) -> list[Let]:
        return [self._let(self.dest, App("orr32", (self.left, apply("lsl32", self.right, self.k))))]

    def prove(self, step: Step) -> None:
        a, b = step.r(self.left), step.r(self.right)
        bound32(step, step.name, f"orr32 {a} (lsl32 {b} {self.k})")


@dataclasses.dataclass(frozen=True, kw_only=True)
class Umaal(Node):
    """`umaal low, high, a, b`: `m := umaal32 low high a b`, then `low := m.1`, `high := m.2`,
    with `low + 2^32 * high = a * b + low + high`."""

    low: str = dst()
    high: str = dst()
    a: Term = src()
    b: Term = src()

    def lets(self) -> list[Let]:
        m = Reg("m")
        return [
            self._let("m", App("umaal32", (Reg(self.low), Reg(self.high), self.a, self.b))),
            Let(self.low, Get(m, "1"), None, f"  `-> {self.source_name('low')}"),
            Let(self.high, Get(m, "2"), None, f"  `-> {self.source_name('high')}"),
        ]

    def label(self, names: list[str]) -> str:
        return names[1] if len(names) == 3 else names[0]

    def prove(self, step: Step) -> None:
        a, b = step.r(self.a), step.r(self.b)
        low, high = step.r(Reg(self.low)), step.r(Reg(self.high))
        total = f"({a} * {b} + {low} + {high})"
        low_name, high_name = step.names[1:3]
        bound32(step, low_name, f"{total} % 2^32")
        step.eq(high_name, f"{total} / 2^32")
        step.line(
            f"  have m_{high_name} : {a} * {b} ≤ (2^32 - 1) * (2^32 - 1) := "
            f"Nat.mul_le_mul (Nat.le_sub_one_of_lt {step.lt64(self.a)}) "
            f"(Nat.le_sub_one_of_lt {step.lt64(self.b)})"
        )
        step.line(f"  have t_{high_name} : {total} < 2^64 := by omega")
        step.line(f"  have b_{high_name} : {high_name} < 2^32 := by")
        step.line(
            f"    rw [e_{high_name}]; exact Nat.div_lt_of_lt_mul "
            f"(lt_of_lt_of_eq t_{high_name} (by norm_num))"
        )
        step.line(f"  have d_{high_name} : {low_name} + 2^32 * {high_name} = {total} := by")
        step.line(f"    rw [e_{low_name}, e_{high_name}]; exact Nat.mod_add_div _ _")
        step.line(f"  clear e_{low_name} e_{high_name}")
        step.state.ren[step.lets[1].name] = low_name
        step.state.ren[step.lets[2].name] = high_name
        step.state.bnd[low_name] = f"b_{low_name}"
        step.state.bnd[high_name] = f"b_{high_name}"


@dataclasses.dataclass(frozen=True, kw_only=True)
class Bne(Node):
    """`bne 2b`: whether the loop's backward branch is taken, `(if z = 0 then 1 else 0)`."""

    dest: str = dst()

    def lets(self) -> list[Let]:
        return [self._let(self.dest, Ite(Reg("z"), Lit("1"), Lit("0")))]

    def prove(self, step: Step) -> None:
        step.eq(step.name, f"(if {step.r(Reg('z'))} = 0 then 1 else 0)")
        unit_flag(step, step.name)


# -- the Rust around the blocks ----------------------------------------------------------------


@dataclasses.dataclass(frozen=True, kw_only=True)
class Join(Node):
    """Two returned 32-bit words joined into a 64-bit limb: `low + wordMod * high`."""

    dest: str = dst()
    low: Term = src()
    high: Term = src()

    def lets(self) -> list[Let]:
        return [self._let(self.dest, Infix("+", self.low, Infix("*", WORD_MOD, self.high)))]

    def prove(self, step: Step) -> None:
        nm = step.name
        step.eq(nm, f"{step.r(self.low)} + 2^32 * {step.r(self.high)}")
        step.line(f"  have b_{nm} : {nm} < 2^64 := by")
        step.line(f"    rw [e_{nm}]; clear * - {step.lt64(self.low)} {step.lt64(self.high)}; omega")
        step.state.bnd[nm] = f"b_{nm}"


@dataclasses.dataclass(frozen=True, kw_only=True)
class Bind(Node):
    """`dest := term`: a call of a factored helper, or a step of the Rust around a block. Its
    step records the equation only. A helper's outputs are bounded by the annotation applying its
    theorem (`ir.Project`); a Rust step's (`annotated`) by the annotation after it."""

    dest: str = dst()
    term: Term = src()
    annotated: bool = False

    def lets(self) -> list[Let]:
        return [self._let(self.dest, self.term)]

    def prove(self, step: Step) -> None:
        step.eq(step.name, step.r(self.term))
        if self.annotated:
            step.state.bnd[step.name] = f"b_{step.name}"


# The node of each mnemonic the lifter maps by table: those writing the carry flag (whose
# bindings form one step: the pair, then its components), those reading it without writing it,
# the other binary operations, and the shifts.
PAIRS: dict[str, type[Carry]] = {
    "adds": Adds,
    "adcs": Adcs,
    "subs": Subs,
    "rsbs": Rsbs,
    "sbcs": Sbcs,
}
FLAG_READING: dict[str, type[CarryIn]] = {"adc": Adc, "sbc": Sbc}
BINARY: dict[str, type[Binary]] = {"add": Add, "rsb": Rsb, "eor": Eor, "and": And}
SHIFTS: dict[str, type[Shift]] = {"lsr": Lsr, "asr": Asr}


# -- the lifter --------------------------------------------------------------------------------


class Armv7emLifter(Lifter):
    """Lifts the backend's Thumb-2 subset, one instruction at a time (see the module's
    docstring for the checks). A block reads memory through pointer operands bound with
    `bind_memory`, whose words are named `<pointer>_<offset>`; a store rebinds that name."""

    def __init__(
        self, directions: Mapping[str, Direction] | None = None, *, loop: bool = False
    ) -> None:
        super().__init__(directions)
        self.memory: dict[tuple[str, int], str] = {}  # (pointer, offset) -> its word's binding
        self.used_offsets: dict[str, list[int]] = {}  # pointer -> the offsets read or written
        self.carry_valid = False
        self.zero_valid = False
        self.loop = loop
        self.label_seen = False
        self.branch_seen = False

    @staticmethod
    def parse_instruction(text: str) -> Instruction:
        if text == "2:":
            return Instruction("2:", (), text)
        found = re.fullmatch(r"\s*([A-Za-z][A-Za-z0-9]*(?:\.w)?)\s*(.*?)\s*", text)
        if not found:
            raise GenerationError(f"cannot parse instruction {text!r}")
        op, rest = found.groups()
        operands: list[str] = []
        start = 0
        depth = 0
        for index, character in enumerate(rest):
            if character == "[":
                depth += 1
            elif character == "]":
                depth -= 1
                if depth < 0:
                    raise GenerationError(f"unmatched memory bracket in instruction {text!r}")
            elif character == "," and depth == 0:
                operands.append(rest[start:index].strip())
                start = index + 1
        if depth:
            raise GenerationError(f"unmatched memory bracket in instruction {text!r}")
        if rest:
            operands.append(rest[start:].strip())
        if any(not operand for operand in operands):
            raise GenerationError(f"empty operand in instruction {text!r}")
        return Instruction(op.lower().removesuffix(".w"), tuple(operands), text)

    # -- operands bound before the block

    @staticmethod
    def slot(pointer: str, offset: int) -> str:
        return f"{pointer}_{offset}"

    def bind_memory(self, pointer: str, argument: str | None) -> None:
        """The eight words of `pointer`'s argument, each half of a 64-bit limb of `argument`, or
        zero-initialized memory when `argument` is `None`."""
        self.check_declared(pointer)
        self.used_offsets[pointer] = []
        for offset in range(0, 32, 4):
            name = self.slot(pointer, offset)
            node: Node
            if argument is None:
                node = Zero(comment=f"zero-initialized {pointer}[{offset}]", dest=name)
            else:
                node = WordLoad(
                    comment=f"input {pointer}[{offset}]",
                    dest=name,
                    arg=argument,
                    limb=f"l{offset // 8}",
                    high=offset % 8 != 0,
                )
            self.nodes.append(node)
            self.known.add(name)
            self.memory[(pointer, offset)] = name

    def bind_argument(self, register: str, argument: str, field: str | None = None) -> None:
        """A register holding a word argument, or a field of one, on entry."""
        self.check_declared(register)
        if register in self.known:
            raise GenerationError(f"duplicate initial binding for {register}")
        comment = f"input {register}"
        node: Node
        if field is None:
            node = Scalar(comment=comment, dest=register, name=argument)
        else:
            node = Field(comment=comment, dest=register, arg=argument, field=field)
        self.nodes.append(node)
        self.known.add(register)

    # -- operands of an instruction

    @staticmethod
    def require_count(ins: Instruction, count: int) -> None:
        if len(ins.operands) != count:
            raise GenerationError(f"{ins.op} expects {count} operands, got {len(ins.operands)}")

    def address(self, token: str) -> tuple[str, int]:
        match = re.fullmatch(
            r"\[(?:\{([A-Za-z_]\w*)\}|(r(?:[0-9]|1[0-4]))),\s*#(0|[1-9][0-9]*)\]", token
        )
        if not match:
            raise GenerationError(f"unsupported memory operand {token}")
        named, fixed, offset_text = match.groups()
        pointer = named or fixed
        if pointer not in self.used_offsets:
            raise GenerationError(f"memory base {pointer} is not a declared pointer")
        offset = int(offset_text)
        if offset % 4 or offset < 0 or offset > 28:
            raise GenerationError(f"memory offset must be one of 0, 4, ..., 28: {offset}")
        self.used_offsets[pointer].append(offset)
        return pointer, offset

    def register(self, token: str) -> str:
        if "[" in token or "]" in token:
            raise GenerationError(f"memory operands are unsupported: {token}")
        match = re.fullmatch(r"\{([A-Za-z_]\w*)\}", token)
        if match:
            register = match.group(1)
        elif token == "lr" and "r14" in self.directions:
            register = "r14"
        elif re.fullmatch(r"r(?:[0-9]|1[0-4])", token) and token in self.directions:
            register = token
        else:
            raise GenerationError(f"unsupported register operand {token}")
        self.check_declared(register)
        return register

    def read(self, token: str) -> Reg:
        register = self.register(token)
        if register not in self.known:
            if self.directions[register] == "out":
                raise GenerationError(f"output-only register {register} read before being written")
            raise GenerationError(f"register {register} read before being written")
        return Reg(register)

    def write(self, token: str) -> str:
        register = self.register(token)
        if self.directions[register] == "in":
            raise GenerationError(f"input-only register {register} cannot be written")
        return register

    def value(self, token: str) -> Term:
        """A register or an immediate, which must fit a word."""
        if "[" in token or "]" in token:
            raise GenerationError(f"memory operands are unsupported: {token}")
        immediate = re.fullmatch(r"#(0|[1-9][0-9]*|0x[0-9a-fA-F]+)", token)
        if immediate:
            if int(immediate.group(1), 0) >= 2**32:
                raise GenerationError(f"immediate does not fit a 32-bit word: {token}")
            return Lit(immediate.group(1).lower())
        return self.read(token)

    @staticmethod
    def shift(token: str) -> int:
        match = re.fullmatch(r"#(0|[1-9][0-9]*)", token)
        if not match:
            raise GenerationError(f"unsupported shift amount {token}")
        amount = int(match.group(1))
        if amount >= 32:
            raise GenerationError(f"32-bit shift amount out of range: {token}")
        return amount

    def require_carry(self) -> None:
        if not self.carry_valid:
            raise GenerationError("carry flag read while invalid")

    # -- instructions

    def lift_instruction(self, ins: Instruction) -> None:
        op, operands = ins.op, ins.operands
        if op == "2:":
            if not self.loop or self.label_seen:
                raise GenerationError("unsupported or duplicate loop label")
            if any(node.pc is not None for node in self.nodes):
                raise GenerationError("loop label must precede every instruction")
            self.label_seen = True
        elif op == "bne":
            self.require_count(ins, 1)
            if not self.loop or operands[0] != "2b" or not self.label_seen:
                raise GenerationError("unsupported branch target or placement")
            if not self.zero_valid:
                raise GenerationError("zero flag read while invalid")
            if self.branch_seen:
                raise GenerationError("duplicate loop branch")
            self.branch_seen = True
            self.emit(Bne(dest="branch"))
        elif op == "ldr":
            self.require_count(ins, 2)
            pointer, offset = self.address(operands[1])
            slot = self.memory.get((pointer, offset))
            if slot is None:
                raise GenerationError(f"uninitialized memory read {pointer}[{offset}]")
            self.emit(Mov(dest=self.write(operands[0]), a=Reg(slot)))
        elif op == "str":
            self.require_count(ins, 2)
            stored = self.read(operands[0])
            pointer, offset = self.address(operands[1])
            slot = self.slot(pointer, offset)
            self.emit(Mov(dest=slot, a=stored))
            self.memory[(pointer, offset)] = slot
        elif op in PAIRS:
            self.lift_carry(ins)
        elif op in FLAG_READING:
            self.require_count(ins, 3)
            self.require_carry()
            left, right = self.value(operands[1]), self.value(operands[2])
            self.emit(FLAG_READING[op](dest=self.write(operands[0]), left=left, right=right))
        elif op == "mov":
            self.require_count(ins, 2)
            value = self.value(operands[1])
            self.emit(Mov(dest=self.write(operands[0]), a=value))
        elif op in BINARY:
            self.require_count(ins, 3)
            left, right = self.value(operands[1]), self.value(operands[2])
            self.emit(BINARY[op](dest=self.write(operands[0]), left=left, right=right))
        elif op in SHIFTS:
            self.require_count(ins, 3)
            a, k = self.read(operands[1]), self.shift(operands[2])
            self.emit(SHIFTS[op](dest=self.write(operands[0]), a=a, k=k))
        elif op == "orr":
            self.require_count(ins, 4)
            left, right = self.read(operands[1]), self.read(operands[2])
            shifted = re.fullmatch(r"lsl\s+#(0|[1-9][0-9]*)", operands[3])
            if not shifted:
                raise GenerationError(f"unsupported shifted operand {operands[3]}")
            amount = int(shifted.group(1))
            if amount >= 32:
                raise GenerationError(f"32-bit shift amount out of range: #{amount}")
            self.emit(Orr(dest=self.write(operands[0]), left=left, right=right, k=amount))
        elif op == "umaal":
            self.require_count(ins, 4)
            low, high = self.read(operands[0]), self.read(operands[1])
            a, b = self.read(operands[2]), self.read(operands[3])
            if low == high:
                raise GenerationError("umaal destinations must be distinct")
            overlap = {low.name, high.name} & {a.name, b.name}
            if overlap:
                raise GenerationError(
                    f"umaal destinations overlap input registers: {', '.join(sorted(overlap))}"
                )
            self.write(operands[0])
            self.write(operands[1])
            self.emit(Umaal(low=low.name, high=high.name, a=a, b=b))
        else:
            raise GenerationError(f"unsupported instruction {op}")

    def lift_carry(self, ins: Instruction) -> None:
        """`adds`, `adcs`, `subs`, `rsbs`, `sbcs`: an operation writing the carry flag, whose
        node fixes its carry-in."""
        node = PAIRS[ins.op]
        operands = ins.operands
        self.require_count(ins, 3)
        a: Term
        if node is Rsbs:
            b = self.value(operands[1])
            if operands[2] != "#0":
                raise GenerationError("rsbs supports only reverse subtraction from #0")
            a = Lit("0")
        else:
            if node.carry_in == Reg("c"):
                self.require_carry()
            a, b = self.value(operands[1]), self.value(operands[2])
        dest = self.write(operands[0])
        if node is Adcs:
            temp = "adcs_pair"
        elif node is Adds:
            temp = "s"
        else:
            temp = "sub" if "d" in self.directions else "d"
        if node is Subs and self.loop:
            self.emit(Subs(dest=dest, temp=temp, a=a, b=b, cin=Subs.carry_in, zero=True))
            self.zero_valid = True
        else:
            # Every flag-setting instruction writes Z, which only a loop's `subs` models: any other
            # leaves the modelled zero flag stale.
            self.emit(node(dest=dest, temp=temp, a=a, b=b, cin=node.carry_in))
            self.zero_valid = False
        self.carry_valid = True


class Armv7em(Isa):
    name = "ARMv7EM"
    width = WIDTH32
    lifter = Armv7emLifter
    reserved_names = frozenset({"c", "d", "m", "s"})
    # A register-only block is pure and touches neither memory nor the stack.
    allowed_options = frozenset({"pure", "nomem", "nostack"})
    required_options = frozenset({"pure", "nomem", "nostack"})

    def check_results(self, lifter: Lifter, results: Sequence[str], block: Block) -> None:
        """Each returned register must be written by an instruction: a primitive's outputs are
        its computation, and one passed through unchanged is a mistake in the block."""
        for register in results:
            if register not in lifter.written:
                raise GenerationError(
                    f"{block.rust_name}: output register {register} was never written"
                )


def lift_all(lifter: Armv7emLifter, instructions: Sequence[str]) -> None:
    """Lift a block's instructions in order, each at its index."""
    for pc, text in enumerate(instructions):
        lifter.lift(text, pc)
