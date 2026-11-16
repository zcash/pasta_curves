"""The x86-64 instructions that the blocks use, lifted to the IR.

Two flags are modelled, CF (`cf`) and OF (`ofl`), as two independent carry chains: `add`, `adc`,
`adcx`, `sub`, `sbb`, and `neg` write CF, `adox` writes OF, `adcx` and `adox` each preserve the
other flag, and `mulx` and moves preserve both. A flag is *valid* while the last instruction that
touched it modelled it; `add`/`adc`/`sub`/`sbb`/`neg` leave OF unmodelled, and `imul` and the
shifts both flags (after a multi-bit shift OF is undefined). Reading an invalid flag is an error:
the transcription's earlier `cf` binding would still be in scope and give a stale value, which Lean
could not notice. The zeroing idiom `xor r, r` (also on the 32-bit subregister, which
zero-extends) writes zero and clears both flags. `bt` sets CF to the tested bit, which `cmovb`
reads.

ZF and SF are not kept as flags: the blocks read them only through `cmove` and `cmovs` right after
the `test` or `add` that sets them, so the condition is that instruction's word (`test`'s AND,
bound as `zs`, or `add`'s sum), and reading it is an error if any other instruction wrote the
flags, or the word was rewritten, in between.

The shifts and bitwise operations are the shared word operations: `sar` is `asr`, `shrd` is
`extr`, `and`, `or`, and `xor` are `andw`, `orrw`, and `eorw`, `bts` is an `orrw`, and `lea` is an
`addw`.

Operands are the placeholders of the template; the multiplicand of `mulx` is the fixed register
`rdx`, and one-operand `mul` multiplies the fixed `rax` into `rax` and `rdx`. A read-only pointer
operand is read only through `qword ptr [{p} + 8k]` for a word `k` of its argument; every other
memory operand, and any write to memory, is rejected. A negative immediate is sign-extended to the
word.
"""

from __future__ import annotations

import dataclasses
import enum
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, ClassVar

from .. import isa
from ..ir import (
    MOD_LT,
    Arg,
    Asr,
    Component,
    FlagState,
    Get,
    Let,
    Lit,
    Mov,
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
    dst,
    lit,
    pair_names,
    src,
)
from ..rust import Direction, GenerationError, ParsedFunction
from . import Instruction

if TYPE_CHECKING:
    from ..block import Block
    from ..skeleton import Step

CF, OF = "cf", "ofl"
RDX, RAX = "rdx", "rax"
ZS = "zs"  # the AND word of the last `test`, from which ZF and SF are read

# The architectural registers a block may name directly, each through exactly one fixed operand
# (`out("rdx") _`, `out("rax") _`): RDX for MULX's implicit multiplicand, and RAX and RDX for
# one-operand MUL.
FIXED_REGISTERS = (RDX, RAX)


class Mnemonic(str, enum.Enum):
    MOV = "mov"
    MOVABS = "movabs"
    MULX = "mulx"
    MUL = "mul"
    IMUL = "imul"
    SHL = "shl"
    SHR = "shr"
    SAR = "sar"
    SHRD = "shrd"
    ADD = "add"
    ADC = "adc"
    ADCX = "adcx"
    ADOX = "adox"
    SUB = "sub"
    SBB = "sbb"
    NEG = "neg"
    CMOVNC = "cmovnc"
    CMOVB = "cmovb"
    CMOVE = "cmove"
    CMOVS = "cmovs"
    XOR = "xor"
    AND = "and"
    OR = "or"
    BT = "bt"
    BTS = "bts"
    TEST = "test"
    LEA = "lea"


# -- the nodes of x86-64's semantics ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True, kw_only=True)
class FlagPair(PairOp):
    """An instruction that writes a register and a flag through a pair: the result in `dest`,
    the flag in `flag`."""

    dest: str = dst()
    flag: str = dst()
    a: Term = src()
    b: Term = src()
    carry: Term = src()

    def pair_args(self) -> tuple[Operand, ...]:
        return (self.a, self.b, self.carry)

    def components(self) -> list[Component]:
        return [
            Component("1", self.dest, f"  `-> {self.source_name('dest')}"),
            Component("2", self.flag, f"  `-> {self.source_name('flag')}"),
        ]

    def label(self, names: list[str]) -> str:
        return names[1]

    def formulas(
        self, a: str, b: str, carry: str, value: str, flag: str, step: Step
    ) -> tuple[str, str, str, str]:
        raise NotImplementedError

    def prove(self, step: Step) -> None:
        a, b, carry = step.r(self.a), step.r(self.b), step.r(self.carry)
        value, flag = step.names[1:3]
        step.eq(value, f"({self.function} {a} {b} {carry}).1")
        step.eq(flag, f"({self.function} {a} {b} {carry}).2")
        linear, linear_proof, value_proof, flag_proof = self.formulas(
            a, b, carry, value, flag, step
        )
        step.bound(value, value_proof)
        step.line(f"  have l_{value} : {linear} := by")
        step.line(f"    rw [e_{value}, e_{flag}]; exact {linear_proof}")
        step.line(f"  have b_{flag} : {flag} ≤ 1 := by rw [e_{flag}]; exact {flag_proof}")
        step.line(f"  clear e_{value} e_{flag}")
        step.define(2)
        step.define(1)
        step.unit(flag)


@dataclasses.dataclass(frozen=True, kw_only=True)
class AddWithFlag(FlagPair):
    """`add`, `adc`, `adcx` (CF), and `adox` (OF): `addc a b carry`."""

    pair = "s"
    function = "addc"

    def formulas(
        self, a: str, b: str, carry: str, value: str, flag: str, step: Step
    ) -> tuple[str, str, str, str]:
        return (
            f"{value} + 2^64 * {flag} = {a} + {b} + {carry}",
            f"addc_lin {a} {b} {carry}",
            f"addc_value_lt {a} {b} {carry}",
            f"addc_carry_le_one {a} {b} {carry} {step.lt64(self.a)} {step.lt64(self.b)} {step.le1(self.carry)}",
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class SubWithBorrow(FlagPair):
    """`sub` and `sbb`: `sbb a b borrow`, whose flag is CF, set on borrow."""

    pair = "d"
    function = "sbb"

    def formulas(
        self, a: str, b: str, carry: str, value: str, flag: str, step: Step
    ) -> tuple[str, str, str, str]:
        return (
            f"{value} + {b} + {carry} = {a} + 2^64 * {flag}",
            f"sbb_lin {a} {b} {carry} {step.lt64(self.a)} {step.lt64(self.b)} {step.le1(self.carry)}",
            f"sbb_value_lt {a} {b} {carry}",
            f"sbb_borrow_le_one {a} {b} {carry}",
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class WideProduct(PairOp):
    """The two words of `a * b` as a pair, the high word first when `high_first`."""

    pair = "m"
    high_first: ClassVar[bool] = True
    high: str = dst()
    low: str = dst()
    a: Term = src()
    b: Term = src()

    def pair_args(self) -> tuple[Operand, ...]:
        return (self.a, self.b)

    def order(self) -> tuple[str, str]:
        """The two words, in the order of the pair."""
        return (self.high, self.low) if self.high_first else (self.low, self.high)

    def components(self) -> list[Component]:
        field = {self.high: "high", self.low: "low"}
        return [
            Component(str(index), register, f"  `-> {self.source_name(field[register])}")
            for index, register in enumerate(self.order(), start=1)
        ]

    def prove(self, step: Step) -> None:
        a, b, fn = step.r(self.a), step.r(self.b), self.function
        # Taken before either word is defined, since one may overwrite an operand.
        product = f"Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' {step.lt64(self.a)} {step.lt64(self.b)})"
        names = step.names[1:3]
        high, low = names if self.high_first else names[::-1]
        for index, name in enumerate(names, start=1):
            step.eq(name, f"({fn} {a} {b}).{index}")
            step.bound(name, product if name == high else MOD_LT)
            step.define(index)
        step.line(f"  have d_{high} : {low} + 2^64 * {high} = {a} * {b} := by")
        step.line(f"    rw [e_{low}, e_{high}]; exact Nat.mod_add_div _ _")


@dataclasses.dataclass(frozen=True, kw_only=True)
class Mulx(WideProduct):
    """`mulx hi, lo, src`: the two words of `rdx * src`, high first."""

    function = "mulx"


@dataclasses.dataclass(frozen=True, kw_only=True)
class Mulq(WideProduct):
    """One-operand `mul src`: the two words of `rax * src`, low into `rax` first, high into
    `rdx`."""

    function = "mulq"
    high_first = False


@dataclasses.dataclass(frozen=True, kw_only=True)
class BitTest(Node):
    """`bt word, bit`: CF is the bit, `bt word bit`."""

    word: Term = src()
    bit: int

    def lets(self) -> list[Let]:
        return [self._let(CF, apply("bt", self.word, self.bit))]

    def prove(self, step: Step) -> None:
        nm = step.name
        step.eq(nm, f"bt {step.r(self.word)} {self.bit}")
        step.line(f"  have b_{nm} : {nm} ≤ 1 := by")
        step.line(f"    rw [e_{nm}]; exact Nat.le_of_lt_succ (Nat.mod_lt _ (by decide))")
        step.unit(nm)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Neg(PairOp):
    """`neg`: the low word of `-a`, and CF, set exactly when `a` is nonzero."""

    pair = "n"
    function = "neg"
    dest: str = dst()
    a: Term = src()

    def pair_args(self) -> tuple[Operand, ...]:
        return (self.a,)

    def components(self) -> list[Component]:
        return [
            Component("1", self.dest, f"  `-> {self.source_name('dest')}"),
            Component("2", CF, f"  `-> {CF}"),
        ]

    def prove(self, step: Step) -> None:
        a = step.r(self.a)
        value, flag = step.names[1:3]
        step.eq(value, f"(neg {a}).1")
        step.bound(value, f"neg_value_lt {a}")
        step.define(1)
        step.eq(flag, f"(neg {a}).2")
        step.line(f"  have b_{flag} : {flag} ≤ 1 := by")
        step.line(f"    rw [e_{flag}]; simp only [neg]; split <;> omega")
        step.define(2)
        step.unit(flag)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ZeroIdiom(Node):
    """`xor r, r`: zero, with CF and OF cleared."""

    dest: str = dst()

    def lets(self) -> list[Let]:
        return [
            self._let(self.dest, Lit("0")),
            self._let(CF, Lit("0"), note=f"  `-> {CF}"),
            self._let(OF, Lit("0"), note=f"  `-> {OF}"),
        ]

    def prove(self, step: Step) -> None:
        data, cf, of = step.names
        step.eq(data, "0")
        step.bound(data, "(by decide)")
        step.define(0)
        for index, flag in ((1, cf), (2, of)):
            step.eq(flag, "0")
            step.line(f"  have b_{flag} : {flag} ≤ 1 := by rw [e_{flag}]; decide")
            step.define(index)
            step.unit(flag)


# -- lifting ------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RegisterToken:
    name: str
    modifier: str | None = None


class Lifter(isa.Lifter):
    """Validates a block's x86-64 instructions and lifts them to nodes."""

    def __init__(self, directions: Mapping[str, Direction] | None = None) -> None:
        super().__init__(directions)
        self.cf_valid = False
        self.of_valid = False
        # Where ZF and SF come from, when valid: `zs` after TEST, whose AND word is bound as
        # `zs`, or the destination of an ADD, whose word must not be rewritten before the flags
        # are read. None when the last flag writer left them undefined or unmodelled.
        self.zs_source: str | None = None

    # -- operands and their checks

    @staticmethod
    def parse_register(token: str) -> RegisterToken:
        token = token.strip()
        if token.startswith(("qword ptr", "[")):
            raise GenerationError(f"memory write is unsupported: {token}")
        if token in FIXED_REGISTERS:
            return RegisterToken(token)
        match = re.fullmatch(r"\{([A-Za-z_]\w*)(?::([A-Za-z_]\w*))?\}", token)
        if not match:
            raise GenerationError(f"unsupported register operand {token}")
        return RegisterToken(match.group(1), match.group(2))

    def read_register(self, token: str) -> str:
        parsed = self.parse_register(token)
        self.check_declared(parsed.name)
        if parsed.modifier:
            raise GenerationError(f"unsupported register modifier :{parsed.modifier} in {token}")
        if parsed.name in self.pointers:
            raise GenerationError(f"pointer register {parsed.name} used outside an address")
        if parsed.name not in self.known:
            self.check_readable(parsed.name)
        return self.require_written(parsed.name)

    def write_register(self, token: str, modifiers: frozenset[str] = frozenset()) -> str:
        """A destination register, which may carry one of the operand `modifiers` (the zeroing
        idiom's `:e`, whose 32-bit write zero-extends)."""
        parsed = self.parse_register(token)
        self.check_writable(parsed.name)
        if parsed.modifier is not None and parsed.modifier not in modifiers:
            raise GenerationError(f"unsupported register modifier :{parsed.modifier} in {token}")
        if parsed.name in self.pointers:
            raise GenerationError(f"cannot overwrite pointer operand {parsed.name}")
        if self.zs_source == parsed.name:
            self.zs_source = None
        return parsed.name

    def source(self, token: str) -> Reg:
        """A register operand that the instruction reads."""
        return Reg(self.read_register(token))

    def read_value(self, token: str) -> Term:
        """A source operand: an immediate (decimal or hex, transcribed in decimal), a word of a
        read-only pointer argument (its field), or a register."""
        token = token.strip()
        if re.fullmatch(r"(?:0|[1-9][0-9]*|0x[0-9A-Fa-f]+)", token):
            return lit(int(token, 0))
        if re.fullmatch(r"-[1-9][0-9]*", token):
            # A negative immediate is sign-extended to the word: its two's complement.
            value = int(token)
            if value < -(2**31):
                raise GenerationError(f"immediate {token} does not fit a sign-extended imm32")
            return lit(value % 2**64)
        memory = re.fullmatch(
            r"qword\s+ptr\s*\[\s*\{([A-Za-z_]\w*)\}\s*(?:\+\s*([0-9]+))?\s*\]", token
        )
        if memory:
            base, displacement_text = memory.groups()
            self.check_declared(base)
            if base not in self.pointers:
                raise GenerationError(f"address base {base} is not a read-only pointer operand")
            displacement = int(displacement_text or "0")
            fields = self.pointer_fields[base]
            if displacement % 8 or not 0 <= displacement < 8 * len(fields):
                raise GenerationError(f"unsupported memory offset {displacement}")
            self.used_memory = True
            return Get(Arg(self.pointers[base]), fields[displacement // 8])
        return self.source(token)

    @staticmethod
    def parse_instruction(text: str) -> Instruction:
        """An instruction of a template line, its operands split at commas, its text as
        written."""
        found = re.fullmatch(r"\s*([A-Za-z][A-Za-z0-9]*)\s*(.*?)\s*", text)
        if not found:
            raise GenerationError(f"cannot parse instruction {text!r}")
        op, rest = found.groups()
        operands = [part.strip() for part in rest.split(",")] if rest else []
        if any(not operand for operand in operands):
            raise GenerationError(f"empty operand in instruction {text!r}")
        return Instruction(op.lower(), tuple(operands), text)

    @staticmethod
    def require_count(op: str, operands: Sequence[str], count: int) -> None:
        if len(operands) != count:
            raise GenerationError(f"{op} expects {count} operands, got {len(operands)}")

    @staticmethod
    def immediate(token: str) -> int:
        if not re.fullmatch(r"(?:0|[1-9][0-9]*)", token.strip()):
            raise GenerationError(f"unsupported immediate {token}")
        return int(token)

    def require(self, flag: str, text: str) -> str:
        valid = self.cf_valid if flag == CF else self.of_valid
        if not valid:
            raise GenerationError(f"{'CF' if flag == CF else 'OF'} read while invalid: {text}")
        return flag

    def invalidate_flags(self) -> None:
        """The instruction writes CF, OF, ZF, and SF in ways the model does not track: none of
        them may be read until another instruction defines it."""
        self.cf_valid = self.of_valid = False
        self.zs_source = None

    def condition(self, text: str) -> str:
        """The word from which ZF and SF are read: `zs`, or the register an ADD wrote."""
        if self.zs_source is None:
            raise GenerationError(f"ZF/SF read while invalid: {text}")
        return self.zs_source

    def shift_amount(self, token: str) -> int:
        amount = self.immediate(token)
        if not 1 <= amount < 64:
            raise GenerationError("only unmasked nonzero 64-bit shift counts are supported")
        return amount

    def bit_index(self, token: str) -> int:
        bit = self.immediate(token)
        if not 0 <= bit < 64:
            raise GenerationError("only bit indices below 64 are supported")
        return bit

    # -- instructions

    def lift_instruction(self, ins: Instruction) -> None:
        try:
            op = Mnemonic(ins.op)
        except ValueError:
            raise GenerationError(f"unsupported instruction {ins.op}: {ins.text}") from None
        self._lift(op, list(ins.operands), ins.text)

    def _lift(self, op: Mnemonic, operands: list[str], text: str) -> None:
        match op:
            case Mnemonic.MOV | Mnemonic.MOVABS:
                self.require_count(op.value, operands, 2)
                a = self.read_value(operands[1])
                self.emit(Mov(dest=self.write_register(operands[0]), a=a))
            case Mnemonic.MULX:
                self.require_count(op.value, operands, 3)
                rdx = self.source(RDX)
                b = self.read_value(operands[2])
                high = self.write_register(operands[0])
                low = self.write_register(operands[1])
                if high == low:
                    raise GenerationError("mulx destinations must be distinct")
                self.emit(Mulx(high=high, low=low, a=rdx, b=b))
            case Mnemonic.MUL:
                self.require_count(op.value, operands, 1)
                rax = self.source(RAX)
                b = self.read_value(operands[0])
                self.emit(
                    Mulq(low=self.write_register(RAX), high=self.write_register(RDX), a=rax, b=b)
                )
                self.invalidate_flags()
            case Mnemonic.IMUL:
                self.require_count(op.value, operands, 2)
                old = self.source(operands[0])
                b = self.read_value(operands[1])
                self.emit(MulLo(dest=self.write_register(operands[0]), a=old, b=b))
                self.invalidate_flags()
            case Mnemonic.SHL | Mnemonic.SHR | Mnemonic.SAR:
                self.require_count(op.value, operands, 2)
                old = self.source(operands[0])
                amount = self.shift_amount(operands[1])
                node = {Mnemonic.SHL: Shl, Mnemonic.SHR: Shr, Mnemonic.SAR: Asr}[op]
                self.emit(node(dest=self.write_register(operands[0]), a=old, k=amount))
                self.invalidate_flags()
            case Mnemonic.SHRD:
                self.require_count(op.value, operands, 3)
                old = self.source(operands[0])
                upper = self.source(operands[1])
                amount = self.shift_amount(operands[2])
                dest = self.write_register(operands[0])
                self.emit(Word(dest=dest, function="extr", args=(upper, old, lit(amount))))
                self.invalidate_flags()
            case Mnemonic.ADD | Mnemonic.ADC | Mnemonic.ADCX | Mnemonic.ADOX:
                self.require_count(op.value, operands, 2)
                old = self.source(operands[0])
                b = self.read_value(operands[1])
                carry: Term = Lit("0")
                if op in (Mnemonic.ADC, Mnemonic.ADCX):
                    carry = Reg(self.require(CF, text))
                elif op is Mnemonic.ADOX:
                    carry = Reg(self.require(OF, text))
                flag = OF if op is Mnemonic.ADOX else CF
                dest = self.write_register(operands[0])
                self.emit(AddWithFlag(dest=dest, flag=flag, a=old, b=b, carry=carry))
                if op is Mnemonic.ADOX:
                    self.of_valid = True
                else:
                    self.cf_valid = True
                    if op in (Mnemonic.ADD, Mnemonic.ADC):
                        self.of_valid = False
                # ADD sets ZF and SF from its result word; the carry-chain forms are not read for
                # them.
                self.zs_source = dest if op is Mnemonic.ADD else None
            case Mnemonic.SUB | Mnemonic.SBB:
                self.require_count(op.value, operands, 2)
                old = self.source(operands[0])
                b = self.read_value(operands[1])
                borrow: Term = Reg(self.require(CF, text)) if op is Mnemonic.SBB else Lit("0")
                dest = self.write_register(operands[0])
                self.emit(SubWithBorrow(dest=dest, flag=CF, a=old, b=b, carry=borrow))
                self.cf_valid, self.of_valid, self.zs_source = True, False, None
            case Mnemonic.NEG:
                self.require_count(op.value, operands, 1)
                old = self.source(operands[0])
                self.emit(Neg(dest=self.write_register(operands[0]), a=old))
                self.cf_valid, self.of_valid, self.zs_source = True, False, None
            case Mnemonic.CMOVNC:
                self.require_count(op.value, operands, 2)
                self.require(CF, text)
                old = self.source(operands[0])
                b = self.read_value(operands[1])
                self.emit(
                    Select(
                        dest=self.write_register(operands[0]),
                        function="cmovnc",
                        flag=Reg(CF),
                        x=old,
                        y=b,
                        x_when=FlagState.SET,
                    )
                )
            case Mnemonic.CMOVB | Mnemonic.CMOVE | Mnemonic.CMOVS:
                self.require_count(op.value, operands, 2)
                if op is Mnemonic.CMOVB:
                    condition = self.require(CF, text)
                else:
                    condition = self.condition(text)
                old = self.source(operands[0])
                b = self.read_value(operands[1])
                function = {Mnemonic.CMOVB: "cmovb", Mnemonic.CMOVE: "cmovz"}.get(op, "cmovs")
                dest = self.write_register(operands[0])
                args = (Reg(condition), old, b)
                self.emit(Word(dest=dest, function=function, args=args, bounds=(1, 2)))
            case Mnemonic.XOR if not self.is_self_xor(operands):
                self.bitwise(op, operands)
            case Mnemonic.XOR:
                self.require_count(op.value, operands, 2)
                self.emit(ZeroIdiom(dest=self.write_register(operands[0], frozenset({"e"}))))
                self.cf_valid = self.of_valid = True
                self.zs_source = None
            case Mnemonic.AND | Mnemonic.OR:
                self.bitwise(op, operands)
            case Mnemonic.BTS:
                self.require_count(op.value, operands, 2)
                old = self.source(operands[0])
                bit = self.bit_index(operands[1])
                dest = self.write_register(operands[0])
                self.emit(Word(dest=dest, function="orrw", args=(old, lit(2**bit)), bounds=(0, 1)))
                self.invalidate_flags()
            case Mnemonic.BT:
                self.require_count(op.value, operands, 2)
                word = self.source(operands[0])
                self.emit(BitTest(word=word, bit=self.bit_index(operands[1])))
                self.invalidate_flags()
                self.cf_valid = True
            case Mnemonic.TEST:
                self.require_count(op.value, operands, 2)
                tested = self.source(operands[0])
                mask = self.read_value(operands[1])
                if isinstance(mask, Get):
                    raise GenerationError("TEST with a memory operand is unsupported")
                self.emit(Word(dest=ZS, function="andw", args=(tested, mask), bounds=(0, 1)))
                self.invalidate_flags()
                self.zs_source = ZS
            case Mnemonic.LEA:
                self.require_count(op.value, operands, 2)
                address = re.fullmatch(
                    r"\[\s*(\{[A-Za-z_]\w*\})\s*\+\s*(\{[A-Za-z_]\w*\}|0x[0-9A-Fa-f]+|[0-9]+)\s*\]",
                    operands[1].strip(),
                )
                if not address:
                    raise GenerationError(f"unsupported LEA address {operands[1]}")
                base = self.source(address.group(1))
                offset = self.read_value(address.group(2))
                if isinstance(offset, Lit) and int(offset.text) >= 2**31:
                    raise GenerationError(f"LEA displacement {offset.text} does not fit a disp32")
                dest = self.write_register(operands[0])
                self.emit(Word(dest=dest, function="addw", args=(base, offset)))

    @staticmethod
    def is_self_xor(operands: Sequence[str]) -> bool:
        """Whether an XOR is the zeroing idiom, both operands one register in one width."""
        return len(operands) == 2 and operands[0].strip() == operands[1].strip()

    def bitwise(self, op: Mnemonic, operands: Sequence[str]) -> None:
        """AND, OR, and the general XOR, on a register and a register or an immediate."""
        self.require_count(op.value, operands, 2)
        old = self.source(operands[0])
        b = self.read_value(operands[1])
        dest = self.write_register(operands[0])
        function = {Mnemonic.AND: "andw", Mnemonic.OR: "orrw", Mnemonic.XOR: "eorw"}[op]
        self.emit(Word(dest=dest, function=function, args=(old, b), bounds=(0, 1)))
        self.invalidate_flags()


# -- the instruction set -------------------------------------------------------------------------


class X86_64(isa.Isa):
    """x86-64: CF and OF as two carry chains, the fixed RDX and RAX, and read-only pointer
    operands."""

    name = "X86_64"
    lifter = Lifter
    # The fixed registers, the flags and `zs`, and the internal pairs of this module's nodes.
    reserved_names = frozenset({RDX, RAX, CF, OF, ZS}) | pair_names(
        AddWithFlag, SubWithBorrow, Mulx, Mulq, Neg
    )
    fixed_registers = FIXED_REGISTERS

    def check_block(self, parsed: ParsedFunction, block: Block) -> None:
        """A fixed register is declared exactly when the block uses it: by name, or implicitly
        through one-operand MUL."""
        for register in FIXED_REGISTERS:
            fixed = [d for d in parsed.declarations if d.fixed and d.name == register]
            uses = any(
                re.search(rf"(?<![A-Za-z0-9_{{]){register}(?![A-Za-z0-9_}}])", instruction)
                or re.match(r"\s*mul\s", instruction)
                for instruction in parsed.instructions
            )
            if uses and len(fixed) != 1:
                raise GenerationError(
                    f"{block.rust_name}: literal {register} requires exactly one fixed "
                    f'out("{register}") _ operand'
                )
            if not uses and fixed:
                raise GenerationError(f"{block.rust_name}: unused fixed {register} operand")


X86_64_ISA = X86_64()
