"""The x86-64 instructions that the blocks use, lifted to the IR.

Two flags are modelled, CF (`cf`) and OF (`ofl`), as two independent carry chains: `add`, `adc`,
`adcx`, `sub`, `sbb`, and `neg` write CF, `adox` writes OF, `adcx` and `adox` each preserve the
other flag, and `mulx` and moves preserve both. A flag is *valid* while the last instruction that
touched it modelled it; `add`/`adc`/`sub`/`sbb`/`neg` leave OF unmodelled, and `imul` and the
shifts both flags (after a multi-bit shift OF is undefined). Reading an invalid flag is an error:
the transcription's earlier `cf` binding would still be in scope and give a stale value, which Lean
could not notice. The zeroing idiom `xor r, r` (also on the 32-bit subregister, which
zero-extends) writes zero and clears both flags.

Operands are the placeholders of the template; the multiplicand of `mulx` is the fixed register
`rdx`. A read-only pointer operand is read only through `qword ptr [{p} + 8k]` for a limb `k` of
its argument; every other memory operand, and any write to memory, is rejected.

Unlike AArch64, the transcription keeps every architectural result, including dead flag writes
(`DeadCode.RETAIN`).
"""

import dataclasses
import enum
import re
from collections.abc import Sequence

from . import rust
from .ir import (
    MOD_LT,
    DeadCode,
    FlagState,
    Load,
    Mov,
    MulLo,
    Node,
    Scalar,
    Select,
    Shl,
    Shr,
    dst,
    src,
)
from .rust import GenerationError

CF, OF = "cf", "ofl"
RDX = "rdx"


class Mnemonic(str, enum.Enum):
    MOV = "mov"
    MOVABS = "movabs"
    MULX = "mulx"
    IMUL = "imul"
    SHL = "shl"
    SHR = "shr"
    ADD = "add"
    ADC = "adc"
    ADCX = "adcx"
    ADOX = "adox"
    SUB = "sub"
    SBB = "sbb"
    NEG = "neg"
    CMOVNC = "cmovnc"
    XOR = "xor"


# -- the nodes of x86-64's semantics ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True, kw_only=True)
class FlagPair(Node):
    """An instruction that writes a register and a flag through a pair: the pair (`pair`, by the
    Lean function `function`), the result in `dest`, the flag in `flag`."""

    pair = ""
    function = ""
    dest: str = dst()
    flag: str = dst()
    a: str = src()
    b: str = src()
    carry: str = src()

    def lets(self):
        expr = f"{self.function} {self.a} {self.b} {self.carry}"
        return [
            self._let(self.pair, expr, self.a, self.b, self.carry),
            self._let(
                self.dest, f"{self.pair}.1", self.pair, note=f"  `-> {self.source_name('dest')}"
            ),
            self._let(
                self.flag, f"{self.pair}.2", self.pair, note=f"  `-> {self.source_name('flag')}"
            ),
        ]

    def label(self, names):
        return names[1]

    def formulas(self, a, b, carry, value, flag, step):
        raise NotImplementedError

    def prove(self, step):
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

    def formulas(self, a, b, carry, value, flag, step):
        return (
            f"{value} + 2^64 * {flag} = {a} + {b} + {carry}",
            f"addc_lin {a} {b} {carry}",
            f"addc_value_lt {a} {b} {carry}",
            f"addc_carry_le_one {a} {b} {carry} {step.lt64(a)} {step.lt64(b)} {step.le1(carry)}",
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class SubWithBorrow(FlagPair):
    """`sub` and `sbb`: `sbb a b borrow`, whose flag is CF, set on borrow."""

    pair = "d"
    function = "sbb"

    def formulas(self, a, b, carry, value, flag, step):
        return (
            f"{value} + {b} + {carry} = {a} + 2^64 * {flag}",
            f"sbb_lin {a} {b} {carry} {step.lt64(a)} {step.lt64(b)} {step.le1(carry)}",
            f"sbb_value_lt {a} {b} {carry}",
            f"sbb_borrow_le_one {a} {b} {carry}",
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class Mulx(Node):
    """`mulx hi, lo, src`: the two words of `rdx * src`, high first."""

    high: str = dst()
    low: str = dst()
    a: str = src()
    b: str = src()

    def lets(self):
        return [
            self._let("m", f"mulx {self.a} {self.b}", self.a, self.b),
            self._let(self.high, "m.1", "m", note=f"  `-> {self.source_name('high')}"),
            self._let(self.low, "m.2", "m", note=f"  `-> {self.source_name('low')}"),
        ]

    def prove(self, step):
        a, b = step.r(self.a), step.r(self.b)
        high, low = step.names[1:3]
        step.eq(high, f"(mulx {a} {b}).1")
        step.bound(high, f"Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' {step.lt64(a)} {step.lt64(b)})")
        step.define(1)
        step.eq(low, f"(mulx {a} {b}).2")
        step.bound(low, MOD_LT)
        step.define(2)
        step.line(f"  have d_{high} : {low} + 2^64 * {high} = {a} * {b} := by")
        step.line(f"    rw [e_{low}, e_{high}]; exact Nat.mod_add_div _ _")


@dataclasses.dataclass(frozen=True, kw_only=True)
class Neg(Node):
    """`neg`: the low word of `-a`, and CF, set exactly when `a` is nonzero."""

    dest: str = dst()
    a: str = src()

    def lets(self):
        return [
            self._let("n", f"neg {self.a}", self.a),
            self._let(self.dest, "n.1", "n", note=f"  `-> {self.source_name('dest')}"),
            self._let(CF, "n.2", "n", note=f"  `-> {CF}"),
        ]

    def prove(self, step):
        a = step.r(self.a)
        value, flag = step.names[1:3]
        step.eq(value, f"(neg {a}).1")
        step.bound(value, f"sbb_value_lt 0 {a} 0")
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

    def lets(self):
        return [
            self._let(self.dest, "0"),
            self._let(CF, "0", note=f"  `-> {CF}"),
            self._let(OF, "0", note=f"  `-> {OF}"),
        ]

    def prove(self, step):
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


class Lifter:
    """Validates a block's x86-64 instructions and lifts them to nodes."""

    def __init__(self, directions: dict[str, str], pointers: dict[str, str]):
        self.directions = dict(directions)
        self.pointers = dict(pointers)
        self.nodes = []
        self.known = set()
        self.cf_valid = False
        self.of_valid = False
        self.used_memory = False
        self.pc, self.text = None, None

    # -- operands and their checks

    def bind_argument(self, node):
        """Bind an input operand's register to its value on entry."""
        register = node.dest
        self._check_declared(register)
        if register in self.known or register in self.pointers:
            raise GenerationError(f"duplicate initial binding for {register}")
        self.nodes.append(node)
        self.known.add(register)

    def bind_pointer(self, register: str, argument: str) -> None:
        self._check_declared(register)
        if register in self.known or register in self.pointers:
            raise GenerationError(f"duplicate initial binding for {register}")
        self.pointers[register] = argument

    def _check_declared(self, register: str) -> None:
        if register not in self.directions:
            raise GenerationError(f"undeclared register {register}")

    def _check_readable(self, register: str) -> None:
        self._check_declared(register)
        if self.directions[register] not in ("in", "inout"):
            raise GenerationError(f"output-only register {register} read before being written")

    def _check_writable(self, register: str) -> None:
        self._check_declared(register)
        if self.directions[register] not in ("out", "inout"):
            raise GenerationError(f"input-only register {register} cannot be written")

    @staticmethod
    def parse_register(token: str) -> RegisterToken:
        token = token.strip()
        if token.startswith(("qword ptr", "[")):
            raise GenerationError(f"memory write is unsupported: {token}")
        if token == RDX:
            return RegisterToken(RDX)
        match = re.fullmatch(r"\{([A-Za-z_]\w*)(?::([A-Za-z_]\w*))?\}", token)
        if not match:
            raise GenerationError(f"unsupported register operand {token}")
        return RegisterToken(match.group(1), match.group(2))

    def read_register(self, token: str) -> str:
        parsed = self.parse_register(token)
        self._check_declared(parsed.name)
        if parsed.modifier:
            raise GenerationError(f"unsupported register modifier :{parsed.modifier} in {token}")
        if parsed.name in self.pointers:
            raise GenerationError(f"pointer register {parsed.name} used outside an address")
        if parsed.name not in self.known:
            self._check_readable(parsed.name)
            raise GenerationError(f"register {parsed.name} read before being written")
        return parsed.name

    def write_register(self, token: str) -> str:
        parsed = self.parse_register(token)
        self._check_writable(parsed.name)
        if parsed.modifier:
            raise GenerationError(f"unsupported register modifier :{parsed.modifier} in {token}")
        if parsed.name in self.pointers:
            raise GenerationError(f"cannot overwrite pointer operand {parsed.name}")
        self.known.add(parsed.name)
        return parsed.name

    def read_value(self, token: str) -> str:
        """A source operand: an immediate (as a decimal literal), a limb of a read-only pointer
        argument (as `arg.lk`), or a register."""
        token = token.strip()
        if re.fullmatch(r"(?:0|[1-9][0-9]*|0x[0-9A-Fa-f]+)", token):
            return str(int(token, 0))
        memory = re.fullmatch(
            r"qword\s+ptr\s*\[\s*\{([A-Za-z_]\w*)\}\s*(?:\+\s*([0-9]+))?\s*\]", token
        )
        if memory:
            base, displacement_text = memory.groups()
            self._check_declared(base)
            if base not in self.pointers:
                raise GenerationError(f"address base {base} is not a read-only pointer operand")
            displacement = int(displacement_text or "0")
            if displacement not in (0, 8, 16, 24):
                raise GenerationError(f"unsupported memory offset {displacement}")
            self.used_memory = True
            return f"{self.pointers[base]}.l{displacement // 8}"
        return self.read_register(token)

    @staticmethod
    def split_instruction(text: str) -> tuple[str, list[str]]:
        found = re.fullmatch(r"\s*([A-Za-z][A-Za-z0-9]*)\s*(.*?)\s*", text)
        if not found:
            raise GenerationError(f"cannot parse instruction {text!r}")
        op, rest = found.groups()
        operands = [part.strip() for part in rest.split(",")] if rest else []
        if any(not operand for operand in operands):
            raise GenerationError(f"empty operand in instruction {text!r}")
        return op.lower(), operands

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

    def emit(self, node_type, **fields):
        self.nodes.append(node_type(comment=self.text, pc=self.pc, **fields))

    # -- instructions

    def lift(self, text: str) -> None:
        op_text, operands = self.split_instruction(text)
        self.text = text
        try:
            try:
                op = Mnemonic(op_text)
            except ValueError:
                raise GenerationError(f"unsupported instruction {op_text}: {text}") from None
            self._lift(op, operands, text)
        except GenerationError as error:
            if str(error).endswith(f": {text}"):
                raise
            raise GenerationError(f"{error}: {text}") from error

    def _lift(self, op, operands, text):
        match op:
            case Mnemonic.MOV | Mnemonic.MOVABS:
                self.require_count(op.value, operands, 2)
                a = self.read_value(operands[1])
                self.emit(Mov, dest=self.write_register(operands[0]), a=a)
            case Mnemonic.MULX:
                self.require_count(op.value, operands, 3)
                rdx = self.read_register(RDX)
                b = self.read_value(operands[2])
                high = self.write_register(operands[0])
                low = self.write_register(operands[1])
                if high == low:
                    raise GenerationError("mulx destinations must be distinct")
                self.emit(Mulx, high=high, low=low, a=rdx, b=b)
            case Mnemonic.IMUL:
                self.require_count(op.value, operands, 2)
                old = self.read_register(operands[0])
                b = self.read_value(operands[1])
                self.emit(MulLo, dest=self.write_register(operands[0]), a=old, b=b)
                self.cf_valid = self.of_valid = False
            case Mnemonic.SHL | Mnemonic.SHR:
                self.require_count(op.value, operands, 2)
                old = self.read_register(operands[0])
                amount = self.immediate(operands[1])
                if not 1 <= amount < 64:
                    raise GenerationError("only unmasked nonzero 64-bit shift counts are supported")
                node = Shl if op is Mnemonic.SHL else Shr
                self.emit(node, dest=self.write_register(operands[0]), a=old, k=amount)
                self.cf_valid = self.of_valid = False
            case Mnemonic.ADD | Mnemonic.ADC | Mnemonic.ADCX | Mnemonic.ADOX:
                self.require_count(op.value, operands, 2)
                old = self.read_register(operands[0])
                b = self.read_value(operands[1])
                carry = "0"
                if op in (Mnemonic.ADC, Mnemonic.ADCX):
                    carry = self.require(CF, text)
                elif op is Mnemonic.ADOX:
                    carry = self.require(OF, text)
                flag = OF if op is Mnemonic.ADOX else CF
                dest = self.write_register(operands[0])
                self.emit(AddWithFlag, dest=dest, flag=flag, a=old, b=b, carry=carry)
                if op is Mnemonic.ADOX:
                    self.of_valid = True
                else:
                    self.cf_valid = True
                    if op in (Mnemonic.ADD, Mnemonic.ADC):
                        self.of_valid = False
            case Mnemonic.SUB | Mnemonic.SBB:
                self.require_count(op.value, operands, 2)
                old = self.read_register(operands[0])
                b = self.read_value(operands[1])
                borrow = self.require(CF, text) if op is Mnemonic.SBB else "0"
                dest = self.write_register(operands[0])
                self.emit(SubWithBorrow, dest=dest, flag=CF, a=old, b=b, carry=borrow)
                self.cf_valid, self.of_valid = True, False
            case Mnemonic.NEG:
                self.require_count(op.value, operands, 1)
                old = self.read_register(operands[0])
                self.emit(Neg, dest=self.write_register(operands[0]), a=old)
                self.cf_valid, self.of_valid = True, False
            case Mnemonic.CMOVNC:
                self.require_count(op.value, operands, 2)
                self.require(CF, text)
                old = self.read_register(operands[0])
                b = self.read_value(operands[1])
                self.emit(
                    Select,
                    dest=self.write_register(operands[0]),
                    function="cmovnc",
                    flag=CF,
                    x=old,
                    y=b,
                    x_when=FlagState.SET,
                )
            case Mnemonic.XOR:
                self.require_count(op.value, operands, 2)
                left, right = self.parse_register(operands[0]), self.parse_register(operands[1])
                self._check_declared(left.name)
                self._check_declared(right.name)
                if left.name != right.name or left.modifier != right.modifier:
                    raise GenerationError("only XOR-self zeroing is supported")
                self._check_writable(left.name)
                if left.modifier not in (None, "e"):
                    raise GenerationError(f"unsupported XOR register modifier :{left.modifier}")
                if left.name in self.pointers:
                    raise GenerationError(f"cannot XOR pointer operand {left.name}")
                self.known.add(left.name)
                self.emit(ZeroIdiom, dest=left.name)
                self.cf_valid = self.of_valid = True


# -- a block ------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Block:
    """An inline `asm!` block to transcribe: the Rust function, the Lean name, the arguments as
    (name, kind), the result's Lean type and number of words, and the docstring."""

    rust_name: str
    lean_name: str
    args: tuple
    result_type: str
    result_count: int
    doc: str


@dataclasses.dataclass(frozen=True)
class Target:
    """What the x86-64 transcription needs to know about the crate: the reserved operand names,
    the named constants an operand may bind (`consts`, name -> value), the proof conventions, and
    the blocks whose rounds are rerolled (`rounds`, by Lean name)."""

    conventions: object
    consts: dict
    rounds: dict = dataclasses.field(default_factory=dict)
    reserved_names: frozenset = frozenset({RDX, CF, OF, "s", "d", "m", "n"})


def parse_function(source, block, target):
    return rust.parse_function(
        source,
        block.rust_name,
        [name for name, _ in block.args],
        block.result_count,
        reserved_names=set(target.reserved_names),
        fixed_registers={RDX},
        const_operands=set(target.consts),
    )


def input_expression(value, locals_map, argument_types):
    """The value an input operand binds, and the comment describing it."""
    direct = re.fullmatch(r"([A-Za-z_]\w*)\[([0-9]+)\]", value)
    if direct:
        argument, index = direct.group(1), int(direct.group(2))
    elif value in locals_map:
        argument, index = locals_map[value]
    elif value in argument_types and argument_types[value] == "Nat":
        return value, None, "scalar input"
    else:
        raise GenerationError(f"unsupported input operand expression {value}")
    if argument not in argument_types:
        raise GenerationError(f"operand reads non-argument {argument}")
    kind = argument_types[argument]
    limit = 8 if kind == "WideLimbs" else 4 if kind == "Limbs" else 0
    if index < 0 or index >= limit:
        raise GenerationError(f"limb {index} is out of range for {argument} : {kind}")
    return argument, f"l{index}", f"input {value}"


def lift_block(source, block, target):
    """The flat program of one block: its operands bound, its instructions lifted and checked."""
    from . import lean

    parsed = parse_function(source, block, target)
    argument_types = dict(block.args)
    lifter = Lifter(rust.declaration_directions(parsed, block.rust_name), {})
    pointer_count = 0

    fixed_rdx = [d for d in parsed.declarations if d.fixed]
    uses_rdx = any(
        re.search(r"(?<![A-Za-z0-9_{])rdx(?![A-Za-z0-9_}])", instruction)
        for instruction in parsed.instructions
    )
    if uses_rdx and len(fixed_rdx) != 1:
        raise GenerationError(
            f'{block.rust_name}: literal rdx requires exactly one fixed out("rdx") _ operand'
        )
    if not uses_rdx and fixed_rdx:
        raise GenerationError(f"{block.rust_name}: unused fixed rdx operand")

    for d in parsed.declarations:
        if d.fixed:
            continue
        if d.kind == "const":
            lifter.bind_argument(
                Mov(
                    comment=f"operand {d.name} = const {d.value}",
                    dest=d.name,
                    a=str(target.consts[d.value]),
                )
            )
        elif d.kind in ("in", "inout"):
            pointer = re.fullmatch(r"([A-Za-z_]\w*)\.as_ptr\(\)", d.value)
            if pointer:
                argument = pointer.group(1)
                if argument_types.get(argument) != "Limbs":
                    raise GenerationError(
                        f"{block.rust_name}: pointer {d.name} is not a Limbs argument"
                    )
                lifter.bind_pointer(d.name, argument)
                pointer_count += 1
            else:
                arg, field, comment = input_expression(d.value, parsed.locals, argument_types)
                if field is None:
                    lifter.bind_argument(Scalar(comment=comment, dest=d.name, name=arg))
                else:
                    lifter.bind_argument(Load(comment=comment, dest=d.name, arg=arg, field=field))
        elif d.kind != "out":
            raise GenerationError(f"{block.rust_name}: unsupported direction {d.kind}")

    if pointer_count:
        if "readonly" not in parsed.options or "nomem" in parsed.options:
            raise GenerationError(
                f"{block.rust_name}: pointer operands require readonly and forbid nomem"
            )
    elif "nomem" not in parsed.options or "readonly" in parsed.options:
        raise GenerationError(
            f"{block.rust_name}: register-only block requires nomem and forbids readonly"
        )

    for pc, instruction in enumerate(parsed.instructions):
        lifter.pc = pc
        lifter.lift(instruction)
    if lifter.used_memory != bool(pointer_count):
        raise GenerationError(
            f"{block.rust_name}: declared pointer operands and memory reads disagree"
        )

    results = list(rust.returned_registers(parsed, block.rust_name))
    for register in results:
        if register not in lifter.known:
            raise GenerationError(
                f"{block.rust_name}: output register {register} was never written"
            )
    signature = lean.signature(block.lean_name, block.args, block.result_type)
    return target.conventions.program(
        "X86_64",
        block.lean_name,
        block.doc,
        signature,
        lifter.nodes,
        results,
        dead_code=DeadCode.RETAIN,
    )


def transcribe(source, block, target):
    """The programs of one block: the block, preceded by its round when its rounds reroll."""
    program = lift_block(source, block, target)
    rounds = target.rounds.get(block.lean_name)
    if rounds is None:
        return [program]

    def make(name, doc, signature, nodes, results, **kw):
        return target.conventions.program(
            "X86_64", name, doc, signature, nodes, results, dead_code=DeadCode.RETAIN, **kw
        )

    return rounds.reroll(program, make, GenerationError, "scalar argument")
