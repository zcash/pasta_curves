"""A small AArch64 interpreter for `src/asm/pasta_mul-armv8.S`.

Executes the shipped assembly instruction by instruction so that the carry
flag becomes observable: `Cpu.trace` records the carry-in of every
flag-consuming instruction. Supports only the 64-bit forms the backend uses,
and treats the `.long`-encoded `paciasp`/`autiasp` as no-ops. It is a model of
the assembly, not a substitute for running it. Pure Python; no SageMath.
"""

import re
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum

M64 = (1 << 64) - 1

#: A decoded instruction: mnemonic, operands, and its line in the source.
Instruction = tuple[str, list[str], int]

#: What every flag-consuming instruction saw, as (instruction index, carry-in);
#: for `csel` the recorded value is 1 when the condition was taken.
Trace = list[tuple[int, int]]

# Where `call` places the operands the routines expect as pointers.
OUT, LHS, RHS, MOD = 0x1000, 0x2000, 0x3000, 0x4000


class Mnemonic(StrEnum):
    """Every instruction the interpreter dispatches on."""

    ADC = "adc"
    ADCS = "adcs"
    ADD = "add"
    ADDS = "adds"
    BL = "bl"
    CSEL = "csel"
    LDP = "ldp"
    LDR = "ldr"
    LSL = "lsl"
    LSR = "lsr"
    MOV = "mov"
    MUL = "mul"
    NOP = "nop"
    RET = "ret"
    SBC = "sbc"
    SBCS = "sbcs"
    STP = "stp"
    STR = "str"
    SUB = "sub"
    SUBS = "subs"
    UMULH = "umulh"


MEMORY_ACCESS = (Mnemonic.LDP, Mnemonic.LDR, Mnemonic.STP, Mnemonic.STR)
PAIRED_ACCESS = (Mnemonic.LDP, Mnemonic.STP)
LOAD_ACCESS = (Mnemonic.LDP, Mnemonic.LDR)
PRODUCTS = (Mnemonic.MUL, Mnemonic.UMULH)
SHIFTS = (Mnemonic.LSL, Mnemonic.LSR)
ADDITIONS = (Mnemonic.ADD, Mnemonic.ADDS, Mnemonic.ADC, Mnemonic.ADCS)
SUBTRACTIONS = (Mnemonic.SUB, Mnemonic.SUBS, Mnemonic.SBC, Mnemonic.SBCS)
CARRY_CONSUMING = (Mnemonic.ADC, Mnemonic.ADCS, Mnemonic.SBC, Mnemonic.SBCS)
FLAG_SETTING = (Mnemonic.ADDS, Mnemonic.ADCS, Mnemonic.SUBS, Mnemonic.SBCS)


def split_operands(text: str) -> list[str]:
    """Split an operand list, keeping bracketed addressing modes intact."""
    operands, depth, current = [], 0, ""
    for char in text:
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        if char == "," and depth == 0:
            operands.append(current.strip())
            current = ""
        else:
            current += char
    if current.strip():
        operands.append(current.strip())
    return operands


class Program:
    """A parsed assembly file: a flat instruction list plus its labels."""

    def __init__(self, source: str) -> None:
        """Parse `source` into a flat instruction list and a label table."""
        self.instructions: list[Instruction] = []
        self.labels: dict[str, int] = {}
        for lineno, raw in enumerate(source.splitlines(), 1):
            line = raw.split("//")[0].strip()
            if not line:
                continue
            if line.endswith(":"):
                self.labels[line[:-1]] = len(self.instructions)
                continue
            if line.startswith("."):
                if line.startswith(".long"):
                    # paciasp / autiasp: no effect on the modelled state.
                    self.instructions.append((Mnemonic.NOP, [], lineno))
                continue
            head, _, rest = line.partition(" ")
            self.instructions.append(
                (head.lower(), split_operands(rest) if rest else [], lineno)
            )

    def replacing(
        self, index: int, replacement: Sequence[Instruction]
    ) -> "Program":
        """A copy with instruction `index` replaced by `replacement` (a list)."""
        clone = Program.__new__(Program)
        clone.instructions = (
            self.instructions[:index]
            + list(replacement)
            + self.instructions[index + 1 :]
        )
        shift = len(clone.instructions) - len(self.instructions)
        clone.labels = {
            name: target + shift if target > index else target
            for name, target in self.labels.items()
        }
        return clone


def register_name(name: str) -> str:
    """Normalise a register name, rejecting the unmodelled 32-bit views."""
    name = name.lower()
    if name == "wzr" or (name.startswith("w") and name[1:].isdigit()):
        raise ValueError(f"32-bit register {name!r} is not modelled")
    return name


class Cpu:
    """Execution state: the 64-bit registers, word-addressed memory, carry."""

    def __init__(self, program: Program) -> None:
        """Start with zeroed registers, empty memory and a clear carry flag."""
        self.program = program
        self.registers: dict[str, int] = {
            f"x{number}": 0 for number in range(31)
        }
        self.registers["sp"] = 1 << 40
        self.memory: dict[int, int] = {}
        self.carry = 0
        self.trace: Trace = []

    def read(self, name: str) -> int:
        """The value of a register, with `xzr` reading as zero."""
        name = register_name(name)
        return 0 if name == "xzr" else self.registers[name]

    def write(self, name: str, value: int) -> None:
        """Write a register, discarding writes to `xzr`."""
        name = register_name(name)
        if name != "xzr":
            self.registers[name] = value & M64

    @staticmethod
    def immediate(token: str) -> int:
        """Evaluate an immediate, which the backend may write scaled as `8*1`."""
        token = token.strip().lstrip("#")
        # The backend writes scaled offsets such as `8*1`.
        return int(eval(token, {"__builtins__": {}}, {}))

    def operand(self, token: str) -> int:
        """Resolve an operand that is either an immediate or a register."""
        token = token.strip()
        if token.startswith("#") or token[0].isdigit():
            return self.immediate(token)
        return self.read(token)

    def address(
        self, tokens: Sequence[str]
    ) -> tuple[int, tuple[str, int] | None]:
        """Resolve `[xN]`, `[xN,#off]`, `[xN,#off]!` and `[xN],#off`."""
        token = tokens[0]
        bare = re.match(r"^\[(\w+)\]$", token)
        if bare:
            base = bare.group(1)
            target = self.read(base)
            if len(tokens) > 1:
                return target, (
                    base,
                    (target + self.immediate(tokens[1])) & M64,
                )
            return target, None
        pre = re.match(r"^\[(\w+),\s*#?([-\w*+ ]+)\](!?)$", token)
        if pre:
            base = pre.group(1)
            target = (self.read(base) + self.immediate(pre.group(2))) & M64
            return target, ((base, target) if pre.group(3) == "!" else None)
        raise ValueError(f"unsupported addressing mode {token!r}")

    def run(self, start: int, step_limit: int = 100000) -> None:
        """Execute from `start` until the outermost `ret` returns."""
        pc, steps = start, 0
        returns: list[int] = []
        while True:
            steps += 1
            if steps > step_limit:
                raise RuntimeError("instruction limit exceeded")
            mnemonic, operands, _ = self.program.instructions[pc]
            following = pc + 1

            if mnemonic == Mnemonic.RET:
                if not returns:
                    return
                following = returns.pop()
            elif mnemonic == Mnemonic.BL:
                returns.append(pc + 1)
                following = self.program.labels[operands[0]]
            elif mnemonic == Mnemonic.NOP:
                pass
            elif mnemonic in PRODUCTS:
                product = self.read(operands[1]) * self.read(operands[2])
                self.write(
                    operands[0],
                    product & M64
                    if mnemonic == Mnemonic.MUL
                    else product >> 64,
                )
            elif mnemonic in SHIFTS:
                value = self.read(operands[1])
                shift = self.immediate(operands[2])
                self.write(
                    operands[0],
                    (value << shift) & M64
                    if mnemonic == Mnemonic.LSL
                    else value >> shift,
                )
            elif mnemonic == Mnemonic.MOV:
                self.write(operands[0], self.operand(operands[1]))
            elif mnemonic in ADDITIONS:
                carry_in = self.carry if mnemonic in CARRY_CONSUMING else 0
                if mnemonic in CARRY_CONSUMING:
                    self.trace.append((pc, carry_in))
                total = (
                    self.read(operands[1])
                    + self.operand(operands[2])
                    + carry_in
                )
                if mnemonic in FLAG_SETTING:
                    self.carry = 1 if total > M64 else 0
                self.write(operands[0], total)
            elif mnemonic in SUBTRACTIONS:
                carry_in = self.carry if mnemonic in CARRY_CONSUMING else 1
                if mnemonic in CARRY_CONSUMING:
                    self.trace.append((pc, carry_in))
                total = (
                    self.read(operands[1])
                    + (~self.operand(operands[2]) & M64)
                    + carry_in
                )
                if mnemonic in FLAG_SETTING:
                    self.carry = 1 if total > M64 else 0
                self.write(operands[0], total)
            elif mnemonic == Mnemonic.CSEL:
                condition = operands[3].lower()
                if condition != "lo":
                    raise ValueError(f"unsupported condition {condition!r}")
                taken = self.carry == 0
                self.trace.append((pc, 1 if taken else 0))
                self.write(operands[0], self.read(operands[1 if taken else 2]))
            elif mnemonic in MEMORY_ACCESS:
                pair = mnemonic in PAIRED_ACCESS
                target, writeback = self.address(operands[2 if pair else 1 :])
                if mnemonic in LOAD_ACCESS:
                    self.write(operands[0], self.memory.get(target, 0))
                    if pair:
                        self.write(operands[1], self.memory.get(target + 8, 0))
                else:
                    self.memory[target] = self.read(operands[0])
                    if pair:
                        self.memory[target + 8] = self.read(operands[1])
                if writeback:
                    self.write(writeback[0], writeback[1])
            else:
                raise ValueError(
                    f"unsupported instruction {mnemonic} {operands}"
                )
            pc = following


def limbs(value: int) -> list[int]:
    """Split a value into its four 64-bit limbs, least significant first."""
    return [(value >> (64 * index)) & M64 for index in range(4)]


def from_limbs(values: Iterable[int]) -> int:
    """Reassemble a value from its limbs, least significant first."""
    return sum(limb << (64 * index) for index, limb in enumerate(values))


def _call(
    program: Program,
    symbol: str,
    registers: Mapping[str, int],
    operands: Mapping[int, int],
) -> tuple[int, Trace]:
    """Run one routine with the given registers and operands in memory."""
    cpu = Cpu(program)
    for name, value in registers.items():
        cpu.write(name, value)
    for base, value in operands.items():
        for index, limb in enumerate(limbs(value)):
            cpu.memory[base + 8 * index] = limb
    cpu.run(program.labels[symbol])
    return from_limbs(
        [cpu.memory.get(OUT + 8 * i, 0) for i in range(4)]
    ), cpu.trace


def mul(
    program: Program, lhs: int, rhs: int, modulus: int, inv: int
) -> tuple[int, Trace]:
    """`_pasta_curves_mul_mont_pasta`: returns (result, trace)."""
    return _call(
        program,
        "_pasta_curves_mul_mont_pasta",
        {"x0": OUT, "x1": LHS, "x2": RHS, "x3": MOD, "x4": inv},
        {LHS: lhs, RHS: rhs, MOD: modulus},
    )


def square(
    program: Program, value: int, modulus: int, inv: int
) -> tuple[int, Trace]:
    """`_pasta_curves_sqr_mont_pasta`: returns (result, trace)."""
    return _call(
        program,
        "_pasta_curves_sqr_mont_pasta",
        {"x0": OUT, "x1": LHS, "x2": MOD, "x3": inv},
        {LHS: value, MOD: modulus},
    )


def from_mont(
    program: Program, value: int, modulus: int, inv: int
) -> tuple[int, Trace]:
    """`_pasta_curves_from_mont_pasta`: returns (result, trace)."""
    return _call(
        program,
        "_pasta_curves_from_mont_pasta",
        {"x0": OUT, "x1": LHS, "x2": MOD, "x3": inv},
        {LHS: value, MOD: modulus},
    )
