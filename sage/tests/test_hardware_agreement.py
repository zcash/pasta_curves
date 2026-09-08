"""Check the interpreter's instruction semantics against the real CPU.

The interpreter's value is that it models the backend faithfully, and the part
easiest to get wrong is the carry flag: on AArch64 a set carry after a
subtraction means it did *not* borrow. Rather than assert what the manual says,
each snippet below is run twice, once through `asm_aarch64` and once on the
hardware, and the two are required to agree on the registers, the carry and a
scratch area of memory. The same text drives both, so they cannot drift apart.

The hardware side is not a subprocess. `assembly_for` wraps every snippet in a
function taking a pointer to one array of slots, `cc -shared` compiles them, and
`ctypes.CDLL` dlopens the result into this process; calling one resolves the
symbol with dlsym and branches to it, the pointer arriving in x0 per the AArch64
ABI. `byref` passes that array without copying, so the wrapper writes the
registers, carry and scratch back into the buffer the test still holds.

Skipped when not on Apple AArch64, or when no C compiler is available. Set
PASTA_CURVES_REQUIRE_HARDWARE_CHECK=1 to turn that skip into a failure, which
is what CI does on the Apple AArch64 job: a silent skip there would mean the
comparison never runs and nobody notices.
"""

import ctypes
import os
import platform
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

import asm_aarch64 as asm
from asm_aarch64 import M64

REGISTERS = 16  # x0 to x15, all carried in and out
SCRATCH = 8  # 64 bytes of memory, addressed through x6
FLAGS_SLOT = 16  # NZCV, at byte offset 128
SCRATCH_SLOT = 20  # the scratch block, at byte offset 160
CARRY_BIT = 29  # C is bit 29 of NZCV
BASE = 0x8000  # where the interpreter puts the same block

# x7 to x15 are untouched by every snippet, so carrying them in and out turns
# any stray write by the interpreter into a failure rather than going unseen.
SENTINELS = {
    f"x{index}": 0x0101010101010100 + index for index in range(7, REGISTERS)
}

# (snippet, preset x0-x5, preset scratch words, initial carry). x6 always
# holds the scratch address, and is reported back as an offset so the two runs
# are comparable. The carry is an input as well as an output, so it is set
# before the snippet rather than always starting clear.
Case = tuple[str, dict[str, int], list[int], int]

FILLED = [11, 22, 33, 44, 55, 66, 77, 88]

CASES: tuple[Case, ...] = (
    # Subtraction sets the carry when it does NOT borrow.
    ("subs x0,x1,x2", {"x1": 0, "x2": 1}, [], 0),
    ("subs x0,x1,x2", {"x1": 5, "x2": 3}, [], 0),
    ("subs x0,x1,x2", {"x1": 3, "x2": 3}, [], 0),
    (
        "subs x0,x2,x3\nsbcs x1,x4,x5",
        {"x2": 0, "x3": 1, "x4": 1, "x5": 0},
        [],
        0,
    ),
    (
        "subs x0,x1,x2\nsbcs x3,x4,x5",
        {"x1": 0, "x2": 0, "x4": 0, "x5": 0},
        [],
        0,
    ),
    # Addition.
    ("adds x0,x1,x2", {"x1": M64, "x2": 1}, [], 0),
    ("adds x0,x1,x2", {"x1": 1, "x2": 1}, [], 0),
    (
        "adds x0,x2,x3\nadcs x1,x4,x5",
        {"x2": M64, "x3": 1, "x4": 7, "x5": 0},
        [],
        0,
    ),
    # `adc` and `sbc` produce a result without disturbing the flags.
    (
        "adds xzr,x1,x2\nadc x0,xzr,xzr\nadcs x3,xzr,xzr",
        {"x1": M64, "x2": 1},
        [],
        0,
    ),
    (
        "adds xzr,x1,x2\nadc x0,x3,x4",
        {"x1": M64, "x2": 1, "x3": 10, "x4": 20},
        [],
        0,
    ),
    (
        "subs xzr,x1,x2\nsbc x0,x3,x4",
        {"x1": 3, "x2": 5, "x3": 10, "x4": 4},
        [],
        0,
    ),
    # The two idioms the mutation harness injects to force a carry.
    ("subs xzr,xzr,xzr", {}, [], 0),
    ("adds xzr,xzr,xzr", {}, [], 0),
    # The backend's own cancellation idiom: carry set exactly when x1 != 0.
    ("subs xzr,x1,#1", {"x1": 0}, [], 0),
    ("subs xzr,x1,#1", {"x1": 1}, [], 0),
    ("subs xzr,x1,#1", {"x1": M64}, [], 0),
    # Immediate operands, as the `adc` carry-set mutation emits.
    ("add x0,x1,#1", {"x1": M64}, [], 0),
    ("adds x0,x1,#1", {"x1": M64}, [], 0),
    # Selection on `lo`, which is carry clear.
    (
        "subs xzr,x1,x2\ncsel x0,x3,x4,lo",
        {"x1": 3, "x2": 5, "x3": 11, "x4": 22},
        [],
        0,
    ),
    (
        "subs xzr,x1,x2\ncsel x0,x3,x4,lo",
        {"x1": 5, "x2": 3, "x3": 11, "x4": 22},
        [],
        0,
    ),
    ("subs xzr,x1,x2\nsbcs xzr,xzr,xzr", {"x1": 5, "x2": 3}, [], 0),
    ("subs xzr,x1,x2\nsbcs xzr,xzr,xzr", {"x1": 3, "x2": 5}, [], 0),
    # Products and shifts, none of which touch the flags.
    (
        "mul x0,x1,x2\numulh x3,x1,x2",
        {"x1": 0xDEADBEEF12345678, "x2": 0xFEEDFACE87654321},
        [],
        0,
    ),
    ("mul x0,x1,x2\numulh x3,x1,x2", {"x1": M64, "x2": M64}, [], 0),
    ("lsl x0,x1,#62\nlsr x3,x1,#2", {"x1": 0xFF}, [], 0),
    ("lsl x0,x1,#62\nlsr x3,x1,#2", {"x1": M64}, [], 0),
    ("mov x0,x1", {"x1": 0x1234}, [], 0),
    # Memory. x6 addresses the scratch block.
    ("stp x0,x1,[x6]", {"x0": 7, "x1": 9}, [], 0),
    ("stp x0,x1,[x6]\nldp x2,x3,[x6]", {"x0": 7, "x1": 9}, [], 0),
    ("ldp x0,x1,[x6]", {}, FILLED, 0),
    ("ldr x0,[x6,#16]", {}, FILLED, 0),
    ("str x0,[x6,#16]\nldr x2,[x6,8*2]", {"x0": 5}, [], 0),
    ("ldp x0,x1,[x6,#16]", {}, FILLED, 0),
    # Pre-index and post-index both write the base register back.
    ("stp x0,x1,[x6,#16]!", {"x0": 7, "x1": 9}, [], 0),
    ("ldr x0,[x6],#8", {}, FILLED, 0),
    ("ldp x0,x1,[x6],#16", {}, FILLED, 0),
    # The carry as an input. `adc`, `sbc` and `csel` read it with no preceding
    # instruction to set it, which is how the reduction rounds and the
    # mutations both use it.
    ("adc x0,x1,x2", {"x1": 10, "x2": 20}, [], 1),
    ("adc x0,x1,x2", {"x1": 10, "x2": 20}, [], 0),
    ("adc x0,xzr,xzr", {}, [], 1),
    ("adc x0,xzr,xzr", {}, [], 0),
    ("adcs x0,x1,xzr", {"x1": M64}, [], 1),
    ("adcs x0,x1,xzr", {"x1": M64}, [], 0),
    ("adcs x0,x1,x2", {"x1": M64, "x2": 0}, [], 1),
    ("sbc x0,x1,x2", {"x1": 10, "x2": 4}, [], 1),
    ("sbc x0,x1,x2", {"x1": 10, "x2": 4}, [], 0),
    ("sbcs x0,x1,x2", {"x1": 5, "x2": 5}, [], 1),
    ("sbcs x0,x1,x2", {"x1": 5, "x2": 5}, [], 0),
    ("sbcs xzr,x1,xzr", {"x1": 0}, [], 1),
    ("sbcs xzr,x1,xzr", {"x1": 0}, [], 0),
    ("csel x0,x3,x4,lo", {"x3": 11, "x4": 22}, [], 1),
    ("csel x0,x3,x4,lo", {"x3": 11, "x4": 22}, [], 0),
)


def assembly_for(cases: tuple[Case, ...]) -> str:
    """One `_case_N(uint64_t *slots)` per snippet, saving x0-x6, flags, memory.

    NZCV is loaded from the block rather than inherited from the caller, so a
    snippet that reads the carry is driven from a known state and one that
    leaves it alone is verifiable. x6 is reported as an offset from the block,
    so a writeback is comparable across the two runs even though the addresses
    differ.
    """
    out = ["    .text"]
    for index, (snippet, _, _, _) in enumerate(cases):
        out += [
            f"    .globl _case_{index}",
            f"_case_{index}:",
            # Claim x19 and x20, and keep the slot pointer in x19.
            "    stp x19,x20,[sp,#-16]!",
            "    mov x19,x0",
            # The registers in.
            "    ldp x0,x1,[x19]",
            "    ldp x2,x3,[x19,#16]",
            "    ldp x4,x5,[x19,#32]",
            "    ldp x6,x7,[x19,#48]",
            "    ldp x8,x9,[x19,#64]",
            "    ldp x10,x11,[x19,#80]",
            "    ldp x12,x13,[x19,#96]",
            "    ldp x14,x15,[x19,#112]",
            # The scratch block is part of the same array, so it needs no
            # copying: x6 just points the snippet at it.
            "    add x6,x19,#160",
            # The carry, driven from the block rather than inherited.
            "    ldr x20,[x19,#128]",
            "    msr nzcv,x20",
        ]
        out += [f"    {line.strip()}" for line in snippet.splitlines()]
        out += [
            # The carry out, before the `sub` below overwrites the flags.
            "    mrs x20,nzcv",
            # x6 as an offset, since the two runs sit at different addresses.
            "    sub x6,x6,x19",
            # The registers out.
            "    stp x0,x1,[x19]",
            "    stp x2,x3,[x19,#16]",
            "    stp x4,x5,[x19,#32]",
            "    stp x6,x7,[x19,#48]",
            "    stp x8,x9,[x19,#64]",
            "    stp x10,x11,[x19,#80]",
            "    stp x12,x13,[x19,#96]",
            "    stp x14,x15,[x19,#112]",
            "    str x20,[x19,#128]",
            # Give x19 and x20 back to the caller.
            "    ldp x19,x20,[sp],#16",
            "    ret",
        ]
    return "\n".join(out) + "\n"


Observation = tuple[list[int], int, list[int]]


def emulate(
    snippet: str, registers: dict[str, int], scratch: list[int], carry: int
) -> Observation:
    """Run a snippet through the interpreter: registers, carry and memory."""
    body = "".join(f"    {line.strip()}\n" for line in snippet.splitlines())
    program = asm.Program("_case:\n" + body + "    ret\n")
    cpu = asm.Cpu(program)
    for name, value in {**SENTINELS, **registers}.items():
        cpu.write(name, value)
    start = BASE + 8 * SCRATCH_SLOT
    cpu.write("x6", start)
    for index in range(SCRATCH):
        cpu.memory[start + 8 * index] = scratch[index] if scratch else 0
    cpu.carry = carry
    cpu.run(program.labels["_case"])
    values = [cpu.read(f"x{index}") for index in range(REGISTERS)]
    values[6] = (values[6] - BASE) & M64
    memory = [cpu.memory.get(start + 8 * index, 0) for index in range(SCRATCH)]
    return values, cpu.carry, memory


def unavailable() -> str | None:
    """Why the hardware comparison cannot run here, or None if it can."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return f"needs Apple AArch64, this is {platform.system()}/{platform.machine()}"
    if shutil.which("cc") is None:
        return "needs a C compiler, found no `cc` on PATH"
    return None


def describe(case: Case) -> str:
    """A readable pytest id: the snippet, its inputs and the incoming carry."""
    snippet, registers, scratch, carry = case
    inputs = " ".join(
        f"{name}={value:#x}" for name, value in sorted(registers.items())
    )
    filled = " mem" if scratch else ""
    return f"{snippet.replace(chr(10), '; ')} [{inputs}{filled} C={carry}]"


@pytest.fixture(scope="module")
def cases_library() -> Iterator[ctypes.CDLL]:
    """Compile every snippet once and hand back the loaded library."""
    reason = unavailable()
    if reason is not None:
        if os.environ.get("PASTA_CURVES_REQUIRE_HARDWARE_CHECK") == "1":
            raise AssertionError(f"hardware comparison required but {reason}")
        pytest.skip(reason)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "cases.S"
        source.write_text(assembly_for(CASES))
        library = root / "libcases.dylib"
        subprocess.run(
            ["cc", "-shared", "-o", str(library), str(source)],
            check=True,
            capture_output=True,
        )
        yield ctypes.CDLL(str(library))


def on_hardware(library: ctypes.CDLL, index: int, case: Case) -> Observation:
    """Run one compiled snippet: registers, carry and memory.

    One `c_uint64` array carries the state both ways. `assembly_for` gave each
    snippet a wrapper that loads x0-x15, the scratch block and NZCV out of it,
    runs the snippet, and stores them back, so calling `case_<index>` through
    ctypes executes those instructions on this CPU and leaves the result in
    the array.
    """
    _, registers, scratch, carry = case
    slots = (ctypes.c_uint64 * (SCRATCH_SLOT + SCRATCH))()
    for name, value in {**SENTINELS, **registers}.items():
        slots[int(name[1:])] = value
    for position in range(SCRATCH):
        slots[SCRATCH_SLOT + position] = scratch[position] if scratch else 0
    slots[FLAGS_SLOT] = carry << CARRY_BIT
    getattr(library, f"case_{index}")(ctypes.byref(slots))
    values = [slots[position] for position in range(REGISTERS)]
    memory = [slots[SCRATCH_SLOT + position] for position in range(SCRATCH)]
    return values, (slots[FLAGS_SLOT] >> CARRY_BIT) & 1, memory


@pytest.mark.parametrize(
    ("index", "case"),
    list(enumerate(CASES)),
    ids=[describe(case) for case in CASES],
)
def test_interpreter_agrees_with_the_cpu(
    cases_library: ctypes.CDLL, index: int, case: Case
) -> None:
    """The interpreter and the CPU agree on registers, carry and memory."""
    snippet, registers, scratch, carry = case
    assert emulate(snippet, registers, scratch, carry) == on_hardware(
        cases_library, index, case
    )
