"""What an instruction set provides: its syntax, and a lifter from its instructions to the IR.

Each instruction set is one module of this package: `aarch64.py` and `x86_64.py`. Adding one is
adding a file here; this package does not import its instruction sets, so nothing shared changes,
and the crate's configuration (`../../pasta/`) imports the one it transcribes.

How the crate writes a transcription (whether it keeps unread bindings, how it comments an
argument) is the crate's convention, in its `block.Target`, not the instruction set's.

The word is 64 bits throughout: the shared nodes and the proof skeleton state their bounds as
`< 2^64`. A 32-bit instruction set would make the word size a parameter of the `Isa` first.

Everything else is shared: the front end reads the Rust function (`rust.py`), the block driver
binds its operands and runs the lifter (`block.py`), the rerolling strategies fold its rounds and
the validator checks them (`reroll.py`, `validate.py`), and the back ends print the programs and
their proof skeletons (`lean.py`, `skeleton.py`). An instruction set is

- an `Isa`: the Lean namespace of its semantics, the names its transcription reserves, the
  registers a block may name directly, the `asm!` options it requires, the checks of a block it
  needs beyond the shared ones, and the node that loads a non-word field of a round's state;
- a `Lifter` subclass, whose `lift_instruction` checks one instruction and emits its nodes;
- node classes for the operations of its Lean semantics that `ir.py` does not already have, each
  stating its `let`s as terms and its own proof step.

The base `Lifter` keeps what every lifter needs: which registers hold a value
(`require_written`), the operand directions the Rust declares (`check_writable`), the read-only
pointer operands and their argument's fields, the nodes it emits (with their instruction as
comment and position, discarding writes to the `zero_register`, then calling `emitted`), and the
context of an error. Shared node shapes are in `ir.py`: the word operations, a `PairOp` for an
instruction returning a pair, and `ScratchKey` for proof facts spanning several instructions.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar

from ..ir import Load, Mov, Node, Scalar
from ..rust import ALLOWED_OPTIONS, REQUIRED_OPTIONS, Direction, GenerationError, ParsedFunction

if TYPE_CHECKING:
    from ..block import Block


@dataclasses.dataclass(frozen=True)
class Instruction:
    """One instruction: its mnemonic as written, its operand tokens, and its text, which the
    transcription keeps as the comment of the `let`s it emits."""

    op: str
    operands: tuple[str, ...]
    text: str


class Lifter:
    """Lifts a block's instructions to nodes, one at a time. A subclass implements
    `parse_instruction` and `lift_instruction`; it reads and writes registers through its own
    syntax, using the checks here."""

    def __init__(self, directions: Mapping[str, Direction] | None = None) -> None:
        self.directions: dict[str, Direction] = dict(directions or {})
        self.nodes: list[Node] = []
        self.known: set[str] = set()  # the registers holding a value
        self.pointers: dict[str, str] = {}  # pointer operand -> the argument it points to
        # Pointer operand -> the fields of the argument it points to, one per word, in order.
        self.pointer_fields: dict[str, tuple[str, ...]] = {}
        self.used_memory = False
        self.pc: int | None = None
        self.text: str | None = None

    # -- the instruction set's part

    @staticmethod
    def parse_instruction(text: str) -> Instruction:
        raise NotImplementedError

    def lift_instruction(self, ins: Instruction) -> None:
        raise NotImplementedError

    # -- registers

    def check_declared(self, register: str) -> None:
        if register not in self.directions:
            raise GenerationError(f"undeclared register {register}")

    def check_readable(self, register: str) -> None:
        self.check_declared(register)
        if self.directions[register] not in ("in", "inout"):
            raise GenerationError(f"output-only register {register} read before being written")

    def check_writable(self, register: str) -> None:
        self.check_declared(register)
        if self.directions[register] not in ("out", "inout"):
            raise GenerationError(f"input-only register {register} cannot be written")

    # -- operands bound before the block

    def bind(self, node: Load | Scalar | Mov) -> None:
        """Bind an input operand's register to its value on entry."""
        register = node.dest
        self.check_declared(register)
        if register in self.known or register in self.pointers:
            raise GenerationError(f"duplicate initial binding for {register}")
        self.nodes.append(node)
        self.known.add(register)

    def bind_pointer(self, register: str, argument: str, fields: tuple[str, ...]) -> None:
        """A read-only pointer operand to `argument`, whose words, the fields `fields`, the block
        may read."""
        self.check_declared(register)
        if register in self.known or register in self.pointers:
            raise GenerationError(f"duplicate initial binding for {register}")
        self.pointers[register] = argument
        self.pointer_fields[register] = fields

    # -- instructions

    # The instruction set's zero register, if it has one: a write to it is discarded.
    zero_register: ClassVar[str | None] = None

    def emit(self, node: Node) -> None:
        """Append `node` as the current instruction's, which its comment and position record;
        its writes now hold values. A node writing only the zero register is discarded."""
        node = dataclasses.replace(node, comment=self.text, pc=self.pc)
        if self.zero_register is not None and node.writes() == [self.zero_register]:
            return
        self.nodes.append(node)
        self.known.update(node.writes())
        self.emitted(node)

    def emitted(self, node: Node) -> None:
        """A hook after `node` is appended, for state the instruction set tracks (its flags)."""

    def require_written(self, register: str) -> str:
        """`register`, which an instruction reads: it must hold a value."""
        if register not in self.known:
            raise GenerationError(f"{register} read before being written")
        return register

    def lift(self, ins: Instruction | str, pc: int | None = None) -> None:
        """Lift one instruction (or its text), naming it in any error."""
        if isinstance(ins, str):
            ins = self.parse_instruction(ins)
        self.pc, self.text = pc, ins.text
        try:
            self.lift_instruction(ins)
        except GenerationError as error:
            if str(error).endswith(f": {ins.text}"):
                raise
            raise GenerationError(f"{error}: {ins.text}") from error


class Isa:
    """An instruction set: subclass it, and set what differs from these defaults."""

    name: ClassVar[str] = ""  # the Lean namespace of its semantics, `PastaCurves.<name>`
    lifter: ClassVar[type[Lifter]] = Lifter
    reserved_names: ClassVar[frozenset[str]] = frozenset()  # names its transcription binds
    # The registers a block names directly, through a fixed operand.
    fixed_registers: ClassVar[tuple[str, ...]] = ()
    allowed_options: ClassVar[frozenset[str]] = ALLOWED_OPTIONS
    required_options: ClassVar[frozenset[str]] = REQUIRED_OPTIONS

    def state_load(self, kind: str) -> type[Load]:
        """The node type loading a field of Lean type `kind` from a round's state structure."""
        return Load

    def check_block(self, parsed: ParsedFunction, block: Block) -> None:
        """Checks of the block that this instruction set needs beyond the shared ones."""
