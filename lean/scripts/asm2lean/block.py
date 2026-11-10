"""The block driver: from one Rust function's `asm!` block to its Lean programs, for any
instruction set.

    parse     the front end reads the function (`rust.parse_function`)
    bind      each input operand is bound to its argument: a word, a field of a structured
              argument, a named constant, or a read-only pointer to an argument
    lift      the instruction set's lifter checks each instruction and emits its nodes
    results   the registers the Rust function returns, each written by the block
    reroll    the target's strategy for the block, if any, folds its repeated rounds
    validate  the rerolled programs are checked to compute the flat program's results

The instruction set comes in through its `Isa` (`isa/`), and the crate's choices through the
`Target`: which kinds of argument there are, how the proofs name things, the named constants,
and how each block's rounds fold.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable
from typing import Any, Protocol

from . import rust, validate
from .ir import Conventions, DeadCode, Load, Mov, Node, Program, Scalar, Signature, lit
from .isa import Instruction, Isa, Lifter
from .reroll import MacroRound, MacroRounds
from .rust import Declaration, GenerationError, ParsedFunction


@dataclasses.dataclass(frozen=True)
class Block:
    """An inline `asm!` block to transcribe: the Rust function whose block to read, the Lean
    name, the docstring, the arguments as (name, kind) in signature order, and the result's
    kind, whose fields are the result words in the source's result order."""

    rust_name: str
    lean_name: str
    doc: str
    args: tuple[tuple[str, str], ...]
    result: str = "Limbs"

    @property
    def arg_names(self) -> list[str]:
        return [name for name, _ in self.args]


class Reroller(Protocol):
    """A rerolling strategy (`reroll.py`): from a block's flat program, its round definitions
    and the block calling them, in that order."""

    def reroll(self, flat: Program, unit: Unit) -> list[Program]: ...


@dataclasses.dataclass(frozen=True)
class Target:
    """What a transcription needs to know about the crate: the instruction set, the fields of each
    argument or result kind (`None` for a word), the proof conventions, the named constants an
    operand may bind (`consts`, name -> value), the strategy that rerolls each block that has one
    (`rerollers`, by Lean name), and the macro arms rerolled as rounds (`macro_rounds`, see
    `reroll.MacroRounds`), used for any other block whose template invokes them.

    It also holds how the crate writes its transcription, which is a convention of the crate and
    not of the instruction set: whether a transcription keeps the bindings nothing reads
    (`dead_code`), and the comment on the binding of an input operand whose Rust value is
    `value` (`input_comment(value, scalar)`)."""

    isa: Isa
    kinds: dict[str, tuple[str, ...] | None]
    conventions: Conventions
    consts: dict[str, int] = dataclasses.field(default_factory=dict)
    rerollers: dict[str, Reroller] = dataclasses.field(default_factory=dict)
    macro_rounds: dict[tuple[str, str], MacroRound] = dataclasses.field(default_factory=dict)
    dead_code: DeadCode = DeadCode.ELIMINATE
    input_comment: Callable[[str, bool], str] = lambda value, scalar: "argument"


@dataclasses.dataclass(frozen=True)
class Unit:
    """What a rerolling strategy sees of one block: the block, its target, its instructions, and
    each instruction's macro origin."""

    block: Block
    target: Target
    instructions: tuple[Instruction, ...]
    origins: tuple[rust.MacroOrigin | None, ...]

    def make(
        self,
        signature: Signature,
        doc: str,
        nodes: list[Node],
        results: list[str],
        **kw: Any,
    ) -> Program:
        """A program of this block's transcription."""
        target = self.target
        return target.conventions.program(
            target.isa.name, signature, doc, nodes, results, dead_code=target.dead_code, **kw
        )

    @property
    def limb_args(self) -> list[str]:
        """The block's structured arguments, in signature order."""
        return [arg for arg, kind in self.block.args if self.target.kinds[kind]]


def parse(source: str, block: Block, target: Target) -> ParsedFunction:
    """The front end's reading of `block`'s Rust function, under the instruction set's rules."""
    isa = target.isa
    return rust.parse_function(
        source,
        block.rust_name,
        block.arg_names,
        len(target.kinds[block.result] or [None]),
        reserved_names=set(isa.reserved_names),
        fixed_registers=set(isa.fixed_registers),
        const_operands=set(target.consts),
        allowed_options=set(isa.allowed_options),
        required_options=set(isa.required_options),
    )


def input_binding(
    declaration: Declaration, parsed: ParsedFunction, block: Block, target: Target
) -> Load | Scalar:
    """The node binding an input operand to its argument: a word argument itself, or a field of a
    structured one, read directly (`lhs[0]`) or through a local copied from one before the block
    (`let mut a0 = value[0];`)."""
    name, value = declaration.name, declaration.value
    kinds = {arg: target.kinds[kind] for arg, kind in block.args}
    arg: str
    index: int  # the field read: `arg[index]`
    direct = re.fullmatch(r"([A-Za-z_]\w*)\[([0-9]+)\]", value)
    if direct:
        arg, index = direct.group(1), int(direct.group(2))
    elif value in parsed.locals:
        arg, index = parsed.locals[value]
    elif value in kinds and kinds[value] is None:
        comment = target.input_comment(value, True)
        return Scalar(comment=comment, dest=name, name=value)
    else:
        raise GenerationError(f"unsupported input operand expression {value}")
    if arg not in kinds:
        raise GenerationError(f"operand {name} reads {value}, not an argument")
    fields = kinds[arg]
    if fields is None or index >= len(fields):
        raise GenerationError(f"operand {name} reads {value}, which {arg} does not have")
    comment = target.input_comment(value, False)
    return Load(comment=comment, dest=name, arg=arg, field=fields[index])


def lift(
    parsed: ParsedFunction, block: Block, target: Target
) -> tuple[Lifter, tuple[Instruction, ...]]:
    """Bind the operands and lift the instructions: the lifter, holding the flat program's
    nodes, and the block's instructions."""
    isa, fn = target.isa, block.rust_name
    lifter = isa.lifter(rust.declaration_directions(parsed, fn))
    isa.check_block(parsed, block)
    kinds = dict(block.args)
    for d in parsed.declarations:
        if d.fixed or d.kind == "out":
            continue
        if d.kind == "const":
            value = lit(target.consts[d.value])
            lifter.bind(Mov(comment=f"operand {d.name} = const {d.value}", dest=d.name, a=value))
        elif d.kind not in ("in", "inout"):
            raise GenerationError(f"{fn}: unsupported operand direction {d.kind}")
        elif pointer := re.fullmatch(r"([A-Za-z_]\w*)\.as_ptr\(\)", d.value):
            kind = kinds.get(pointer.group(1))
            fields = target.kinds.get(kind) if kind is not None else None
            if not fields:
                raise GenerationError(f"{fn}: pointer {d.name} is not a word argument")
            lifter.bind_pointer(d.name, pointer.group(1), fields)
        else:
            try:
                lifter.bind(input_binding(d, parsed, block, target))
            except GenerationError as error:
                raise GenerationError(f"{fn}: {error}") from error

    # Memory is read only through pointer operands, and the options say so.
    if lifter.pointers:
        if "readonly" not in parsed.options or "nomem" in parsed.options:
            raise GenerationError(f"{fn}: pointer operands require readonly and forbid nomem")
    elif "nomem" not in parsed.options or "readonly" in parsed.options:
        raise GenerationError(f"{fn}: register-only block requires nomem and forbids readonly")

    instructions = tuple(lifter.parse_instruction(text) for text in parsed.instructions)
    for pc, ins in enumerate(instructions):
        lifter.lift(ins, pc)
    if lifter.used_memory != bool(lifter.pointers):
        raise GenerationError(f"{fn}: declared pointer operands and memory reads disagree")
    return lifter, instructions


def transcribe(source: str, block: Block, target: Target) -> list[Program]:
    """The programs of one block: the block itself, preceded by its round definitions when its
    rounds reroll, and checked to compute what the flat block computes."""
    parsed = parse(source, block, target)
    lifter, instructions = lift(parsed, block, target)
    results = list(rust.returned_registers(parsed, block.rust_name))
    for register in results:
        if register not in lifter.known:
            raise GenerationError(
                f"{block.rust_name}: output register {register} was never written"
            )
    unit = Unit(block, target, instructions, parsed.origins)
    flat = unit.make(
        Signature(block.lean_name, block.args, block.result),
        block.doc,
        lifter.nodes,
        results,
        arg_fields={arg: target.kinds[kind] for arg, kind in block.args if target.kinds[kind]},
    )
    strategy: Reroller | None = target.rerollers.get(block.lean_name)
    if strategy is None and any(parsed.origins):
        strategy = MacroRounds(target.macro_rounds)
    if strategy is None:
        return [flat]
    programs = strategy.reroll(flat, unit)
    validate.check(flat, programs)
    return programs
