"""What every architecture's configuration module shares: transcribing its blocks in order, and
printing their definitions. Each module of `architectures/` adds its own header, module docstring,
and comment-column policy."""

from collections.abc import Sequence

from asm2lean import lean
from asm2lean.block import Block, Target, transcribe
from asm2lean.ir import Program


def programs(source: str, blocks: Sequence[Block], target: Target) -> list[Program]:
    """Every program of the transcription of `blocks` from `source`, in order: each block's
    rounds, then the block."""
    return [program for block in blocks for program in transcribe(source, block, target)]


def definitions(programs: Sequence[Program], column: int | None = None) -> str:
    """The programs' Lean definitions, their instruction comments at `column`, or, when it is
    `None`, each definition's at its own column."""
    return "\n".join(
        lean.definition(p, lean.comment_column([p]) if column is None else column) for p in programs
    )
