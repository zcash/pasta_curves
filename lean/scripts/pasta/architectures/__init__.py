"""The crate's transcribed architectures: one configuration module each, in this package, listed
here in output order.

A new architecture is a module of this package (its blocks, its `Target`, its output path), listed
here, and its instruction set a module of `asm2lean/isa/`; `gen.py` reads only this list."""

from pathlib import Path
from typing import Protocol

from asm2lean.block import Target
from asm2lean.ir import Program

from . import aarch64, x86_64


class Architecture(Protocol):
    """What `gen.py` needs of an architecture's configuration module."""

    OUTPUT: Path
    TARGET: Target

    def programs(self, source: str | None = None) -> list[Program]: ...

    def text(self, source: str | None = None) -> str: ...


ARCHITECTURES: tuple[Architecture, ...] = (aarch64, x86_64)
