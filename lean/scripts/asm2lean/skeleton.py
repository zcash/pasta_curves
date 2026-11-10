"""Back end: the mechanical part of a program's correctness proof.

The skeleton is symbolic execution of the program inside a Lean proof. The theorem's hypothesis
`hres : r = <program> ...` is unfolded and its `let`s lifted with merging off, so that every binding
is its own `let`; then, one node at a time, a `word_step` (the tactic of
`PastaCurves/Tactic/WordStep.lean`) extracts the node's `let`s from `hres` under fresh names,
records their defining equations (by `rfl`, in `%`/`/` form), makes the locals opaque when the
program asks for it, and proves the bound `b_x : x < 2^64` of each result whose bound is one lemma
instance (`x := v using proof`). The lines after the step derive the node's other facts from the
equations (a carry-chain equation, a carry's bound by `1`, a narrower bound, a product's
decomposition), each an instance of one lemma, and clear the equations that later steps do not
need. Every fact is linear, so the hand-written annotations close the proof with `omega`.

The names are the program's static single assignment form: the first binding of a register keeps
its name, later ones get `_1`, `_2`, ...; an argument bound under its own name (`inv := inv`) is
primed, so that extracting it does not shadow the theorem's variable.

Extracting one node at a time (`extract_lets +onlyGivenNames`) keeps the rest of the chain folded
inside `hres`, so that `clear_value` has one hypothesis to revert and re-check. With every let
extracted up front, each `clear_value` re-checks all the later locals and equations, which is
quadratic in the chain's length and exhausted the heartbeat budget on the multiplication
routine's 264 locals.

Each node knows its own step (`Node.prove`); this module only walks the program and keeps the
state the steps share.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar, cast

from .ir import Arg, Get, Let, Lit, Node, Program, Reg, ScratchKey, Term, rename_term, show
from .lean import wrap_words
from .rust import GenerationError

T = TypeVar("T")


def ssa_names(lets: Sequence[Let]) -> list[str]:
    """Unique names: the first binding of a register keeps its name, later ones get `_1`,
    `_2`, ...; a binding `x := x` is primed. A suffixed name skips any name already taken, or
    held by a register's first binding (registers `t` and `t_1`: `t`'s second binding is `t_2`)."""
    counts: dict[str, int] = {}
    taken = {let.name for let in lets}
    names: list[str] = []
    for let in lets:
        base = let.name + "'" if let.term == Arg(let.name) else let.name
        n = counts.get(let.name, 0)
        name = base
        if n:
            while f"{base}_{n}" in taken:
                n += 1
            name = f"{base}_{n}"
        counts[let.name] = n + 1
        if name in names:
            raise GenerationError(f"two bindings named {name}")
        taken.add(name)
        names.append(name)
    return names


def projection(fields: Sequence[str], field: str) -> str:
    """The projection of a `Bounded` conjunction over `fields` that bounds `field`."""
    i = fields.index(field)
    return ".".join(["2"] * i + (["1"] if i < len(fields) - 1 else []))


class State:
    """What the steps share while walking one program."""

    def __init__(self, program: Program) -> None:
        self.program = program
        self.ren: dict[str, str] = {}  # register -> its current SSA name
        self.bnd: dict[str, str] = {}  # SSA name -> the fact bounding it below 2^64 (or by 1)
        self.narrow: set[str] = set()  # `lsr` results, bounded below 2^(64 - k)
        self.unit_bound: set[str] = set()  # carries, bounded by 1
        self._scratch: dict[ScratchKey[Any], object] = {}

    def scratch(self, key: ScratchKey[T]) -> T:
        """State that nodes share across steps, of the kind `key` names (the low and high words
        of a product, a sum awaiting its carry), created on first use: an instruction set adds
        such state by defining a key, without changing this class."""
        if key not in self._scratch:
            self._scratch[key] = key.factory()
        # Keys compare by identity, and `key` created this entry: its value is a `T`.
        return cast(T, self._scratch[key])


class Step:
    """One node's step: its kept `let`s under their SSA names, the equations and bounds the
    `word_step` states, and the lines that follow it."""

    def __init__(self, state: State, node: Node, lets: list[Let], names: list[str]) -> None:
        self.state, self.node, self.lets, self.names = state, node, lets, names
        self.values: dict[str, str] = {}
        self.bounds: dict[str, str] = {}
        self.lines: list[str] = []

    @property
    def name(self) -> str:
        return self.names[0]

    def r(self, operand: Term | str) -> str:
        """The Lean text of an operand under the registers' current SSA names: a term, or the
        name of a register (a call's state, a projection's call)."""
        if isinstance(operand, str):
            return self.state.ren.get(operand, operand)
        return show(rename_term(operand, lambda r: self.state.ren.get(r, r)))

    def define(self, index: int) -> None:
        """Make the node's `index`-th binding the current one for its register."""
        self.state.ren[self.lets[index].name] = self.names[index]

    def eq(self, name: str, value: str) -> None:
        self.values[name] = value

    def bound(self, name: str, proof: str) -> None:
        """`proof : v < 2^64` for the value `v` of `name`; the step proves `b_{name}`."""
        self.bounds[name] = proof
        self.state.bnd[name] = f"b_{name}"

    def unit(self, name: str) -> None:
        """`name` is a carry: bounded by `b_{name} : name ≤ 1`."""
        self.state.bnd[name] = f"b_{name}"
        self.state.unit_bound.add(name)

    def line(self, text: str) -> None:
        self.lines.append(text)

    def argument_bound(self, arg: str, field: str) -> str:
        """The theorem's `Bounded` hypothesis on argument `arg`, projected to `field`."""
        program = self.state.program
        return f"{program.bound_hyp(arg)}.{projection(program.arg_fields[arg], field)}"

    def lt64(self, operand: Term) -> str:
        """A proof that the operand is below `2^64`: a literal by evaluation, an argument's field
        by the theorem's hypothesis, a register by its fact."""
        match operand:
            case Lit():
                return "(by decide)"
            case Get(Arg(arg), field) if arg in self.state.program.arg_fields:
                return self.argument_bound(arg, field)
            case Reg(register):
                name = self.r(register)
                if name in self.state.narrow or name in self.state.unit_bound:
                    return f"(lt_of_lt_of_le {self.state.bnd[name]} (by norm_num))"
                return self.state.bnd[name]
        raise GenerationError(f"no bound for the operand {operand!r}")

    def le1(self, operand: Term) -> str:
        """A proof that the carry operand is at most `1`."""
        match operand:
            case Lit():
                return "(by decide)"
            case Reg(register):
                return self.state.bnd[self.r(register)]
        raise GenerationError(f"no carry bound for the operand {operand!r}")


def skeleton(program: Program) -> list[str]:
    """The generated part of the correctness proof of `program`, as lines."""
    state = State(program)
    kept = [(node, let) for node, let, live in program.kept() if live]
    names = ssa_names([let for _, let in kept])
    # (node, [let], [name]): the kept bindings of each node, in order
    groups: list[tuple[Node, list[Let], list[str]]] = []
    for (node, let), name in zip(kept, names, strict=True):
        if groups and groups[-1][0] is node:
            groups[-1][1].append(let)
            groups[-1][2].append(name)
        else:
            groups.append((node, [let], [name]))
    out = [
        f"  -- generated skeleton for `{program.name}`: do not edit between the annotations",
        f"  unfold {program.name} at hres",
        "  lift_lets -merge at hres",
    ]
    head = "word_step" if program.clear_values else "word_step -clear"
    for node, lets, group in groups:
        step = Step(state, node, lets, group)
        node.prove(step)
        if len(lets) == 1:
            step.define(0)
        out.append(f"  -- {node.label(group)}: {lets[0].comment}")
        items = []
        for name in group:
            item = name
            if name in step.values:
                item += f" := {step.values[name]}"
            if name in step.bounds:
                item += f" using {step.bounds[name]}"
            items.append(item)
        out += wrap_words(head, [f"{item}," for item in items[:-1]] + items[-1:], "")
        out += step.lines
    out.append("  subst hres")
    return out
