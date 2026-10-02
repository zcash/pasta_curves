"""Back end: the mechanical part of a program's correctness proof.

The skeleton is symbolic execution of the program inside a Lean proof. The theorem's hypothesis
`hr : r = <program> ...` is unfolded and its `let`s lifted with merging off, so that every binding
is its own `let`; then, one node at a time, a `word_step` (the tactic of
`PastaCurves/Tactic/WordStep.lean`) extracts the node's `let`s from `hr` under fresh names,
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
inside `hr`, so that `clear_value` has one hypothesis to revert and re-check. With every let
extracted up front, each `clear_value` re-checks all the later locals and equations, which is
quadratic in the chain's length and exhausted the heartbeat budget on the multiplication
routine's 264 locals.

Each node knows its own step (`Node.prove`); this module only walks the program and keeps the
state the steps share.
"""

import re

from .ir import LITERAL
from .lean import wrap_words

# An argument's limb read directly as an operand, as a memory read of x86-64 produces it.
ARGUMENT_FIELD = re.compile(r"(\w+)\.(l[0-7])")


def ssa_names(lets):
    """Unique names: the first binding of a register keeps its name, later ones get `_1`,
    `_2`, ...; a binding `x := x` is primed."""
    counts, names = {}, []
    for let in lets:
        n = counts.get(let.name, 0)
        counts[let.name] = n + 1
        base = let.name + "'" if let.expr == let.name else let.name
        names.append(base if n == 0 else f"{base}_{n}")
    return names


def projection(fields, field):
    """The projection of a `Bounded` conjunction over `fields` that bounds `field`."""
    i = fields.index(field)
    return ".".join(["2"] * i + (["1"] if i < len(fields) - 1 else []))


class State:
    """What the steps share while walking one program."""

    def __init__(self, program):
        self.program = program
        self.ren = {}  # register -> its current SSA name
        self.bnd = {}  # SSA name -> the fact bounding it below 2^64 (or a carry by 1)
        self.narrow = set()  # `lsr` results, bounded below 2^(64 - k)
        self.unit_bound = set()  # carries, bounded by 1
        self.products = {}  # (a, b) -> the `mulLo` result awaiting its `umulh`


class Step:
    """One node's step: its kept `let`s under their SSA names, the equations and bounds the
    `word_step` states, and the lines that follow it."""

    def __init__(self, state, node, lets, names):
        self.state, self.node, self.lets, self.names = state, node, lets, names
        self.values, self.bounds, self.lines = {}, {}, []

    @property
    def name(self):
        return self.names[0]

    @property
    def inv_bound(self):
        return self.state.program.inv_bound

    def r(self, operand):
        """An operand as the node wrote it, under its current SSA name."""
        return self.state.ren.get(operand, operand)

    def define(self, index):
        """Make the node's `index`-th binding the current one for its register."""
        self.state.ren[self.lets[index].name] = self.names[index]

    def eq(self, name, value):
        self.values[name] = value

    def bound(self, name, proof):
        """`proof : v < 2^64` for the value `v` of `name`; the step proves `b_{name}`."""
        self.bounds[name] = proof
        self.state.bnd[name] = f"b_{name}"

    def unit(self, name):
        """`name` is a carry: bounded by `b_{name} : name ≤ 1`."""
        self.state.bnd[name] = f"b_{name}"
        self.state.unit_bound.add(name)

    def line(self, text):
        self.lines.append(text)

    def argument_bound(self, arg, field):
        """The theorem's `Bounded` hypothesis on argument `arg`, projected to `field`."""
        program = self.state.program
        return f"{program.bound_hyp(arg)}.{projection(program.arg_fields[arg], field)}"

    def lt64(self, operand):
        """A proof that the operand is below `2^64`."""
        if LITERAL.fullmatch(operand):
            return "(by decide)"
        field = ARGUMENT_FIELD.fullmatch(operand)
        if field and field.group(1) in self.state.program.arg_fields:
            return self.argument_bound(*field.groups())
        if operand in self.state.narrow or operand in self.state.unit_bound:
            return f"(lt_of_lt_of_le {self.state.bnd[operand]} (by norm_num))"
        return self.state.bnd[operand]

    def le1(self, operand):
        """A proof that the carry operand is at most `1`."""
        return "(by decide)" if LITERAL.fullmatch(operand) else self.state.bnd[operand]


def skeleton(program):
    """The generated part of the correctness proof of `program`, as lines."""
    state = State(program)
    kept = [(node, let) for node, let, live in program.kept() if live]
    names = ssa_names([let for _, let in kept])
    groups = []  # (node, [let], [name]): the kept bindings of each node, in order
    for (node, let), name in zip(kept, names, strict=True):
        if groups and groups[-1][0] is node:
            groups[-1][1].append(let)
            groups[-1][2].append(name)
        else:
            groups.append((node, [let], [name]))
    out = [
        f"  -- generated skeleton for `{program.name}`: do not edit between the annotations",
        f"  unfold {program.name} at hr",
        "  lift_lets -merge at hr",
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
    out.append("  subst hr")
    return out
