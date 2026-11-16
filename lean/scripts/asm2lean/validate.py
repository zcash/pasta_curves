"""Translation validation: a rerolled transcription computes what the flat one computes.

Rerolling (`reroll.py`) is the inverse of inlining: it folds repeated instructions into a round and
replaces each repetition by a call. Each strategy checks what it can see locally, but the property
the proofs rely on is global: the rerolled block, with every call expanded, returns the same
values as the block the lifter produced. This module checks exactly that, once, for every
strategy and every instruction set.

The check evaluates both programs symbolically in the free term algebra over the Lean operations,
treated as uninterpreted. Every `let`'s right-hand side is a term (`ir.py`); folding it with each
register replaced by the number of the term that register holds and interning each node
(hash-consing) gives equal terms equal numbers: this is global value numbering. A copy
(`x := y`) adds no term, a call evaluates the round's body on the call's arguments, `n` times for
`round^[n]`, as does an application of a helper factored out of a block, and a projection selects
a field of the call's result. The two programs are equivalent when their result tuples are the same
numbers.

Equal terms in the free algebra are equal in every interpretation of the operations, so the check
needs no semantics of the instructions: it is sound for the Lean semantics whatever they are, and
for any instruction set. It is complete for rerolling, which only renames registers and moves
instructions into a definition; it does not see through algebraic identities, which rerolling
never needs. Its cost is linear in the size of the expanded programs.

References: A. Pnueli, M. Siegel, E. Singerman, *Translation Validation*, TACAS 1998; G. Necula,
*Translation Validation for an Optimizing Compiler*, PLDI 2000; B. Alpern, M. Wegman, F. Zadeck,
*Detecting Equality of Variables in Programs*, POPL 1988 (value numbering).
"""

from __future__ import annotations

import dataclasses

from .ir import App, Arg, Call, Get, Lit, Program, Reg, Term, Tuple, subterms
from .rust import GenerationError


@dataclasses.dataclass(frozen=True)
class Record:
    """A value of a round's state structure: its fields, and the term number of each."""

    fields: tuple[str, ...]
    values: tuple[int, ...]

    def get(self, field: str) -> int:
        return self.values[self.fields.index(field)]


# What a register or parameter holds during evaluation: a term number, or a structure's fields.
Value = int | Record


class Evaluator:
    """Evaluates programs to term numbers, expanding the calls of the `rounds` (by name): a
    round's `Call`, or an application of any of them by name to its arguments, as a helper
    factored out of a block is applied."""

    def __init__(self, rounds: dict[str, Program]) -> None:
        self.rounds = rounds
        self.ids: dict[tuple[object, ...], int] = {}

    def intern(self, key: tuple[object, ...]) -> int:
        return self.ids.setdefault(key, len(self.ids))

    def run(self, program: Program, params: dict[str, Value] | None = None) -> tuple[int, ...]:
        """The terms of `program`'s results. `params` maps each parameter to its argument's
        value: a term number, or a `Record` for a state structure. Without `params` every
        argument is free."""
        return tuple(self.number(v) for v in self.values(program, params))

    def values(self, program: Program, params: dict[str, Value] | None = None) -> tuple[Value, ...]:
        """The values of `program`'s results, which may be structures."""
        params = params or {}
        env: dict[str, Value] = {}  # register -> a term number, or a call's `Record`
        for node in program.nodes:
            if isinstance(node, Call):
                env[node.dest] = self.call(node, env, params)
            else:
                for let in node.lets():
                    env[let.name] = self.value(let.term, env, params)
        return tuple(env[r] for r in program.results)

    def number(self, value: Value) -> int:
        """A value that must be a word's term, not a structure."""
        if isinstance(value, Record):
            raise GenerationError("a structure where a word was expected")
        return value

    def value(self, term: Term, env: dict[str, Value], params: dict[str, Value]) -> Value:
        """The term number of `term`: a fold over it, interning each node. A register or an
        argument is its value, so copies add no term."""
        match term:
            case Reg(name):
                if name not in env:
                    raise GenerationError(f"register {name} read before it is bound")
                return env[name]
            case Arg(name):
                # Lean resolves a bare name to its latest `let`, so a register of this name, if
                # any, is what the text reads (and `Program` rejects one that would differ).
                if name in env:
                    return env[name]
                return params[name] if name in params else self.intern(("arg", name))
            case Lit(text):
                return self.intern(("lit", text))
            case Get(base, field):
                value = self.value(base, env, params)
                if isinstance(value, Record):
                    return value.get(field)
                return self.intern(("get", value, field))
            case App(function, args) if function in self.rounds:
                return self.apply(self.rounds[function], args, env, params)
            case App(function, args):
                values = (self.number(self.value(a, env, params)) for a in args)
                return self.intern(("app", function, *values))
            case Tuple():
                raise GenerationError(f"cannot evaluate {term!r}")
        # Any other term is an uninterpreted operation on its subterms, keyed by its own data.
        parts = tuple(self.number(self.value(t, env, params)) for t in subterms(term))
        return self.intern((type(term).__name__, _label(term), *parts))

    def apply(
        self,
        program: Program,
        args: tuple[Term, ...],
        env: dict[str, Value],
        params: dict[str, Value],
    ) -> Value:
        """`program` applied to `args`, one per parameter: an anonymous constructor binds a
        structure parameter's fields. Its value is its one result, or its results as a structure:
        the fields of its state structure, or a pair's `1` and `2`."""
        names = program.signature.names
        if len(args) != len(names):
            raise GenerationError(f"application of `{program.name}` to {len(args)} arguments")
        inner: dict[str, Value] = {}
        for name, arg in zip(names, args, strict=True):
            if isinstance(arg, Tuple):
                fields = program.arg_fields.get(name, ())
                if len(fields) != len(arg.items):
                    raise GenerationError(
                        f"`{program.name}`: {name} does not take {len(arg.items)} fields"
                    )
                items = tuple(self.number(self.value(i, env, params)) for i in arg.items)
                inner[name] = Record(fields, items)
            else:
                inner[name] = self.value(arg, env, params)
        results = self.values(program, inner)
        if program.state_fields:
            return Record(program.state_fields, tuple(self.number(r) for r in results))
        if len(results) == 1:
            return results[0]
        if len(results) == 2:
            return Record(("1", "2"), tuple(self.number(r) for r in results))
        raise GenerationError(f"`{program.name}` returns {len(results)} values and no structure")

    def call(self, node: Call, env: dict[str, Value], params: dict[str, Value]) -> Record:
        round_ = self.rounds.get(node.round)
        if round_ is None:
            raise GenerationError(f"call of `{node.round}`, which is not a round of this block")
        if not round_.signature.names:
            raise GenerationError(f"`{node.round}` takes no state, so it is not a round")
        *words, state = round_.signature.names
        *args, tuple_ = node.term().args
        if len(args) != len(words) or not isinstance(tuple_, Tuple):
            raise GenerationError(
                f"call of `{node.round}` does not pass one argument per parameter"
            )
        inner = {p: self.value(a, env, params) for p, a in zip(words, args, strict=True)}
        values = tuple(self.number(self.value(i, env, params)) for i in tuple_.items)
        for _ in range(node.count):
            inner[state] = Record(round_.state_fields, values)
            values = self.run(round_, inner)
        return Record(round_.state_fields, values)


def _label(term: Term) -> str:
    """What distinguishes a term from another of its class with the same subterms."""
    return str(getattr(term, "op", "") or getattr(term, "name", "") or getattr(term, "type", ""))


def applied(program: Program) -> set[str]:
    """The functions a program's terms apply."""
    functions: set[str] = set()

    def walk(term: Term) -> None:
        if isinstance(term, App):
            functions.add(term.function)
        for t in subterms(term):
            walk(t)

    for _, let in program.bindings():
        walk(let.term)
    return functions


def check(flat: Program, programs: list[Program]) -> None:
    """Require the rerolled `programs` (rounds first, the block last) to return what `flat`
    returns, as terms."""
    *rounds, block = programs
    # A round is expanded wherever it is applied by name, so its name must not be that of an
    # operation the flat block applies: that operation would be read as a call of the round.
    names = {r.name for r in rounds}
    captured = sorted(names & applied(flat))
    if captured:
        raise GenerationError(
            f"{block.name}: rounds {captured} are named like operations the flat block applies"
        )
    evaluator = Evaluator({r.name: r for r in rounds})
    expected, got = evaluator.run(flat), evaluator.run(block)
    if expected != got:
        differing = [r for r, a, b in zip(block.results, expected, got, strict=True) if a != b]
        raise GenerationError(
            f"{block.name}: the rerolled block does not compute the flat block's results: "
            f"{', '.join(differing)} differ"
        )
