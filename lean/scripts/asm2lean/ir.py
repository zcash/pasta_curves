"""The intermediate representation: straight-line programs in A-normal form.

A program is a sequence of nodes, each one operation of an instruction set's Lean semantics
applied to registers and literals, and a tuple of result registers. A node binds one or more
`let`s (`Let`): every intermediate value is named, and a register that an instruction writes again
is rebound, shadowing the old binding. That is also the form of the generated Lean. A binding's
right-hand side is a term (`Reg`, `Lit`, `Arg`, `Get`, `App`, `Tuple`), data rather than text: the
Lean text is one printing of it (`show`), and any other interpretation is a fold over it.

A node is the unit of everything downstream:

- the transcription prints its `lets`;
- liveness reads their `reads`, the registers of their terms;
- translation validation evaluates their terms (`validate.py`);
- loop rerolling renames it (`renamed`) and compares it up to renaming (`key`);
- the proof skeleton asks it for its proof step (`prove`), one `word_step` per node.

So an instruction set adds node classes and nothing else changes: this module holds the nodes
that every instruction set shares (arguments, moves, the two words of a product, shifts, the word
operations of `PastaCurves/Semantics.lean`, selection, calls), and each instruction set's file in
`isa/` holds the rest.

Liveness is the backward dataflow analysis on the `let`s; on straight-line code one backward pass
reaches the fixed point.

References: C. Flanagan, A. Sabry, B. Duba, M. Felleisen, *The Essence of Compiling with
Continuations*, PLDI 1993 (A-normal form); A. Appel, *Modern Compiler Implementation*, ch. 10
(liveness).
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

from .rust import GenerationError

if TYPE_CHECKING:
    from .skeleton import Step

MOD_LT = "Nat.mod_lt _ (Nat.two_pow_pos _)"

T = TypeVar("T")


@dataclasses.dataclass(frozen=True, eq=False)
class ScratchKey(Generic[T]):
    """A kind of state that nodes share across proof steps (`skeleton.State.scratch`): its name,
    for messages, and how to create it on first use. Its type parameter is the state's type. Keys
    compare by identity, so two keys never share state, whatever their names."""

    name: str
    factory: Callable[[], T]


# Each `mulLo` result, by its operands, for the node computing the high word of the same product.
PRODUCTS: ScratchKey[dict[tuple[str, str], str]] = ScratchKey("products", dict)


# -- terms ---------------------------------------------------------------------------------------
#
# The right-hand side of a binding is a term over the Lean semantics' operations: data, not text.
# The transcription prints it (`show`), liveness reads its registers (`term_registers`), and every
# other interpretation (the translation validator, a future interpreter) is a fold over it, so no
# pass ever parses Lean back.


@dataclasses.dataclass(frozen=True)
class Reg:
    """A register: the value of the latest binding of `name`."""

    name: str


@dataclasses.dataclass(frozen=True)
class Lit:
    """A literal, as its text: decimal, or hex as the assembly wrote it."""

    text: str


@dataclasses.dataclass(frozen=True)
class Arg:
    """An argument of the definition, by name: a word, or a structure whose fields `Get` reads."""

    name: str


@dataclasses.dataclass(frozen=True)
class Get:
    """A projection: a structure's field (`lhs.l0`, `st.f`), or a pair's component (`s.1`)."""

    base: Term
    field: str


@dataclasses.dataclass(frozen=True)
class App:
    """An operation applied to its arguments: `function a b ...`."""

    function: str
    args: tuple[Term, ...]


@dataclasses.dataclass(frozen=True)
class Tuple:
    """An anonymous constructor, `⟨a, b, ...⟩`."""

    items: tuple[Term, ...]


# A term: data the transcription prints and every other interpretation folds over.
Term = Reg | Lit | Arg | Get | App | Tuple
# An operand of `apply`: a term the lifter built, or an integer immediate.
Operand = Term | int


def lit(value: int) -> Lit:
    """A word literal, in `[0, 2^64)`; a negative one has no word: write the two's complement it
    encodes."""
    if value < 0:
        raise GenerationError(f"negative literal {value}: a word is its two's complement")
    if value >= 2**64:
        raise GenerationError(f"literal {value} does not fit a word")
    return Lit(str(value))


def apply(function: str, *operands: Operand) -> App:
    """`function` applied to the operands: terms, or integers (a shift amount, a bit index)."""
    return App(function, tuple(lit(o) if isinstance(o, int) else o for o in operands))


def show(term: Term, nested: bool = False) -> str:
    """The Lean text of a term; an application inside another term is parenthesized."""
    match term:
        case Reg(name) | Arg(name):
            return name
        case Lit(text):
            return text
        case Get(base, field):
            return f"{show(base, nested=True)}.{field}"
        case App(function, args):
            text = " ".join([function, *(show(a, nested=True) for a in args)])
            return f"({text})" if nested else text
        case Tuple(items):
            return f"⟨{', '.join(show(i) for i in items)}⟩"
    raise TypeError(f"not a term: {term!r}")


def term_registers(term: Term) -> frozenset[str]:
    """The registers a term reads."""
    match term:
        case Reg(name):
            return frozenset({name})
        case Get(base, _):
            return term_registers(base)
        case App(_, items) | Tuple(items):
            return frozenset[str]().union(*(term_registers(i) for i in items))
    return frozenset()


def term_args(term: Term) -> frozenset[str]:
    """The arguments a term reads, by name: `Arg`s, and the bases of their fields."""
    match term:
        case Arg(name):
            return frozenset({name})
        case Get(base, _):
            return term_args(base)
        case App(_, items) | Tuple(items):
            return frozenset[str]().union(*(term_args(i) for i in items))
    return frozenset()


def rename_term(term: Term, rename: Callable[[str], str]) -> Term:
    """The term with each register `r` replaced by the register `rename(r)`."""
    match term:
        case Reg(name):
            return Reg(rename(name))
        case Get(base, field):
            return Get(rename_term(base, rename), field)
        case App(function, args):
            return App(function, tuple(rename_term(a, rename) for a in args))
        case Tuple(items):
            return Tuple(tuple(rename_term(i, rename) for i in items))
    return term


class Dead(enum.Enum):
    """What a binding that nothing reads becomes in a transcription that drops dead bindings."""

    ERROR = "error"  # a computed value: dead code in the block, rejected
    COMMENT = "comment"  # an argument, or an optional call output: left as a comment
    DROP = "drop"  # a flag write: an ordinary unread flag


@dataclasses.dataclass(frozen=True)
class Let:
    """One `let name := term`. `comment` is the instruction it models; `note`, when given,
    replaces it in the transcription, for a binding that continues the line above."""

    name: str
    term: Term
    comment: str | None
    note: str | None = None
    dead: Dead = Dead.ERROR

    @property
    def expr(self) -> str:
        """The Lean text of the right-hand side."""
        return show(self.term)

    @property
    def reads(self) -> frozenset[str]:
        return term_registers(self.term)

    @property
    def trailing(self) -> str | None:
        return self.note or self.comment


def src() -> Any:
    """A field holding what a node reads: a term (or a tuple of terms). Only a call's state and a
    projection's call, which name a binding rather than read a value, are register names."""
    return dataclasses.field(metadata={"operand": "read"})


def dst() -> Any:
    """A field holding a register that a node writes."""
    return dataclasses.field(metadata={"operand": "write"})


@dataclasses.dataclass(frozen=True, kw_only=True)
class Node:
    """One operation. `comment` is the source instruction (or what binds an argument), `pc`
    the index of that instruction, `None` for a node with no instruction of its own."""

    comment: str | None = None
    pc: int | None = None
    # The registers this node writes as the source wrote them, by field, once it has been renamed:
    # a continuation note names the register of the source line it continues.
    written_as: tuple[tuple[str, str], ...] = ()

    def lets(self) -> list[Let]:
        raise NotImplementedError

    def source_name(self, field: str) -> str:
        """The register in `field` as the source wrote it."""
        value = getattr(self, field)
        assert isinstance(value, str), f"{field} is not a register field"
        return dict(self.written_as).get(field, value)

    def label(self, names: list[str]) -> str:
        """The register named in the skeleton's marker for this node's step."""
        return names[0]

    def prove(self, step: Step) -> None:
        """Emit this node's proof step (see `skeleton.Step`)."""
        raise ValueError(f"unsupported skeleton fact {type(self).__name__}")

    def renamed(
        self,
        mapping: Mapping[str, str],
        writes: bool = True,
        terms: Mapping[Term, Term] | None = None,
    ) -> Node:
        """The node with the registers it reads renamed by `mapping`, and those it writes too if
        `writes`; an operand equal to a key of `terms` is replaced by its value first."""
        substitution = terms or {}

        def operand(value: object) -> object:
            """A field's value renamed: a register name, a term, or a tuple of either."""
            if isinstance(value, str):
                return mapping.get(value, value)
            if isinstance(value, tuple):
                return tuple(operand(v) for v in value)
            if isinstance(value, (Reg, Lit, Arg, Get, App, Tuple)):
                return rename_term(substitution.get(value, value), lambda r: mapping.get(r, r))
            return value

        changes: dict[str, Any] = {}
        if writes and not self.written_as:
            changes["written_as"] = tuple(
                (f.name, getattr(self, f.name))
                for f in dataclasses.fields(self)
                if f.metadata.get("operand") == "write"
            )
        for f in dataclasses.fields(self):
            role = f.metadata.get("operand")
            if role is None or (role == "write" and not writes):
                continue
            changes[f.name] = operand(getattr(self, f.name))
        return dataclasses.replace(self, **changes)

    def key(self) -> tuple[object, ...]:
        """The node up to its comment and position: what two rounds must agree on."""
        return (type(self).__name__,) + tuple(
            getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name not in ("comment", "pc", "written_as")
        )

    def writes(self) -> list[str]:
        return [let.name for let in self.lets()]

    def implicit_writes(self) -> frozenset[str]:
        """The names this node writes that no `dst()` field holds: an instruction set's fixed
        names (a flag, a pair). `renamed` cannot rename them, so a renaming must fix them."""
        named = {
            getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.metadata.get("operand") == "write"
        }
        return frozenset(self.writes()) - named

    def _let(
        self,
        name: str,
        term: Term,
        note: str | None = None,
        comment: str | None = None,
        dead: Dead = Dead.ERROR,
    ) -> Let:
        """A binding of this node, commented with its instruction unless `comment` is given."""
        return Let(name, term, comment if comment is not None else self.comment, note, dead)


# -- arguments -------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, kw_only=True)
class Load(Node):
    """A field of a structured argument: `dest := arg.field`."""

    dest: str = dst()
    arg: str
    field: str

    def lets(self) -> list[Let]:
        return [self._let(self.dest, Get(Arg(self.arg), self.field), dead=Dead.COMMENT)]

    def prove(self, step: Step) -> None:
        step.eq(step.name, f"{self.arg}.{self.field}")
        step.bound(step.name, step.argument_bound(self.arg, self.field))


@dataclasses.dataclass(frozen=True, kw_only=True)
class Scalar(Node):
    """A word argument: `dest := name`, bounded by the hypothesis the conventions name for it."""

    dest: str = dst()
    name: str

    def lets(self) -> list[Let]:
        return [self._let(self.dest, Arg(self.name))]

    def prove(self, step: Step) -> None:
        step.eq(step.name, self.name)
        step.bound(step.name, step.state.program.bound_hyp(self.name))


# -- the operations every instruction set shares ---------------------------------------------


@dataclasses.dataclass(frozen=True, kw_only=True)
class Mov(Node):
    dest: str = dst()
    a: Term = src()

    def lets(self) -> list[Let]:
        return [self._let(self.dest, self.a)]

    def prove(self, step: Step) -> None:
        a = step.r(self.a)
        step.eq(step.name, a)
        step.bound(step.name, step.lt64(self.a))


@dataclasses.dataclass(frozen=True, kw_only=True)
class MulLo(Node):
    """`mulLo a b`: the low word of the product."""

    dest: str = dst()
    a: Term = src()
    b: Term = src()

    def lets(self) -> list[Let]:
        return [self._let(self.dest, apply("mulLo", self.a, self.b))]

    def prove(self, step: Step) -> None:
        a, b = step.r(self.a), step.r(self.b)
        step.eq(step.name, f"{a} * {b} % 2^64")
        step.bound(step.name, MOD_LT)
        # The high word of the same product, if a node computes it, pairs with this one.
        step.state.scratch(PRODUCTS)[(a, b)] = step.name


@dataclasses.dataclass(frozen=True, kw_only=True)
class MulHi(Node):
    """The high word of the product, as the Lean function `function` spells it (AArch64's
    `umulh`). With the `mulLo` of the same operands, the two equations combine into
    `lo + 2^64 * hi = a * b`; without one, the low word is named as a ghost, so that later steps
    need no `%`."""

    dest: str = dst()
    function: str
    a: Term = src()
    b: Term = src()

    def lets(self) -> list[Let]:
        return [self._let(self.dest, apply(self.function, self.a, self.b))]

    def prove(self, step: Step) -> None:
        a, b, nm = step.r(self.a), step.r(self.b), step.name
        step.eq(nm, f"{a} * {b} / 2^64")
        step.bound(
            nm, f"Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' {step.lt64(self.a)} {step.lt64(self.b)})"
        )
        lo = step.state.scratch(PRODUCTS).pop((a, b), None)
        if lo is not None:
            step.line(f"  have d_{nm} : {lo} + 2^64 * {nm} = {a} * {b} := by")
            step.line(f"    rw [e_{lo}, e_{nm}]; exact Nat.mod_add_div _ _")
            step.line(f"  clear e_{lo} e_{nm}")
        else:
            step.line(f"  obtain ⟨lo_{nm}, b_lo_{nm}, d_{nm}⟩ :")
            step.line(f"      ∃ lo, lo < 2^64 ∧ lo + 2^64 * {nm} = {a} * {b} :=")
            step.line(f"    ⟨{a} * {b} % 2^64, {MOD_LT},")
            step.line(f"      by rw [e_{nm}]; exact Nat.mod_add_div _ _⟩")
            step.line(f"  clear e_{nm}")


def bounded(step: Step, value: str, proof: str) -> None:
    """The step's one binding is `value`, below `2^64` by `proof`."""
    step.eq(step.name, value)
    step.bound(step.name, proof)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Shl(Node):
    """`lsl a k`. For an amount the conventions name a split lemma for, the split fact is
    stated on the spot, with the high part as `a / 2^(64 - k)`, so that `omega` connects it with
    the matching `lsr`."""

    dest: str = dst()
    a: Term = src()
    k: int

    def lets(self) -> list[Let]:
        return [self._let(self.dest, apply("lsl", self.a, self.k))]

    def prove(self, step: Step) -> None:
        a, k, nm = step.r(self.a), self.k, step.name
        step.eq(nm, f"{a} * 2^{k} % 2^64")
        step.bound(nm, MOD_LT)
        lemma = step.state.program.conventions.shift_splits.get(k)
        if lemma is not None:
            step.line(f"  have sh_{nm} : {nm} + 2^64 * ({a} / 2^{64 - k}) = {a} * 2^{k} := by")
            step.line(f"    rw [e_{nm}]; exact {lemma} _")


@dataclasses.dataclass(frozen=True, kw_only=True)
class Shr(Node):
    """`lsr a k`, bounded below `2^(64 - k)`."""

    dest: str = dst()
    a: Term = src()
    k: int

    def lets(self) -> list[Let]:
        return [self._let(self.dest, apply("lsr", self.a, self.k))]

    def prove(self, step: Step) -> None:
        a, k, nm = step.r(self.a), self.k, step.name
        step.eq(nm, f"{a} / 2^{k}")
        step.line(f"  have b_{nm} : {nm} < 2^{64 - k} := by")
        step.line(
            f"    rw [e_{nm}]; exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq {step.lt64(self.a)} (by norm_num))"
        )
        step.state.bnd[nm] = f"b_{nm}"
        step.state.narrow.add(nm)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Word(Node):
    """A word operation `function args...` whose bound is the instance
    `<function>_lt args... proofs...` of a lemma, where `proofs` are the bounds of the arguments at
    the indices `bounds`. The definitions are in `Semantics.lean` (the shared `addw`, `subw`,
    `negw`, `extr`, `andw`, `orrw`, `eorw`, and an instruction set's own, such as AArch64's `madd`,
    `cselNe`, `cnegGe` or x86-64's `cmovz`), the lemmas in `Spec.lean`. An operation whose bound
    needs none of its arguments' (`addw`) has no `bounds`; a bitwise one (`andw a b`) needs both;
    a flag-dependent one (`cselNe fl a b`) needs its words'."""

    dest: str = dst()
    function: str
    args: tuple[Term, ...] = src()
    bounds: tuple[int, ...] = ()

    def lets(self) -> list[Let]:
        return [self._let(self.dest, apply(self.function, *self.args))]

    def prove(self, step: Step) -> None:
        args = " ".join(step.r(a) for a in self.args)
        proofs = "".join(f" {step.lt64(self.args[i])}" for i in self.bounds)
        bounded(step, f"{self.function} {args}", f"{self.function}_lt {args}{proofs}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class Asr(Node):
    """`asr a k`: the arithmetic shift right."""

    dest: str = dst()
    a: Term = src()
    k: int

    def lets(self) -> list[Let]:
        return [self._let(self.dest, apply("asr", self.a, self.k))]

    def prove(self, step: Step) -> None:
        a = step.r(self.a)
        bounded(step, f"asr {a} {self.k}", f"asr_lt {a} {self.k} {step.lt64(self.a)}")


class FlagState(enum.Enum):
    """The state of the flag in which a `Select` takes its first operand."""

    CLEAR = "clear"
    SET = "set"


@dataclasses.dataclass(frozen=True, kw_only=True)
class Select(Node):
    """`x` when `flag` is in the state `x_when`, else `y`, as the Lean function `function`
    spells it (`function flag x y`); in the skeleton, `if flag = 0 then if_zero else
    if_nonzero`."""

    dest: str = dst()
    function: str
    flag: Term = src()
    x: Term = src()
    y: Term = src()
    x_when: FlagState

    def lets(self) -> list[Let]:
        return [self._let(self.dest, apply(self.function, self.flag, self.x, self.y))]

    def prove(self, step: Step) -> None:
        clear = self.x_when is FlagState.CLEAR
        if_zero, if_nonzero = (self.x, self.y) if clear else (self.y, self.x)
        zero, nonzero = step.r(if_zero), step.r(if_nonzero)
        step.eq(step.name, f"(if {step.r(self.flag)} = 0 then {zero} else {nonzero})")
        step.bound(step.name, f"ite_lt {step.lt64(if_zero)} {step.lt64(if_nonzero)}")


# -- operations returning a pair ----------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Component:
    """A component of a pair that a `PairOp` binds: its index (`1` or `2`), the register it binds,
    the note continuing the instruction's line, and what it becomes when nothing reads it."""

    index: str
    register: str
    note: str
    dead: Dead = Dead.ERROR


@dataclasses.dataclass(frozen=True, kw_only=True)
class PairOp(Node):
    """An instruction whose Lean function returns a pair: `pair := function args`, under the
    internal name `pair`, then each component bound to its register, `r := pair.i`, on a line
    continuing the instruction's. A carry chain's (value, carry), a wide product's two words,
    and a negation's (value, borrow) are all one. An instruction set reserves the names of its
    pairs (`pair_names`), and a node that writes a fixed name (`implicit_writes`) is never
    renamed there."""

    pair: ClassVar[str] = ""
    function: ClassVar[str] = ""

    def pair_args(self) -> tuple[Operand, ...]:
        raise NotImplementedError

    def components(self) -> list[Component]:
        raise NotImplementedError

    def lets(self) -> list[Let]:
        unpacked = [
            self._let(c.register, Get(Reg(self.pair), c.index), note=c.note, dead=c.dead)
            for c in self.components()
        ]
        return [self._let(self.pair, apply(self.function, *self.pair_args())), *unpacked]


def pair_names(*nodes: type[PairOp]) -> frozenset[str]:
    """The internal names of the pairs of the `PairOp`s `nodes`, which their instruction set
    reserves as operand names."""
    return frozenset(cls.pair for cls in nodes)


# -- calls of rerolled rounds ----------------------------------------------------------------


@dataclasses.dataclass(frozen=True, kw_only=True)
class Call(Node):
    """`dest := round^[count] passed... scalars... ⟨state...⟩`: a call of a round definition (see
    `reroll.py`), iterated `count` times, with the caller's arguments `passed` through by name,
    then word operands, then the registers that make up the round's state."""

    dest: str = dst()
    round: str
    count: int = 1
    passed: tuple[str, ...] = ()
    scalars: tuple[Term, ...] = src()
    state: tuple[str, ...] = src()

    @property
    def head(self) -> str:
        return self.round if self.count == 1 else f"{self.round}^[{self.count}]"

    def term(self) -> App:
        args: list[Term] = [Arg(a) for a in self.passed]
        args += self.scalars
        return App(self.head, (*args, Tuple(tuple(Reg(s) for s in self.state))))

    def lets(self) -> list[Let]:
        return [self._let(self.dest, self.term())]

    def prove(self, step: Step) -> None:
        step.eq(step.name, show(rename_term(self.term(), step.r)))


@dataclasses.dataclass(frozen=True, kw_only=True)
class Project(Node):
    """`dest := call.field`: an output of a round's call. `optional` when the block may leave
    it unread; its bound is supplied by the annotation that applies the round's theorem."""

    dest: str = dst()
    call: str = src()
    field: str
    optional: bool = False

    def lets(self) -> list[Let]:
        dead = Dead.COMMENT if self.optional else Dead.ERROR
        return [self._let(self.dest, Get(Reg(self.call), self.field), dead=dead)]

    def prove(self, step: Step) -> None:
        step.eq(step.name, f"{step.r(self.call)}.{self.field}")
        step.state.bnd[step.name] = f"b_{step.name}"


# -- programs ----------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Signature:
    """A definition's name, its parameters as (name, Lean type) in order, and its result type:
    built once, printed by `text`, and read by anything that binds arguments to parameters."""

    name: str
    params: tuple[tuple[str, str], ...]
    result: str

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.params)

    def text(self) -> str:
        """`def name (a b : K) (c : K') : result :=`, grouping consecutive parameters of one
        type."""
        groups: list[tuple[list[str], str]] = []
        for param, kind in self.params:
            if groups and groups[-1][1] == kind:
                groups[-1][0].append(param)
            else:
                groups.append(([param], kind))
        params = "".join(f" ({' '.join(names)} : {kind})" for names, kind in groups)
        return f"def {self.name}{params} : {self.result} :="


@dataclasses.dataclass(frozen=True)
class Conventions:
    """How the correctness theorems name things, which the skeleton must match: the hypothesis
    bounding each argument, by name (`hypotheses`; any other argument `a` has `ha`), and the
    lemmas of the proofs' preamble that the skeleton cites (`shift_splits`)."""

    hypotheses: dict[str, str] = dataclasses.field(default_factory=dict)
    # The lemma splitting `lsl a k` into its word and the bits shifted out, by `k`.
    shift_splits: dict[int, str] = dataclasses.field(default_factory=dict)

    def hypothesis(self, arg: str) -> str:
        return self.hypotheses.get(arg, f"h{arg}")

    def program(
        self,
        architecture: str,
        signature: Signature,
        doc: str,
        nodes: Sequence[Node],
        results: Sequence[str],
        arg_fields: dict[str, Sequence[str]] | None = None,
        **kw: Any,
    ) -> Program:
        return Program(
            signature,
            doc,
            list(nodes),
            list(results),
            arg_fields={arg: tuple(f) for arg, f in (arg_fields or {}).items()},
            conventions=self,
            architecture=architecture,
            **kw,
        )


class DeadCode(enum.Enum):
    """Whether a transcription drops the bindings that nothing reads, or keeps every
    architectural result."""

    ELIMINATE = "eliminate"
    RETAIN = "retain"


@dataclasses.dataclass
class Program:
    """A Lean definition: its docstring, signature, nodes, and result registers.

    `struct` is the Lean text of a structure declared before it (a round's state); `arg_fields`
    gives each structured argument's fields, and `conventions` the hypotheses bounding the
    arguments, for the skeleton; `clear_values` says whether the skeleton makes each extracted
    value opaque."""

    signature: Signature
    doc: str
    nodes: list[Node]
    results: list[str]
    dead_code: DeadCode = DeadCode.ELIMINATE
    struct: str | None = None
    arg_fields: dict[str, tuple[str, ...]] = dataclasses.field(default_factory=dict)
    conventions: Conventions = dataclasses.field(default_factory=Conventions)
    clear_values: bool = True
    architecture: str = ""
    # For a round, the fields of the state structure it takes and returns, in order: its results
    # are their values, and a call's projections name them.
    state_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.check_hygiene()

    def check_hygiene(self) -> None:
        """Every argument a term reads is the argument itself, as Lean resolves the name.

        The Lean text names an argument and a register alike, by its bare name, and Lean resolves
        a name to its latest `let`. So a binding named like an argument captures every later read
        of that argument, unless it binds the argument to itself (`inv := inv`), which leaves the
        value unchanged. Anything else would make the Lean read a register where the program
        means the argument: accepted by Lean, and silently a different function.

        And every argument a term reads is a parameter of this definition: a round's body that
        read one of the block's arguments would name something out of its scope, which Lean
        resolves to whatever else bears that name, if anything does. The validator relies on this:
        it treats an argument that is not a parameter as free."""
        params = set(self.signature.names)
        identity: dict[str, bool] = {}  # name -> whether its latest binding is `name := name`
        for node, let in self.bindings():
            for name in term_args(let.term):
                if name not in params:
                    raise GenerationError(
                        f"{self.name}: `{let.comment}` reads {name}, which is not a parameter of "
                        f"`{self.signature.name}`"
                    )
                if identity.get(name) is False:
                    raise GenerationError(
                        f"{self.name}: `{let.comment}` reads the argument {name}, which the "
                        f"binding of the register {name} captures"
                    )
            identity[let.name] = let.term == Arg(let.name)

    def bindings(self) -> list[tuple[Node, Let]]:
        """Every `let`, with the node it belongs to."""
        return [(node, let) for node in self.nodes for let in node.lets()]

    def liveness(self) -> list[bool]:
        """For each binding, whether a later binding or the result reads it."""
        bindings = self.bindings()
        needed = set(self.results)
        live = [False] * len(bindings)
        for i in reversed(range(len(bindings))):
            let = bindings[i][1]
            if let.name in needed:
                live[i] = True
                needed.discard(let.name)
                needed |= let.reads
        return live

    def kept(self) -> list[tuple[Node, Let, bool]]:
        """The bindings the transcription and the skeleton keep, with whether each is live."""
        bindings = self.bindings()
        if self.dead_code is DeadCode.RETAIN:
            return [(node, let, True) for node, let in bindings]
        return [
            (node, let, live) for (node, let), live in zip(bindings, self.liveness(), strict=True)
        ]

    @property
    def name(self) -> str:
        return self.signature.name

    def bound_hyp(self, arg: str) -> str:
        return self.conventions.hypothesis(arg)
