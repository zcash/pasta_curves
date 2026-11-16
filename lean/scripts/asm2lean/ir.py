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


@dataclasses.dataclass(frozen=True)
class Const:
    """A named constant of the Lean semantics, such as `wordMod`."""

    name: str


@dataclasses.dataclass(frozen=True)
class Infix:
    """A binary operator of Lean's arithmetic, `left op right`: `+` and `-`, or `*`, `/` and `%`,
    each group left-associative and the second binding tighter. It prints with the parentheses
    the precedences need, and in parentheses wherever it is an operand of an application."""

    op: str
    left: Term
    right: Term


@dataclasses.dataclass(frozen=True)
class Ite:
    """`(if cond = 0 then then_ else else_)`: a choice on whether a word is zero."""

    cond: Term
    then_: Term
    else_: Term


@dataclasses.dataclass(frozen=True)
class Ascribe:
    """A type ascription, `(term : type)`."""

    term: Term
    type: str


@dataclasses.dataclass(frozen=True)
class Prod:
    """A tuple of Lean's product type, `(a, b, ...)`."""

    items: tuple[Term, ...]


# A term: data the transcription prints and every other interpretation folds over.
Term = Reg | Lit | Arg | Get | App | Tuple | Const | Infix | Ite | Ascribe | Prod

# The term classes, for telling a term from the other operands a node holds.
TERMS = (Reg, Lit, Arg, Get, App, Tuple, Const, Infix, Ite, Ascribe, Prod)

# The precedence of each `Infix` operator: `*`, `/` and `%` bind tighter than `+` and `-`.
PRECEDENCE = {"+": 65, "-": 65, "*": 70, "/": 70, "%": 70}
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
        case Reg(name) | Arg(name) | Const(name):
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
        case Infix():
            text = _show_infix(term)
            return f"({text})" if nested else text
        case Ite(cond, then_, else_):
            return f"(if {show(cond)} = 0 then {show(then_)} else {show(else_)})"
        case Ascribe(inner, kind):
            return f"({show(inner)} : {kind})"
        case Prod(items):
            return f"({', '.join(show(i) for i in items)})"
    raise TypeError(f"not a term: {term!r}")


def _show_infix(term: Infix) -> str:
    """An `Infix` term's text, its operands parenthesized only where the precedences need it."""
    level = PRECEDENCE[term.op]

    def operand(t: Term, right: bool) -> str:
        if isinstance(t, Infix):
            inner = PRECEDENCE[t.op]
            text = _show_infix(t)
            return f"({text})" if inner < level or (inner == level and right) else text
        return show(t, nested=True)

    return f"{operand(term.left, False)} {term.op} {operand(term.right, True)}"


def subterms(term: Term) -> tuple[Term, ...]:
    """The immediate subterms of a term."""
    match term:
        case Get(base, _):
            return (base,)
        case App(_, items) | Tuple(items) | Prod(items):
            return items
        case Infix(_, left, right):
            return (left, right)
        case Ite(cond, then_, else_):
            return (cond, then_, else_)
        case Ascribe(inner, _):
            return (inner,)
    return ()


def term_registers(term: Term) -> frozenset[str]:
    """The registers a term reads."""
    if isinstance(term, Reg):
        return frozenset({term.name})
    return frozenset[str]().union(*(term_registers(t) for t in subterms(term)))


def term_args(term: Term) -> frozenset[str]:
    """The arguments a term reads, by name: `Arg`s, and the bases of their fields."""
    if isinstance(term, Arg):
        return frozenset({term.name})
    return frozenset[str]().union(*(term_args(t) for t in subterms(term)))


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
        case Prod(items):
            return Prod(tuple(rename_term(i, rename) for i in items))
        case Infix(op, left, right):
            return Infix(op, rename_term(left, rename), rename_term(right, rename))
        case Ite(cond, then_, else_):
            return Ite(
                rename_term(cond, rename), rename_term(then_, rename), rename_term(else_, rename)
            )
        case Ascribe(inner, kind):
            return Ascribe(rename_term(inner, rename), kind)
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
            if isinstance(value, TERMS):
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


@dataclasses.dataclass(frozen=True)
class Width:
    """The word of an instruction set: its bits, the suffix of the Lean names of its word
    operations (`addc`, `addc32`), and whether `word_step` proves a word's bound. It does for a
    64-bit word, `x < 2^64` being the bound it states (`x := v using proof`); a narrower word's
    bound is a line after the step instead."""

    bits: int
    suffix: str = ""

    @property
    def in_step(self) -> bool:
        return self.bits == 64

    def bound(self, step: Step, name: str, value: str, proof: str) -> None:
        """`name := value`, below `2^bits` by `proof`."""
        step.eq(name, value)
        if self.in_step:
            step.bound(name, proof)
            return
        step.line(f"  have b_{name} : {name} < 2^{self.bits} := by rw [e_{name}]; exact {proof}")
        step.state.bnd[name] = f"b_{name}"


WIDTH64 = Width(64)
WIDTH32 = Width(32, "32")


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
    note: str | None
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

    def pair_name(self) -> str:
        """The internal name of the pair: the class's, unless an instruction set chooses it per
        instruction."""
        return self.pair

    def pair_args(self) -> tuple[Operand, ...]:
        raise NotImplementedError

    def components(self) -> list[Component]:
        raise NotImplementedError

    def lets(self) -> list[Let]:
        pair = self.pair_name()
        unpacked = [
            self._let(c.register, Get(Reg(pair), c.index), note=c.note, dead=c.dead)
            for c in self.components()
        ]
        return [self._let(pair, apply(self.function, *self.pair_args())), *unpacked]


@dataclasses.dataclass(frozen=True, kw_only=True)
class CarryChain(PairOp):
    """A link of a carry chain: `pair := function a b cin` (`addc`, or `subc`, whose carry is set
    when no borrow occurred), its value in `dest`, and its carry in `c`.

    Its step bounds the value, then derives the chain's linear equation (`dest + 2^w * c =
    a + b + cin`, or `dest + 2^w * c + b + 1 = a + 2^w + cin`) and the carry's bound by `1`, and
    clears the `%` and `/` equations. A carry nothing reads is, with `ghost_carry`, obtained as a
    ghost for the linear equation, and otherwise left out. A binding after the carry, which the
    step names without a value, gets a step of its own (`prove_rest`)."""

    subtraction: ClassVar[bool] = False
    ghost_carry: ClassVar[bool] = True

    dest: str = dst()
    a: Term = src()
    b: Term = src()
    cin: Term = src()

    def pair_args(self) -> tuple[Operand, ...]:
        return (self.a, self.b, self.cin)

    def components(self) -> list[Component]:
        return [
            Component("1", self.dest, f"  `-> {self.source_name('dest')}"),
            Component("2", "c", "  `-> carry", Dead.DROP),
        ]

    def label(self, names: list[str]) -> str:
        return names[1]

    def prove(self, step: Step) -> None:
        width = step.state.program.width
        w, suffix = f"2^{width.bits}", width.suffix
        a, b, cin = step.r(self.a), step.r(self.b), step.r(self.cin)
        names, lets = step.names, step.lets
        value = names[1]
        carry = names[2] if len(names) > 2 else f"k_{value}"
        if self.subtraction:
            total = f"({a} + {w} - {b} - (1 - {cin}))"
            linear = f"{value} + {w} * {carry} + {b} + 1 = {a} + {w} + {cin}"
            linear_proof = (
                f"subc{suffix}_lin {a} {b} {cin} {step.lt64(self.b)} {step.le1(self.cin)}"
            )
            carry_proof = f"subc{suffix}_carry_le_one {a} {b} {cin} {step.lt64(self.a)}"
        else:
            total = f"({a} + {b} + {cin})"
            linear = f"{value} + {w} * {carry} = {a} + {b} + {cin}"
            linear_proof = "Nat.mod_add_div _ _"
            carry_proof = (
                f"addc{suffix}_carry_le_one {a} {b} {cin} {step.lt64(self.a)} "
                f"{step.lt64(self.b)} {step.le1(self.cin)}"
            )
        width.bound(step, value, f"{total} % {w}", MOD_LT)
        step.define(1)
        if len(names) == 2:
            if self.ghost_carry:
                step.line(f"  obtain ⟨{carry}, b_{carry}, l_{value}⟩ :")
                step.line(f"      ∃ k, k ≤ 1 ∧ {linear.replace(carry, 'k')} :=")
                step.line(f"    ⟨{total} / {w}, {carry_proof},")
                step.line(f"      by rw [e_{value}]; exact {linear_proof}⟩")
                step.line(f"  clear e_{value}")
            return
        step.eq(carry, f"{total} / {w}")
        step.line(f"  have l_{value} : {linear} := by")
        step.line(f"    rw [e_{value}, e_{carry}]; exact {linear_proof}")
        step.line(f"  have b_{carry} : {carry} ≤ 1 := by")
        step.line(f"    rw [e_{carry}]; exact {carry_proof}")
        step.line(f"  clear e_{value} e_{carry}")
        step.define(2)
        step.unit(carry)
        for index in range(3, len(names)):
            self.prove_rest(step, lets[index], names[index])

    def prove_rest(self, step: Step, let: Let, name: str) -> None:
        """The step of a binding after the carry, which the pair's step names without a value."""
        raise GenerationError(f"`{self.comment}`: no step for its binding {let.name}")


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
    value opaque, and `lift_lets` whether it lifts every `let` of the unfolded definition to the
    top before extracting them (a definition whose state equality Lean would otherwise normalize
    recursively extracts them as they come). `result_text`, when given, is the result line as
    written, for a definition that returns more than its result registers' tuple (`results` are
    then the registers it reads, for liveness). `shadow` lets the skeleton name a later binding
    like an earlier one, which Lean then shadows (`skeleton.ssa_names`). `width` is the word of
    its instruction set (`Width`)."""

    signature: Signature
    doc: str
    nodes: list[Node]
    results: list[str]
    dead_code: DeadCode = DeadCode.ELIMINATE
    struct: str | None = None
    arg_fields: dict[str, tuple[str, ...]] = dataclasses.field(default_factory=dict)
    conventions: Conventions = dataclasses.field(default_factory=Conventions)
    clear_values: bool = True
    lift_lets: bool = True
    result_text: str | None = None
    shadow: bool = False
    width: Width = WIDTH64
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
