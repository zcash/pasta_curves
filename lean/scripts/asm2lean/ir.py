"""The intermediate representation: straight-line programs in A-normal form.

A program is a sequence of nodes, each one operation of an instruction set's Lean semantics
applied to registers and literals, and a tuple of result registers. A node's Lean text is one or
more `let` bindings (`Let`): every intermediate value is named, and a register that an
instruction writes again is rebound, shadowing the old binding. That is also the form of the
generated Lean.

A node is the unit of everything downstream:

- the transcription prints its `lets`;
- liveness reads their `reads`;
- loop rerolling renames it (`renamed`) and compares it up to renaming (`key`);
- the proof skeleton asks it for its proof step (`prove`), one `word_step` per node.

So an instruction set adds node classes and nothing else changes: this module holds the nodes
that every instruction set shares (arguments, moves, products, shifts, selection, calls), and
`aarch64.py` and `x86_64.py` hold the rest.

Liveness is the backward dataflow analysis on the `let`s; on straight-line code one backward pass
reaches the fixed point.

References: C. Flanagan, A. Sabry, B. Duba, M. Felleisen, *The Essence of Compiling with
Continuations*, PLDI 1993 (A-normal form); A. Appel, *Modern Compiler Implementation*, ch. 10
(liveness).
"""

import dataclasses
import enum
import re

IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
# A literal operand: decimal, or hex as the assembly wrote it.
LITERAL = re.compile(r"[0-9]+|0x[0-9a-f]+")

MOD_LT = "Nat.mod_lt _ (Nat.two_pow_pos _)"


def registers(*operands):
    """The registers among operands: identifiers, not literals or argument fields."""
    return frozenset(o for o in operands if isinstance(o, str) and IDENTIFIER.fullmatch(o))


class Dead(enum.Enum):
    """What a binding that nothing reads becomes in a transcription that drops dead bindings."""

    ERROR = "error"  # a computed value: dead code in the block, rejected
    COMMENT = "comment"  # an argument, or an optional call output: left as a comment
    DROP = "drop"  # a flag write: an ordinary unread flag


@dataclasses.dataclass(frozen=True)
class Let:
    """One `let name := expr`. `comment` is the instruction it models; `note`, when given,
    replaces it in the transcription, for a binding that continues the line above."""

    name: str
    expr: str
    comment: str | None
    reads: frozenset
    note: str | None = None
    dead: Dead = Dead.ERROR

    @property
    def trailing(self):
        return self.note or self.comment


def src():
    """A field holding a register or literal that a node reads."""
    return dataclasses.field(metadata={"operand": "read"})


def dst():
    """A field holding a register that a node writes."""
    return dataclasses.field(metadata={"operand": "write"})


@dataclasses.dataclass(frozen=True, kw_only=True)
class Node:
    """One operation. `comment` is the source instruction (or what binds an argument), `pc`
    the index of that instruction, `None` for a node with no instruction of its own."""

    comment: str | None
    pc: int | None = None
    # The registers this node writes as the source wrote them, by field, once it has been renamed:
    # a continuation note names the register of the source line it continues.
    written_as: tuple = ()

    def lets(self):
        raise NotImplementedError

    def source_name(self, field):
        """The register in `field` as the source wrote it."""
        return dict(self.written_as).get(field, getattr(self, field))

    def label(self, names):
        """The register named in the skeleton's marker for this node's step."""
        return names[0]

    def prove(self, step):
        """Emit this node's proof step (see `skeleton.Step`)."""
        raise ValueError(f"unsupported skeleton fact {type(self).__name__}")

    def renamed(self, mapping, writes=True):
        """The node with its operands renamed by `mapping`; written registers too if `writes`."""
        changes = {}
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
            value = getattr(self, f.name)
            if isinstance(value, str):
                changes[f.name] = mapping.get(value, value)
            elif isinstance(value, tuple):
                changes[f.name] = tuple(mapping.get(v, v) for v in value)
        return dataclasses.replace(self, **changes)

    def key(self):
        """The node up to its comment and position: what two rounds must agree on."""
        return (type(self).__name__,) + tuple(
            getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name not in ("comment", "pc", "written_as")
        )

    def writes(self):
        return [let.name for let in self.lets()]

    def _let(self, name, expr, *reads, note=None, comment=..., dead=Dead.ERROR):
        return Let(
            name,
            expr,
            self.comment if comment is ... else comment,
            registers(*reads),
            note,
            dead,
        )


# -- arguments -------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, kw_only=True)
class Load(Node):
    """A field of a structured argument: `dest := arg.field`."""

    dest: str = dst()
    arg: str = src()
    field: str

    def lets(self):
        return [self._let(self.dest, f"{self.arg}.{self.field}", dead=Dead.COMMENT)]

    def prove(self, step):
        step.eq(step.name, f"{self.arg}.{self.field}")
        step.bound(step.name, step.argument_bound(self.arg, self.field))


@dataclasses.dataclass(frozen=True, kw_only=True)
class Scalar(Node):
    """A word argument: `dest := name`, bounded by the hypothesis on it (`inv`'s, or
    `h<name>` for a round's parameter)."""

    dest: str = dst()
    name: str = src()

    def lets(self):
        return [self._let(self.dest, self.name)]

    @property
    def is_inv(self):
        return self.name == "inv"

    def prove(self, step):
        step.eq(step.name, self.name)
        step.bound(step.name, step.inv_bound if self.is_inv else f"h{self.name}")


# -- the operations every instruction set shares ---------------------------------------------


@dataclasses.dataclass(frozen=True, kw_only=True)
class Mov(Node):
    dest: str = dst()
    a: str = src()

    def lets(self):
        return [self._let(self.dest, self.a, self.a)]

    def prove(self, step):
        a = step.r(self.a)
        step.eq(step.name, a)
        step.bound(step.name, step.lt64(a))


@dataclasses.dataclass(frozen=True, kw_only=True)
class MulLo(Node):
    """`mulLo a b`: the low word of the product."""

    dest: str = dst()
    a: str = src()
    b: str = src()

    def lets(self):
        return [self._let(self.dest, f"mulLo {self.a} {self.b}", self.a, self.b)]

    def prove(self, step):
        a, b = step.r(self.a), step.r(self.b)
        step.eq(step.name, f"{a} * {b} % 2^64")
        step.bound(step.name, MOD_LT)
        step.state.products[(a, b)] = step.name  # cleared at the matching `umulh`


@dataclasses.dataclass(frozen=True, kw_only=True)
class MulHi(Node):
    """`umulh a b`: the high word of the product. With the `mulLo` of the same operands, the
    two equations combine into `lo + 2^64 * hi = a * b`; without one, the low word is named as
    a ghost, so that later steps need no `%`."""

    dest: str = dst()
    a: str = src()
    b: str = src()

    def lets(self):
        return [self._let(self.dest, f"umulh {self.a} {self.b}", self.a, self.b)]

    def prove(self, step):
        a, b, nm = step.r(self.a), step.r(self.b), step.name
        step.eq(nm, f"{a} * {b} / 2^64")
        step.bound(nm, f"Nat.div_lt_of_lt_mul (Nat.mul_lt_mul'' {step.lt64(a)} {step.lt64(b)})")
        lo = step.state.products.pop((a, b), None)
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


@dataclasses.dataclass(frozen=True, kw_only=True)
class Shl(Node):
    """`lsl a k`. By 62, its split fact is stated on the spot, with the high part as
    `a / 2^2`, so that `omega` connects it with the matching `lsr`."""

    dest: str = dst()
    a: str = src()
    k: int

    def lets(self):
        return [self._let(self.dest, f"lsl {self.a} {self.k}", self.a)]

    def prove(self, step):
        a, k, nm = step.r(self.a), self.k, step.name
        if k != 62:
            raise ValueError(f"lsl by {k}: add a lemma to the spec preamble")
        step.eq(nm, f"{a} * 2^{k} % 2^64")
        step.bound(nm, MOD_LT)
        step.line(f"  have sh_{nm} : {nm} + 2^64 * ({a} / 2^2) = {a} * 2^62 := by")
        step.line(f"    rw [e_{nm}]; exact lsl62_lsr2_split _")


@dataclasses.dataclass(frozen=True, kw_only=True)
class Shr(Node):
    """`lsr a k`, bounded below `2^(64 - k)`."""

    dest: str = dst()
    a: str = src()
    k: int

    def lets(self):
        return [self._let(self.dest, f"lsr {self.a} {self.k}", self.a)]

    def prove(self, step):
        a, k, nm = step.r(self.a), self.k, step.name
        step.eq(nm, f"{a} / 2^{k}")
        step.line(f"  have b_{nm} : {nm} < 2^{64 - k} := by")
        step.line(
            f"    rw [e_{nm}]; exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq {step.lt64(a)} (by norm_num))"
        )
        step.state.bnd[nm] = f"b_{nm}"
        step.state.narrow.add(nm)


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
    flag: str = src()
    x: str = src()
    y: str = src()
    x_when: FlagState

    def lets(self):
        return [
            self._let(
                self.dest,
                f"{self.function} {self.flag} {self.x} {self.y}",
                self.flag,
                self.x,
                self.y,
            )
        ]

    def prove(self, step):
        c, x, y = step.r(self.flag), step.r(self.x), step.r(self.y)
        if_zero, if_nonzero = (x, y) if self.x_when is FlagState.CLEAR else (y, x)
        step.eq(step.name, f"(if {c} = 0 then {if_zero} else {if_nonzero})")
        step.bound(step.name, f"ite_lt {step.lt64(if_zero)} {step.lt64(if_nonzero)}")


# -- calls of rerolled rounds ----------------------------------------------------------------


@dataclasses.dataclass(frozen=True, kw_only=True)
class Call(Node):
    """`dest := head scalars... ⟨state...⟩`: a call of a round definition (see `reroll.py`)."""

    dest: str = dst()
    head: str
    scalars: tuple = src()
    state: tuple = src()

    def expr(self, rename=lambda r: r):
        scalars = "".join(f" {rename(s)}" for s in self.scalars)
        return f"{self.head}{scalars} ⟨{', '.join(rename(s) for s in self.state)}⟩"

    def lets(self):
        return [Let(self.dest, self.expr(), self.comment, frozenset(self.scalars + self.state))]

    def prove(self, step):
        step.eq(step.name, self.expr(step.r))


@dataclasses.dataclass(frozen=True, kw_only=True)
class Project(Node):
    """`dest := call.field`: an output of a round's call. `optional` when the block may leave
    it unread; its bound is supplied by the annotation that applies the round's theorem."""

    dest: str = dst()
    call: str = src()
    field: str
    optional: bool = False

    def lets(self):
        dead = Dead.COMMENT if self.optional else Dead.ERROR
        return [self._let(self.dest, f"{self.call}.{self.field}", self.call, dead=dead)]

    def prove(self, step):
        step.eq(step.name, f"{step.r(self.call)}.{self.field}")
        step.state.bnd[step.name] = f"b_{step.name}"


# -- programs ----------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Conventions:
    """How the correctness theorems name things, which the skeleton must match: the fields of
    each structured argument (`arg_fields`, by argument name), the hypothesis bounding it
    (`bound_hyps`, else `h<arg>`), and the hypothesis bounding `inv`."""

    arg_fields: dict
    bound_hyps: dict
    inv_bound: str = "hinv_lt"

    def program(self, architecture, name, doc, signature, nodes, results, arg_fields=None, **kw):
        fields = {arg: list(f) for arg, f in self.arg_fields.items()}
        fields.update({arg: list(f) for arg, f in (arg_fields or {}).items()})
        return Program(
            name,
            doc,
            signature,
            list(nodes),
            list(results),
            arg_fields=fields,
            bound_hyps=dict(self.bound_hyps),
            inv_bound=self.inv_bound,
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
    gives each structured argument's fields and `bound_hyps` the hypothesis bounding it, for the
    skeleton; `clear_values` says whether the skeleton makes each extracted value opaque."""

    name: str
    doc: str
    signature: str
    nodes: list
    results: list
    dead_code: DeadCode = DeadCode.ELIMINATE
    struct: str | None = None
    arg_fields: dict = dataclasses.field(default_factory=dict)
    bound_hyps: dict = dataclasses.field(default_factory=dict)
    inv_bound: str = "hinv_lt"
    clear_values: bool = True
    architecture: str = ""

    def bindings(self):
        """Every `let`, with the node it belongs to."""
        return [(node, let) for node in self.nodes for let in node.lets()]

    def liveness(self):
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

    def kept(self):
        """The bindings the transcription and the skeleton keep, with whether each is live."""
        bindings = self.bindings()
        if self.dead_code is DeadCode.RETAIN:
            return [(node, let, True) for node, let in bindings]
        return [
            (node, let, live) for (node, let), live in zip(bindings, self.liveness(), strict=True)
        ]

    def bound_hyp(self, arg):
        return self.bound_hyps.get(arg, f"h{arg}")
