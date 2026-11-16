"""Loop rerolling: folding repeated instruction sequences back into a round called per repetition.

Assembly generators unroll loops, so a block's instruction stream is a word `u v_1 ... v_n w` in
which the `v_i` are one body up to renaming registers. Rerolling recovers the body as a definition
(a "round") over the registers that cross its boundary, and replaces each `v_i` by a call of it,
after checking that the `v_i` really are the same body up to renaming, that is, alpha-equivalent.
The registers crossing the boundary come from the same dataflow facts as liveness: the body's
inputs are the registers it reads before writing them (its upward-exposed uses), and its outputs
the registers it writes that the code after it reads before writing them again.

The blocks repeat in three ways, one strategy each:

- `PrologueLoop` (AArch64 `mul`): a prologue, `count` rounds that differ only in the register
  holding the round's `rhs` limb, and an epilogue; the rounds end at a marker instruction.
- `MacroRounds` (the inversion's `divstep59`, on every instruction set): the source invokes a
  `macro_rules!` macro per step, so the repetitions are known from the front end's macro origins;
  a run of consecutive invocations of one arm becomes one call of the round iterated over the run
  (`round^[n]`).
- `RotatedRounds` (x86-64 `mul`): two full rounds that agree once the accumulator registers are
  rotated, at fixed instruction ranges; the round is checked to flatten back to each of them.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING

from .ir import App, Arg, Call, Get, Load, Mov, Node, Project, Reg, Scalar, Signature, Term, Tuple
from .lean import state_structure
from .rust import GenerationError, MacroOrigin

if TYPE_CHECKING:
    from .block import Unit
    from .ir import Program


def upward_exposed(nodes: Sequence[Node], exclude: Collection[str] = ()) -> list[str]:
    """The registers that `nodes` read before writing them, in the order first read."""
    written: set[str] = set()
    exposed: list[str] = []
    for node in nodes:
        for let in node.lets():
            for r in sorted(let.reads):
                if r not in written and r not in exposed and r not in exclude:
                    exposed.append(r)
            written.add(let.name)
    return exposed


def written(nodes: Sequence[Node]) -> set[str]:
    """The registers that `nodes` write."""
    return {let.name for node in nodes for let in node.lets()}


@dataclasses.dataclass(frozen=True)
class Role:
    """A register carried from one round to the next, its field in the round's state structure,
    the field's docstring, and its Lean type: `Nat` for a word, bounded below `2^64` and
    projected by the skeleton, or another type of the instruction set's semantics (AArch64's
    `Flags`), loaded by the node its `Isa.state_load` names."""

    register: str
    field: str
    doc: str | None = None
    type: str = "Nat"

    @property
    def is_word(self) -> bool:
        return self.type == "Nat"


def position(node: Node) -> int:
    """The instruction index of a node that comes from an instruction."""
    if node.pc is None:
        raise GenerationError(f"`{node.comment}` has no instruction")
    return node.pc


def alpha_match(source: list[Node], target: list[Node]) -> dict[str, str]:
    """The renaming of registers under which `source` is `target`, node by node: a bijection
    `source register -> target register`, found by walking the two in lockstep (first-order
    matching where the only variables are register names). Every other field, and each node's
    position relative to the first, must agree exactly; otherwise it raises, naming the first
    disagreement.

    A name that a node of either list writes implicitly (`Node.implicit_writes`: a flag, a pair)
    is rigid, matched only by itself: `Node.renamed` renames a read of it but not the write, so a
    renaming that moved it would not commute with the nodes. Under the bijection returned,
    `[n.renamed(m) for n in source]` is `target` node by node, up to comments and positions."""
    if len(source) != len(target):
        raise GenerationError(f"rounds of {len(source)} and {len(target)} operations")
    rigid = frozenset[str]().union(*(n.implicit_writes() for n in (*source, *target)))
    forward: dict[str, str] = {}
    backward: dict[str, str] = {}

    def register(a: str, b: str, where: str) -> None:
        if (a in rigid or b in rigid) and a != b:
            raise GenerationError(
                f"{where}: {a} and {b} differ, and {a if a in rigid else b} is fixed"
            )
        if forward.setdefault(a, b) != b or backward.setdefault(b, a) != a:
            raise GenerationError(f"{where}: {a} and {b} break the renaming")

    def pair(a: object, b: object, where: str) -> None:
        """Match two operands: register names pair up; terms match by structure, their
        registers pairing up; anything else must be equal."""
        match a, b:
            case str(), str():
                register(a, b, where)
            case Reg(x), Reg(y):
                register(x, y, where)
            case Get(x, f), Get(y, g) if f == g:
                pair(x, y, where)
            case App(f, xs), App(g, ys) if f == g and len(xs) == len(ys):
                for x, y in zip(xs, ys, strict=True):
                    pair(x, y, where)
            case Tuple(xs), Tuple(ys) if len(xs) == len(ys):
                for x, y in zip(xs, ys, strict=True):
                    pair(x, y, where)
            case _ if a != b:
                raise GenerationError(f"{where}: {a!r} and {b!r} differ")

    base_a = position(source[0]) if source else 0
    base_b = position(target[0]) if target else 0
    for a, b in zip(source, target, strict=True):
        where = f"`{a.comment}` and `{b.comment}`"
        if type(a) is not type(b) or position(a) - base_a != position(b) - base_b:
            raise GenerationError(f"{where}: different operations")
        for f in dataclasses.fields(a):
            if f.name in ("comment", "pc", "written_as"):
                continue
            va, vb = getattr(a, f.name), getattr(b, f.name)
            if f.metadata.get("operand") and isinstance(va, tuple) and isinstance(vb, tuple):
                if len(va) != len(vb):
                    raise GenerationError(f"{where}: different operand counts")
                for x, y in zip(va, vb, strict=True):
                    pair(x, y, where)
            elif f.metadata.get("operand"):
                pair(va, vb, where)
            elif va != vb:
                raise GenerationError(f"{where}: {f.name} differs")
    return forward


# -- a prologue, rounds, and an epilogue -----------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PrologueLoop:
    """`end` is the instruction that ends the prologue and each round (as written), `count` the
    number of rounds, which must agree up to `operand`, the register holding the round's `rhs`
    limb (`{i}` is the round number),
    which the round takes as its parameter `param`. The round `round` carries the registers of
    `roles` in the structure `state`, passed as `arg`; `value` names the fields that make up the
    accumulator, and `emit_struct` says whether this block's transcription declares the
    structure."""

    end: str
    count: int
    operand: str
    round: str
    state: str
    param: str
    arg: str
    value: tuple[str, ...]
    roles: tuple[Role, ...]
    emit_struct: bool = True

    def register(self, k: int) -> str:
        """The register holding round `k`'s `rhs` limb."""
        return self.operand.format(i=k)

    def reroll(self, flat: Program, unit: Unit) -> list[Program]:
        """The round and the rerolled block, from the `flat` program of the block `unit`."""
        block_name, doc, nodes, results = flat.name, flat.doc, flat.nodes, flat.results
        instructions, limb_args, make = unit.instructions, unit.limb_args, unit.make
        first_pc = min(n.pc for n in nodes if n.pc is not None)
        ends = [p for p in range(first_pc, len(instructions)) if instructions[p].text == self.end]
        if len(ends) != self.count + 1:
            raise GenerationError(
                f"{block_name}: expected {self.count + 1} `{self.end}`, found {len(ends)}"
            )
        first = self.register(1)

        def segment(node: Node) -> int:
            """0 for the prologue, k for round k, count + 1 for the epilogue."""
            if node.pc is None or node.pc <= ends[0]:
                return 0
            return next((k for k in range(1, self.count + 1) if node.pc <= ends[k]), self.count + 1)

        prologue = [n for n in nodes if segment(n) == 0]
        body = [n for n in nodes if segment(n) == 1]
        # Round k is round 1 with its `rhs` register for round 1's, and every other register the
        # same: the renaming that matches them must be the identity.
        for k in range(2, self.count + 1):
            round_k = [n.renamed({self.register(k): first}) for n in nodes if segment(n) == k]
            try:
                renaming = alpha_match(round_k, body)
            except GenerationError as failure:
                raise GenerationError(
                    f"{block_name}: the round bodies differ beyond `{first}`: {failure}"
                ) from failure
            moved = sorted(a for a, b in renaming.items() if a != b)
            if moved:
                raise GenerationError(
                    f"{block_name}: the round bodies differ beyond `{first}`: round {k} uses "
                    f"{moved} where round 1 uses {[renaming[a] for a in moved]}"
                )
        after = [n for n in nodes if segment(n) > 1]
        epilogue = [n for n in nodes if segment(n) == self.count + 1]

        inputs = upward_exposed(body, exclude={first})
        body_writes = written(body)
        outputs = [r for r in upward_exposed(after) if r in body_writes]
        regs = [role.register for role in self.roles]
        fields = [role.field for role in self.roles]
        if sorted(outputs) != sorted(regs):
            raise GenerationError(
                f"{block_name}: registers carried between rounds are {sorted(outputs)}, "
                f"`roles` lists {sorted(regs)}"
            )
        # An input is an invariant argument if its value on entry is an argument's field or a word
        # argument;
        # otherwise it is carried state.
        defined = {let.name: n for n in prologue for let in n.lets()}
        invariant = [(r, defined[r]) for r in inputs if _is_argument(defined[r])]
        carried = [r for r in inputs if not _is_argument(defined[r])]
        if sorted(carried) != sorted(regs):
            raise GenerationError(
                f"{block_name}: registers read from the previous round are {sorted(carried)}, "
                f"`roles` lists {sorted(regs)}"
            )
        args = [
            arg
            for arg in limb_args
            if any(isinstance(n, Load) and n.arg == arg for _, n in invariant)
        ]
        scalars = [(r, n) for r, n in invariant if isinstance(n, Scalar)]
        kinds = dict(unit.block.args)

        def order(item: tuple[str, Node]) -> tuple[int, int, str]:
            n = item[1]
            return (0, args.index(n.arg), n.field) if isinstance(n, Load) else (1, 0, "")

        round_nodes = (
            [
                dataclasses.replace(n, comment="argument", pc=None)
                for _, n in sorted(invariant, key=order)
            ]
            + [Scalar(comment="argument", dest=first, name=self.param)]
            + [
                unit.target.isa.state_load(role.type)(
                    comment="argument", dest=role.register, arg=self.arg, field=role.field
                )
                for role in self.roles
            ]
            + body
        )
        signature = Signature(
            self.round,
            tuple((a, kinds[a]) for a in args)
            + tuple((n.name, kinds[n.name]) for _, n in scalars if isinstance(n, Scalar))
            + ((self.param, "Nat"), (self.arg, self.state)),
            self.state,
        )
        round_doc = (
            f"One round of `{block_name}`: the instructions between one `{self.end}` and the next "
            f"in each of its {self.count} rounds, on the round's `rhs` limb `{self.param}` and the "
            f"registers `{self.arg}` carried from the previous round."
        )
        struct = None
        if self.emit_struct:
            struct = state_structure(
                self.state,
                f"/-- The registers that `{block_name}` carries from one round to the next. -/",
                [(r.field, r.doc, r.type) for r in self.roles],
                words_doc="field",
                value=self.value,
            )
        round_program = make(
            signature,
            round_doc,
            round_nodes,
            regs,
            arg_fields={
                **{a: flat.arg_fields[a] for a in args},
                self.arg: [role.field for role in self.roles if role.is_word],
            },
            struct=struct,
            state_fields=tuple(fields),
        )

        calls: list[Node] = []
        for k in range(1, self.count + 1):
            name = f"round{k}"
            calls.append(
                Call(
                    comment=f"round {k}",
                    dest=name,
                    round=self.round,
                    passed=tuple(args),
                    scalars=(*(Reg(r) for r, _ in scalars), Reg(self.register(k))),
                    state=tuple(regs),
                )
            )
            calls += [
                Project(comment=f"round {k} output", dest=r, call=name, field=f)
                for r, f in zip(regs, fields)
            ]
        block_program = make(
            flat.signature, doc, prologue + calls + epilogue, results, arg_fields=flat.arg_fields
        )
        return [round_program, block_program]


def _is_argument(node: Node) -> bool:
    """Whether a node binds an argument: a field of a structured one, or a word one."""
    return isinstance(node, (Load, Scalar))


# -- runs of macro invocations ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MacroRound:
    """A macro arm transcribed as a round: the definition `round` with docstring `doc`, the
    structure `state` of the registers it carries in and out (passed as `arg`), `roles` for that
    structure, whether this arm's transcription declares it, and the name prefix `call` of a call's
    result. A round's body may read only carried registers."""

    round: str
    doc: str
    state: str
    arg: str
    call: str
    roles: tuple[Role, ...]
    emit_struct: bool


@dataclasses.dataclass
class Run:
    """Consecutive invocations of one arm (`key`, a (macro, arm) pair), rerolled as one call of
    its round iterated over them: the invocation sites `first` to `last`."""

    round: MacroRound
    key: tuple[str, str]
    first: int
    last: int


class MacroRounds:
    """The arms of `rounds` (keyed by (macro, arm)) become round definitions, and each run of
    consecutive invocations of one arm a call of it iterated over the run; any other arm stays
    expanded in place."""

    def __init__(self, rounds: dict[tuple[str, str], MacroRound]) -> None:
        self.rounds = rounds

    def reroll(self, flat: Program, unit: Unit) -> list[Program]:
        """The round definitions and the rerolled block, from the `flat` program of the block
        `unit`."""
        name, nodes, origins, make = flat.name, flat.nodes, unit.origins, unit.make
        state_load = unit.target.isa.state_load

        def origin(node: Node) -> MacroOrigin | None:
            return origins[node.pc] if node.pc is not None else None

        def round_of(o: MacroOrigin | None) -> MacroRound | None:
            return self.rounds.get((o.macro, o.arm)) if o is not None else None

        # Consecutive nodes of one round invocation form a segment; a segment of other nodes has
        # site `None`. Sites are numbered from 0, so -1 marks "no segment yet".
        segments: list[tuple[int | None, list[Node]]] = []
        current: list[Node] = []
        current_site: int | None = -1
        for node in nodes:
            o = origin(node)
            site = o.site if o is not None and round_of(o) else None
            if site != current_site:
                if current:
                    segments.append((current_site, current))
                current, current_site = [], site
            current.append(node)
        if current:
            segments.append((current_site, current))

        # The round definitions, one per arm, from each arm's first invocation.
        rounds: dict[tuple[str, str], Program] = {}
        round_of_site: dict[int, MacroRound] = {}
        for site, segment in segments:
            if site is None:
                continue
            o = origin(segment[0])
            assert o is not None  # a segment with a site is an invocation
            key = (o.macro, o.arm)
            cfg = self.rounds[key]
            round_of_site[site] = cfg
            regs = [r.register for r in cfg.roles]
            fields = [r.field for r in cfg.roles]
            inputs, body_writes = upward_exposed(segment), written(segment)
            if key in rounds:
                continue
            stray = sorted(set(inputs) - set(regs))
            if stray:
                raise GenerationError(
                    f"{name}: {cfg.round} reads {stray}, which `roles` does not carry"
                )
            unused = sorted(set(regs) - set(inputs) - body_writes)
            if unused:
                raise GenerationError(f"{name}: {cfg.round} neither reads nor writes {unused}")
            loads: list[Node] = [
                state_load(role.type)(
                    comment="argument", dest=role.register, arg=cfg.arg, field=role.field
                )
                for role in cfg.roles
            ]
            struct = None
            if cfg.emit_struct:
                struct = state_structure(
                    cfg.state,
                    f"/-- The registers that `{cfg.round}` carries in and out. -/",
                    [(r.field, r.doc, r.type) for r in cfg.roles],
                    words_doc="word",
                )
            rounds[key] = make(
                Signature(cfg.round, ((cfg.arg, cfg.state),), cfg.state),
                f"`{o.macro}!({o.arm})`, {cfg.doc}",
                loads + segment,
                regs,
                arg_fields={cfg.arg: [r.field for r in cfg.roles if r.is_word]},
                struct=struct,
                state_fields=tuple(fields),
            )

        # A round's call passes its state positionally (`⟨...⟩`), so every round of a structure
        # must list its fields in the order of the round that declares the structure.
        declared = {c.state: c for c in round_of_site.values() if c.emit_struct}
        for cfg in round_of_site.values():
            owner = declared.get(cfg.state)
            if owner is None:
                raise GenerationError(f"{name}: no round of this block declares {cfg.state}")
            if [r.field for r in cfg.roles] != [r.field for r in owner.roles]:
                raise GenerationError(
                    f"{name}: {cfg.round} lists the fields of {cfg.state} as "
                    f"{[r.field for r in cfg.roles]}, which {owner.round} declares as "
                    f"{[r.field for r in owner.roles]}"
                )

        # The block: a run of consecutive invocations of one arm is one call, the round iterated
        # over the run, and the outputs it carries; a proof about the run is then the step
        # theorem iterated, without one binding per invocation.
        runs: list[Run | list[Node]] = []  # a run of invocations, or a stretch of other nodes
        for site, segment in segments:
            if site is None:
                runs.append(segment)
                continue
            o = origin(segment[0])
            assert o is not None  # a segment with a site is an invocation
            key = (o.macro, o.arm)
            previous = runs[-1] if runs else None
            # Only consecutive invocations merge: an invocation between two of one arm that lifts
            # to no nodes would otherwise be counted as an iteration of the round.
            if isinstance(previous, Run) and previous.key == key and previous.last == site - 1:
                previous.last = site
            else:
                runs.append(Run(round_of_site[site], key, site, site))
        block_nodes: list[Node] = []
        for run in runs:
            if not isinstance(run, Run):
                block_nodes += run
                continue
            cfg, key, first, last = run.round, run.key, run.first, run.last
            regs = [r.register for r in cfg.roles]
            fields = [r.field for r in cfg.roles]
            dest = f"{cfg.call}{last + 1}"
            arm = f"{key[0]}!({key[1]})"
            count = last - first + 1
            which = (
                f"invocation {first + 1}"
                if count == 1
                else f"invocations {first + 1} to {last + 1}"
            )
            block_nodes.append(
                Call(
                    comment=f"{arm}, {which}",
                    dest=dest,
                    round=cfg.round,
                    count=count,
                    scalars=(),
                    state=tuple(regs),
                )
            )
            block_nodes += [
                Project(comment=f"{arm}, {which} output", dest=r, call=dest, field=f, optional=True)
                for r, f in zip(regs, fields)
            ]
        main = make(flat.signature, flat.doc, block_nodes, flat.results, arg_fields=flat.arg_fields)
        return list(rounds.values()) + [main]


# -- rounds that agree under a register rotation ---------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RotatedRounds:
    """Full rounds at the instruction ranges `ranges` (inclusive pcs) that agree once their
    registers are renamed and round i's `rhs` limb (`rhs_limb`) is the parameter `param`. The
    renaming of each round is derived, not configured: round 1's is `entry` named by the state's
    fields, and round i's is the bijection under which it matches round 1 (`alpha_match`)
    followed by round 1's.

    Each round begins by loading that limb into `load_register`; the round definition `round`
    takes it as `param` instead, and carries the fields of `roles` in the structure `state`
    (generated, with `state_doc` and, when `value` names the accumulator's limbs, `toNat`
    documented by `value_doc`), passed as `arg`. The last field is the one `load_register`
    fills, so the round returns it and does not read it. The round also takes the block's
    arguments `passed` by name and its word arguments `invariants`. The calls pass `entry`
    rotated per round; `result` lists the round's result registers, in the order of `roles`.
    The comments bind `invariants` as `scalar_comment` and the state as `state_comment`: they are
    the round's parameters, which no Rust operand binds, so the crate's `Target.input_comment`
    does not name them."""

    ranges: tuple[tuple[int, int], ...]
    rhs: str
    rhs_field: str
    entry: tuple[str, ...]
    result: tuple[str, ...]
    round: str
    doc: str
    state: str
    state_doc: str
    roles: tuple[Role, ...]
    arg: str
    load_register: str
    param: str
    scalar_comment: str
    state_comment: str
    value: tuple[str, ...] | None = None
    value_doc: str | None = None
    passed: tuple[str, ...] = ()
    invariants: tuple[str, ...] = ()

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(role.field for role in self.roles)

    @property
    def struct(self) -> str:
        """The Lean declaration of the round's state structure."""
        return state_structure(
            self.state,
            self.state_doc,
            [(r.field, r.doc, r.type) for r in self.roles],
            words_doc="field",
            value=self.value,
            value_doc=self.value_doc,
        )

    def rhs_limb(self, index: int) -> Term:
        """The `rhs` limb that round `index` (from 0) loads: the field `rhs_field.format(i=...)`
        of the argument `rhs`, numbered from 1."""
        return Get(Arg(self.rhs), self.rhs_field.format(i=index + 1))

    def source_round(self, nodes: list[Node], index: int) -> list[Node]:
        """Round `index` (from 0) as the source wrote it, its `rhs` limb read as `param`."""
        first, last = self.ranges[index]
        limb: dict[Term, Term] = {self.rhs_limb(index): Reg(self.param)}
        return [
            n.renamed({}, terms=limb) for n in nodes if n.pc is not None and first <= n.pc <= last
        ]

    def rotations(self, nodes: list[Node]) -> list[dict[str, str]]:
        """Each round's renaming to the canonical names, derived by matching it with round 1."""
        canonical = dict(zip(self.entry, self.fields, strict=False))
        first = self.source_round(nodes, 0)
        renamings = [canonical]
        for index in range(1, len(self.ranges)):
            try:
                match = alpha_match(self.source_round(nodes, index), first)
            except GenerationError as failure:
                raise GenerationError(
                    f"{self.round}: flattened full rounds 1 and {index + 1} differ after "
                    f"accumulator rotation: {failure}"
                ) from failure
            renamings.append({a: canonical.get(b, b) for a, b in match.items()})
        return renamings

    def normalized(self, nodes: list[Node], index: int, rotation: dict[str, str]) -> list[Node]:
        """Round `index` (from 0) under the canonical register names that `rotation` gives it, its
        `rhs` limb as `param`, and its positions relative to its first instruction."""
        first = self.ranges[index][0]
        return [
            dataclasses.replace(n.renamed(rotation), pc=position(n) - first)
            for n in self.source_round(nodes, index)
        ]

    @staticmethod
    def fingerprint(nodes: Sequence[Node]) -> list[tuple[tuple[object, ...], int | None]]:
        """What two rounds must agree on: each node up to its comment, and its position."""
        return [(n.key(), n.pc) for n in nodes]

    def body(self, nodes: list[Node]) -> tuple[Mov, list[Node]]:
        """The round's `rhs` load and its body under canonical names, from round 1, once the
        other rounds are checked to be the same round under their rotations. The body reads the
        limb as `param` where the source read the `load_register` that the load had filled."""
        rotations = self.rotations(nodes)
        rounds = [self.normalized(nodes, i, r) for i, r in enumerate(rotations)]
        if not rounds[0] or any(
            self.fingerprint(r) != self.fingerprint(rounds[0]) for r in rounds[1:]
        ):
            raise GenerationError(
                f"{self.round}: flattened full rounds differ after accumulator rotation"
            )
        first_pc = min(position(n) for n in rounds[0])
        load = next(
            (n for n in rounds[0] if n.pc == first_pc and n.writes() == [self.load_register]),
            None,
        )
        if not isinstance(load, Mov) or load.a != Reg(self.param):
            raise GenerationError(
                f"{self.round}: the round does not begin by loading its RHS limb into "
                f"{self.load_register}"
            )
        # The load aliases `load_register` to `param` only until the source writes it again:
        # later reads must see that new value (the Montgomery quotient), never `param`.
        body: list[Node] = []
        aliases = {self.load_register: self.param}
        for n in rounds[0]:
            if n.pc == first_pc:
                continue
            body.append(n.renamed(aliases, writes=False))
            for name in n.writes():
                aliases.pop(name, None)
        return load, body

    def flattened(
        self, load: Mov, body: list[Node], registers: list[str], rhs_limb: Term, first: int
    ) -> list[Node]:
        """The round instantiated at one call: the `rhs` load, then the body with the call's
        registers for the state's fields and `load_register` for `param`, at the source's
        positions."""
        inverse = dict(zip(self.fields, registers))
        inverse[self.param] = self.load_register
        loaded: list[Node] = [dataclasses.replace(load, a=rhs_limb, pc=first)]
        return loaded + [
            dataclasses.replace(n.renamed(inverse), pc=position(n) + first) for n in body
        ]

    def check_call(
        self,
        nodes: list[Node],
        load: Mov,
        body: list[Node],
        registers: list[str],
        rhs_limb: Term,
        index: int,
    ) -> None:
        """Require the round, instantiated at a call, to be the source's round exactly, so that
        the call's registers and the round's rotation cannot disagree."""
        first, last = self.ranges[index]
        source = [n for n in nodes if n.pc is not None and first <= n.pc <= last]
        if self.fingerprint(
            self.flattened(load, body, registers, rhs_limb, first)
        ) != self.fingerprint(source):
            raise GenerationError(
                f"{self.round}: the factored round at call {index + 1} does not flatten to the "
                f"source's round {index + 1}"
            )

    def reroll(self, flat: Program, unit: Unit) -> list[Program]:
        """The round and the block that calls it, from the `flat` program of the block `unit`."""
        kinds = dict(unit.block.args)
        nodes = flat.nodes
        load, body = self.body(nodes)
        round_nodes: list[Node] = [
            Scalar(comment=self.scalar_comment, dest=name, name=name) for name in self.invariants
        ]
        round_nodes.append(Scalar(comment=load.comment, dest=self.param, name=self.param))
        # The last field is the one the round's load register fills: an output only.
        state_load = unit.target.isa.state_load
        round_nodes += [
            state_load(role.type)(
                comment=self.state_comment, dest=role.field, arg=self.arg, field=role.field
            )
            for role in self.roles[:-1]
        ]
        signature = Signature(
            self.round,
            tuple((a, kinds[a]) for a in self.passed)
            + tuple((a, kinds[a]) for a in self.invariants)
            + ((self.param, "Nat"), (self.arg, self.state)),
            self.state,
        )
        round_program = unit.make(
            signature,
            self.doc,
            round_nodes + body,
            list(self.result),
            arg_fields={
                **{a: flat.arg_fields[a] for a in self.passed},
                self.arg: tuple(role.field for role in self.roles if role.is_word),
            },
            struct=self.struct,
            clear_values=False,
            state_fields=self.fields,
        )

        main_nodes = [n for n in nodes if n.pc is None or n.pc < self.ranges[0][0]]
        current = list(self.entry)
        for index in range(len(self.ranges)):
            rhs_limb = self.rhs_limb(index)
            registers = current + [self.load_register]
            self.check_call(nodes, load, body, registers, rhs_limb, index)
            name = f"round{index + 1}"
            main_nodes.append(
                Call(
                    comment=f"factored round {index + 1}",
                    dest=name,
                    round=self.round,
                    passed=self.passed,
                    scalars=(*(Reg(name) for name in self.invariants), rhs_limb),
                    state=tuple(registers),
                )
            )
            current = current[1:] + current[:1]
            main_nodes += [
                Project(comment=f"round {index + 1} output", dest=r, call=name, field=f)
                for r, f in zip(current + [self.load_register], self.fields)
            ]
        main_nodes += [n for n in nodes if n.pc is not None and n.pc > self.ranges[-1][1]]
        main = unit.make(
            flat.signature,
            flat.doc,
            main_nodes,
            flat.results,
            arg_fields=flat.arg_fields,
            clear_values=False,
        )
        return [round_program, main]
