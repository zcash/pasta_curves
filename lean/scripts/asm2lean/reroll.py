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
- `MacroRounds` (AArch64 inversion): the source invokes a `macro_rules!` macro per step, so the
  repetitions are known from the front end's macro origins; a run of consecutive invocations of
  one arm becomes one call of the round iterated over the run (`round^[n]`).
- `RotatedRounds` (x86-64 `mul`): two full rounds that agree once the accumulator registers are
  rotated, at fixed instruction ranges; the round is checked to flatten back to each of them.
"""

import dataclasses

from .ir import Call, Load, Project, Scalar
from .lean import state_structure


def upward_exposed(nodes, exclude=()):
    """The registers that `nodes` read before writing them, in the order first read."""
    written, exposed = set(), []
    for node in nodes:
        for let in node.lets():
            for r in sorted(let.reads):
                if r not in written and r not in exposed and r not in exclude:
                    exposed.append(r)
            written.add(let.name)
    return exposed


def written(nodes):
    return {let.name for node in nodes for let in node.lets()}


@dataclasses.dataclass(frozen=True)
class Role:
    """A register carried from one round to the next, its field in the round's state structure,
    and the field's docstring."""

    register: str
    field: str
    doc: str | None = None


# -- a prologue, rounds, and an epilogue -----------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PrologueLoop:
    """`end` is the instruction that ends the prologue and each round, `count` the number of
    rounds, `operand` the register holding the round's `rhs` limb (`{i}` is the round number),
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
    value: tuple
    roles: tuple
    emit_struct: bool = True

    def register(self, k):
        return self.operand.format(i=k)

    def reroll(self, block_name, doc, lifter, instructions, limb_args, results, make):
        """The round and the rerolled block, from the flat transcription in `lifter`."""
        nodes = lifter.nodes
        first_pc = min(n.pc for n in nodes if n.pc is not None)
        ends = [p for p in range(first_pc, len(instructions)) if instructions[p].text == self.end]
        if len(ends) != self.count + 1:
            raise ValueError(
                f"{block_name}: expected {self.count + 1} `{self.end}`, found {len(ends)}"
            )
        first = self.register(1)

        def round_body(k):
            rename = {self.register(k): first}
            return [
                (ins.op, tuple(rename.get(t, t) for t in ins.operands))
                for ins in instructions[ends[k - 1] + 1 : ends[k] + 1]
            ]

        if any(round_body(k) != round_body(1) for k in range(2, self.count + 1)):
            raise ValueError(f"{block_name}: the round bodies differ beyond `{first}`")

        def segment(node):
            """0 for the prologue, k for round k, count + 1 for the epilogue."""
            if node.pc is None or node.pc <= ends[0]:
                return 0
            return next((k for k in range(1, self.count + 1) if node.pc <= ends[k]), self.count + 1)

        prologue = [n for n in nodes if segment(n) == 0]
        body = [n for n in nodes if segment(n) == 1]
        after = [n for n in nodes if segment(n) > 1]
        epilogue = [n for n in nodes if segment(n) == self.count + 1]

        inputs = upward_exposed(body, exclude={first})
        body_writes = written(body)
        outputs = [r for r in upward_exposed(after) if r in body_writes]
        regs = [role.register for role in self.roles]
        fields = [role.field for role in self.roles]
        if sorted(outputs) != sorted(regs):
            raise ValueError(
                f"{block_name}: registers carried between rounds are {sorted(outputs)}, "
                f"`roles` lists {sorted(regs)}"
            )
        # An input is an invariant argument if its value on entry is an argument limb or `inv`;
        # otherwise it is carried state.
        defined = {let.name: n for n in prologue for let in n.lets()}
        invariant = [(r, defined[r]) for r in inputs if _is_argument(defined[r])]
        carried = [r for r in inputs if not _is_argument(defined[r])]
        if sorted(carried) != sorted(regs):
            raise ValueError(
                f"{block_name}: registers read from the previous round are {sorted(carried)}, "
                f"`roles` lists {sorted(regs)}"
            )
        args = [
            arg
            for arg in limb_args
            if any(isinstance(n, Load) and n.arg == arg for _, n in invariant)
        ]
        inv = [r for r, n in invariant if isinstance(n, Scalar)]

        def order(item):
            n = item[1]
            return (0, args.index(n.arg), n.field) if isinstance(n, Load) else (1, 0, "")

        round_nodes = (
            [
                dataclasses.replace(n, comment="argument", pc=None)
                for _, n in sorted(invariant, key=order)
            ]
            + [Scalar(comment="argument", dest=first, name=self.param)]
            + [
                Load(comment="argument", dest=r, arg=self.arg, field=f)
                for r, f in zip(regs, fields)
            ]
            + body
        )
        signature = (
            f"def {self.round} ({' '.join(args)} : Limbs) "
            f"({'inv ' if inv else ''}{self.param} : Nat) "
            f"({self.arg} : {self.state}) : {self.state} :="
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
                [(r.register, r.field, r.doc) for r in self.roles],
                words_doc="field",
                value=self.value,
            )
        round_program = make(
            self.round,
            round_doc,
            signature,
            round_nodes,
            regs,
            arg_fields={self.arg: fields},
            struct=struct,
        )

        calls = []
        for k in range(1, self.count + 1):
            name = f"round{k}"
            calls.append(
                Call(
                    comment=f"round {k}",
                    dest=name,
                    head=f"{self.round} {' '.join(args)}",
                    scalars=tuple(inv + [self.register(k)]),
                    state=tuple(regs),
                )
            )
            calls += [
                Project(comment=f"round {k} output", dest=r, call=name, field=f)
                for r, f in zip(regs, fields)
            ]
        block_program = make(
            block_name,
            doc,
            f"def {block_name} ({' '.join(limb_args)} : Limbs) (inv : Nat) : Limbs :=",
            prologue + calls + epilogue,
            results,
        )
        return [round_program, block_program]


def _is_argument(node):
    return isinstance(node, Load) or (isinstance(node, Scalar) and node.is_inv)


# -- runs of macro invocations ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MacroRound:
    """A macro arm transcribed as a round: the definition `round` with docstring `doc`, the
    structure `state` of the registers it carries in and out (passed as `arg`), `roles` for that
    structure, whether this arm's transcription declares it, and the name prefix `call` of a call's
    result. The field `fl` holds the flags; every other field is a word. A round's body may read
    only carried registers."""

    round: str
    doc: str
    state: str
    arg: str
    call: str
    roles: tuple
    emit_struct: bool


class MacroRounds:
    """The arms of `rounds` (keyed by (macro, arm)) become round definitions, and each run of
    consecutive invocations of one arm a call of it iterated over the run; any other arm stays
    expanded in place."""

    def __init__(self, rounds, load_flags):
        self.rounds = rounds
        self.load_flags = load_flags  # the node type loading the flags field

    def reroll(self, block, lifter, origins, results, make, signature, arg_fields):
        name = block.lean_name

        def origin(node):
            return origins[node.pc] if node.pc is not None else None

        def round_of(o):
            return self.rounds.get((o.macro, o.arm)) if o is not None else None

        # Consecutive nodes of one round invocation form a segment.
        segments, current, current_site = [], [], object()
        for node in lifter.nodes:
            o = origin(node)
            site = o.site if round_of(o) else None
            if site != current_site:
                if current:
                    segments.append((current_site, current))
                current, current_site = [], site
            current.append(node)
        if current:
            segments.append((current_site, current))

        # The round definitions, one per arm, from each arm's first invocation.
        rounds = {}
        round_of_site = {}
        for site, nodes in segments:
            if site is None:
                continue
            o = origin(nodes[0])
            key = (o.macro, o.arm)
            cfg = self.rounds[key]
            round_of_site[site] = cfg
            regs = [r.register for r in cfg.roles]
            fields = [r.field for r in cfg.roles]
            inputs, body_writes = upward_exposed(nodes), written(nodes)
            if key in rounds:
                continue
            stray = sorted(set(inputs) - set(regs))
            if stray:
                raise ValueError(f"{name}: {cfg.round} reads {stray}, which `roles` does not carry")
            unused = sorted(set(regs) - set(inputs) - body_writes)
            if unused:
                raise ValueError(f"{name}: {cfg.round} neither reads nor writes {unused}")
            loads = [
                (self.load_flags if f == "fl" else Load)(
                    comment="argument", dest=r, arg=cfg.arg, field=f
                )
                for r, f in zip(regs, fields)
            ]
            struct = None
            if cfg.emit_struct:
                struct = state_structure(
                    cfg.state,
                    f"/-- The registers that `{cfg.round}` carries in and out. -/",
                    [(r.register, r.field, r.doc) for r in cfg.roles],
                    words_doc="word",
                )
            rounds[key] = make(
                cfg.round,
                f"`{o.macro}!({o.arm})`, {cfg.doc}",
                f"def {cfg.round} ({cfg.arg} : {cfg.state}) : {cfg.state} :=",
                loads + nodes,
                regs,
                arg_fields={cfg.arg: [f for f in fields if f != "fl"]},
                struct=struct,
            )

        # The block: a run of consecutive invocations of one arm is one call, the round iterated
        # over the run, and the outputs it carries; a proof about the run is then the step
        # theorem iterated, without one binding per invocation.
        runs = []  # (cfg, key, first site, last site), or (None, None, None, nodes)
        for site, nodes in segments:
            if site is None:
                runs.append((None, None, None, nodes))
                continue
            o = origin(nodes[0])
            key = (o.macro, o.arm)
            if runs and runs[-1][1] == key:
                runs[-1] = (runs[-1][0], key, runs[-1][2], site)
            else:
                runs.append((round_of_site[site], key, site, site))
        block_nodes = []
        for cfg, key, first, last in runs:
            if cfg is None:
                block_nodes += last
                continue
            regs = [r.register for r in cfg.roles]
            fields = [r.field for r in cfg.roles]
            dest = f"{cfg.call}{last + 1}"
            arm = f"{key[0]}!({key[1]})"
            count = last - first + 1
            head = cfg.round if count == 1 else f"{cfg.round}^[{count}]"
            which = (
                f"invocation {first + 1}"
                if count == 1
                else f"invocations {first + 1} to {last + 1}"
            )
            block_nodes.append(
                Call(comment=f"{arm}, {which}", dest=dest, head=head, scalars=(), state=tuple(regs))
            )
            block_nodes += [
                Project(comment=f"{arm}, {which} output", dest=r, call=dest, field=f, optional=True)
                for r, f in zip(regs, fields)
            ]
        main = make(name, block.doc, signature, block_nodes, results, arg_fields=arg_fields)
        return list(rounds.values()) + [main]


# -- rounds that agree under a register rotation ---------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RotatedRounds:
    """Full rounds at the instruction ranges `ranges` (inclusive pcs) that agree once the
    registers are renamed by `rotations[i]` and round i's `rhs` limb `rhs_limb.format(i=...)` is
    the parameter `b`. Each round begins by loading that limb into `rdx`; the round definition
    `round` takes it as `b` instead, carrying `fields` (the last, `q`, only out) in `state`. The
    calls pass `entry` rotated per round. `result` lists the round's result registers, `doc` and
    `signature` its declaration, `struct` its state structure's text."""

    ranges: tuple
    rotations: tuple
    rhs_limb: str
    fields: tuple
    entry: tuple
    result: tuple
    round: str
    doc: str
    signature: str
    struct: str
    load_register: str = "rdx"
    param: str = "b"

    def normalized(self, nodes, index):
        """Round `index` (from 0) under canonical register names, its `rhs` limb as `b`, and its
        positions relative to its first instruction."""
        first, last = self.ranges[index]
        mapping = dict(self.rotations[index])
        mapping[self.rhs_limb.format(i=index + 1)] = self.param
        return [
            dataclasses.replace(n.renamed(mapping), pc=n.pc - first)
            for n in nodes
            if n.pc is not None and first <= n.pc <= last
        ]

    @staticmethod
    def fingerprint(nodes):
        """What two rounds must agree on: each node up to its comment, and its position."""
        return [(n.key(), n.pc) for n in nodes]

    def body(self, nodes, error):
        """The round's `rhs` load and its body under canonical names, from round 1, once round 2
        is checked to be the same round under its rotation. The body reads the limb as the
        parameter `b` where the source read the `rdx` that the load had filled."""
        rounds = [self.normalized(nodes, i) for i in range(len(self.ranges))]
        if not rounds[0] or any(
            self.fingerprint(r) != self.fingerprint(rounds[0]) for r in rounds[1:]
        ):
            raise error("mul: flattened full rounds 1 and 2 differ after accumulator rotation")
        first_pc = min(n.pc for n in rounds[0])
        load = next(
            (n for n in rounds[0] if n.pc == first_pc and n.writes() == [self.load_register]),
            None,
        )
        if load is None or getattr(load, "a", None) != self.param:
            raise error("mul: factored round does not begin by loading its RHS limb into RDX")
        # The load aliases `rdx` to `b` only until the source writes `rdx` again: later reads
        # must see that new value (the Montgomery quotient), never `b`.
        body, aliases = [], {self.load_register: self.param}
        for n in rounds[0]:
            if n.pc == first_pc:
                continue
            body.append(n.renamed(aliases, writes=False))
            for name in n.writes():
                aliases.pop(name, None)
        return load, body

    def flattened(self, load, body, registers, rhs_limb, first):
        """The round instantiated at one call: the `rhs` load, then the body with the call's
        registers for the state's fields and `rdx` for `b`, at the source's positions."""
        inverse = dict(zip(self.fields, registers))
        inverse[self.param] = self.load_register
        return [dataclasses.replace(load, a=rhs_limb, pc=first)] + [
            dataclasses.replace(n.renamed(inverse), pc=n.pc + first) for n in body
        ]

    def check_call(self, nodes, load, body, registers, rhs_limb, index, error):
        """Require the round, instantiated at a call, to be the source's round exactly, so that
        the call's registers and the round's rotation cannot disagree."""
        first, last = self.ranges[index]
        source = [n for n in nodes if n.pc is not None and first <= n.pc <= last]
        if self.fingerprint(
            self.flattened(load, body, registers, rhs_limb, first)
        ) != self.fingerprint(source):
            raise error(
                f"mul: the factored round at call {index + 1} does not flatten to the source's "
                f"round {index + 1}"
            )

    def reroll(self, block, make, error, scalar_comment):
        """The round and the block that calls it, from the flat `block` program."""
        nodes = block.nodes
        load, body = self.body(nodes, error)
        round_nodes = [
            Scalar(comment=scalar_comment, dest="inv", name="inv"),
            Scalar(comment=load.comment, dest=self.param, name=self.param),
        ]
        round_nodes += [
            Load(comment="accumulator argument", dest=f, arg="acc", field=f)
            for f in self.fields
            if f != "q"
        ]
        round_program = make(
            self.round,
            self.doc,
            self.signature,
            round_nodes + body,
            list(self.result),
            arg_fields={"acc": list(self.fields)},
            struct=self.struct,
            clear_values=False,
        )

        main_nodes = [n for n in nodes if n.pc is None or n.pc < self.ranges[0][0]]
        current = list(self.entry)
        for index in range(len(self.ranges)):
            rhs_limb = self.rhs_limb.format(i=index + 1)
            registers = current + [self.load_register]
            self.check_call(nodes, load, body, registers, rhs_limb, index, error)
            name = f"round{index + 1}"
            main_nodes.append(
                Call(
                    comment=f"factored round {index + 1}",
                    dest=name,
                    head=f"{self.round} lhs modulus",
                    scalars=("inv", rhs_limb),
                    state=tuple(registers),
                )
            )
            current = current[1:] + current[:1]
            main_nodes += [
                Project(comment=f"round {index + 1} output", dest=r, call=name, field=f)
                for r, f in zip(current + [self.load_register], self.fields)
            ]
        main_nodes += [n for n in nodes if n.pc is not None and n.pc > self.ranges[-1][1]]
        main = make(
            block.name, block.doc, block.signature, main_nodes, block.results, clear_values=False
        )
        return [round_program, main]
