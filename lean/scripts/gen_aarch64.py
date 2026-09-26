#!/usr/bin/env python3
"""AArch64 backend for the unified Lean transcription generator.

The shared fail-closed Rust/``asm!`` parser feeds this architecture-specific
instruction emitter. It produces

- `lean/PastaAsm/AArch64/Transcription.lean`: the AArch64 blocks as Lean definitions; and
- `lean/PastaAsm/AArch64/Vectors.lean`: kernel-checked AArch64 reference vectors.

Invoke it through ``python3 lean/scripts/gen.py``.

The transcription is deliberately mechanical. A block is read from its template lines, with
the operand placeholders as register names, rebound by each instruction that writes them:
the `in` and `inout` operands bind argument limbs and `inv`, the named `out` and the `inout`
operands are the result limbs, and the block ends as a routine does. The compiler's
allocation of registers to the operands is not modelled; the script checks that every
register the block reads was written by the block or bound by an operand.

A binding that nothing later reads is not emitted. For an operand this records that the
block binds a value it never uses, and the dropped binding is left as a comment; for a carry
flag it is an ordinary unread flag write. A computed register that is never read would be
dead code in the block and is reported as an error, since none is expected.

Run from the repository root:

    python3 lean/scripts/gen.py

`lean/scripts/check.sh` regenerates and fails if the output differs from the committed
files.

The script also generates the mechanical part of each block's correctness proof in
`lean/PastaAsm/AArch64/Spec.lean`: `--skeleton NAME` prints it (see `skeleton`), and
`--check-spec FILE` checks that FILE contains every block's skeleton verbatim once its
`-- BEGIN ... -- END` annotation blocks are removed; the check script runs that too.
Python 3.9+; stdlib only.
"""
import re
from pathlib import Path

import asm_source
import gen

INLINE = Path("src/asm/aarch64.rs")
OUT_PROGRAM = Path("lean/PastaAsm/AArch64/Transcription.lean")

HEADER = """/-
Copyright Supranational LLC (the routines, transcribed from Semolina v0.1.4).
Copyright (c) 2026 the pasta-asm contributors (the transcription).
Released under the Apache License, Version 2.0, as described in the file LICENSE.
-/
"""

# Code longer than this does not set the instruction-comment column (see `Routine.text`).
COMMENT_COLUMN_MAX = 40

# Blocks whose instruction stream Semolina's generator (`pasta_mul-armv8.pl`) emitted as a
# prologue, a loop body repeated a fixed number of times, and an epilogue. The body is
# transcribed once, as a function of the registers that cross its boundary; the block calls it
# once per round. `end` is the instruction that ends the prologue and each body. The rounds
# differ in one place only, which is checked: `operand`, the register holding the round's `rhs`
# limb (`{i}` is the round number), which the round takes as its parameter `param`. `roles`
# lists the registers carried from one round to the next as (register, field, description); the
# structure `state` has those fields, `value` names the ones that make up the accumulator, and
# `emit_struct` says whether this block's transcription declares the structure.
LOOPS = {
    "mulMont": dict(
        end="lsl t3,q,#62", count=3, operand="b{i}",
        round="mulMontRound", state="MulMontAcc", emit_struct=True, param="b", arg="acc",
        value=["r0", "r1", "r2", "r3", "r4"],
        roles=[("r0", "r0", "accumulator limb 0"), ("r1", "r1", "accumulator limb 1"),
               ("r2", "r2", "accumulator limb 2"), ("r3", "r3", "accumulator limb 3"),
               ("r4", "r4", "accumulator limb 4"),
               ("q", "q", "the round's Montgomery quotient `q`"),
               ("t1", "t1", "`low(p1 * q)`, the first term of the round's reduction"),
               ("t3", "t3", "`low(q * 2^62)`, the third term of the round's reduction")]),
}

# The inline `asm!` blocks of the crate: the function whose block to read, the Lean name, the
# docstring, and the argument names in signature order. The block's template lines are the
# instruction stream; its `in` and `inout` operands bind the arguments (`lhs[0]` is `lhs.l0`,
# `inv` is `inv`, and a `let mut a0 = value[0];` before the block makes the `inout` operand
# `a0` the limb `value.l0`), and its named `out` and its `inout` operands are the result limbs.
INLINE_ROUTINES = [
    ("mul", "mulMont",
     "The inline `asm!` block of `mul`: Montgomery multiplication, `lhs * rhs * 2^-256 mod p`, "
     "with the result in the block's output operands. Its rounds are those of Semolina's "
     "`mul_mont_pasta`; its epilogue keeps four limbs of the final candidate.",
     ["lhs", "rhs", "modulus"]),
    ("square", "sqrMont",
     "The inline `asm!` block of `square`: Montgomery squaring, `value^2 * 2^-256 mod p`, the "
     "squaring loop body of Semolina's `sqr_n_mul_mont_pasta` followed by a conditional "
     "subtraction, with the result in the block's `inout` operands.",
     ["value", "modulus"]),
    ("add", "addMod",
     "The inline `asm!` block of `add`: modular addition, `lhs + rhs mod p`, as a full-width "
     "addition, a subtraction of the modulus, and the selection of the reduced sum when that "
     "subtraction did not borrow, with the result in the block's `inout` operands.",
     ["lhs", "rhs", "modulus"]),
    ("sub", "subMod",
     "The inline `asm!` block of `sub`: modular subtraction, `lhs - rhs mod p`, as a full-width "
     "subtraction and the addition of the modulus when it borrowed, with the result in the "
     "block's `inout` operands.",
     ["lhs", "rhs", "modulus"]),
]

def tokenize(rest):
    return re.findall(r"\[[^\]]*\]!?|[^,\s]+", rest)


def imm(tok):
    """An immediate operand such as `#62` or `8*1`, as an integer."""
    s = tok.lstrip("#")
    if re.fullmatch(r"0x[0-9a-fA-F]+", s):
        return int(s, 16)
    if not re.fullmatch(r"-?[0-9]+(\*[0-9]+)?", s):
        raise ValueError(f"unexpected immediate {tok}")
    return eval(s)


class Emitter(gen.Emitter):
    """AArch64 instruction decoder backed by the shared binding and liveness IR."""

    def __init__(self, ins, directions=None):
        super().__init__()
        self.ins = ins
        self.directions = directions or {}

    def read(self, tok):
        if tok == "xzr":
            return "0"
        if tok.startswith("#"):
            return str(imm(tok))
        if tok not in self.known:
            raise ValueError(f"{tok} read before being written")
        self.cur_reads.add(tok)
        return tok

    def bind(self, name, expr, comment, reads=None, load=False, fact=None, note=None):
        if name != "xzr":
            super().bind(name, expr, comment, reads=reads, load=load, fact=fact, note=note)

    def step(self, op, t, text):
        arity = {"mov": 2, "mul": 3, "umulh": 3, "lsl": 3, "lsr": 3,
                 "adds": 3, "adcs": 3, "adc": 3, "subs": 3, "sbcs": 3, "csel": 4}
        if op not in arity:
            raise ValueError(f"unhandled instruction: {text}")
        if len(t) != arity[op]:
            raise ValueError(f"{op} expects {arity[op]} operands, got {len(t)}: {text}")
        if t[0] != "xzr":
            if self.directions.get(t[0]) == "in":
                raise ValueError(f"input-only register {t[0]} cannot be written: {text}")
            if t[0] not in self.directions:
                raise ValueError(f"undeclared destination {t[0]}: {text}")
        self.cur_reads = set()
        if op == "mov":
            a = self.read(t[1])
            self.bind(t[0], a, text, fact=("mov", a))
        elif op == "mul":
            a, b = self.read(t[1]), self.read(t[2])
            self.bind(t[0], f"mulLo {a} {b}", text, fact=("mul", a, b))
        elif op == "umulh":
            a, b = self.read(t[1]), self.read(t[2])
            self.bind(t[0], f"umulh {a} {b}", text, fact=("umulh", a, b))
        elif op == "lsl":
            a, k = self.read(t[1]), imm(t[2])
            self.bind(t[0], f"lsl {a} {k}", text, fact=("lsl", a, k))
        elif op == "lsr":
            a, k = self.read(t[1]), imm(t[2])
            self.bind(t[0], f"lsr {a} {k}", text, fact=("lsr", a, k))
        elif op in ("adds", "adcs", "adc"):
            cin = "0" if op == "adds" else self.read("c")
            a, b = self.read(t[1]), self.read(t[2])
            expr = f"addc {a} {b} {cin}"
            if op == "adc":
                self.bind(t[0], f"({expr}).1", text, fact=("adc", a, b, cin))
            else:
                self.bind("s", expr, text, fact=("adds", a, b, cin))
                self.bind(t[0], "s.1", text, reads=("s",), fact=("fst",), note=f"  `-> {t[0]}")
                self.bind("c", "s.2", text, reads=("s",), fact=("snd",), note="  `-> carry")
        elif op in ("subs", "sbcs"):
            cin = "1" if op == "subs" else self.read("c")
            a, b = self.read(t[1]), self.read(t[2])
            expr = f"subc {a} {b} {cin}"
            if t[0] == "xzr":
                self.bind("c", f"({expr}).2", text, fact=("subs_carry", a, b, cin))
            else:
                self.bind("s", expr, text, fact=("subs", a, b, cin))
                self.bind(t[0], "s.1", text, reads=("s",), fact=("fst",), note=f"  `-> {t[0]}")
                self.bind("c", "s.2", text, reads=("s",), fact=("snd",), note="  `-> carry")
        elif op == "csel":
            c, a, b = self.read("c"), self.read(t[1]), self.read(t[2])
            if t[3] in ("lo", "cc"):
                self.bind(t[0], f"cselLo {c} {a} {b}", text, fact=("select", c, a, b))
            elif t[3] in ("cs", "hs"):
                self.bind(t[0], f"cselCs {c} {a} {b}", text, fact=("select", c, b, a))
            else:
                raise ValueError(f"unexpected condition: {text}")
        else:
            raise ValueError(f"unhandled instruction: {text}")

    def run(self, start):
        pc = start
        while True:
            op, tokens, text = self.ins[pc]
            if op == "ret":
                self.end_pc = pc
                return
            self.pc = pc
            self.step(op, tokens, text)
            pc += 1


# Fields of the argument structures, by argument name, for the skeleton's bound hypotheses.
LIMB_FIELDS = ["l0", "l1", "l2", "l3"]
ARG_FIELDS = {arg: LIMB_FIELDS for arg in ("t", "lhs", "rhs", "value", "modulus")}


def loop_routines(e, ins, name, doc, args, result, cfg):
    """Split the transcription of a routine that the assembly's generator emitted as prologue,
    repeated body, and epilogue (see `LOOPS`): the body becomes its own definition over the
    registers that cross its boundary, and the routine calls it once per round."""
    first_pc = min(en["pc"] for en in e.entries if en["pc"] is not None)
    ends = [p for p in range(first_pc, e.end_pc) if ins[p][2] == cfg["end"]]
    if len(ends) != cfg["count"] + 1:
        raise ValueError(f"{name}: expected {cfg['count'] + 1} `{cfg['end']}`, found {len(ends)}")
    count = cfg["count"]
    template = cfg["operand"]
    varying = [None] + [template.format(i=k) for k in range(1, count + 1)]
    bodies = []
    for k in range(1, count + 1):
        texts = [ins[p][2] for p in range(ends[k - 1] + 1, ends[k] + 1)]
        texts = [re.sub(rf"\b{re.escape(varying[k])}\b", varying[1], t) for t in texts]
        bodies.append(texts)
    if any(b != bodies[0] for b in bodies):
        raise ValueError(f"{name}: the round bodies differ beyond `{varying[1]}`")

    def part(en):
        pc = en["pc"]
        if pc is None or pc <= ends[0]:
            return "prologue"
        for k in range(1, count + 1):
            if ends[k - 1] < pc <= ends[k]:
                return ("body", k)
        return "epilogue"

    prologue = [en for en in e.entries if part(en) == "prologue"]
    body = [en for en in e.entries if part(en) == ("body", 1)]
    after = [en for en in e.entries if part(en) not in ("prologue", ("body", 1))]
    # The register holding the round's `rhs` limb: the operand.
    loaded = varying[1]
    body_entries = body
    # Registers the body reads before writing (its inputs), and registers it writes that a
    # later instruction reads before they are written again (its outputs).
    bound, live_in = set(), []
    for en in body_entries:
        for r in sorted(en["reads"]):
            if r not in bound and r not in live_in and r != loaded:
                live_in.append(r)
        bound.add(en["name"])
    rebound, live_out = set(), []
    for en in after:
        for r in sorted(en["reads"]):
            if r in bound and r not in rebound and r not in live_out:
                live_out.append(r)
        rebound.add(en["name"])
    regs = [r for r, _, _ in cfg["roles"]]
    fields = [f for _, f, _ in cfg["roles"]]
    if sorted(live_out) != sorted(regs):
        raise ValueError(f"{name}: registers carried between rounds are {sorted(live_out)}, "
                         f"`roles` lists {sorted(regs)}")
    # An input is an invariant argument if its value on entry is an argument limb or `inv`;
    # otherwise it is carried state.
    defs = {}
    for en in prologue:
        defs[en["name"]] = en
    invariant, carried = [], []
    for r in live_in:
        d = defs[r]
        if d["fact"][0] in ("load", "inv"):
            invariant.append((r, d))
        else:
            carried.append(r)
    if sorted(carried) != sorted(regs):
        raise ValueError(f"{name}: registers read from the previous round are {sorted(carried)}, "
                         f"`roles` lists {sorted(regs)}")
    limb_args = [arg for arg in args if any(d["fact"][0] == "load" and d["fact"][1] == arg
                                            for _, d in invariant)]
    uses_inv = any(d["fact"][0] == "inv" for _, d in invariant)
    param, sarg = cfg["param"], cfg["arg"]
    # The round: its inputs bound as arguments, then the body.
    re_ = Emitter(ins)
    order = {arg: i for i, arg in enumerate(limb_args)}

    def key(rd):
        d = rd[1]
        return (0, order[d["fact"][1]], d["fact"][2]) if d["fact"][0] == "load" else (1, 0, "")

    for r, d in sorted(invariant, key=key):
        re_.bind(r, d["expr"], "argument", reads=(), load=(d["fact"][0] == "load"), fact=d["fact"])
    re_.bind(loaded, param, "argument", reads=(), fact=("param", param))
    for r, f in zip(regs, fields):
        re_.bind(r, f"{sarg}.{f}", "argument", reads=(), load=True, fact=("load", sarg, f))
    re_.entries += [dict(en) for en in body_entries]
    re_.cur_reads = set()
    round_result = [re_.read(r) for r in regs]
    sig = (f"def {cfg['round']} ({' '.join(limb_args)} : Limbs) "
           f"({'inv ' if uses_inv else ''}{param} : Nat) "
           f"({sarg} : {cfg['state']}) : {cfg['state']} :=")
    where = f"the instructions between one `{cfg['end']}` and the next"
    round_doc = (f"One round of `{name}`: {where} in each of its {count} rounds, on the "
                 f"round's `rhs` limb `{param}` and the registers `{sarg}` carried from the "
                 "previous round.")
    struct = None
    if cfg["emit_struct"]:
        value = cfg["value"]
        lines = [f"/-- The registers that `{name}` carries from one round to the next. -/",
                 f"structure {cfg['state']} where"]
        for _, f, desc in cfg["roles"]:
            lines += [f"  /-- {desc} -/", f"  {f} : Nat"]
        lines += ["  deriving DecidableEq, Repr", "", f"namespace {cfg['state']}", "",
                  "/-- Every field is below `2^64`. -/",
                  f"def Bounded (s : {cfg['state']}) : Prop :="]
        lines += wrap_tactic("", [f"s.{f} < 2^64" + (" ∧" if i < len(fields) - 1 else "")
                                  for i, f in enumerate(fields)], "", indent="  ")
        lines += ["", f"/-- The accumulator's value: limbs `{'`, `'.join(value)}` with weights "
                  f"`2^0` to `2^{64 * (len(value) - 1)}`. -/",
                  f"def toNat (s : {cfg['state']}) : Nat :="]
        terms = [f"{'2^%d * ' % (64 * i) if i else ''}s.{f}" for i, f in enumerate(value)]
        lines += wrap_tactic("", [t + (" +" if i < len(value) - 1 else "")
                                  for i, t in enumerate(terms)], "", indent="  ")
        lines += ["", f"end {cfg['state']}", ""]
        struct = "\n".join(lines)
    rnd = Routine(round_doc, sig, re_.render(round_result), f"  ⟨{', '.join(round_result)}⟩",
                  cfg["round"], re_, round_result, struct=struct,
                  arg_fields={sarg: fields})
    # The block: prologue, then per round the call and the outputs.
    me = Emitter(ins)
    me.entries = [dict(en) for en in prologue]
    inv_regs = [r for r, _ in invariant if defs[r]["fact"][0] == "inv"]
    n_scalar = len(inv_regs) + 1
    fmt = (f"{cfg['round']} {' '.join(limb_args)} " + " ".join("{%d}" % i for i in range(n_scalar))
           + " ⟨" + ", ".join("{%d}" % (n_scalar + i) for i in range(len(regs))) + "⟩")
    for k in range(1, count + 1):
        call_regs = inv_regs + [varying[k]] + regs
        rname = f"round{k}"
        me.entries.append(dict(name=rname, expr=fmt.format(*call_regs), comment=f"round {k}",
                               reads=set(call_regs), load=False, fact=("call", fmt, call_regs),
                               pc=None))
        for r, f in zip(regs, fields):
            me.entries.append(dict(name=r, expr=f"{rname}.{f}", comment=f"round {k} output",
                                   reads={rname}, load=False, fact=("callout", rname, f), pc=None))
    me.entries += [dict(en) for en in e.entries if part(en) == "epilogue"]
    main = Routine(doc, f"def {name} ({' '.join(args)} : Limbs) (inv : Nat) : Limbs :=",
                   me.render(result), f"  ⟨{', '.join(result)}⟩", name, me, result)
    return [rnd, main]


def parse_inline(path, fn, args):
    """Adapt the shared Rust asm parser to the AArch64 emitter's instruction representation."""
    rust_args = args + (["inv"] if fn in ("mul", "square") else [])
    parsed = asm_source.parse_function(
        path.read_text(), fn, rust_args, 4,
        allowed_options={"pure", "nomem", "nostack"},
        required_options={"pure", "nomem", "nostack"},
    )
    ins = []
    for template in parsed.instructions:
        text = re.sub(r"\{(\w+)\}", r"\1", template)
        text = re.sub(r"\s+", " ", text.replace(", ", ",")).strip()
        match = re.match(r"(\S+)\s*(.*)", text)
        ins.append((match.group(1), tokenize(match.group(2)), text))
    ins.append(("ret", [], "ret"))
    asm_source.declaration_directions(parsed, fn)
    outputs = asm_source.output_bindings(parsed, fn)
    returned = asm_source.returned_registers(parsed, fn)
    decls = [(decl.name, decl.kind, decl.value) for decl in parsed.declarations]
    return ins, decls, parsed.locals, outputs, returned


def emit_inline(fn, name, doc, args):
    ins, decls, lets, named_outputs, returned = parse_inline(INLINE, fn, args)
    e = Emitter(ins, {n: kind for n, kind, _ in decls})
    outs = []
    for n, kind, v in decls:
        if kind in ("in", "inout"):
            m = re.fullmatch(r"(\w+)\[(\d)\]", v)
            if m:
                arg, i = m.group(1), m.group(2)
            elif v in lets:
                arg, i = lets[v][0], str(lets[v][1])
            elif v == "inv":
                e.bind(n, "inv", "argument", reads=(), fact=("inv",))
                continue
            else:
                raise ValueError(f"{name}: unexpected input operand {n} = {v}")
            if arg not in args:
                raise ValueError(f"{name}: operand {n} reads {v}, not an argument")
            e.bind(n, f"{arg}.l{i}", "argument", reads=(), load=True, fact=("load", arg, f"l{i}"))
            if kind == "inout":
                outs.append((n, n))
        elif kind == "out":
            if v != "_":
                if named_outputs.get(v) != n:
                    raise ValueError(f"{name}: output binding {v} does not name operand {n}")
                outs.append((v, n))
        else:
            raise ValueError(f"{name}: unsupported operand direction {kind}")
    ordered_outputs = tuple(reg for _, reg in sorted(outs))
    if ordered_outputs != returned:
        raise ValueError(
            f"{name}: source output order {returned} differs from the order of the output "
            f"names {ordered_outputs}"
        )
    e.run(0)
    e.cur_reads = set()
    result = [e.read(reg) for reg in ordered_outputs]
    if len(result) != 4:
        raise ValueError(f"{name}: {len(result)} output operands")
    if name in LOOPS:
        return loop_routines(e, ins, name, doc, args, result, LOOPS[name])
    uses_inv = any(kind in ("in", "inout") and v == "inv" for _, kind, v in decls)
    return [Routine(doc, f"def {name} ({' '.join(args)} : Limbs){' (inv : Nat)' if uses_inv else ''} : Limbs :=",
                    e.render(result), f"  ⟨{', '.join(result)}⟩", name, e, result)]


class SkeletonBackend(gen.SkeletonBackend):
    """AArch64 grouping and proof facts used by the shared skeleton traversal."""

    def prepare(self, emitter, entries):
        entries = [dict(entry) for entry in entries]
        names = gen.ssa_names(entries)
        i = 0
        while i < len(entries):
            kind = entries[i]["fact"][0]
            if kind not in ("adds", "subs"):
                i += 1
                continue
            # An `adds`/`subs` whose carry nothing reads has no carry binding (the inline
            # block's last shift), so its group is the pair and the carry is a ghost, as for
            # `adc`.
            dead_carry = (i + 2 >= len(entries) or entries[i + 2]["fact"] != ("snd",))
            group_count = 2 if dead_carry else 3
            entries[i]["group"] = list(zip(entries[i:i + group_count], names[i:i + group_count]))
            entries[i]["group_label"] = names[i + 1]
            entries[i]["dead_carry"] = dead_carry
            i += group_count
        return gen.SkeletonPreparation(entries, names)

    def fact(self, kind, ops, context):
        if kind in ("adds", "subs"):
            a, b, cin = ops
            i, entries = context.index, context.entries
            xn = context.group_names[1]
            dead_carry = context.entry["dead_carry"]
            cn = f"k_{xn}" if dead_carry else context.group_names[2]
            if kind == "adds":
                val = f"({a} + {b} + {cin})"
                lin = f"{xn} + 2^64 * {cn} = {a} + {b} + {cin}"
                lin_proof = "Nat.mod_add_div _ _"
                carry_proof = f"addc_carry_le_one {a} {b} {cin} {context.lt64(a)} {context.lt64(b)} {context.le1(cin)}"
            else:
                val = f"({a} + 2^64 - {b} - (1 - {cin}))"
                lin = f"{xn} + 2^64 * {cn} + {b} + 1 = {a} + 2^64 + {cin}"
                lin_proof = f"subc_lin {a} {b} {cin} {context.lt64(b)} {context.le1(cin)}"
                carry_proof = f"subc_carry_le_one {a} {b} {cin} {context.lt64(a)}"
            context.eq(xn, f"{val} % 2^64")
            if dead_carry:
                context.lines.append(f"  have b_{xn} : {xn} < 2^64 := by rw [e_{xn}]; "
                                     "exact Nat.mod_lt _ (Nat.two_pow_pos _)")
                context.lines.append(f"  obtain ⟨{cn}, b_{cn}, l_{xn}⟩ :")
                context.lines.append(f"      ∃ k, k ≤ 1 ∧ {lin.replace(cn, 'k')} :=")
                context.lines.append(f"    ⟨{val} / 2^64, {carry_proof},")
                context.lines.append(f"      by rw [e_{xn}]; exact {lin_proof}⟩")
                context.lines.append(f"  clear e_{xn}")
                context.ren[entries[i + 1]["name"]] = xn
                context.bnd[xn] = f"b_{xn}"
                context.consumed = 2
            else:
                context.eq(cn, f"{val} / 2^64")
                context.lines.append(f"  have l_{xn} : {lin} := by")
                context.lines.append(f"    rw [e_{xn}, e_{cn}]; exact {lin_proof}")
                context.lines.append(f"  have b_{xn} : {xn} < 2^64 := by rw [e_{xn}]; exact Nat.mod_lt _ (Nat.two_pow_pos _)")
                context.lines.append(f"  have b_{cn} : {cn} ≤ 1 := by")
                context.lines.append(f"    rw [e_{cn}]; exact {carry_proof}")
                context.lines.append(f"  clear e_{xn} e_{cn}")
                context.ren[entries[i + 1]["name"]] = xn
                context.ren[entries[i + 2]["name"]] = cn
                context.bnd[xn], context.bnd[cn] = f"b_{xn}", f"b_{cn}"
                context.unit_bound.add(cn)
                context.consumed = 3
            return True
        if kind == "subs_carry":
            a, b, cin = ops
            nm = context.name
            context.eq(nm, f"({a} + 2^64 - {b} - (1 - {cin})) / 2^64")
            context.lines.append(f"  have b_{nm} : {nm} ≤ 1 := by rw [e_{nm}]; exact subc_carry_le_one {a} {b} {cin} {context.lt64(a)}")
            context.lines.append(f"  have l_{nm} : ({nm} = 1 ∧ {b} + 1 ≤ {a} + {cin}) ∨ ({nm} = 0 ∧ {a} + {cin} < {b} + 1) :=")
            context.lines.append(f"    subc_carry_cases {a} {b} {cin} _ e_{nm} {context.lt64(a)} {context.lt64(b)} {context.le1(cin)}")
            context.lines.append(f"  clear e_{nm}")
            context.bnd[nm] = f"b_{nm}"
            context.unit_bound.add(nm)
            return True
        return False


SKELETON_BACKEND = SkeletonBackend()


class Routine(gen.Routine):
    architecture = "AArch64"
    skeleton_backend = SKELETON_BACKEND


wrap_tactic = gen.wrap_tactic


def gen_program():
    parts = [HEADER, "import PastaAsm.AArch64.Semantics\n", """
/-!
# The crate's inline Pasta field blocks, transcribed

GENERATED by `lean/scripts/gen.py` from the `asm!` blocks of `mul`, `square`, `add`, and `sub`
in `src/asm/aarch64.rs`; do not edit by hand. Each definition follows its block instruction by
instruction (the instruction is the trailing comment; the two lines that unpack an
instruction's (result, carry) pair are marked as its continuation), over the semantics of
`PastaAsm.AArch64.Semantics`. Registers are rebound by the instructions that write them, `c`
is the carry flag, `s` is the (result, carry) pair of the instruction that last set both,
argument limbs are read where the block's operands bind them, and the output limbs are bound
where the block's output operands hold them. Bindings that nothing reads are left as
comments. See the generator's docstring for what it checks.
-/

namespace PastaAsm.AArch64

"""]
    routines = all_routines()
    # One comment column for the whole file: two spaces past the widest ordinary `let`. Lines
    # longer than COMMENT_COLUMN_MAX are outliers (the round calls) and do not set the column.
    column = 2 + max(len(code) for r in routines for code, _ in r.lines
                     if code is not None and len(code) <= COMMENT_COLUMN_MAX)
    parts.append("\n".join(r.text(column) for r in routines))
    parts.append("\nend PastaAsm.AArch64\n")
    return "".join(parts)


# Proof skeleton construction and checking are shared in gen.py.

def all_routines():
    routines = []
    for fn, name, doc, args in INLINE_ROUTINES:
        routines += emit_inline(fn, name, doc, args)
    return routines


def generated_outputs():
    """Return the AArch64 generated paths and contents without writing files."""
    return [(OUT_PROGRAM, gen_program())]
