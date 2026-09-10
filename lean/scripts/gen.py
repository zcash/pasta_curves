#!/usr/bin/env python3
"""Generate the Lean transcription of the crate's inline AArch64 Pasta Montgomery blocks.

Reads the inline `asm!` blocks of `mul` and `square` in `src/asm/aarch64.rs`, and writes

- `lean/PastaAArch64Asm/Transcription.lean`: each block as a Lean definition over the
  instruction semantics of `PastaAArch64Asm.Semantics`, one `let` per instruction, in the
  block's order, with the instruction as a trailing comment;
- `lean/PastaAArch64Asm/Vectors.lean`: one kernel-checked example per line of
  `test-vectors/pasta_mul-armv8-vectors.txt` whose operands are inside the proved
  contracts, the outputs of the real routines on an Apple M-series machine.

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
`lean/PastaAArch64Asm/Spec.lean`: `--skeleton NAME` prints it (see `skeleton`), and
`--check-spec FILE` checks that FILE contains every block's skeleton verbatim once its
`-- BEGIN ... -- END` annotation blocks are removed; the check script runs that too.
Python 3.9+; stdlib only.
"""
import re
import sys
import textwrap
from pathlib import Path

INLINE = Path("src/asm/aarch64.rs")
VECTORS = Path("test-vectors/pasta_mul-armv8-vectors.txt")
OUT_PROGRAM = Path("lean/PastaAArch64Asm/Transcription.lean")
OUT_VECTORS = Path("lean/PastaAArch64Asm/Vectors.lean")

HEADER = """/-
Copyright Supranational LLC (the routines, transcribed from Semolina v0.1.4).
Copyright (c) 2026 the pasta-aarch64-asm contributors (the transcription).
Released under the Apache License, Version 2.0, as described in the file LICENSE.
-/
"""

HEADER_VECTORS = """/-
Copyright (c) 2026 the pasta-aarch64-asm contributors.
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


class Emitter:
    """Transcribes one block, from its first instruction to the appended `ret`, into Lean `let`
    bindings."""

    def __init__(self, ins):
        self.ins = ins
        self.entries = []   # dicts: name, expr, comment, reads, load (bool)
        self.known = set()  # names holding a value the program may read
        self.cur_reads = set()
        self.pc = None      # index of the instruction being transcribed (None: an argument)

    # -- operands ----------------------------------------------------------------

    def read(self, tok):
        if tok == "xzr":
            return "0"
        if tok.startswith("#"):
            return str(imm(tok))
        if tok not in self.known:
            raise ValueError(f"{tok} read before being written")
        self.cur_reads.add(tok)
        return tok

    def bind(self, name, expr, comment, reads=None, load=False, fact=None):
        """Record a binding. `fact` is the skeleton's description of it: a tuple whose head
        names the kind of instruction and whose remaining items are the operand names."""
        if name == "xzr":
            return
        self.known.add(name)
        self.entries.append(dict(name=name, expr=expr, comment=comment,
                                 reads=set(self.cur_reads) if reads is None else set(reads),
                                 load=load, fact=fact, pc=self.pc))

    # -- instructions --------------------------------------------------------------

    def step(self, op, t, text):
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
                self.bind(t[0], "s.1", text, reads=("s",), fact=("fst",))
                self.bind("c", "s.2", text, reads=("s",), fact=("snd",))
        elif op in ("subs", "sbcs"):
            cin = "1" if op == "subs" else self.read("c")
            a, b = self.read(t[1]), self.read(t[2])
            expr = f"subc {a} {b} {cin}"
            if t[0] == "xzr":
                self.bind("c", f"({expr}).2", text, fact=("subs_carry", a, b, cin))
            else:
                self.bind("s", expr, text, fact=("subs", a, b, cin))
                self.bind(t[0], "s.1", text, reads=("s",), fact=("fst",))
                self.bind("c", "s.2", text, reads=("s",), fact=("snd",))
        elif op == "csel":
            if t[3] != "lo":
                raise ValueError(f"unexpected condition: {text}")
            c, a, b = self.read("c"), self.read(t[1]), self.read(t[2])
            self.bind(t[0], f"cselLo {c} {a} {b}", text, fact=("csel", c, a, b))
        else:
            raise ValueError(f"unhandled instruction: {text}")

    def run(self, start):
        pc = start
        while True:
            op, t, text = self.ins[pc]
            if op == "ret":
                self.end_pc = pc
                return
            self.pc = pc
            self.step(op, t, text)
            pc += 1

    # -- output ------------------------------------------------------------------------

    def liveness(self, result_names):
        """Which entries something later reads, by a backward pass from the result names."""
        needed = set(result_names)
        live = [False] * len(self.entries)
        for i in range(len(self.entries) - 1, -1, -1):
            e = self.entries[i]
            if e["name"] in needed:
                live[i] = True
                needed.discard(e["name"])
                needed |= e["reads"]
        return live

    def render(self, result_names):
        """The `let` lines, with bindings that nothing reads dropped (see the module doc)."""
        live = self.liveness(result_names)
        lines = []  # (code, comment): a `let` with its instruction, or (None, whole-line comment)
        for e, keep in zip(self.entries, live):
            if keep:
                lines.append((f"  let {e['name']} := {e['expr']}", e["comment"]))
            elif e["load"]:
                lines.append((None, f"  -- {e['comment']}: {e['name']} = {e['expr']} is never read"))
            elif e["name"] != "c":
                raise ValueError(f"dead computation: {e['name']} := {e['expr']} ({e['comment']})")
        return lines


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
    ARG_FIELDS[sarg] = fields
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
                  cfg["round"], re_, round_result, struct=struct)
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


def parse_inline(path, fn):
    """The instruction list of the `asm!` block of function `fn` in the crate's root module
    (operand placeholders become register names; a `ret` is appended), the block's operand
    declarations as (name, direction, expression), and the `let mut <name> = <arg>[<i>];`
    bindings made before the block, as {name: (arg, i)}."""
    src = path.read_text()
    found = re.search(rf"^\s*pub(?:\([^)]*\))? fn {fn}\(", src, re.MULTILINE)
    if found is None:
        raise ValueError(f"{path}: no `fn {fn}(`")
    start = found.start()
    blk = src[start:]
    blk = blk[:blk.index("options(")]
    lets = {n: (arg, int(i)) for n, arg, i in re.findall(r"let mut (\w+) = (\w+)\[(\d)\];", blk)}
    ins = []
    for t in re.findall(r'^\s*"([^"]*)",', blk, re.M):
        text = re.sub(r"\{(\w+)\}", r"\1", t)
        text = re.sub(r"\s+", " ", text.replace(", ", ",")).strip()
        m = re.match(r"(\S+)\s*(.*)", text)
        ins.append((m.group(1), tokenize(m.group(2)), text))
    ins.append(("ret", [], "ret"))
    decls = re.findall(r'^\s*(\w+) = (in|out|inout)\(reg\) ([^,]+),', blk, re.M)
    return ins, decls, lets


def emit_inline(fn, name, doc, args):
    ins, decls, lets = parse_inline(INLINE, fn)
    e = Emitter(ins)
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
                outs.append((v, n))
        else:
            raise ValueError(f"{name}: unsupported operand direction {kind}")
    e.run(0)
    e.cur_reads = set()
    result = [e.read(reg) for _, reg in sorted(outs)]
    if len(result) != 4:
        raise ValueError(f"{name}: {len(result)} output operands")
    if name in LOOPS:
        return loop_routines(e, ins, name, doc, args, result, LOOPS[name])
    return [Routine(doc, f"def {name} ({' '.join(args)} : Limbs) (inv : Nat) : Limbs :=",
                    e.render(result), f"  ⟨{', '.join(result)}⟩", name, e, result)]


class Routine:
    """A transcribed routine: docstring, signature line, body lines, and result line, plus the
    emitter and result names for the proof skeleton, and for a round definition the text of its
    state structure."""

    def __init__(self, doc, signature, lines, result, name, emitter, result_names, struct=None):
        self.doc, self.signature, self.lines, self.result = doc, signature, lines, result
        self.name, self.emitter, self.result_names = name, emitter, result_names
        self.struct = struct

    def text(self, column):
        """The definition, with the instruction comments aligned at `column`; a line whose code
        reaches the column (an outlier, such as the helper call) gets its comment two spaces
        after the code instead."""
        body = []
        for code, comment in self.lines:
            if code is None:
                body.append(comment)
            else:
                body.append(f"{code.ljust(column) if len(code) + 2 <= column else code + '  '}-- {comment}")
        head = f"{self.struct}\n" if self.struct else ""
        return head + f"{docstring(self.doc)}\n{self.signature}\n" + "\n".join(body) + f"\n{self.result}\n"


def docstring(text, width=100):
    """A `/-- ... -/` docstring wrapped to the repository's line width."""
    lines = textwrap.TextWrapper(width=width, break_long_words=False, break_on_hyphens=False,
                                 initial_indent="/-- ").wrap(text)
    if len(lines[-1]) + 3 <= width:
        lines[-1] += " -/"
    else:
        lines.append("-/")
    return "\n".join(lines)


def gen_program():
    parts = [HEADER, "import PastaAArch64Asm.Semantics\n", """
/-!
# The crate's inline Pasta Montgomery blocks, transcribed

GENERATED by `lean/scripts/gen.py` from the `asm!` blocks of `mul` and `square` in
`src/asm/aarch64.rs`; do not edit by hand. Each definition follows its block instruction by
instruction (the instruction is the trailing comment), over the semantics of
`PastaAArch64Asm.Semantics`: registers are rebound by the instructions that write them, `c`
is the carry flag, `s` is the (result, carry) pair of the instruction that last set both,
argument limbs are read where the block's operands bind them, and the output limbs are bound
where the block's output operands hold them. Bindings that nothing reads are left as
comments. See the generator's docstring for what it checks.
-/

namespace PastaAArch64Asm

"""]
    routines = all_routines()
    # One comment column for the whole file: two spaces past the widest ordinary `let`. Lines
    # longer than COMMENT_COLUMN_MAX are outliers (the round calls) and do not set the column.
    column = 2 + max(len(code) for r in routines for code, _ in r.lines
                     if code is not None and len(code) <= COMMENT_COLUMN_MAX)
    parts.append("\n".join(r.text(column) for r in routines))
    parts.append("\nend PastaAArch64Asm\n")
    return "".join(parts)


FIELDS = {
    "Fp": ("pallasBase", "⟨0x992d30ed00000001, 0x224698fc094cf91b, 0, 0x4000000000000000⟩",
           "0x992d30ecffffffff"),
    "Fq": ("vestaBase", "⟨0x8c46eb2100000001, 0x224698fc0994a8dd, 0, 0x4000000000000000⟩",
           "0x8c46eb20ffffffff"),
}


# The two moduli as integers, for classifying the vectors' operands.
MODULUS_INT = {
    "Fp": 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001,
    "Fq": 0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001,
}


def in_contract(op, key, operands):
    """Whether the vector's operands are inside the contract the proofs establish: for the
    multiplication, a canonical left operand, or a canonical right operand whose limbs 1 to 3
    are at most `2^64 - 3`; for the squaring, a canonical input; for the conversion, any."""
    p = MODULUS_INT[key]
    if op == "MUL":
        lhs, rhs = operands
        limbs = [(rhs >> (64 * i)) % 2**64 for i in range(1, 4)]
        return lhs < p or (rhs < p and all(l <= 2**64 - 3 for l in limbs))
    if op == "SQR":
        return operands[0] < p
    return True


def gen_vectors(lines):
    out = [HEADER_VECTORS, "import PastaAArch64Asm.Compositions\n", """
/-!
# Reference vectors for the transcribed blocks

GENERATED by `lean/scripts/gen.py` from `test-vectors/pasta_mul-armv8-vectors.txt`; do not
edit by hand. Each vector is the output of the real assembly (Semolina's `mul_mont_pasta`,
`sqr_mont_pasta`, and `from_mont_pasta`, as vendored by pasta_curves at
`8ad85e9fab7929f6236960e472f432a4bd9ccd74` and run on an Apple M-series machine) on the
given operands, and each example asks the kernel to evaluate the transcription on the same
operands. The multiplication and squaring examples exercise the inline blocks, which
transcribe those routines; the conversion examples exercise `fromMont`, the multiplication
block with `1` as its right operand, as the crate composes it. The vectors file also records
the routines' outputs on operands outside the proved contracts (unreduced operands), where the
block's dropped fifth limb can change the result, and those are left out here, with their
number recorded at the end.

The modulus limbs and `inv` are the crate's constants for its `Fp` (the Pallas base field)
and `Fq` (the Vesta base field).
-/

namespace PastaAArch64Asm

"""]
    for key, (prefix, limbs, inv) in FIELDS.items():
        field = "Pallas" if key == "Fp" else "Vesta"
        out.append(f"/-- The {field} base field modulus, as the crate's `MODULUS` limbs. -/\n")
        out.append(f"def {prefix}Modulus : Limbs := {limbs}\n\n")
        out.append(f"/-- `-p^-1 mod 2^64` for the {field} base field, the crate's `INV`. -/\n")
        out.append(f"def {prefix}Inv : Nat := {inv}\n\n")
    n, skipped = 0, {}
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        op, key, *vals = parts
        prefix = FIELDS[key][0]
        fn = {"MUL": "mulMont", "SQR": "sqrMont", "FROM": "fromMont"}.get(op)
        if fn is None:
            raise ValueError(line)
        *operands, r = vals
        if not in_contract(op, key, [int(v, 16) for v in operands]):
            skipped[op] = skipped.get(op, 0) + 1
            continue
        vals = [f"(Limbs.ofNat 0x{v})" for v in operands]
        # One operand per line keeps every line within the repository's width.
        out.append(f"example :\n    {fn}\n")
        for v in vals:
            out.append(f"      {v}\n")
        out.append(f"      {prefix}Modulus {prefix}Inv =\n    (Limbs.ofNat 0x{r}) := by\n  decide +kernel\n\n")
        n += 1
    omitted = ", ".join(f"{k} {op}" for op, k in sorted(skipped.items())) or "none"
    out.append(f"\n-- {n} vectors; omitted as outside the proved contracts: {omitted}.\n\n"
               "end PastaAArch64Asm\n")
    return "".join(out)


# --- proof skeletons --------------------------------------------------------------------

# Bounds hypotheses the annotated spec theorems must provide, by argument name.
BOUND_HYPS = {"t": "ht", "modulus": "hm", "lhs": "hlhs", "rhs": "hrhs", "value": "hv", "acc": "hacc"}
INV_BOUND_HYP = "hinv_lt"
SKELETON_WIDTH = 100


def proj(arg, field):
    """The projection of the `Bounded` conjunction of argument `arg` that bounds `field`."""
    fields = ARG_FIELDS[arg]
    i = fields.index(field)
    return ".".join(["2"] * i + (["1"] if i < len(fields) - 1 else []))


def ssa_names(entries):
    """Unique names for the live bindings: the first binding of a register keeps its name,
    later ones get `_1`, `_2`, ... An argument bound under the argument's own name (the inline
    block's `inv`) is primed, so that extracting it does not shadow the theorem's variable."""
    counts, names = {}, []
    for e in entries:
        n = counts.get(e["name"], 0)
        counts[e["name"]] = n + 1
        base = e["name"] + "'" if e["expr"] == e["name"] else e["name"]
        names.append(base if n == 0 else f"{base}_{n}")
    return names


def wrap_tactic(head, words, tail, indent="  "):
    """`head w1 w2 ... tail`, broken over lines at SKELETON_WIDTH with a 4-space continuation."""
    lines, cur = [], indent + head
    for w in words:
        if len(cur) + 1 + len(w) > SKELETON_WIDTH:
            lines.append(cur)
            cur = indent + "    " + w
        else:
            cur += " " + w
    lines.append(cur + tail)
    return lines


def skeleton(routine):
    """The generated part of the correctness proof of `routine`: unfold the routine in `hr` and
    lift its lets to the top; then, instruction by instruction, extract that instruction's lets
    from `hr` under SSA names, record their defining equations (by `rfl`, in `%`/`/` form), make
    the locals opaque, and derive the linear facts from the equations, clearing the equations
    the later steps do not need. Each derived fact is an instance of one lemma
    (`Nat.mod_add_div`, `Nat.mod_lt`, `Nat.div_lt_of_lt_mul`, or a carry lemma from the spec
    file's preamble), so a step costs nothing wherever it sits and names the facts it rests on;
    `omega` is left to the hand-written annotations, which go after the facts of the group whose
    marker (`-- <register>: <instruction>`) names the register they need.

    Extracting one instruction at a time (`extract_lets +onlyGivenNames`) keeps the rest of the
    chain folded inside `hr`, so that `clear_value` has one hypothesis to revert and re-check.
    With every let extracted up front, each `clear_value` re-checks all the later locals and
    equations, which is quadratic in the chain's length and exhausted the heartbeat budget on
    the multiplication routine's 264 locals."""
    e = routine.emitter
    live = e.liveness(routine.result_names)
    entries = [en for en, keep in zip(e.entries, live) if keep]
    names = ssa_names(entries)
    ren = {}  # current SSA name of each register at each point: resolved while walking
    bnd = {}  # SSA name -> the fact bounding it below 2^64 (registers) or by 1 (carries)
    narrow = set()  # `lsr` results, whose bound is below 2^64 and needs weakening
    out = [f"  -- generated skeleton for `{routine.name}`: do not edit between the annotations",
           f"  unfold {routine.name} at hr", "  lift_lets at hr"]
    products, shifts = {}, {}
    eqs = []      # the current group's `have e_... := rfl` lines

    def r(op):  # operand as written in the entry, renamed to its SSA name at that point
        return ren.get(op, op)

    def lt64(op):  # a proof that the operand is below 2^64
        if re.fullmatch(r"[0-9]+", op):
            return "(by decide)"
        if op in narrow:
            return f"(lt_of_lt_of_le {bnd[op]} (by norm_num))"
        return bnd[op]

    def le1(op):  # a proof that the carry operand is at most 1
        return "(by decide)" if re.fullmatch(r"[0-9]+", op) else bnd[op]

    def eq(nm, rhs):
        eqs.append(f"  have e_{nm} : {nm} = {rhs} := rfl")

    i = 0
    while i < len(entries):
        en, nm = entries[i], names[i]
        kind, *ops = en["fact"]
        if kind not in ("call", "callout", "load", "param"):
            ops = [r(o) if isinstance(o, str) else o for o in ops]
        # The group's marker: the register it writes, then the instruction. Annotation blocks
        # are placed after the group they name. An `adds`/`subs` whose carry nothing reads has
        # no carry binding (the inline block's last shift), so its group is the pair and the
        # result, and its carry is a ghost, as for `adc`.
        dead_carry = (kind in ("adds", "subs")
                      and (i + 2 >= len(entries) or entries[i + 2]["fact"] != ("snd",)))
        label = names[i + 1] if kind in ("adds", "subs") else nm
        group = names[i:i + 2 if dead_carry else i + 3] if kind in ("adds", "subs") else [nm]
        eqs, lines = [], []
        # Every step records only facts `omega` handles cheaply later: linear equations, bounds,
        # and at most a disjunction. The `%`/`/` equations are derived by `rfl`, used to prove
        # those facts, and cleared.
        if kind == "load":
            arg, field = ops
            hyp = BOUND_HYPS[arg]
            eq(nm, f"{arg}.{field}")
            lines.append(f"  have b_{nm} : {nm} < 2^64 := by rw [e_{nm}]; exact {hyp}.{proj(arg, field)}")
            bnd[nm] = f"b_{nm}"
        elif kind == "inv":
            eq(nm, "inv")
            lines.append(f"  have b_{nm} : {nm} < 2^64 := by rw [e_{nm}]; exact {INV_BOUND_HYP}")
            bnd[nm] = f"b_{nm}"
        elif kind == "param":
            (p,) = ops
            eq(nm, p)
            lines.append(f"  have b_{nm} : {nm} < 2^64 := by rw [e_{nm}]; exact h{p}")
            bnd[nm] = f"b_{nm}"
        elif kind == "mov":
            (a,) = ops
            eq(nm, a)
            proof = "decide" if re.fullmatch(r"[0-9]+", a) else f"exact {lt64(a)}"
            lines.append(f"  have b_{nm} : {nm} < 2^64 := by rw [e_{nm}]; {proof}")
            bnd[nm] = f"b_{nm}"
        elif kind == "mul":
            a, b = ops
            eq(nm, f"{a} * {b} % 2^64")
            lines.append(f"  have b_{nm} : {nm} < 2^64 := by rw [e_{nm}]; exact Nat.mod_lt _ (Nat.two_pow_pos _)")
            bnd[nm] = f"b_{nm}"
            products[(a, b)] = nm  # its `%` equation is cleared at the matching `umulh`
        elif kind == "umulh":
            a, b = ops
            eq(nm, f"{a} * {b} / 2^64")
            lines.append(f"  have p_{nm} : {a} * {b} < 2^64 * 2^64 := Nat.mul_lt_mul'' {lt64(a)} {lt64(b)}")
            lines.append(f"  have b_{nm} : {nm} < 2^64 := by rw [e_{nm}]; exact Nat.div_lt_of_lt_mul p_{nm}")
            bnd[nm] = f"b_{nm}"
            if (a, b) in products:
                lo = products.pop((a, b))
                lines.append(f"  have d_{nm} : {lo} + 2^64 * {nm} = {a} * {b} := by")
                lines.append(f"    rw [e_{lo}, e_{nm}]; exact Nat.mod_add_div _ _")
                lines.append(f"  clear e_{lo} e_{nm}")
            else:
                # The low half is never computed (its cancellation is arranged by `subs`); name
                # it as a ghost so that later steps need no `%`.
                lines.append(f"  obtain ⟨lo_{nm}, b_lo_{nm}, d_{nm}⟩ :")
                lines.append(f"      ∃ lo, lo < 2^64 ∧ lo + 2^64 * {nm} = {a} * {b} :=")
                lines.append(f"    ⟨{a} * {b} % 2^64, Nat.mod_lt _ (Nat.two_pow_pos _),")
                lines.append(f"      by rw [e_{nm}]; exact Nat.mod_add_div _ _⟩")
                lines.append(f"  clear e_{nm}")
        elif kind == "lsl":
            a, k = ops
            eq(nm, f"{a} * 2^{k} % 2^64")
            lines.append(f"  have b_{nm} : {nm} < 2^64 := by rw [e_{nm}]; exact Nat.mod_lt _ (Nat.two_pow_pos _)")
            bnd[nm] = f"b_{nm}"
            shifts[(a, k)] = nm
        elif kind == "lsr":
            a, k = ops
            eq(nm, f"{a} / 2^{k}")
            lines.append(f"  have b_{nm} : {nm} < 2^{64 - k} := by")
            lines.append(f"    rw [e_{nm}]; exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq {lt64(a)} (by norm_num))")
            bnd[nm] = f"b_{nm}"
            narrow.add(nm)
            if (a, 64 - k) in shifts:
                lo = shifts.pop((a, 64 - k))
                if k != 2:
                    raise ValueError(f"lsl/lsr split by {64 - k}/{k}: add a lemma to the spec preamble")
                lines.append(f"  have sh_{nm} : {lo} + 2^64 * {nm} = {a} * 2^{64 - k} := by")
                lines.append(f"    rw [e_{lo}, e_{nm}]; exact lsl62_lsr2_split _")
                lines.append(f"  clear e_{lo} e_{nm}")
        elif kind in ("adds", "subs"):
            a, b, cin = ops
            xn = names[i + 1]
            cn = f"k_{xn}" if dead_carry else names[i + 2]
            if kind == "adds":
                val = f"({a} + {b} + {cin})"
                lin = f"{xn} + 2^64 * {cn} = {a} + {b} + {cin}"
                lin_proof = "Nat.mod_add_div _ _"
                carry_proof = f"addc_carry_le_one {a} {b} {cin} {lt64(a)} {lt64(b)} {le1(cin)}"
            else:
                val = f"({a} + 2^64 - {b} - (1 - {cin}))"
                lin = f"{xn} + 2^64 * {cn} + {b} + 1 = {a} + 2^64 + {cin}"
                lin_proof = f"subc_lin {a} {b} {cin} {lt64(b)} {le1(cin)}"
                carry_proof = f"subc_carry_le_one {a} {b} {cin} {lt64(a)}"
            eq(xn, f"{val} % 2^64")
            if dead_carry:
                lines.append(f"  have b_{xn} : {xn} < 2^64 := by rw [e_{xn}]; "
                             "exact Nat.mod_lt _ (Nat.two_pow_pos _)")
                lines.append(f"  obtain ⟨{cn}, b_{cn}, l_{xn}⟩ :")
                lines.append(f"      ∃ k, k ≤ 1 ∧ {lin.replace(cn, 'k')} :=")
                lines.append(f"    ⟨{val} / 2^64, {carry_proof},")
                lines.append(f"      by rw [e_{xn}]; exact {lin_proof}⟩")
                lines.append(f"  clear e_{xn}")
                ren[entries[i + 1]["name"]] = xn
                bnd[xn] = f"b_{xn}"
                i += 1
            else:
                eq(cn, f"{val} / 2^64")
                lines.append(f"  have l_{xn} : {lin} := by")
                lines.append(f"    rw [e_{xn}, e_{cn}]; exact {lin_proof}")
                lines.append(f"  have b_{xn} : {xn} < 2^64 := by rw [e_{xn}]; exact Nat.mod_lt _ (Nat.two_pow_pos _)")
                lines.append(f"  have b_{cn} : {cn} ≤ 1 := by")
                lines.append(f"    rw [e_{cn}]; exact {carry_proof}")
                lines.append(f"  clear e_{xn} e_{cn}")
                ren[entries[i + 1]["name"]] = xn
                ren[entries[i + 2]["name"]] = cn
                bnd[xn], bnd[cn] = f"b_{xn}", f"b_{cn}"
                i += 2
        elif kind == "adc":
            a, b, cin = ops
            eq(nm, f"({a} + {b} + {cin}) % 2^64")
            lines.append(f"  have b_{nm} : {nm} < 2^64 := by rw [e_{nm}]; exact Nat.mod_lt _ (Nat.two_pow_pos _)")
            lines.append(f"  obtain ⟨k_{nm}, b_k_{nm}, l_{nm}⟩ :")
            lines.append(f"      ∃ k, k ≤ 1 ∧ {nm} + 2^64 * k = {a} + {b} + {cin} :=")
            lines.append(f"    ⟨({a} + {b} + {cin}) / 2^64, addc_carry_le_one {a} {b} {cin} {lt64(a)} {lt64(b)} {le1(cin)},")
            lines.append(f"      by rw [e_{nm}]; exact Nat.mod_add_div _ _⟩")
            lines.append(f"  clear e_{nm}")
            bnd[nm] = f"b_{nm}"
        elif kind == "subs_carry":
            a, b, cin = ops
            eq(nm, f"({a} + 2^64 - {b} - (1 - {cin})) / 2^64")
            lines.append(f"  have b_{nm} : {nm} ≤ 1 := by rw [e_{nm}]; exact subc_carry_le_one {a} {b} {cin} {lt64(a)}")
            lines.append(f"  have l_{nm} : ({nm} = 1 ∧ {b} + 1 ≤ {a} + {cin}) ∨ ({nm} = 0 ∧ {a} + {cin} < {b} + 1) :=")
            lines.append(f"    subc_carry_cases {a} {b} {cin} _ e_{nm} {lt64(a)} {lt64(b)} {le1(cin)}")
            lines.append(f"  clear e_{nm}")
            bnd[nm] = f"b_{nm}"
        elif kind == "csel":
            c, a, b = ops
            eq(nm, f"(if {c} = 0 then {a} else {b})")
            lines.append(f"  have b_{nm} : {nm} < 2^64 := by")
            lines.append(f"    rw [e_{nm}]; split <;> first | exact {lt64(a)} | exact {lt64(b)}")
            bnd[nm] = f"b_{nm}"
        elif kind == "call":
            fmt, cargs = ops
            eq(nm, fmt.format(*[r(o) for o in cargs]))
        elif kind == "callout":
            callee, field = ops
            eq(nm, f"{r(callee)}.{field}")
            bnd[nm] = f"b_{nm}"  # supplied by the annotation that applies the callee's theorem
        else:
            raise ValueError(kind)
        ren[en["name"]] = nm
        out.append(f"  -- {label}: {en['comment']}")
        out += wrap_tactic("extract_lets +onlyGivenNames", group, " at hr")
        out += eqs
        out.append(f"  clear_value {' '.join(group)}")
        out += lines
        i += 1
    out.append("  subst hr")
    return out

def check_spec(path, routines):
    """Verify that each routine's skeleton appears verbatim and contiguously in `path` once its
    `-- BEGIN ... -- END` annotation blocks are removed and blank lines dropped. Text outside the
    skeletons (theorem statements, lemmas, the closing steps) is free; text between two
    skeleton lines must be inside an annotation block."""
    text = Path(path).read_text()
    stripped = re.sub(r"(?ms)^\s*-- BEGIN[^\n]*\n.*?^\s*-- END[^\n]*\n", "", text)
    remaining = [l for l in stripped.splitlines() if l.strip()]
    ok = True
    for rt in routines:
        sk = [l for l in skeleton(rt) if l.strip()]
        n = len(sk)
        if sk[1] not in remaining:  # `unfold <routine> at hr`: the theorem is not in this file
            continue
        for start in range(len(remaining) - n + 1):
            if remaining[start:start + n] == sk:
                del remaining[start:start + n]
                break
        else:
            i = remaining.index(sk[1])
            for j, l in enumerate(sk):
                if i + j >= len(remaining) or remaining[i + j] != l:
                    print(f"{path}: skeleton of {rt.name} diverges at skeleton line {j}:",
                          file=sys.stderr)
                    print(f"  expected: {l}", file=sys.stderr)
                    print(f"  found:    {remaining[i + j] if i + j < len(remaining) else '<eof>'}",
                          file=sys.stderr)
                    break
            ok = False
    return ok


def all_routines():
    routines = []
    for fn, name, doc, args in INLINE_ROUTINES:
        routines += emit_inline(fn, name, doc, args)
    return routines


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--skeleton":
        for rt in all_routines():
            if rt.name == sys.argv[2]:
                print("\n".join(skeleton(rt)))
                return 0
        print(f"no routine {sys.argv[2]}", file=sys.stderr)
        return 1
    if len(sys.argv) >= 3 and sys.argv[1] == "--check-spec":
        ok = check_spec(sys.argv[2], all_routines())
        print(f"{sys.argv[2]}: skeletons {'current' if ok else 'STALE'}")
        return 0 if ok else 1
    OUT_PROGRAM.write_text(gen_program())
    OUT_VECTORS.write_text(gen_vectors(VECTORS.read_text().splitlines()))
    print(f"wrote {OUT_PROGRAM} and {OUT_VECTORS}")


if __name__ == "__main__":
    sys.exit(main())
