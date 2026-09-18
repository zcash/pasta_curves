#!/usr/bin/env python3
"""Generate the Lean transcription of the crate's inline Pasta Montgomery blocks.

Reads the inline `asm!` blocks in `src/asm/aarch64.rs` and `src/asm/x86_64.rs`, and writes

- `lean/PastaAsm/<Architecture>/Transcription.lean`: each block as a Lean definition over
  its instruction semantics, one `let` per instruction result, in the block's order,
  with the instruction as a trailing comment;
- `lean/PastaAsm/<Architecture>/Vectors.lean`: one kernel-checked example per line of
  `test-vectors/pasta_mul-armv8-vectors.txt` whose operands are inside the backend's
  contracts, the outputs of the real routines on an Apple M-series machine.

The transcription is deliberately mechanical. A block is read from its template lines, with
the operand placeholders as register names, rebound by each instruction that writes them:
the `in` and `inout` operands bind argument limbs and `inv`, the named `out` and the `inout`
operands are the result limbs, and the block ends as a routine does. The compiler's
allocation of registers to the operands is not modelled; the script checks that every
register the block reads was written by the block or bound by an operand.

AArch64 omits bindings that nothing later reads: unused operands are left as comments,
unused carry writes are dropped, and unused computed registers are reported as errors.
x86-64 retains architectural results, including dead flag writes.

For both architectures, the proof skeletons lift and extract the `let`s with merging off, so
every binding is its own `let`, equal values or not.

Run from the repository root:

    python3 lean/scripts/gen.py

`lean/scripts/check.sh` fails if the generated output differs from the committed files;
`--check` compares all outputs without rewriting them.

The script also generates the mechanical part of each block's correctness proof:
`--skeleton ARCH:NAME` prints it (see `skeleton`), and `--check-spec FILE` checks that
FILE contains its registered skeletons verbatim once its `-- BEGIN ... -- END` annotation
blocks are removed; the check script runs that too. Bare skeleton names select AArch64
for compatibility. Python 3.9+; stdlib only.
"""

import argparse
import re
import sys
import textwrap
from pathlib import Path

import asm_source

# Backends import this module by name; keep a single shared state when this file runs as a script.
sys.modules.setdefault("gen", sys.modules[__name__])

ROOT = Path(__file__).resolve().parents[2]
VECTORS = ROOT / "test-vectors/pasta_mul-armv8-vectors.txt"

# Fields of the argument structures, by argument name, for the skeleton's bound hypotheses.
LIMB_FIELDS = ["l0", "l1", "l2", "l3"]
ARG_FIELDS = {arg: LIMB_FIELDS for arg in ("t", "lhs", "rhs", "value", "modulus")}
ARG_FIELDS["product"] = [f"l{i}" for i in range(8)]

# Bounds hypotheses the annotated spec theorems must provide, by argument name.
BOUND_HYPS = {
    "t": "ht", "modulus": "hm", "lhs": "hlhs", "rhs": "hrhs",
    "value": "hv", "product": "hproduct", "acc": "hacc",
}
INV_BOUND_HYP = "hinv_lt"
SKELETON_WIDTH = 100


class Emitter:
    """Records one block's Lean `let` bindings and their proof facts."""

    def __init__(self):
        self.entries = []   # dicts: name, expr, comment, reads, load (bool)
        self.known = set()  # names holding a value the program may read
        self.cur_reads = set()
        self.pc = None      # index of the instruction being transcribed (None: an argument)

    def bind(self, name, expr, comment=None, reads=None, load=False, fact=None, note=None):
        """Record a binding. `fact` is the skeleton's description of it: a tuple whose head
        names the kind of instruction and whose remaining items are the operand names. `note`,
        when given, replaces the instruction as the binding's trailing comment, for a binding
        that continues the instruction of the line above it."""
        self.known.add(name)
        self.entries.append(dict(name=name, expr=expr, comment=comment, note=note,
                                 reads=set(self.cur_reads) if reads is None else set(reads),
                                 load=load, fact=fact, pc=self.pc))

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
        """The `let` lines, with bindings that nothing reads dropped."""
        live = self.liveness(result_names)
        lines = []  # (code, comment): a `let` with its instruction, or (None, whole-line comment)
        for e, keep in zip(self.entries, live):
            if keep:
                lines.append((f"  let {e['name']} := {e['expr']}", e.get("note") or e["comment"]))
            elif e["load"]:
                lines.append((None, f"  -- {e['comment']}: {e['name']} = {e['expr']} is never read"))
            elif e["name"] != "c":
                raise ValueError(f"dead computation: {e['name']} := {e['expr']} ({e['comment']})")
        return lines


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


# --- reference vectors ------------------------------------------------------------------

HEADER_VECTORS = """/-
Copyright (c) 2026 the pasta-asm contributors.
Released under the Apache License, Version 2.0, as described in the file LICENSE.
-/
"""

# The crate's two fields, by the vectors file's key, as `Fields.lean` names them.
FIELDS = {"Fp": "pallasBase", "Fq": "vestaBase"}


# The two moduli as integers, for classifying the vectors' operands.
MODULUS_INT = {
    "Fp": 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001,
    "Fq": 0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001,
}

VECTOR_FUNCTIONS = {"MUL": "mulMont", "SQR": "sqrMont", "FROM": "fromMont"}
VECTOR_OPERAND_COUNTS = {"MUL": 2, "SQR": 1, "FROM": 1}
HEX_VALUE = re.compile(r"[0-9a-fA-F]{64}")


def parse_vectors(lines):
    """Parse and strictly validate nonempty lines from the shared hardware corpus."""
    vectors = []
    for line_number, line in enumerate(lines, 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 2:
            raise ValueError(f"vector line {line_number}: malformed row")
        op, key, *vals = parts
        if op not in VECTOR_OPERAND_COUNTS:
            raise ValueError(f"vector line {line_number}: unknown operation {op}")
        if key not in FIELDS:
            raise ValueError(f"vector line {line_number}: unknown field {key}")
        expected = VECTOR_OPERAND_COUNTS[op] + 1
        if len(vals) != expected:
            raise ValueError(
                f"vector line {line_number}: {op} expects {expected} values, got {len(vals)}"
            )
        if any(HEX_VALUE.fullmatch(v) is None for v in vals):
            raise ValueError(f"vector line {line_number}: values must be 64 hexadecimal digits")
        vectors.append((op, key, vals))
    return vectors


def is_canonical(key, value):
    """Whether a 256-bit input is below the selected Pasta modulus."""
    return value < MODULUS_INT[key]


def in_public_contract(op, key, operands):
    """Whether the vector's operands are inside the public contract: for the multiplication,
    a canonical left operand, or a canonical right operand whose limbs 1 to 3 are at most
    `2^64 - 3`; for the squaring, a canonical input; for the conversion, any."""
    p = MODULUS_INT[key]
    if op == "MUL":
        lhs, rhs = operands
        limbs = [(rhs >> (64 * i)) % 2**64 for i in range(1, 4)]
        return lhs < p or (rhs < p and all(l <= 2**64 - 3 for l in limbs))
    if op == "SQR":
        return operands[0] < p
    if op == "FROM":
        return True
    raise ValueError(f"unknown operation {op}")


def render_vectors(lines, *, imports, introduction, namespace, in_contract, omission_scope):
    """Render one backend's accepted corpus rows as kernel-checked Lean examples."""
    out = [HEADER_VECTORS, imports, introduction]
    n, skipped = 0, {}
    for op, key, vals in parse_vectors(lines):
        prefix = FIELDS[key]
        fn = VECTOR_FUNCTIONS[op]
        *operands, r = vals
        if not in_contract(op, key, [int(v, 16) for v in operands]):
            skipped[op] = skipped.get(op, 0) + 1
            continue
        vals = [f"(Limbs.ofNat 0x{v})" for v in operands]
        # One operand per line keeps every line within the repository's width.
        out.append(f"example :\n    {fn}\n")
        for v in vals:
            out.append(f"      {v}\n")
        out.append(f"      {prefix}.modulus {prefix}.inv =\n    (Limbs.ofNat 0x{r}) := by\n  decide +kernel\n\n")
        n += 1
    omitted = ", ".join(f"{k} {op}" for op, k in sorted(skipped.items())) or "none"
    out.append(f"\n-- {n} vectors; omitted as outside {omission_scope}: {omitted}.\n\n"
               f"end {namespace}\n")
    return "".join(out)


# --- proof skeletons --------------------------------------------------------------------


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


class SkeletonBackend:
    """Hooks for ISA-specific proof grouping and facts in the shared skeleton traversal."""

    def prepare(self, emitter, entries):
        return SkeletonPreparation(entries, ssa_names(entries))

    def fact(self, kind, ops, context):
        return False


class SkeletonPreparation:
    """The entries and backend-selected grouping metadata consumed by the shared traversal."""

    def __init__(self, entries, names):
        self.entries = entries
        self.names = names


class SkeletonFactContext:
    """Shared skeleton state exposed narrowly to an ISA backend's fact hook."""

    def __init__(self, entries, names, index, entry, name, group_entries, group_names,
                 lines, eq, lt64, le1, ren, bnd, unit_bound, consumed):
        self.entries, self.names, self.index = entries, names, index
        self.entry, self.name = entry, name
        self.group_entries = group_entries
        self.group_names = group_names
        self.lines, self.eq = lines, eq
        self.lt64, self.le1 = lt64, le1
        self.ren, self.bnd, self.unit_bound = ren, bnd, unit_bound
        self.consumed = consumed


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


def proj(arg, field):
    """The projection of the `Bounded` conjunction of argument `arg` that bounds `field`."""
    fields = ARG_FIELDS[arg]
    i = fields.index(field)
    return ".".join(["2"] * i + (["1"] if i < len(fields) - 1 else []))


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
    live_entries = [en for en, keep in zip(e.entries, live) if keep]
    backend = routine.skeleton_backend
    prepared = backend.prepare(e, live_entries)
    entries, names = prepared.entries, prepared.names
    ren = {}  # current SSA name of each register at each point: resolved while walking
    bnd = {}  # SSA name -> the fact bounding it below 2^64 (registers) or by 1 (carries)
    narrow = set()  # `lsr` results, whose bound is below 2^64 and needs weakening
    unit_bound = set()
    out = [f"  -- generated skeleton for `{routine.name}`: do not edit between the annotations",
           f"  unfold {routine.name} at hr", "  lift_lets -merge at hr"]
    products = {}
    eqs = []      # the current group's `have e_... := rfl` lines

    def r(op):  # operand as written in the entry, renamed to its SSA name at that point
        return ren.get(op, op)

    def expression_bound(op):
        field = re.fullmatch(r"(\w+)\.(l[0-7])", op)
        if field and field.group(1) in BOUND_HYPS:
            arg, limb = field.groups()
            return f"{BOUND_HYPS[arg]}.{proj(arg, limb)}"
        return None

    def lt64(op):  # a proof that the operand is below 2^64
        if re.fullmatch(r"[0-9]+", op):
            return "(by decide)"
        direct = expression_bound(op)
        if direct:
            return direct
        if op in narrow or op in unit_bound:
            return f"(lt_of_lt_of_le {bnd[op]} (by norm_num))"
        return bnd[op]

    def le1(op):  # a proof that the carry operand is at most 1
        if re.fullmatch(r"[0-9]+", op):
            return "(by decide)"
        return bnd[op]

    def eq(nm, rhs):
        eqs.append(f"  have e_{nm} : {nm} = {rhs} := rfl")

    i = 0
    while i < len(entries):
        en, nm = entries[i], names[i]
        kind, *ops = en.get("group_fact", en["fact"])
        if kind not in ("call", "callout", "load", "param"):
            ops = [r(o) if isinstance(o, str) else o for o in ops]
        # The group's marker: the register it writes, then the instruction. Annotation blocks
        # are placed after the group they name. Backend metadata groups ISA instructions that
        # produce several bindings; a member not marked as kept (a proof-only wrapper) has a
        # name and facts but no `let` to extract.
        group_entries = en.get("group")
        if group_entries:
            group = [group_name for _, group_name, keep in group_entries if keep]
            group_names = [group_name for _, group_name, _ in group_entries]
            nm = group_names[0]
        else:
            group = [nm]
            group_names = group
        label = en.get("group_label", nm)
        eqs, lines = [], []
        # Every step records only facts `omega` handles cheaply later: linear equations, bounds,
        # and at most a disjunction. The `%`/`/` equations are derived by `rfl`, used to prove
        # those facts, and cleared.
        fact_context = SkeletonFactContext(
            entries, names, i, en, nm,
            group_entries or [(en, nm, True)], group_names,
            lines, eq, lt64, le1, ren, bnd, unit_bound, len(group),
        )

        if backend.fact(kind, ops, fact_context):
            pass
        elif kind == "load":
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
            direct = expression_bound(a)
            proof = "decide" if re.fullmatch(r"[0-9]+", a) else f"exact {direct or lt64(a)}"
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
            # No pairing with the matching `lsr`: the split fact is stated here, with the
            # high part as `a / 2^(64 - k)`, and the `lsr`'s equation is kept, so that `omega`
            # connects the two, through a `mov` copy's equation if there is one.
            a, k = ops
            if k != 62:
                raise ValueError(f"lsl by {k}: add a lemma to the spec preamble")
            eq(nm, f"{a} * 2^{k} % 2^64")
            lines.append(f"  have b_{nm} : {nm} < 2^64 := by rw [e_{nm}]; exact Nat.mod_lt _ (Nat.two_pow_pos _)")
            lines.append(f"  have sh_{nm} : {nm} + 2^64 * ({a} / 2^2) = {a} * 2^62 := by")
            lines.append(f"    rw [e_{nm}]; exact lsl62_lsr2_split _")
            bnd[nm] = f"b_{nm}"
        elif kind == "lsr":
            a, k = ops
            eq(nm, f"{a} / 2^{k}")
            lines.append(f"  have b_{nm} : {nm} < 2^{64 - k} := by")
            lines.append(f"    rw [e_{nm}]; exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq {lt64(a)} (by norm_num))")
            bnd[nm] = f"b_{nm}"
            narrow.add(nm)
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
        elif kind == "select":
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
            raise ValueError(f"unsupported skeleton fact {kind}")

        if not group_entries:
            ren[en["name"]] = nm
        out.append(f"  -- {label}: {group_entries[0][0]['comment'] if group_entries else en['comment']}")
        out += wrap_tactic("extract_lets -merge +onlyGivenNames", group, " at hr")
        out += eqs
        out.append(f"  clear_value {' '.join(group)}")
        out += lines
        i += fact_context.consumed
    out.append("  subst hr")
    return out


def _strip_annotations(path, text):
    """Remove checked annotation blocks and reject malformed marker structure."""
    remaining = []
    active = None
    ok = True
    for line_number, line in enumerate(text.splitlines(), 1):
        marker = re.match(r"^\s*-- (BEGIN|END)(?:\s+(.*?))?\s*$", line)
        if not marker:
            if active is None:
                remaining.append(line)
            continue
        kind, label = marker.group(1), marker.group(2) or ""
        if kind == "BEGIN":
            if active is not None:
                print(f"{path}:{line_number}: nested BEGIN inside `{active[0]}`", file=sys.stderr)
                ok = False
            else:
                active = (label, line_number)
        elif active is None:
            print(f"{path}:{line_number}: END without BEGIN", file=sys.stderr)
            ok = False
        else:
            if label != active[0]:
                print(
                    f"{path}:{line_number}: END `{label}` does not match BEGIN `{active[0]}` "
                    f"at line {active[1]}", file=sys.stderr,
                )
                ok = False
            active = None
    if active is not None:
        print(f"{path}:{active[1]}: unterminated BEGIN `{active[0]}`", file=sys.stderr)
        ok = False
    return ok, [line for line in remaining if line.strip()]


def check_spec(path, routines):
    """Require exactly one current skeleton for every routine in this file's manifest."""
    path = Path(path)
    try:
        source = path.read_text()
    except OSError as error:
        print(f"{path}: cannot read proof file: {error.strerror}", file=sys.stderr)
        return False
    ok, remaining = _strip_annotations(path, source)
    expected_names = {routine.name for routine in routines}
    markers = [
        (index, match.group(1))
        for index, line in enumerate(remaining)
        if (match := re.match(r"^\s*-- generated skeleton for `([^`]+)`:", line))
    ]
    unexpected = sorted({name for _, name in markers if name not in expected_names})
    for name in unexpected:
        print(f"{path}: unexpected generated skeleton `{name}`", file=sys.stderr)
        ok = False
    for routine in routines:
        generated = [line for line in skeleton(routine) if line.strip()]
        starts = [index for index, name in markers if name == routine.name]
        if len(starts) != 1:
            print(
                f"{path}: expected one skeleton of {routine.name}, found {len(starts)}",
                file=sys.stderr,
            )
            ok = False
            continue
        start = starts[0]
        found = remaining[start:start + len(generated)]
        if found != generated:
            for index, expected in enumerate(generated):
                actual = found[index] if index < len(found) else "<eof>"
                if actual != expected:
                    print(
                        f"{path}: skeleton of {routine.name} diverges at skeleton line {index}:",
                        file=sys.stderr,
                    )
                    print(f"  expected: {expected}", file=sys.stderr)
                    print(f"  found:    {actual}", file=sys.stderr)
                    break
            ok = False
    return ok


# Import backends lazily: they import the shared vector API above to configure it,
# while this common CLI imports them only when orchestration needs them.
def _backends():
    import gen_aarch64
    import gen_x86_64

    return gen_aarch64, gen_x86_64


def generated_outputs():
    """Return every generated path and its expected contents without writing files."""
    gen_aarch64, gen_x86_64 = _backends()
    return gen_aarch64.generated_outputs() + gen_x86_64.generated_outputs()


# Existing proof files only. None selects every generated routine of that architecture.
SPEC_MANIFEST = {
    "lean/PastaAsm/AArch64/Spec/Add.lean": ("AArch64", ("addMod",)),
    "lean/PastaAsm/AArch64/Spec/Sub.lean": ("AArch64", ("subMod",)),
    "lean/PastaAsm/AArch64/Spec/Mul.lean": ("AArch64", ("mulMont", "mulMontRound")),
    "lean/PastaAsm/AArch64/Spec/Square.lean": ("AArch64", ("sqrMont",)),
    "lean/PastaAsm/X86_64/Spec/Add.lean": ("X86_64", ("addMod",)),
    "lean/PastaAsm/X86_64/Spec/Sub.lean": ("X86_64", ("subMod",)),
    "lean/PastaAsm/X86_64/Spec/Square.lean": ("X86_64", ("squareLo",)),
}

# Missing proofs are tracked by routine, not by hypothetical files.
UNPROVED_ROUTINES = {
    "X86_64": ("mulMont", "squareHi", "fromMont"),
}


def architecture_routines():
    """Return the proof-capable routine manifest for each assembly architecture."""
    gen_aarch64, gen_x86_64 = _backends()
    return {"AArch64": gen_aarch64.all_routines(), "X86_64": gen_x86_64.all_routines()}


def find_routine(specification):
    """Resolve ``ARCH:name``; legacy bare routine names continue to mean AArch64."""
    if ":" in specification:
        architecture, name = specification.split(":", 1)
    else:
        architecture, name = "AArch64", specification
    manifests = architecture_routines()
    if architecture not in manifests:
        raise ValueError(f"unknown architecture {architecture}")
    matches = [routine for routine in manifests[architecture] if routine.name == name]
    if len(matches) != 1:
        raise ValueError(f"no routine {architecture}:{name}")
    return matches[0]


def parse_spec_manifest(specification):
    """Resolve one explicitly registered proof file to its required routine skeletons."""
    requested_architecture = None
    path_text = specification
    if ":" in specification:
        requested_architecture, path_text = specification.split(":", 1)
    path = Path(path_text)
    try:
        key = path.resolve().relative_to(ROOT).as_posix()
    except ValueError as error:
        raise ValueError(f"spec path is outside the repository: {path}") from error
    if key not in SPEC_MANIFEST:
        raise ValueError(f"unregistered proof skeleton manifest: {key}")
    architecture, routine_names = SPEC_MANIFEST[key]
    if requested_architecture is not None and requested_architecture != architecture:
        raise ValueError(
            f"manifest {key} belongs to {architecture}, not {requested_architecture}"
        )
    available = architecture_routines()
    if architecture not in available:
        raise ValueError(f"manifest {key} names unknown architecture {architecture}")
    routines = available[architecture]
    if routine_names is None:
        selected = routines
    else:
        by_name = {routine.name: routine for routine in routines}
        missing = [name for name in routine_names if name not in by_name]
        if missing:
            raise ValueError(f"manifest {key} names missing routines: {', '.join(missing)}")
        selected = [by_name[name] for name in routine_names]
    return path, selected


def check_specs(strict=False):
    """Check every registered file and account for every generated routine exactly once.

    Explicitly unproved routines are reported, never counted as checked skeletons. Strict
    mode rejects them as well. Kernel checking remains the responsibility of the Lean build.
    """
    available = architecture_routines()
    inventory = {(arch, routine.name) for arch, routines in available.items()
                 for routine in routines}
    covered, unproved = {}, set()
    ok = True
    for filename, (arch, names) in SPEC_MANIFEST.items():
        if arch not in available:
            print(f"{filename}: unknown architecture {arch}", file=sys.stderr)
            ok = False
            continue
        by_name = {routine.name: routine for routine in available[arch]}
        selected = tuple(by_name) if names is None else names
        routines = []
        for name in selected:
            key = (arch, name)
            if key not in inventory:
                print(f"{filename}: unknown routine {arch}:{name}", file=sys.stderr)
                ok = False
                continue
            if key in covered:
                print(f"{arch}:{name}: duplicate coverage in {covered[key]} and {filename}",
                      file=sys.stderr)
                ok = False
            covered[key] = filename
            routines.append(by_name[name])
        if not check_spec(ROOT / filename, routines):
            ok = False
    for arch, names in UNPROVED_ROUTINES.items():
        if arch not in available:
            print(f"unproved list: unknown architecture {arch}", file=sys.stderr)
            ok = False
        for name in names:
            key = (arch, name)
            if key not in inventory:
                print(f"unproved list: unknown routine {arch}:{name}", file=sys.stderr)
                ok = False
            if key in unproved:
                print(f"unproved list: duplicate routine {arch}:{name}", file=sys.stderr)
                ok = False
            unproved.add(key)
    for arch, name in sorted(set(covered) & unproved):
        print(f"{arch}:{name}: both covered and explicitly unproved", file=sys.stderr)
        ok = False
    for arch, name in sorted(inventory - set(covered) - unproved):
        print(f"{arch}:{name}: no Spec coverage or explicit unproved entry", file=sys.stderr)
        ok = False
    for arch, name in sorted(unproved & inventory):
        print(f"unproved: {arch}:{name}")
    if strict and unproved:
        print("strict Spec coverage requires every routine to have a checked skeleton",
              file=sys.stderr)
        ok = False
    print(f"Spec coverage: {len(covered)} registered, {len(unproved & inventory)} unproved; "
          f"{'checks passed' if ok else 'FAILED'}")
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="compare every output without writing")
    action.add_argument(
        "--check-spec", metavar="[ARCH:]FILE",
        help="check the architecture's registered proof skeletons",
    )
    action.add_argument(
        "--skeleton", metavar="[ARCH:]NAME",
        help="print one proof skeleton (bare names retain legacy AArch64 meaning)",
    )
    action.add_argument("--check-specs", action="store_true",
                        help="check all Spec files and account for every generated routine")
    parser.add_argument("--strict", action="store_true",
                        help="with --check-specs, reject explicitly unproved routines")
    args = parser.parse_args(argv)
    if args.strict and not args.check_specs:
        parser.error("--strict requires --check-specs")
    if args.check_specs:
        return 0 if check_specs(strict=args.strict) else 1

    if args.skeleton:
        try:
            routine = find_routine(args.skeleton)
        except ValueError as error:
            print(error, file=sys.stderr)
            return 1
        print("\n".join(skeleton(routine)))
        return 0
    if args.check_spec:
        try:
            path, routines = parse_spec_manifest(args.check_spec)
        except ValueError as error:
            print(error, file=sys.stderr)
            return 1
        ok = check_spec(path, routines)
        print(f"{path}: skeletons {'current' if ok else 'STALE'}")
        return 0 if ok else 1

    outputs = generated_outputs()
    if args.check:
        checks = [asm_source.check_output(path, expected) for path, expected in outputs]
        ok = all(checks)
        print(f"generated transcriptions: {'current' if ok else 'STALE'}")
        return 0 if ok else 1

    for path, expected in outputs:
        path.write_text(expected)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
