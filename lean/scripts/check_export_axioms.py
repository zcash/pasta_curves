#!/usr/bin/env python3
"""Check the axiom census of a lean4export ndjson export against the nanoda config.

nanoda's strict mode (`unpermitted_axiom_hard_error: true`) rejects any axiom *declared*
outside the permitted list, but must permit axioms Lean core declares whether or not
anything uses them (`sorryAx` and the legacy compiler-trust axioms). Permitting a
declaration says nothing about use, so this script closes that gap from the export
itself:

  * the axioms declared in the export are exactly the config's `permitted_axioms`;
  * `sorryAx`, `Lean.ofReduceBool`, and `Lean.ofReduceNat` are cited by no expression
    at all — in particular, a `sorry` anywhere in the library fails here;
  * `Lean.trustCompiler` is cited only by `Lean.reduceBool` and `Lean.reduceNat` —
    the opaque evaluation functions of the deprecated compiler-trust route, distinct
    from their propositional bridge axioms `Lean.ofReduceBool`/`Lean.ofReduceNat`
    above — and nothing in this repository consumes them.

Failures come in two kinds, mirroring `check_native_optin.py`:

  * VIOLATION (exit 1) — an undesired outcome in structurally well-formed data: a
    declared axiom outside the permitted list, a stale permitted entry, or a citation
    of an axiom that must be unreferenced. Violations are collected, the full axiom
    census is printed regardless, and the process exits non-zero at the end.
  * ERROR (exit 2) — the export's structure is not as this script assumes: an
    unrecognized line shape or kind, a non-monotonic or undefined id, a cyclic name
    chain, or an unpinned format version. Conclusions drawn from such data would be
    unreliable, so the scan stops immediately without printing a census.

The scan does not silently rely on the export's structural conventions; it verifies
them:

  * the format version is pinned (bump deliberately after re-checking the assumptions);
  * every line's shape is from the closed known set, so a citation cannot hide inside
    an unhandled construct;
  * name, expression, and level ids are dense — each new id increments the previous
    by exactly 1 (names and levels from 1, id 0 being the reserved anonymous name and
    zero level; expressions from 0) — and every referenced sub-id is strictly smaller
    than the id being defined, so a referenced id is always already defined and the
    single-pass citation propagation over the expression DAG cannot miss a forward or
    dangling reference.

Usage: scripts/check_export_axioms.py [nanoda-config.json]
The export path is read from the config (single source of truth). Runs from the
repository root.
"""
import json
import sys
from pathlib import Path
from typing import NoReturn

FORMAT_VERSION = "3.1.0"
MAX_REPORTED_VIOLATIONS = 100

# Axioms that must be cited by no expression in the export.
UNREFERENCED = {"sorryAx", "Lean.ofReduceBool", "Lean.ofReduceNat"}
# Axioms that only the named declarations may cite.
RESTRICTED = {"Lean.trustCompiler": {"Lean.reduceBool", "Lean.reduceNat"}}
TARGETS = UNREFERENCED | set(RESTRICTED)
TARGET_COMPONENTS = {t.rsplit(".", 1)[-1] for t in TARGETS}

EXPR_KINDS = {"app", "bvar", "const", "forallE", "lam", "letE", "natVal", "proj",
              "sort", "strVal"}
LEVEL_KINDS = {"imax", "max", "param", "succ"}
DECL_KINDS = {"axiom", "def", "inductive", "opaque", "quot", "thm"}
# Expression sub-ids per expression kind (everything else in the payload is a name id,
# a level id, or plain data).
EXPR_SUBFIELDS = {"app": ("fn", "arg"), "forallE": ("type", "body"),
                  "lam": ("type", "body"), "letE": ("type", "value", "body"),
                  "proj": ("struct",)}
# Keys holding expression ids inside declaration payloads (at any nesting depth).
DECL_EXPR_KEYS = {"type", "value", "rhs"}

violations = []


def violation(msg):
    violations.append(msg)


def error(msg) -> NoReturn:
    print(f"ERROR: {msg} — the export's structure is not as this script assumes, so "
          f"its conclusions would be unreliable; re-verify the assumptions and update "
          f"this script", file=sys.stderr)
    sys.exit(2)


def main():
    config_path = Path(sys.argv[1] if len(sys.argv) > 1 else "scripts/nanoda-config.json")
    config = json.loads(config_path.read_text())
    permitted = set(config["permitted_axioms"])
    export_path = Path(config["export_file_path"])

    names = {}  # name id -> (prefix id, component)

    def resolve(i):
        parts, seen = [], set()
        while i != 0:
            if i in seen:
                error(f"name id {i} has a cyclic prefix chain")
            seen.add(i)
            entry = names.get(i)
            if entry is None:
                error(f"name id {i} is undefined")
            pre, comp = entry
            parts.append(comp)
            i = pre
        return ".".join(reversed(parts))

    def decl_expr_ids(payload):
        todo, out = [payload], []
        while todo:
            x = todo.pop()
            if isinstance(x, dict):
                for k, v in x.items():
                    if k in DECL_EXPR_KEYS and isinstance(v, int):
                        out.append(v)
                    else:
                        todo.append(v)
            elif isinstance(x, list):
                todo.extend(x)
        return out

    target_name_ids = {}  # name id -> target full name
    taint = {}  # expr id -> frozenset of target full names
    citers = {}  # target full name -> set of citing declaration names
    const_cited = {t: False for t in TARGETS}
    declared_axioms = set()
    max_in = max_il = 0  # id 0: the reserved anonymous name / zero level
    max_ie = -1
    meta_seen = False

    with open(export_path, "rb") as f:
        for raw in f:
            o = json.loads(raw)
            if "in" in o:
                i = o["in"]
                if i != max_in + 1:
                    error(f"name id {i} does not follow {max_in} densely")
                max_in = i
                if "str" in o:
                    pre, comp = o["str"]["pre"], o["str"]["str"]
                elif "num" in o:
                    pre, comp = o["num"]["pre"], str(o["num"]["i"])
                else:
                    error(f"unknown name entry shape: {sorted(o.keys())}")
                if len(o) != 2:
                    error(f"unknown name line shape: {sorted(o.keys())}")
                if not (pre == 0 or pre < i):
                    error(f"name id {i} references non-earlier prefix {pre}")
                names[i] = (pre, comp)
                if comp in TARGET_COMPONENTS:
                    full = resolve(i)
                    if full in TARGETS:
                        target_name_ids[i] = full
            elif "ie" in o:
                i = o["ie"]
                if i != max_ie + 1:
                    error(f"expression id {i} does not follow {max_ie} densely")
                max_ie = i
                rest = set(o.keys()) - {"ie"}
                if len(rest) != 1 or len(o) != 2:
                    error(f"unknown expression line shape: {sorted(o.keys())}")
                (kind,) = rest
                if kind not in EXPR_KINDS:
                    error(f"unknown expression kind '{kind}' at expression id {i}")
                t = set()
                if kind == "const":
                    nid = o["const"]["name"]
                    if nid in target_name_ids:
                        full = target_name_ids[nid]
                        t.add(full)
                        const_cited[full] = True
                for f2 in EXPR_SUBFIELDS.get(kind, ()):
                    v = o[kind][f2]
                    if not 0 <= v < i:
                        error(f"expression id {i} references non-earlier sub-id {v}")
                    if v in taint:
                        t |= taint[v]
                if t:
                    taint[i] = frozenset(t)
            elif "il" in o:
                i = o["il"]
                if i != max_il + 1:
                    error(f"level id {i} does not follow {max_il} densely")
                max_il = i
                rest = set(o.keys()) - {"il"}
                if len(rest) != 1 or len(o) != 2:
                    error(f"unknown level line shape: {sorted(o.keys())}")
                (kind,) = rest
                if kind not in LEVEL_KINDS:
                    error(f"unknown level kind '{kind}' at level id {i}")
                if kind == "succ":
                    subs = [o[kind]]
                elif kind in ("max", "imax"):
                    subs = o[kind]
                else:  # param references a name id, not a level id
                    subs = []
                for v in subs:
                    if v >= i:
                        error(f"level id {i} references non-earlier sub-id {v}")
            elif "meta" in o:
                meta_seen = True
                version = o["meta"]["format"]["version"]
                if version != FORMAT_VERSION:
                    error(f"export format version {version} != pinned {FORMAT_VERSION}")
            else:
                kinds = set(o.keys()) & DECL_KINDS
                if len(kinds) != 1 or len(o) != 1:
                    error(f"unknown line shape: {sorted(o.keys())}")
                (kind,) = kinds
                d = o[kind]
                if kind == "axiom":
                    declared_axioms.add(resolve(d["name"]))
                hit = set()
                for v in decl_expr_ids(d):
                    if not 0 <= v <= max_ie:
                        error(f"declaration references undefined expression id {v}")
                    if v in taint:
                        hit |= taint[v]
                if hit:
                    if kind == "inductive":
                        nm = ", ".join(resolve(ty["name"]) for ty in d["types"])
                    else:
                        nm = resolve(d["name"])
                    for t2 in hit:
                        citers.setdefault(t2, set()).add(nm)

    if not meta_seen:
        error("export has no meta line; format version unverified")

    if declared_axioms != permitted:
        unpermitted = sorted(declared_axioms - permitted)
        undeclared = sorted(permitted - declared_axioms)
        if unpermitted:
            violation(f"axiom(s) declared but not permitted: {unpermitted}")
        if undeclared:
            violation(f"permitted axiom(s) not declared in the export (stale census "
                      f"entry): {undeclared}")

    for t in sorted(UNREFERENCED):
        if const_cited[t] or citers.get(t):
            violation(f"'{t}' is cited by: {sorted(citers.get(t, {'<expression>'}))}")
    for t, allowed in RESTRICTED.items():
        extra = citers.get(t, set()) - allowed
        if extra:
            violation(f"'{t}' is cited outside its allowance {sorted(allowed)}: "
                      f"{sorted(extra)}")

    # Print the full census whether or not anything was flagged: this is the actionable
    # state when a check above has flagged a stale or widened axiom list.
    print(f"export axiom census: {len(declared_axioms)} axiom(s) declared:")
    for a in sorted(declared_axioms):
        cited = ([] if a not in TARGETS else
                 sorted(citers.get(a, set())) or ["nothing"])
        note = f"  (cited by: {', '.join(cited)})" if cited else ""
        print(f"  {a}{note}")

    if violations:
        for msg in violations[:MAX_REPORTED_VIOLATIONS]:
            print(f"VIOLATION: {msg}", file=sys.stderr)
        if len(violations) > MAX_REPORTED_VIOLATIONS:
            print(f"... and {len(violations) - MAX_REPORTED_VIOLATIONS} further "
                  f"violation(s)", file=sys.stderr)
        sys.exit(1)
    print("export axiom census: all checks passed")


if __name__ == "__main__":
    main()
