#!/usr/bin/env python3
"""Check the axiom census of a lean4export ndjson export against the nanoda config.

The export format is lean4export's ndjson format, described in its `format_ndjson.md`;
the link is to the tag that CI builds lean4export from, matching `lean-toolchain`:
https://github.com/leanprover/lean4export/blob/v4.31.0/format_ndjson.md

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

Aeneas' Lean library, which the translation of the portable inversion blocks imports,
declares axioms of its own for opaque Rust items, and two of its tests leave `sorryAx` citations
behind. A violation is accepted only when:

  * its axiom is `sorryAx` and every declaration that depends on it is one of Aeneas' tests (an
    `Aeneas.` name with a `Test` component); or
  * its axiom is Aeneas' and every declaration that depends on it is in Aeneas.

A declaration depends on an axiom when it cites the axiom, or cites a declaration that depends on
it. Checking only the citations would let a declaration of the package rely on a `sorry` or on
one of Aeneas' axioms through one of Aeneas' declarations, which the exceptions accept.

The scan reads the export once and records which expressions and declarations refer to which, as
a graph. A declaration depends on an axiom exactly when the axiom is reachable from it, so a
depth-first search over the reversed edges, starting at the axiom, finds its dependents; one
that stops at the first declarations finds its citers. Each search costs time linear in what it
visits, and the graph is held in dense arrays, in memory linear in the export.

The export does not record the module of a declaration, so a check of the package's own
sources makes sure none declares into the `Aeneas` namespace. With `--nanoda-config OUT`, a
clean census writes nanoda's config to OUT, permitting Aeneas' axioms as well, since
nanoda's strict mode rejects any declared axiom it is not told of.

Failures come in two kinds, mirroring CompElliptic's `check_native_optin.py`:

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
    than the id being defined, so a referenced expression is always already defined;
    and every name that an expression cites or a declaration declares is defined, so the
    graph has no dangling reference.

Usage: scripts/check_export_axioms.py [--nanoda-config OUT] [nanoda-config.json]
The export path is read from the config (single source of truth). Runs from `lean/`.
"""

import json
import re
import sys
from array import array
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

EXPR_KINDS = {
    "app",
    "bvar",
    "const",
    "forallE",
    "lam",
    "letE",
    "mdata",
    "natVal",
    "proj",
    "sort",
    "strVal",
}
LEVEL_KINDS = {"imax", "max", "param", "succ"}
DECL_KINDS = {"axiom", "def", "inductive", "opaque", "quot", "thm"}
# Expression sub-ids per expression kind (everything else in the payload is a name id,
# a level id, or plain data). A kind added to EXPR_KINDS MUST be given its expression-valued
# fields here in the SAME change: the citation propagation descends only through these, so a
# kind accepted with no entry would let a citation hide in an unvisited subterm. `mdata`'s
# `data` holds `Lean.DataValue`s (string, bool, name, nat, int, syntax), none of which is an
# expression, so `expr` is its only expression-valued field.
EXPR_SUBFIELDS = {
    "app": ("fn", "arg"),
    "forallE": ("type", "body"),
    "lam": ("type", "body"),
    "letE": ("type", "value", "body"),
    "mdata": ("expr",),
    "proj": ("struct",),
}
# Keys holding expression ids inside declaration payloads (at any nesting depth).
DECL_EXPR_KEYS = {"type", "value", "rhs"}

violations = []


def violation(msg):
    violations.append(msg)


def error(msg) -> NoReturn:
    print(
        f"ERROR: {msg} — the export's structure is not as this script assumes, so "
        f"its conclusions would be unreliable; re-verify the assumptions and update "
        f"this script",
        file=sys.stderr,
    )
    sys.exit(2)


def grouped(n, keys, values):
    """`values` grouped by `keys`, which are below `n`, in compressed sparse row form: the values
    for the key `k` are `out[start[k]:start[k + 1]]`. A counting sort, in two linear passes."""
    start = array("I", [0]) * (n + 1)
    for k in keys:
        start[k + 1] += 1
    for k in range(n):
        start[k + 1] += start[k]
    out = array("I", [0]) * len(keys)
    fill = array("I", start)
    for k, v in zip(keys, values):
        out[fill[k]] = v
        fill[k] += 1
    return start, out


class Export:
    """What the census needs from an export: its declared axioms, and which expressions and
    declarations refer to which, as a graph stored in dense arrays.

    The graph's edges run from a declaration to its expressions, from an expression to its
    subexpressions, and from a `const` expression to the name that it cites; a declaration
    depends on an axiom exactly when the axiom's name is reachable from it. The arrays hold the
    edges reversed, for searches that start at an axiom: each expression's parents, the `const`
    expressions that cite each name, and the declarations that hold each expression at the top
    level."""

    def __init__(self, axioms, parents, citing, holders, owns, labels):
        self.axioms = axioms  # axiom name -> name id
        self.parents = parents  # expression id -> the expressions it is a subexpression of
        self.citing = citing  # name id -> the `const` expressions citing it
        self.holders = holders  # expression id -> the declarations holding it at the top level
        self.owns = owns  # declaration index -> the name ids that it declares
        self.labels = labels  # declaration index -> its name, as the census reports it

    def cited(self, axiom):
        """Whether any expression cites `axiom`."""
        start, _ = self.citing
        i = self.axioms.get(axiom)
        return i is not None and start[i + 1] > start[i]

    def reaching(self, axiom, across):
        """The names of the declarations that cite `axiom`, and if `across`, also those that
        cite a declaration that reaches it: a depth-first search over the reversed edges,
        which visits each expression, declaration, and name at most once. Without `across`, it
        stops at the first declarations, which are the citers."""
        i = self.axioms.get(axiom)
        if i is None:
            return set()
        p_start, p_out = self.parents
        c_start, c_out = self.citing
        h_start, h_out = self.holders
        names, seen_names = [i], {i}
        seen_exprs, seen_decls = set(), set()
        while names:
            n = names.pop()
            todo = []
            for e in c_out[c_start[n] : c_start[n + 1]]:
                if e not in seen_exprs:
                    seen_exprs.add(e)
                    todo.append(e)
            while todo:
                e = todo.pop()
                for d in h_out[h_start[e] : h_start[e + 1]]:
                    if d not in seen_decls:
                        seen_decls.add(d)
                        if across:
                            for m in self.owns[d]:
                                if m not in seen_names:
                                    seen_names.add(m)
                                    names.append(m)
                for q in p_out[p_start[e] : p_start[e + 1]]:
                    if q not in seen_exprs:
                        seen_exprs.add(q)
                        todo.append(q)
        return {self.labels[d] for d in seen_decls}

    def citers(self, axiom):
        """The declarations whose own expressions cite `axiom`."""
        return self.reaching(axiom, across=False)

    def dependents(self, axiom):
        """The declarations that depend on `axiom`, directly or through other declarations."""
        return self.reaching(axiom, across=True)


def scan(export_path):
    """One pass over the export, building the `Export` graph. Stops with an ERROR on any
    structural surprise."""
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

    # The graph's edges, reversed, as parallel arrays of (key, value) pairs; `grouped` turns
    # each into compressed sparse rows after the pass.
    sub_ids, sub_parents = array("I"), array("I")  # subexpression -> expression
    cited_names, citing_exprs = array("I"), array("I")  # name -> `const` expression
    held_exprs, holding_decls = array("I"), array("I")  # top-level expression -> declaration
    owns, labels = [], []  # declaration index -> its own name ids; its reported name
    axioms = {}  # declared axiom name -> name id
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
                if kind == "const":
                    cited_names.append(o["const"]["name"])
                    citing_exprs.append(i)
                for f2 in EXPR_SUBFIELDS.get(kind, ()):
                    v = o[kind][f2]
                    if not 0 <= v < i:
                        error(f"expression id {i} references non-earlier sub-id {v}")
                    sub_ids.append(v)
                    sub_parents.append(i)
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
                    axioms[resolve(d["name"])] = d["name"]
                if kind == "inductive":
                    own = tuple(x["name"] for part in ("types", "ctors", "recs") for x in d[part])
                    nm = ", ".join(resolve(ty["name"]) for ty in d["types"])
                else:
                    own = (d["name"],)
                    nm = resolve(d["name"])
                k = len(owns)
                owns.append(own)
                labels.append(nm)
                for v in decl_expr_ids(d):
                    if not 0 <= v <= max_ie:
                        error(f"declaration references undefined expression id {v}")
                    held_exprs.append(v)
                    holding_decls.append(k)

    if not meta_seen:
        error("export has no meta line; format version unverified")
    for nid in cited_names:
        if not 0 < nid <= max_in:
            error(f"an expression cites undefined name id {nid}")
    for own in owns:
        for nid in own:
            if not 0 < nid <= max_in:
                error(f"a declaration declares undefined name id {nid}")

    n_exprs, n_names = max_ie + 1, max_in + 1
    parents = grouped(n_exprs, sub_ids, sub_parents)
    del sub_ids, sub_parents
    citing = grouped(n_names, cited_names, citing_exprs)
    del cited_names, citing_exprs
    holders = grouped(n_exprs, held_exprs, holding_decls)
    return Export(axioms, parents, citing, holders, owns, labels)


def is_aeneas(name):
    """Whether a declaration counts as Aeneas' library's: its name is in the `Aeneas` namespace,
    or is a private name of one of Aeneas' modules.

    The export gives each declaration's name, type, and value, but not the module it was declared
    in, so the name is what is checked. Any module can declare into the `Aeneas` namespace, so
    `check_no_aeneas_declarations` checks that the package's own modules do not. That guards
    against an accident, not against code written to evade it."""
    return name.startswith(("Aeneas.", "_private.Aeneas."))


# A `namespace Aeneas`, or a declaration whose name is in the `Aeneas` namespace, in the
# package's own sources. A textual check: it catches the ways a module would declare into
# `Aeneas` by accident, and is not a defence against malicious code. It is not anchored to the
# start of a line, so that whatever precedes the keyword (attributes, modifiers, a docstring,
# `set_option ... in`) cannot hide the declaration; a comment that reads like one is flagged too.
AENEAS_DECLARATION = re.compile(
    r"\bnamespace\s+(?:_root_\.)?Aeneas\b"
    r"|\b(?:def|theorem|lemma|abbrev|instance|structure|inductive|class|axiom|opaque"
    r"|irreducible_def|alias)\s+(?:\([^)]*\)\s*)?(?:_root_\.)?Aeneas\."
    r"|\(name\s*:=\s*(?:_root_\.)?Aeneas\.",
)


def check_no_aeneas_declarations(root):
    """Report a violation for each place in the package's own `.lean` files, the root module
    `root.lean` and the modules under `root/`, that declares into the `Aeneas` namespace, which
    would make `is_aeneas` wrongly vouch for it."""
    if not root.is_dir():
        violation(f"{root}/ holds no modules to check; run from `lean/`")
        return
    paths = sorted(root.rglob("*.lean"))
    module = root.with_name(f"{root.name}.lean")
    if module.is_file():
        paths.insert(0, module)
    for path in paths:
        text = path.read_text()
        for m in AENEAS_DECLARATION.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            violation(f"{path}:{line} declares into the `Aeneas` namespace: {m.group(0).strip()}")


def is_aeneas_test(name):
    """Whether a declaration is one of Aeneas' tests: in the `Aeneas` namespace, with a `Test`
    component, like the inductives that `Aeneas.Data.ListN` checks the kernel rejects."""
    return name.startswith("Aeneas.") and "Test" in name.split(".")[1:]


def main():
    args = sys.argv[1:]
    nanoda_out = None
    if args[:1] == ["--nanoda-config"]:
        if len(args) < 2:
            error("--nanoda-config needs an output path")
        nanoda_out, args = Path(args[1]), args[2:]
    config_path = Path(args[0] if args else "scripts/nanoda-config.json")
    config = json.loads(config_path.read_text())
    permitted = set(config["permitted_axioms"])
    export_path = Path(config["export_file_path"])

    # Run from `lean/`: the package's own modules are `PastaCurves.lean` and `PastaCurves/`.
    check_no_aeneas_declarations(Path("PastaCurves"))
    export = scan(export_path)
    declared_axioms = set(export.axioms)
    # The axioms that Aeneas' library declares and the permitted list does not name: only
    # Aeneas' own declarations may depend on them.
    aeneas_axioms = {a for a in declared_axioms - permitted if is_aeneas(a)}
    citers = {t: export.citers(t) for t in TARGETS | aeneas_axioms}

    def citers_of(t):
        """The declarations citing `t`, with `<expression>` for a citation that no declaration
        reaches."""
        found = set(citers.get(t, set()))
        if export.cited(t) and not found:
            found.add("<expression>")
        return found

    def dependents_of(t):
        """The declarations depending on `t`, directly or through other declarations, with
        `<expression>` for a citation that no declaration reaches."""
        return export.dependents(t) | citers_of(t)

    def listed(names, limit=10):
        """`names`, sorted, with at most `limit` of them spelled out."""
        names = sorted(names)
        more = f" and {len(names) - limit} more" if len(names) > limit else ""
        return f"{names[:limit]}{more}"

    unpermitted = sorted(declared_axioms - permitted - aeneas_axioms)
    undeclared = sorted(permitted - declared_axioms)
    if unpermitted:
        violation(f"axiom(s) declared but not permitted, and not Aeneas': {unpermitted}")
    if undeclared:
        violation(
            f"permitted axiom(s) not declared in the export (stale census entry): {undeclared}"
        )

    for a in sorted(aeneas_axioms):
        outside = [c for c in dependents_of(a) if not is_aeneas(c)]
        if outside:
            violation(
                f"Aeneas' axiom '{a}' is depended on outside Aeneas' library: {listed(outside)}"
            )
    for t in sorted(UNREFERENCED):
        # `sorryAx` may be depended on by Aeneas' tests alone; the others by nothing.
        allowed = is_aeneas_test if t == "sorryAx" else (lambda _: False)
        outside = [c for c in dependents_of(t) if not allowed(c)]
        if outside:
            violation(f"'{t}' is depended on by: {listed(outside)}")
    for t, allowed in RESTRICTED.items():
        extra = citers.get(t, set()) - allowed
        if extra:
            violation(f"'{t}' is cited outside its allowance {sorted(allowed)}: {sorted(extra)}")

    # Print the full census whether or not anything was flagged: this is the actionable
    # state when a check above has flagged a stale or widened axiom list.
    watched = TARGETS | aeneas_axioms
    print(f"export axiom census: {len(declared_axioms)} axiom(s) declared:")
    for a in sorted(declared_axioms):
        cited = [] if a not in watched else sorted(citers_of(a)) or ["nothing"]
        note = f"  (cited by: {', '.join(cited)})" if cited else ""
        print(f"  {a}{note}")

    if violations:
        for msg in violations[:MAX_REPORTED_VIOLATIONS]:
            print(f"VIOLATION: {msg}", file=sys.stderr)
        if len(violations) > MAX_REPORTED_VIOLATIONS:
            print(
                f"... and {len(violations) - MAX_REPORTED_VIOLATIONS} further violation(s)",
                file=sys.stderr,
            )
        sys.exit(1)
    if nanoda_out is not None:
        # nanoda's strict mode rejects any declared axiom it is not told of, so it is told of
        # Aeneas', now that the census has checked that only Aeneas' library cites them.
        config["permitted_axioms"] = config["permitted_axioms"] + sorted(aeneas_axioms)
        nanoda_out.write_text(json.dumps(config, indent=4) + "\n")
    print("export axiom census: all checks passed")


if __name__ == "__main__":
    main()
