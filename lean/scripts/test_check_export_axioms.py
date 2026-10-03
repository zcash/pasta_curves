#!/usr/bin/env python3
"""Tests for scripts/check_export_axioms.py.

The checker is a security control: it is what makes "no `sorry` and no
`native_decide` anywhere in the development" a machine-checked claim rather than a
convention, so its own failure modes need to be pinned. A checker that silently
stops detecting something is worse than no checker, because the green CI job is
then evidence of nothing.

The tests drive the real entry point as a SUBPROCESS over a synthetic export,
because the exit code is the contract CI consumes: 0 clean, 1 VIOLATION (well
formed data, undesired outcome), 2 ERROR (the export's structure is not as the
script assumes). Importing and calling `main()` would not exercise that, and the
script deliberately has no seam for it.

Three groups carry the weight:

  * `Citations` walks EVERY expression-valued field in `EXPR_SUBFIELDS` and every
    declaration path in `DECL_EXPR_KEYS`, hiding a forbidden const behind exactly
    that one field. Dropping a field from either table turns a test red instead of
    turning the audit into a no-op. The builder raises rather than skipping when it
    meets a kind it cannot construct, so ADDING a kind also fails until the test is
    extended.
  * `SpecParity` pins the three kind tables against lean4export's published format,
    so a format bump cannot widen or narrow them unnoticed.
  * `StructuralErrors` pins the fail-closed behaviour: every malformed shape must
    exit 2 and print NO census, since a census computed from misparsed data would
    be a false reassurance.

Stdlib only, no test runner to install, matching the crate's zero-dependency stance
and the stdlib-python convention of its sibling scripts.

Usage, from the `lean/` directory:  python3 scripts/test_check_export_axioms.py
"""

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

SCRIPT = Path(__file__).resolve().parent / "check_export_axioms.py"


def _load(path):
    spec = importlib.util.spec_from_file_location("check_export_axioms", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Importing is safe: everything is behind `if __name__ == "__main__"`. Reading the
# tables from the script (rather than copying them) is what lets the parametrized
# tests below stay in step with it.
CHECKER = _load(SCRIPT)

# The kinds of lean4export's ndjson format, transcribed from `format_ndjson.md` at
# the tag named by `lean/lean-toolchain`. `zero` is deliberately absent from the
# levels: it is never emitted as a line, it is the reserved id 0. Constructors and
# recursors are deliberately absent from the declarations: they are nested inside
# the `inductive` object as `ctors` and `recs`, not top-level kinds. Update these
# three sets and FORMAT_VERSION together, from the spec, never from the script.
SPEC_EXPR_KINDS = {
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
SPEC_LEVEL_KINDS = {"imax", "max", "param", "succ"}
# The expression-valued fields of each kind, likewise from the spec. The
# parametrized tests below are driven from HERE and not from the script's own
# table: a test parametrized by the thing it tests cannot detect a deletion from
# it, because the deleted case simply stops being generated.
SPEC_EXPR_SUBFIELDS = {
    "app": ("fn", "arg"),
    "forallE": ("type", "body"),
    "lam": ("type", "body"),
    "letE": ("type", "value", "body"),
    "mdata": ("expr",),
    "proj": ("struct",),
}
SPEC_DECL_KINDS = {"axiom", "def", "inductive", "opaque", "quot", "thm"}
SPEC_FORMAT_VERSION = "3.1.0"
# The policy the checker exists to enforce, stated here independently of it: these
# three axioms must be cited by nothing, so a `sorry` or a `native_decide` anywhere
# in the development fails the run.
SPEC_UNREFERENCED = {"sorryAx", "Lean.ofReduceBool", "Lean.ofReduceNat"}


class Export:
    """Builds a structurally well-formed ndjson export, line by line.

    Ids are allocated exactly as the format requires, so a test opts into a
    malformation explicitly with `raw`, and never by accident.
    """

    def __init__(self, version=SPEC_FORMAT_VERSION, meta=True):
        self.lines = []
        self._names = {}
        self._n, self._e, self._l = 0, -1, 0
        if meta:
            self.raw(
                {
                    "meta": {
                        "exporter": {"name": "test", "version": "0"},
                        "lean": {"githash": "0", "version": "0"},
                        "format": {"version": version},
                    }
                }
            )

    def raw(self, obj):
        """Append a line verbatim, bypassing id allocation."""
        self.lines.append(obj)
        return self

    def name(self, dotted):
        """Intern a dotted name, emitting the prefix chain it needs."""
        if dotted in self._names:
            return self._names[dotted]
        pre, acc = 0, ""
        for comp in dotted.split("."):
            acc = f"{acc}.{comp}" if acc else comp
            if acc not in self._names:
                self._n += 1
                self.raw({"str": {"pre": pre, "str": comp}, "in": self._n})
                self._names[acc] = self._n
            pre = self._names[acc]
        return self._names[dotted]

    # -- expressions --------------------------------------------------------
    def _expr(self, payload):
        self._e += 1
        self.raw({**payload, "ie": self._e})
        return self._e

    def sort(self, level=0):
        return self._expr({"sort": level})

    def bvar(self, idx=0):
        return self._expr({"bvar": idx})

    def const(self, dotted, us=()):
        return self._expr({"const": {"name": self.name(dotted), "us": list(us)}})

    def app(self, fn, arg):
        return self._expr({"app": {"fn": fn, "arg": arg}})

    def lam(self, ty, body, binder="x"):
        return self._expr(
            {"lam": {"name": self.name(binder), "type": ty, "body": body, "binderInfo": "default"}}
        )

    def forall_e(self, ty, body, binder="x"):
        return self._expr(
            {
                "forallE": {
                    "name": self.name(binder),
                    "type": ty,
                    "body": body,
                    "binderInfo": "default",
                }
            }
        )

    def let_e(self, ty, value, body, binder="x"):
        return self._expr(
            {
                "letE": {
                    "name": self.name(binder),
                    "type": ty,
                    "value": value,
                    "body": body,
                    "nondep": False,
                }
            }
        )

    def proj(self, struct, type_name="S", idx=0):
        return self._expr(
            {"proj": {"typeName": self.name(type_name), "idx": idx, "struct": struct}}
        )

    def mdata(self, expr):
        return self._expr({"mdata": {"expr": expr, "data": {"x": True}}})

    def nat_val(self, v="0"):
        return self._expr({"natVal": str(v)})

    def str_val(self, v="s"):
        return self._expr({"strVal": v})

    # -- levels -------------------------------------------------------------
    def _level(self, payload):
        self._l += 1
        self.raw({**payload, "il": self._l})
        return self._l

    def level_succ(self, u=0):
        return self._level({"succ": u})

    def level_max(self, a=0, b=0):
        return self._level({"max": [a, b]})

    def level_imax(self, a=0, b=0):
        return self._level({"imax": [a, b]})

    def level_param(self, dotted="u"):
        return self._level({"param": self.name(dotted)})

    # -- declarations -------------------------------------------------------
    def axiom(self, dotted, ty=None):
        return self.raw(
            {
                "axiom": {
                    "name": self.name(dotted),
                    "levelParams": [],
                    "type": self.sort() if ty is None else ty,
                    "isUnsafe": False,
                }
            }
        )

    def defn(self, dotted, ty=None, value=None, hints="opaque", all_=()):
        return self.raw(
            {
                "def": {
                    "name": self.name(dotted),
                    "levelParams": [],
                    "type": self.sort() if ty is None else ty,
                    "value": self.sort() if value is None else value,
                    "hints": hints,
                    "safety": "safe",
                    "all": list(all_),
                }
            }
        )

    def thm(self, dotted, ty=None, value=None):
        return self.raw(
            {
                "thm": {
                    "name": self.name(dotted),
                    "levelParams": [],
                    "type": self.sort() if ty is None else ty,
                    "value": self.sort() if value is None else value,
                    "all": [],
                }
            }
        )

    def opaque(self, dotted, ty=None, value=None):
        return self.raw(
            {
                "opaque": {
                    "name": self.name(dotted),
                    "levelParams": [],
                    "type": self.sort() if ty is None else ty,
                    "value": self.sort() if value is None else value,
                    "isUnsafe": False,
                    "all": [],
                }
            }
        )

    def quot(self, dotted="Quot", ty=None, kind="type"):
        return self.raw(
            {
                "quot": {
                    "name": self.name(dotted),
                    "levelParams": [],
                    "type": self.sort() if ty is None else ty,
                    "kind": kind,
                }
            }
        )

    def inductive(self, dotted="I", ty=None, ctor_ty=None, rec_ty=None, rule_rhs=None):
        ty = self.sort() if ty is None else ty
        ctor_ty = self.sort() if ctor_ty is None else ctor_ty
        rec_ty = self.sort() if rec_ty is None else rec_ty
        rule_rhs = self.sort() if rule_rhs is None else rule_rhs
        return self.raw(
            {
                "inductive": {
                    "types": [
                        {
                            "name": self.name(dotted),
                            "levelParams": [],
                            "type": ty,
                            "numParams": 0,
                            "numIndices": 0,
                            "all": [],
                            "ctors": [],
                            "numNested": 0,
                            "isRec": False,
                            "isUnsafe": False,
                            "isReflexive": False,
                        }
                    ],
                    "ctors": [
                        {
                            "name": self.name(f"{dotted}.mk"),
                            "levelParams": [],
                            "type": ctor_ty,
                            "induct": self.name(dotted),
                            "cidx": 0,
                            "numParams": 0,
                            "numFields": 0,
                            "isUnsafe": False,
                        }
                    ],
                    "recs": [
                        {
                            "name": self.name(f"{dotted}.rec"),
                            "levelParams": [],
                            "type": rec_ty,
                            "all": [],
                            "numParams": 0,
                            "numIndices": 0,
                            "numMotives": 1,
                            "numMinors": 1,
                            "rules": [
                                {"ctor": self.name(f"{dotted}.mk"), "nfields": 0, "rhs": rule_rhs}
                            ],
                            "k": False,
                            "isUnsafe": False,
                        }
                    ],
                }
            }
        )

    def render(self):
        return "".join(json.dumps(o) + "\n" for o in self.lines).encode()


def run(export, permitted=("propext",)):
    """Run the checker over `export` and return the CompletedProcess."""
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ndjson = tmp / "export.ndjson"
        ndjson.write_bytes(export.render())
        config = tmp / "config.json"
        config.write_text(
            json.dumps({"export_file_path": str(ndjson), "permitted_axioms": list(permitted)})
        )
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(config)], capture_output=True, text=True, check=False
        )


def baseline(**kw):
    """The smallest export the checker accepts: one declared, permitted axiom."""
    return Export(**kw).axiom("propext")


def expr_of_kind(e, kind):
    """One expression of `kind`. Raises on a kind with no constructor here, so a
    kind added to EXPR_KINDS fails this suite until it is modelled."""
    build = {
        "app": lambda: e.app(e.sort(), e.sort()),
        "bvar": lambda: e.bvar(0),
        "const": lambda: e.const("Foo.bar"),
        "forallE": lambda: e.forall_e(e.sort(), e.sort()),
        "lam": lambda: e.lam(e.sort(), e.sort()),
        "letE": lambda: e.let_e(e.sort(), e.sort(), e.sort()),
        "mdata": lambda: e.mdata(e.sort()),
        "natVal": lambda: e.nat_val(7),
        "proj": lambda: e.proj(e.sort()),
        "sort": lambda: e.sort(0),
        "strVal": lambda: e.str_val("hi"),
    }.get(kind)
    if build is None:
        raise AssertionError(
            f"the test builder cannot construct kind {kind!r}; "
            f"model it here when adding it to EXPR_KINDS"
        )
    return build()


def level_of_kind(e, kind):
    build = {
        "imax": lambda: e.level_imax(),
        "max": lambda: e.level_max(),
        "param": lambda: e.level_param(),
        "succ": lambda: e.level_succ(),
    }.get(kind)
    if build is None:
        raise AssertionError(f"the test builder cannot construct level kind {kind!r}")
    return build()


def hide_in(e, kind, field, hidden):
    """Build one expression of `kind` whose `field` is `hidden` and whose every
    other expression field is inert filler."""
    filler = e.sort()

    def pick(f):
        return hidden if f == field else filler

    build = {
        "app": lambda: e.app(pick("fn"), pick("arg")),
        "forallE": lambda: e.forall_e(pick("type"), pick("body")),
        "lam": lambda: e.lam(pick("type"), pick("body")),
        "letE": lambda: e.let_e(pick("type"), pick("value"), pick("body")),
        "mdata": lambda: e.mdata(pick("expr")),
        "proj": lambda: e.proj(pick("struct")),
    }.get(kind)
    if build is None:
        raise AssertionError(
            f"the test builder cannot construct kind {kind!r}; "
            f"model it here when adding it to EXPR_SUBFIELDS"
        )
    return build()


class SpecParity(unittest.TestCase):
    """The kind tables must match the published format, not merely each other."""

    def test_expr_kinds(self):
        self.assertEqual(CHECKER.EXPR_KINDS, SPEC_EXPR_KINDS)

    def test_level_kinds(self):
        self.assertEqual(CHECKER.LEVEL_KINDS, SPEC_LEVEL_KINDS)

    def test_decl_kinds(self):
        self.assertEqual(CHECKER.DECL_KINDS, SPEC_DECL_KINDS)

    def test_unreferenced_policy(self):
        self.assertEqual(CHECKER.UNREFERENCED, SPEC_UNREFERENCED)

    def test_format_version(self):
        self.assertEqual(CHECKER.FORMAT_VERSION, SPEC_FORMAT_VERSION)

    def test_expr_subfields(self):
        self.assertEqual(
            {k: tuple(v) for k, v in CHECKER.EXPR_SUBFIELDS.items()}, SPEC_EXPR_SUBFIELDS
        )

    def test_every_subfield_kind_is_a_known_kind(self):
        self.assertLessEqual(set(CHECKER.EXPR_SUBFIELDS), CHECKER.EXPR_KINDS)


class Accepts(unittest.TestCase):
    """What a clean export looks like, and that every declared kind parses."""

    def test_baseline_passes(self):
        p = run(baseline())
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("all checks passed", p.stdout)
        self.assertIn("propext", p.stdout)

    def test_every_expr_kind_parses(self):
        for kind in sorted(SPEC_EXPR_KINDS):
            with self.subTest(kind=kind):
                e = baseline()
                expr_of_kind(e, kind)
                self.assertEqual(run(e).returncode, 0)

    def test_every_level_kind_parses(self):
        for kind in sorted(SPEC_LEVEL_KINDS):
            with self.subTest(kind=kind):
                e = baseline()
                level_of_kind(e, kind)
                self.assertEqual(run(e).returncode, 0)

    def test_every_decl_kind_parses(self):
        builders = {
            "axiom": lambda e: e.axiom("Other"),
            "def": lambda e: e.defn("D"),
            "inductive": lambda e: e.inductive(),
            "opaque": lambda e: e.opaque("O"),
            "quot": lambda e: e.quot(),
            "thm": lambda e: e.thm("T"),
        }
        self.assertEqual(set(builders), SPEC_DECL_KINDS)
        for kind, build in sorted(builders.items()):
            with self.subTest(kind=kind):
                e = baseline()
                build(e)
                permitted = ("propext", "Other") if kind == "axiom" else ("propext",)
                self.assertEqual(run(e, permitted).returncode, 0)

    def test_non_target_const_is_not_a_citation(self):
        e = baseline()
        e.defn("User", value=e.const("Nat.succ"))
        self.assertEqual(run(e).returncode, 0)

    def test_integers_outside_expression_keys_are_not_expression_ids(self):
        # `hints: {"regular": N}` and `all: [name ids]` hold integers that are not
        # expression ids. Treating them as such would error on an undefined id, so
        # this pins the DECL_EXPR_KEYS filter rather than a happy accident.
        e = baseline()
        e.defn("D", hints={"regular": 9999}, all_=[8888])
        p = run(e)
        self.assertEqual(p.returncode, 0, p.stderr)

    def test_trust_compiler_cited_by_exactly_its_allowance(self):
        e = baseline()
        e.axiom("Lean.trustCompiler")
        e.opaque("Lean.reduceBool", value=e.const("Lean.trustCompiler"))
        e.opaque("Lean.reduceNat", value=e.const("Lean.trustCompiler"))
        p = run(e, ("propext", "Lean.trustCompiler"))
        self.assertEqual(p.returncode, 0, p.stderr)


class Citations(unittest.TestCase):
    """The audit proper: a forbidden const must be found wherever it hides."""

    def _cited_via_expr(self, kind, field):
        e = baseline()
        e.axiom("sorryAx")
        top = hide_in(e, kind, field, e.const("sorryAx"))
        e.defn("User", value=top)
        return run(e, ("propext", "sorryAx"))

    def test_found_behind_every_expression_field(self):
        for kind, fields in sorted(SPEC_EXPR_SUBFIELDS.items()):
            for field in fields:
                with self.subTest(kind=kind, field=field):
                    p = self._cited_via_expr(kind, field)
                    self.assertEqual(p.returncode, 1, p.stdout)
                    self.assertIn("sorryAx", p.stderr)
                    self.assertIn("User", p.stderr)

    def test_found_through_a_chain_of_expressions(self):
        e = baseline()
        e.axiom("sorryAx")
        deep = e.const("sorryAx")
        for _ in range(8):
            deep = e.lam(e.sort(), e.app(e.sort(), deep))
        e.thm("Deep", value=deep)
        p = run(e, ("propext", "sorryAx"))
        self.assertEqual(p.returncode, 1, p.stdout)
        self.assertIn("Deep", p.stderr)

    def test_found_in_every_declaration_position(self):
        def via(build, permitted=("propext", "sorryAx")):
            e = baseline()
            e.axiom("sorryAx")
            build(e, e.const("sorryAx"))
            return run(e, permitted)

        cases = {
            "def.value": lambda e, c: e.defn("User", value=c),
            "def.type": lambda e, c: e.defn("User", ty=c),
            "thm.value": lambda e, c: e.thm("User", value=c),
            "thm.type": lambda e, c: e.thm("User", ty=c),
            "opaque.value": lambda e, c: e.opaque("User", value=c),
            "quot.type": lambda e, c: e.quot("User", ty=c),
            "inductive.type": lambda e, c: e.inductive("User", ty=c),
            "inductive.ctor.type": lambda e, c: e.inductive("User", ctor_ty=c),
            "inductive.rec.type": lambda e, c: e.inductive("User", rec_ty=c),
            "inductive.rec.rule.rhs": lambda e, c: e.inductive("User", rule_rhs=c),
        }
        for where, build in sorted(cases.items()):
            with self.subTest(where=where):
                p = via(build)
                self.assertEqual(p.returncode, 1, p.stdout)
                self.assertIn("sorryAx", p.stderr)
                self.assertIn("User", p.stderr)

    def test_found_with_no_declaration_reaching_it(self):
        # The `const_cited` path: an expression naming a forbidden axiom is a
        # violation even when no declaration payload points at it, so the audit
        # does not depend on reachability from a declaration.
        e = baseline()
        e.axiom("sorryAx")
        e.const("sorryAx")
        p = run(e, ("propext", "sorryAx"))
        self.assertEqual(p.returncode, 1, p.stdout)
        self.assertIn("sorryAx", p.stderr)

    def test_each_unreferenced_axiom_is_watched(self):
        for target in sorted(SPEC_UNREFERENCED):
            with self.subTest(target=target):
                e = baseline()
                e.axiom(target)
                e.defn("User", value=e.const(target))
                p = run(e, ("propext", target))
                self.assertEqual(p.returncode, 1, p.stdout)
                self.assertIn(target, p.stderr)

    def test_restricted_axiom_cited_outside_its_allowance(self):
        e = baseline()
        e.axiom("Lean.trustCompiler")
        e.opaque("Lean.reduceBool", value=e.const("Lean.trustCompiler"))
        e.defn("Sneaky", value=e.const("Lean.trustCompiler"))
        p = run(e, ("propext", "Lean.trustCompiler"))
        self.assertEqual(p.returncode, 1, p.stdout)
        # The offending set must be exactly {Sneaky}: the two allowed citers are
        # named in the message as the allowance, not as offenders.
        self.assertIn("['Sneaky']", p.stderr)


class AxiomCensus(unittest.TestCase):
    """The declared set must equal the permitted set, in both directions."""

    def test_declared_but_not_permitted(self):
        e = baseline()
        e.axiom("MyAxiom")
        p = run(e)
        self.assertEqual(p.returncode, 1, p.stdout)
        self.assertIn("MyAxiom", p.stderr)
        self.assertIn("not permitted", p.stderr)

    def test_permitted_but_not_declared_is_a_stale_entry(self):
        p = run(baseline(), ("propext", "Gone.axiom"))
        self.assertEqual(p.returncode, 1, p.stdout)
        self.assertIn("Gone.axiom", p.stderr)
        self.assertIn("stale", p.stderr)

    def test_census_prints_even_when_violations_are_found(self):
        e = baseline()
        e.axiom("MyAxiom")
        p = run(e)
        self.assertEqual(p.returncode, 1)
        self.assertIn("export axiom census", p.stdout)
        self.assertIn("MyAxiom", p.stdout)


class StructuralErrors(unittest.TestCase):
    """Anything the script cannot parse confidently must exit 2 with no census."""

    def assertStructuralError(self, export, needle):
        p = run(export)
        self.assertEqual(p.returncode, 2, p.stdout)
        self.assertIn(needle, p.stderr)
        self.assertNotIn("export axiom census", p.stdout)

    def test_unknown_expression_kind(self):
        e = baseline()
        e._e += 1
        e.raw({"fvar": 3, "ie": e._e})
        self.assertStructuralError(e, "unknown expression kind")

    def test_zero_level_emitted_as_a_line_is_rejected(self):
        # `zero` is the reserved id 0 and never a line; emitting one means the
        # format changed under us.
        e = baseline()
        e.raw({"zero": 0, "il": 1})
        self.assertStructuralError(e, "unknown level kind")

    def test_unknown_declaration_kind(self):
        e = baseline()
        e.raw({"ctor": {"name": 1}})
        self.assertStructuralError(e, "unknown line shape")

    def test_extra_key_on_an_expression_line(self):
        e = baseline()
        e._e += 1
        e.raw({"sort": 0, "note": "x", "ie": e._e})
        self.assertStructuralError(e, "unknown expression line shape")

    def test_extra_key_on_a_name_line(self):
        e = baseline()
        e._n += 1
        e.raw({"str": {"pre": 0, "str": "x"}, "note": 1, "in": e._n})
        self.assertStructuralError(e, "unknown name line shape")

    def test_two_declaration_keys_on_one_line(self):
        e = baseline()
        e.raw({"def": {"name": 1}, "thm": {"name": 1}})
        self.assertStructuralError(e, "unknown line shape")

    def test_unknown_name_entry_shape(self):
        e = baseline()
        e._n += 1
        e.raw({"sym": {"pre": 0, "s": "x"}, "in": e._n})
        self.assertStructuralError(e, "unknown name entry shape")

    def test_non_dense_expression_id(self):
        e = baseline()
        e.raw({"sort": 0, "ie": e._e + 5})
        self.assertStructuralError(e, "does not follow")

    def test_non_dense_name_id(self):
        e = baseline()
        e.raw({"str": {"pre": 0, "str": "x"}, "in": e._n + 5})
        self.assertStructuralError(e, "does not follow")

    def test_non_dense_level_id(self):
        e = baseline()
        e.raw({"succ": 0, "il": 7})
        self.assertStructuralError(e, "does not follow")

    def test_expression_referencing_a_later_sub_id(self):
        e = baseline()
        e._e += 1
        e.raw({"app": {"fn": e._e + 3, "arg": 0}, "ie": e._e})
        self.assertStructuralError(e, "non-earlier sub-id")

    def test_expression_referencing_itself(self):
        e = baseline()
        e._e += 1
        e.raw({"app": {"fn": e._e, "arg": 0}, "ie": e._e})
        self.assertStructuralError(e, "non-earlier sub-id")

    def test_level_referencing_a_later_sub_id(self):
        e = baseline()
        e.raw({"succ": 9, "il": 1})
        self.assertStructuralError(e, "non-earlier sub-id")

    def test_name_referencing_a_later_prefix(self):
        e = baseline()
        e._n += 1
        e.raw({"str": {"pre": e._n + 2, "str": "x"}, "in": e._n})
        self.assertStructuralError(e, "non-earlier prefix")

    def test_declaration_referencing_an_undefined_expression(self):
        e = baseline()
        e.raw(
            {
                "def": {
                    "name": e.name("D"),
                    "levelParams": [],
                    "type": 9999,
                    "value": 0,
                    "hints": "opaque",
                    "safety": "safe",
                    "all": [],
                }
            }
        )
        self.assertStructuralError(e, "undefined expression id")

    def test_missing_meta_line(self):
        self.assertStructuralError(baseline(meta=False), "no meta line")

    def test_wrong_format_version(self):
        self.assertStructuralError(baseline(version="2.0.0"), "format version")


if __name__ == "__main__":
    unittest.main()
