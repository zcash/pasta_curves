#!/usr/bin/env python3
"""Tests of `blueprint/sources.py`: the Lean and Rust indexes, the book's results, the conversion
to LaTeX, the expansion of LeanArchitect's nodes, the checks, and, when LeanArchitect's output is
present, the whole assembly on the repository's sources. Standard library only; run directly or
by `blueprint/build.sh`."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sources


def declarations_of(text):
    """The Lean declarations of one file with the given text."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "PastaCurves").mkdir()
        (root / "PastaCurves" / "A.lean").write_text(text)
        (root / "PastaCurves.lean").write_text("")
        saved = sources.ROOT
        sources.ROOT = root
        try:
            return {name: line for name, (_, line) in sources.lean_declarations(root).items()}
        finally:
            sources.ROOT = saved


class StripComments(unittest.TestCase):
    def test_keeps_lines(self):
        text = "a\n/- one\ntwo -/ b\n-- c\nd"
        stripped = sources.strip_comments(text)
        self.assertEqual(stripped.count("\n"), text.count("\n"))
        self.assertEqual([line.strip() for line in stripped.splitlines()], ["a", "", "b", "", "d"])

    def test_nested_and_docstrings(self):
        stripped = sources.strip_comments("/-- doc /- inner -/ still doc -/ theorem t")
        self.assertEqual(stripped.strip(), "theorem t")


class LeanDeclarations(unittest.TestCase):
    def test_namespaces(self):
        found = declarations_of(
            "namespace A.B\n"
            "theorem t : True := trivial\n"
            "end A.B\n"
            "namespace C\n"
            "section S\n"
            "/-- doc mentioning def fake -/\n"
            "@[simp] protected def d : Nat := 0\n"
            "end S\n"
            "theorem _root_.top : True := trivial\n"
            "end C\n"
            "def outside : Nat := 1\n"
        )
        self.assertEqual(found, {"A.B.t": 2, "C.d": 7, "top": 9, "outside": 11})

    def test_structure_fields(self):
        found = declarations_of("namespace N\nstructure S where\n  x : Nat\n  y : Nat\nend N\n")
        self.assertEqual(found, {"N.S": 2, "N.S.x": 3, "N.S.y": 4})


class Markdown(unittest.TestCase):
    def test_conversion(self):
        tex = sources.markdown_to_tex("*Some* `a_b` and $x_1^2$ cost 50% #1\n\n$$\ny\n$$")
        self.assertEqual(
            tex, r"\emph{Some} \texttt{a\_b} and $x_1^2$ cost 50\% \#1" + "\n\n" + r"\[y\]"
        )


class Expansion(unittest.TestCase):
    NODE = (
        "\\begin{lemma}[\\booktitle{lemma-1}]\n"
        "\\uses{b,a,b}\n"
        "\\label{lemma-1}\n"
        "\\lean{N.t}\n"
        "% at /home/someone/N.lean:1.0-2.0\n"
        "\\leanok\n"
        "\\bookquote{lemma-1}\n"
        "\\touches{f:g}\n"
        "\\end{lemma}\n"
    )

    def expander(self):
        book = {"lemma-1": ("Lemma 1 (one)", "It holds for $x_1$.", ["t"])}
        return sources.Expander("abc", {}, {"f:g": ("src/asm/f.rs", 7)}, book, {})

    def test_node(self):
        expander = self.expander()
        tex = expander.node(self.NODE)
        self.assertEqual(expander.errors, [])
        self.assertIn("[Lemma 1 (one)]", tex)
        self.assertIn(r"\uses{a,b}", tex)
        self.assertNotIn("% at", tex)
        self.assertIn("It holds for $x_1$.", tex)
        self.assertIn(r"\href{../design/inversion.html\#lemma-1}{Lemma 1 (one)}", tex)
        self.assertIn(r"/blob/abc/src/asm/f.rs\#L7}{\texttt{g}}", tex)

    def test_unknown_names(self):
        expander = self.expander()
        expander.node("\\bookquote{lemma-2} \\leandoc{N.u} \\touches{f:h}")
        self.assertEqual(
            expander.errors,
            [
                "no result anchored lemma-2 in the book page",
                "no docstring for the Lean declaration N.u",
                "no Rust item f:h",
            ],
        )


class Checks(unittest.TestCase):
    def test_book_nodes(self):
        book = {"lemma-1": ("Lemma 1", "", ["t"])}
        lean = {"N.t": ("", 1), "N.u": ("", 2)}
        nodes = {"lemma-1": "\\lean{N.t,N.u}", "N.v": "\\lean{N.v}"}
        self.assertEqual(
            sources.check_book_nodes(nodes, book, lean),
            ["N.u is on the node lemma-1, but the book does not name it there"],
        )

    def test_cycle(self):
        self.assertIsNone(sources.find_cycle({"a": {"b"}, "b": {"c"}, "c": set()}))
        self.assertEqual(sources.find_cycle({"a": {"b"}, "b": {"a"}}), ["a", "b", "a"])


class Repository(unittest.TestCase):
    """The indexes and the assembly on the actual sources."""

    def test_lean_names(self):
        lean = sources.lean_declarations()
        for name in (
            "PastaCurves.Inversion.M_spec",
            "PastaCurves.Inversion.Hull.terminationBound_256",
            "PastaCurves.AArch64.signMagBlock_spec",
            "PastaCurves.invert_entry_spec",
            "PastaCurves.AArch64.invert_entry_spec",
        ):
            self.assertIn(name, lean)

    def test_rust_items(self):
        rust = sources.rust_items()
        path, line = rust["aarch64:sign_mag"]
        self.assertEqual(path, "src/asm/aarch64.rs")
        text = (sources.ROOT / path).read_text().splitlines()[line - 1]
        self.assertIn("fn sign_mag(", text)
        self.assertIn("aarch64:divstep", rust)
        self.assertIn("inversion:invert", rust)
        self.assertNotIn("aarch64:sign_mag_known_answers", rust)

    def test_book_results(self):
        title, statement, names = sources.book_results()["lemma-6-prime"]
        self.assertEqual(title, "Lemma 6\u2032 (the sum does not wrap)")
        self.assertTrue(statement.startswith("For $j < k$"))
        self.assertNotIn("*Proof.*", statement)
        self.assertEqual(names, ["divsteps_packedStart_g_abs_lt"])

    @unittest.skipUnless(sources.ARCHITECT_MODULE.is_file(), "LeanArchitect's output is absent")
    def test_assembly(self):
        content, errors = sources.assemble("0" * 40)
        self.assertEqual(errors, [])
        self.assertEqual(content.count(r"\label{"), len(sources.architect_nodes()))
        self.assertNotIn(r"\bookquote", content)
        self.assertNotIn("% at ", content)


if __name__ == "__main__":
    unittest.main()
