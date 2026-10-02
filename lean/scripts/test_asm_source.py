#!/usr/bin/env python3
"""Regression tests for fail-closed Rust surrounding-code parsing."""

import re
import sys
import unittest

# Running this source-tree test should not leave lean/scripts/__pycache__ behind.
sys.dont_write_bytecode = True

from asm2lean import rust
from pasta import aarch64_blocks, x86_64_blocks


class SurroundingCodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.aarch64_source = aarch64_blocks.SOURCE.read_text()
        cls.x86_64_source = x86_64_blocks.SOURCE.read_text()

    @staticmethod
    def mutate_function(source, name, old, new):
        masked = rust.masked_noncode(source)
        match = re.search(rf"(?m)^.*\bfn\s+{re.escape(name)}\s*\(", masked)
        if match is None:
            raise AssertionError(f"function {name} not found")
        signature_open = masked.find("(", match.start())
        signature_close = rust.matching_delimiter(source, signature_open, "(", ")")
        body_open = masked.find("{", signature_close)
        body_close = rust.matching_delimiter(source, body_open, "{", "}")
        function_source = source[match.start() : body_close + 1]
        if function_source.count(old) != 1:
            raise AssertionError(f"expected one {old!r} in {name}")
        return (
            source[: match.start()] + function_source.replace(old, new) + source[body_close + 1 :]
        )

    @staticmethod
    def generate_aarch64(source):
        return aarch64_blocks.text(source)

    def test_current_sources_generate_committed_output(self):
        self.assertEqual(
            self.generate_aarch64(self.aarch64_source),
            aarch64_blocks.OUTPUT.read_text(),
        )
        self.assertEqual(
            x86_64_blocks.text(self.x86_64_source),
            x86_64_blocks.OUTPUT.read_text(),
        )

    def test_inv_shadowing_is_rejected_by_both_backends(self):
        for architecture, source, generate in (
            ("aarch64", self.aarch64_source, self.generate_aarch64),
            ("x86_64", self.x86_64_source, x86_64_blocks.text),
        ):
            with self.subTest(architecture=architecture):
                mutated = self.mutate_function(
                    source,
                    "mul",
                    "    let (o0, o1, o2, o3): (u64, u64, u64, u64);",
                    "    let inv = 0;\n    let (o0, o1, o2, o3): (u64, u64, u64, u64);",
                )
                with self.assertRaisesRegex(
                    rust.GenerationError,
                    "local inv shadows a function argument",
                ):
                    generate(mutated)

    def test_postasm_output_mutation_is_rejected_by_both_backends(self):
        for architecture, source, generate in (
            ("aarch64", self.aarch64_source, self.generate_aarch64),
            ("x86_64", self.x86_64_source, x86_64_blocks.text),
        ):
            with self.subTest(architecture=architecture):
                mutated = self.mutate_function(
                    source,
                    "add",
                    "    }\n    [r0, r1, r2, r3]",
                    "    }\n    r0 = 0;\n    [r0, r1, r2, r3]",
                )
                with self.assertRaisesRegex(
                    rust.GenerationError,
                    "unsupported code after asm!",
                ):
                    generate(mutated)

    def test_comments_are_allowed_between_surrounding_grammar_tokens(self):
        for architecture, source, generate, expected in (
            (
                "aarch64",
                self.aarch64_source,
                self.generate_aarch64,
                aarch64_blocks.OUTPUT.read_text(),
            ),
            (
                "x86_64",
                self.x86_64_source,
                x86_64_blocks.text,
                x86_64_blocks.OUTPUT.read_text(),
            ),
        ):
            with self.subTest(architecture=architecture):
                commented = self.mutate_function(
                    source,
                    "add",
                    "    let [mut r0, mut r1, mut r2, mut r3] = *lhs;",
                    "    let /* outputs */ [mut r0, mut r1, mut r2, mut r3] = * /* input */ lhs;",
                )
                self.assertEqual(generate(commented), expected)


class MacroTests(unittest.TestCase):
    def parse(self, definitions):
        return rust.parse_macros(definitions)

    def test_arms_expand_literals_and_earlier_arms(self):
        macros = self.parse(
            "macro_rules! step {\n"
            '    (core) => { concat!("add {a}, {a}, #1\\n", "sub {b}, {b}, #1\\n") };\n'
            '    () => { concat!(step!(core), "tst {b}, #2\\n") };\n'
            '    (last) => { "add {a}, {a}, #2\\n" };\n'
            "}\n"
        )
        self.assertEqual(
            macros,
            {
                "step": {
                    "core": ("add {a}, {a}, #1", "sub {b}, {b}, #1"),
                    "": ("add {a}, {a}, #1", "sub {b}, {b}, #1", "tst {b}, #2"),
                    "last": ("add {a}, {a}, #2",),
                }
            },
        )

    def test_malformed_macros_are_rejected(self):
        cases = {
            "must end with a newline": 'macro_rules! m { () => { "add {a}, {a}, #1" }; }',
            "escape only": 'macro_rules! m { () => { "add {a}, {a}, #1\\t\\n" }; }',
            "no arm for `x`": 'macro_rules! m { () => { concat!(m!(x), "a\\n") }; }',
            "unknown macro n!": 'macro_rules! m { () => { concat!(n!(), "a\\n") }; }',
            "duplicate arm": 'macro_rules! m { () => { "a\\n" }; () => { "b\\n" }; }',
            "unsupported macro pattern": 'macro_rules! m { ($x:tt) => { "a\\n" }; }',
            "string literal or `concat!`": 'macro_rules! m { () => { format!("a\\n") }; }',
            "empty line": 'macro_rules! m { () => { "a\\n\\n" }; }',
        }
        for message, source in cases.items():
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(rust.GenerationError, message),
            ):
                self.parse(source)

    def test_invocations_in_template_position_expand_with_origins(self):
        source = (
            'macro_rules! step { () => { concat!("add {a}, {a}, #1\\n", "add {a}, {a}, #1\\n") }; }\n'
            "fn f(mut a: u64) -> [u64; 1] {\n"
            "    unsafe {\n"
            '        asm!("mov {a}, {a}", step!(), step!(), "mov {a}, {a}", a = inout(reg) a,\n'
            "            options(pure, nomem, nostack));\n"
            "    }\n"
            "    [a]\n"
            "}\n"
        )
        parsed = rust.parse_function(
            source, "f", ["a"], 1, required_options={"pure", "nomem", "nostack"}
        )
        self.assertEqual(len(parsed.instructions), 6)
        self.assertEqual(
            parsed.origins,
            (
                None,
                rust.MacroOrigin("step", "", 0),
                rust.MacroOrigin("step", "", 0),
                rust.MacroOrigin("step", "", 1),
                rust.MacroOrigin("step", "", 1),
                None,
            ),
        )
        with self.assertRaisesRegex(rust.GenerationError, "unknown macro other!"):
            rust.parse_function(
                source.replace("step!(), step!()", "other!()"),
                "f",
                ["a"],
                1,
                required_options={"pure", "nomem", "nostack"},
            )
        with self.assertRaisesRegex(rust.GenerationError, "has no arm for `last`"):
            rust.parse_function(
                source.replace("step!(), step!()", "step!(last)"),
                "f",
                ["a"],
                1,
                required_options={"pure", "nomem", "nostack"},
            )

    def test_block_function_is_the_top_level_one(self):
        # A trait implementation's method of the same name, which forwards to the block, is
        # indented and so not the block; a second top-level definition is an error.
        block = (
            "fn f(mut a: u64) -> [u64; 1] {\n"
            "    unsafe {\n"
            '        asm!("mov {a}, {a}", a = inout(reg) a, options(pure, nomem, nostack));\n'
            "    }\n"
            "    [a]\n"
            "}\n"
        )
        forwarding = (
            "impl Blocks for Backend {\n"
            "    fn f(a: u64) -> [u64; 1] {\n"
            "        f(a)\n"
            "    }\n"
            "}\n"
        )  # fmt: skip
        parsed = rust.parse_function(
            block + forwarding, "f", ["a"], 1, required_options={"pure", "nomem", "nostack"}
        )
        self.assertEqual(len(parsed.instructions), 1)
        with self.assertRaisesRegex(rust.GenerationError, "expected exactly one `fn f\\(`"):
            rust.parse_function(
                block + block, "f", ["a"], 1, required_options={"pure", "nomem", "nostack"}
            )


if __name__ == "__main__":
    unittest.main()
