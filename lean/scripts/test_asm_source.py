#!/usr/bin/env python3
# Copyright (c) 2026 the pasta-asm contributors.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for fail-closed Rust surrounding-code parsing."""

from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest import mock

# Running this source-tree test should not leave lean/scripts/__pycache__ behind.
sys.dont_write_bytecode = True

import asm_source
import gen_aarch64
import gen_x86_64


class SurroundingCodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.aarch64_source = gen_aarch64.INLINE.read_text()
        cls.x86_64_source = gen_x86_64.SOURCE.read_text()

    @staticmethod
    def mutate_function(source, name, old, new):
        masked = asm_source.masked_noncode(source)
        match = re.search(rf"(?m)^.*\bfn\s+{re.escape(name)}\s*\(", masked)
        if match is None:
            raise AssertionError(f"function {name} not found")
        signature_open = masked.find("(", match.start())
        signature_close = asm_source.matching_delimiter(
            source, signature_open, "(", ")"
        )
        body_open = masked.find("{", signature_close)
        body_close = asm_source.matching_delimiter(source, body_open, "{", "}")
        function_source = source[match.start():body_close + 1]
        if function_source.count(old) != 1:
            raise AssertionError(f"expected one {old!r} in {name}")
        return (
            source[:match.start()]
            + function_source.replace(old, new)
            + source[body_close + 1:]
        )

    @staticmethod
    def generate_aarch64(source):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aarch64.rs"
            path.write_text(source)
            with mock.patch.object(gen_aarch64, "INLINE", path):
                return gen_aarch64.gen_program()

    def test_current_sources_generate_committed_output(self):
        self.assertEqual(
            self.generate_aarch64(self.aarch64_source),
            gen_aarch64.OUT_PROGRAM.read_text(),
        )
        self.assertEqual(
            gen_x86_64.gen_program(self.x86_64_source),
            gen_x86_64.OUTPUT.read_text(),
        )

    def test_inv_shadowing_is_rejected_by_both_backends(self):
        for architecture, source, generate in (
            ("aarch64", self.aarch64_source, self.generate_aarch64),
            ("x86_64", self.x86_64_source, gen_x86_64.gen_program),
        ):
            with self.subTest(architecture=architecture):
                mutated = self.mutate_function(
                    source,
                    "mul",
                    "    let (o0, o1, o2, o3): (u64, u64, u64, u64);",
                    "    let inv = 0;\n"
                    "    let (o0, o1, o2, o3): (u64, u64, u64, u64);",
                )
                with self.assertRaisesRegex(
                    asm_source.GenerationError,
                    "local inv shadows a function argument",
                ):
                    generate(mutated)

    def test_postasm_output_mutation_is_rejected_by_both_backends(self):
        for architecture, source, generate in (
            ("aarch64", self.aarch64_source, self.generate_aarch64),
            ("x86_64", self.x86_64_source, gen_x86_64.gen_program),
        ):
            with self.subTest(architecture=architecture):
                mutated = self.mutate_function(
                    source,
                    "add",
                    "    }\n    [r0, r1, r2, r3]",
                    "    }\n    r0 = 0;\n    [r0, r1, r2, r3]",
                )
                with self.assertRaisesRegex(
                    asm_source.GenerationError,
                    "unsupported code after asm!",
                ):
                    generate(mutated)

    def test_comments_are_allowed_between_surrounding_grammar_tokens(self):
        for architecture, source, generate, expected in (
            (
                "aarch64",
                self.aarch64_source,
                self.generate_aarch64,
                gen_aarch64.OUT_PROGRAM.read_text(),
            ),
            (
                "x86_64",
                self.x86_64_source,
                gen_x86_64.gen_program,
                gen_x86_64.OUTPUT.read_text(),
            ),
        ):
            with self.subTest(architecture=architecture):
                commented = self.mutate_function(
                    source,
                    "add",
                    "    let [mut r0, mut r1, mut r2, mut r3] = *lhs;",
                    "    let /* outputs */ [mut r0, mut r1, mut r2, mut r3] "
                    "= * /* input */ lhs;",
                )
                self.assertEqual(generate(commented), expected)


if __name__ == "__main__":
    unittest.main()
