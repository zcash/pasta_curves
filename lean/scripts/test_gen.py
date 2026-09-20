#!/usr/bin/env python3
# Copyright (c) 2026 the pasta-asm contributors.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for shared asm parsing and architecture-specific generators."""

from collections import Counter
import contextlib
import io
import re
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace

# Running this source-tree test should not leave lean/scripts/__pycache__ behind.
sys.dont_write_bytecode = True

import asm_source
import gen
import gen_aarch64 as gen_aarch64


class AArch64WriteDirectionTests(unittest.TestCase):
    def test_input_only_destination_is_rejected(self):
        emitter = gen_aarch64.Emitter([], {"b0": "in", "r0": "inout"})
        emitter.bind("r0", "0", "argument", reads=())
        with self.assertRaisesRegex(ValueError, "input-only register b0"):
            emitter.step("mov", ["b0", "r0"], "mov b0, r0")

    def test_undeclared_destination_is_rejected(self):
        emitter = gen_aarch64.Emitter([], {"r0": "inout"})
        with self.assertRaisesRegex(ValueError, "undeclared destination"):
            emitter.step("mov", ["bad", "r0"], "mov bad, r0")

    def test_output_and_inout_writes_are_allowed(self):
        for direction in ("out", "inout"):
            emitter = gen_aarch64.Emitter([], {"r0": direction})
            emitter.step("mov", ["r0", "xzr"], "mov r0, xzr")
            self.assertIn("r0", emitter.known)

    def test_zero_register_write_is_allowed(self):
        gen_aarch64.Emitter([]).step("mov", ["xzr", "xzr"], "mov xzr, xzr")


class AArch64OperandCountTests(unittest.TestCase):
    def test_shifted_add_is_rejected(self):
        emitter = gen_aarch64.Emitter([])
        with self.assertRaisesRegex(ValueError, "adds expects 3 operands"):
            emitter.step("adds", gen_aarch64.tokenize("r0, r0, b0, lsl #1"),
                         "adds r0, r0, b0, lsl #1")

    def test_missing_and_extra_operands_are_rejected_before_reads(self):
        arities = {"mov": 2, "mul": 3, "umulh": 3, "lsl": 3, "lsr": 3,
                   "adds": 3, "adcs": 3, "adc": 3, "subs": 3, "sbcs": 3, "csel": 4}
        for op, count in arities.items():
            for actual in (count - 1, count + 1):
                with self.subTest(op=op, actual=actual):
                    with self.assertRaisesRegex(ValueError, "expects .* operands"):
                        gen_aarch64.Emitter([]).step(op, ["r0"] * actual, op)


class SharedAArch64ParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = gen_aarch64.INLINE.read_text()

    def test_all_real_blocks_use_shared_parsed_function_model(self):
        for rust_name, _lean_name, _doc, arguments in gen_aarch64.INLINE_ROUTINES:
            with self.subTest(routine=rust_name):
                rust_arguments = arguments + (["inv"] if rust_name in ("mul", "square") else [])
                parsed = asm_source.parse_function(
                    self.source, rust_name, rust_arguments, 4,
                    allowed_options={"pure", "nomem", "nostack"},
                    required_options={"pure", "nomem", "nostack"},
                )
                instructions, declarations, locals_map, outputs, returned = gen_aarch64.parse_inline(
                    gen_aarch64.INLINE, rust_name, arguments
                )
                self.assertIsInstance(parsed, asm_source.ParsedFunction)
                self.assertEqual(len(instructions), len(parsed.instructions) + 1)
                self.assertEqual(declarations, [
                    (declaration.name, declaration.kind, declaration.value)
                    for declaration in parsed.declarations
                ])
                self.assertEqual(locals_map, parsed.locals)
                self.assertEqual(outputs, asm_source.output_bindings(parsed, rust_name))
                self.assertEqual(returned, asm_source.returned_registers(parsed, rust_name))

    def test_real_named_and_implicit_outputs_are_extracted(self):
        mul = asm_source.parse_function(
            self.source, "mul", ["lhs", "rhs", "modulus", "inv"], 4,
            allowed_options={"pure", "nomem", "nostack"},
            required_options={"pure", "nomem", "nostack"},
        )
        add = asm_source.parse_function(
            self.source, "add", ["lhs", "rhs", "modulus"], 4,
            allowed_options={"pure", "nomem", "nostack"},
            required_options={"pure", "nomem", "nostack"},
        )
        self.assertEqual(mul.returns, ("o0", "o1", "o2", "o3"))
        self.assertEqual(
            asm_source.output_bindings(mul, "mul"),
            {"o0": "r0", "o1": "r1", "o2": "r2", "o3": "r3"},
        )
        self.assertEqual(add.returns, ("r0", "r1", "r2", "r3"))
        self.assertEqual(
            asm_source.output_bindings(add, "add"),
            {"r0": "r0", "r1": "r1", "r2": "r2", "r3": "r3"},
        )


class SharedGeneratorTests(unittest.TestCase):
    def test_skeleton_hooks_are_backend_owned(self):
        aarch_routines = gen_aarch64.all_routines()
        self.assertTrue(all(routine.skeleton_backend is gen_aarch64.SKELETON_BACKEND
                            for routine in aarch_routines))

        aarch_add = next(routine for routine in aarch_routines if routine.name == "addMod")
        aarch_live = [entry for entry, keep in zip(
            aarch_add.emitter.entries, aarch_add.emitter.liveness(aarch_add.result_names)
        ) if keep]
        aarch_prepared = gen_aarch64.SKELETON_BACKEND.prepare(
            aarch_add.emitter, aarch_live
        )
        self.assertTrue(any(entry.get("group_fact", entry["fact"])[0] == "subs"
                            and len(entry.get("group", ())) == 3
                            for entry in aarch_prepared.entries))

    def test_shared_skeleton_has_no_architecture_dispatch_or_isa_fact_rules(self):
        source = gen.Path(gen.__file__).read_text()
        skeleton_source = source[source.index("def skeleton(routine):"):
                                 source.index("def _strip_annotations")]
        self.assertNotIn("routine.architecture", skeleton_source)
        for isa_fact in (
            "subs_carry", "subc_carry_cases", "sbb_borrow_le_one",
        ):
            with self.subTest(fact=isa_fact):
                self.assertNotIn(isa_fact, skeleton_source)
        self.assertIn('elif kind == "select":', skeleton_source)
        self.assertIn('elif kind == "mul":', skeleton_source)
        self.assertIn('elif kind == "lsr":', skeleton_source)

    def test_transcription_snapshots_match_committed_files(self):
        self.assertEqual(gen_aarch64.gen_program(), gen_aarch64.OUT_PROGRAM.read_text())

    def test_bare_skeleton_lookup_retains_aarch64_legacy(self):
        self.assertEqual(gen.find_routine("addMod").architecture, "AArch64")


class SkeletonCheckerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.routine = gen_aarch64.all_routines()[3]
        cls.skeleton = "\n".join(gen.skeleton(cls.routine)) + "\n"

    def check_text(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = gen.Path(directory) / "Spec.lean"
            path.write_text(text)
            diagnostics = io.StringIO()
            with contextlib.redirect_stderr(diagnostics):
                ok = gen.check_spec(path, [self.routine])
            return ok, diagnostics.getvalue()

    def test_handwritten_annotation_blocks_are_ignored(self):
        ok, diagnostics = self.check_text(
            "-- BEGIN handwritten proof\nanything may appear here\n-- END handwritten proof\n"
            + self.skeleton
        )
        self.assertTrue(ok, diagnostics)

    def test_missing_and_duplicate_skeletons_are_rejected(self):
        for text, message in (
            ("", "expected one skeleton of addMod, found 0"),
            (self.skeleton + self.skeleton, "expected one skeleton of addMod, found 2"),
        ):
            with self.subTest(message=message):
                ok, diagnostics = self.check_text(text)
                self.assertFalse(ok)
                self.assertIn(message, diagnostics)

    def test_malformed_annotations_are_rejected(self):
        malformed = {
            "END without BEGIN": "-- END loose\n" + self.skeleton,
            "unterminated BEGIN": self.skeleton + "-- BEGIN loose\n",
            "nested BEGIN": (
                "-- BEGIN outer\n-- BEGIN inner\n-- END outer\n" + self.skeleton
            ),
            "does not match BEGIN": (
                "-- BEGIN one\n-- END two\n" + self.skeleton
            ),
        }
        for message, text in malformed.items():
            with self.subTest(message=message):
                ok, diagnostics = self.check_text(text)
                self.assertFalse(ok)
                self.assertIn(message, diagnostics)

    def test_unreadable_spec_reports_controlled_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            diagnostics = io.StringIO()
            with contextlib.redirect_stderr(diagnostics):
                self.assertFalse(gen.check_spec(Path(directory) / "missing.lean", []))
            self.assertIn("cannot read proof file", diagnostics.getvalue())


if __name__ == "__main__":
    unittest.main()
