#!/usr/bin/env python3
# Copyright (c) 2026 the pasta-asm contributors.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for shared asm parsing and architecture-specific generators."""

import contextlib
import io
import re
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

# Running this source-tree test should not leave lean/scripts/__pycache__ behind.
sys.dont_write_bytecode = True

import asm_source
import gen
import gen_aarch64
import gen_x86_64


class DeclarationTests(unittest.TestCase):
    def test_inout_discard_and_named_outputs_are_parsed(self):
        discarded = gen_x86_64.parse_declaration("z3 = inout(reg) product[3] => _,")
        named = gen_x86_64.parse_declaration("z0 = inout(reg) product[0] => o1,")
        self.assertEqual(
            discarded,
            gen_x86_64.Declaration("z3", "inout", "product[3]", "_"),
        )
        self.assertEqual(
            named,
            gen_x86_64.Declaration("z0", "inout", "product[0]", "o1"),
        )

    def test_unsupported_constraint_is_rejected(self):
        with self.assertRaisesRegex(gen_x86_64.GenerationError, "constraint"):
            gen_x86_64.parse_declaration("x = in(xmm_reg) value,")

    def test_unsupported_fixed_register_is_rejected(self):
        with self.assertRaisesRegex(gen_x86_64.GenerationError, "fixed-register"):
            gen_x86_64.parse_declaration('in("rax") value,')

    def test_unsupported_const_is_rejected(self):
        with self.assertRaisesRegex(gen_x86_64.GenerationError, "const operand"):
            gen_x86_64.parse_declaration("p3 = const OTHER_CONSTANT,")

    def test_named_rdx_and_internal_names_are_reserved(self):
        for name in ("rdx", "cf", "ofl", "s", "d", "m", "n"):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(gen_x86_64.GenerationError, "reserved operand name"),
            ):
                gen_x86_64.parse_declaration(f"{name} = out(reg) _,")

    def test_high_limb_value_is_validated(self):
        source = gen_x86_64.SOURCE.read_text().replace(
            "const PASTA_HIGH_LIMB: u64 = 1 << 62;",
            "const PASTA_HIGH_LIMB: u64 = 1 << 61;",
        )
        with self.assertRaisesRegex(gen_x86_64.GenerationError, "1 << 62"):
            gen_x86_64.validate_high_limb(source)


class EmitterTests(unittest.TestCase):
    @staticmethod
    def emitter(*registers):
        directions = {register: "inout" for register in registers}
        emitter = gen_x86_64.Emitter(directions, {})
        for register in registers:
            if register != "rdx":
                emitter.bind_argument(register, "0", "test input")
        return emitter

    def test_unsupported_instruction_is_rejected(self):
        emitter = self.emitter("a")
        with self.assertRaisesRegex(gen_x86_64.GenerationError, "unsupported instruction or"):
            emitter.emit_instruction("or {a}, 1")

    def test_input_only_register_cannot_be_written(self):
        emitter = gen_x86_64.Emitter({"a": "in", "b": "in"}, {})
        emitter.bind_argument("a", "1", "test input")
        emitter.bind_argument("b", "2", "test input")
        with self.assertRaisesRegex(gen_x86_64.GenerationError, "input-only register a"):
            emitter.emit_instruction("add {a}, {b}")

    def test_input_only_register_cannot_be_xor_zeroed(self):
        emitter = gen_x86_64.Emitter({"z": "in"}, {})
        emitter.bind_argument("z", "1", "test input")
        with self.assertRaisesRegex(gen_x86_64.GenerationError, "input-only register z"):
            emitter.emit_instruction("xor {z:e}, {z:e}")

    def test_register_read_before_write_is_rejected(self):
        emitter = gen_x86_64.Emitter({"a": "inout", "b": "in"}, {})
        emitter.bind_argument("a", "0", "test input")
        with self.assertRaisesRegex(gen_x86_64.GenerationError, "b read before"):
            emitter.emit_instruction("add {a}, {b}")

    def test_memory_write_is_rejected(self):
        emitter = gen_x86_64.Emitter({"a": "inout", "p": "in"}, {"p": "value"})
        emitter.bind_argument("a", "0", "test input")
        with self.assertRaisesRegex(gen_x86_64.GenerationError, "memory write"):
            emitter.emit_instruction("mov qword ptr [{p}], {a}")

    def test_non_limb_memory_offset_is_rejected(self):
        emitter = gen_x86_64.Emitter({"a": "inout", "p": "in"}, {"p": "value"})
        emitter.bind_argument("a", "0", "test input")
        with self.assertRaisesRegex(gen_x86_64.GenerationError, "offset 32"):
            emitter.emit_instruction("mov {a}, qword ptr [{p} + 32]")

    def test_undeclared_address_base_is_rejected(self):
        emitter = gen_x86_64.Emitter({"a": "inout"}, {})
        emitter.bind_argument("a", "0", "test input")
        with self.assertRaisesRegex(gen_x86_64.GenerationError, "undeclared register p"):
            emitter.emit_instruction("mov {a}, qword ptr [{p}]")

    def test_register_is_not_an_address_base_without_pointer_binding(self):
        emitter = self.emitter("a", "p")
        with self.assertRaisesRegex(gen_x86_64.GenerationError, "not a read-only pointer"):
            emitter.emit_instruction("mov {a}, qword ptr [{p}]")

    def test_adc_rejects_invalid_cf(self):
        emitter = self.emitter("a", "b")
        with self.assertRaisesRegex(gen_x86_64.GenerationError, "CF read while invalid"):
            emitter.emit_instruction("adc {a}, {b}")

    def test_cmovnc_rejects_invalid_cf(self):
        emitter = self.emitter("a", "b")
        with self.assertRaisesRegex(gen_x86_64.GenerationError, "CF read while invalid"):
            emitter.emit_instruction("cmovnc {a}, {b}")

    def test_add_invalidates_of(self):
        emitter = self.emitter("a", "b", "z")
        emitter.emit_instruction("xor {z:e}, {z:e}")
        emitter.emit_instruction("add {a}, {b}")
        with self.assertRaisesRegex(gen_x86_64.GenerationError, "OF read while invalid"):
            emitter.emit_instruction("adox {a}, {b}")

    def test_neg_invalidates_of(self):
        emitter = self.emitter("a", "b", "z")
        emitter.emit_instruction("xor {z:e}, {z:e}")
        emitter.emit_instruction("neg {a}")
        with self.assertRaisesRegex(gen_x86_64.GenerationError, "OF read while invalid"):
            emitter.emit_instruction("adox {a}, {b}")

    def test_imul_invalidates_flags(self):
        emitter = self.emitter("a", "b", "z")
        emitter.emit_instruction("xor {z:e}, {z:e}")
        emitter.emit_instruction("imul {a}, {b}")
        with self.assertRaisesRegex(gen_x86_64.GenerationError, "CF read while invalid"):
            emitter.emit_instruction("adc {a}, {b}")

    def test_masked_or_zero_shift_counts_are_rejected(self):
        for amount in (0, 64, 65, 256):
            with self.subTest(amount=amount):
                emitter = self.emitter("a")
                with self.assertRaisesRegex(gen_x86_64.GenerationError, "shift counts"):
                    emitter.emit_instruction(f"shl {{a}}, {amount}")

    def test_shift_invalidates_flags(self):
        emitter = self.emitter("a", "b", "z")
        emitter.emit_instruction("xor {z:e}, {z:e}")
        emitter.emit_instruction("shl {a}, 62")
        with self.assertRaisesRegex(gen_x86_64.GenerationError, "CF read while invalid"):
            emitter.emit_instruction("adcx {a}, {b}")

    def test_xor_self_allows_uninitialized_output_and_clears_both_flags(self):
        emitter = gen_x86_64.Emitter({"a": "inout", "b": "in", "z": "out"}, {})
        emitter.bind_argument("a", "1", "test input")
        emitter.bind_argument("b", "2", "test input")
        emitter.emit_instruction("xor {z:e}, {z:e}")
        emitter.emit_instruction("adcx {a}, {b}")
        emitter.emit_instruction("adox {a}, {b}")

    def test_adcx_and_adox_preserve_the_other_flag(self):
        emitter = self.emitter("a", "b", "z")
        emitter.emit_instruction("xor {z:e}, {z:e}")
        emitter.emit_instruction("adcx {a}, {b}")
        emitter.emit_instruction("adox {a}, {b}")
        # OF from ADOX and CF from ADCX are both still valid.
        emitter.emit_instruction("adox {a}, {b}")
        emitter.emit_instruction("adcx {a}, {b}")

    def test_mov_mulx_and_cmov_preserve_flags(self):
        emitter = self.emitter("a", "b", "z", "rdx", "hi", "lo")
        emitter.bind_argument("rdx", "3", "fixed register test input")
        emitter.emit_instruction("xor {z:e}, {z:e}")
        emitter.emit_instruction("mov {a}, {b}")
        emitter.emit_instruction("mulx {hi}, {lo}, {a}")
        emitter.emit_instruction("cmovnc {a}, {b}")
        emitter.emit_instruction("adox {a}, {b}")
        emitter.emit_instruction("adcx {a}, {b}")


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
            emitter.step(
                "adds", gen_aarch64.tokenize("r0, r0, b0, lsl #1"), "adds r0, r0, b0, lsl #1"
            )

    def test_missing_and_extra_operands_are_rejected_before_reads(self):
        arities = {
            "mov": 2,
            "mul": 3,
            "umulh": 3,
            "lsl": 3,
            "lsr": 3,
            "adds": 3,
            "adcs": 3,
            "adc": 3,
            "subs": 3,
            "sbcs": 3,
            "csel": 4,
        }
        for op, count in arities.items():
            for actual in (count - 1, count + 1):
                with (
                    self.subTest(op=op, actual=actual),
                    self.assertRaisesRegex(ValueError, "expects .* operands"),
                ):
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
                    self.source,
                    rust_name,
                    rust_arguments,
                    4,
                    allowed_options={"pure", "nomem", "nostack"},
                    required_options={"pure", "nomem", "nostack"},
                )
                instructions, declarations, locals_map, outputs, returned = (
                    gen_aarch64.parse_inline(gen_aarch64.INLINE, rust_name, arguments)
                )
                self.assertIsInstance(parsed, asm_source.ParsedFunction)
                self.assertEqual(len(instructions), len(parsed.instructions) + 1)
                self.assertEqual(
                    declarations,
                    [
                        (declaration.name, declaration.kind, declaration.value)
                        for declaration in parsed.declarations
                    ],
                )
                self.assertEqual(locals_map, parsed.locals)
                self.assertEqual(outputs, asm_source.output_bindings(parsed, rust_name))
                self.assertEqual(returned, asm_source.returned_registers(parsed, rust_name))

    def test_real_named_and_implicit_outputs_are_extracted(self):
        mul = asm_source.parse_function(
            self.source,
            "mul",
            ["lhs", "rhs", "modulus", "inv"],
            4,
            allowed_options={"pure", "nomem", "nostack"},
            required_options={"pure", "nomem", "nostack"},
        )
        add = asm_source.parse_function(
            self.source,
            "add",
            ["lhs", "rhs", "modulus"],
            4,
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


class X86RealSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = gen_x86_64.SOURCE.read_text()

    def mutate(self, old, new):
        self.assertEqual(self.source.count(old), 1, old)
        return self.source.replace(old, new)

    def assert_mutation_rejected(self, old, new, message):
        source = self.mutate(old, new)
        with self.assertRaisesRegex(gen_x86_64.GenerationError, message):
            gen_x86_64.gen_program(source)

    def mutate_function(self, name, old, new):
        marker = f"pub(super) fn {name}("
        start = self.source.index(marker)
        next_function = self.source.find("pub(super) fn ", start + len(marker))
        end = len(self.source) if next_function < 0 else next_function
        function = self.source[start:end]
        self.assertEqual(function.count(old), 1, old)
        return self.source[:start] + function.replace(old, new) + self.source[end:]

    def assert_function_mutation_rejected(self, name, old, new, message):
        source = self.mutate_function(name, old, new)
        with self.assertRaisesRegex(gen_x86_64.GenerationError, message):
            gen_x86_64.gen_program(source)

    def test_all_six_blocks_generate(self):
        generated = gen_x86_64.gen_program(self.source)
        for config in gen_x86_64.ROUTINES:
            self.assertIn(f"def {config.lean_name} ", generated)

    def test_source_output_tuple_order_is_used(self):
        mul = gen_x86_64.transcribe(self.source, gen_x86_64.ROUTINES[2])
        square_hi = gen_x86_64.transcribe(self.source, gen_x86_64.ROUTINES[4])
        self.assertTrue(mul.rstrip().endswith("⟨ee, ae, be, ce⟩"))
        self.assertTrue(square_hi.rstrip().endswith("⟨a, z0, z1, z2⟩"))

    def test_each_unfactored_source_instruction_has_one_generated_comment(self):
        for config in gen_x86_64.ROUTINES:
            if config.lean_name == "mulMont":
                continue
            with self.subTest(routine=config.lean_name):
                parsed = gen_x86_64.parse_function(self.source, config)
                generated = gen_x86_64.transcribe(self.source, config)
                comments = Counter(re.findall(r"-- (.+)$", generated, re.MULTILINE))
                self.assertEqual(
                    Counter(parsed.instructions),
                    Counter(
                        {instruction: comments[instruction] for instruction in parsed.instructions}
                    ),
                )

    def test_factored_mul_rounds_match_under_rotation(self):
        routine = gen_x86_64.emit_routine(self.source, gen_x86_64.ROUTINES[2])
        rounds = [
            gen_x86_64._normalized_mul_round_entries(
                routine.emitter, pc_range, rotation, f"rhs.l{index}"
            )
            for index, (pc_range, rotation) in enumerate(
                zip(gen_x86_64.MUL_ROUND_RANGES, gen_x86_64.MUL_ROUND_ROTATIONS), start=1
            )
        ]
        self.assertEqual(
            gen_x86_64._round_fingerprint(rounds[0]),
            gen_x86_64._round_fingerprint(rounds[1]),
        )
        self.assertEqual({entry["pc"] for entry in rounds[0]}, set(range(36)))

    def test_factored_mul_round_difference_is_rejected(self):
        source = self.mutate_function(
            "mul",
            '            "add {ce}, {s1}",\n            "adc {de}, {s2}",\n            "adc {ee}, 0",\n',
            '            "add {ce}, {s1}",\n            "adc {de}, {s2}",\n            "adc {ee}, {s1}",\n',
        )
        with self.assertRaisesRegex(
            gen_x86_64.GenerationError,
            "flattened full rounds 1 and 2 differ after accumulator rotation",
        ):
            gen_x86_64.gen_program(source)

    def test_factored_round_flattens_back_to_each_source_round(self):
        routine = gen_x86_64.emit_routine(self.source, gen_x86_64.ROUTINES[2])
        source_load, body = gen_x86_64._factored_round_body(routine.emitter)
        registers = ["be", "ce", "de", "ee", "ae", "rdx"]
        for round_number, (first, last) in enumerate(gen_x86_64.MUL_ROUND_RANGES, start=1):
            source = [
                entry
                for entry in routine.emitter.entries
                if entry["pc"] is not None and first <= entry["pc"] <= last
            ]
            flattened = gen_x86_64._flattened_round_call(
                source_load, body, registers, f"rhs.l{round_number}", first
            )
            self.assertEqual(
                gen_x86_64._round_fingerprint(flattened), gen_x86_64._round_fingerprint(source)
            )
            self.assertEqual({entry["pc"] for entry in flattened}, set(range(first, last + 1)))
            registers = registers[1:5] + registers[:1] + ["rdx"]

    def test_factored_round_call_with_misordered_registers_is_rejected(self):
        routine = gen_x86_64.emit_routine(self.source, gen_x86_64.ROUTINES[2])
        source_load, body = gen_x86_64._factored_round_body(routine.emitter)
        with self.assertRaisesRegex(
            gen_x86_64.GenerationError, "does not flatten to the source's round 1"
        ):
            gen_x86_64._check_round_call(
                routine.emitter,
                source_load,
                body,
                ["ce", "be", "de", "ee", "ae", "rdx"],
                "rhs.l1",
                gen_x86_64.MUL_ROUND_RANGES[0],
                1,
            )

    def test_factored_mul_comments_cover_one_validated_full_round(self):
        generated = gen_x86_64.transcribe(self.source, gen_x86_64.ROUTINES[2])
        comments = Counter(re.findall(r"-- (.+)$", generated, re.MULTILINE))
        parsed = gen_x86_64.parse_function(self.source, gen_x86_64.ROUTINES[2])
        omitted = []
        second_first, second_last = gen_x86_64.MUL_ROUND_RANGES[1]
        for pc, instruction in enumerate(parsed.instructions):
            if not second_first <= pc <= second_last:
                omitted.append(instruction)
        self.assertEqual(
            Counter(omitted),
            Counter({instruction: comments[instruction] for instruction in omitted}),
        )

    def test_declared_pointer_blocks_are_readonly(self):
        for config in (gen_x86_64.ROUTINES[2], gen_x86_64.ROUTINES[4]):
            with self.subTest(routine=config.lean_name):
                parsed = gen_x86_64.parse_function(self.source, config)
                self.assertIn("readonly", parsed.options)
                self.assertNotIn("nomem", parsed.options)

    def test_raw_template_is_rejected(self):
        self.assert_mutation_rejected(
            '            "add {r0}, {b0}",',
            '            r"add {r0}, {b0}",',
            "raw asm template",
        )

    def test_escaped_template_is_rejected(self):
        self.assert_mutation_rejected(
            '            "add {r0}, {b0}",',
            '            "add {r0}, {b0}\\n",',
            "escaped asm template",
        )

    def test_instruction_string_in_block_comment_is_ignored(self):
        source = self.mutate(
            '            "add {r0}, {b0}",',
            '            /* "sub {r0}, {b0}", */\n            "add {r0}, {b0}",',
        )
        self.assertEqual(gen_x86_64.gen_program(source), gen_x86_64.gen_program(self.source))

    def test_junk_between_templates_is_rejected(self):
        self.assert_mutation_rejected(
            '            "add {r0}, {b0}",',
            '            "add {r0}, {b0}",\n            unsupported_token,',
            "unsupported operand binding",
        )

    def test_junk_after_options_is_rejected(self):
        old = """            p3 = inout(reg) modulus[3] => _,
            z = out(reg) _,
            options(pure, nomem, nostack),
        );"""
        new = old.replace(
            "            options(pure, nomem, nostack),",
            "            options(pure, nomem, nostack),\n            junk,",
        )
        self.assert_function_mutation_rejected("add", old, new, "operand after options")

    def test_second_asm_block_is_rejected(self):
        old = """            z = out(reg) _,
            options(pure, nomem, nostack),
        );
    }
    [r0, r1, r2, r3]
}"""
        new = old.replace(
            "        );\n    }",
            '        );\n        asm!("mov rax, rax");\n    }',
        )
        self.assert_function_mutation_rejected("add", old, new, "exactly one asm! block")

    def test_input_only_operand_write_is_rejected_in_real_source(self):
        self.assert_mutation_rejected(
            '            "add {r0}, {b0}",',
            '            "add {b0}, {r0}",',
            "input-only register b0",
        )

    def test_input_only_operand_xor_is_rejected_in_real_source(self):
        self.assert_mutation_rejected(
            '            "add {r0}, {b0}",',
            '            "xor {b0:e}, {b0:e}",',
            "input-only register b0",
        )

    def test_literal_rdx_without_fixed_operand_is_rejected(self):
        old = """            t1 = out(reg) _,
            t2 = out(reg) _,
            out("rdx") _,
            options(pure, nomem, nostack),"""
        new = old.replace('            out("rdx") _,\n', "")
        self.assert_mutation_rejected(old, new, "literal rdx requires exactly one fixed")

    def test_named_rdx_cannot_spoof_fixed_operand(self):
        old = """            t1 = out(reg) _,
            t2 = out(reg) _,
            out("rdx") _,
            options(pure, nomem, nostack),"""
        new = old.replace(
            '            out("rdx") _,',
            "            rdx = out(reg) _,",
        )
        self.assert_mutation_rejected(old, new, "reserved operand name rdx")

    def test_bound_local_use_before_asm_is_rejected(self):
        self.assert_function_mutation_rejected(
            "add",
            "    let [mut r0, mut r1, mut r2, mut r3] = *lhs;\n",
            "    let [mut r0, mut r1, mut r2, mut r3] = *lhs;\n    r0 = 0;\n",
            "unsupported use of bound local r0",
        )


class SharedGeneratorTests(unittest.TestCase):
    def test_skeleton_hooks_are_backend_owned(self):
        aarch_routines = gen_aarch64.all_routines()
        x86_routines = gen_x86_64.all_routines()
        self.assertTrue(
            all(
                routine.skeleton_backend is gen_aarch64.SKELETON_BACKEND
                for routine in aarch_routines
            )
        )
        self.assertTrue(
            all(routine.skeleton_backend is gen_x86_64.SKELETON_BACKEND for routine in x86_routines)
        )
        self.assertIsNot(gen_aarch64.SKELETON_BACKEND, gen_x86_64.SKELETON_BACKEND)

        aarch_add = next(routine for routine in aarch_routines if routine.name == "addMod")
        aarch_live = [
            entry
            for entry, keep in zip(
                aarch_add.emitter.entries, aarch_add.emitter.liveness(aarch_add.result_names)
            )
            if keep
        ]
        aarch_prepared = gen_aarch64.SKELETON_BACKEND.prepare(aarch_add.emitter, aarch_live)
        self.assertTrue(
            any(
                entry.get("group_fact", entry["fact"])[0] == "subs"
                and len(entry.get("group", ())) == 3
                for entry in aarch_prepared.entries
            )
        )

        x86_square = next(routine for routine in x86_routines if routine.name == "squareLo")
        x86_prepared = gen_x86_64.SKELETON_BACKEND.prepare(
            x86_square.emitter, x86_square.emitter.entries
        )
        self.assertTrue(
            any(
                entry.get("group_fact", entry["fact"])[0] == "x86_mulx"
                for entry in x86_prepared.entries
            )
        )

    def test_skeleton_facts_come_from_the_routine_backend(self):
        # The shared traversal knows the common facts; an ISA-specific one reaches it only
        # through the routine's backend hook, and one that no hook handles is an error.
        class StubBackend(gen.SkeletonBackend):
            def fact(self, kind, ops, context):
                if kind != "stub":
                    return False
                (operand,) = ops
                context.eq(context.name, f"stub {operand}")
                context.lines.append(f"  have b_{context.name} : {context.name} < 2^64 := by sorry")
                context.bnd[context.name] = f"b_{context.name}"
                return True

        def routine(backend):
            emitter = gen.Emitter()
            emitter.bind("x", "lhs.l0", "argument", reads=(), load=True, fact=("load", "lhs", "l0"))
            emitter.bind("y", "stub x", "stub x", reads={"x"}, fact=("stub", "x"))

            class StubRoutine(gen.Routine):
                architecture = "Stub"
                skeleton_backend = backend

            return StubRoutine(
                "doc", "def stub", emitter.render(["y"]), "  y", "stub", emitter, ["y"]
            )

        skeleton = "\n".join(gen.skeleton(routine(StubBackend())))
        self.assertIn("have e_x : x = lhs.l0 := rfl", skeleton)
        self.assertIn("have e_y : y = stub x := rfl", skeleton)
        self.assertIn("have b_y : y < 2^64 := by sorry", skeleton)
        with self.assertRaisesRegex(ValueError, "unsupported skeleton fact stub"):
            gen.skeleton(routine(gen.SkeletonBackend()))

    def test_backward_liveness_and_backend_retention_share_the_same_ir(self):
        emitter = gen.Emitter()
        emitter.bind("input", "lhs.l0", reads=set(), fact=("param", "lhs", "l0"))
        emitter.bind("dead", "input + 1", reads={"input"}, fact=("call", "{} + 1", ["input"]))
        emitter.bind("result", "input", reads={"input"}, fact=("call", "{}", ["input"]))
        self.assertEqual(
            [
                entry["name"]
                for entry, keep in zip(emitter.entries, emitter.liveness(["result"]))
                if keep
            ],
            ["input", "result"],
        )

        retained = gen_x86_64.Emitter({}, {})
        retained.entries = list(emitter.entries)
        self.assertEqual(
            [
                entry["name"]
                for entry, keep in zip(retained.entries, retained.liveness(["result"]))
                if keep
            ],
            ["input", "result"],
        )
        self.assertIn(
            ("  let dead := input + 1", None),
            retained.render(["result"]),
        )

    def test_x86_skeleton_extracts_every_retained_instruction_let(self):
        routines = {routine.name: routine for routine in gen_x86_64.all_routines()}
        # Every retained binding is extracted exactly once, equal values included (the three
        # zeros of an `xor r, r`, a repeated `mov` from `rdx`), since merging is off.
        for name in ("squareLo", "mulMontRound", "fromMont"):
            with self.subTest(routine=name):
                routine = routines[name]
                prepared = gen_x86_64.SKELETON_BACKEND.prepare(
                    routine.emitter, routine.emitter.entries
                )
                extracted, current = [], None
                for line in gen.skeleton(routine):
                    if line.startswith("  extract_lets -merge +onlyGivenNames "):
                        current = line[len("  extract_lets -merge +onlyGivenNames ") :]
                    elif current is not None:
                        current += " " + line.strip()
                    if current is not None and current.endswith(" at hr"):
                        extracted += current[: -len(" at hr")].split()
                        current = None
                self.assertEqual(sorted(extracted), sorted(prepared.names))
                self.assertEqual(len(extracted), len(set(extracted)))
        square_lo = "\n".join(gen.skeleton(routines["squareLo"]))
        self.assertIn("extract_lets -merge +onlyGivenNames s_2 z4_1 cf_5 at hr", square_lo)
        self.assertNotIn("obtain ⟨cf_5, b_cf_5, l_z4_1⟩", square_lo)

        mul_round_routine = routines["mulMontRound"]
        mul_round = "\n".join(gen.skeleton(mul_round_routine))
        # The factored round is rendered like every other block: an instruction's pair wrapper
        # and its two projections, extracted together, and read under their own names.
        self.assertIn("extract_lets -merge +onlyGivenNames m s2 s1_1 at hr", mul_round)
        self.assertIn("extract_lets -merge +onlyGivenNames s r0_1 cf_1 at hr", mul_round)
        self.assertIn("have e_r0_1 : r0_1 = (addc r0 s1_1 cf).1 := rfl", mul_round)
        self.assertIn("have e_cf_1 : cf_1 = (addc r0 s1_1 cf).2 := rfl", mul_round)
        # The round's and the calling block's local definitions stay transparent.
        self.assertNotIn("clear_value", mul_round)
        self.assertNotIn("clear_value", "\n".join(gen.skeleton(routines["mulMont"])))
        helper_text = "\n".join(code for code, _ in mul_round_routine.lines)
        self.assertIn("let m := mulx b lhs.l0", helper_text)
        self.assertIn("let s := addc r0 s1 cf", helper_text)
        self.assertIn("let r0 := s.1", helper_text)
        self.assertIn("let cf := s.2", helper_text)
        # The omitted entry load aliases RDX to b only until the source writes RDX again.
        # Reduction products must use the rebound Montgomery quotient, never b.
        expressions = [entry["expr"] for entry in mul_round_routine.emitter.entries]
        quotient_index = expressions.index("mulLo rdx inv")
        self.assertTrue(any("mulx rdx modulus.l1" in expr for expr in expressions[quotient_index:]))
        self.assertFalse(any("mulx b modulus.l1" in expr for expr in expressions[quotient_index:]))

        from_mont = "\n".join(gen.skeleton(routines["fromMont"]))
        self.assertIn("extract_lets -merge +onlyGivenNames n z0_1 cf at hr", from_mont)
        self.assertIn("have e_z0_1 : z0_1 = (neg z0).1 := rfl", from_mont)
        self.assertIn("have e_cf : cf = (neg z0).2 := rfl", from_mont)
        self.assertIn("clear_value", from_mont)

    def test_transcription_snapshots_match_committed_files(self):
        self.assertEqual(gen_aarch64.gen_program(), gen_aarch64.OUT_PROGRAM.read_text())
        self.assertEqual(gen_x86_64.gen_program(), gen_x86_64.OUTPUT.read_text())

    def test_all_x86_routines_and_factored_round_generate_shared_skeletons(self):
        routines = gen_x86_64.all_routines()
        self.assertEqual(
            [routine.name for routine in routines],
            ["addMod", "subMod", "mulMontRound", "mulMont", "squareLo", "squareHi", "fromMont"],
        )
        for routine in routines:
            with self.subTest(routine=routine.name):
                generated = gen.skeleton(routine)
                self.assertEqual(
                    generated[0],
                    f"  -- generated skeleton for `{routine.name}`: do not edit between the annotations",
                )
                self.assertEqual(generated[-1], "  subst hr")
                self.assertIs(
                    gen.find_routine(f"X86_64:{routine.name}").emitter.__class__,
                    routine.emitter.__class__,
                )

    def test_round_argument_fields_are_routine_local(self):
        aarch_round = next(
            routine for routine in gen_aarch64.all_routines() if routine.name == "mulMontRound"
        )
        x86_round = next(
            routine for routine in gen_x86_64.all_routines() if routine.name == "mulMontRound"
        )
        self.assertEqual(
            aarch_round.arg_fields["acc"],
            [
                "r0",
                "r1",
                "r2",
                "r3",
                "r4",
                "q",
                "t1",
                "t3",
            ],
        )
        self.assertEqual(
            x86_round.arg_fields["acc"],
            [
                "r0",
                "r1",
                "r2",
                "r3",
                "r4",
                "q",
            ],
        )
        self.assertEqual(gen.proj("acc", "q", aarch_round.arg_fields), "2.2.2.2.2.1")
        self.assertEqual(gen.proj("acc", "q", x86_round.arg_fields), "2.2.2.2.2")
        self.assertEqual(gen.proj("acc", "r4", aarch_round.arg_fields), "2.2.2.2.1")
        self.assertEqual(gen.proj("acc", "r4", x86_round.arg_fields), "2.2.2.2.1")

    def test_bare_skeleton_lookup_retains_aarch64_legacy(self):
        self.assertEqual(gen.find_routine("addMod").architecture, "AArch64")
        self.assertEqual(gen.find_routine("X86_64:addMod").architecture, "X86_64")


class SkeletonCheckerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.routine = gen_x86_64.all_routines()[0]
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
            "nested BEGIN": ("-- BEGIN outer\n-- BEGIN inner\n-- END outer\n" + self.skeleton),
            "does not match BEGIN": ("-- BEGIN one\n-- END two\n" + self.skeleton),
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

    def test_manifest_selects_existing_files_and_rejects_mismatched(self):
        add_path = gen.ROOT / "lean/PastaAsm/X86_64/Spec/Add.lean"
        available = gen.architecture_routines()
        for name, (arch, expected_names) in gen.SPEC_MANIFEST.items():
            with self.subTest(path=name):
                manifest_path = gen.ROOT / name
                if Path(manifest_path).exists:
                    path, routines = gen.parse_spec_manifest(str(manifest_path))
                    self.assertEqual(path, manifest_path)
                    expected = (
                        expected_names
                        if expected_names is not None
                        else tuple(routine.name for routine in available[arch])
                    )
                    self.assertEqual([routine.name for routine in routines], list(expected))
                else:
                    with self.assertRaisesRegex(ValueError, "pending migration"):
                        gen.parse_spec_manifest(str(manifest_path))
        with self.assertRaisesRegex(ValueError, "belongs to X86_64, not AArch64"):
            gen.parse_spec_manifest(f"AArch64:{add_path}")


if __name__ == "__main__":
    unittest.main()
