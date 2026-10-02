#!/usr/bin/env python3
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

import gen
from asm2lean import aarch64, ir, lean, rust, x86_64
from asm2lean.ir import Call, Load, Mov, Node
from asm2lean.skeleton import projection, skeleton, ssa_names
from asm2lean.specs import check_spec
from pasta import aarch64_blocks, data, x86_64_blocks
from pasta.conventions import CONVENTIONS
from pasta.paths import ROOT

GenerationError = rust.GenerationError
X86 = x86_64_blocks.TARGET
MUL_ROUNDS = x86_64_blocks.MUL_ROUNDS


def parse_declaration(text):
    """An operand declaration, under the x86-64 target's restrictions."""
    return rust.parse_declaration(
        text,
        reserved_names=set(X86.reserved_names),
        fixed_registers={x86_64.RDX},
        const_operands=set(X86.consts),
    )


def x86_block(name):
    return next(block for block in x86_64_blocks.BLOCKS if block.lean_name == name)


def transcribed(source, block):
    """The x86-64 transcription of one block, with its round when it has one."""
    return "\n".join(
        lean.definition(p, lean.comment_column([p])) for p in x86_64.transcribe(source, block, X86)
    )


def x86_bind(lifter, register, value, comment="test input"):
    lifter.bind_argument(Mov(comment=comment, dest=register, a=value))


def aarch64_step(lifter, op, operands, text):
    lifter.lift(aarch64.Instruction(op, tuple(operands), text))


def bound_registers(lifter, registers):
    for register in registers:
        lifter.bind(Mov(comment="argument", dest=register, a="0"))


def word_step_groups(skeleton):
    """The names that each `word_step` of a skeleton extracts, in order: a step's items are
    separated by commas outside brackets, and continue over its indented continuation lines."""

    def names(step):
        found, depth, item = [], 0, ""
        for char in step + ",":
            if char in "([⟨{":
                depth += 1
            elif char in ")]⟩}":
                depth -= 1
            if char == "," and depth == 0:
                found.append(item.split()[0])
                item = ""
            else:
                item += char
        return found

    groups, step = [], None
    for line in list(skeleton) + [""]:
        if step is not None and line.startswith("      "):
            step += " " + line.strip()
            continue
        if step is not None:
            groups.append(names(step))
            step = None
        if line.startswith("  word_step "):
            step = line[len("  word_step ") :].removeprefix("-clear ")
    return groups


class DeclarationTests(unittest.TestCase):
    def test_inout_discard_and_named_outputs_are_parsed(self):
        discarded = parse_declaration("z3 = inout(reg) product[3] => _,")
        named = parse_declaration("z0 = inout(reg) product[0] => o1,")
        self.assertEqual(
            discarded,
            rust.Declaration("z3", "inout", "product[3]", "_"),
        )
        self.assertEqual(
            named,
            rust.Declaration("z0", "inout", "product[0]", "o1"),
        )

    def test_unsupported_constraint_is_rejected(self):
        with self.assertRaisesRegex(GenerationError, "constraint"):
            parse_declaration("x = in(xmm_reg) value,")

    def test_unsupported_fixed_register_is_rejected(self):
        with self.assertRaisesRegex(GenerationError, "fixed-register"):
            parse_declaration('in("rax") value,')

    def test_unsupported_const_is_rejected(self):
        with self.assertRaisesRegex(GenerationError, "const operand"):
            parse_declaration("p3 = const OTHER_CONSTANT,")

    def test_named_rdx_and_internal_names_are_reserved(self):
        for name in ("rdx", "cf", "ofl", "s", "d", "m", "n"):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(GenerationError, "reserved operand name"),
            ):
                parse_declaration(f"{name} = out(reg) _,")

    def test_high_limb_value_is_validated(self):
        source = x86_64_blocks.SOURCE.read_text().replace(
            "const PASTA_HIGH_LIMB: u64 = 1 << 62;",
            "const PASTA_HIGH_LIMB: u64 = 1 << 61;",
        )
        with self.assertRaisesRegex(GenerationError, "1 << 62"):
            x86_64_blocks.validate_high_limb(source)


class EmitterTests(unittest.TestCase):
    @staticmethod
    def emitter(*registers):
        directions = {register: "inout" for register in registers}
        emitter = x86_64.Lifter(directions, {})
        for register in registers:
            if register != "rdx":
                x86_bind(emitter, register, "0")
        return emitter

    def test_unsupported_instruction_is_rejected(self):
        emitter = self.emitter("a")
        with self.assertRaisesRegex(GenerationError, "unsupported instruction or"):
            emitter.lift("or {a}, 1")

    def test_input_only_register_cannot_be_written(self):
        emitter = x86_64.Lifter({"a": "in", "b": "in"}, {})
        x86_bind(emitter, "a", "1", "test input")
        x86_bind(emitter, "b", "2", "test input")
        with self.assertRaisesRegex(GenerationError, "input-only register a"):
            emitter.lift("add {a}, {b}")

    def test_input_only_register_cannot_be_xor_zeroed(self):
        emitter = x86_64.Lifter({"z": "in"}, {})
        x86_bind(emitter, "z", "1", "test input")
        with self.assertRaisesRegex(GenerationError, "input-only register z"):
            emitter.lift("xor {z:e}, {z:e}")

    def test_register_read_before_write_is_rejected(self):
        emitter = x86_64.Lifter({"a": "inout", "b": "in"}, {})
        x86_bind(emitter, "a", "0", "test input")
        with self.assertRaisesRegex(GenerationError, "b read before"):
            emitter.lift("add {a}, {b}")

    def test_memory_write_is_rejected(self):
        emitter = x86_64.Lifter({"a": "inout", "p": "in"}, {"p": "value"})
        x86_bind(emitter, "a", "0", "test input")
        with self.assertRaisesRegex(GenerationError, "memory write"):
            emitter.lift("mov qword ptr [{p}], {a}")

    def test_non_limb_memory_offset_is_rejected(self):
        emitter = x86_64.Lifter({"a": "inout", "p": "in"}, {"p": "value"})
        x86_bind(emitter, "a", "0", "test input")
        with self.assertRaisesRegex(GenerationError, "offset 32"):
            emitter.lift("mov {a}, qword ptr [{p} + 32]")

    def test_undeclared_address_base_is_rejected(self):
        emitter = x86_64.Lifter({"a": "inout"}, {})
        x86_bind(emitter, "a", "0", "test input")
        with self.assertRaisesRegex(GenerationError, "undeclared register p"):
            emitter.lift("mov {a}, qword ptr [{p}]")

    def test_register_is_not_an_address_base_without_pointer_binding(self):
        emitter = self.emitter("a", "p")
        with self.assertRaisesRegex(GenerationError, "not a read-only pointer"):
            emitter.lift("mov {a}, qword ptr [{p}]")

    def test_adc_rejects_invalid_cf(self):
        emitter = self.emitter("a", "b")
        with self.assertRaisesRegex(GenerationError, "CF read while invalid"):
            emitter.lift("adc {a}, {b}")

    def test_cmovnc_rejects_invalid_cf(self):
        emitter = self.emitter("a", "b")
        with self.assertRaisesRegex(GenerationError, "CF read while invalid"):
            emitter.lift("cmovnc {a}, {b}")

    def test_add_invalidates_of(self):
        emitter = self.emitter("a", "b", "z")
        emitter.lift("xor {z:e}, {z:e}")
        emitter.lift("add {a}, {b}")
        with self.assertRaisesRegex(GenerationError, "OF read while invalid"):
            emitter.lift("adox {a}, {b}")

    def test_neg_invalidates_of(self):
        emitter = self.emitter("a", "b", "z")
        emitter.lift("xor {z:e}, {z:e}")
        emitter.lift("neg {a}")
        with self.assertRaisesRegex(GenerationError, "OF read while invalid"):
            emitter.lift("adox {a}, {b}")

    def test_imul_invalidates_flags(self):
        emitter = self.emitter("a", "b", "z")
        emitter.lift("xor {z:e}, {z:e}")
        emitter.lift("imul {a}, {b}")
        with self.assertRaisesRegex(GenerationError, "CF read while invalid"):
            emitter.lift("adc {a}, {b}")

    def test_masked_or_zero_shift_counts_are_rejected(self):
        for amount in (0, 64, 65, 256):
            with self.subTest(amount=amount):
                emitter = self.emitter("a")
                with self.assertRaisesRegex(GenerationError, "shift counts"):
                    emitter.lift(f"shl {{a}}, {amount}")

    def test_shift_invalidates_flags(self):
        emitter = self.emitter("a", "b", "z")
        emitter.lift("xor {z:e}, {z:e}")
        emitter.lift("shl {a}, 62")
        with self.assertRaisesRegex(GenerationError, "CF read while invalid"):
            emitter.lift("adcx {a}, {b}")

    def test_xor_self_allows_uninitialized_output_and_clears_both_flags(self):
        emitter = x86_64.Lifter({"a": "inout", "b": "in", "z": "out"}, {})
        x86_bind(emitter, "a", "1", "test input")
        x86_bind(emitter, "b", "2", "test input")
        emitter.lift("xor {z:e}, {z:e}")
        emitter.lift("adcx {a}, {b}")
        emitter.lift("adox {a}, {b}")

    def test_adcx_and_adox_preserve_the_other_flag(self):
        emitter = self.emitter("a", "b", "z")
        emitter.lift("xor {z:e}, {z:e}")
        emitter.lift("adcx {a}, {b}")
        emitter.lift("adox {a}, {b}")
        # OF from ADOX and CF from ADCX are both still valid.
        emitter.lift("adox {a}, {b}")
        emitter.lift("adcx {a}, {b}")

    def test_mov_mulx_and_cmov_preserve_flags(self):
        emitter = self.emitter("a", "b", "z", "rdx", "hi", "lo")
        x86_bind(emitter, "rdx", "3", "fixed register test input")
        emitter.lift("xor {z:e}, {z:e}")
        emitter.lift("mov {a}, {b}")
        emitter.lift("mulx {hi}, {lo}, {a}")
        emitter.lift("cmovnc {a}, {b}")
        emitter.lift("adox {a}, {b}")
        emitter.lift("adcx {a}, {b}")


class AArch64WriteDirectionTests(unittest.TestCase):
    def test_input_only_destination_is_rejected(self):
        emitter = aarch64.Lifter({"b0": "in", "r0": "inout"})
        bound_registers(emitter, ["r0"])
        with self.assertRaisesRegex(ValueError, "input-only register b0"):
            aarch64_step(emitter, "mov", ["b0", "r0"], "mov b0, r0")

    def test_undeclared_destination_is_rejected(self):
        emitter = aarch64.Lifter({"r0": "inout"})
        with self.assertRaisesRegex(ValueError, "undeclared destination"):
            aarch64_step(emitter, "mov", ["bad", "r0"], "mov bad, r0")

    def test_output_and_inout_writes_are_allowed(self):
        for direction in ("out", "inout"):
            emitter = aarch64.Lifter({"r0": direction})
            aarch64_step(emitter, "mov", ["r0", "xzr"], "mov r0, xzr")
            self.assertIn("r0", emitter.known)

    def test_zero_register_write_is_allowed(self):
        aarch64_step(aarch64.Lifter(), "mov", ["xzr", "xzr"], "mov xzr, xzr")


class AArch64FlagFormTests(unittest.TestCase):
    """A condition reads the flags in the form the last flag-setting instruction produced."""

    @staticmethod
    def emitter(*registers):
        emitter = aarch64.Lifter({register: "inout" for register in registers})
        bound_registers(emitter, registers)
        return emitter

    def run_steps(self, emitter, *lines):
        for line in lines:
            op, rest = line.split(" ", 1)
            aarch64_step(emitter, op, aarch64.tokenize(rest), line)

    def test_carry_condition_after_four_flags_is_rejected(self):
        emitter = self.emitter("a", "b")
        self.run_steps(emitter, "tst a, #1")
        with self.assertRaisesRegex(ValueError, "c read while the flags are not in that form"):
            self.run_steps(emitter, "csel a, a, b, cs")

    def test_four_flag_condition_after_carry_chain_is_rejected(self):
        emitter = self.emitter("a", "b")
        self.run_steps(emitter, "adds a, a, b")
        with self.assertRaisesRegex(ValueError, "fl read while the flags are not in that form"):
            self.run_steps(emitter, "csel a, a, b, ne")

    def test_divstep_conditions_and_signed_operations_transcribe(self):
        emitter = self.emitter("d", "pf", "pg", "t", "m")
        self.run_steps(
            emitter,
            "tst pg, #1",
            "csel t, pf, xzr, ne",
            "ccmp d, xzr, #8, ne",
            "cneg d, d, ge",
            "csel pf, pg, pf, ge",
            "add pg, pg, t",
            "add d, d, #2",
            "asr pg, pg, #1",
            "add m, pf, #0x100, lsl #12",
            "sbfx m, m, #21, #21",
            "add m, m, m, lsl #21",
            "cmp m, xzr",
            "csetm t, mi",
            "cneg m, m, mi",
            "mneg t, m, d",
            "msub m, t, d, pf",
            "madd m, t, d, pf",
            "extr m, pf, pg, #59",
            "eor m, m, t",
            "neg m, m",
            "sub m, m, t",
        )
        expressions = [let.expr for node in emitter.nodes[5:] for let in node.lets()]
        self.assertEqual(
            expressions,
            [
                "tstFlags (andw pg 1)",
                "cselNe fl pf 0",
                "ccmpNe fl d 0 8",
                "cnegGe fl d",
                "cselGe fl pg pf",
                "addw pg t",
                "addw d 2",
                "asr pg 1",
                "addw pf 0x100000",
                "sbfx m 21 21",
                "addw m (lsl m 21)",
                "cmpFlags m 0",
                "csetmMi fl",
                "cnegMi fl m",
                "mneg m d",
                "msub t d pf",
                "madd t d pf",
                "extr pf pg 59",
                "eorw m t",
                "negw m",
                "subw m t",
            ],
        )

    def test_unsupported_conditions_are_rejected(self):
        for line in ("csel a, a, b, eq", "cneg a, a, lt", "csetm a, pl", "ccmp a, xzr, #8, ge"):
            with self.subTest(line=line):
                emitter = self.emitter("a", "b")
                self.run_steps(emitter, "tst a, #1")
                with self.assertRaisesRegex(ValueError, "unexpected condition"):
                    self.run_steps(emitter, line)


class AArch64OperandCountTests(unittest.TestCase):
    def test_shifted_add_is_rejected(self):
        emitter = aarch64.Lifter()
        with self.assertRaisesRegex(ValueError, "adds expects 3 operands"):
            aarch64_step(
                emitter, "adds", aarch64.tokenize("r0, r0, b0, lsl #1"), "adds r0, r0, b0, lsl #1"
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
                    aarch64_step(aarch64.Lifter(), op, ["r0"] * actual, op)


class AArch64StreamTests(unittest.TestCase):
    """The instruction stream back end: each instruction as a term of the leakage model's
    `Instr`, and the instructions it cannot state."""

    def term(self, text):
        return aarch64.instruction_term(aarch64.parse_instruction(text))

    def test_operand_forms(self):
        self.assertEqual(
            self.term("csel {pf}, {pg}, {pf}, ge"),
            '⟨.csel, [.reg "pf", .reg "pg", .reg "pf", .cond .ge]⟩',
        )
        self.assertEqual(
            self.term("ccmp {d}, xzr, #8, ne"),
            '⟨.ccmp, [.reg "d", .zero, .imm 8, .cond .ne]⟩',
        )
        self.assertEqual(
            self.term("add {a}, {b}, {c}, lsl #20"),
            '⟨.add, [.reg "a", .reg "b", .reg "c", .lsl 20]⟩',
        )
        self.assertEqual(
            self.term("and {pf}, {f}, #0xfffff"),
            '⟨.and, [.reg "pf", .reg "f", .imm 0xfffff]⟩',
        )
        self.assertEqual(self.term("csetm {s}, mi"), '⟨.csetm, [.reg "s", .cond .mi]⟩')

    def test_unstateable_instructions_are_rejected(self):
        for text, message in [
            ("ldr x0, [x1, #8]", "unhandled instruction"),
            ("b.ne 1f", "unhandled instruction"),
            ("udiv x0, x1, x2", "unhandled instruction"),
            ("add x0, x1, [x2]", "unsupported operand"),
            ("add x0, x1, #-1", "negative immediate"),
            ("csel x0, x1, x2, eq", "unexpected condition"),
        ]:
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, message):
                self.term(text)

    def test_every_real_block_has_a_stream_of_its_instructions(self):
        source = aarch64_blocks.SOURCE.read_text()
        for block in aarch64_blocks.BLOCKS:
            with self.subTest(block=block.rust_name):
                name, _, terms = aarch64.stream(source, block, aarch64_blocks.TARGET)
                parsed = aarch64.parse_block(
                    source, block, aarch64_blocks.KINDS, aarch64_blocks.TARGET.reserved_names
                )
                self.assertEqual(name, f"{block.lean_name}Program")
                self.assertEqual([text for _, text in terms], [i.text for i in parsed.instructions])


class SharedAArch64ParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = aarch64_blocks.SOURCE.read_text()

    @staticmethod
    def parse_block(block):
        return aarch64.parse_block(
            aarch64_blocks.SOURCE.read_text(),
            block,
            aarch64_blocks.KINDS,
            aarch64_blocks.TARGET.reserved_names,
        )

    def test_all_real_blocks_use_shared_parsed_function_model(self):
        for block in aarch64_blocks.BLOCKS:
            rust_name = block.rust_name
            with self.subTest(routine=rust_name):
                parsed = rust.parse_function(
                    self.source,
                    rust_name,
                    block.arg_names,
                    len(aarch64_blocks.KINDS[block.result]),
                    allowed_options={"pure", "nomem", "nostack"},
                    required_options={"pure", "nomem", "nostack"},
                )
                adapted = self.parse_block(block)
                self.assertIsInstance(parsed, rust.ParsedFunction)
                self.assertEqual(len(adapted.instructions), len(parsed.instructions))
                self.assertEqual(adapted.declarations, parsed.declarations)
                self.assertEqual(adapted.locals, parsed.locals)
                self.assertEqual(adapted.outputs, rust.output_bindings(parsed, rust_name))
                self.assertEqual(adapted.returned, rust.returned_registers(parsed, rust_name))
                self.assertEqual(adapted.origins, parsed.origins)
                self.assertEqual(len(adapted.origins), len(parsed.instructions))

    def test_divstep_macro_expands_to_the_step_body_at_every_site(self):
        macros = rust.parse_macros(self.source)
        self.assertEqual(set(macros["divstep"]), {"core", "", "last"})
        core = macros["divstep"]["core"]
        self.assertEqual(len(core), 7)
        self.assertEqual(macros["divstep"][""], core + ("tst {pg}, #2", "asr {pg}, {pg}, #1"))
        self.assertEqual(macros["divstep"]["last"], core + ("asr {pg}, {pg}, #1",))
        block = next(b for b in aarch64_blocks.BLOCKS if b.rust_name == "divstep59")
        parsed = self.parse_block(block)
        sites = {}
        for instruction, origin in zip(parsed.instructions, parsed.origins, strict=True):
            if origin is not None:
                sites.setdefault(origin, []).append(instruction)
        arms = [
            (origin.arm, len(body))
            for origin, body in sorted(sites.items(), key=lambda s: s[0].site)
        ]
        # Batches of 20, 20, and 19 steps, each of full steps and a last one, in invocation order.
        expected = ([""] * 19 + ["last"]) * 2 + [""] * 18 + ["last"]
        self.assertEqual([a for a, _ in arms], expected)
        self.assertEqual({n for a, n in arms if a == ""}, {9})
        self.assertEqual({n for a, n in arms if a == "last"}, {8})
        self.assertEqual(
            [origin.site for origin in sorted(sites, key=lambda o: o.site)], list(range(59))
        )

    def test_divstep_rounds_are_factored_from_the_macro(self):
        programs = {program.name: program for program in aarch64_blocks.programs()}
        block = programs["divstep59Block"]
        calls = [node for node in block.nodes if isinstance(node, Call)]
        # A run of consecutive invocations is one call of the round iterated over the run; the
        # third batch's run breaks where the second batch's matrix products are scheduled.
        self.assertEqual(
            [node.expr().split(" ")[0] for node in calls],
            ["divstepRound^[19]", "divstepLast"] * 2
            + ["divstepRound^[10]", "divstepRound^[8]", "divstepLast"],
        )
        for name in ("divstepRound", "divstepLast"):
            body = [code for code, _ in lean.body_lines(programs[name]) if code is not None]
            self.assertEqual(
                body[:4],
                ["  let d := st.d", "  let pf := st.f", "  let pg := st.g", "  let fl := st.fl"],
            )
        self.assertEqual(len(lean.body_lines(programs["divstepRound"])), 4 + 9)
        self.assertEqual(len(lean.body_lines(programs["divstepLast"])), 4 + 8)
        # The last step's flags are carried but never read; the block leaves the binding as a comment.
        comments = [comment for code, comment in lean.body_lines(block) if code is None]
        self.assertEqual(len([c for c in comments if "fl = step" in c]), 3)

    def test_real_named_and_implicit_outputs_are_extracted(self):
        mul = rust.parse_function(
            self.source,
            "mul",
            ["lhs", "rhs", "modulus", "inv"],
            4,
            allowed_options={"pure", "nomem", "nostack"},
            required_options={"pure", "nomem", "nostack"},
        )
        add = rust.parse_function(
            self.source,
            "add",
            ["lhs", "rhs", "modulus"],
            4,
            allowed_options={"pure", "nomem", "nostack"},
            required_options={"pure", "nomem", "nostack"},
        )
        self.assertEqual(mul.returns, ("o0", "o1", "o2", "o3"))
        self.assertEqual(
            rust.output_bindings(mul, "mul"),
            {"o0": "r0", "o1": "r1", "o2": "r2", "o3": "r3"},
        )
        self.assertEqual(add.returns, ("r0", "r1", "r2", "r3"))
        self.assertEqual(
            rust.output_bindings(add, "add"),
            {"r0": "r0", "r1": "r1", "r2": "r2", "r3": "r3"},
        )


class X86RealSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = x86_64_blocks.SOURCE.read_text()

    def mutate(self, old, new):
        self.assertEqual(self.source.count(old), 1, old)
        return self.source.replace(old, new)

    def assert_mutation_rejected(self, old, new, message):
        source = self.mutate(old, new)
        with self.assertRaisesRegex(GenerationError, message):
            x86_64_blocks.text(source)

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
        with self.assertRaisesRegex(GenerationError, message):
            x86_64_blocks.text(source)

    def flat_mul(self):
        return x86_64.lift_block(self.source, x86_block("mulMont"), X86)

    def test_all_six_blocks_generate(self):
        generated = x86_64_blocks.text(self.source)
        for block in x86_64_blocks.BLOCKS:
            self.assertIn(f"def {block.lean_name} ", generated)

    def test_source_output_tuple_order_is_used(self):
        mul = transcribed(self.source, x86_block("mulMont"))
        square_hi = transcribed(self.source, x86_block("squareHi"))
        self.assertTrue(mul.rstrip().endswith("⟨ee, ae, be, ce⟩"))
        self.assertTrue(square_hi.rstrip().endswith("⟨a, z0, z1, z2⟩"))

    def test_each_unfactored_source_instruction_has_one_generated_comment(self):
        for block in x86_64_blocks.BLOCKS:
            if block.lean_name == "mulMont":
                continue
            with self.subTest(routine=block.lean_name):
                parsed = x86_64.parse_function(self.source, block, X86)
                generated = transcribed(self.source, block)
                comments = Counter(re.findall(r"-- (.+)$", generated, re.MULTILINE))
                self.assertEqual(
                    Counter(parsed.instructions),
                    Counter(
                        {instruction: comments[instruction] for instruction in parsed.instructions}
                    ),
                )

    def test_factored_mul_rounds_match_under_rotation(self):
        nodes = self.flat_mul().nodes
        rounds = [MUL_ROUNDS.normalized(nodes, index) for index in range(len(MUL_ROUNDS.ranges))]
        self.assertEqual(MUL_ROUNDS.fingerprint(rounds[0]), MUL_ROUNDS.fingerprint(rounds[1]))
        self.assertEqual({node.pc for node in rounds[0]}, set(range(36)))

    def test_factored_mul_round_difference_is_rejected(self):
        source = self.mutate_function(
            "mul",
            '            "add {ce}, {s1}",\n            "adc {de}, {s2}",\n            "adc {ee}, 0",\n',
            '            "add {ce}, {s1}",\n            "adc {de}, {s2}",\n            "adc {ee}, {s1}",\n',
        )
        with self.assertRaisesRegex(
            GenerationError,
            "flattened full rounds 1 and 2 differ after accumulator rotation",
        ):
            x86_64_blocks.text(source)

    def test_factored_round_flattens_back_to_each_source_round(self):
        nodes = self.flat_mul().nodes
        load, body = MUL_ROUNDS.body(nodes, GenerationError)
        registers = ["be", "ce", "de", "ee", "ae", "rdx"]
        for round_number, (first, last) in enumerate(MUL_ROUNDS.ranges, start=1):
            source = [node for node in nodes if node.pc is not None and first <= node.pc <= last]
            flattened = MUL_ROUNDS.flattened(load, body, registers, f"rhs.l{round_number}", first)
            self.assertEqual(MUL_ROUNDS.fingerprint(flattened), MUL_ROUNDS.fingerprint(source))
            self.assertEqual({node.pc for node in flattened}, set(range(first, last + 1)))
            registers = registers[1:5] + registers[:1] + ["rdx"]

    def test_factored_round_call_with_misordered_registers_is_rejected(self):
        nodes = self.flat_mul().nodes
        load, body = MUL_ROUNDS.body(nodes, GenerationError)
        with self.assertRaisesRegex(GenerationError, "does not flatten to the source's round 1"):
            MUL_ROUNDS.check_call(
                nodes,
                load,
                body,
                ["ce", "be", "de", "ee", "ae", "rdx"],
                "rhs.l1",
                0,
                GenerationError,
            )

    def test_factored_mul_comments_cover_one_validated_full_round(self):
        block = x86_block("mulMont")
        generated = transcribed(self.source, block)
        comments = Counter(re.findall(r"-- (.+)$", generated, re.MULTILINE))
        parsed = x86_64.parse_function(self.source, block, X86)
        omitted = []
        second_first, second_last = MUL_ROUNDS.ranges[1]
        for pc, instruction in enumerate(parsed.instructions):
            if not second_first <= pc <= second_last:
                omitted.append(instruction)
        self.assertEqual(
            Counter(omitted),
            Counter({instruction: comments[instruction] for instruction in omitted}),
        )

    def test_declared_pointer_blocks_are_readonly(self):
        for block in (x86_block("mulMont"), x86_block("squareHi")):
            with self.subTest(routine=block.lean_name):
                parsed = x86_64.parse_function(self.source, block, X86)
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
        self.assertEqual(x86_64_blocks.text(source), x86_64_blocks.text(self.source))

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

    def test_x86_skeleton_extracts_every_retained_instruction_let(self):
        programs = {program.name: program for program in x86_64_blocks.programs()}
        # Every retained binding is extracted exactly once, equal values included (the three
        # zeros of an `xor r, r`, a repeated `mov` from `rdx`), since merging is off.
        for name in ("squareLo", "mulMontRound", "fromMont"):
            with self.subTest(routine=name):
                program = programs[name]
                names = ssa_names([let for _, let, _ in program.kept()])
                groups = word_step_groups(skeleton(program))
                extracted = [name for group in groups for name in group]
                self.assertEqual(sorted(extracted), sorted(names))
                self.assertEqual(len(extracted), len(set(extracted)))
        square_lo_lines = skeleton(programs["squareLo"])
        square_lo = "\n".join(square_lo_lines)
        self.assertIn(["s_2", "z4_1", "cf_5"], word_step_groups(square_lo_lines))
        self.assertNotIn("obtain ⟨cf_5, b_cf_5, l_z4_1⟩", square_lo)

        mul_round = programs["mulMontRound"]
        mul_round_lines = skeleton(mul_round)
        mul_round_text = "\n".join(mul_round_lines)
        # The factored round is rendered like every other block: an instruction's pair wrapper
        # and its two projections, extracted together, and read under their own names.
        self.assertIn(["m", "s2", "s1_1"], word_step_groups(mul_round_lines))
        self.assertIn(["s", "r0_1", "cf_1"], word_step_groups(mul_round_lines))
        self.assertIn("r0_1 := (addc r0 s1_1 cf).1 using", mul_round_text)
        self.assertIn("cf_1 := (addc r0 s1_1 cf).2", mul_round_text)
        # The round's and the calling block's local definitions stay transparent.
        for lines in (mul_round_lines, skeleton(programs["mulMont"])):
            steps = [line for line in lines if line.startswith("  word_step")]
            self.assertTrue(steps)
            self.assertTrue(all(line.startswith("  word_step -clear ") for line in steps))
        helper_text = "\n".join(code for code, _ in lean.body_lines(mul_round))
        self.assertIn("let m := mulx b lhs.l0", helper_text)
        self.assertIn("let s := addc r0 s1 cf", helper_text)
        self.assertIn("let r0 := s.1", helper_text)
        self.assertIn("let cf := s.2", helper_text)
        # The omitted entry load aliases RDX to b only until the source writes RDX again.
        # Reduction products must use the rebound Montgomery quotient, never b.
        expressions = [let.expr for _, let in mul_round.bindings()]
        quotient_index = expressions.index("mulLo rdx inv")
        self.assertTrue(any("mulx rdx modulus.l1" in expr for expr in expressions[quotient_index:]))
        self.assertFalse(any("mulx b modulus.l1" in expr for expr in expressions[quotient_index:]))

        from_mont_lines = skeleton(programs["fromMont"])
        from_mont = "\n".join(from_mont_lines)
        self.assertIn(["n", "z0_1", "cf"], word_step_groups(from_mont_lines))
        self.assertIn("z0_1 := (neg z0).1 using", from_mont)
        self.assertIn("cf := (neg z0).2", from_mont)
        self.assertIn("\n  word_step n,", from_mont)

    def test_all_x86_routines_and_factored_round_generate_shared_skeletons(self):
        programs = x86_64_blocks.programs()
        self.assertEqual(
            [program.name for program in programs],
            ["addMod", "subMod", "mulMontRound", "mulMont", "squareLo", "squareHi", "fromMont"],
        )
        for program in programs:
            with self.subTest(routine=program.name):
                generated = skeleton(program)
                self.assertEqual(
                    generated[0],
                    f"  -- generated skeleton for `{program.name}`: do not edit between the annotations",
                )
                self.assertEqual(generated[-1], "  subst hr")
                self.assertEqual(gen.find_routine(f"X86_64:{program.name}").architecture, "X86_64")


class SharedGeneratorTests(unittest.TestCase):
    def test_skeleton_hooks_are_backend_owned(self):
        # Each architecture's proof steps belong to its own nodes: a program holds only the
        # shared nodes of `ir` and its own architecture's.
        def modules(programs):
            return {type(node).__module__ for program in programs for node in program.nodes}

        aarch_programs = aarch64_blocks.programs()
        x86_programs = x86_64_blocks.programs()
        self.assertLessEqual(modules(aarch_programs), {ir.__name__, aarch64.__name__})
        self.assertLessEqual(modules(x86_programs), {ir.__name__, x86_64.__name__})

        aarch_add = next(program for program in aarch_programs if program.name == "addMod")
        kept = [(node, let) for node, let, live in aarch_add.kept() if live]
        self.assertTrue(
            any(
                isinstance(node, aarch64.SubBorrow) and sum(1 for n, _ in kept if n is node) == 3
                for node, _ in kept
            )
        )
        x86_square = next(program for program in x86_programs if program.name == "squareLo")
        self.assertTrue(any(isinstance(node, x86_64.Mulx) for node in x86_square.nodes))

    def test_skeleton_facts_come_from_the_routine_backend(self):
        # The shared traversal knows only how to walk; each node states its own step, and a node
        # that states none is an error.
        import dataclasses

        @dataclasses.dataclass(frozen=True, kw_only=True)
        class Stub(Node):
            dest: str = ir.dst()
            a: str = ir.src()

            def lets(self):
                return [self._let(self.dest, f"stub {self.a}", self.a)]

            def prove(self, step):
                step.eq(step.name, f"stub {step.r(self.a)}")
                step.line(f"  have b_{step.name} : {step.name} < 2^64 := by sorry")
                step.state.bnd[step.name] = f"b_{step.name}"

        @dataclasses.dataclass(frozen=True, kw_only=True)
        class Unproved(Stub):
            def prove(self, step):
                return Node.prove(self, step)

        def program(stub):
            nodes = [
                Load(comment="argument", dest="x", arg="lhs", field="l0"),
                stub(comment="stub x", dest="y", a="x"),
            ]
            return CONVENTIONS.program("Stub", "stub", "doc", "def stub", nodes, ["y"])

        text = "\n".join(skeleton(program(Stub)))
        self.assertIn("  word_step x := lhs.l0 using hlhs.1\n", text)
        self.assertIn("  word_step y := stub x\n", text)
        self.assertIn("have b_y : y < 2^64 := by sorry", text)
        with self.assertRaisesRegex(ValueError, "unsupported skeleton fact Unproved"):
            skeleton(program(Unproved))

    def test_backward_liveness_and_backend_retention_share_the_same_ir(self):
        nodes = [
            Mov(comment="argument", dest="input", a="lhs.l0"),
            Mov(comment="dead", dest="dead", a="input"),
            Mov(comment="result", dest="result", a="input"),
        ]
        eliminated = CONVENTIONS.program("Test", "t", "doc", "def t", nodes, ["result"])
        retained = CONVENTIONS.program(
            "Test", "t", "doc", "def t", nodes, ["result"], dead_code=ir.DeadCode.RETAIN
        )
        for program in (eliminated, retained):
            self.assertEqual(
                [
                    let.name
                    for (_, let), live in zip(program.bindings(), program.liveness())
                    if live
                ],
                ["input", "result"],
            )
        # Eliminating it, a dead computed value is dead code in the block, an error; retaining
        # it, the binding is kept.
        with self.assertRaisesRegex(ValueError, "dead computation: dead := input"):
            lean.body_lines(eliminated)
        self.assertIn(("  let dead := input", "dead"), lean.body_lines(retained))

    def test_transcription_snapshots_match_committed_files(self):
        self.assertEqual(aarch64_blocks.text(), aarch64_blocks.OUTPUT.read_text())
        self.assertEqual(x86_64_blocks.text(), x86_64_blocks.OUTPUT.read_text())

    def test_round_argument_fields_are_routine_local(self):
        aarch_round = next(p for p in aarch64_blocks.programs() if p.name == "mulMontRound")
        x86_round = next(p for p in x86_64_blocks.programs() if p.name == "mulMontRound")
        self.assertEqual(
            aarch_round.arg_fields["acc"], ["r0", "r1", "r2", "r3", "r4", "q", "t1", "t3"]
        )
        self.assertEqual(x86_round.arg_fields["acc"], ["r0", "r1", "r2", "r3", "r4", "q"])
        self.assertEqual(projection(aarch_round.arg_fields["acc"], "q"), "2.2.2.2.2.1")
        self.assertEqual(projection(x86_round.arg_fields["acc"], "q"), "2.2.2.2.2")
        self.assertEqual(projection(aarch_round.arg_fields["acc"], "r4"), "2.2.2.2.1")
        self.assertEqual(projection(x86_round.arg_fields["acc"], "r4"), "2.2.2.2.1")

    def test_skeleton_lookup_names_the_architecture(self):
        self.assertEqual(gen.find_routine("AArch64:addMod").architecture, "AArch64")
        self.assertEqual(gen.find_routine("X86_64:addMod").architecture, "X86_64")
        # Both backends have an `addMod`, so a bare name would be ambiguous.
        with self.assertRaisesRegex(ValueError, "expected ARCH:NAME, not addMod"):
            gen.find_routine("addMod")


class SkeletonCheckerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.routine = gen.find_routine("X86_64:addMod")
        cls.skeleton = "\n".join(skeleton(cls.routine)) + "\n"

    def check_text(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Spec.lean"
            path.write_text(text)
            diagnostics = io.StringIO()
            with contextlib.redirect_stderr(diagnostics):
                ok = check_spec(path, [self.routine])
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
                self.assertFalse(check_spec(Path(directory) / "missing.lean", []))
            self.assertIn("cannot read proof file", diagnostics.getvalue())

    def test_manifest_selects_existing_files_and_rejects_mismatched(self):
        add_path = ROOT / "lean/PastaCurves/X86_64/Spec/Add.lean"
        available = gen.architecture_routines()
        for name, (arch, expected_names) in gen.SPEC_MANIFEST.items():
            with self.subTest(path=name):
                manifest_path = ROOT / name
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


class KnownAnswerTests(unittest.TestCase):
    """The known-answer literals of the backend tests, read for `KnownAnswers.lean`."""

    SOURCE = """
const FP: Field = Field {
    modulus: fp::MODULUS.0,
    two_r: [
        0x1,
        0x2,
        0x3,
        0x4,
    ],
};
"""

    def test_literals_are_read_in_order_and_references_skipped(self):
        self.assertEqual(data.parse_known_answers(self.SOURCE), [("FP", "two_r", [1, 2, 3, 4])])

    def test_a_literal_without_a_definition_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no known-answer definition"):
            data.parse_known_answers(self.SOURCE.replace("two_r", "five_r"))

    def test_an_unknown_field_constant_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown field constant"):
            data.parse_known_answers(self.SOURCE.replace("const FP", "const FR"))

    def test_the_tests_have_literals(self):
        with self.assertRaisesRegex(ValueError, "no known-answer literals"):
            data.parse_known_answers("")

    def test_each_literal_becomes_one_kernel_checked_example(self):
        rendered = data.render_known_answers(self.SOURCE)
        self.assertEqual(rendered.count("decide +kernel"), 1)
        self.assertIn(
            "let x := Limbs.toNat\n"
            "      ⟨0x0000000000000001, 0x0000000000000002, 0x0000000000000003, "
            "0x0000000000000004⟩\n",
            rendered,
        )


class InversionPairTests(unittest.TestCase):
    """The inversion pairs of the backend tests, read for `KnownAnswers.lean`."""

    SOURCE = """
const FP: Field = Field {
    inversions: [
        (
            [0x1, 0x2, 0x3, 0x4],
            [0x5, 0x6, 0x7, 0x8],
        ),
    ],
};
"""

    def test_pairs_are_read_in_order(self):
        self.assertEqual(
            data.parse_inversions(self.SOURCE), [("FP", 0, [1, 2, 3, 4], [5, 6, 7, 8])]
        )

    def test_a_pair_that_is_not_two_four_limb_values_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not two four-limb values"):
            data.parse_inversions(self.SOURCE.replace("0x3, 0x4]", "0x3]"))

    def test_each_pair_becomes_one_kernel_checked_example(self):
        rendered = data.render_known_answers(
            self.SOURCE.replace("inversions", "two_r: [0x1, 0x2, 0x3, 0x4,],\n    inversions")
        )
        self.assertIn("z < p ∧ (x*z) % p = (if x = 0 then 0 else R^2 % p)", rendered)


class FieldTypeTests(unittest.TestCase):
    """The field types' constants, read for `FieldTypes.lean`."""

    SOURCE = """
pub(crate) const MODULUS: Fp = Fp([0x1, 0x2, 0x0, 0x4]);
pub(crate) const INV: u64 = 0x5;
pub(crate) const R: Fp = Fp([0x6, 0x7, 0x8, 0x9]);
pub(crate) const R2: Fp = Fp([
    0xa,
    0xb,
    0xc,
    0xd,
]);
pub(crate) const R3: Fp = Fp([0xe, 0xf, 0x10, 0x11]);
"""

    def test_constants_are_read_as_the_source_spells_them(self):
        modulus, inv, powers = data.parse_field_type(self.SOURCE, "fp.rs")
        self.assertEqual(modulus, ["0x1", "0x2", "0x0", "0x4"])
        self.assertEqual(inv, "0x5")
        self.assertEqual(powers["R2"], ["0xa", "0xb", "0xc", "0xd"])

    def test_a_missing_constant_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing R3"):
            data.parse_field_type(self.SOURCE.replace("const R3", "const S3"), "fp.rs")
        with self.assertRaisesRegex(ValueError, "missing INV"):
            data.parse_field_type(self.SOURCE.replace("const INV", "const NV"), "fp.rs")

    def test_each_constant_becomes_one_example(self):
        rendered = data.render_field_types({"fp": self.SOURCE, "fq": self.SOURCE})
        self.assertEqual(rendered.count("decide +kernel"), 6)
        self.assertEqual(rendered.count("  decide\n"), 4)
        self.assertIn("example : pallasBase.modulus =\n    ⟨0x1, 0x2, 0x0, 0x4⟩ := by\n", rendered)
        self.assertIn("example : vestaBase.inv = 0x5 := by\n", rendered)
        self.assertIn("      ⟨0xa, 0xb, 0xc, 0xd⟩\n    x = R^2 % p := by\n", rendered)


if __name__ == "__main__":
    unittest.main()
