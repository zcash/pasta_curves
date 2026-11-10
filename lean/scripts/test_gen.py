#!/usr/bin/env python3
"""Unit tests for shared asm parsing and architecture-specific generators."""

import contextlib
import dataclasses
import io
import random
import re
import sys
import tempfile
import unittest
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

# Running this source-tree test should not leave lean/scripts/__pycache__ behind.
sys.dont_write_bytecode = True

import gen
from asm2lean import block as driver
from asm2lean import ir, isa, lean, reroll, rust, validate
from asm2lean.ir import Call, Load, Mov, Node
from asm2lean.isa import aarch64, x86_64
from asm2lean.skeleton import State, Step, projection, skeleton, ssa_names
from asm2lean.specs import check_spec
from pasta import data
from pasta.architectures import aarch64 as aarch64_blocks
from pasta.architectures import x86_64 as x86_64_blocks
from pasta.conventions import CONVENTIONS
from pasta.paths import ROOT
from pasta.specs import SPEC_MANIFEST

GenerationError = rust.GenerationError
LIMBS = ("l0", "l1", "l2", "l3")
X86 = x86_64_blocks.TARGET
MUL_ROUNDS = x86_64_blocks.MUL_ROUNDS


def parse_declaration(text: str) -> rust.Declaration:
    """An operand declaration, under the x86-64 target's restrictions."""
    target_isa = X86.isa
    return rust.parse_declaration(
        text,
        reserved_names=set(target_isa.reserved_names),
        fixed_registers=set(x86_64.FIXED_REGISTERS),
        const_operands=set(X86.consts),
    )


def x86_block(name: str) -> driver.Block:
    return next(block for block in x86_64_blocks.BLOCKS if block.lean_name == name)


def transcribed(source: str, block: driver.Block) -> str:
    """The x86-64 transcription of one block, with its round when it has one."""
    return "\n".join(
        lean.definition(p, lean.comment_column([p])) for p in driver.transcribe(source, block, X86)
    )


def x86_bind(lifter: isa.Lifter, register: str, value: int, comment: str = "test input") -> None:
    lifter.bind(Mov(comment=comment, dest=register, a=ir.lit(value)))


def word(operand: str) -> ir.Term:
    """A test operand: a literal if it is a number, else a register."""
    return ir.lit(int(operand)) if operand.isdigit() else ir.Reg(operand)


def expressions(lifter: isa.Lifter) -> list[str]:
    """The right-hand side of every `let` of a lifter's nodes."""
    return [let.expr for node in lifter.nodes for let in node.lets()]


def aarch64_step(lifter: isa.Lifter, op: str, operands: Sequence[str], text: str) -> None:
    lifter.lift(isa.Instruction(op, tuple(operands), text))


def bound_registers(lifter: isa.Lifter, registers: Iterable[str]) -> None:
    for register in registers:
        lifter.bind(Mov(comment="argument", dest=register, a=ir.lit(0)))


def word_step_groups(skeleton: Iterable[str]) -> list[list[str]]:
    """The names that each `word_step` of a skeleton extracts, in order: a step's items are
    separated by commas outside brackets, and continue over its indented continuation lines."""

    def names(step: str) -> list[str]:
        found: list[str] = []
        depth, item = 0, ""
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

    groups: list[list[str]] = []
    step: str | None = None
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
    def test_inout_discard_and_named_outputs_are_parsed(self) -> None:
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

    def test_unsupported_constraint_is_rejected(self) -> None:
        with self.assertRaisesRegex(GenerationError, "constraint"):
            parse_declaration("x = in(xmm_reg) value,")

    def test_unsupported_fixed_register_is_rejected(self) -> None:
        with self.assertRaisesRegex(GenerationError, "fixed-register"):
            parse_declaration('in("rax") value,')

    def test_unsupported_const_is_rejected(self) -> None:
        with self.assertRaisesRegex(GenerationError, "const operand"):
            parse_declaration("p3 = const OTHER_CONSTANT,")

    def test_named_rdx_and_internal_names_are_reserved(self) -> None:
        for name in ("rdx", "rax", "cf", "ofl", "zs", "s", "d", "m", "n"):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(GenerationError, "reserved operand name"),
            ):
                parse_declaration(f"{name} = out(reg) _,")

    def test_high_limb_value_is_validated(self) -> None:
        source = x86_64_blocks.SOURCE.read_text().replace(
            "const PASTA_HIGH_LIMB: u64 = 1 << 62;",
            "const PASTA_HIGH_LIMB: u64 = 1 << 61;",
        )
        with self.assertRaisesRegex(GenerationError, "1 << 62"):
            x86_64_blocks.validate_high_limb(source)


class X86LifterTests(unittest.TestCase):
    @staticmethod
    def emitter(*registers: str) -> x86_64.Lifter:
        directions: dict[str, rust.Direction] = {register: "inout" for register in registers}
        emitter = x86_64.Lifter(directions)
        for register in registers:
            if register != "rdx":
                x86_bind(emitter, register, 0)
        return emitter

    def test_unsupported_instruction_is_rejected(self) -> None:
        emitter = self.emitter("a")
        with self.assertRaisesRegex(GenerationError, "unsupported instruction rol"):
            emitter.lift("rol {a}, 1")

    def test_zf_sf_read_without_a_defining_instruction_is_rejected(self) -> None:
        emitter = self.emitter("a", "b")
        with self.assertRaisesRegex(GenerationError, "ZF/SF read while invalid"):
            emitter.lift("cmove {a}, {b}")

    def test_sign_of_a_rewritten_add_destination_is_rejected(self) -> None:
        emitter = self.emitter("a", "b")
        emitter.lift("add {a}, 2")
        emitter.lift("mov {a}, {b}")
        with self.assertRaisesRegex(GenerationError, "ZF/SF read while invalid"):
            emitter.lift("cmovs {b}, {a}")

    def test_zf_sf_after_another_flag_writer_is_rejected(self) -> None:
        emitter = self.emitter("a", "b")
        emitter.lift("test {a}, 1")
        emitter.lift("xor {b}, {a}")
        with self.assertRaisesRegex(GenerationError, "ZF/SF read while invalid"):
            emitter.lift("cmove {a}, {b}")

    def test_zf_sf_survive_flag_preserving_instructions(self) -> None:
        emitter = self.emitter("a", "b")
        emitter.lift("add {a}, 2")
        emitter.lift("lea {b}, [{b} + {a}]")
        emitter.lift("cmovs {b}, {a}")
        self.assertIn("cmovs a b a", expressions(emitter))

    def test_cf_after_bt_is_readable(self) -> None:
        emitter = self.emitter("a", "b")
        emitter.lift("bt {a}, 63")
        emitter.lift("cmovb {a}, {b}")
        self.assertIn("bt a 63", expressions(emitter))

    def test_negative_immediate_is_its_twos_complement(self) -> None:
        emitter = self.emitter("a")
        emitter.lift("mov {a}, -2")
        self.assertIn(str(2**64 - 2), expressions(emitter))
        with self.assertRaisesRegex(GenerationError, "imm32"):
            emitter.lift(f"mov {{a}}, -{2**31 + 1}")

    def test_five_word_pointer_reads_its_last_word_only(self) -> None:
        emitter = x86_64.Lifter({"a": "out", "p": "in"})
        emitter.bind_pointer("p", "t", ("l0", "l1", "l2", "l3", "l4"))
        emitter.lift("mov {a}, qword ptr [{p} + 32]")
        self.assertIn("t.l4", expressions(emitter))
        with self.assertRaisesRegex(GenerationError, "unsupported memory offset 40"):
            emitter.lift("mov {a}, qword ptr [{p} + 40]")

    def test_input_only_register_cannot_be_written(self) -> None:
        emitter = x86_64.Lifter({"a": "in", "b": "in"})
        x86_bind(emitter, "a", 1, "test input")
        x86_bind(emitter, "b", 2, "test input")
        with self.assertRaisesRegex(GenerationError, "input-only register a"):
            emitter.lift("add {a}, {b}")

    def test_input_only_register_cannot_be_xor_zeroed(self) -> None:
        emitter = x86_64.Lifter({"z": "in"})
        x86_bind(emitter, "z", 1, "test input")
        with self.assertRaisesRegex(GenerationError, "input-only register z"):
            emitter.lift("xor {z:e}, {z:e}")

    def test_register_read_before_write_is_rejected(self) -> None:
        emitter = x86_64.Lifter({"a": "inout", "b": "in"})
        x86_bind(emitter, "a", 0, "test input")
        with self.assertRaisesRegex(GenerationError, "b read before"):
            emitter.lift("add {a}, {b}")

    def test_memory_write_is_rejected(self) -> None:
        emitter = x86_64.Lifter({"a": "inout", "p": "in"})
        emitter.bind_pointer("p", "value", LIMBS)
        x86_bind(emitter, "a", 0, "test input")
        with self.assertRaisesRegex(GenerationError, "memory write"):
            emitter.lift("mov qword ptr [{p}], {a}")

    def test_non_limb_memory_offset_is_rejected(self) -> None:
        emitter = x86_64.Lifter({"a": "inout", "p": "in"})
        emitter.bind_pointer("p", "value", LIMBS)
        x86_bind(emitter, "a", 0, "test input")
        with self.assertRaisesRegex(GenerationError, "offset 32"):
            emitter.lift("mov {a}, qword ptr [{p} + 32]")

    def test_undeclared_address_base_is_rejected(self) -> None:
        emitter = x86_64.Lifter({"a": "inout"})
        x86_bind(emitter, "a", 0, "test input")
        with self.assertRaisesRegex(GenerationError, "undeclared register p"):
            emitter.lift("mov {a}, qword ptr [{p}]")

    def test_register_is_not_an_address_base_without_pointer_binding(self) -> None:
        emitter = self.emitter("a", "p")
        with self.assertRaisesRegex(GenerationError, "not a read-only pointer"):
            emitter.lift("mov {a}, qword ptr [{p}]")

    def test_adc_rejects_invalid_cf(self) -> None:
        emitter = self.emitter("a", "b")
        with self.assertRaisesRegex(GenerationError, "CF read while invalid"):
            emitter.lift("adc {a}, {b}")

    def test_cmovnc_rejects_invalid_cf(self) -> None:
        emitter = self.emitter("a", "b")
        with self.assertRaisesRegex(GenerationError, "CF read while invalid"):
            emitter.lift("cmovnc {a}, {b}")

    def test_add_invalidates_of(self) -> None:
        emitter = self.emitter("a", "b", "z")
        emitter.lift("xor {z:e}, {z:e}")
        emitter.lift("add {a}, {b}")
        with self.assertRaisesRegex(GenerationError, "OF read while invalid"):
            emitter.lift("adox {a}, {b}")

    def test_neg_invalidates_of(self) -> None:
        emitter = self.emitter("a", "b", "z")
        emitter.lift("xor {z:e}, {z:e}")
        emitter.lift("neg {a}")
        with self.assertRaisesRegex(GenerationError, "OF read while invalid"):
            emitter.lift("adox {a}, {b}")

    def test_imul_invalidates_flags(self) -> None:
        emitter = self.emitter("a", "b", "z")
        emitter.lift("xor {z:e}, {z:e}")
        emitter.lift("imul {a}, {b}")
        with self.assertRaisesRegex(GenerationError, "CF read while invalid"):
            emitter.lift("adc {a}, {b}")

    def test_masked_or_zero_shift_counts_are_rejected(self) -> None:
        for amount in (0, 64, 65, 256):
            with self.subTest(amount=amount):
                emitter = self.emitter("a")
                with self.assertRaisesRegex(GenerationError, "shift counts"):
                    emitter.lift(f"shl {{a}}, {amount}")

    def test_shift_invalidates_flags(self) -> None:
        emitter = self.emitter("a", "b", "z")
        emitter.lift("xor {z:e}, {z:e}")
        emitter.lift("shl {a}, 62")
        with self.assertRaisesRegex(GenerationError, "CF read while invalid"):
            emitter.lift("adcx {a}, {b}")

    def test_xor_self_allows_uninitialized_output_and_clears_both_flags(self) -> None:
        emitter = x86_64.Lifter({"a": "inout", "b": "in", "z": "out"})
        x86_bind(emitter, "a", 1, "test input")
        x86_bind(emitter, "b", 2, "test input")
        emitter.lift("xor {z:e}, {z:e}")
        emitter.lift("adcx {a}, {b}")
        emitter.lift("adox {a}, {b}")

    def test_adcx_and_adox_preserve_the_other_flag(self) -> None:
        emitter = self.emitter("a", "b", "z")
        emitter.lift("xor {z:e}, {z:e}")
        emitter.lift("adcx {a}, {b}")
        emitter.lift("adox {a}, {b}")
        # OF from ADOX and CF from ADCX are both still valid.
        emitter.lift("adox {a}, {b}")
        emitter.lift("adcx {a}, {b}")

    def test_mov_mulx_and_cmov_preserve_flags(self) -> None:
        emitter = self.emitter("a", "b", "z", "rdx", "hi", "lo")
        x86_bind(emitter, "rdx", 3, "fixed register test input")
        emitter.lift("xor {z:e}, {z:e}")
        emitter.lift("mov {a}, {b}")
        emitter.lift("mulx {hi}, {lo}, {a}")
        emitter.lift("cmovnc {a}, {b}")
        emitter.lift("adox {a}, {b}")
        emitter.lift("adcx {a}, {b}")


class AArch64WriteDirectionTests(unittest.TestCase):
    def test_input_only_destination_is_rejected(self) -> None:
        emitter = aarch64.Lifter({"b0": "in", "r0": "inout"})
        bound_registers(emitter, ["r0"])
        with self.assertRaisesRegex(ValueError, "input-only register b0"):
            aarch64_step(emitter, "mov", ["{b0}", "{r0}"], "mov b0, r0")

    def test_undeclared_destination_is_rejected(self) -> None:
        emitter = aarch64.Lifter({"r0": "inout"})
        with self.assertRaisesRegex(ValueError, "undeclared register bad"):
            aarch64_step(emitter, "mov", ["{bad}", "{r0}"], "mov bad, r0")

    def test_output_and_inout_writes_are_allowed(self) -> None:
        for direction in ("out", "inout"):
            emitter = aarch64.Lifter({"r0": direction})
            aarch64_step(emitter, "mov", ["{r0}", "xzr"], "mov r0, xzr")
            self.assertIn("r0", emitter.known)

    def test_zero_register_write_is_allowed(self) -> None:
        aarch64_step(aarch64.Lifter(), "mov", ["xzr", "xzr"], "mov xzr, xzr")


class AArch64FlagFormTests(unittest.TestCase):
    """A condition reads the flags in the form the last flag-setting instruction produced."""

    @staticmethod
    def emitter(*registers: str) -> aarch64.Lifter:
        emitter = aarch64.Lifter({register: "inout" for register in registers})
        bound_registers(emitter, registers)
        return emitter

    def run_steps(self, emitter: aarch64.Lifter, *lines: str) -> None:
        """Lift template lines, whose operands are `{name}` placeholders."""
        for line in lines:
            emitter.lift(line)

    def test_carry_condition_after_four_flags_is_rejected(self) -> None:
        emitter = self.emitter("a", "b")
        self.run_steps(emitter, "tst {a}, #1")
        with self.assertRaisesRegex(ValueError, "c read while the flags are not in that form"):
            self.run_steps(emitter, "csel {a}, {a}, {b}, cs")

    def test_four_flag_condition_after_carry_chain_is_rejected(self) -> None:
        emitter = self.emitter("a", "b")
        self.run_steps(emitter, "adds {a}, {a}, {b}")
        with self.assertRaisesRegex(ValueError, "fl read while the flags are not in that form"):
            self.run_steps(emitter, "csel {a}, {a}, {b}, ne")

    def test_divstep_conditions_and_signed_operations_transcribe(self) -> None:
        emitter = self.emitter("d", "pf", "pg", "t", "m")
        self.run_steps(
            emitter,
            "tst {pg}, #1",
            "csel {t}, {pf}, xzr, ne",
            "ccmp {d}, xzr, #8, ne",
            "cneg {d}, {d}, ge",
            "csel {pf}, {pg}, {pf}, ge",
            "add {pg}, {pg}, {t}",
            "add {d}, {d}, #2",
            "asr {pg}, {pg}, #1",
            "add {m}, {pf}, #0x100, lsl #12",
            "sbfx {m}, {m}, #21, #21",
            "add {m}, {m}, {m}, lsl #21",
            "cmp {m}, xzr",
            "csetm {t}, mi",
            "cneg {m}, {m}, mi",
            "mneg {t}, {m}, {d}",
            "msub {m}, {t}, {d}, {pf}",
            "madd {m}, {t}, {d}, {pf}",
            "extr {m}, {pf}, {pg}, #59",
            "eor {m}, {m}, {t}",
            "neg {m}, {m}",
            "sub {m}, {m}, {t}",
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

    def test_unsupported_conditions_are_rejected(self) -> None:
        for line in (
            "csel {a}, {a}, {b}, eq",
            "cneg {a}, {a}, lt",
            "csetm {a}, pl",
            "ccmp {a}, xzr, #8, ge",
        ):
            with self.subTest(line=line):
                emitter = self.emitter("a", "b")
                self.run_steps(emitter, "tst {a}, #1")
                with self.assertRaisesRegex(ValueError, "unexpected condition"):
                    self.run_steps(emitter, line)


class AArch64OperandCountTests(unittest.TestCase):
    def test_shifted_add_is_rejected(self) -> None:
        emitter = aarch64.Lifter()
        with self.assertRaisesRegex(ValueError, "adds expects 3 operands"):
            aarch64_step(
                emitter, "adds", aarch64.tokenize("r0, r0, b0, lsl #1"), "adds r0, r0, b0, lsl #1"
            )

    def test_missing_and_extra_operands_are_rejected_before_reads(self) -> None:
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


class SharedAArch64ParserTests(unittest.TestCase):
    source: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = aarch64_blocks.SOURCE.read_text()

    @staticmethod
    def parse_block(block: driver.Block) -> rust.ParsedFunction:
        return driver.parse(aarch64_blocks.SOURCE.read_text(), block, aarch64_blocks.TARGET)

    def test_all_real_blocks_use_shared_parsed_function_model(self) -> None:
        for block in aarch64_blocks.BLOCKS:
            rust_name = block.rust_name
            with self.subTest(routine=rust_name):
                fields = aarch64_blocks.KINDS[block.result]
                assert fields is not None
                parsed = rust.parse_function(
                    self.source,
                    rust_name,
                    block.arg_names,
                    len(fields),
                    allowed_options={"pure", "nomem", "nostack"},
                    required_options={"pure", "nomem", "nostack"},
                )
                adapted = self.parse_block(block)
                self.assertIsInstance(parsed, rust.ParsedFunction)
                self.assertEqual(adapted, parsed)
                self.assertEqual(len(adapted.origins), len(parsed.instructions))

    def test_divstep_macro_expands_to_the_step_body_at_every_site(self) -> None:
        macros = rust.parse_macros(self.source)
        self.assertEqual(set(macros["divstep"]), {"core", "", "last"})
        core = macros["divstep"]["core"]
        self.assertEqual(len(core), 7)
        self.assertEqual(macros["divstep"][""], core + ("tst {pg}, #2", "asr {pg}, {pg}, #1"))
        self.assertEqual(macros["divstep"]["last"], core + ("asr {pg}, {pg}, #1",))
        block = next(b for b in aarch64_blocks.BLOCKS if b.rust_name == "divstep59")
        parsed = self.parse_block(block)
        sites: dict[rust.MacroOrigin, list[str]] = {}
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

    def test_divstep_rounds_are_factored_from_the_macro(self) -> None:
        programs = {program.name: program for program in aarch64_blocks.programs()}
        block = programs["divstep59Block"]
        calls = [node for node in block.nodes if isinstance(node, Call)]
        # A run of consecutive invocations is one call of the round iterated over the run; the
        # third batch's run breaks where the second batch's matrix products are scheduled.
        self.assertEqual(
            [node.head for node in calls],
            ["divstepRound^[19]", "divstepLast"] * 2
            + ["divstepRound^[10]", "divstepRound^[8]", "divstepLast"],
        )
        for name in ("divstepRound", "divstepLast"):
            body = [code for code, _ in lean.body_lines(programs[name]) if code is not None]
            self.assertEqual(
                body[:4],
                [
                    "  let two_delta := st.two_delta",
                    "  let pf := st.f",
                    "  let pg := st.g",
                    "  let fl := st.fl",
                ],
            )
        self.assertEqual(len(lean.body_lines(programs["divstepRound"])), 4 + 9)
        self.assertEqual(len(lean.body_lines(programs["divstepLast"])), 4 + 8)
        # The last step's flags are carried but never read; the block leaves the binding as a comment.
        comments = [comment for code, comment in lean.body_lines(block) if code is None]
        self.assertEqual(len([c for c in comments if c is not None and "fl = step" in c]), 3)

    def test_real_named_and_implicit_outputs_are_extracted(self) -> None:
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
    source: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = x86_64_blocks.SOURCE.read_text()

    def mutate(self, old: str, new: str) -> str:
        self.assertEqual(self.source.count(old), 1, old)
        return self.source.replace(old, new)

    def assert_mutation_rejected(self, old: str, new: str, message: str) -> None:
        source = self.mutate(old, new)
        with self.assertRaisesRegex(GenerationError, message):
            x86_64_blocks.text(source)

    def mutate_function(self, name: str, old: str, new: str) -> str:
        marker = f"pub(super) fn {name}("
        start = self.source.index(marker)
        next_function = self.source.find("pub(super) fn ", start + len(marker))
        end = len(self.source) if next_function < 0 else next_function
        function = self.source[start:end]
        self.assertEqual(function.count(old), 1, old)
        return self.source[:start] + function.replace(old, new) + self.source[end:]

    def assert_function_mutation_rejected(
        self, name: str, old: str, new: str, message: str
    ) -> None:
        source = self.mutate_function(name, old, new)
        with self.assertRaisesRegex(GenerationError, message):
            x86_64_blocks.text(source)

    def flat_mul(self) -> isa.Lifter:
        mul = x86_block("mulMont")
        lifter, _ = driver.lift(driver.parse(self.source, mul, X86), mul, X86)
        return lifter

    def test_all_six_blocks_generate(self) -> None:
        generated = x86_64_blocks.text(self.source)
        for block in x86_64_blocks.BLOCKS:
            self.assertIn(f"def {block.lean_name} ", generated)

    def test_source_output_tuple_order_is_used(self) -> None:
        mul = transcribed(self.source, x86_block("mulMont"))
        square_hi = transcribed(self.source, x86_block("squareHi"))
        self.assertTrue(mul.rstrip().endswith("⟨ee, ae, be, ce⟩"))
        self.assertTrue(square_hi.rstrip().endswith("⟨a, z0, z1, z2⟩"))

    def test_each_unfactored_source_instruction_has_one_generated_comment(self) -> None:
        for block in x86_64_blocks.BLOCKS:
            if block.lean_name == "mulMont":
                continue
            parsed = driver.parse(self.source, block, X86)
            # A block whose template invokes a rerolled macro arm transcribes those invocations
            # as calls of the round, not one comment per instruction.
            if any(
                origin is not None and (origin.macro, origin.arm) in X86.macro_rounds
                for origin in parsed.origins
            ):
                continue
            with self.subTest(routine=block.lean_name):
                generated = transcribed(self.source, block)
                comments = Counter(re.findall(r"-- (.+)$", generated, re.MULTILINE))
                self.assertEqual(
                    Counter(parsed.instructions),
                    Counter(
                        {instruction: comments[instruction] for instruction in parsed.instructions}
                    ),
                )

    def test_factored_mul_rounds_match_under_rotation(self) -> None:
        nodes = self.flat_mul().nodes
        rotations = MUL_ROUNDS.rotations(nodes)
        self.assertEqual(
            {k: v for k, v in rotations[1].items() if k != v},
            {"ce": "r0", "de": "r1", "ee": "r2", "ae": "r3", "be": "r4"},
        )
        rounds = [MUL_ROUNDS.normalized(nodes, i, r) for i, r in enumerate(rotations)]
        self.assertEqual(MUL_ROUNDS.fingerprint(rounds[0]), MUL_ROUNDS.fingerprint(rounds[1]))
        self.assertEqual({node.pc for node in rounds[0]}, set(range(36)))

    def test_factored_mul_round_difference_is_rejected(self) -> None:
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

    def test_factored_round_flattens_back_to_each_source_round(self) -> None:
        nodes = self.flat_mul().nodes
        load, body = MUL_ROUNDS.body(nodes)
        registers = ["be", "ce", "de", "ee", "ae", "rdx"]
        for round_number, (first, last) in enumerate(MUL_ROUNDS.ranges, start=1):
            source = [node for node in nodes if node.pc is not None and first <= node.pc <= last]
            flattened = MUL_ROUNDS.flattened(
                load, body, registers, MUL_ROUNDS.rhs_limb(round_number - 1), first
            )
            self.assertEqual(MUL_ROUNDS.fingerprint(flattened), MUL_ROUNDS.fingerprint(source))
            self.assertEqual({node.pc for node in flattened}, set(range(first, last + 1)))
            registers = registers[1:5] + registers[:1] + ["rdx"]

    def test_factored_round_call_with_misordered_registers_is_rejected(self) -> None:
        nodes = self.flat_mul().nodes
        load, body = MUL_ROUNDS.body(nodes)
        with self.assertRaisesRegex(GenerationError, "does not flatten to the source's round 1"):
            MUL_ROUNDS.check_call(
                nodes,
                load,
                body,
                ["ce", "be", "de", "ee", "ae", "rdx"],
                MUL_ROUNDS.rhs_limb(0),
                0,
            )

    def test_factored_mul_comments_cover_one_validated_full_round(self) -> None:
        block = x86_block("mulMont")
        generated = transcribed(self.source, block)
        comments = Counter(re.findall(r"-- (.+)$", generated, re.MULTILINE))
        parsed = driver.parse(self.source, block, X86)
        omitted = []
        second_first, second_last = MUL_ROUNDS.ranges[1]
        for pc, instruction in enumerate(parsed.instructions):
            if not second_first <= pc <= second_last:
                omitted.append(instruction)
        self.assertEqual(
            Counter(omitted),
            Counter({instruction: comments[instruction] for instruction in omitted}),
        )

    def test_declared_pointer_blocks_are_readonly(self) -> None:
        for block in (x86_block("mulMont"), x86_block("squareHi")):
            with self.subTest(routine=block.lean_name):
                parsed = driver.parse(self.source, block, X86)
                self.assertIn("readonly", parsed.options)
                self.assertNotIn("nomem", parsed.options)

    def test_raw_template_is_rejected(self) -> None:
        self.assert_mutation_rejected(
            '            "add {r0}, {b0}",',
            '            r"add {r0}, {b0}",',
            "raw asm template",
        )

    def test_escaped_template_is_rejected(self) -> None:
        self.assert_mutation_rejected(
            '            "add {r0}, {b0}",',
            '            "add {r0}, {b0}\\n",',
            "escaped asm template",
        )

    def test_instruction_string_in_block_comment_is_ignored(self) -> None:
        source = self.mutate(
            '            "add {r0}, {b0}",',
            '            /* "sub {r0}, {b0}", */\n            "add {r0}, {b0}",',
        )
        self.assertEqual(x86_64_blocks.text(source), x86_64_blocks.text(self.source))

    def test_junk_between_templates_is_rejected(self) -> None:
        self.assert_mutation_rejected(
            '            "add {r0}, {b0}",',
            '            "add {r0}, {b0}",\n            unsupported_token,',
            "unsupported operand binding",
        )

    def test_junk_after_options_is_rejected(self) -> None:
        old = """            p3 = inout(reg) modulus[3] => _,
            z = out(reg) _,
            options(pure, nomem, nostack),
        );"""
        new = old.replace(
            "            options(pure, nomem, nostack),",
            "            options(pure, nomem, nostack),\n            junk,",
        )
        self.assert_function_mutation_rejected("add", old, new, "operand after options")

    def test_second_asm_block_is_rejected(self) -> None:
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

    def test_input_only_operand_write_is_rejected_in_real_source(self) -> None:
        self.assert_mutation_rejected(
            '            "add {r0}, {b0}",',
            '            "add {b0}, {r0}",',
            "input-only register b0",
        )

    def test_input_only_operand_xor_is_rejected_in_real_source(self) -> None:
        self.assert_mutation_rejected(
            '            "add {r0}, {b0}",',
            '            "xor {b0:e}, {b0:e}",',
            "input-only register b0",
        )

    def test_literal_rdx_without_fixed_operand_is_rejected(self) -> None:
        old = """            t1 = out(reg) _,
            t2 = out(reg) _,
            out("rdx") _,
            options(pure, nomem, nostack),"""
        new = old.replace('            out("rdx") _,\n', "")
        self.assert_mutation_rejected(old, new, "literal rdx requires exactly one fixed")

    def test_named_rdx_cannot_spoof_fixed_operand(self) -> None:
        old = """            t1 = out(reg) _,
            t2 = out(reg) _,
            out("rdx") _,
            options(pure, nomem, nostack),"""
        new = old.replace(
            '            out("rdx") _,',
            "            rdx = out(reg) _,",
        )
        self.assert_mutation_rejected(old, new, "reserved operand name rdx")

    def test_bound_local_use_before_asm_is_rejected(self) -> None:
        self.assert_function_mutation_rejected(
            "add",
            "    let [mut r0, mut r1, mut r2, mut r3] = *lhs;\n",
            "    let [mut r0, mut r1, mut r2, mut r3] = *lhs;\n    r0 = 0;\n",
            "unsupported use of bound local r0",
        )

    def test_x86_skeleton_extracts_every_retained_instruction_let(self) -> None:
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
        helper_text = "\n".join(code for code, _ in lean.body_lines(mul_round) if code is not None)
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

    def test_all_x86_routines_and_factored_round_generate_shared_skeletons(self) -> None:
        programs = x86_64_blocks.programs()
        self.assertEqual(
            [program.name for program in programs],
            [
                "addMod",
                "subMod",
                "mulMontRound",
                "mulMont",
                "squareLo",
                "squareHi",
                "fromMont",
                "divstepRound",
                "divstep59Block",
                "signMagBlock",
                "fgRowBlock",
                "deRowBlock",
                "amontredBlock",
                "condSubBlock",
            ],
        )
        for program in programs:
            with self.subTest(routine=program.name):
                generated = skeleton(program)
                self.assertEqual(
                    generated[0],
                    f"  -- generated skeleton for `{program.name}`: do not edit between the annotations",
                )
                self.assertEqual(generated[-1], "  subst hres")
                self.assertEqual(gen.find_routine(f"X86_64:{program.name}").architecture, "X86_64")


class TranslationValidationTests(unittest.TestCase):
    """The rerolled blocks are checked to compute what the flat blocks compute."""

    def test_every_rerolled_block_is_validated(self) -> None:
        checked: list[tuple[str, str]] = []
        real = validate.check

        def spy(flat: ir.Program, programs: list[ir.Program]) -> None:
            checked.append((programs[-1].architecture, programs[-1].name))
            return real(flat, programs)

        validate.check = spy
        try:
            aarch64_blocks.programs()
            x86_64_blocks.programs()
        finally:
            validate.check = real
        self.assertEqual(
            sorted(checked),
            [
                ("AArch64", "divstep59Block"),
                ("AArch64", "mulMont"),
                ("X86_64", "divstep59Block"),
                ("X86_64", "mulMont"),
            ],
        )

    def test_a_stale_read_after_a_run_is_rejected(self) -> None:
        # `k` is written by every step of the run but not carried out of it: read after the run,
        # the rerolled block would see the `k` from before the run.
        source = x86_64_blocks.SOURCE.read_text()
        arm = source.index("(last) => {", source.index("macro_rules! divstep"))
        at = source.index("concat!(\n", arm) + len("concat!(\n")
        mutated = source[:at] + '            "add {td}, {k}\\n",\n' + source[at:]
        with self.assertRaisesRegex(GenerationError, "does not compute the flat block"):
            x86_64_blocks.programs(mutated)

    def test_projections_of_different_pairs_are_different_terms(self) -> None:
        def program(b: str) -> ir.Program:
            nodes: list[Node] = [
                ir.Scalar(comment="argument", dest="x", name="x"),
                ir.Scalar(comment="argument", dest=b, name=b),
                x86_64.Mulx(comment="mulx", high="h", low="l", a=ir.Reg("x"), b=ir.Reg(b)),
            ]
            return CONVENTIONS.program(
                "X86_64",
                ir.Signature("p", (("x", "Nat"), (b, "Nat")), "Nat"),
                "",
                nodes,
                ["h", "l"],
            )

        evaluator = validate.Evaluator({})
        self.assertNotEqual(evaluator.run(program("y")), evaluator.run(program("z")))
        self.assertEqual(evaluator.run(program("y")), evaluator.run(program("y")))


class SoundnessRegressionTests(unittest.TestCase):
    """Source and configuration edits the generator once accepted while emitting Lean that does
    not model the assembly; each must now fail."""

    @staticmethod
    def aarch64_mutated(function: str, *edits: tuple[str, str]) -> str:
        source = aarch64_blocks.SOURCE.read_text()
        start = source.index(f"pub(super) fn {function}(")
        end = source.index("\n}\n", start)
        body = source[start:end]
        for old, new in edits:
            assert old in body, old
            body = body.replace(old, new)
        return source[:start] + body + source[end:]

    @staticmethod
    def aarch64_block(rust_name: str) -> driver.Block:
        return next(b for b in aarch64_blocks.BLOCKS if b.rust_name == rust_name)

    def test_a_binding_cannot_capture_a_round_parameter(self) -> None:
        # The operand renamed `b` is bound in `mulMontRound` before its parameter `b` is read:
        # Lean would read the operand where the round means its `rhs` limb.
        source = self.aarch64_mutated(
            "mul", ("{a0}", "{b}"), ("a0 = in(reg) lhs[0]", "b = in(reg) lhs[0]")
        )
        with self.assertRaisesRegex(GenerationError, "captures"):
            driver.transcribe(source, self.aarch64_block("mul"), aarch64_blocks.TARGET)

    def test_hygiene_allows_an_argument_bound_to_itself(self) -> None:
        nodes = [
            ir.Scalar(comment="argument", dest="inv", name="inv"),
            ir.Word(comment="use", dest="x", function="addw", args=(ir.Reg("inv"), ir.Reg("inv"))),
        ]
        CONVENTIONS.program("Test", ir.Signature("p", (("inv", "Nat"),), "Nat"), "", nodes, ["x"])

    def test_an_operand_cannot_be_named_xzr(self) -> None:
        source = self.aarch64_mutated(
            "add", ("{b0}", "{xzr}"), ("b0 = in(reg) rhs[0]", "xzr = in(reg) rhs[0]")
        )
        with self.assertRaisesRegex(GenerationError, "reserved operand name"):
            driver.transcribe(source, self.aarch64_block("add"), aarch64_blocks.TARGET)

    def test_a_bare_register_is_not_an_operand_of_the_same_name(self) -> None:
        # rustc reads a bare `x1` as the hardware register, not as the operand named `x1`.
        source = self.aarch64_mutated(
            "add", ("{r1}, {r1}, {b1}", "{r1}, {r1}, x1"), ("b1 = in(reg)", "x1 = in(reg)")
        )
        with self.assertRaisesRegex(GenerationError, "bare register x1"):
            driver.transcribe(source, self.aarch64_block("add"), aarch64_blocks.TARGET)

    def test_a_placeholder_is_not_a_condition_of_the_same_name(self) -> None:
        emitter = aarch64.Lifter({"a": "inout", "b": "inout", "ne": "in"})
        bound_registers(emitter, ["a", "b", "ne"])
        emitter.lift("tst {a}, #1")
        with self.assertRaisesRegex(GenerationError, "unexpected condition"):
            emitter.lift("csel {a}, {a}, {b}, {ne}")

    def test_round_fields_follow_the_declared_structure(self) -> None:
        last = aarch64_blocks.MACRO_ROUNDS[("divstep", "last")]
        swapped = dataclasses.replace(
            last,
            roles=(
                reroll.Role("two_delta", "two_delta", None),
                reroll.Role("pg", "g", None),
                reroll.Role("pf", "f", None),
                reroll.Role("fl", "fl", None),
            ),
        )
        target = dataclasses.replace(
            aarch64_blocks.TARGET,
            macro_rounds={**aarch64_blocks.MACRO_ROUNDS, ("divstep", "last"): swapped},
        )
        with self.assertRaisesRegex(GenerationError, "divstepRound declares"):
            driver.transcribe(
                aarch64_blocks.SOURCE.read_text(), self.aarch64_block("divstep59"), target
            )

    def test_a_rotated_round_structure_is_generated_from_its_roles(self) -> None:
        # The structure's field order and the order a call binds have one source, `roles`.
        rounds = x86_64_blocks.MUL_ROUNDS
        declared = re.findall(r"^  (\w+) : Nat$", rounds.struct, re.MULTILINE)
        self.assertEqual(tuple(declared), rounds.fields)
        program = next(p for p in x86_64_blocks.programs() if p.name == rounds.round)
        self.assertEqual(program.state_fields, rounds.fields)

    def test_aarch64_immediates_are_checked(self) -> None:
        for line, message in (
            ("add {d}, {a}, #-1", "negative immediate"),
            ("mov {d}, #-1", "negative immediate"),
            ("lsl {d}, {a}, #64", "shift 64"),
            ("asr {d}, {a}, #0", "shift 0"),
            ("extr {d}, {a}, {a}, #64", "extr lsb 64"),
            ("sbfx {d}, {a}, #60, #5", "sbfx width 5"),
            ("mov {d}, #0x10000000000000000", "does not fit a word"),
            ("add {d}, {a}, {a}, lsl #64", "shift 64"),
            # Only `subs xzr` (a comparison) has a node; the others need the value they drop.
            ("adds xzr, {a}, {a}", "adds to xzr is unsupported"),
        ):
            with self.subTest(line=line):
                emitter = aarch64.Lifter({"a": "in", "d": "out"})
                bound_registers(emitter, ["a"])
                with self.assertRaisesRegex(GenerationError, message):
                    emitter.lift(line)

    def test_a_literal_is_a_word(self) -> None:
        self.assertEqual(ir.lit(0), ir.Lit("0"))
        self.assertEqual(ir.lit(2**64 - 1), ir.Lit(str(2**64 - 1)))
        with self.assertRaisesRegex(GenerationError, "negative literal"):
            ir.lit(-1)
        with self.assertRaisesRegex(GenerationError, "does not fit a word"):
            ir.lit(2**64)

    def test_a_ccmp_mask_is_four_bits(self) -> None:
        emitter = aarch64.Lifter({"a": "in"})
        bound_registers(emitter, ["a"])
        emitter.lift("tst {a}, {a}")
        with self.assertRaisesRegex(GenerationError, "ccmp nzcv 16"):
            emitter.lift("ccmp {a}, {a}, #16, ne")


class RejectionPathTests(unittest.TestCase):
    """Each check that rejects a block, exercised once, so that deleting it fails a test."""

    PROBE = """
pub(super) fn probe(lhs: &Limbs, rhs: &Limbs) -> Limbs {
    let [mut r0, mut r1, mut r2, mut r3] = *lhs;
    // SAFETY: a test block.
    unsafe {
        asm!(
            INSTRUCTIONS
            r0 = inout(reg) r0,
            r1 = inout(reg) r1,
            r2 = inout(reg) r2,
            R3
            EXTRA
            options(OPTIONS),
        );
    }
    [r0, r1, r2, r3]
}
"""

    def probe(
        self,
        instructions: str = '"add {r0}, {r1}",',
        options: str = "pure, nomem, nostack",
        extra: str = "",
        r3: str = "r3 = inout(reg) r3,",
    ) -> list[ir.Program]:
        source = (
            self.PROBE.replace("INSTRUCTIONS", instructions)
            .replace("OPTIONS", options)
            .replace("EXTRA", extra)
            .replace("R3", r3)
        )
        block = driver.Block(
            "probe", "probe", "A test block.", (("lhs", "Limbs"), ("rhs", "Limbs"))
        )
        return driver.transcribe(source, block, X86)

    def test_the_probe_itself_transcribes(self) -> None:
        (program,) = self.probe()
        self.assertEqual(program.results, ["r0", "r1", "r2", "r3"])

    def test_pointer_operands_require_readonly(self) -> None:
        with self.assertRaisesRegex(GenerationError, "pointer operands require readonly"):
            self.probe(
                instructions='"mov {r0}, qword ptr [{p}]",',
                extra="p = in(reg) rhs.as_ptr(),",
            )

    def test_a_register_only_block_requires_nomem(self) -> None:
        with self.assertRaisesRegex(GenerationError, "register-only block requires nomem"):
            self.probe(options="pure, readonly, nostack")

    def test_a_pointer_must_be_read(self) -> None:
        with self.assertRaisesRegex(GenerationError, "pointer operands and memory reads disagree"):
            self.probe(options="pure, readonly, nostack", extra="p = in(reg) rhs.as_ptr(),")

    def test_a_returned_output_must_be_written(self) -> None:
        with self.assertRaisesRegex(GenerationError, "output register r3 was never written"):
            self.probe(r3="r3 = out(reg) r3,")

    def test_an_operand_is_bound_once(self) -> None:
        lifter = x86_64.Lifter({"a": "in"})
        lifter.bind(Mov(comment="argument", dest="a", a=ir.lit(0)))
        with self.assertRaisesRegex(GenerationError, "duplicate initial binding for a"):
            lifter.bind(Mov(comment="argument", dest="a", a=ir.lit(1)))

    def test_prologue_loop_rounds_must_agree(self) -> None:
        # With every round's `rhs` register named `b1`, round 2's `b2` is a difference.
        rounds = dataclasses.replace(aarch64_blocks.MUL_ROUNDS, operand="b1")
        target = dataclasses.replace(aarch64_blocks.TARGET, rerollers={"mulMont": rounds})
        mul = next(b for b in aarch64_blocks.BLOCKS if b.rust_name == "mul")
        with self.assertRaisesRegex(GenerationError, "round bodies differ"):
            driver.transcribe(aarch64_blocks.SOURCE.read_text(), mul, target)

    def divstep(self, roles: tuple[reroll.Role, ...]) -> None:
        body = dataclasses.replace(aarch64_blocks.MACRO_ROUNDS[("divstep", "")], roles=roles)
        target = dataclasses.replace(
            aarch64_blocks.TARGET,
            macro_rounds={**aarch64_blocks.MACRO_ROUNDS, ("divstep", ""): body},
        )
        block = next(b for b in aarch64_blocks.BLOCKS if b.rust_name == "divstep59")
        driver.transcribe(aarch64_blocks.SOURCE.read_text(), block, target)

    def test_a_round_reads_only_what_it_carries(self) -> None:
        roles = aarch64_blocks.MACRO_ROUNDS[("divstep", "")].roles
        with self.assertRaisesRegex(
            GenerationError, "reads \\['pf'\\], which `roles` does not carry"
        ):
            self.divstep(tuple(r for r in roles if r.register != "pf"))

    def test_a_round_uses_every_role(self) -> None:
        roles = aarch64_blocks.MACRO_ROUNDS[("divstep", "")].roles
        with self.assertRaisesRegex(GenerationError, "neither reads nor writes \\['zz'\\]"):
            self.divstep((*roles, reroll.Role("zz", "zz", "unused")))

    def test_a_rotated_round_begins_with_its_rhs_load(self) -> None:
        rounds = dataclasses.replace(x86_64_blocks.MUL_ROUNDS, load_register="rax")
        target = dataclasses.replace(X86, rerollers={"mulMont": rounds})
        with self.assertRaisesRegex(GenerationError, "does not begin by loading its RHS limb"):
            driver.transcribe(x86_64_blocks.SOURCE.read_text(), x86_block("mulMont"), target)

    def test_a_rotated_round_loads_its_rhs_limb_not_another_value(self) -> None:
        # Each round begins by loading `rdx`, but from the register `q`, not from its `rhs` limb.
        def round_(pc: int) -> list[Node]:
            return [
                Mov(comment="mov rdx, q", pc=pc, dest="rdx", a=ir.Reg("q")),
                ir.Word(
                    comment="add",
                    pc=pc + 1,
                    dest="x",
                    function="addw",
                    args=(ir.Reg("rdx"), ir.Reg("x")),
                ),
            ]

        rounds = dataclasses.replace(x86_64_blocks.MUL_ROUNDS, ranges=((0, 1), (2, 3)))
        with self.assertRaisesRegex(GenerationError, "does not begin by loading its RHS limb"):
            rounds.body(round_(0) + round_(2))

    def test_matched_rounds_keep_their_relative_positions(self) -> None:
        source = AlphaMatchTests.adds(("x", "x", "y"), ("y", "y", "x"))
        target = [dataclasses.replace(n, pc=2 * (n.pc or 0)) for n in source]
        with self.assertRaisesRegex(GenerationError, "different operations"):
            reroll.alpha_match(source, target)

    def test_a_register_is_not_split_by_the_renaming(self) -> None:
        # `x` would have to become both `p` and `q`.
        source = AlphaMatchTests.adds(("x", "x", "x"))
        target = AlphaMatchTests.adds(("p", "p", "q"))
        with self.assertRaisesRegex(GenerationError, "break the renaming"):
            reroll.alpha_match(source, target)

    def test_the_two_components_of_a_pair_are_different_terms(self) -> None:
        nodes = [
            ir.Scalar(comment="argument", dest="x", name="x"),
            ir.Scalar(comment="argument", dest="y", name="y"),
            x86_64.Mulx(comment="mulx", high="h", low="l", a=ir.Reg("x"), b=ir.Reg("y")),
        ]
        program = CONVENTIONS.program(
            "X86_64", ir.Signature("p", (("x", "Nat"), ("y", "Nat")), "Nat"), "", nodes, ["h", "l"]
        )
        high, low = validate.Evaluator({}).run(program)
        self.assertNotEqual(high, low)


class AlphaMatchTests(unittest.TestCase):
    """The renaming under which one round is another is found, and must be a bijection."""

    @staticmethod
    def adds(*pairs: tuple[str, str, str]) -> list[Node]:
        return [
            ir.Word(comment=f"add {d}", pc=pc, dest=d, function="addw", args=(word(a), word(b)))
            for pc, (d, a, b) in enumerate(pairs)
        ]

    def test_a_renaming_is_found(self) -> None:
        source = self.adds(("x", "x", "y"), ("y", "y", "x"))
        target = self.adds(("p", "p", "q"), ("q", "q", "p"))
        self.assertEqual(reroll.alpha_match(source, target), {"x": "p", "y": "q"})

    def test_two_registers_are_not_merged(self) -> None:
        # `x` and `y` would both have to become `p`: not a renaming, so not the same round.
        source = self.adds(("x", "x", "y"))
        target = self.adds(("p", "p", "p"))
        with self.assertRaisesRegex(GenerationError, "break the renaming"):
            reroll.alpha_match(source, target)

    def test_literals_must_agree(self) -> None:
        with self.assertRaisesRegex(GenerationError, "differ"):
            reroll.alpha_match(self.adds(("x", "x", "1")), self.adds(("x", "x", "2")))


# -- a toy RISC-V-shaped instruction set, defined entirely here -----------------------------------
#
# No flags: an addition's carry is a separate `sltu c, s, b` (`c = s < b`, set exactly when
# `s = a + b` wrapped). Its proof fact `s + 2^64 * c = a + b` spans two instructions, so the
# addition records its pending sum under the toy's own `ScratchKey`, which `sltu` consumes:
# nothing shared changes. The Lean names (`sltu`, `add_sltu_carry`) are the toy's.

PENDING_SUMS: ir.ScratchKey[dict[str, tuple[str, str]]] = ir.ScratchKey("rv-sums", dict)


@dataclasses.dataclass(frozen=True, kw_only=True)
class RvAdd(ir.Word):
    """`add d, a, b`: `addw a b`, recording the sum for a later `sltu`."""

    def prove(self, step: Step) -> None:
        super().prove(step)
        a, b = (step.r(x) for x in self.args if isinstance(x, ir.Reg))
        step.state.scratch(PENDING_SUMS)[step.name] = (a, b)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Sltu(Node):
    """`sltu c, s, b`: `c = s < b`, the carry of the sum `s` when `s = a + b`."""

    dest: str = ir.dst()
    s: ir.Term = ir.src()
    b: ir.Term = ir.src()

    def lets(self) -> list[ir.Let]:
        return [self._let(self.dest, ir.apply("sltu", self.s, self.b))]

    def prove(self, step: Step) -> None:
        s, b, c = step.r(self.s), step.r(self.b), step.name
        step.eq(c, f"sltu {s} {b}")
        step.line(f"  have b_{c} : {c} ≤ 1 := by rw [e_{c}]; exact sltu_le_one {s} {b}")
        step.unit(c)
        summed = step.state.scratch(PENDING_SUMS).get(s)
        if summed is not None and summed[1] == b:
            a = summed[0]
            step.line(f"  have l_{s} : {s} + 2^64 * {c} = {a} + {b} := by")
            step.line(f"    exact add_sltu_carry e_{s} e_{c}")


class RvLifter(isa.Lifter):
    @staticmethod
    def parse_instruction(text: str) -> isa.Instruction:
        op, rest = text.split(" ", 1)
        return isa.Instruction(op, tuple(t.strip() for t in rest.split(",")), text)

    def operand(self, token: str) -> str:
        if not (token.startswith("{") and token.endswith("}")):
            raise GenerationError(f"bare register {token}")
        return token[1:-1]

    def lift_instruction(self, ins: isa.Instruction) -> None:
        if ins.op not in ("add", "xor", "sltu"):
            raise GenerationError(f"unsupported instruction {ins.op}")
        dest, x, y = (self.operand(t) for t in ins.operands)
        a, b = ir.Reg(self.require_written(x)), ir.Reg(self.require_written(y))
        self.check_writable(dest)
        if ins.op == "add":
            self.emit(RvAdd(dest=dest, function="addw", args=(a, b)))
        elif ins.op == "xor":
            self.emit(ir.Word(dest=dest, function="eorw", args=(a, b), bounds=(0, 1)))
        else:
            self.emit(Sltu(dest=dest, s=a, b=b))


class RiscV(isa.Isa):
    name = "RV64"
    lifter = RvLifter
    required_options = frozenset({"pure", "nomem", "nostack"})


class NewInstructionSetTests(unittest.TestCase):
    """A new instruction set is its syntax, a lifter, and its semantics: here the toy RISC-V
    above, transcribed and given its skeleton by the shared pipeline, with a carry fact spanning
    two instructions."""

    SOURCE = """
pub(super) fn mix(lhs: &Limbs, rhs: &Limbs) -> Limbs {
    let [mut r0, mut r1, mut r2, mut r3] = *lhs;
    // SAFETY: register-only arithmetic with declared inputs and outputs.
    unsafe {
        asm!(
            "add {r0}, {r0}, {b0}",
            "sltu {c}, {r0}, {b0}",
            "add {r1}, {r1}, {b1}",
            "add {r1}, {r1}, {c}",
            "xor {r2}, {r2}, {b1}",
            r0 = inout(reg) r0,
            r1 = inout(reg) r1,
            r2 = inout(reg) r2,
            r3 = inout(reg) r3,
            b0 = in(reg) rhs[0],
            b1 = in(reg) rhs[1],
            c = out(reg) _,
            options(pure, nomem, nostack),
        );
    }
    [r0, r1, r2, r3]
}
"""

    BLOCK = driver.Block("mix", "mixRv", "A toy block.", (("lhs", "Limbs"), ("rhs", "Limbs")))

    @staticmethod
    def target() -> driver.Target:
        return driver.Target(RiscV(), aarch64_blocks.KINDS, CONVENTIONS)

    def test_a_toy_instruction_set_is_transcribed_and_proved_by_the_shared_pipeline(self) -> None:
        (program,) = driver.transcribe(self.SOURCE, self.BLOCK, self.target())
        text = lean.definition(program, 32)
        self.assertIn("def mixRv (lhs rhs : Limbs) : Limbs :=", text)
        self.assertIn("  let r0 := addw r0 b0", text)
        self.assertIn("  let c := sltu r0 b0", text)
        self.assertIn("  let r2 := eorw r2 b1", text)
        steps = "\n".join(skeleton(program))
        self.assertIn("word_step r0_1 := addw r0 b0 using addw_lt r0 b0", steps)
        # The carry fact joins the `add` and the `sltu` that follows it.
        self.assertIn("have l_r0_1 : r0_1 + 2^64 * c = r0 + b0 := by", steps)

    def test_its_checks_and_the_shared_ones_both_apply(self) -> None:
        with self.assertRaisesRegex(GenerationError, "unsupported instruction sub"):
            driver.transcribe(self.SOURCE.replace('"xor ', '"sub '), self.BLOCK, self.target())
        with self.assertRaisesRegex(GenerationError, "bare register x1"):
            driver.transcribe(
                self.SOURCE.replace("{r2}, {b1}", "{r2}, x1"), self.BLOCK, self.target()
            )
        # The shared front end requires the options the instruction set declares.
        with self.assertRaisesRegex(GenerationError, "requires options"):
            driver.transcribe(
                self.SOURCE.replace("pure, nomem,", "pure,"), self.BLOCK, self.target()
            )


class SharedGeneratorTests(unittest.TestCase):
    def test_skeleton_hooks_are_backend_owned(self) -> None:
        # Each architecture's proof steps belong to its own nodes: a program holds only the
        # shared nodes of `ir` and its own architecture's.
        def modules(programs: list[ir.Program]) -> set[str]:
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

    def test_skeleton_facts_come_from_the_routine_backend(self) -> None:
        # The shared traversal knows only how to walk; each node states its own step, and a node
        # that states none is an error.
        import dataclasses

        @dataclasses.dataclass(frozen=True, kw_only=True)
        class Stub(Node):
            dest: str = ir.dst()
            a: ir.Term = ir.src()

            def lets(self) -> list[ir.Let]:
                return [self._let(self.dest, ir.apply("stub", self.a))]

            def prove(self, step: Step) -> None:
                step.eq(step.name, f"stub {step.r(self.a)}")
                step.line(f"  have b_{step.name} : {step.name} < 2^64 := by sorry")
                step.state.bnd[step.name] = f"b_{step.name}"

        @dataclasses.dataclass(frozen=True, kw_only=True)
        class Unproved(Stub):
            def prove(self, step: Step) -> None:
                return Node.prove(self, step)

        def program(stub: type[Stub]) -> ir.Program:
            nodes: list[Node] = [
                Load(comment="argument", dest="x", arg="lhs", field="l0"),
                stub(comment="stub x", dest="y", a=ir.Reg("x")),
            ]
            return CONVENTIONS.program(
                "Stub",
                ir.Signature("stub", (("lhs", "Limbs"),), "Nat"),
                "doc",
                nodes,
                ["y"],
                arg_fields={"lhs": ("l0", "l1", "l2", "l3")},
            )

        text = "\n".join(skeleton(program(Stub)))
        self.assertIn("  word_step x := lhs.l0 using hlhs.1\n", text)
        self.assertIn("  word_step y := stub x\n", text)
        self.assertIn("have b_y : y < 2^64 := by sorry", text)
        with self.assertRaisesRegex(ValueError, "unsupported skeleton fact Unproved"):
            skeleton(program(Unproved))

    def test_backward_liveness_and_backend_retention_share_the_same_ir(self) -> None:
        nodes = [
            Mov(comment="argument", dest="input", a=ir.Get(ir.Arg("lhs"), "l0")),
            Mov(comment="dead", dest="dead", a=ir.Reg("input")),
            Mov(comment="result", dest="result", a=ir.Reg("input")),
        ]
        eliminated = CONVENTIONS.program(
            "Test", ir.Signature("t", (("lhs", "Limbs"),), "Nat"), "doc", nodes, ["result"]
        )
        retained = CONVENTIONS.program(
            "Test",
            ir.Signature("t", (("lhs", "Limbs"),), "Nat"),
            "doc",
            nodes,
            ["result"],
            dead_code=ir.DeadCode.RETAIN,
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

    def test_transcription_snapshots_match_committed_files(self) -> None:
        self.assertEqual(aarch64_blocks.text(), aarch64_blocks.OUTPUT.read_text())
        self.assertEqual(x86_64_blocks.text(), x86_64_blocks.OUTPUT.read_text())

    def test_round_argument_fields_are_routine_local(self) -> None:
        aarch_round = next(p for p in aarch64_blocks.programs() if p.name == "mulMontRound")
        x86_round = next(p for p in x86_64_blocks.programs() if p.name == "mulMontRound")
        self.assertEqual(
            aarch_round.arg_fields["acc"], ("r0", "r1", "r2", "r3", "r4", "q", "t1", "t3")
        )
        self.assertEqual(x86_round.arg_fields["acc"], ("r0", "r1", "r2", "r3", "r4", "q"))
        self.assertEqual(projection(aarch_round.arg_fields["acc"], "q"), "2.2.2.2.2.1")
        self.assertEqual(projection(x86_round.arg_fields["acc"], "q"), "2.2.2.2.2")
        self.assertEqual(projection(aarch_round.arg_fields["acc"], "r4"), "2.2.2.2.1")
        self.assertEqual(projection(x86_round.arg_fields["acc"], "r4"), "2.2.2.2.1")

    def test_skeleton_lookup_names_the_architecture(self) -> None:
        self.assertEqual(gen.find_routine("AArch64:addMod").architecture, "AArch64")
        self.assertEqual(gen.find_routine("X86_64:addMod").architecture, "X86_64")
        # Both backends have an `addMod`, so a bare name would be ambiguous.
        with self.assertRaisesRegex(ValueError, "expected ARCH:NAME, not addMod"):
            gen.find_routine("addMod")


class SkeletonCheckerTests(unittest.TestCase):
    routine: ir.Program
    skeleton: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.routine = gen.find_routine("X86_64:addMod")
        cls.skeleton = "\n".join(skeleton(cls.routine)) + "\n"

    def check_text(self, text: str) -> tuple[bool, str]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Spec.lean"
            path.write_text(text)
            diagnostics = io.StringIO()
            with contextlib.redirect_stderr(diagnostics):
                ok = check_spec(path, [self.routine])
            return ok, diagnostics.getvalue()

    def test_handwritten_annotation_blocks_are_ignored(self) -> None:
        ok, diagnostics = self.check_text(
            "-- BEGIN handwritten proof\nanything may appear here\n-- END handwritten proof\n"
            + self.skeleton
        )
        self.assertTrue(ok, diagnostics)

    def test_missing_and_duplicate_skeletons_are_rejected(self) -> None:
        for text, message in (
            ("", "expected one skeleton of addMod, found 0"),
            (self.skeleton + self.skeleton, "expected one skeleton of addMod, found 2"),
        ):
            with self.subTest(message=message):
                ok, diagnostics = self.check_text(text)
                self.assertFalse(ok)
                self.assertIn(message, diagnostics)

    def test_malformed_annotations_are_rejected(self) -> None:
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

    def test_unreadable_spec_reports_controlled_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            diagnostics = io.StringIO()
            with contextlib.redirect_stderr(diagnostics):
                self.assertFalse(check_spec(Path(directory) / "missing.lean", []))
            self.assertIn("cannot read proof file", diagnostics.getvalue())

    def test_manifest_selects_existing_files_and_rejects_mismatched(self) -> None:
        add_path = ROOT / "lean/PastaCurves/X86_64/Spec/Add.lean"
        available = gen.architecture_routines()
        for name, (arch, expected_names) in SPEC_MANIFEST.items():
            with self.subTest(path=name):
                manifest_path = ROOT / name
                if Path(manifest_path).exists():
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

    def test_literals_are_read_in_order_and_references_skipped(self) -> None:
        self.assertEqual(data.parse_known_answers(self.SOURCE), [("FP", "two_r", [1, 2, 3, 4])])

    def test_a_literal_without_a_definition_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "no known-answer definition"):
            data.parse_known_answers(self.SOURCE.replace("two_r", "five_r"))

    def test_an_unknown_field_constant_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown field constant"):
            data.parse_known_answers(self.SOURCE.replace("const FP", "const FR"))

    def test_the_tests_have_literals(self) -> None:
        with self.assertRaisesRegex(ValueError, "no known-answer literals"):
            data.parse_known_answers("")

    def test_each_literal_becomes_one_kernel_checked_example(self) -> None:
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

    def test_pairs_are_read_in_order(self) -> None:
        self.assertEqual(
            data.parse_inversions(self.SOURCE), [("FP", 0, [1, 2, 3, 4], [5, 6, 7, 8])]
        )

    def test_a_pair_that_is_not_two_four_limb_values_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not two four-limb values"):
            data.parse_inversions(self.SOURCE.replace("0x3, 0x4]", "0x3]"))

    def test_each_pair_becomes_one_kernel_checked_example(self) -> None:
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
pub(crate) const E0: Fp = Fp([0x12, 0x13, 0x14, 0x15]);
"""

    def test_constants_are_read_as_the_source_spells_them(self) -> None:
        modulus, inv, powers = data.parse_field_type(self.SOURCE, "fp.rs")
        self.assertEqual(modulus, ["0x1", "0x2", "0x0", "0x4"])
        self.assertEqual(inv, "0x5")
        self.assertEqual(powers["R2"], ["0xa", "0xb", "0xc", "0xd"])

    def test_a_missing_constant_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing R3"):
            data.parse_field_type(self.SOURCE.replace("const R3", "const S3"), "fp.rs")
        with self.assertRaisesRegex(ValueError, "missing E0"):
            data.parse_field_type(self.SOURCE.replace("const E0", "const W0"), "fp.rs")
        with self.assertRaisesRegex(ValueError, "missing INV"):
            data.parse_field_type(self.SOURCE.replace("const INV", "const NV"), "fp.rs")

    def test_each_constant_becomes_one_example(self) -> None:
        rendered = data.render_field_types({"fp": self.SOURCE, "fq": self.SOURCE})
        self.assertEqual(rendered.count("decide +kernel"), 8)
        self.assertEqual(rendered.count("  decide\n"), 4)
        self.assertIn("example : pallasBase.modulus =\n    ⟨0x1, 0x2, 0x0, 0x4⟩ := by\n", rendered)
        self.assertIn("example : vestaBase.inv = 0x5 := by\n", rendered)
        self.assertIn("      ⟨0xa, 0xb, 0xc, 0xd⟩\n    x = R^2 % p := by\n", rendered)


# -- the laws of renaming, and the rejections the reviews found untested --------------------------


def operand_registers(nodes: Sequence[Node]) -> set[str]:
    """The registers that the nodes' operand fields name: what `alpha_match` pairs up."""
    found: set[str] = set()
    for node in nodes:
        for f in dataclasses.fields(node):
            if not f.metadata.get("operand"):
                continue
            value = getattr(node, f.name)
            for item in value if isinstance(value, tuple) else (value,):
                found |= {item} if isinstance(item, str) else ir.term_registers(item)
    return found


def flat_and_rerolled() -> list[tuple[ir.Program, list[ir.Program]]]:
    """Each rerolled block's flat program and its rerolled programs, as the validator sees them."""
    captured: list[tuple[ir.Program, list[ir.Program]]] = []
    real = validate.check

    def spy(flat: ir.Program, programs: list[ir.Program]) -> None:
        captured.append((flat, programs))
        real(flat, programs)

    validate.check = spy
    try:
        aarch64_blocks.programs()
        x86_64_blocks.programs()
    finally:
        validate.check = real
    return captured


class RenamingLawTests(unittest.TestCase):
    """`alpha_match` is the inverse of renaming: matching a list with its image under a renaming
    `m` gives back `m`, matching the other way gives its inverse, and renaming by the match turns
    the source into the target. Checked on every block of both instruction sets, under random
    renamings that fix the names nodes write implicitly."""

    flats: list[ir.Program]

    @classmethod
    def setUpClass(cls) -> None:
        cls.flats = [flat for flat, _ in flat_and_rerolled()]

    def test_matching_a_renamed_list_recovers_the_renaming(self) -> None:
        rng = random.Random(0)
        for flat in self.flats:
            nodes = [n for n in flat.nodes if n.pc is not None]
            rigid = frozenset[str]().union(*(n.implicit_writes() for n in nodes))
            names = operand_registers(nodes)
            free = sorted(names - rigid)
            for _ in range(3):
                image = free[:]
                rng.shuffle(image)
                m = {**dict(zip(free, image)), **{r: r for r in names & rigid}}
                renamed = [n.renamed(m) for n in nodes]
                with self.subTest(block=f"{flat.architecture}:{flat.name}"):
                    found = reroll.alpha_match(nodes, renamed)
                    self.assertEqual(found, m)
                    self.assertEqual(
                        reroll.alpha_match(renamed, nodes), {b: a for a, b in found.items()}
                    )
                    self.assertEqual(
                        [n.renamed(found).key() for n in nodes], [n.key() for n in renamed]
                    )

    def test_renaming_a_term_renames_its_registers(self) -> None:
        rename = {"x": "p", "y": "q"}.get
        term = ir.apply("f", ir.Reg("x"), ir.Get(ir.Reg("y"), "1"), ir.Arg("x"), 7)
        renamed = ir.rename_term(term, lambda r: rename(r) or r)
        self.assertEqual(ir.term_registers(renamed), {"p", "q"})
        # An argument is not a register: `Arg("x")` is untouched.
        self.assertEqual(ir.term_args(renamed), {"x"})

    def test_a_name_written_implicitly_is_matched_only_by_itself(self) -> None:
        # `SubFlag` writes the carry `c` without a field naming it. Pairing `c` with `k` would
        # make the second `adc` read an outer `k` instead of that carry.
        def nodes(carry: str, first: int) -> list[Node]:
            x, y = ir.Reg("x"), ir.Reg("y")
            return [
                aarch64.SubFlag(pc=first, a=x, b=y, cin=ir.Reg("c0")),
                aarch64.AddTrunc(pc=first + 1, dest="z", a=x, b=y, cin=ir.Reg(carry)),
            ]

        self.assertEqual(reroll.alpha_match(nodes("c", 0), nodes("c", 5))["c"], "c")
        with self.assertRaisesRegex(GenerationError, "c is fixed"):
            reroll.alpha_match(nodes("c", 0), nodes("k", 5))

    def test_operand_counts_must_agree(self) -> None:
        def word(*args: str) -> list[Node]:
            return [ir.Word(pc=0, dest="d", function="f", args=tuple(ir.Reg(a) for a in args))]

        with self.assertRaisesRegex(GenerationError, "different operand counts"):
            reroll.alpha_match(word("a", "b"), word("a"))


class ValidatorTests(unittest.TestCase):
    """The validator accepts every real rerolling and rejects a rerolling that computes something
    else."""

    pairs: list[tuple[ir.Program, list[ir.Program]]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.pairs = flat_and_rerolled()

    def test_swapping_two_results_is_rejected(self) -> None:
        for flat, programs in self.pairs:
            *rounds, block = programs
            values = validate.Evaluator({r.name: r for r in rounds}).run(flat)
            i, j = next(
                (i, j)
                for i in range(len(values))
                for j in range(i + 1, len(values))
                if values[i] != values[j]
            )
            results = list(block.results)
            results[i], results[j] = results[j], results[i]
            swapped = dataclasses.replace(block, results=results)
            name = f"{block.architecture}:{block.name}"
            with (
                self.subTest(block=name),
                self.assertRaisesRegex(GenerationError, "does not compute"),
            ):
                validate.check(flat, [*rounds, swapped])

    def test_one_iteration_fewer_is_rejected(self) -> None:
        mutated = 0
        for flat, programs in self.pairs:
            *rounds, block = programs
            for index, node in enumerate(block.nodes):
                if isinstance(node, Call) and node.count > 1:
                    nodes = list(block.nodes)
                    nodes[index] = dataclasses.replace(node, count=node.count - 1)
                    fewer = dataclasses.replace(block, nodes=nodes)
                    with self.assertRaisesRegex(GenerationError, "does not compute"):
                        validate.check(flat, [*rounds, fewer])
                    mutated += 1
        self.assertGreater(mutated, 0)

    def test_malformed_calls_and_reads_are_rejected(self) -> None:
        def program(nodes: list[Node], name: str = "p") -> ir.Program:
            return CONVENTIONS.program("Test", ir.Signature(name, (), "Nat"), "", nodes, ["d"])

        evaluator = validate.Evaluator({})
        with self.assertRaisesRegex(GenerationError, "read before it is bound"):
            evaluator.run(program([Mov(dest="d", a=ir.Reg("ghost"))]))
        call = Call(dest="d", round="nowhere", scalars=(), state=())
        with self.assertRaisesRegex(GenerationError, "not a round of this block"):
            evaluator.run(program([call]))
        signature = ir.Signature("r", (("x", "Nat"), ("st", "S")), "S")
        round_ = CONVENTIONS.program("Test", signature, "", [Mov(dest="d", a=ir.lit(1))], ["d"])
        call = Call(dest="d", round="r", scalars=(), state=())
        with self.assertRaisesRegex(GenerationError, "one argument per parameter"):
            validate.Evaluator({"r": round_}).run(program([call]))
        stateless = {"r": program([Mov(dest="d", a=ir.lit(1))], "r")}
        with self.assertRaisesRegex(GenerationError, "takes no state"):
            validate.Evaluator(stateless).run(program([call]))


class ScopeAndNamingTests(unittest.TestCase):
    def test_a_definition_reads_only_its_parameters(self) -> None:
        # A round that read the block's `rhs` would name something out of its scope.
        nodes: list[Node] = [Mov(comment="read", dest="d", a=ir.Get(ir.Arg("rhs"), "l0"))]
        with self.assertRaisesRegex(GenerationError, "rhs, which is not a parameter of `round`"):
            CONVENTIONS.program(
                "Test", ir.Signature("round", (("st", "S"),), "Nat"), "", nodes, ["d"]
            )

    def test_a_round_reading_a_block_argument_is_rejected(self) -> None:
        source = x86_64_blocks.SOURCE.read_text()
        arm = source.index("(last) => {", source.index("macro_rules! divstep"))
        at = source.index("concat!(\n", arm) + len("concat!(\n")
        reads = '            "add {td}, qword ptr [{p}]\n",\n'
        with self.assertRaises(GenerationError):
            x86_64_blocks.programs(source[:at] + reads + source[at:])

    def test_ssa_names_skip_names_in_use(self) -> None:
        lets = [
            ir.Let("t", ir.Arg("x"), None),
            ir.Let("t_1", ir.Arg("y"), None),
            ir.Let("t", ir.apply("addw", ir.Reg("t"), 1), None),
            ir.Let("t_1", ir.apply("addw", ir.Reg("t_1"), ir.Reg("t")), None),
        ]
        names = ssa_names(lets)
        self.assertEqual(names, ["t", "t_1", "t_2", "t_1_1"])
        self.assertEqual(len(set(names)), len(names))

    def test_scratch_keys_with_one_name_do_not_share_state(self) -> None:
        program = CONVENTIONS.program(
            "Test", ir.Signature("p", (), "Nat"), "", [Mov(dest="d", a=ir.lit(0))], ["d"]
        )
        state = State(program)
        first: ir.ScratchKey[dict[str, int]] = ir.ScratchKey("k", dict)
        second: ir.ScratchKey[list[int]] = ir.ScratchKey("k", list)
        state.scratch(first)["a"] = 1
        self.assertEqual(state.scratch(second), [])


class MacroRunTests(unittest.TestCase):
    SOURCE = r"""
macro_rules! inc {
    () => { "add {a}, {a}, #1\n" };
}
macro_rules! nop {
    () => { "mov xzr, {a}\n" };
}
pub fn probe(lhs: &[u64; 4], rhs: &[u64; 4]) -> [u64; 4] {
    let mut r0 = lhs[0];
    let (r1, r2, r3): (u64, u64, u64);
    unsafe {
        asm!(
            "mov {r1}, {a}",
            INVOCATIONS
            "mov {r2}, {a}",
            "mov {r3}, {a}",
            a = inout(reg) r0,
            r1 = out(reg) r1,
            r2 = out(reg) r2,
            r3 = out(reg) r3,
            options(pure, nomem, nostack),
        );
    }
    [r0, r1, r2, r3]
}
"""

    def calls(self, invocations: str, emit_struct: bool = True) -> list[str]:
        rounds = {
            ("inc", ""): reroll.MacroRound(
                round="incRound",
                doc="one increment",
                state="IncState",
                arg="st",
                call="s",
                roles=(reroll.Role("a", "a", "the counter"),),
                emit_struct=emit_struct,
            )
        }
        target = dataclasses.replace(aarch64_blocks.TARGET, rerollers={}, macro_rounds=rounds)
        block = driver.Block(
            "probe", "probe", "A test block.", (("lhs", "Limbs"), ("rhs", "Limbs"))
        )
        *_, main = driver.transcribe(self.SOURCE.replace("INVOCATIONS", invocations), block, target)
        return [let.expr for node in main.nodes if isinstance(node, Call) for let in node.lets()]

    def test_consecutive_invocations_are_one_iterated_call(self) -> None:
        self.assertEqual(self.calls("inc!(), inc!(),"), ["incRound^[2] ⟨a⟩"])

    def test_an_empty_invocation_between_two_runs_is_not_an_iteration(self) -> None:
        # `nop!()` lifts to no nodes (a write to `xzr`); it must not be counted as an `inc!()`.
        self.assertEqual(self.calls("inc!(), nop!(), inc!(),"), ["incRound ⟨a⟩", "incRound ⟨a⟩"])

    def test_a_round_whose_structure_no_round_declares_is_rejected(self) -> None:
        with self.assertRaisesRegex(GenerationError, "no round of this block declares IncState"):
            self.calls("inc!(),", emit_struct=False)


class X86RejectionTests(unittest.TestCase):
    """The x86-64 operand checks, one each."""

    @staticmethod
    def lifter() -> x86_64.Lifter:
        emitter = x86_64.Lifter({"rdx": "inout", "a": "inout", "b": "in", "p": "in"})
        for register in ("rdx", "a", "b"):
            x86_bind(emitter, register, 0)
        emitter.bind_pointer("p", "rhs", ("l0", "l1", "l2", "l3"))
        return emitter

    def test_each_check(self) -> None:
        for line, message in (
            ("add {a:e}, {b}", "unsupported register modifier :e"),
            # A 16-bit XOR leaves the upper bits: it is not the zeroing idiom.
            ("xor {a:x}, {a:x}", "unsupported register modifier :x"),
            ("mov {a}, {p}", "pointer register p used outside an address"),
            ("mulx {a}, {a}, {b}", "mulx destinations must be distinct"),
            ("lea {a}, [{a} + 0x80000000]", "does not fit a disp32"),
            ("lea {a}, [{a} * 2]", "unsupported LEA address"),
            ("bt {a}, 64", "bit indices below 64"),
            ("test {a}, qword ptr [{p}]", "TEST with a memory operand"),
            ("mov {a}, 18446744073709551616", "does not fit a word"),
            ("mov {a}, qword ptr [{p} + 32]", "unsupported memory offset 32"),
        ):
            with self.subTest(line=line), self.assertRaisesRegex(GenerationError, message):
                self.lifter().lift(line)

    def test_the_front_end_rejects_what_it_cannot_model(self) -> None:
        probe = RejectionPathTests().probe
        for edits, message in (
            # A late output may share a register with an input: not a register of its own.
            ({"r3": "r3 = lateout(reg) r3,"}, "unsupported operand binding"),
            ({"extra": "r0 = in(reg) r1,"}, "duplicate operand declaration"),
            ({"r3": "r3 = inout(reg) r0,"}, "duplicate output target r0"),
            ({"instructions": '"add {r0}, {r1}\\t",'}, "escaped asm template strings"),
        ):
            with self.subTest(edits=edits), self.assertRaisesRegex(GenerationError, message):
                probe(**edits)
        for source, message in (
            ('macro_rules! m { () => { "a\\n" }; }\n' * 2, "defined twice"),
            ('macro_rules! m { () => { "a\\t" }; }\n', "may escape only"),
            ('macro_rules! m { () => { "a\\n\\n" }; }\n', "has an empty line"),
        ):
            with self.subTest(source=source), self.assertRaisesRegex(GenerationError, message):
                rust.parse_macros(source)

    def test_the_zeroing_idiom_takes_the_32_bit_register(self) -> None:
        emitter = self.lifter()
        emitter.lift("xor {a:e}, {a:e}")
        self.assertEqual(expressions(emitter)[-3:], ["0", "0", "0"])


# -- a toy instruction set's further features -----------------------------------------------------


class RvMoreLifter(RvLifter):
    """The toy, with a negative immediate (`addi`) and a flagless conditional zero (`czeqz`,
    `d = c = 0 ? 0 : a`), the shapes a RISC-V port meets beyond `add`."""

    def lift_instruction(self, ins: isa.Instruction) -> None:
        if ins.op == "addi":
            dest, source, immediate = ins.operands
            a = ir.Reg(self.require_written(self.operand(source)))
            self.check_writable(self.operand(dest))
            value = ir.lit(int(immediate) % 2**64)  # sign-extended to the word
            self.emit(ir.Word(dest=self.operand(dest), function="addw", args=(a, value)))
        elif ins.op == "czeqz":
            dest, x, c = (self.operand(t) for t in ins.operands)
            a, flag = ir.Reg(self.require_written(x)), ir.Reg(self.require_written(c))
            self.check_writable(dest)
            self.emit(
                ir.Select(
                    dest=dest,
                    function="czeroEqz",
                    flag=flag,
                    x=ir.lit(0),
                    y=a,
                    x_when=ir.FlagState.CLEAR,
                )
            )
        else:
            super().lift_instruction(ins)


class RiscVMore(RiscV):
    lifter = RvMoreLifter


class NewInstructionSetFeatureTests(unittest.TestCase):
    """The toy's immediates, flagless selection, and a rerolled macro run, all through shared
    code."""

    SOURCE = """
macro_rules! step {
    () => { "add {r0}, {r0}, {r1}\n" };
}
pub(super) fn more(lhs: &Limbs, rhs: &Limbs) -> Limbs {
    let [mut r0, mut r1, mut r2, mut r3] = *lhs;
    // SAFETY: register-only arithmetic with declared inputs and outputs.
    unsafe {
        asm!(
            "addi {r2}, {r2}, -1",
            "czeqz {r3}, {r3}, {r2}",
            step!(),
            step!(),
            step!(),
            r0 = inout(reg) r0,
            r1 = inout(reg) r1,
            r2 = inout(reg) r2,
            r3 = inout(reg) r3,
            options(pure, nomem, nostack),
        );
    }
    [r0, r1, r2, r3]
}
"""

    def test_immediates_selection_and_runs(self) -> None:
        rounds = {
            ("step", ""): reroll.MacroRound(
                round="stepRound",
                doc="one step",
                state="StepState",
                arg="st",
                call="s",
                roles=(reroll.Role("r0", "acc"), reroll.Role("r1", "inc")),
                emit_struct=True,
            )
        }
        target = driver.Target(RiscVMore(), aarch64_blocks.KINDS, CONVENTIONS, macro_rounds=rounds)
        block = driver.Block("more", "moreRv", "A toy block.", (("lhs", "Limbs"), ("rhs", "Limbs")))
        step, main = driver.transcribe(self.SOURCE, block, target)
        text = lean.definition(main, 32)
        self.assertIn(f"let r2 := addw r2 {2**64 - 1}", text)
        self.assertIn("let r3 := czeroEqz r2 0 r3", text)
        self.assertIn("stepRound^[3] ⟨r0, r1⟩", text)
        self.assertIn("let r0 := addw r0 r1", lean.definition(step, 32))
        steps = "\n".join(skeleton(main))
        self.assertIn("(if r2_1 = 0 then 0 else r3)", steps)


if __name__ == "__main__":
    unittest.main()
