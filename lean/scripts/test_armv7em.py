#!/usr/bin/env python3
"""Focused tests for the ARMv7E-M transcription: its lifter (`asm2lean/isa/armv7em.py`) and its
blocks and factorings (`pasta/architectures/armv7em.py`)."""

import dataclasses
import re
import sys
import unittest

# Running this source-tree test should not leave lean/scripts/__pycache__ behind.
sys.dont_write_bytecode = True


from asm2lean import rust, validate
from asm2lean.ir import App, Lit, Mov, Node, Program, Tuple
from asm2lean.isa.armv7em import Bind
from asm2lean.skeleton import skeleton
from pasta.architectures import armv7em


def programs(source: str | None = None) -> dict[str, Program]:
    return {program.name: program for program in armv7em.programs(source)}


def changed(nodes: list[Node]) -> list[Node]:
    """`nodes` with the first one replaced by a move of zero to its register."""
    first = nodes[0]
    return [Mov(dest=first.writes()[-1], a=Lit("0"), pc=first.pc), *nodes[1:]]


class ARMv7EMGeneratorTests(unittest.TestCase):
    source: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = armv7em.SOURCE.read_text()

    @staticmethod
    def mutate_function(source: str, name: str, old: str, new: str) -> str:
        masked = rust.masked_noncode(source)
        matches = list(re.finditer(rf"(?m)^fn\s+{re.escape(name)}\s*\(", masked))
        for match in matches:
            signature_open = masked.find("(", match.start())
            signature_close = rust.matching_delimiter(source, signature_open, "(", ")")
            body_open = masked.find("{", signature_close)
            body_close = rust.matching_delimiter(source, body_open, "{", "}")
            function_source = source[match.start() : body_close + 1]
            if "asm!" not in function_source or old not in function_source:
                continue
            if function_source.count(old) != 1:
                raise AssertionError(f"expected one {old!r} in inline-asm {name}")
            return (
                source[: match.start()]
                + function_source.replace(old, new)
                + source[body_close + 1 :]
            )
        raise AssertionError(f"inline-asm function {name} containing {old!r} not found")

    def assert_rejected(self, source: str, message: str) -> None:
        with self.assertRaisesRegex(rust.GenerationError, message):
            armv7em.text(source)

    @staticmethod
    def parse_from_mont(source: str) -> rust.ParsedFunction:
        return rust.parse_function(
            source,
            "from_mont",
            ["value", "modulus", "inv"],
            8,
            fixed_registers={
                "r0",
                "r1",
                "r2",
                "r3",
                "r4",
                "r5",
                "r8",
                "r9",
                "r10",
                "r11",
                "r12",
                "r14",
            },
            fixed_register_kinds={"in", "out", "inout"},
            allowed_options={"pure", "readonly", "nostack"},
            required_options={"pure", "readonly", "nostack"},
        )

    def test_from_mont_real_source_parses_and_mutations_fail_closed(self) -> None:
        parsed = self.parse_from_mont(self.source)
        self.assertEqual(len(parsed.instructions), 208)
        self.assertEqual(
            tuple(
                (declaration.name, declaration.kind, declaration.value)
                for declaration in parsed.declarations
                if declaration.kind in {"in", "inout"}
            ),
            (("r0", "inout", "value.as_ptr()"), ("r1", "in", "modulus.as_ptr()")),
        )
        self.assertEqual(
            rust.returned_registers(parsed, "from_mont"),
            ("r14", "r0", "r2", "r3", "r4", "r5", "r8", "r9"),
        )
        self.assertEqual(
            parsed.returns,
            ("word0", "word1", "word2", "word3", "word4", "word5", "word6", "word7"),
        )
        macro_origins = [origin for origin in parsed.origins if origin is not None]
        self.assertEqual(len(macro_origins), 8 * 21)
        self.assertEqual({origin.macro for origin in macro_origins}, {"from_mont_round"})
        self.assertEqual({origin.site for origin in macro_origins}, set(range(8)))

        cases = (
            (
                self.source.replace(
                    "debug_assert_eq!(inv as u32", "debug_assert_ne!(inv as u32", 1
                ),
                "unsupported code before asm",
            ),
            (
                self.source.replace(
                    "debug_assert_eq!(inv as u32, u32::MAX);",
                    "debug_assert_eq!(inv as u32, u32::MAX)",
                    1,
                ),
                "debug_assert! must end with `;`",
            ),
            (
                self.source.replace("from_mont_round!(r2, r3", "from_mont_round!(2, r3", 1),
                "has no arm",
            ),
            (
                self.source.replace('lateout("r10") _', 'lateout("r6") _', 1),
                "unsupported fixed-register binding",
            ),
            (
                self.source.replace(
                    "u64::from(word4) | (u64::from(word5) << 32),\n        u64::from(word6)",
                    "u64::from(word4) | (u64::from(word5) << 32)\n        u64::from(word6)",
                    1,
                ),
                "unsupported joined-word return",
            ),
        )
        for mutated, message in cases:
            with self.subTest(message=message):
                self.assertNotEqual(mutated, self.source)
                with self.assertRaisesRegex(rust.GenerationError, message):
                    self.parse_from_mont(mutated)

    def test_generated_header_imports_semantics_and_excludes_isa_definitions(self) -> None:
        generated = armv7em.text(self.source)
        self.assertIn("import PastaCurves.ARMv7EM.Semantics", generated)
        for name in (
            "wordMod",
            "addc32",
            "subc32",
            "umaal32",
            "eor32",
            "orr32",
            "lsl32",
            "lsr32",
            "asr32",
            "join32",
            "signExtend32",
        ):
            with self.subTest(definition=name):
                self.assertNotRegex(generated, rf"(?m)^(?:abbrev|def) {name}\b")

    def test_transcribed_source_blocks_and_full_inventory(self) -> None:
        self.assertEqual(rust.asm_functions(self.source), armv7em.ASM_BLOCKS)
        self.assertEqual(
            [program.name for program in armv7em.programs(self.source)],
            [
                "adc",
                "addWord",
                "sbb",
                "macWord",
                "inversionMacWords",
                "fromMontRoundPrefixQuotient",
                "fromMontRoundPrefixSelfSub",
                "fromMontRoundPrefixMask",
                "fromMontRoundPrefix",
                "fromMontRoundSuffix",
                "fromMontRound",
                "fromMontFinalSubtractionBlock",
                "fromMontFinalAddBackBlock",
                "fromMontFinalReductionBlock",
                "fromMontBlock",
                "batchRoundFinish",
                "batchRound",
                "batchBlock",
                "divstep59Block",
                "condSub",
            ],
        )
        generated = armv7em.text(self.source)
        for instruction in (
            "adds {sum}, {a}, {b}",
            "adc {carry}, {carry}, #0",
            "rsbs {borrow}, {borrow}, #0",
            "sbcs {difference}, {a}, {b}",
            "umaal {low}, {high}, {a}, {b}",
            "umaal {r0}, {r2}, {b0}, {c0}",
            "add {r3}, {r3}, {c0}",
            "ldr r2, [{value}, #28]",
            "sbcs r2, r2, r3",
            "sbc.w {keep}, {keep}, {keep}",
            "and r3, r3, {keep}",
            "str r2, [{out}, #28]",
        ):
            self.assertIn(f"-- {instruction}", generated)
        self.assertIn("let result3 := out_24 + wordMod * out_28", generated)
        self.assertIn("structure FromMontRoundPrefix where", generated)
        self.assertIn("  quotient : Nat", generated)
        self.assertIn("  carry : Nat", generated)
        self.assertIn("def fromMontRoundPrefixQuotient (w0 : Nat) : Nat × Nat :=", generated)
        self.assertIn(
            "def fromMontRoundPrefixSelfSub (quotient subCarry : Nat) : Nat :=", generated
        )
        self.assertIn("def fromMontRoundPrefixMask (selfSub : Nat) : Nat :=", generated)
        self.assertIn("def fromMontRoundPrefix (w0 : Nat) : FromMontRoundPrefix :=", generated)
        self.assertIn("fromMontRoundPrefixQuotient w0", generated)
        self.assertIn("fromMontRoundPrefixSelfSub quotient subCarry", generated)
        self.assertIn("fromMontRoundPrefixMask selfSub", generated)
        self.assertIn(
            "def fromMontRoundSuffix (modulus : Limbs) (st : FromMontState) "
            "(q carry : Nat) : FromMontState :=",
            generated,
        )
        self.assertIn("def fromMontRound (modulus : Limbs) (st : FromMontState)", generated)
        self.assertIn("let quotientResult := fromMontRoundPrefix st.w0", generated)
        round_body = generated.split("def fromMontRound (", 1)[1].split(
            "structure FromMontFinalSubtraction ", 1
        )[0]
        self.assertIn("let q := quotientResult.quotient", round_body)
        self.assertIn("let carry := quotientResult.carry", round_body)
        self.assertNotIn("quotientResult.1", round_body)
        self.assertNotIn("quotientResult.2", round_body)
        self.assertIn("let suffix := fromMontRoundSuffix modulus st q carry", round_body)
        self.assertIn("structure FromMontFinalSubtraction where", generated)
        self.assertIn(
            "def fromMontFinalSubtractionBlock (modulus : Limbs) (st : FromMontState)",
            generated,
        )
        self.assertIn(
            "def fromMontFinalAddBackBlock (modulus : Limbs) (sub : FromMontFinalSubtraction)",
            generated,
        )
        self.assertIn(
            "def fromMontFinalReductionBlock (modulus : Limbs) (st : FromMontState)", generated
        )
        final = generated.split("def fromMontFinalReductionBlock ", 1)[1].split(
            "def fromMontBlock ", 1
        )[0]
        self.assertIn("fromMontFinalSubtractionBlock modulus st", final)
        self.assertIn("fromMontFinalAddBackBlock modulus subtraction", final)
        self.assertIn("def fromMontBlock (value modulus : Limbs) : Limbs :=", generated)
        caller = generated.split("def fromMontBlock ", 1)[1].split("structure BatchState", 1)[0]
        self.assertEqual(caller.count("fromMontRound modulus ⟨"), 8)
        self.assertEqual(caller.count("fromMontFinalReductionBlock modulus ⟨"), 1)
        self.assertNotIn("-- sbc.w r12, r10, r10", caller)
        self.assertNotIn("-- adc.w r9, r9, r11", caller)
        subtraction = generated.split("def fromMontFinalSubtractionBlock ", 1)[1].split(
            "def fromMontFinalAddBackBlock ", 1
        )[0]
        addback = generated.split("def fromMontFinalAddBackBlock ", 1)[1].split(
            "def fromMontFinalReductionBlock ", 1
        )[0]
        self.assertIn("-- sbc.w r12, r10, r10", subtraction)
        self.assertNotIn("-- adc.w r9, r9, r11", subtraction)
        self.assertNotIn("-- sbc.w r12, r10, r10", addback)
        self.assertIn("-- adc.w r9, r9, r11", addback)
        self.assertEqual(generated, armv7em.OUTPUT.read_text())

    def test_from_mont_factored_routines_generate_proof_skeletons(self) -> None:
        routine_names = {
            "fromMontRoundPrefixQuotient",
            "fromMontRoundPrefixSelfSub",
            "fromMontRoundPrefixMask",
            "fromMontRoundPrefix",
            "fromMontRoundSuffix",
            "fromMontRound",
            "fromMontFinalSubtractionBlock",
            "fromMontFinalAddBackBlock",
            "fromMontFinalReductionBlock",
            "fromMontBlock",
        }
        routines = {
            name: program
            for name, program in programs(self.source).items()
            if name in routine_names
        }
        self.assertEqual(set(routines), routine_names)
        skeletons = {name: skeleton(program) for name, program in routines.items()}
        for name, lines in skeletons.items():
            with self.subTest(routine=name):
                self.assertIn(f"generated skeleton for `{name}`", "\n".join(lines))
        self.assertEqual(len(skeletons["fromMontRoundPrefixQuotient"]), 15)
        self.assertEqual(len(skeletons["fromMontRoundPrefixSelfSub"]), 12)
        self.assertEqual(len(skeletons["fromMontRoundPrefixMask"]), 10)
        self.assertEqual(len(skeletons["fromMontRoundPrefix"]), 13)
        self.assertEqual(len(skeletons["fromMontRoundSuffix"]), 168)
        self.assertEqual(len(skeletons["fromMontRound"]), 30)
        self.assertEqual(len(skeletons["fromMontFinalSubtractionBlock"]), 132)
        self.assertEqual(len(skeletons["fromMontFinalAddBackBlock"]), 142)
        self.assertEqual(len(skeletons["fromMontFinalReductionBlock"]), 8)
        quotient = "\n".join(skeletons["fromMontRoundPrefixQuotient"])
        self.assertIn("subc32_lin 0 w0' 1", quotient)
        self_sub = "\n".join(skeletons["fromMontRoundPrefixSelfSub"])
        self.assertIn("sbc.w r12, r2, r2", self_sub)
        mask = "\n".join(skeletons["fromMontRoundPrefixMask"])
        self.assertIn("Nat.and_lt_two_pow", mask)
        prefix = "\n".join(skeletons["fromMontRoundPrefix"])
        self.assertIn("fromMontRoundPrefixQuotient w0", prefix)
        self.assertIn("fromMontRoundPrefixSelfSub quotient subCarry", prefix)
        self.assertIn("fromMontRoundPrefixMask selfSub", prefix)
        suffix = "\n".join(skeletons["fromMontRoundSuffix"])
        self.assertEqual(len(re.findall(r"(?m)^  -- .*: umaal ", suffix)), 4)
        self.assertIn("addc32_carry_le_one", suffix)
        round_skeleton = "\n".join(skeletons["fromMontRound"])
        self.assertIn("quotientResult := fromMontRoundPrefix st.w0", round_skeleton)
        self.assertIn("q := quotientResult.quotient", round_skeleton)
        self.assertIn("carry := quotientResult.carry", round_skeleton)
        self.assertNotIn("quotientResult.1", round_skeleton)
        self.assertNotIn("quotientResult.2", round_skeleton)
        self.assertIn("fromMontRoundSuffix modulus st q carry", round_skeleton)
        caller = "\n".join(skeletons["fromMontBlock"])
        self.assertEqual(caller.count("fromMontRound modulus"), 8)
        self.assertEqual(caller.count("fromMontFinalReductionBlock modulus"), 1)
        self.assertNotIn("subc32_lin", caller)
        self.assertNotIn("addc32_carry_le_one", caller)
        subtraction = "\n".join(skeletons["fromMontFinalSubtractionBlock"])
        self.assertIn("subc32_lin", subtraction)
        self.assertNotIn("addc32_carry_le_one", subtraction)
        addback = "\n".join(skeletons["fromMontFinalAddBackBlock"])
        self.assertNotIn("subc32_lin", addback)
        self.assertIn("addc32_carry_le_one", addback)
        final = "\n".join(skeletons["fromMontFinalReductionBlock"])
        self.assertIn("fromMontFinalSubtractionBlock modulus st", final)
        self.assertIn("fromMontFinalAddBackBlock modulus subtraction", final)
        self.assertNotIn("subc32_lin", final)
        self.assertNotIn("addc32_carry_le_one", final)

    def test_from_mont_helper_accounting_rejects_mutations(self) -> None:
        _, flat = armv7em.lift_from_mont(self.source)
        round_nodes = armv7em.round_nodes(flat.nodes, 0)
        prefix = [n for n in round_nodes if n.pc is not None and n.pc < 3]
        suffix = [n for n in round_nodes if n.pc is not None and n.pc >= 3]
        armv7em.check_round_split(round_nodes, prefix, suffix)
        self.assertEqual([n.pc for n in prefix], [0, 1, 2])
        self.assertEqual(
            [(type(n).__name__, n.writes()) for n in prefix],
            [("Rsbs", ["d", "w0", "c"]), ("Sbc", ["r12"]), ("And", ["r12"])],
        )
        self.assertEqual({n.pc for n in suffix}, set(range(3, 21)))
        quotient, self_sub, mask = ([n for n in prefix if n.pc == pc] for pc in range(3))
        armv7em.check_prefix_split(prefix, quotient, self_sub, mask)
        final = armv7em.final_nodes(flat.nodes)
        subtraction = [n for n in final if n.pc is not None and n.pc < 14]
        addback = [n for n in final if n.pc is not None and n.pc >= 14]
        armv7em.check_final_split(final, subtraction, addback)
        self.assertEqual({n.pc for n in subtraction}, set(range(14)))
        self.assertEqual({n.pc for n in addback}, set(range(14, 31)))

        with self.assertRaisesRegex(rust.GenerationError, "sparse suffix does not cover"):
            armv7em.check_round_split(round_nodes, prefix, suffix[:-1])
        with self.assertRaisesRegex(rust.GenerationError, "quotient prefix does not cover"):
            armv7em.check_round_split(round_nodes, prefix[:-1], [prefix[-1], *suffix])
        with self.assertRaisesRegex(rust.GenerationError, "do not reconstruct"):
            armv7em.check_round_split(round_nodes, prefix, changed(suffix))
        with self.assertRaisesRegex(rust.GenerationError, "quotient helper does not cover"):
            armv7em.check_prefix_split(prefix, [*quotient, *self_sub], [], mask)
        with self.assertRaisesRegex(rust.GenerationError, "self-subtraction helper does not cover"):
            armv7em.check_prefix_split(prefix, quotient, [*self_sub, *mask], [])
        with self.assertRaisesRegex(
            rust.GenerationError, "short prefix helpers do not reconstruct"
        ):
            armv7em.check_prefix_split(prefix, quotient, self_sub, changed(mask))
        with self.assertRaisesRegex(
            rust.GenerationError, "final subtraction helper does not cover"
        ):
            armv7em.check_final_split(final, subtraction[:-1], [subtraction[-1], *addback])
        with self.assertRaisesRegex(rust.GenerationError, "final add-back helper does not cover"):
            armv7em.check_final_split(final, subtraction, addback[1:])
        with self.assertRaisesRegex(
            rust.GenerationError, "final reduction helpers do not reconstruct"
        ):
            armv7em.check_final_split(final, subtraction, changed(addback))

    def test_factorings_are_validated_and_mutations_rejected(self) -> None:
        checked: list[tuple[Program, list[Program]]] = []
        real = validate.check

        def spy(flat: Program, programs: list[Program]) -> None:
            checked.append((flat, programs))
            real(flat, programs)

        validate.check = spy
        try:
            armv7em.programs(self.source)
        finally:
            validate.check = real
        self.assertEqual(
            [programs[-1].name for _, programs in checked], ["fromMontBlock", "batchRound"]
        )
        flat, programs = checked[0]

        def swapped(node: Node) -> Node:
            if isinstance(node, Bind) and node.dest == "round1":
                assert isinstance(node.term, App) and isinstance(node.term.args[1], Tuple)
                items = node.term.args[1].items
                state = Tuple((items[1], items[0], *items[2:]))
                return dataclasses.replace(
                    node, term=App(node.term.function, (node.term.args[0], state))
                )
            return node

        block = programs[-1]
        mutated_block = dataclasses.replace(block, nodes=[swapped(n) for n in block.nodes])
        with self.assertRaisesRegex(rust.GenerationError, "does not compute the flat block"):
            real(flat, [*programs[:-1], mutated_block])
        suffix = next(p for p in programs if p.name == "fromMontRoundSuffix")
        index = next(i for i, n in enumerate(suffix.nodes) if n.pc is not None)
        mutated_suffix = dataclasses.replace(
            suffix,
            nodes=[*suffix.nodes[:index], *changed(suffix.nodes[index:])],
        )
        with self.assertRaisesRegex(rust.GenerationError, "does not compute the flat block"):
            real(flat, [mutated_suffix if p is suffix else p for p in programs])

    def test_from_mont_rejects_offset_rotation_alias_selection_and_output_mutations(self) -> None:
        cases = (
            (
                self.source.replace('"ldr.w r2, [r0, #0]",', '"ldr.w r2, [r0, #4]",', 1),
                "value memory offsets differ",
            ),
            (
                self.source.replace(
                    "from_mont_round!(r3, r4, r5, r8, r9, r10, r11, lr, r0, r2)",
                    "from_mont_round!(r3, r5, r4, r8, r9, r10, r11, lr, r0, r2)",
                    1,
                ),
                "expected register rotation|rounds differ",
            ),
            (
                self.source.replace(
                    'stringify!($tmp), ", ", stringify!($a0), "\\n"),',
                    'stringify!($a1), ", ", stringify!($a0), "\\n"),',
                    1,
                ),
                "umaal destinations overlap input registers",
            ),
            (
                self.source.replace('"subs.w lr, lr, #1",', '"subs.w lr, lr, #2",', 1),
                "final borrow selection differs",
            ),
            (
                self.source.replace(
                    '"ldr.w r11, [r1, #4]",\n            "sbcs.w r0, r0, r11",',
                    '"ldr.w r11, [r1, #8]",\n            "sbcs.w r0, r0, r11",',
                    1,
                ),
                "modulus memory offsets differ|final borrow selection differs",
            ),
            (
                self.source.replace('"sbc.w r12, r10, r10",', '"sbc.w r12, r9, r9",', 1),
                "final borrow selection differs",
            ),
            (
                self.source.replace('lateout("r14") word0', 'lateout("r14") _', 1),
                "fixed-register declarations differ|returned registers differ",
            ),
        )
        for source, message in cases:
            with self.subTest(message=message):
                self.assertNotEqual(source, self.source)
                self.assert_rejected(source, message)

    def test_existing_proved_and_cond_sub_routines_generate_proof_skeletons(self) -> None:
        proof_names = {"adc", "addWord", "sbb", "macWord", "inversionMacWords", "condSub"}
        routines = [
            program for program in armv7em.programs(self.source) if program.name in proof_names
        ]
        self.assertTrue(all(program.architecture == "ARMv7EM" for program in routines))
        skeletons = {program.name: "\n".join(skeleton(program)) for program in routines}
        for name, text in skeletons.items():
            with self.subTest(routine=name):
                self.assertIn(f"generated skeleton for `{name}`", text)
                self.assertIn("< 2^32", text)
        self.assertIn("b_carry'", skeletons["adc"])
        self.assertIn("b_c", skeletons["addWord"])
        self.assertIn("subc32_carry_le_one 0 borrow' 1", skeletons["sbb"])
        self.assertIn("d_high", skeletons["macWord"])
        self.assertIn("d_r3", skeletons["inversionMacWords"])
        self.assertIn("subc32_lin r2 r3 1", skeletons["condSub"])
        self.assertIn("sbc.w {keep}, {keep}, {keep}", skeletons["condSub"])
        self.assertIn("Nat.and_lt_two_pow", skeletons["condSub"])
        self.assertIn("b_result3 : result3 < 2^64", skeletons["condSub"])

    def test_batch_round_is_factored_and_divstep59_composes_three_batches(self) -> None:
        generated = armv7em.text(self.source)
        self.assertIn("structure BatchState where", generated)
        self.assertIn("def batchRound (st : BatchState) : BatchState :=", generated)
        self.assertIn("let st := batchRound^[k] st", generated)
        for instruction in (
            "and {odd}, {pglo}, #1",
            "rsbs {tmp}, {d}, #0",
            "orr.w {pglo}, {tmp}, {pghi}, lsl #31",
            "subs {remaining}, {remaining}, #1",
            "bne 2b",
        ):
            self.assertGreaterEqual(
                len(re.findall(rf"-- {re.escape(instruction)}$", generated, re.MULTILINE)), 1
            )
        asm_inner = rust.checked_function(self.source, "batch", armv7em.BATCH_RUST)
        instructions, _, _, _ = rust.parse_asm_invocation(
            self.source,
            asm_inner,
            "batch",
            reserved_names={"c", "m", "s"},
            allowed_options={"pure", "nomem", "nostack"},
            required_options={"pure", "nomem", "nostack"},
        )
        self.assertEqual(len(instructions), 32)
        self.assertEqual(generated.count("def batchRound (st : BatchState)"), 1)
        self.assertEqual(generated.count("def batchRoundFinish (pglo pghi remaining : Nat)"), 1)
        self.assertIn("let finish := batchRoundFinish pglo pghi remaining", generated)
        routines = programs()
        finish = routines["batchRoundFinish"]
        self.assertEqual(
            [n.pc for n in finish.nodes[3:] for _ in n.lets()], [27, 28, 29, 30, 30, 30, 30, 31]
        )
        finish_skeleton = "\n".join(skeleton(finish))
        self.assertIn("word_step -clear pglo' := pglo", finish_skeleton)
        self.assertIn(
            "word_step -clear sub, remaining_1 := (remaining' + 2^32 - 1 - (1 - 1)) % 2^32",
            finish_skeleton,
        )
        round_skeleton = "\n".join(skeleton(routines["batchRound"]))
        self.assertNotIn("lift_lets -merge at hres", round_skeleton)
        self.assertIn("word_step -clear pflo := st.pflo", round_skeleton)
        self.assertIn(
            "word_step -clear sub, tmp := (0 + 2^32 - d - (1 - 1)) % 2^32", round_skeleton
        )
        self.assertIn("word_step -clear swap := asr32 tmp 31", round_skeleton)
        for name in ("batchBlock", "divstep59Block"):
            text = "\n".join(skeleton(routines[name]))
            self.assertIn(f"generated skeleton for `{name}`", text)
            self.assertIn("lift_lets -merge at hres", text)
        # A live carry must be retained after every addback ADCS in the factored suffix.
        final_addback = generated.split("def fromMontFinalAddBackBlock ", 1)[1].split(
            "def fromMontFinalReductionBlock ", 1
        )[0]
        self.assertIn("let c := s.2", final_addback)
        self.assertEqual(final_addback.count("let c := adcs_pair.2"), 6)
        self.assertIn("let b1 := batchBlock 20 two_delta f0 g0", generated)
        self.assertIn("let b2 := batchBlock 20 d1 f1 g1", generated)
        self.assertIn("let b3 := batchBlock 19 d2 f2 g2", generated)
        self.assertIn("let m21 := matMulWords (u2, v2, q2, r2) (u1, v1, q1, r1)", generated)
        self.assertIn("let matrix := matMulWords (b3.u, b3.v, b3.q, b3.r) m21", generated)

    def test_batch_and_divstep59_source_changes_are_rejected(self) -> None:
        cases = (
            (
                self.source.replace("let remaining = k;", "let remaining = k + 1;", 1),
                "batch: surrounding Rust differs",
            ),
            (
                self.source.replace('"bne 2b",', '"bne 3b",', 1),
                "expected one loop from label",
            ),
            (
                self.source.replace(
                    "batch(20, two_delta, f0, g0)", "batch(19, two_delta, f0, g0)", 1
                ),
                "divstep59: surrounding Rust differs",
            ),
        )
        for source, message in cases:
            with self.subTest(message=message):
                self.assert_rejected(source, message)

    def test_batch_rejects_invalid_zero_flag_and_operand_declarations(self) -> None:
        invalid_flag = self.source.replace(
            '"subs {remaining}, {remaining}, #1",',
            '"sub.w {remaining}, {remaining}, #1",',
            1,
        )
        self.assert_rejected(invalid_flag, "unsupported instruction sub")
        declaration = self.source.replace(
            "remaining = inout(reg) remaining => _,",
            "remaining = inout(reg) remaining,",
            1,
        )
        self.assert_rejected(declaration, "assembly operand declarations differ")

    def test_cond_sub_rejects_malformed_offsets_uninitialized_memory_and_flags(self) -> None:
        malformed_offset = self.source.replace("cond_sub_word!(4),", "cond_sub_word!(2),", 1)
        self.assert_rejected(malformed_offset, "memory offset must be one of")

        uninitialized = self.mutate_function(
            self.source,
            "cond_sub",
            "let mut difference = [0u64; 4];",
            "let mut difference = [1u64; 4];",
        )
        self.assert_rejected(uninitialized, "unsupported code before asm")

        invalid_flag = self.source.replace("cond_sub_word!(first, 0),", "cond_sub_word!(0),", 1)
        self.assert_rejected(invalid_flag, "carry flag read while invalid")

    def test_full_asm_inventory_changes_are_rejected(self) -> None:
        renamed = self.source.replace("fn batch(k:", "fn batch_changed(k:", 1)
        self.assert_rejected(renamed, "ARMv7EM: asm block inventory differs")

    def test_unsupported_instruction_is_rejected(self) -> None:
        source = self.mutate_function(
            self.source,
            "adc",
            '"mov {carry_out}, #0",',
            '"mvn {carry_out}, #0",',
        )
        self.assert_rejected(source, "unsupported instruction mvn")

    def test_carry_read_before_flag_write_is_rejected(self) -> None:
        for function, old, new in (
            ("adc", '"adds {sum}, {a}, {b}",', '"mov {sum}, {a}",'),
            ("sbb", '"rsbs {borrow}, {borrow}, #0",', '"mov {borrow}, #0",'),
        ):
            with self.subTest(function=function):
                source = self.mutate_function(self.source, function, old, new)
                self.assert_rejected(source, "carry flag read while invalid")

    def test_inversion_mac_words_rejects_invalid_flags_memory_and_attributes(self) -> None:
        invalid_flag = self.mutate_function(
            self.source,
            "inversion_mac_words",
            '"adds {r2}, {r2}, {r3}",',
            '"add {r2}, {r2}, {r3}",',
        )
        self.assert_rejected(invalid_flag, "carry flag read while invalid")

        memory = self.mutate_function(
            self.source,
            "inversion_mac_words",
            '"add {r3}, {r3}, {c0}",',
            '"add {r3}, [{r3}], {c0}",',
        )
        self.assert_rejected(memory, "memory operands are unsupported")

        attribute = self.source.replace(
            '#[expect(unused_assignments, reason = "c0 is an assembly carry temporary")]',
            "#[allow(unused_assignments)]",
            1,
        )
        self.assert_rejected(attribute, "unsupported attribute before asm")

    def test_uninitialized_and_input_only_register_use_is_rejected(self) -> None:
        unread = self.mutate_function(
            self.source,
            "adc",
            '"adds {sum}, {a}, {b}",',
            '"adds {sum}, {sum}, {b}",',
        )
        self.assert_rejected(unread, "output-only register sum read before being written")

        input_write = self.mutate_function(
            self.source,
            "adc",
            '"adds {sum}, {a}, {b}",',
            '"adds {a}, {a}, {b}",',
        )
        self.assert_rejected(input_write, "input-only register a cannot be written")

    def test_unwritten_read_write_output_is_rejected(self) -> None:
        read_write = self.mutate_function(
            self.source,
            "mac_word",
            '"umaal {low}, {high}, {a}, {b}",',
            '"adds {low}, {a}, {b}",',
        )
        self.assert_rejected(read_write, "output register high was never written")

    def test_memory_operands_are_rejected(self) -> None:
        for function, old, new in (
            ("adc", '"adds {sum}, {a}, {b}",', '"adds {sum}, [{a}], {b}",'),
            (
                "mac_word",
                '"umaal {low}, {high}, {a}, {b}",',
                '"umaal {low}, {high}, [{a}], {b}",',
            ),
        ):
            with self.subTest(function=function):
                source = self.mutate_function(self.source, function, old, new)
                self.assert_rejected(source, "memory operands are unsupported")

    def test_umaal_destination_aliases_are_rejected(self) -> None:
        same_destination = self.mutate_function(
            self.source,
            "mac_word",
            '"umaal {low}, {high}, {a}, {b}",',
            '"umaal {low}, {low}, {a}, {b}",',
        )
        self.assert_rejected(same_destination, "umaal destinations must be distinct")

        for destination, left, right in (
            ("low", "low", "b"),
            ("low", "a", "low"),
            ("high", "high", "b"),
            ("high", "a", "high"),
        ):
            with self.subTest(destination=destination, left=left, right=right):
                overlap = self.mutate_function(
                    self.source,
                    "mac_word",
                    '"umaal {low}, {high}, {a}, {b}",',
                    f'"umaal {{low}}, {{high}}, {{{left}}}, {{{right}}}",',
                )
                self.assert_rejected(
                    overlap,
                    f"umaal destinations overlap input registers: {destination}",
                )

    def test_rsbs_form_is_restricted(self) -> None:
        source = self.mutate_function(
            self.source,
            "sbb",
            '"rsbs {borrow}, {borrow}, #0",',
            '"rsbs {borrow}, {borrow}, #1",',
        )
        self.assert_rejected(source, "rsbs supports only reverse subtraction from #0")

    def test_required_options_and_output_binding_are_validated(self) -> None:
        options = self.mutate_function(
            self.source,
            "adc",
            "options(pure, nomem, nostack),",
            "options(pure, nostack),",
        )
        self.assert_rejected(options, "requires options .*nomem")

        output = self.mutate_function(
            self.source,
            "adc",
            "sum = out(reg) sum,",
            "sum = out(reg) _,",
        )
        self.assert_rejected(output, "returned value sum is not an asm output")

    def test_second_inline_asm_definition_is_rejected(self) -> None:
        inline = rust.inline_asm_function(self.source, "adc")
        block = armv7em.Primitive("adc_extra", "adcExtra", ("a", "b", "carry"), "test")
        renamed = inline.replace("fn adc(", "fn adc_extra(")
        duplicated = self.source + "\n" + renamed + "\n" + renamed
        with self.assertRaisesRegex(
            rust.GenerationError,
            "exactly one top-level `fn adc_extra\\(` containing asm!, found 2",
        ):
            armv7em.primitive(duplicated, block)


if __name__ == "__main__":
    unittest.main()
