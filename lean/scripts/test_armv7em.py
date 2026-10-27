#!/usr/bin/env python3
"""Focused tests for the ARMv7E-M inline-assembly generator backend."""

import re
import sys
import unittest

# Running this source-tree test should not leave lean/scripts/__pycache__ behind.
sys.dont_write_bytecode = True

import asm_source
import gen
import gen_armv7em


class ARMv7EMGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = gen_armv7em.SOURCE.read_text()

    @staticmethod
    def mutate_function(source: str, name: str, old: str, new: str) -> str:
        masked = asm_source.masked_noncode(source)
        matches = list(re.finditer(rf"(?m)^fn\s+{re.escape(name)}\s*\(", masked))
        for match in matches:
            signature_open = masked.find("(", match.start())
            signature_close = asm_source.matching_delimiter(source, signature_open, "(", ")")
            body_open = masked.find("{", signature_close)
            body_close = asm_source.matching_delimiter(source, body_open, "{", "}")
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
        with self.assertRaisesRegex(asm_source.GenerationError, message):
            gen_armv7em.gen_program(source)

    @staticmethod
    def parse_from_mont(source: str) -> asm_source.ParsedFunction:
        return asm_source.parse_function(
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

    def test_from_mont_real_source_parses_and_mutations_fail_closed(self):
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
            asm_source.returned_registers(parsed, "from_mont"),
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
                with self.assertRaisesRegex(asm_source.GenerationError, message):
                    self.parse_from_mont(mutated)

    def test_generated_header_imports_semantics_and_excludes_isa_definitions(self):
        generated = gen_armv7em.gen_program(self.source)
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

    def test_transcribed_source_blocks_and_full_inventory(self):
        self.assertEqual(gen_armv7em.asm_block_inventory(self.source), gen_armv7em.ASM_BLOCKS)
        routines = gen_armv7em.all_routines(self.source)
        self.assertEqual(
            [routine.name for routine in routines],
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
        generated = gen_armv7em.gen_program(self.source)
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
        self.assertEqual(generated, gen_armv7em.OUTPUT.read_text())

    def test_from_mont_factored_routines_generate_proof_skeletons(self):
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
            routine.name: routine
            for routine in gen_armv7em.all_routines(self.source)
            if routine.name in routine_names
        }
        self.assertEqual(set(routines), routine_names)
        skeletons = {name: gen.skeleton(routine) for name, routine in routines.items()}
        for name, skeleton in skeletons.items():
            with self.subTest(routine=name):
                self.assertIn(f"generated skeleton for `{name}`", "\n".join(skeleton))
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

    def test_from_mont_helper_accounting_rejects_mutations(self):
        parsed = gen_armv7em.parse_from_mont(self.source)
        directions = asm_source.declaration_directions(parsed, "from_mont")
        flat = gen_armv7em.Emitter(directions, {"r0": "value", "r1": "modulus"})
        flat.bind_pointer("r0", "value")
        flat.bind_pointer("r1", "modulus")
        for pc, instruction in enumerate(parsed.instructions):
            flat.pc = pc
            flat.emit_instruction(instruction)
        registers = list(gen_armv7em.FROM_MONT_ACCUMULATOR) + ["r0"]
        round_entries = gen_armv7em._normalized_from_mont_round(flat, 9, registers)
        prefix = [dict(entry) for entry in round_entries if entry["pc"] < 3]
        suffix = [dict(entry) for entry in round_entries if entry["pc"] >= 3]
        gen_armv7em._check_from_mont_helper_accounting(round_entries, prefix, suffix)
        self.assertEqual([entry["pc"] for entry in prefix], [0, 0, 0, 1, 2])
        self.assertEqual(
            [(entry["name"], entry["fact"][0]) for entry in prefix],
            [("d", "rsbs"), ("w0", "fst"), ("c", "snd"), ("r12", "sbc"), ("r12", "and")],
        )
        self.assertEqual({entry["pc"] for entry in suffix}, set(range(3, 21)))
        self.assertEqual(
            gen_armv7em._from_mont_fingerprint(prefix + suffix),
            gen_armv7em._from_mont_fingerprint(round_entries),
        )
        quotient = [dict(entry) for entry in prefix if entry["pc"] == 0]
        self_sub = [dict(entry) for entry in prefix if entry["pc"] == 1]
        mask = [dict(entry) for entry in prefix if entry["pc"] == 2]
        gen_armv7em._check_from_mont_prefix_accounting(prefix, quotient, self_sub, mask)
        self.assertEqual(
            gen_armv7em._from_mont_fingerprint(quotient + self_sub + mask),
            gen_armv7em._from_mont_fingerprint(prefix),
        )
        final_entries = gen_armv7em._normalized_from_mont_final_reduction(flat)
        subtraction = [dict(entry) for entry in final_entries if entry["pc"] < 14]
        addback = [dict(entry) for entry in final_entries if entry["pc"] >= 14]
        gen_armv7em._check_from_mont_final_reduction_accounting(final_entries, subtraction, addback)
        self.assertEqual({entry["pc"] for entry in subtraction}, set(range(14)))
        self.assertEqual({entry["pc"] for entry in addback}, set(range(14, 31)))
        self.assertEqual(
            gen_armv7em._from_mont_fingerprint(subtraction + addback),
            gen_armv7em._from_mont_fingerprint(final_entries),
        )

        with self.assertRaisesRegex(asm_source.GenerationError, "sparse suffix does not cover"):
            gen_armv7em._check_from_mont_helper_accounting(round_entries, prefix, suffix[:-1])
        with self.assertRaisesRegex(asm_source.GenerationError, "quotient prefix does not cover"):
            gen_armv7em._check_from_mont_helper_accounting(
                round_entries, prefix[:-1], [prefix[-1], *suffix]
            )
        changed_suffix = [dict(entry) for entry in suffix]
        changed_suffix[0]["expr"] = "0"
        with self.assertRaisesRegex(asm_source.GenerationError, "do not reconstruct"):
            gen_armv7em._check_from_mont_helper_accounting(round_entries, prefix, changed_suffix)
        with self.assertRaisesRegex(
            asm_source.GenerationError, "self-subtraction helper does not cover"
        ):
            gen_armv7em._check_from_mont_prefix_accounting(
                prefix, quotient[:-1], [quotient[-1], *self_sub], mask
            )
        changed_mask = [dict(entry) for entry in mask]
        changed_mask[0]["expr"] = "0"
        with self.assertRaisesRegex(
            asm_source.GenerationError, "short prefix helpers do not reconstruct"
        ):
            gen_armv7em._check_from_mont_prefix_accounting(prefix, quotient, self_sub, changed_mask)
        with self.assertRaisesRegex(
            asm_source.GenerationError, "final subtraction helper does not cover"
        ):
            gen_armv7em._check_from_mont_final_reduction_accounting(
                final_entries, subtraction[:-1], [subtraction[-1], *addback]
            )
        with self.assertRaisesRegex(
            asm_source.GenerationError, "final add-back helper does not cover"
        ):
            gen_armv7em._check_from_mont_final_reduction_accounting(
                final_entries, subtraction, addback[1:]
            )
        changed_addback = [dict(entry) for entry in addback]
        changed_addback[0]["expr"] = "0"
        with self.assertRaisesRegex(
            asm_source.GenerationError, "final reduction helpers do not reconstruct"
        ):
            gen_armv7em._check_from_mont_final_reduction_accounting(
                final_entries, subtraction, changed_addback
            )

    def test_from_mont_rejects_offset_rotation_alias_selection_and_output_mutations(self):
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

    def test_existing_proved_and_cond_sub_routines_generate_proof_skeletons(self):
        proof_names = {"adc", "addWord", "sbb", "macWord", "inversionMacWords", "condSub"}
        routines = [
            routine
            for routine in gen_armv7em.all_routines(self.source)
            if routine.name in proof_names
        ]
        self.assertTrue(
            all(routine.skeleton_backend is gen_armv7em.SKELETON_BACKEND for routine in routines)
        )
        skeletons = {routine.name: "\n".join(gen.skeleton(routine)) for routine in routines}
        for name, skeleton in skeletons.items():
            with self.subTest(routine=name):
                self.assertIn(f"generated skeleton for `{name}`", skeleton)
                self.assertIn("< 2^32", skeleton)
        self.assertIn("b_carry'", skeletons["adc"])
        self.assertIn("b_c", skeletons["addWord"])
        self.assertIn("subc32_carry_le_one 0 borrow' 1", skeletons["sbb"])
        self.assertIn("d_high", skeletons["macWord"])
        self.assertIn("d_r3", skeletons["inversionMacWords"])
        self.assertIn("subc32_lin r2 r3 1", skeletons["condSub"])
        self.assertIn("sbc.w {keep}, {keep}, {keep}", skeletons["condSub"])
        self.assertIn("Nat.and_lt_two_pow", skeletons["condSub"])
        self.assertIn("b_result3 : result3 < 2^64", skeletons["condSub"])

    def test_batch_round_is_factored_and_divstep59_composes_three_batches(self):
        generated = gen_armv7em.gen_program(self.source)
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
        asm_inner = gen_armv7em._validate_exact_function(
            self.source, "batch", gen_armv7em.BATCH_RUST
        )
        instructions, _, _, _ = asm_source.parse_asm_invocation(
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
        routines = {routine.name: routine for routine in gen_armv7em.all_routines()}
        finish = routines["batchRoundFinish"]
        self.assertEqual(
            [entry["pc"] for entry in finish.emitter.entries[3:]], [27, 28, 29, 30, 30, 30, 30, 31]
        )
        finish_skeleton = "\n".join(gen.skeleton(finish))
        self.assertIn("word_step -clear pglo' := pglo", finish_skeleton)
        self.assertIn(
            "word_step -clear sub, remaining_1 := (remaining' + 2^32 - 1 - (1 - 1)) % 2^32",
            finish_skeleton,
        )
        round_skeleton = "\n".join(gen.skeleton(routines["batchRound"]))
        self.assertNotIn("lift_lets -merge at hres", round_skeleton)
        self.assertIn("word_step -clear pflo := st.pflo", round_skeleton)
        self.assertIn(
            "word_step -clear sub, tmp := (0 + 2^32 - d - (1 - 1)) % 2^32", round_skeleton
        )
        self.assertIn("word_step -clear swap := asr32 tmp 31", round_skeleton)
        for name in ("batchBlock", "divstep59Block"):
            skeleton = "\n".join(gen.skeleton(routines[name]))
            self.assertIn(f"generated skeleton for `{name}`", skeleton)
            self.assertIn("lift_lets -merge at hres", skeleton)
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

    def test_batch_and_divstep59_source_changes_are_rejected(self):
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

    def test_batch_rejects_invalid_zero_flag_and_operand_declarations(self):
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

    def test_cond_sub_rejects_malformed_offsets_uninitialized_memory_and_flags(self):
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

    def test_full_asm_inventory_changes_are_rejected(self):
        renamed = self.source.replace("fn batch(k:", "fn batch_changed(k:", 1)
        self.assert_rejected(renamed, "ARM asm block inventory differs")

    def test_unsupported_instruction_is_rejected(self):
        source = self.mutate_function(
            self.source,
            "adc",
            '"mov {carry_out}, #0",',
            '"mvn {carry_out}, #0",',
        )
        self.assert_rejected(source, "unsupported instruction mvn")

    def test_carry_read_before_flag_write_is_rejected(self):
        for function, old, new in (
            ("adc", '"adds {sum}, {a}, {b}",', '"mov {sum}, {a}",'),
            ("sbb", '"rsbs {borrow}, {borrow}, #0",', '"mov {borrow}, #0",'),
        ):
            with self.subTest(function=function):
                source = self.mutate_function(self.source, function, old, new)
                self.assert_rejected(source, "carry flag read while invalid")

    def test_inversion_mac_words_rejects_invalid_flags_memory_and_attributes(self):
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

    def test_uninitialized_and_input_only_register_use_is_rejected(self):
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

    def test_unwritten_read_write_output_is_rejected(self):
        read_write = self.mutate_function(
            self.source,
            "mac_word",
            '"umaal {low}, {high}, {a}, {b}",',
            '"adds {low}, {a}, {b}",',
        )
        self.assert_rejected(read_write, "output register high was never written")

    def test_memory_operands_are_rejected(self):
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

    def test_umaal_destination_aliases_are_rejected(self):
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

    def test_rsbs_form_is_restricted(self):
        source = self.mutate_function(
            self.source,
            "sbb",
            '"rsbs {borrow}, {borrow}, #0",',
            '"rsbs {borrow}, {borrow}, #1",',
        )
        self.assert_rejected(source, "rsbs supports only reverse subtraction from #0")

    def test_required_options_and_output_binding_are_validated(self):
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

    def test_second_inline_asm_definition_is_rejected(self):
        inline = gen_armv7em._inline_asm_function(self.source, "adc")
        config = gen_armv7em.RoutineConfig("adc_extra", "adcExtra", ("a", "b", "carry"), "test")
        renamed = inline.replace("fn adc(", "fn adc_extra(")
        duplicated = self.source + "\n" + renamed + "\n" + renamed
        with self.assertRaisesRegex(
            asm_source.GenerationError,
            "exactly one top-level `fn adc_extra\\(` containing asm!, found 2",
        ):
            gen_armv7em.emit_routine(duplicated, config)


if __name__ == "__main__":
    unittest.main()
