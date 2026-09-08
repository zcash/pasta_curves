"""Unit tests for the AArch64 interpreter in `asm_aarch64`.

Run with `python3 -m unittest discover -s tests -t .` from the parent
directory.
"""

import unittest

import asm_aarch64 as asm
from asm_aarch64 import M64
from asm_mutant_witnesses import ASM_SOURCE, MODULI


def idle_cpu() -> asm.Cpu:
    """A Cpu over a program that does nothing, for exercising registers."""
    return asm.Cpu(asm.Program("_test:\n    ret\n"))


def run(body: str, **registers: int) -> asm.Cpu:
    """Execute a snippet with the given registers preset, returning the Cpu."""
    program = asm.Program("_test:\n" + body + "\n    ret\n")
    cpu = asm.Cpu(program)
    for name, value in registers.items():
        cpu.write(name, value)
    cpu.run(program.labels["_test"])
    return cpu


class Parsing(unittest.TestCase):
    def test_operands_keep_bracketed_addressing_together(self) -> None:
        self.assertEqual(
            asm.split_operands("x1, x2, [x3, #16]"), ["x1", "x2", "[x3, #16]"]
        )

    def test_comments_labels_and_directives(self) -> None:
        program = asm.Program(
            ".text\n"
            ".globl _f\n"
            "_f:\n"
            "    mul x1,x2,x3 // a comment\n"
            "    .long 3573752639\n"
            "    ret\n"
        )
        self.assertEqual(program.labels, {"_f": 0})
        self.assertEqual(
            [mnemonic for mnemonic, _, _ in program.instructions],
            ["mul", "nop", "ret"],
        )
        self.assertEqual(program.instructions[0][1], ["x1", "x2", "x3"])

    def test_replacing_shifts_only_the_labels_after_it(self) -> None:
        program = asm.Program("_a:\n    mul x1,x2,x3\n_b:\n    ret\n")
        self.assertEqual((program.labels["_a"], program.labels["_b"]), (0, 1))
        grown = program.replacing(0, [("adds", ["x1", "x2", "x3"], 1)] * 3)
        self.assertEqual((grown.labels["_a"], grown.labels["_b"]), (0, 3))


class Registers(unittest.TestCase):
    def test_xzr_reads_zero_and_discards_writes(self) -> None:
        cpu = idle_cpu()
        cpu.write("xzr", 42)
        self.assertEqual(cpu.read("xzr"), 0)

    def test_writes_are_truncated_to_64_bits(self) -> None:
        cpu = idle_cpu()
        cpu.write("x1", (1 << 64) + 7)
        self.assertEqual(cpu.read("x1"), 7)

    def test_32_bit_views_are_rejected(self) -> None:
        cpu = idle_cpu()
        for name in ("w0", "w30", "wzr"):
            with self.assertRaises(ValueError):
                cpu.read(name)


class Flags(unittest.TestCase):
    def test_adds_sets_carry_only_on_overflow(self) -> None:
        self.assertEqual(run("    adds x0,x1,x2", x1=M64, x2=1).carry, 1)
        self.assertEqual(run("    adds x0,x1,x2", x1=M64, x2=1).read("x0"), 0)
        self.assertEqual(run("    adds x0,x1,x2", x1=1, x2=1).carry, 0)

    def test_adcs_adds_the_incoming_carry(self) -> None:
        # 128-bit addition: (2^64 - 1) + 1 carries into the high limb.
        cpu = run(
            "    adds x0,x2,x3\n    adcs x1,x4,x5", x2=M64, x3=1, x4=7, x5=0
        )
        self.assertEqual((cpu.read("x1"), cpu.read("x0")), (8, 0))

    def test_adc_does_not_disturb_the_flags(self) -> None:
        # `adc` must leave the carry for whatever consumes it next.
        cpu = run(
            "    adds x0,x1,x2\n    adc x3,xzr,xzr\n    adcs x4,xzr,xzr",
            x1=M64,
            x2=1,
        )
        self.assertEqual(cpu.read("x3"), 1)
        self.assertEqual(cpu.read("x4"), 1)

    def test_subs_sets_carry_when_there_is_no_borrow(self) -> None:
        # AArch64 inverts the intuition: carry set means the subtraction did
        # not borrow, which is what `lo` tests for.
        self.assertEqual(run("    subs x0,x1,x2", x1=5, x2=3).carry, 1)
        self.assertEqual(run("    subs x0,x1,x2", x1=3, x2=5).carry, 0)
        self.assertEqual(run("    subs x0,x1,x2", x1=3, x2=3).carry, 1)

    def test_subs_wraps_on_borrow(self) -> None:
        self.assertEqual(run("    subs x0,x1,x2", x1=0, x2=1).read("x0"), M64)

    def test_sbcs_consumes_the_incoming_borrow(self) -> None:
        # 128-bit subtraction of 1 from 2^64 borrows into the high limb.
        cpu = run(
            "    subs x0,x2,x3\n    sbcs x1,x4,x5", x2=0, x3=1, x4=1, x5=0
        )
        self.assertEqual((cpu.read("x1"), cpu.read("x0")), (0, M64))

    def test_sbcs_with_zero_operands_passes_the_borrow_through(self) -> None:
        # The identity the backend relies on at its final comparison.
        for x1, x2, expected in ((5, 3, 1), (3, 5, 0)):
            cpu = run("    subs xzr,x1,x2\n    sbcs xzr,xzr,xzr", x1=x1, x2=x2)
            self.assertEqual(cpu.carry, expected)


class Arithmetic(unittest.TestCase):
    def test_mul_and_umulh_split_the_full_product(self) -> None:
        a, b = 0xDEAD_BEEF_1234_5678, 0xFEED_FACE_8765_4321
        cpu = run("    mul x0,x2,x3\n    umulh x1,x2,x3", x2=a, x3=b)
        self.assertEqual(cpu.read("x0"), (a * b) & M64)
        self.assertEqual(cpu.read("x1"), (a * b) >> 64)

    def test_shifts_truncate_to_64_bits(self) -> None:
        cpu = run("    lsl x0,x2,#62\n    lsr x1,x2,#2", x2=0xFF)
        self.assertEqual(cpu.read("x0"), (0xFF << 62) & M64)
        self.assertEqual(cpu.read("x1"), 0xFF >> 2)

    def test_csel_lo_selects_on_borrow(self) -> None:
        borrowed = run(
            "    subs xzr,x1,x2\n    csel x0,x3,x4,lo", x1=3, x2=5, x3=11, x4=22
        )
        self.assertEqual(borrowed.read("x0"), 11)
        not_borrowed = run(
            "    subs xzr,x1,x2\n    csel x0,x3,x4,lo", x1=5, x2=3, x3=11, x4=22
        )
        self.assertEqual(not_borrowed.read("x0"), 22)


class Memory(unittest.TestCase):
    def test_pair_store_and_load_round_trip(self) -> None:
        cpu = run(
            "    stp x2,x3,[x1]\n    ldp x4,x5,[x1]", x1=0x900, x2=7, x3=9
        )
        self.assertEqual((cpu.read("x4"), cpu.read("x5")), (7, 9))
        self.assertEqual(cpu.memory[0x900], 7)
        self.assertEqual(cpu.memory[0x908], 9)

    def test_offsets_including_scaled_expressions(self) -> None:
        cpu = run("    str x2,[x1,#16]\n    ldr x3,[x1,8*2]", x1=0x900, x2=5)
        self.assertEqual(cpu.read("x3"), 5)

    def test_pre_index_writes_the_base_back(self) -> None:
        cpu = run("    stp x2,x3,[x1,#-64]!", x1=0x900, x2=1, x3=2)
        self.assertEqual(cpu.read("x1"), 0x900 - 64)
        self.assertEqual(cpu.memory[0x900 - 64], 1)

    def test_post_index_writes_the_base_back_after_the_access(self) -> None:
        cpu = run("    ldr x2,[x1],#64", x1=0x900)
        self.assertEqual(cpu.read("x1"), 0x900 + 64)

    def test_unwritten_memory_reads_as_zero(self) -> None:
        self.assertEqual(run("    ldr x2,[x1]", x1=0x4000).read("x2"), 0)


class ControlFlow(unittest.TestCase):
    def test_bl_returns_to_the_following_instruction(self) -> None:
        program = asm.Program(
            "_f:\n"
            "    bl _g\n"
            "    add x1,x1,#1\n"
            "    ret\n"
            "_g:\n"
            "    add x1,x1,#10\n"
            "    ret\n"
        )
        cpu = asm.Cpu(program)
        cpu.run(program.labels["_f"])
        self.assertEqual(cpu.read("x1"), 11)

    def test_runaway_execution_is_reported(self) -> None:
        program = asm.Program("_f:\n    bl _f\n    ret\n")
        with self.assertRaises(RuntimeError):
            asm.Cpu(program).run(program.labels["_f"], step_limit=100)


class Limbs(unittest.TestCase):
    """`limbs` and `from_limbs` are inverse to each other, both ways round."""

    VALUES = (
        0,
        1,
        M64,
        1 << 64,
        (1 << 128) - 1,
        (1 << 255),
        (1 << 256) - 1,
        0x0123456789ABCDEF_FEDCBA9876543210_0F1E2D3C4B5A6978_8796A5B4C3D2E1F0,
    )

    def test_from_limbs_undoes_limbs(self) -> None:
        for value in self.VALUES:
            with self.subTest(value=value):
                self.assertEqual(asm.from_limbs(asm.limbs(value)), value)

    def test_limbs_undoes_from_limbs(self) -> None:
        for value in self.VALUES:
            split = asm.limbs(value)
            with self.subTest(value=value):
                self.assertEqual(asm.limbs(asm.from_limbs(split)), split)

    def test_always_four_limbs_each_in_range(self) -> None:
        for value in self.VALUES:
            with self.subTest(value=value):
                split = asm.limbs(value)
                self.assertEqual(len(split), 4)
                self.assertTrue(all(0 <= limb <= M64 for limb in split))

    def test_least_significant_limb_first(self) -> None:
        self.assertEqual(asm.limbs((1 << 64) + 3), [3, 1, 0, 0])
        self.assertEqual(asm.from_limbs([3, 1, 0, 0]), (1 << 64) + 3)

    def test_limbs_keeps_only_the_low_256_bits(self) -> None:
        value = (1 << 256) + 5
        self.assertEqual(asm.limbs(value), asm.limbs(5))
        self.assertEqual(asm.from_limbs(asm.limbs(value)), 5)


class ShippedBackend(unittest.TestCase):
    """The interpreter against the real assembly and the mathematical result."""

    def setUp(self) -> None:
        self.program = asm.Program(ASM_SOURCE.read_text())

    def test_routines_match_the_montgomery_definitions(self) -> None:
        import random

        for field, p in MODULI.items():
            inv = (-pow(p, -1, 1 << 64)) % (1 << 64)
            r_inverse = pow(1 << 256, -1, p)
            rng = random.Random(0)
            values = [0, 1, p - 1, (1 << 256) % p] + [
                rng.randrange(p) for _ in range(50)
            ]
            for a in values:
                b = rng.randrange(p)
                with self.subTest(field=field, a=a, b=b):
                    self.assertEqual(
                        asm.mul(self.program, a, b, p, inv)[0],
                        a * b * r_inverse % p,
                    )
                    self.assertEqual(
                        asm.square(self.program, a, p, inv)[0],
                        a * a * r_inverse % p,
                    )
                    self.assertEqual(
                        asm.from_mont(self.program, a, p, inv)[0],
                        a * r_inverse % p,
                    )

    def test_the_trace_records_every_flag_consuming_instruction(self) -> None:
        p = MODULI["Fp"]
        inv = (-pow(p, -1, 1 << 64)) % (1 << 64)
        _, trace = asm.mul(self.program, p - 1, p - 2, p, inv)
        consuming = {"adc", "adcs", "sbcs", "csel"}

        for index, carry in trace:
            self.assertIn(self.program.instructions[index][0], consuming)
            self.assertIn(carry, (0, 1))

        # `mul` is straight-line, so each of its flag-consuming instructions
        # is traced exactly once.
        start = self.program.labels["_pasta_curves_mul_mont_pasta"]
        end = next(
            index
            for index in range(start, len(self.program.instructions))
            if self.program.instructions[index][0] == "ret"
        )
        expected = sum(
            1
            for mnemonic, _, _ in self.program.instructions[start:end]
            if mnemonic in consuming
        )
        self.assertEqual(len(trace), expected)
        self.assertEqual(len({index for index, _ in trace}), expected)


if __name__ == "__main__":
    unittest.main()
