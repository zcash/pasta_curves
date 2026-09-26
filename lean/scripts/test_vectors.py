#!/usr/bin/env python3
# Copyright (c) 2026 the pasta-asm contributors.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for shared reference-vector parsing, filtering, and emission."""

from collections import Counter
import sys
import unittest

# Running this source-tree test should not leave lean/scripts/__pycache__ behind.
sys.dont_write_bytecode = True

import gen
import gen_aarch64
import gen_x86_64


def contract_counts(vectors, in_contract):
    counts = Counter()
    omitted = Counter()
    for op, key, vals in vectors:
        operands = [int(value, 16) for value in vals[:-1]]
        (counts if in_contract(op, key, operands) else omitted)[op] += 1
    return counts, omitted


class SharedVectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lines = gen.VECTORS.read_text().splitlines()
        cls.vectors = gen.parse_vectors(cls.lines)

    def test_corpus_shape_covers_both_fields_and_all_operations(self):
        self.assertEqual(
            Counter((key, op) for op, key, _ in self.vectors),
            Counter({
                ("Fp", "MUL"): 493,
                ("Fq", "MUL"): 493,
                ("Fp", "SQR"): 17,
                ("Fq", "SQR"): 17,
                ("Fp", "FROM"): 17,
                ("Fq", "FROM"): 17,
            }),
        )

    def test_aarch64_shared_emission_is_byte_identical(self):
        expected = gen_aarch64.OUT_VECTORS.read_text()
        self.assertEqual(gen_aarch64.gen_vectors(self.lines), expected)

    def test_public_contract_counts_match_existing_aarch64_coverage(self):
        included, omitted = contract_counts(self.vectors, gen.in_public_contract)
        self.assertEqual(included, Counter({"MUL": 806, "FROM": 34, "SQR": 34}))
        self.assertEqual(omitted, Counter({"MUL": 180}))


class X86VectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lines = gen.VECTORS.read_text().splitlines()
        cls.vectors = gen.parse_vectors(cls.lines)
        cls.generated = gen_x86_64.gen_vectors(cls.lines)

    def test_generated_file_records_cross_backend_provenance_and_compositions(self):
        self.assertIn("produced by the real AArch64 assembly", self.generated)
        self.assertIn("cross-backend checks", self.generated)
        self.assertIn("not captures from x86-64 hardware", self.generated)
        self.assertIn("`sqrMont`, the Rust composition `squareHi (squareLo value)`", self.generated)
        self.assertIn("x86-64's standalone `fromMont` assembly block", self.generated)
        self.assertIn("import PastaAsm.X86_64.Compositions", self.generated)
        self.assertEqual(self.generated.count("example :"), 874)
        self.assertIn(
            "-- 874 vectors; omitted as outside the public contracts: 180 MUL.",
            self.generated,
        )

    def test_backend_registers_current_generated_vector_file(self):
        outputs = dict(gen_x86_64.generated_outputs())
        self.assertIn(gen_x86_64.OUTPUT_VECTORS, outputs)
        self.assertEqual(outputs[gen_x86_64.OUTPUT_VECTORS], self.generated)
        self.assertEqual(gen_x86_64.OUTPUT_VECTORS.read_text(), self.generated)


if __name__ == "__main__":
    unittest.main()
