#!/usr/bin/env python3
"""Regression tests for fail-closed Rust surrounding-code parsing."""

import re
import sys
import unittest

# Running this source-tree test should not leave lean/scripts/__pycache__ behind.
sys.dont_write_bytecode = True

from asm2lean import rust
from pasta.architectures import aarch64 as aarch64_blocks
from pasta.architectures import x86_64 as x86_64_blocks


class SurroundingCodeTests(unittest.TestCase):
    aarch64_source: str
    x86_64_source: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.aarch64_source = aarch64_blocks.SOURCE.read_text()
        cls.x86_64_source = x86_64_blocks.SOURCE.read_text()

    @staticmethod
    def mutate_function(source: str, name: str, old: str, new: str) -> str:
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
    def generate_aarch64(source: str) -> str:
        return aarch64_blocks.text(source)

    def test_current_sources_generate_committed_output(self) -> None:
        self.assertEqual(
            self.generate_aarch64(self.aarch64_source),
            aarch64_blocks.OUTPUT.read_text(),
        )
        self.assertEqual(
            x86_64_blocks.text(self.x86_64_source),
            x86_64_blocks.OUTPUT.read_text(),
        )

    def test_inv_shadowing_is_rejected_by_both_backends(self) -> None:
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

    def test_u32_tuple_outputs_are_parsed(self) -> None:
        source = """fn f(a: u32) -> (u32, u32) {
    let low: u32;
    let high: u32;
    unsafe {
        asm!(
            "umull {low}, {high}, {a}, {a}",
            a = in(reg) a,
            low = out(reg) low,
            high = out(reg) high,
            options(pure, nomem, nostack),
        );
    }
    (low, high)
}
"""
        parsed = rust.parse_function(
            source,
            "f",
            ["a"],
            2,
            required_options={"pure", "nomem", "nostack"},
        )
        self.assertEqual(parsed.returns, ("low", "high"))
        self.assertEqual(rust.returned_registers(parsed, "f"), ("low", "high"))

    def test_direct_scalar_local_initializers_are_parsed(self) -> None:
        source = """fn f(a: u32, b: u32) -> (u32, u32) {
    let mut low = a;
    let mut high = b;
    unsafe {
        asm!(
            "umaal {low}, {high}, {a}, {b}",
            low = inout(reg) low,
            high = inout(reg) high,
            a = in(reg) a,
            b = in(reg) b,
            options(pure, nomem, nostack),
        );
    }
    (low, high)
}
"""
        parsed = rust.parse_function(
            source,
            "f",
            ["a", "b"],
            2,
            required_options={"pure", "nomem", "nostack"},
        )
        self.assertEqual(parsed.locals, {"low": ("a", None), "high": ("b", None)})
        self.assertEqual(rust.returned_registers(parsed, "f"), ("low", "high"))

    def test_zero_initialized_array_and_direct_return_are_parsed(self) -> None:
        source = """fn f(a: u32) -> [u64; 4] {
    let mut output = [0u64; 4];
    unsafe {
        asm!("str {a}, [{output}, #0]",
            a = in(reg) a, output = in(reg) output.as_mut_ptr(), options(nostack));
    }
    output
}
"""
        parsed = rust.parse_function(
            source,
            "f",
            ["a"],
            1,
            allowed_options={"nostack"},
            required_options={"nostack"},
        )
        self.assertEqual(parsed.zero_arrays, {"output": 4})
        self.assertEqual(parsed.returns, ("output",))

        with self.assertRaisesRegex(rust.GenerationError, "unsupported code before asm"):
            rust.parse_function(
                source.replace("[0u64; 4]", "[1u64; 4]"),
                "f",
                ["a"],
                1,
                allowed_options={"nostack"},
                required_options={"nostack"},
            )

    def test_identity_scalar_shadow_is_parsed_but_other_shadowing_is_rejected(self) -> None:
        source = """fn f(a: u32) -> (u32, u32) {
    let mut a = a;
    let high: u32;
    #[expect(unused_assignments, reason = "temporary assembly value")]
    unsafe {
        asm!(
            "umaal {a}, {high}, {a_in}, {a_in}",
            a = inout(reg) a,
            high = out(reg) high,
            a_in = in(reg) a,
            options(pure, nomem, nostack),
        );
    }
    (a, high)
}
"""
        parsed = rust.parse_function(
            source,
            "f",
            ["a"],
            2,
            required_options={"pure", "nomem", "nostack"},
        )
        self.assertEqual(parsed.locals, {"a": ("a", None)})
        with self.assertRaisesRegex(rust.GenerationError, "local reads non-argument missing"):
            rust.parse_function(
                source.replace("let mut a = a;", "let mut a = missing;"),
                "f",
                ["a"],
                2,
                required_options={"pure", "nomem", "nostack"},
            )
        with self.assertRaisesRegex(rust.GenerationError, "unsupported attribute before asm"):
            rust.parse_function(
                source.replace(
                    '#[expect(unused_assignments, reason = "temporary assembly value")]',
                    "#[allow(unused_assignments)]",
                ),
                "f",
                ["a"],
                2,
                required_options={"pure", "nomem", "nostack"},
            )

    def test_explicit_discarded_outputs_are_validated(self) -> None:
        source = """fn f(a: u32) -> (u32, u32) {
    let mut low = a;
    let mut high = a;
    let mut temporary = a;
    unsafe {
        asm!(
            "umaal {low}, {high}, {a}, {temporary}",
            low = inout(reg) low,
            high = inout(reg) high,
            temporary = inout(reg) temporary,
            a = in(reg) a,
            options(pure, nomem, nostack),
        );
    }
    (low, high)
}
"""
        parsed = rust.parse_function(
            source,
            "f",
            ["a"],
            2,
            required_options={"pure", "nomem", "nostack"},
        )
        with self.assertRaisesRegex(
            rust.GenerationError, "named asm outputs not returned.*temporary"
        ):
            rust.returned_registers(parsed, "f")
        self.assertEqual(
            rust.returned_registers(
                parsed,
                "f",
                discarded_outputs={"temporary"},
            ),
            ("low", "high"),
        )
        with self.assertRaisesRegex(
            rust.GenerationError, "configured discarded values are not asm outputs"
        ):
            rust.returned_registers(parsed, "f", discarded_outputs={"missing"})

    def test_direct_scalar_local_must_read_an_argument_and_only_initialize(self) -> None:
        source = """fn f(a: u32) -> (u32, u32) {
    let mut low = missing;
    let mut high = a;
    unsafe {
        asm!(
            "umaal {low}, {high}, {a}, {a}",
            low = inout(reg) low,
            high = inout(reg) high,
            a = in(reg) a,
            options(pure, nomem, nostack),
        );
    }
    (low, high)
}
"""
        for mutated, message in (
            (source, "local reads non-argument missing"),
            (
                source.replace("let mut low = missing;", "let mut low = a + 1;"),
                "unsupported code before asm!",
            ),
        ):
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(rust.GenerationError, message),
            ):
                rust.parse_function(
                    mutated,
                    "f",
                    ["a"],
                    2,
                    required_options={"pure", "nomem", "nostack"},
                )

    def test_qualified_asm_path_is_parsed(self) -> None:
        source = """fn f(a: u32) -> (u32, u32) {
    let low: u32;
    let high: u32;
    unsafe {
        core::arch::asm!(
            "umull {low}, {high}, {a}, {a}",
            a = in(reg) a,
            low = out(reg) low,
            high = out(reg) high,
            options(pure, nomem, nostack),
        );
    }
    (low, high)
}
"""
        parsed = rust.parse_function(
            source,
            "f",
            ["a"],
            2,
            required_options={"pure", "nomem", "nostack"},
        )
        self.assertEqual(parsed.instructions, ("umull {low}, {high}, {a}, {a}",))

    def test_unsupported_output_declarations_and_tuple_returns_are_rejected(self) -> None:
        source = """fn f(a: u32) -> (u32, u32) {
    let (low, high): (u32, u32);
    unsafe {
        asm!(
            "umull {low}, {high}, {a}, {a}",
            a = in(reg) a,
            low = out(reg) low,
            high = out(reg) high,
            options(pure, nomem, nostack),
        );
    }
    (low, high)
}
"""
        cases = {
            "unsupported output declaration": source.replace("(u32, u32);", "(u32, u64);"),
            "unsupported code after asm!": source.replace("(low, high)\n}", "(low,)\n}"),
            "source output tuple has 2 values, expected 1": source,
        }
        for message, mutated in cases.items():
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(rust.GenerationError, message),
            ):
                rust.parse_function(
                    mutated,
                    "f",
                    ["a"],
                    1 if "expected 1" in message else 2,
                    required_options={"pure", "nomem", "nostack"},
                )

    def test_postasm_output_mutation_is_rejected_by_both_backends(self) -> None:
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

    def test_comments_are_allowed_between_surrounding_grammar_tokens(self) -> None:
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


class TriviaNormalizationTests(unittest.TestCase):
    def test_code_without_trivia_preserves_literals_and_removes_spacing(self) -> None:
        source = "fn f(a: u32) { let text = \"a b\"; /* ignored */ let c = ' '; }"
        self.assertEqual(
            rust.code_without_trivia(source),
            "fnf(a:u32){lettext=\"a b\";letc=' ';}",
        )


class MacroTests(unittest.TestCase):
    def parse(
        self, definitions: str
    ) -> dict[str, dict[str, tuple[str, ...] | rust.ParameterizedMacro]]:
        return rust.parse_macros(definitions)

    def test_arms_expand_literals_and_earlier_arms(self) -> None:
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

    def test_literal_macro_arguments_expand_stringify(self) -> None:
        source = """macro_rules! load {
    ($offset:literal) => { concat!("ldr {word}, [{base}, #", stringify!($offset), "]") };
}
fn f(a: u32) -> (u32, u32) {
    let low: u32;
    let high: u32;
    unsafe {
        asm!(load!(12), "mov {high}, #0",
            base = in(reg) a, word = out(reg) low, high = out(reg) high,
            options(nostack));
    }
    (low, high)
}
"""
        parsed = rust.parse_function(
            source,
            "f",
            ["a"],
            2,
            allowed_options={"nostack"},
            required_options={"nostack"},
        )
        self.assertEqual(parsed.instructions[0], "ldr {word}, [{base}, #12]")
        self.assertEqual(parsed.origins[0], rust.MacroOrigin("load", "12", 0))

        with self.assertRaisesRegex(rust.GenerationError, "has no arm for `OFFSET`"):
            rust.parse_function(
                source.replace("load!(12)", "load!(OFFSET)"),
                "f",
                ["a"],
                2,
                allowed_options={"nostack"},
                required_options={"nostack"},
            )

    def test_identifier_macro_arguments_expand_nested_concat_and_stringify(self) -> None:
        source = """macro_rules! rotate {
    ($a:ident, $b:ident) => { concat!(
        concat!("add ", stringify!($a), ", ", stringify!($a), ", #1\\n"),
        concat!("add ", stringify!($b), ", ", stringify!($b), ", #1\\n"),
    ) };
}
fn f(mut a: u32, mut b: u32) -> (u32, u32) {
    unsafe {
        asm!(rotate!(a, b), a = inout(reg) a, b = inout(reg) b,
            options(pure, nomem, nostack));
    }
    (a, b)
}
"""
        parsed = rust.parse_function(
            source,
            "f",
            ["a", "b"],
            2,
            required_options={"pure", "nomem", "nostack"},
        )
        self.assertEqual(parsed.instructions, ("add a, a, #1", "add b, b, #1"))
        self.assertEqual(
            parsed.origins,
            (
                rust.MacroOrigin("rotate", "a, b", 0),
                rust.MacroOrigin("rotate", "a, b", 0),
            ),
        )

    def test_malformed_macros_are_rejected(self) -> None:
        cases = {
            "escape only": 'macro_rules! m { () => { "add {a}, {a}, #1\\t\\n" }; }',
            "no arm for `x`": 'macro_rules! m { () => { concat!(m!(x), "a\\n") }; }',
            "unknown macro n!": 'macro_rules! m { () => { concat!(n!(), "a\\n") }; }',
            "duplicate arm": 'macro_rules! m { () => { "a\\n" }; () => { "b\\n" }; }',
            "unsupported macro pattern": 'macro_rules! m { ($x:tt) => { "a\\n" }; }',
            "declared metavariable": 'macro_rules! m { ($x:ident) => { concat!(stringify!($y), "\\n") }; }',
            "string literal or `concat!`": 'macro_rules! m { () => { format!("a\\n") }; }',
            "empty line": 'macro_rules! m { () => { "a\\n\\n" }; }',
        }
        for message, source in cases.items():
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(rust.GenerationError, message),
            ):
                self.parse(source)

    def test_parameterized_macro_argument_kinds_are_rejected(self) -> None:
        source = """macro_rules! m {
    ($word:ident, $offset:literal) => {
        concat!("ldr ", stringify!($word), ", [r0, #", stringify!($offset), "]\\n")
    };
}
fn f(a: u32) -> (u32, u32) {
    let low: u32;
    let high: u32;
    unsafe {
        asm!(m!(r2, 4), "mov {high}, #0", low = out(reg) low, high = out(reg) high,
            options(nostack));
    }
    (low, high)
}
"""
        parsed = rust.parse_function(
            source, "f", ["a"], 2, allowed_options={"nostack"}, required_options={"nostack"}
        )
        self.assertEqual(parsed.instructions[0], "ldr r2, [r0, #4]")
        with self.assertRaisesRegex(rust.GenerationError, "has no arm for `4, r2`"):
            rust.parse_function(
                source.replace("m!(r2, 4)", "m!(4, r2)"),
                "f",
                ["a"],
                2,
                allowed_options={"nostack"},
                required_options={"nostack"},
            )

    def test_invocations_in_template_position_expand_with_origins(self) -> None:
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

    def test_block_function_is_the_top_level_one(self) -> None:
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
