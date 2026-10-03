"""Shared fail-closed extraction of Rust inline-assembly source.

This module parses only the Rust and ``asm!`` surface syntax needed by the Pasta
backends. Architecture-specific instruction semantics belong in their emitters.
Unsupported syntax is rejected rather than guessed.

A template line of an ``asm!`` block may be an invocation of a ``macro_rules!`` macro of the
same source file (a repeated instruction sequence, written once). `parse_macros` reads the
definitions, restricted to arms whose pattern is a sequence of plain identifiers and whose body
is a string literal or a ``concat!`` of literals and earlier arms, and `parse_function` expands
each invocation into its instructions, recording where they came from in `origins`.
"""

import dataclasses
import difflib
import re
import sys
from collections.abc import Sequence
from pathlib import Path


class GenerationError(ValueError):
    """The Rust source uses a construct the mechanical model does not accept."""


_IDENTIFIER_PATTERN = r"[A-Za-z_]\w*"


@dataclasses.dataclass(frozen=True)
class Declaration:
    name: str
    kind: str
    value: str
    output: str | None
    fixed: bool = False


@dataclasses.dataclass(frozen=True)
class MacroOrigin:
    """Where an instruction of an `asm!` block came from, when a macro invocation in template
    position expanded to it: the macro's name, the arm's pattern (its tokens, space-separated),
    and the invocation's index among the block's macro invocations."""

    macro: str
    arm: str
    site: int


@dataclasses.dataclass(frozen=True)
class ParsedFunction:
    instructions: tuple[str, ...]
    declarations: tuple[Declaration, ...]
    locals: dict[str, tuple[str, int]]
    returns: tuple[str, ...]
    options: set[str]
    # One entry per instruction: `None` for a literal template string, else its macro origin.
    origins: tuple[MacroOrigin | None, ...] = ()


def _line_comment_end(text: str, start: int) -> int:
    end = text.find("\n", start + 2)
    return len(text) if end < 0 else end


def _block_comment_end(text: str, start: int) -> int:
    depth = 1
    pos = start + 2
    while pos < len(text):
        if text.startswith("/*", pos):
            depth += 1
            pos += 2
        elif text.startswith("*/", pos):
            depth -= 1
            pos += 2
            if depth == 0:
                return pos
        else:
            pos += 1
    raise GenerationError("unterminated block comment")


def _quoted_end(text: str, start: int, quote: str = '"') -> tuple[int, bool]:
    """The end of an ordinary Rust string/character token and whether it has an escape."""
    pos = start + 1
    escaped = False
    while pos < len(text):
        if text[pos] == "\\":
            escaped = True
            pos += 2
        elif text[pos] == quote:
            return pos + 1, escaped
        else:
            pos += 1
    raise GenerationError("unterminated quoted literal")


def _raw_string_end(text: str, start: int) -> int | None:
    """The end of a Rust raw string beginning at `start`, or `None` if there is none."""
    pos = start
    if text.startswith("br", pos):
        pos += 1
    if pos >= len(text) or text[pos] != "r":
        return None
    pos += 1
    hashes = 0
    while pos < len(text) and text[pos] == "#":
        hashes += 1
        pos += 1
    if pos >= len(text) or text[pos] != '"':
        return None
    terminator = '"' + "#" * hashes
    end = text.find(terminator, pos + 1)
    if end < 0:
        raise GenerationError("unterminated raw string")
    return end + len(terminator)


def _char_literal_end(text: str, start: int) -> int | None:
    """Recognize a character literal without treating a Rust lifetime as one."""
    if start + 2 < len(text) and text[start + 2] == "'":
        return start + 3
    if start + 1 < len(text) and text[start + 1] == "\\":
        end, _ = _quoted_end(text, start, "'")
        return end
    return None


def _skip_trivia(text: str, start: int) -> int:
    pos = start
    while pos < len(text):
        if text[pos].isspace():
            pos += 1
        elif text.startswith("//", pos):
            pos = _line_comment_end(text, pos)
        elif text.startswith("/*", pos):
            pos = _block_comment_end(text, pos)
        else:
            break
    return pos


def masked_noncode(text: str) -> str:
    """Replace comments and literals with spaces, preserving length and newlines."""
    masked = list(text)
    pos = 0
    while pos < len(text):
        end: int | None = None
        if text.startswith("//", pos):
            end = _line_comment_end(text, pos)
        elif text.startswith("/*", pos):
            end = _block_comment_end(text, pos)
        else:
            end = _raw_string_end(text, pos)
            if end is None and text[pos] == '"':
                end, _ = _quoted_end(text, pos)
            elif end is None and text[pos] == "'":
                end = _char_literal_end(text, pos)
        if end is None:
            pos += 1
            continue
        for index in range(pos, end):
            if masked[index] != "\n":
                masked[index] = " "
        pos = end
    return "".join(masked)


def matching_delimiter(text: str, opening: int, left: str, right: str) -> int:
    """Return the matching delimiter, ignoring Rust literals and nested comments."""
    if opening >= len(text) or text[opening] != left:
        raise GenerationError("internal delimiter search error")
    depth = 1
    pos = opening + 1
    while pos < len(text):
        next_pos = _skip_trivia(text, pos)
        if next_pos != pos:
            pos = next_pos
            continue
        raw_end = _raw_string_end(text, pos)
        if raw_end is not None:
            pos = raw_end
            continue
        if text[pos] == '"':
            pos, _ = _quoted_end(text, pos)
            continue
        if text[pos] == "'":
            char_end = _char_literal_end(text, pos)
            if char_end is not None:
                pos = char_end
                continue
        if text[pos] == left:
            depth += 1
        elif text[pos] == right:
            depth -= 1
            if depth == 0:
                return pos
        pos += 1
    raise GenerationError(f"unclosed {left}")


def _strip_comments(text: str) -> str:
    """Remove comments from one asm argument while retaining literals verbatim."""
    pieces: list[str] = []
    pos = 0
    while pos < len(text):
        if text.startswith("//", pos):
            end = _line_comment_end(text, pos)
            pieces.append("\n" if end < len(text) else " ")
            pos = end
        elif text.startswith("/*", pos):
            pos = _block_comment_end(text, pos)
            pieces.append(" ")
        else:
            raw_end = _raw_string_end(text, pos)
            if raw_end is not None:
                pieces.append(text[pos:raw_end])
                pos = raw_end
            elif text[pos] == '"':
                end, _ = _quoted_end(text, pos)
                pieces.append(text[pos:end])
                pos = end
            else:
                pieces.append(text[pos])
                pos += 1
    return "".join(pieces)


def _argument_end(text: str, start: int) -> int:
    """Find an asm argument's top-level comma, validating its delimiters."""
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack: list[str] = []
    pos = start
    while pos < len(text):
        next_pos = _skip_trivia(text, pos)
        if next_pos != pos:
            pos = next_pos
            continue
        raw_end = _raw_string_end(text, pos)
        if raw_end is not None:
            pos = raw_end
            continue
        if text[pos] == '"':
            pos, _ = _quoted_end(text, pos)
            continue
        char = text[pos]
        if char in pairs:
            stack.append(pairs[char])
        elif char in pairs.values():
            if not stack or stack.pop() != char:
                raise GenerationError(f"unmatched delimiter {char} in asm argument")
        elif char == "," and not stack:
            return pos
        pos += 1
    if stack:
        raise GenerationError(f"unclosed delimiter {stack[-1]} in asm argument")
    return len(text)


def parse_declaration(
    text: str,
    *,
    reserved_names: set[str] = frozenset(),
    fixed_registers: set[str] = frozenset(),
    const_operands: set[str] = frozenset(),
) -> Declaration:
    """Parse one inline-assembly declaration under backend-specific restrictions."""
    declaration = text.strip().rstrip(",").strip()
    const = re.fullmatch(r"([A-Za-z_]\w*)\s*=\s*const\s+(.+)", declaration)
    if const:
        name, value = const.group(1), const.group(2).strip()
        if name in reserved_names:
            raise GenerationError(f"reserved operand name {name}: {text.strip()}")
        if value not in const_operands:
            raise GenerationError(f"unsupported const operand binding: {text.strip()}")
        return Declaration(name, "const", value, None)

    fixed = re.fullmatch(r'(in|out|inout)\("([^"]+)"\)\s+(.+)', declaration)
    if fixed:
        kind, register, value = fixed.groups()
        if register not in fixed_registers or kind != "out" or value.strip() != "_":
            raise GenerationError(f"unsupported fixed-register binding: {text.strip()}")
        return Declaration(register, kind, value.strip(), None, fixed=True)

    named = re.fullmatch(r"([A-Za-z_]\w*)\s*=\s*(in|out|inout)\(([^)]+)\)\s+(.+)", declaration)
    if not named:
        raise GenerationError(f"unsupported operand binding: {text.strip()}")
    name, kind, constraint, body = named.groups()
    if name in reserved_names:
        raise GenerationError(f"reserved operand name {name}: {text.strip()}")
    if constraint.strip() != "reg":
        raise GenerationError(f"unsupported operand constraint: {text.strip()}")
    pieces = [piece.strip() for piece in body.split("=>")]
    if len(pieces) > 2 or (len(pieces) == 2 and kind != "inout"):
        raise GenerationError(f"unsupported operand binding: {text.strip()}")
    value = pieces[0]
    output = pieces[1] if len(pieces) == 2 else None
    if kind == "in" and output is not None:
        raise GenerationError(f"input operand has an output: {text.strip()}")
    if kind == "out" and output is not None:
        raise GenerationError(f"output operand uses `=>`: {text.strip()}")
    if kind == "out":
        output = value
    elif kind == "inout" and output is None:
        if not re.fullmatch(r"[A-Za-z_]\w*", value):
            raise GenerationError(f"inout without `=>` must bind a local variable: {text.strip()}")
        output = value
    return Declaration(name, kind, value, output)


_MACRO_INVOCATION = re.compile(rf"({_IDENTIFIER_PATTERN})\s*!\s*\(")


def _arm_key(pattern: str) -> str:
    """A macro arm's pattern as a key: its tokens, which must be plain identifiers, joined by
    single spaces. The empty pattern is the empty key."""
    tokens = pattern.split()
    if any(not re.fullmatch(_IDENTIFIER_PATTERN, token) for token in tokens):
        raise GenerationError(f"unsupported macro pattern `{pattern.strip()}`")
    return " ".join(tokens)


def _macro_literal(text: str, start: int) -> tuple[str, int]:
    """A string literal of a macro body: only the escape `\\n` is allowed in it."""
    end, escaped = _quoted_end(text, start)
    body = text[start + 1 : end - 1]
    if escaped:
        if re.search(r"\\[^n]", body):
            raise GenerationError("macro template strings may escape only `\\n`")
        body = body.replace("\\n", "\n")
    return body, end


def _expand_macro_body(body: str, macros: dict[str, dict[str, str]], depth: int) -> str:
    """The text of a macro arm's body: a string literal, or `concat!` of string literals and
    invocations of already-parsed arms."""
    pos = _skip_trivia(body, 0)
    if pos < len(body) and body[pos] == '"':
        text, end = _macro_literal(body, pos)
        if _skip_trivia(body, end) != len(body):
            raise GenerationError("unsupported macro body")
        return text
    concat = re.compile(r"concat\s*!\s*\(").match(body, pos)
    if not concat:
        raise GenerationError("a macro body must be a string literal or `concat!`")
    close = matching_delimiter(body, concat.end() - 1, "(", ")")
    if _skip_trivia(body, close + 1) != len(body):
        raise GenerationError("unsupported macro body")
    inner = body[concat.end() : close]
    pieces: list[str] = []
    pos = 0
    while True:
        pos = _skip_trivia(inner, pos)
        if pos == len(inner):
            break
        if inner[pos] == '"':
            text, end = _macro_literal(inner, pos)
            pieces.append(text)
        else:
            invocation = _MACRO_INVOCATION.match(inner, pos)
            if not invocation:
                raise GenerationError("unsupported item in a macro's `concat!`")
            end = matching_delimiter(inner, invocation.end() - 1, "(", ")")
            pieces.append(
                _expand_invocation(
                    invocation.group(1), inner[invocation.end() : end], macros, depth + 1
                )
            )
            end += 1
        pos = _skip_trivia(inner, end)
        if pos < len(inner):
            if inner[pos] != ",":
                raise GenerationError("items of a macro's `concat!` must be comma-separated")
            pos += 1
    return "".join(pieces)


def _expand_invocation(
    name: str, pattern: str, macros: dict[str, dict[str, str]], depth: int = 0
) -> str:
    if depth > 8:
        raise GenerationError(f"macro {name}! expands too deeply")
    arms = macros.get(name)
    if arms is None:
        raise GenerationError(f"unknown macro {name}!")
    key = _arm_key(pattern)
    if key not in arms:
        raise GenerationError(f"macro {name}! has no arm for `{key}`")
    return arms[key]


def parse_macros(source: str) -> dict[str, dict[str, tuple[str, ...]]]:
    """The `macro_rules!` definitions of a source file, restricted to the shape the blocks use:
    every arm's pattern is a sequence of plain identifiers, and its body is a string literal or
    a `concat!` of string literals and invocations of arms defined earlier. The result maps a
    macro's name to its arms, each expanded to its instruction lines; the body's text must end
    with a newline, and its lines are the instructions."""
    masked = masked_noncode(source)
    macros: dict[str, dict[str, str]] = {}
    for match in re.finditer(rf"\bmacro_rules\s*!\s*({_IDENTIFIER_PATTERN})\s*\{{", masked):
        name = match.group(1)
        if name in macros:
            raise GenerationError(f"macro {name}! is defined twice")
        close = matching_delimiter(source, match.end() - 1, "{", "}")
        body = source[match.end() : close]
        arms: dict[str, str] = {}
        macros[name] = arms
        pos = _skip_trivia(body, 0)
        while pos < len(body):
            if body[pos] != "(":
                raise GenerationError(f"macro {name}!: unsupported arm syntax")
            pattern_close = matching_delimiter(body, pos, "(", ")")
            key = _arm_key(body[pos + 1 : pattern_close])
            if key in arms:
                raise GenerationError(f"macro {name}!: duplicate arm `{key}`")
            pos = _skip_trivia(body, pattern_close + 1)
            if not body.startswith("=>", pos):
                raise GenerationError(f"macro {name}!: unsupported arm syntax")
            pos = _skip_trivia(body, pos + 2)
            if pos >= len(body) or body[pos] != "{":
                raise GenerationError(f"macro {name}!: unsupported arm syntax")
            body_close = matching_delimiter(body, pos, "{", "}")
            try:
                arms[key] = _expand_macro_body(body[pos + 1 : body_close], macros, 0)
            except GenerationError as error:
                raise GenerationError(f"macro {name}!: {error}") from error
            pos = _skip_trivia(body, body_close + 1)
            if pos < len(body) and body[pos] == ";":
                pos = _skip_trivia(body, pos + 1)
    lines: dict[str, dict[str, tuple[str, ...]]] = {}
    for name, arms in macros.items():
        lines[name] = {}
        for key, text in arms.items():
            if not text.endswith("\n"):
                raise GenerationError(f"macro {name}!: the arm `{key}` must end with a newline")
            instructions = tuple(line.strip() for line in text[:-1].split("\n"))
            if any(not instruction for instruction in instructions):
                raise GenerationError(f"macro {name}!: the arm `{key}` has an empty line")
            lines[name][key] = instructions
    return lines


def _parse_asm(
    inner: str,
    function: str,
    *,
    macros: dict[str, dict[str, tuple[str, ...]]],
    reserved_names: set[str],
    fixed_registers: set[str],
    const_operands: set[str],
    allowed_options: set[str],
    required_options: set[str],
) -> tuple[tuple[str, ...], tuple[MacroOrigin | None, ...], tuple[Declaration, ...], set[str]]:
    """Consume the complete restricted grammar of one `asm!` invocation."""
    instructions: list[str] = []
    origins: list[MacroOrigin | None] = []
    declarations: list[Declaration] = []
    options: set[str] | None = None
    operands_started = False
    sites = 0
    pos = 0
    while True:
        pos = _skip_trivia(inner, pos)
        if pos == len(inner):
            break
        if _raw_string_end(inner, pos) is not None:
            raise GenerationError(f"{function}: raw asm template strings are unsupported")
        if inner[pos] == '"':
            if operands_started:
                raise GenerationError(f"{function}: instruction template after asm operands")
            end, escaped = _quoted_end(inner, pos)
            if escaped:
                raise GenerationError(f"{function}: escaped asm template strings are unsupported")
            instruction = inner[pos + 1 : end - 1]
            if "\n" in instruction or "\r" in instruction:
                raise GenerationError(f"{function}: multiline asm template strings are unsupported")
            pos = _skip_trivia(inner, end)
            if pos >= len(inner) or inner[pos] != ",":
                raise GenerationError(f"{function}: asm template must be followed by a comma")
            instructions.append(instruction.strip())
            origins.append(None)
            pos += 1
            continue
        invocation = _MACRO_INVOCATION.match(inner, pos)
        if invocation and not operands_started:
            name = invocation.group(1)
            end = matching_delimiter(inner, invocation.end() - 1, "(", ")")
            arms = macros.get(name)
            if arms is None:
                raise GenerationError(f"{function}: unknown macro {name}! in template position")
            key = _arm_key(inner[invocation.end() : end])
            if key not in arms:
                raise GenerationError(f"{function}: macro {name}! has no arm for `{key}`")
            pos = _skip_trivia(inner, end + 1)
            if pos >= len(inner) or inner[pos] != ",":
                raise GenerationError(f"{function}: asm template must be followed by a comma")
            origin = MacroOrigin(name, key, sites)
            sites += 1
            instructions.extend(arms[key])
            origins.extend([origin] * len(arms[key]))
            pos += 1
            continue

        operands_started = True
        end = _argument_end(inner, pos)
        argument = _strip_comments(inner[pos:end]).strip()
        if not argument:
            raise GenerationError(f"{function}: empty or unsupported asm argument")
        option_match = re.fullmatch(r"options\(([^()]*)\)", argument)
        if option_match:
            if options is not None:
                raise GenerationError(f"{function}: duplicate options")
            options = {part.strip() for part in option_match.group(1).split(",") if part.strip()}
        else:
            if options is not None:
                raise GenerationError(f"{function}: operand after options: {argument}")
            declarations.append(
                parse_declaration(
                    argument,
                    reserved_names=reserved_names,
                    fixed_registers=fixed_registers,
                    const_operands=const_operands,
                )
            )
        pos = end if end == len(inner) else end + 1

    if not instructions:
        raise GenerationError(f"{function}: asm! block has no instructions")
    if options is None:
        raise GenerationError(f"{function}: missing options")
    unsupported_options = options - allowed_options
    if unsupported_options:
        raise GenerationError(f"{function}: unsupported options {sorted(unsupported_options)}")
    if not required_options.issubset(options):
        raise GenerationError(f"{function}: requires options {sorted(required_options)}")
    return tuple(instructions), tuple(origins), tuple(declarations), options


_IDENTIFIER = r"[A-Za-z_]\w*"
_INDEX_BINDING = re.compile(
    rf"let\s+mut\s+({_IDENTIFIER})\s*=\s*({_IDENTIFIER})\s*\[\s*([0-9]+)\s*\]\s*;"
)
_DESTRUCTURING_BINDING = re.compile(
    rf"let\s*\[\s*([^\]]+)\s*\]\s*=\s*(?:\*\s*)?({_IDENTIFIER})\s*;"
)
_OUTPUT_DECLARATION = re.compile(r"let\s*\(\s*([^()]*)\s*\)\s*:\s*\(\s*([^()]*)\s*\)\s*;")
_DEBUG_ASSERT = re.compile(r"debug_assert\s*!\s*\(")
_UNSAFE_BLOCK = re.compile(r"unsafe\s*\{")
_DIRECT_BINDING = re.compile(rf"let\s+(?:mut\s+)?({_IDENTIFIER})\b")
_RETURN_ARRAY = re.compile(rf"\[\s*({_IDENTIFIER}(?:\s*,\s*{_IDENTIFIER})*)\s*\]")


def _parse_function_prefix(
    body: str,
    masked_body: str,
    asm_start: int,
    function: str,
    expected: set[str],
) -> dict[str, tuple[str, int]]:
    """Consume the supported statements before an inline-assembly block."""
    locals_map: dict[str, tuple[str, int]] = {}
    bound_names: set[str] = set()

    def bind(name: str, binding: tuple[str, int] | None = None) -> None:
        if name in expected:
            raise GenerationError(f"{function}: local {name} shadows a function argument")
        if name in bound_names:
            raise GenerationError(f"{function}: duplicate local {name}")
        bound_names.add(name)
        if binding is not None:
            locals_map[name] = binding

    pos = _skip_trivia(body, 0)
    while pos < asm_start:
        debug_assert = _DEBUG_ASSERT.match(masked_body, pos, asm_start)
        if debug_assert:
            opening = debug_assert.end() - 1
            closing = matching_delimiter(body, opening, "(", ")")
            if closing >= asm_start:
                raise GenerationError(f"{function}: debug_assert! overlaps asm!")
            pos = _skip_trivia(body, closing + 1)
            if pos >= asm_start or body[pos] != ";":
                raise GenerationError(f"{function}: debug_assert! must end with `;`")
            pos = _skip_trivia(body, pos + 1)
            continue

        local_match = _INDEX_BINDING.match(masked_body, pos, asm_start)
        if local_match:
            local, argument, index = local_match.groups()
            if argument not in expected:
                raise GenerationError(f"{function}: local reads non-argument {argument}")
            bind(local, (argument, int(index)))
            pos = _skip_trivia(body, local_match.end())
            continue

        destructuring = _DESTRUCTURING_BINDING.match(masked_body, pos, asm_start)
        if destructuring:
            names, argument = destructuring.groups()
            if argument not in expected:
                raise GenerationError(f"{function}: destructures non-argument {argument}")
            for index, item in enumerate(names.split(",")):
                local_match = re.fullmatch(rf"\s*(?:mut\s+)?({_IDENTIFIER})\s*", item)
                if not local_match:
                    raise GenerationError(f"{function}: unsupported local binding {item}")
                bind(local_match.group(1), (argument, index))
            pos = _skip_trivia(body, destructuring.end())
            continue

        output_declaration = _OUTPUT_DECLARATION.match(masked_body, pos, asm_start)
        if output_declaration:
            names = tuple(part.strip() for part in output_declaration.group(1).split(","))
            types = tuple(part.strip() for part in output_declaration.group(2).split(","))
            if (
                not names
                or len(names) != len(types)
                or any(not re.fullmatch(_IDENTIFIER, name) for name in names)
                or any(type_name != "u64" for type_name in types)
            ):
                raise GenerationError(f"{function}: unsupported output declaration")
            for name in names:
                bind(name)
            pos = _skip_trivia(body, output_declaration.end())
            continue

        unsafe_block = _UNSAFE_BLOCK.match(masked_body, pos, asm_start)
        if unsafe_block:
            pos = _skip_trivia(body, unsafe_block.end())
            if pos != asm_start:
                raise GenerationError(f"{function}: unsupported code before asm!")
            return locals_map

        direct_binding = _DIRECT_BINDING.match(masked_body, pos, asm_start)
        if direct_binding and direct_binding.group(1) in expected:
            name = direct_binding.group(1)
            raise GenerationError(f"{function}: local {name} shadows a function argument")
        local_use = next(
            (
                name
                for name in locals_map
                if re.match(rf"\b{re.escape(name)}\b", masked_body[pos:asm_start])
            ),
            None,
        )
        if local_use is not None:
            raise GenerationError(
                f"{function}: unsupported use of bound local {local_use} before asm!"
            )
        raise GenerationError(f"{function}: unsupported code before asm!")

    raise GenerationError(f"{function}: asm! must be the sole statement in an unsafe block")


def _parse_function_suffix(
    body: str,
    masked_body: str,
    asm_close: int,
    function: str,
    result_count: int,
) -> tuple[str, ...]:
    """Consume the unsafe-block close and the function's exact result expression."""
    pos = _skip_trivia(body, asm_close + 1)
    if pos >= len(body) or body[pos] != ";":
        raise GenerationError(f"{function}: asm! invocation must end with `;`")
    pos = _skip_trivia(body, pos + 1)
    if pos >= len(body) or body[pos] != "}":
        raise GenerationError(f"{function}: unsupported code after asm!")
    pos = _skip_trivia(body, pos + 1)

    returned = _RETURN_ARRAY.match(masked_body, pos)
    if not returned:
        raise GenerationError(f"{function}: unsupported code after asm!")
    returns = tuple(part.strip() for part in returned.group(1).split(","))
    pos = _skip_trivia(body, returned.end())
    if pos != len(body):
        raise GenerationError(f"{function}: unsupported code after asm!")
    if len(returns) != result_count:
        raise GenerationError(
            f"{function}: source output tuple has {len(returns)} values, expected {result_count}"
        )
    return returns


def parse_function(
    source: str,
    function: str,
    expected_args: Sequence[str],
    result_count: int,
    *,
    reserved_names: set[str] = frozenset(),
    fixed_registers: set[str] = frozenset(),
    const_operands: set[str] = frozenset(),
    allowed_options: set[str] = frozenset(("pure", "readonly", "nomem", "nostack")),
    required_options: set[str] = frozenset(("pure", "nostack")),
) -> ParsedFunction:
    """Extract one Rust function's sole inline-asm block and surrounding bindings."""
    masked_source = masked_noncode(source)
    pattern = re.compile(rf"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?fn\s+{re.escape(function)}\s*\(")
    functions = list(pattern.finditer(masked_source))
    if len(functions) != 1:
        raise GenerationError(f"expected exactly one `fn {function}(`, found {len(functions)}")
    found = functions[0]
    signature_open = found.end() - 1
    signature_close = matching_delimiter(source, signature_open, "(", ")")
    parameters = set(
        re.findall(
            r"\b([A-Za-z_]\w*)\s*:",
            masked_source[signature_open + 1 : signature_close],
        )
    )
    expected = set(expected_args)
    if parameters != expected:
        raise GenerationError(
            f"{function}: parameters {sorted(parameters)} do not match {sorted(expected)}"
        )

    body_open = masked_noncode(source[signature_close + 1 :]).find("{")
    if body_open < 0:
        raise GenerationError(f"{function}: missing function body")
    body_open += signature_close + 1
    body_close = matching_delimiter(source, body_open, "{", "}")
    body = source[body_open + 1 : body_close]
    masked_body = masked_noncode(body)
    macros = list(re.finditer(r"\basm\s*!\s*\(", masked_body))
    if len(macros) != 1:
        raise GenerationError(f"{function}: expected exactly one asm! block, found {len(macros)}")
    asm = macros[0]
    asm_open = masked_body.find("(", asm.start())
    asm_close = matching_delimiter(body, asm_open, "(", ")")
    instructions, origins, declarations, options = _parse_asm(
        body[asm_open + 1 : asm_close],
        function,
        macros=parse_macros(source),
        reserved_names=reserved_names,
        fixed_registers=fixed_registers,
        const_operands=const_operands,
        allowed_options=allowed_options,
        required_options=required_options,
    )

    locals_map = _parse_function_prefix(body, masked_body, asm.start(), function, expected)
    returns = _parse_function_suffix(body, masked_body, asm_close, function, result_count)
    return ParsedFunction(instructions, declarations, locals_map, returns, options, origins)


def declaration_directions(parsed: ParsedFunction, function: str) -> dict[str, str]:
    """Validate unique operand names and return their read/write directions."""
    names = [declaration.name for declaration in parsed.declarations]
    if len(names) != len(set(names)):
        raise GenerationError(f"{function}: duplicate operand declaration")
    return {declaration.name: declaration.kind for declaration in parsed.declarations}


def output_bindings(parsed: ParsedFunction, function: str) -> dict[str, str]:
    """Map Rust asm output variables to operand names, rejecting ambiguous bindings."""
    outputs: dict[str, str] = {}
    for declaration in parsed.declarations:
        output = declaration.output
        if not output or output == "_":
            continue
        if not re.fullmatch(r"[A-Za-z_]\w*", output):
            raise GenerationError(f"{function}: unsupported output target {output}")
        if output in outputs:
            raise GenerationError(f"{function}: duplicate output target {output}")
        outputs[output] = declaration.name
    return outputs


def returned_registers(parsed: ParsedFunction, function: str) -> tuple[str, ...]:
    """Resolve the Rust result tuple to asm operands and require a complete binding."""
    outputs = output_bindings(parsed, function)
    registers: list[str] = []
    for output in parsed.returns:
        if output not in outputs:
            raise GenerationError(f"{function}: returned value {output} is not an asm output")
        registers.append(outputs[output])
    unused = set(outputs) - set(parsed.returns)
    if unused:
        raise GenerationError(f"{function}: named asm outputs not returned: {sorted(unused)}")
    return tuple(registers)


def check_output(path: Path, expected: str, root: Path | None = None) -> bool:
    """Compare one generated file without writing it, printing a unified diff."""
    display = path.relative_to(root) if root is not None else path
    if not path.exists():
        print(f"{display} does not exist", file=sys.stderr)
        return False
    actual = path.read_text()
    if actual == expected:
        return True
    diff = difflib.unified_diff(
        actual.splitlines(True),
        expected.splitlines(True),
        fromfile=str(display),
        tofile="generated",
    )
    sys.stderr.writelines(diff)
    return False
