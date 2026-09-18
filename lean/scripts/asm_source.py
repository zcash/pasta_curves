# Copyright (c) 2026 the pasta-asm contributors.
# SPDX-License-Identifier: Apache-2.0
"""Shared fail-closed extraction of Rust inline-assembly source.

This module parses only the Rust and ``asm!`` surface syntax needed by the Pasta
backends. Architecture-specific instruction semantics belong in their emitters.
Unsupported syntax is rejected rather than guessed.
"""

import dataclasses
import difflib
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple


class GenerationError(ValueError):
    """The Rust source uses a construct the mechanical model does not accept."""


@dataclasses.dataclass(frozen=True)
class Declaration:
    name: str
    kind: str
    value: str
    output: Optional[str]
    fixed: bool = False


@dataclasses.dataclass(frozen=True)
class ParsedFunction:
    instructions: Tuple[str, ...]
    declarations: Tuple[Declaration, ...]
    locals: Dict[str, Tuple[str, int]]
    returns: Tuple[str, ...]
    options: Set[str]


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


def _quoted_end(text: str, start: int, quote: str = '"') -> Tuple[int, bool]:
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


def _raw_string_end(text: str, start: int) -> Optional[int]:
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


def _char_literal_end(text: str, start: int) -> Optional[int]:
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
        end: Optional[int] = None
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
    pieces: List[str] = []
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
    stack: List[str] = []
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
    reserved_names: Set[str] = frozenset(),
    fixed_registers: Set[str] = frozenset(),
    const_operands: Set[str] = frozenset(),
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

    named = re.fullmatch(
        r"([A-Za-z_]\w*)\s*=\s*(in|out|inout)\(([^)]+)\)\s+(.+)", declaration
    )
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
            raise GenerationError(
                f"inout without `=>` must bind a local variable: {text.strip()}"
            )
        output = value
    return Declaration(name, kind, value, output)


def _parse_asm(
    inner: str,
    function: str,
    *,
    reserved_names: Set[str],
    fixed_registers: Set[str],
    const_operands: Set[str],
    allowed_options: Set[str],
    required_options: Set[str],
) -> Tuple[Tuple[str, ...], Tuple[Declaration, ...], Set[str]]:
    """Consume the complete restricted grammar of one `asm!` invocation."""
    instructions: List[str] = []
    declarations: List[Declaration] = []
    options: Optional[Set[str]] = None
    operands_started = False
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
            instruction = inner[pos + 1:end - 1]
            if "\n" in instruction or "\r" in instruction:
                raise GenerationError(f"{function}: multiline asm template strings are unsupported")
            pos = _skip_trivia(inner, end)
            if pos >= len(inner) or inner[pos] != ",":
                raise GenerationError(f"{function}: asm template must be followed by a comma")
            instructions.append(instruction.strip())
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
            options = {
                part.strip() for part in option_match.group(1).split(",") if part.strip()
            }
        else:
            if options is not None:
                raise GenerationError(f"{function}: operand after options: {argument}")
            declarations.append(parse_declaration(
                argument,
                reserved_names=reserved_names,
                fixed_registers=fixed_registers,
                const_operands=const_operands,
            ))
        pos = end if end == len(inner) else end + 1

    if not instructions:
        raise GenerationError(f"{function}: asm! block has no instructions")
    if options is None:
        raise GenerationError(f"{function}: missing options")
    unsupported_options = options - allowed_options
    if unsupported_options:
        raise GenerationError(f"{function}: unsupported options {sorted(unsupported_options)}")
    if not required_options.issubset(options):
        raise GenerationError(
            f"{function}: requires options {sorted(required_options)}"
        )
    return tuple(instructions), tuple(declarations), options


def parse_function(
    source: str,
    function: str,
    expected_args: Sequence[str],
    result_count: int,
    *,
    reserved_names: Set[str] = frozenset(),
    fixed_registers: Set[str] = frozenset(),
    const_operands: Set[str] = frozenset(),
    allowed_options: Set[str] = frozenset(("pure", "readonly", "nomem", "nostack")),
    required_options: Set[str] = frozenset(("pure", "nostack")),
) -> ParsedFunction:
    """Extract one Rust function's sole inline-asm block and surrounding bindings."""
    masked_source = masked_noncode(source)
    pattern = re.compile(
        rf"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?fn\s+{re.escape(function)}\s*\("
    )
    functions = list(pattern.finditer(masked_source))
    if len(functions) != 1:
        raise GenerationError(
            f"expected exactly one `fn {function}(`, found {len(functions)}"
        )
    found = functions[0]
    signature_open = found.end() - 1
    signature_close = matching_delimiter(source, signature_open, "(", ")")
    parameters = set(re.findall(
        r"\b([A-Za-z_]\w*)\s*:",
        masked_source[signature_open + 1:signature_close],
    ))
    expected = set(expected_args)
    if parameters != expected:
        raise GenerationError(
            f"{function}: parameters {sorted(parameters)} do not match {sorted(expected)}"
        )

    body_open = masked_noncode(source[signature_close + 1:]).find("{")
    if body_open < 0:
        raise GenerationError(f"{function}: missing function body")
    body_open += signature_close + 1
    body_close = matching_delimiter(source, body_open, "{", "}")
    body = source[body_open + 1:body_close]
    masked_body = masked_noncode(body)
    macros = list(re.finditer(r"\basm\s*!\s*\(", masked_body))
    if len(macros) != 1:
        raise GenerationError(
            f"{function}: expected exactly one asm! block, found {len(macros)}"
        )
    asm = macros[0]
    asm_open = masked_body.find("(", asm.start())
    asm_close = matching_delimiter(body, asm_open, "(", ")")
    instructions, declarations, options = _parse_asm(
        body[asm_open + 1:asm_close], function,
        reserved_names=reserved_names,
        fixed_registers=fixed_registers,
        const_operands=const_operands,
        allowed_options=allowed_options,
        required_options=required_options,
    )

    locals_map: Dict[str, Tuple[str, int]] = {}
    prefix = masked_body[:asm.start()]
    local_patterns = (
        r"let\s+mut\s+([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\[([0-9]+)\]\s*;",
        r"let\s+\[([^\]]+)\]\s*=\s*(?:\*)?([A-Za-z_]\w*)\s*;",
    )
    for local_match in re.finditer(local_patterns[0], prefix):
        local, argument, index = local_match.groups()
        if argument not in expected:
            raise GenerationError(f"{function}: local reads non-argument {argument}")
        locals_map[local] = (argument, int(index))
    for local_match in re.finditer(local_patterns[1], prefix):
        names, argument = local_match.groups()
        if argument not in expected:
            raise GenerationError(f"{function}: destructures non-argument {argument}")
        for index, item in enumerate(names.split(",")):
            local = re.sub(r"^\s*mut\s+", "", item).strip()
            if not re.fullmatch(r"[A-Za-z_]\w*", local):
                raise GenerationError(f"{function}: unsupported local binding {item}")
            if local in locals_map:
                raise GenerationError(f"{function}: duplicate local {local}")
            locals_map[local] = (argument, index)

    prefix_without_bindings = prefix
    for local_pattern in local_patterns:
        prefix_without_bindings = re.sub(local_pattern, " ", prefix_without_bindings)
    for local in locals_map:
        if re.search(rf"\b{re.escape(local)}\b", prefix_without_bindings):
            raise GenerationError(
                f"{function}: unsupported use of bound local {local} before asm!"
            )

    suffix = masked_body[asm_close + 1:]
    return_matches = re.findall(
        r"(?m)^\s*\[\s*([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*\]\s*$",
        suffix,
    )
    if len(return_matches) != 1:
        raise GenerationError(
            f"{function}: expected one source output tuple, found {len(return_matches)}"
        )
    returns = tuple(part.strip() for part in return_matches[0].split(","))
    if len(returns) != result_count:
        raise GenerationError(
            f"{function}: source output tuple has {len(returns)} values, expected {result_count}"
        )
    return ParsedFunction(instructions, declarations, locals_map, returns, options)


def declaration_directions(parsed: ParsedFunction, function: str) -> Dict[str, str]:
    """Validate unique operand names and return their read/write directions."""
    names = [declaration.name for declaration in parsed.declarations]
    if len(names) != len(set(names)):
        raise GenerationError(f"{function}: duplicate operand declaration")
    return {declaration.name: declaration.kind for declaration in parsed.declarations}


def output_bindings(parsed: ParsedFunction, function: str) -> Dict[str, str]:
    """Map Rust asm output variables to operand names, rejecting ambiguous bindings."""
    outputs: Dict[str, str] = {}
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


def returned_registers(parsed: ParsedFunction, function: str) -> Tuple[str, ...]:
    """Resolve the Rust result tuple to asm operands and require a complete binding."""
    outputs = output_bindings(parsed, function)
    registers: List[str] = []
    for output in parsed.returns:
        if output not in outputs:
            raise GenerationError(f"{function}: returned value {output} is not an asm output")
        registers.append(outputs[output])
    unused = set(outputs) - set(parsed.returns)
    if unused:
        raise GenerationError(f"{function}: named asm outputs not returned: {sorted(unused)}")
    return tuple(registers)


def check_output(path: Path, expected: str, root: Optional[Path] = None) -> bool:
    """Compare one generated file without writing it, printing a unified diff."""
    display = path.relative_to(root) if root is not None else path
    if not path.exists():
        print(f"{display} does not exist", file=sys.stderr)
        return False
    actual = path.read_text()
    if actual == expected:
        return True
    diff = difflib.unified_diff(
        actual.splitlines(True), expected.splitlines(True),
        fromfile=str(display), tofile="generated",
    )
    sys.stderr.writelines(diff)
    return False
