"""Front end: fail-closed extraction of Rust inline-assembly source.

This module parses only the Rust and ``asm!`` surface syntax that the blocks use: one function
with one ``asm!`` block, its template strings, its operand declarations, the bindings before it,
and what it returns. The instruction set's semantics belong to the lifters (``isa/aarch64.py``,
``isa/armv7em.py``, ``isa/x86_64.py``). Unsupported syntax is rejected rather than guessed.

A template line of an ``asm!`` block may be an invocation of a ``macro_rules!`` macro of the
same source file (a repeated instruction sequence, written once). `parse_macros` reads the
definitions, restricted to arms whose pattern is a sequence of plain identifiers or decimal
``literal`` metavariables and whose body is a string literal or a ``concat!`` of literals,
``stringify!`` uses of those metavariables, and earlier arms. `parse_function` expands each
invocation into its instructions, recording where they came from in `origins`.
"""

import dataclasses
import re
from collections.abc import Collection, Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import Literal


class GenerationError(ValueError):
    """The Rust source uses a construct the mechanical model does not accept."""


_IDENTIFIER = r"[A-Za-z_]\w*"

# The `asm!` options a block may give, and those it must, unless its instruction set says otherwise
# (`isa.Isa`).
ALLOWED_OPTIONS = frozenset({"pure", "readonly", "nomem", "nostack"})
REQUIRED_OPTIONS = frozenset({"pure", "nostack"})


# The direction of an operand: read, written, both, or a named constant.
Direction = Literal["in", "out", "inout", "const"]
DIRECTIONS: dict[str, Direction] = {"in": "in", "out": "out", "inout": "inout"}


@dataclasses.dataclass(frozen=True)
class Declaration:
    name: str
    kind: Direction
    value: str
    output: str | None
    fixed: bool = False
    # A `lateout`: written only after the block has read every input, so that the compiler may
    # give it the register of an input it allocates.
    late: bool = False


@dataclasses.dataclass(frozen=True)
class MacroOrigin:
    """Where an instruction of an `asm!` block came from, when a macro invocation in template
    position expanded to it: the macro's name, the invocation arguments in canonical form, and
    the invocation's index among the block's macro invocations."""

    macro: str
    arm: str
    site: int


@dataclasses.dataclass(frozen=True)
class ParameterizedMacro:
    """An instruction template with typed identifier or decimal-literal metavariables."""

    variables: tuple[tuple[str, str], ...]
    instructions: tuple[str, ...]


# A macro arm while the definitions are read: its expanded text, or, for an arm with
# metavariables, that text as a template.
Arm = str | ParameterizedMacro


@dataclasses.dataclass(frozen=True)
class ParsedFunction:
    instructions: tuple[str, ...]
    declarations: tuple[Declaration, ...]
    locals: dict[str, tuple[str, int | None]]
    returns: tuple[str, ...]
    options: set[str]
    # One entry per instruction: `None` for a literal template string, else its macro origin.
    origins: tuple[MacroOrigin | None, ...] = ()
    # Locals initialized as `[0u64; N]`, by local name and 64-bit element count.
    zero_arrays: dict[str, int] = dataclasses.field(default_factory=dict)
    # Each parameter's Rust type, and the return type, as written without whitespace.
    param_types: dict[str, str] = dataclasses.field(default_factory=dict)
    return_type: str = ""
    # The Rust types of the locals declared for the block's outputs (`let (a, b): (u64, u64);`).
    output_types: frozenset[str] = frozenset()


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


def code_without_trivia(text: str) -> str:
    """Return Rust code with comments and insignificant whitespace removed.

    Backends use this for small, explicitly supported surrounding Rust fragments whose
    expression syntax is outside the common inline-assembly grammar. String and character
    literals retain their contents; callers therefore compare only fragments with no
    whitespace-sensitive literal contents.
    """
    stripped = _strip_comments(text)
    out: list[str] = []
    pos = 0
    while pos < len(stripped):
        raw = _raw_string_end(stripped, pos)
        quoted = _quoted_end(stripped, pos) if stripped[pos] == '"' else None
        char = _char_literal_end(stripped, pos) if stripped[pos] == "'" else None
        end = raw if raw is not None else quoted[0] if quoted is not None else char
        if end is not None:
            out.append(stripped[pos:end])
            pos = end
        elif stripped[pos].isspace():
            pos += 1
        else:
            out.append(stripped[pos])
            pos += 1
    return "".join(out)


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
    reserved_names: AbstractSet[str] = frozenset(),
    fixed_registers: AbstractSet[str] = frozenset(),
    fixed_register_kinds: AbstractSet[str] = frozenset(("out",)),
    const_operands: AbstractSet[str] = frozenset(),
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

    fixed = re.fullmatch(r'(in|out|lateout|inout)\("([^"]+)"\)\s+(.+)', declaration)
    if fixed:
        kind_text, register, body = fixed.groups()
        normalized_kind = DIRECTIONS["out" if kind_text == "lateout" else kind_text]
        pieces = [piece.strip() for piece in body.split("=>")]
        valid = (
            register in fixed_registers
            and normalized_kind in fixed_register_kinds
            and len(pieces) <= 2
            and (len(pieces) != 2 or normalized_kind == "inout")
        )
        if valid:
            value = pieces[0]
            output = pieces[1] if len(pieces) == 2 else None
            if normalized_kind == "out":
                valid = output is None and (value == "_" or bool(re.fullmatch(_IDENTIFIER, value)))
                output = None if value == "_" else value
            elif normalized_kind == "in":
                valid = output is None
            elif normalized_kind == "inout":
                valid = output is not None and bool(re.fullmatch(_IDENTIFIER, output))
        if not valid:
            raise GenerationError(f"unsupported fixed-register binding: {text.strip()}")
        late = kind_text == "lateout"
        return Declaration(register, normalized_kind, value, output, fixed=True, late=late)

    named = re.fullmatch(r"([A-Za-z_]\w*)\s*=\s*(in|out|inout)\(([^)]+)\)\s+(.+)", declaration)
    if not named:
        raise GenerationError(f"unsupported operand binding: {text.strip()}")
    name, kind_text, constraint, body = named.groups()
    kind = DIRECTIONS[kind_text]
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


_MACRO_INVOCATION = re.compile(rf"({_IDENTIFIER})\s*!\s*\(")


def _split_macro_tokens(text: str) -> list[str]:
    """Split the deliberately tiny macro pattern/invocation grammar at commas."""
    stripped = text.strip()
    if not stripped:
        return []
    pieces = stripped.split(",") if "," in stripped else stripped.split()
    tokens = [token.strip() for token in pieces]
    if any(not token for token in tokens):
        raise GenerationError(f"unsupported macro pattern `{stripped}`")
    return tokens


def _macro_pattern(pattern: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Normalize one supported arm pattern and return its typed metavariables."""
    key: list[str] = []
    variables: list[tuple[str, str]] = []
    for token in _split_macro_tokens(pattern):
        identifier = re.fullmatch(_IDENTIFIER, token)
        variable = re.fullmatch(rf"\$({_IDENTIFIER})\s*:\s*(ident|literal)", token)
        if identifier:
            key.append(identifier.group(0))
        elif variable:
            name, kind = variable.groups()
            key.append(f"${kind}")
            variables.append((name, kind))
        else:
            raise GenerationError(f"unsupported macro pattern `{pattern.strip()}`")
    names = [name for name, _ in variables]
    if len(names) != len(set(names)):
        raise GenerationError(f"duplicate macro metavariable in `{pattern.strip()}`")
    return ", ".join(key), tuple(variables)


def _invocation_tokens(pattern: str) -> tuple[tuple[str, str], ...]:
    """Classify the deliberately tiny identifier-or-decimal invocation grammar."""
    arguments: list[tuple[str, str]] = []
    for token in _split_macro_tokens(pattern):
        if re.fullmatch(_IDENTIFIER, token):
            arguments.append(("ident", token))
        elif re.fullmatch(r"0|[1-9][0-9]*", token):
            arguments.append(("literal", token))
        else:
            raise GenerationError(f"unsupported macro invocation `{pattern.strip()}`")
    return tuple(arguments)


def _resolve_macro_invocation(
    arms: Mapping[str, object], pattern: str
) -> tuple[str, tuple[str, ...]]:
    """Select the macro arm an invocation expands, and return its metavariable arguments. As
    `macro_rules!` does, the arm is the FIRST, in definition order, whose pattern matches: an arm
    of literal tokens defined after a metavariable arm that matches too is never used."""
    arguments = _invocation_tokens(pattern)
    for key in arms:
        parts = _split_macro_tokens(key)
        if len(parts) != len(arguments):
            continue
        values: list[str] = []
        for part, (kind, value) in zip(parts, arguments, strict=True):
            if part in ("$ident", "$literal"):
                if part != f"${kind}":
                    break
                values.append(value)
            elif kind != "ident" or part != value:
                break
        else:
            return key, tuple(values)
    display = ", ".join(_split_macro_tokens(pattern))
    raise GenerationError(f"macro has no arm for `{display}`")


def _macro_literal(text: str, start: int) -> tuple[str, int]:
    """A string literal of a macro body: only the escape `\\n` is allowed in it."""
    end, escaped = _quoted_end(text, start)
    body = text[start + 1 : end - 1]
    if escaped:
        if re.search(r"\\[^n]", body):
            raise GenerationError("macro template strings may escape only `\\n`")
        body = body.replace("\\n", "\n")
    return body, end


def _expand_macro_body(
    body: str,
    macros: Mapping[str, Mapping[str, Arm]],
    depth: int,
    variables: dict[str, str] | None = None,
) -> str:
    """Expand a supported macro body, preserving typed substitutions as ``${name}``."""
    if depth > 8:
        raise GenerationError("macro body expands too deeply")
    variables = variables or {}
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
            argument = inner[invocation.end() : end].strip()
            if invocation.group(1) == "stringify":
                variable = re.fullmatch(rf"\$({_IDENTIFIER})", argument)
                if not variable or variable.group(1) not in variables:
                    raise GenerationError("stringify! must name a declared metavariable")
                pieces.append(f"${{{variable.group(1)}}}")
            elif invocation.group(1) == "concat":
                pieces.append(
                    _expand_macro_body(inner[pos : end + 1], macros, depth + 1, variables)
                )
            else:
                pieces.append(_expand_invocation(invocation.group(1), argument, macros, depth + 1))
            end += 1
        pos = _skip_trivia(inner, end)
        if pos < len(inner):
            if inner[pos] != ",":
                raise GenerationError("items of a macro's `concat!` must be comma-separated")
            pos += 1
    return "".join(pieces)


def _expand_invocation(
    name: str, pattern: str, macros: Mapping[str, Mapping[str, Arm]], depth: int = 0
) -> str:
    if depth > 8:
        raise GenerationError(f"macro {name}! expands too deeply")
    arms = macros.get(name)
    if arms is None:
        raise GenerationError(f"unknown macro {name}!")
    try:
        key, arguments = _resolve_macro_invocation(arms, pattern)
    except GenerationError as error:
        raise GenerationError(f"macro {name}! {error}") from error
    template = arms[key]
    if arguments or isinstance(template, ParameterizedMacro):
        raise GenerationError(f"nested parameterized macro invocation is unsupported in {name}!")
    return template


def parse_macros(
    source: str, required: AbstractSet[str] | None = None
) -> dict[str, dict[str, tuple[str, ...] | ParameterizedMacro]]:
    """Parse the supported `macro_rules!` definitions.

    When ``required`` is provided, unrelated definitions are skipped. This lets a backend consume
    one deliberately restricted assembly macro without pretending to understand other Rust macros
    in the same source file.
    """
    masked = masked_noncode(source)
    macros: dict[str, dict[str, str | ParameterizedMacro]] = {}
    for match in re.finditer(rf"\bmacro_rules\s*!\s*({_IDENTIFIER})\s*\{{", masked):
        name = match.group(1)
        close = matching_delimiter(source, match.end() - 1, "{", "}")
        if required is not None and name not in required:
            continue
        if name in macros:
            raise GenerationError(f"macro {name}! is defined twice")
        body = source[match.end() : close]
        arms: dict[str, str | ParameterizedMacro] = {}
        macros[name] = arms
        pos = _skip_trivia(body, 0)
        while pos < len(body):
            if body[pos] != "(":
                raise GenerationError(f"macro {name}!: unsupported arm syntax")
            pattern_close = matching_delimiter(body, pos, "(", ")")
            key, variables = _macro_pattern(body[pos + 1 : pattern_close])
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
                expanded = _expand_macro_body(
                    body[pos + 1 : body_close], macros, 0, dict(variables)
                )
                arms[key] = ParameterizedMacro(variables, (expanded,)) if variables else expanded
            except GenerationError as error:
                raise GenerationError(f"macro {name}!: {error}") from error
            pos = _skip_trivia(body, body_close + 1)
            if pos < len(body) and body[pos] == ";":
                pos = _skip_trivia(body, pos + 1)
    lines: dict[str, dict[str, tuple[str, ...] | ParameterizedMacro]] = {}
    for name, arms in macros.items():
        lines[name] = {}
        for key, arm in arms.items():
            variables = arm.variables if isinstance(arm, ParameterizedMacro) else ()
            text = arm.instructions[0] if isinstance(arm, ParameterizedMacro) else arm
            instruction_text = text.removesuffix("\n")
            instructions = tuple(line.strip() for line in instruction_text.split("\n"))
            if any(not instruction for instruction in instructions):
                raise GenerationError(f"macro {name}!: the arm `{key}` has an empty line")
            lines[name][key] = (
                ParameterizedMacro(variables, instructions) if variables else instructions
            )
    return lines


def parse_asm_invocation(
    source: str,
    inner: str,
    function: str,
    *,
    reserved_names: AbstractSet[str] = frozenset(),
    fixed_registers: AbstractSet[str] = frozenset(),
    fixed_register_kinds: AbstractSet[str] = frozenset(("out",)),
    const_operands: AbstractSet[str] = frozenset(),
    allowed_options: AbstractSet[str] = ALLOWED_OPTIONS,
    required_options: AbstractSet[str] = REQUIRED_OPTIONS,
) -> tuple[tuple[str, ...], tuple[MacroOrigin | None, ...], tuple[Declaration, ...], set[str]]:
    """Parse one complete restricted ``asm!`` invocation body.

    This is the instruction-and-operand half of :func:`parse_function`, for a backend
    that separately validates a function's architecture-specific surrounding Rust.
    """
    referenced_macros = {
        match.group(1)
        for match in _MACRO_INVOCATION.finditer(masked_noncode(inner))
        if match.group(1) != "options"
    }
    return _parse_asm(
        inner,
        function,
        macros=parse_macros(source, referenced_macros),
        reserved_names=reserved_names,
        fixed_registers=fixed_registers,
        fixed_register_kinds=fixed_register_kinds,
        const_operands=const_operands,
        allowed_options=allowed_options,
        required_options=required_options,
    )


def _substitute(template: str, substitutions: Mapping[str, str]) -> str:
    """A parameterized arm's instruction, each `${name}` replaced by its argument."""
    return re.sub(r"\$\{([A-Za-z_]\w*)\}", lambda found: substitutions[found.group(1)], template)


def _parse_asm(
    inner: str,
    function: str,
    *,
    macros: dict[str, dict[str, tuple[str, ...] | ParameterizedMacro]],
    reserved_names: AbstractSet[str],
    fixed_registers: AbstractSet[str],
    fixed_register_kinds: AbstractSet[str],
    const_operands: AbstractSet[str],
    allowed_options: AbstractSet[str],
    required_options: AbstractSet[str],
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
            arguments = inner[invocation.end() : end]
            try:
                key, values = _resolve_macro_invocation(arms, arguments)
            except GenerationError as error:
                raise GenerationError(f"{function}: macro {name}! {error}") from error
            pos = _skip_trivia(inner, end + 1)
            if pos >= len(inner) or inner[pos] != ",":
                raise GenerationError(f"{function}: asm template must be followed by a comma")
            arm = arms[key]
            if isinstance(arm, ParameterizedMacro):
                if len(values) != len(arm.variables):
                    raise GenerationError(f"{function}: argument mismatch for macro {name}!")
                substitutions = dict(zip((variable for variable, _ in arm.variables), values))
                expanded = tuple(_substitute(i, substitutions) for i in arm.instructions)
            else:
                if values:
                    raise GenerationError(f"{function}: unexpected arguments for macro {name}!")
                expanded = arm
            origin_text = ", ".join(_split_macro_tokens(arguments))
            origin = MacroOrigin(name, origin_text, sites)
            sites += 1
            instructions.extend(expanded)
            origins.extend([origin] * len(expanded))
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
                    fixed_register_kinds=fixed_register_kinds,
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
    if not required_options <= options:
        raise GenerationError(f"{function}: requires options {sorted(required_options)}")
    # A transcription gives every operand a register of its own. The compiler may give a `lateout`
    # the register of an input it allocates (`in(reg)`), which the block then overwrites before the
    # input is read: only when every input is in a fixed register can no `lateout` share one.
    late = [d.name for d in declarations if d.late]
    allocated = [d.name for d in declarations if not d.fixed and d.kind in ("in", "inout")]
    if late and allocated:
        raise GenerationError(
            f"{function}: lateout {late} may share a register with the allocated inputs "
            f"{allocated}, which the transcription would model as distinct"
        )
    return tuple(instructions), tuple(origins), tuple(declarations), options


_INDEX_BINDING = re.compile(
    rf"let\s+mut\s+({_IDENTIFIER})\s*=\s*({_IDENTIFIER})\s*\[\s*([0-9]+)\s*\]\s*;"
)
_SCALAR_BINDING = re.compile(rf"let\s+mut\s+({_IDENTIFIER})\s*=\s*({_IDENTIFIER})\s*;")
_DESTRUCTURING_BINDING = re.compile(
    rf"let\s*\[\s*([^\]]+)\s*\]\s*=\s*(?:\*\s*)?({_IDENTIFIER})\s*;"
)
_OUTPUT_DECLARATION = re.compile(r"let\s*\(\s*([^()]*)\s*\)\s*:\s*\(\s*([^()]*)\s*\)\s*;")
_SINGLE_OUTPUT_DECLARATION = re.compile(rf"let\s+({_IDENTIFIER})\s*:\s*(u32|u64)\s*;")
_ZERO_ARRAY_BINDING = re.compile(
    rf"let\s+mut\s+({_IDENTIFIER})\s*=\s*\[\s*0u64\s*;\s*([1-9][0-9]*)\s*\]\s*;"
)
_DEBUG_ASSERT = re.compile(r"debug_assert(?:_eq)?\s*!\s*\(")
_EXPECT_ATTRIBUTE = re.compile(r"#\s*\[")
_UNSAFE_BLOCK = re.compile(r"unsafe\s*\{")
_DIRECT_BINDING = re.compile(rf"let\s+(?:mut\s+)?({_IDENTIFIER})\b")
_RETURN_ARRAY = re.compile(rf"\[\s*({_IDENTIFIER}(?:\s*,\s*{_IDENTIFIER})*)\s*\]")
_RETURN_TUPLE = re.compile(rf"\(\s*({_IDENTIFIER}(?:\s*,\s*{_IDENTIFIER})+)\s*\)")
_JOINED_U32_RETURN_ITEM = re.compile(
    rf"u64\s*::\s*from\s*\(\s*({_IDENTIFIER})\s*\)\s*\|\s*"
    rf"\(\s*u64\s*::\s*from\s*\(\s*({_IDENTIFIER})\s*\)\s*<<\s*32\s*\)"
)


def _parse_function_prefix(
    body: str,
    masked_body: str,
    asm_start: int,
    function: str,
    expected: set[str],
) -> tuple[dict[str, tuple[str, int | None]], dict[str, int], frozenset[str]]:
    """Consume the supported statements before an inline-assembly block."""
    locals_map: dict[str, tuple[str, int | None]] = {}
    output_types: set[str] = set()
    zero_arrays: dict[str, int] = {}
    bound_names: set[str] = set()

    def bind(
        name: str,
        binding: tuple[str, int | None] | None = None,
        *,
        allow_identity_shadow: bool = False,
    ) -> None:
        if name in expected and not (allow_identity_shadow and binding == (name, None)):
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

        attribute = _EXPECT_ATTRIBUTE.match(masked_body, pos, asm_start)
        if attribute:
            opening = masked_body.find("[", attribute.start(), attribute.end())
            closing = matching_delimiter(body, opening, "[", "]")
            if closing >= asm_start:
                raise GenerationError(f"{function}: attribute overlaps asm!")
            text = _strip_comments(body[opening + 1 : closing]).strip()
            if not re.fullmatch(
                r'expect\s*\(\s*unused_assignments\s*,\s*reason\s*=\s*"[^"\\]*"\s*\)',
                text,
            ):
                raise GenerationError(f"{function}: unsupported attribute before asm!")
            pos = _skip_trivia(body, closing + 1)
            continue

        local_match = _INDEX_BINDING.match(masked_body, pos, asm_start)
        if local_match:
            local, argument, index = local_match.groups()
            if argument not in expected:
                raise GenerationError(f"{function}: local reads non-argument {argument}")
            bind(local, (argument, int(index)))
            pos = _skip_trivia(body, local_match.end())
            continue

        scalar_binding = _SCALAR_BINDING.match(masked_body, pos, asm_start)
        if scalar_binding:
            local, argument = scalar_binding.groups()
            if argument not in expected:
                raise GenerationError(f"{function}: local reads non-argument {argument}")
            bind(
                local,
                (argument, None),
                allow_identity_shadow=local == argument,
            )
            pos = _skip_trivia(body, scalar_binding.end())
            continue

        zero_array = _ZERO_ARRAY_BINDING.match(masked_body, pos, asm_start)
        if zero_array:
            name, count = zero_array.groups()
            bind(name)
            zero_arrays[name] = int(count)
            pos = _skip_trivia(body, zero_array.end())
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
            names = tuple(
                part.strip() for part in output_declaration.group(1).split(",") if part.strip()
            )
            types = tuple(
                part.strip() for part in output_declaration.group(2).split(",") if part.strip()
            )
            if (
                not names
                or len(names) != len(types)
                or any(not re.fullmatch(_IDENTIFIER, name) for name in names)
                or len(set(types)) != 1
                or types[0] not in ("u32", "u64")
            ):
                raise GenerationError(f"{function}: unsupported output declaration")
            output_types.add(types[0])
            for name in names:
                bind(name)
            pos = _skip_trivia(body, output_declaration.end())
            continue

        single_output = _SINGLE_OUTPUT_DECLARATION.match(masked_body, pos, asm_start)
        if single_output:
            output_types.add(single_output.group(2))
            bind(single_output.group(1))
            pos = _skip_trivia(body, single_output.end())
            continue

        unsafe_block = _UNSAFE_BLOCK.match(masked_body, pos, asm_start)
        if unsafe_block:
            pos = _skip_trivia(body, unsafe_block.end())
            if pos != asm_start:
                raise GenerationError(f"{function}: unsupported code before asm!")
            return locals_map, zero_arrays, frozenset(output_types)

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
    direct_returns: AbstractSet[str] = frozenset(),
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
        returned = _RETURN_TUPLE.match(masked_body, pos)
    if returned:
        returns = tuple(part.strip() for part in returned.group(1).split(","))
        pos = _skip_trivia(body, returned.end())
    elif result_count == 8 and pos < len(body) and body[pos] == "[":
        close = matching_delimiter(body, pos, "[", "]")
        items: list[str] = []
        item_pos = pos + 1
        while item_pos < close:
            item_pos = _skip_trivia(body, item_pos)
            if item_pos == close:
                break
            item = _JOINED_U32_RETURN_ITEM.match(masked_body, item_pos, close)
            if not item:
                raise GenerationError(f"{function}: unsupported joined-word return")
            items.extend(item.groups())
            item_pos = _skip_trivia(body, item.end())
            if item_pos < close:
                if body[item_pos] != ",":
                    raise GenerationError(f"{function}: unsupported joined-word return")
                item_pos += 1
        if len(items) != result_count:
            raise GenerationError(f"{function}: unsupported joined-word return")
        returns = tuple(items)
        pos = _skip_trivia(body, close + 1)
    else:
        direct = re.match(rf"({_IDENTIFIER})\b", masked_body[pos:])
        if not direct or result_count != 1 or direct.group(1) not in direct_returns:
            raise GenerationError(f"{function}: unsupported code after asm!")
        returns = (direct.group(1),)
        pos = _skip_trivia(body, pos + direct.end())
    if pos != len(body):
        raise GenerationError(f"{function}: unsupported code after asm!")
    if len(returns) != result_count:
        raise GenerationError(
            f"{function}: source output tuple has {len(returns)} values, expected {result_count}"
        )
    return returns


def _parameter_types(text: str, function: str) -> dict[str, str]:
    """Each parameter of a signature (`[mut] name: Type, ...`) and its type, without whitespace."""
    types: dict[str, str] = {}
    masked = masked_noncode(text)
    depth, start, pieces = 0, 0, []
    for index, char in enumerate(masked):
        if char in "([<":
            depth += 1
        elif char in ")]>":
            depth -= 1
        elif char == "," and depth == 0:
            pieces.append(text[start:index])
            start = index + 1
    pieces.append(text[start:])
    for piece in pieces:
        if not piece.strip():
            continue
        found = re.fullmatch(rf"\s*(?:mut\s+)?({_IDENTIFIER})\s*:(.*)", piece, re.DOTALL)
        if found is None:
            raise GenerationError(f"{function}: unsupported parameter `{piece.strip()}`")
        types[found.group(1)] = re.sub(r"\s+", "", found.group(2))
    return types


def parse_function(
    source: str,
    function: str,
    expected_args: Sequence[str],
    result_count: int,
    *,
    reserved_names: AbstractSet[str] = frozenset(),
    fixed_registers: AbstractSet[str] = frozenset(),
    fixed_register_kinds: AbstractSet[str] = frozenset(("out",)),
    const_operands: AbstractSet[str] = frozenset(),
    allowed_options: AbstractSet[str] = ALLOWED_OPTIONS,
    required_options: AbstractSet[str] = REQUIRED_OPTIONS,
) -> ParsedFunction:
    """Extract one Rust function's sole inline-asm block and surrounding bindings.

    The function is a top-level item, at the start of a line: a method of the same name in a
    trait implementation, which forwards to it, is indented and so not matched. A source may also
    define, under the same name, a portable function for other targets: the one with an `asm!`
    block is read."""
    masked_source = masked_noncode(source)
    pattern = re.compile(rf"(?m)^(?:pub(?:\([^)]*\))?\s+)?fn\s+{re.escape(function)}\s*\(")
    functions = list(pattern.finditer(masked_source))
    if len(functions) > 1:
        with_asm = [
            match
            for match in functions
            if re.search(
                r"\basm\s*!\s*\(",
                masked_source[
                    slice(*_function_span(source, masked_source, match.start(), function))
                ],
            )
        ]
        if len(with_asm) != 1:
            raise GenerationError(
                f"expected exactly one top-level `fn {function}(` containing asm!, "
                f"found {len(with_asm)}"
            )
        functions = with_asm
    if len(functions) != 1:
        raise GenerationError(f"expected exactly one `fn {function}(`, found {len(functions)}")
    found = functions[0]
    signature_open = found.end() - 1
    signature_close = matching_delimiter(source, signature_open, "(", ")")
    param_types = _parameter_types(source[signature_open + 1 : signature_close], function)
    parameters = set(param_types)
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
    macros = list(re.finditer(r"(?:(?:core\s*::\s*arch\s*::\s*)?asm)\s*!\s*\(", masked_body))
    if len(macros) != 1:
        raise GenerationError(f"{function}: expected exactly one asm! block, found {len(macros)}")
    asm = macros[0]
    asm_open = masked_body.find("(", asm.start())
    asm_close = matching_delimiter(body, asm_open, "(", ")")
    asm_inner = body[asm_open + 1 : asm_close]
    instructions, origins, declarations, options = parse_asm_invocation(
        source,
        asm_inner,
        function,
        reserved_names=reserved_names,
        fixed_registers=fixed_registers,
        fixed_register_kinds=fixed_register_kinds,
        const_operands=const_operands,
        allowed_options=allowed_options,
        required_options=required_options,
    )

    locals_map, zero_arrays, output_types = _parse_function_prefix(
        body, masked_body, asm.start(), function, expected
    )
    returns = _parse_function_suffix(
        body, masked_body, asm_close, function, result_count, set(zero_arrays)
    )
    return_type = re.sub(r"\s+", "", source[signature_close + 1 : body_open])
    if not return_type.startswith("->"):
        raise GenerationError(f"{function}: the function returns nothing")
    return ParsedFunction(
        instructions,
        declarations,
        locals_map,
        returns,
        options,
        origins,
        zero_arrays,
        param_types,
        return_type.removeprefix("->"),
        output_types,
    )


def declaration_directions(parsed: ParsedFunction, function: str) -> dict[str, Direction]:
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


def returned_registers(
    parsed: ParsedFunction,
    function: str,
    *,
    discarded_outputs: AbstractSet[str] = frozenset(),
) -> tuple[str, ...]:
    """Resolve the Rust result tuple to asm operands and validate named outputs.

    A backend may explicitly allow named local outputs that the function intentionally
    discards after the block. This remains fail-closed: every allowed name must be an asm
    output, and every other named output must occur in the function's result.
    """
    outputs = output_bindings(parsed, function)
    unknown_discards = discarded_outputs - set(outputs)
    if unknown_discards:
        raise GenerationError(
            f"{function}: configured discarded values are not asm outputs: "
            f"{sorted(unknown_discards)}"
        )
    returned = set(parsed.returns)
    discarded_returns = discarded_outputs & returned
    if discarded_returns:
        raise GenerationError(
            f"{function}: returned values are also configured as discarded: "
            f"{sorted(discarded_returns)}"
        )
    registers: list[str] = []
    for output in parsed.returns:
        if output not in outputs:
            raise GenerationError(f"{function}: returned value {output} is not an asm output")
        registers.append(outputs[output])
    unused = set(outputs) - returned - discarded_outputs
    if unused:
        raise GenerationError(f"{function}: named asm outputs not returned: {sorted(unused)}")
    return tuple(registers)


# -- the functions of a source file ------------------------------------------------------------


def _function_span(source: str, masked: str, start: int, name: str) -> tuple[int, int]:
    """The braces of the body of the function whose definition starts at `start`."""
    signature_open = masked.find("(", start)
    signature_close = matching_delimiter(source, signature_open, "(", ")")
    body_open = masked.find("{", signature_close)
    if body_open < 0:
        raise GenerationError(f"{name}: missing function body")
    return body_open, matching_delimiter(source, body_open, "{", "}")


def asm_functions(source: str) -> tuple[str, ...]:
    """Every top-level function with an `asm!` block, in source order."""
    masked = masked_noncode(source)
    blocks = []
    for match in re.finditer(r"(?m)^(?:pub(?:\([^)]*\))?\s+)?fn\s+([A-Za-z_]\w*)\s*\(", masked):
        body_open, body_close = _function_span(source, masked, match.start(), match.group(1))
        if re.search(r"\basm\s*!\s*\(", masked[body_open + 1 : body_close]):
            blocks.append(match.group(1))
    return tuple(blocks)


def check_asm_inventory(source: str, covered: Collection[str], architecture: str) -> None:
    """Require the functions of `source` with an `asm!` block to be exactly those an
    architecture transcribes (`covered`), each once: a block added, removed, or renamed fails
    generation until the architecture decides how to cover it."""
    found = asm_functions(source)
    if (
        len(set(found)) != len(found)
        or set(found) != set(covered)
        or len(set(covered)) != len(covered)
    ):
        raise GenerationError(
            f"{architecture}: asm block inventory differs: transcribed {sorted(covered)}, "
            f"source has {list(found)}"
        )


def _functions(source: str, name: str) -> list[str]:
    """The text of each top-level `fn name(` of the source."""
    masked = masked_noncode(source)
    found = []
    for match in re.finditer(rf"(?m)^fn\s+{re.escape(name)}\s*\(", masked):
        _, body_close = _function_span(source, masked, match.start(), name)
        found.append(source[match.start() : body_close + 1])
    return found


def top_level_function(source: str, name: str) -> str:
    """The unique top-level definition of `name`."""
    found = _functions(source, name)
    if len(found) != 1:
        raise GenerationError(f"expected exactly one top-level `fn {name}(`, found {len(found)}")
    return found[0]


def inline_asm_function(source: str, name: str) -> str:
    """The unique top-level definition of `name` with an `asm!` block, where the source may also
    define a portable function of that name for other targets."""
    found = [
        text
        for text in _functions(source, name)
        if re.search(r"\basm\s*!\s*\(", masked_noncode(text))
    ]
    if len(found) != 1:
        raise GenerationError(
            f"expected exactly one top-level `fn {name}(` containing asm!, found {len(found)}"
        )
    return found[0]


def asm_placeholder(function: str) -> tuple[str, str]:
    """The function's code without trivia, its one `asm!(...)` replaced by `asm!(ASM)`, and the
    invocation's contents."""
    masked = masked_noncode(function)
    matches = list(re.finditer(r"\basm\s*!\s*\(", masked))
    if len(matches) != 1:
        raise GenerationError(f"expected one asm! invocation, found {len(matches)}")
    opening = masked.find("(", matches[0].start())
    closing = matching_delimiter(function, opening, "(", ")")
    normalized = code_without_trivia(
        function[: matches[0].start()] + "asm!(ASM)" + function[closing + 1 :]
    )
    return normalized, function[opening + 1 : closing]


def checked_function(source: str, name: str, expected: str) -> str:
    """Require the Rust of `name`, around its `asm!` block if `expected` has one (`asm!(ASM)`), to
    be `expected` up to trivia, for Rust that a transcription models by hand; the block's
    contents, or `""`."""
    function = top_level_function(source, name)
    if "ASM" in expected:
        normalized, inner = asm_placeholder(function)
    else:
        normalized, inner = code_without_trivia(function), ""
    if normalized != code_without_trivia(expected):
        raise GenerationError(f"{name}: surrounding Rust differs from the checked transcription")
    return inner
