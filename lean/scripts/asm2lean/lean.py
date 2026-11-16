"""Back end: the Lean text of programs.

A program prints as a definition whose body is its kept `let` bindings, each with its source
instruction as a trailing comment aligned at a column. Under dead-code elimination, a binding that
nothing reads is not printed: an argument (or an optional call output) the block never uses is
left as a comment, a flag write is dropped, and any other dead value is an error, since it would
be dead code in the block.
"""

from __future__ import annotations

import re
import textwrap
from collections.abc import Sequence

from .ir import Dead, Program, show
from .rust import GenerationError

WIDTH = 100


_STRUCTURE = re.compile(r"^structure (\w+) where\n((?:[ \t]+.*\n?)*)", re.MULTILINE)
_FIELD = re.compile(r"^[ \t]+(\w+) : ")
_BOUNDED = re.compile(r"^def Bounded \(s : (\w+)\) : Prop :=\n((?:[ \t]+.*\n?)*)", re.MULTILINE)


def structures(text: str) -> dict[str, tuple[str, ...]]:
    """The fields, in declaration order, of each `structure` declared in Lean text."""
    found: dict[str, tuple[str, ...]] = {}
    for declaration in _STRUCTURE.finditer(text):
        fields = (_FIELD.match(line) for line in declaration.group(2).splitlines())
        found[declaration.group(1)] = tuple(f.group(1) for f in fields if f)
    return found


def bounded_fields(text: str) -> dict[str, tuple[str, ...]]:
    """For each structure whose `Bounded` predicate the text declares, the fields that
    conjunction bounds, in its order."""
    return {
        d.group(1): tuple(re.findall(r"\bs\.(\w+) <", d.group(2))) for d in _BOUNDED.finditer(text)
    }


def check_structures(programs: Sequence[Program]) -> None:
    """Require every field order a program assumes of a structure the programs declare to be the
    declared one. A round's state is projected by its fields (`state_fields`), so those are the
    structure's; a structured argument's fields (`arg_fields`) are those of the structure's
    `Bounded` conjunction, which the skeleton projects a field's bound from, or, for a structure
    without one, the structure's, which the validator binds positionally. The declarations are
    text, and a reordered field would otherwise bind a value or a bound to the wrong name, Lean
    accepting it whenever the fields' types agree."""
    declared: dict[str, tuple[str, ...]] = {}
    bounded: dict[str, tuple[str, ...]] = {}
    for program in programs:
        text = program.struct or ""
        for table, found in ((declared, structures(text)), (bounded, bounded_fields(text))):
            for name, fields in found.items():
                if table.setdefault(name, fields) != fields:
                    raise GenerationError(f"structure {name} is declared twice, differently")
    for program in programs:
        signature = program.signature
        assumed = [(signature.result, program.state_fields, declared, "its result")]
        for name, kind in signature.params:
            table = bounded if kind in bounded else declared
            assumed.append((kind, program.arg_fields.get(name, ()), table, name))
        for kind, fields, table, what in assumed:
            if kind in table and fields and fields != table[kind]:
                raise GenerationError(
                    f"{program.name}: {what} takes {kind}'s fields as {list(fields)}, declared "
                    f"{list(table[kind])}"
                )


def doc_text(text: str) -> str:
    """Text for inside a `/-- ... -/` comment. Lean's block comments nest, so a `-/` in it would
    end the comment early and a `/-` would open one that swallows the code after it: either way,
    the rest of the text would change what Lean reads, so both are rejected."""
    if "-/" in text or "/-" in text:
        raise GenerationError(f"a docstring may not contain `-/` or `/-`: {text!r}")
    return text


def comment_text(text: str) -> str:
    """Text for after a `--` comment, which ends at the line's end: a line break in it would make
    the rest of the text Lean code, so it is rejected."""
    if "\n" in text or "\r" in text:
        raise GenerationError(f"a line comment may not contain a line break: {text!r}")
    return text


def docstring(text: str, width: int = WIDTH) -> str:
    """A `/-- ... -/` docstring wrapped to the repository's line width."""
    doc_text(text)
    lines = textwrap.TextWrapper(
        width=width, break_long_words=False, break_on_hyphens=False, initial_indent="/-- "
    ).wrap(text)
    if len(lines[-1]) + 3 <= width:
        lines[-1] += " -/"
    else:
        lines.append("-/")
    return "\n".join(lines)


def wrap_words(head: str, words: Sequence[str], tail: str, indent: str = "  ") -> list[str]:
    """`head w1 w2 ... tail`, broken over lines at WIDTH with a 4-space continuation; the first
    word stays on the head's line however long it is."""
    lines, cur = [], indent + head
    for w in words:
        if cur != indent + head and len(cur) + 1 + len(w) > WIDTH:
            lines.append(cur)
            cur = indent + "    " + w
        else:
            cur += " " + w
    lines.append(cur + tail)
    return lines


def joined(words: list[str], separator: str) -> list[str]:
    """`words`, each but the last followed by `separator`."""
    return [w + separator for w in words[:-1]] + words[-1:]


def body_lines(program: Program) -> list[tuple[str | None, str | None]]:
    """The body as (code, comment) pairs: code is `None` for a whole-line comment, the comment
    `None` for a binding without one."""
    lines: list[tuple[str | None, str | None]] = []
    for _, let, live in program.kept():
        if live:
            trailing = let.trailing
            lines.append((f"  let {let.name} := {let.expr}", trailing and comment_text(trailing)))
        elif let.dead is Dead.COMMENT:
            text = comment_text(f"{let.comment}: {let.name} = {let.expr} is never read")
            lines.append((None, f"  -- {text}"))
        elif let.dead is Dead.ERROR:
            raise ValueError(f"dead computation: {let.name} := {let.expr} ({let.comment})")
    return lines


def comment_column(programs: Sequence[Program], widest: int = 40) -> int:
    """Two spaces past the widest commented binding of `programs` up to `widest` characters;
    longer bindings (the round calls) keep their comment two spaces away instead."""
    return 2 + max(
        len(code)
        for program in programs
        for code, comment in body_lines(program)
        if code is not None and comment is not None and len(code) <= widest
    )


def definition(program: Program, column: int) -> str:
    """The definition, with the instruction comments aligned at `column`; code that reaches the
    column gets its comment two spaces after it instead."""
    body: list[str] = []
    for code, comment in body_lines(program):
        if code is None:
            assert comment is not None  # a line without code is a comment
            body.append(comment)
        elif comment is None:
            body.append(code)
        elif len(code) + 2 <= column:
            body.append(f"{code.ljust(column)}-- {comment}")
        else:
            body.append(f"{code}  -- {comment}")
    head = f"{program.struct}\n" if program.struct else ""
    if program.result_term is not None:
        result = f"  {show(program.result_term)}"
    elif " × " in program.signature.result:
        # A product is a Lean tuple, `(a, b)`; any other result a structure, `⟨a, b⟩`.
        result = f"  ({', '.join(program.results)})"
    else:
        result = f"  ⟨{', '.join(program.results)}⟩"
    return (
        head
        + f"{docstring(program.doc)}\n{program.signature.text()}\n"
        + "\n".join(body)
        + f"\n{result}\n"
    )


def state_structure(
    name: str,
    doc: str,
    fields: Sequence[tuple[str, str | None, str]],
    *,
    words_doc: str,
    value: Sequence[str] | None = None,
    value_doc: str | None = None,
) -> str:
    """The declaration of a round's carried registers: the structure, `Bounded` over its words,
    and, when `value` names the accumulator's limbs, `toNat`. A field is (name, doc, Lean type);
    the `Nat` ones are the words."""
    lines = [doc, f"structure {name} where"]
    for field, field_doc, kind in fields:
        lines += [f"  /-- {doc_text(str(field_doc))} -/", f"  {field} : {kind}"]
    words = [field for field, _, kind in fields if kind == "Nat"]
    lines += [
        "  deriving DecidableEq, Repr",
        "",
        f"namespace {name}",
        "",
        f"/-- Every {doc_text(words_doc)} is below `2^64`. -/",
        f"def Bounded (s : {name}) : Prop :=",
    ]
    lines += wrap_words("", joined([f"s.{f} < 2^64" for f in words], " ∧"), "")
    if value is not None:
        lines += [
            "",
            (
                f"/-- {doc_text(value_doc)} -/"
                if value_doc is not None
                else f"/-- The accumulator's value: limbs `{'`, `'.join(value)}` with weights "
                f"`2^0` to `2^{64 * (len(value) - 1)}`. -/"
            ),
            f"def toNat (s : {name}) : Nat :=",
        ]
        terms = [f"{f'2^{64 * i} * ' if i else ''}s.{f}" for i, f in enumerate(value)]
        lines += wrap_words("", joined(terms, " +"), "")
    lines += ["", f"end {name}", ""]
    return "\n".join(lines)
