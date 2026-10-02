"""Back end: the Lean text of programs.

A program prints as a definition whose body is its kept `let` bindings, each with its source
instruction as a trailing comment aligned at a column. Under dead-code elimination, a binding that
nothing reads is not printed: an argument (or an optional call output) the block never uses is
left as a comment, a flag write is dropped, and any other dead value is an error, since it would
be dead code in the block.
"""

import textwrap

from .ir import Dead

WIDTH = 100


def docstring(text, width=WIDTH):
    """A `/-- ... -/` docstring wrapped to the repository's line width."""
    lines = textwrap.TextWrapper(
        width=width, break_long_words=False, break_on_hyphens=False, initial_indent="/-- "
    ).wrap(text)
    if len(lines[-1]) + 3 <= width:
        lines[-1] += " -/"
    else:
        lines.append("-/")
    return "\n".join(lines)


def wrap_words(head, words, tail, indent="  "):
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


def joined(words, separator):
    """`words`, each but the last followed by `separator`."""
    return [w + separator for w in words[:-1]] + words[-1:]


def signature(name, args, result):
    """`def name (a b : K) (c : K') : result :=`, grouping consecutive arguments of one kind."""
    groups = []
    for arg, kind in args:
        if groups and groups[-1][1] == kind:
            groups[-1][0].append(arg)
        else:
            groups.append(([arg], kind))
    params = " ".join(f"({' '.join(names)} : {kind})" for names, kind in groups)
    return f"def {name} {params} : {result} :="


def body_lines(program):
    """The body as (code, comment) pairs: code is `None` for a whole-line comment, the comment
    `None` for a binding without one."""
    lines = []
    for _, let, live in program.kept():
        if live:
            lines.append((f"  let {let.name} := {let.expr}", let.trailing))
        elif let.dead is Dead.COMMENT:
            lines.append((None, f"  -- {let.comment}: {let.name} = {let.expr} is never read"))
        elif let.dead is Dead.ERROR:
            raise ValueError(f"dead computation: {let.name} := {let.expr} ({let.comment})")
    return lines


def comment_column(programs, widest=40):
    """Two spaces past the widest commented binding of `programs` up to `widest` characters;
    longer bindings (the round calls) keep their comment two spaces away instead."""
    return 2 + max(
        len(code)
        for program in programs
        for code, comment in body_lines(program)
        if code is not None and comment is not None and len(code) <= widest
    )


def definition(program, column):
    """The definition, with the instruction comments aligned at `column`; code that reaches the
    column gets its comment two spaces after it instead."""
    body = []
    for code, comment in body_lines(program):
        if code is None:
            body.append(comment)
        elif comment is None:
            body.append(code)
        elif len(code) + 2 <= column:
            body.append(f"{code.ljust(column)}-- {comment}")
        else:
            body.append(f"{code}  -- {comment}")
    head = f"{program.struct}\n" if program.struct else ""
    result = f"  ⟨{', '.join(program.results)}⟩"
    return (
        head
        + f"{docstring(program.doc)}\n{program.signature}\n"
        + "\n".join(body)
        + f"\n{result}\n"
    )


def state_structure(name, doc, roles, *, words_doc, value=None):
    """The declaration of a round's carried registers: the structure, `Bounded` over its words,
    and, when `value` names the accumulator's limbs, `toNat`. A role is (register, field, doc);
    the field `fl` holds the flags, every other one a word."""
    lines = [doc, f"structure {name} where"]
    for _, field, field_doc in roles:
        lines += [f"  /-- {field_doc} -/", f"  {field} : {'Flags' if field == 'fl' else 'Nat'}"]
    words = [field for _, field, _ in roles if field != "fl"]
    lines += [
        "  deriving DecidableEq, Repr",
        "",
        f"namespace {name}",
        "",
        f"/-- Every {words_doc} is below `2^64`. -/",
        f"def Bounded (s : {name}) : Prop :=",
    ]
    lines += wrap_words("", joined([f"s.{f} < 2^64" for f in words], " ∧"), "")
    if value is not None:
        lines += [
            "",
            (
                f"/-- The accumulator's value: limbs `{'`, `'.join(value)}` with weights "
                f"`2^0` to `2^{64 * (len(value) - 1)}`. -/"
            ),
            f"def toNat (s : {name}) : Nat :=",
        ]
        terms = [f"{f'2^{64 * i} * ' if i else ''}s.{f}" for i, f in enumerate(value)]
        lines += wrap_words("", joined(terms, " +"), "")
    lines += ["", f"end {name}", ""]
    return "\n".join(lines)
