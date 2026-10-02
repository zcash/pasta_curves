"""Back end: a block's instruction stream as data, for the leakage model.

The transcription (`lean.py`) states what a block computes; this back end states what it is, as
syntax: one term of the instruction set's `Instr` per instruction, in the block's order, with the
instruction as a trailing comment. The constant-time proofs classify these terms, so they must be
the block's own instructions, which is why they are generated from the same front end as the
transcription and checked with it.
"""

from .lean import docstring


def definition(name, doc, terms, column):
    """`def name : List Instr := [...]`, one instruction per line, the comments aligned at
    `column`; a term that reaches the column gets its comment two spaces after it instead."""
    lines = [docstring(doc), f"def {name} : List Instr := ["]
    for k, (term, text) in enumerate(terms):
        code = f"  {term}{',' if k + 1 < len(terms) else ''}"
        if len(code) + 2 <= column:
            lines.append(f"{code.ljust(column)}-- {text}")
        else:
            lines.append(f"{code}  -- {text}")
    lines.append("]")
    return "\n".join(lines) + "\n"
