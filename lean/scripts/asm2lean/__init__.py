"""A compiler from inline assembly in Rust to Lean definitions and their proof skeletons.

It is organised as compilers usually are, as a pipeline over one intermediate representation:

    rust.py      front end: one Rust function's `asm!` block, fail-closed: its template lines
                 (with `macro_rules!` invocations expanded, and their origins kept), its operand
                 declarations, the bindings before it, and the array it returns
    block.py     the driver, for any instruction set: binds a block's operands to its
                 arguments, runs the lifter, collects the results, rerolls, validates
    isa/         the instruction sets. `isa/__init__.py` is what an instruction set provides:
                 its syntax, and a lifter (`Lifter`, whose base keeps the register state,
                 operand directions, and pointer operands). Each other file is one instruction
                 set, lifting its instructions to the IR with its own checks (the form or
                 validity of the flags, its operand syntax): `isa/aarch64.py`, `isa/x86_64.py`
    ir.py        the IR: a straight-line program in A-normal form, one node per operation of the
                 Lean semantics, and liveness on its bindings
    reroll.py    a pass folding unrolled loops back into a round called per repetition, after
                 checking that the repetitions are alpha-equivalent
    validate.py  translation validation of that pass: the rerolled block, its calls expanded,
                 returns the same terms as the flat block, by value numbering
    lean.py      back end: the Lean text of each program
    skeleton.py  back end: the mechanical part of each program's correctness proof, as symbolic
                 execution; each node states its own proof step
    specs.py     the check that the hand-written proofs contain the generated skeletons

A node is the unit of everything after lifting: the transcription prints its `let`s, liveness reads
them, rerolling renames and compares it, the validator evaluates its terms, and the skeleton asks
it for its step. So another instruction set is an `Isa`, a `Lifter` subclass, and node classes for
the operations `ir.py` does not already have, in one new file of `isa/`; nothing else changes (see
`isa/__init__.py`, and the toy instruction set in the tests).

The crate's configuration (which blocks, how their rounds fold, how the proofs name things, which
proof file proves which routine) is in `../pasta/`, and `../gen.py` is the command line.
"""
