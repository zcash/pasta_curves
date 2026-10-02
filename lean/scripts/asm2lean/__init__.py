"""A compiler from inline assembly in Rust to Lean definitions and their proof skeletons.

It is organised as compilers usually are, as a pipeline over one intermediate representation:

    rust.py      front end: one Rust function's `asm!` block, fail-closed: its template lines
                 (with `macro_rules!` invocations expanded, and their origins kept), its operand
                 declarations, the bindings before it, and the array it returns
    aarch64.py   lifting: each instruction set's semantics, from its instructions to the IR,
    x86_64.py    with its own checks (registers written before read, operand directions,
                 the form or validity of the flags, read-only pointers)
    ir.py        the IR: a straight-line program in A-normal form, one node per operation of the
                 Lean semantics, and liveness on its bindings
    reroll.py    a pass folding unrolled loops back into a round called per repetition, after
                 checking that the repetitions are alpha-equivalent
    lean.py      back end: the Lean text of each program
    skeleton.py  back end: the mechanical part of each program's correctness proof, as symbolic
                 execution; each node states its own proof step
    specs.py     the check that the hand-written proofs contain the generated skeletons

A node is the unit of everything after lifting: the transcription prints its `let`s, liveness reads
them, rerolling renames and compares it, and the skeleton asks it for its step. So another
instruction set is another lifter with its own node classes; nothing downstream changes.

The crate's configuration (which blocks, how their rounds fold, how the proofs name things, which
proof file proves which routine) is in `../pasta/`, and `../gen.py` is the command line.
"""
