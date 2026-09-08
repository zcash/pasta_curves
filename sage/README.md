# SageMath derivations of the crate's constants

The scripts in this directory recompute, from the curve definitions
alone, the magic constants that appear in the crate's source, printing
them in the exact shape of the Rust code so the two can be diffed:

- `glv_constants.sage` — the GLV short-basis and Babai-rounding
  constants in the `GlvParams` implementations in `src/glv.rs`
  (`V1A`, `V1B_NEG`, `V2A`, `V2B`, `G1`, `G2`, for both curves).
- `glv_boundary_scalars.sage` — the `*_BOUNDARY_SCALAR` witnesses used
  by the `babai_boundary_*` and `native_vs_glv_boundary_*` regression
  tests in `src/glv.rs`.
- `asm_mutant_witnesses.py`: the `AARCH64_ASM_CARRY_PRODUCTS` and
  `AARCH64_ASM_CARRY_REDUCTIONS` vectors used by the
  `aarch64_asm_matches_portable_at_carry_boundaries` tests in
  `src/fields/fp.rs` and `src/fields/fq.rs`. It builds on
  `asm_aarch64.py`, an interpreter for `src/asm/pasta_mul-armv8.S`.
  Pure Python; SageMath is not required for these two.

The scripts use exact integer/rational arithmetic and hand-rolled
lattice reduction only, so their output is deterministic and does not
depend on the SageMath version.

## Running

Sage is pinned via [uv](https://docs.astral.sh/uv/) using
[passagemath](https://pypi.org/project/passagemath-standard/), the
pip-installable distribution of SageMath (see `pyproject.toml` /
`uv.lock`). From this directory:

```console
$ uv run sage glv_constants.sage
$ uv run sage glv_boundary_scalars.sage
```

Any reasonably recent standalone SageMath installation works too:
`sage glv_constants.sage`.

`asm_mutant_witnesses.py` needs no Sage at all:

```console
$ python3 asm_mutant_witnesses.py
```

## The AArch64 carry-boundary vectors

`cargo xtask asm-mutants` flips one carry decision in the AArch64 backend
at a time and reports the mutants that no test kills. A surviving mutant is
a carry propagation the suite never exercises. These are not reachable by
random testing: they need a limb to be exactly all-ones with a carry
arriving into it, which random operands hit with probability about 2^-64.

`asm_mutant_witnesses.py` reaches them by construction rather than by
sampling. In the shared reduction helper, and in any `mul` round whose
`rhs` limb is zero, a round is exactly one Montgomery cancellation step,

    t_next = (t + q * p) / 2^64,   q = limb0(t) * inv mod 2^64,

and because `inv` is `-p[0]^-1` that step is self-consistent for every
`q`, so it inverts: the predecessor of a state below `p` is `t * 2^64 mod
p`, again below `p`. Iterating gives, for a chosen internal state `t` at
step `k`, a `from_mont` input `t * 2^(64k) mod p`, and similarly a `mul`
operand pair. Any state can therefore be requested on demand, including
the all-ones states that make a carry propagate the full width.

Some vectors have a left operand that is NOT canonical. That is
deliberate and matches how the backend is really called: `from_u512`
multiplies an arbitrary 256-bit value by `R2`/`R3`. Those inputs are what
make the per-round fifth accumulator limb live at all. They stay inside the
operand range in `src/fields/aarch64_asm.rs`, which allows an unreduced left
operand only against a right operand whose every limb is at most `2^64 - 4`;
`R2` and `R3` are, and `tests/test_operand_range.py` pins both that and the
divergence just outside it.

### Mutants that no vector can kill

Ten surviving mutants are equivalent mutants: no input distinguishes them
from the original, so no test can kill them. They mark code that is
unreachable given the routines' preconditions, not gaps in the tests.

Two different mechanisms produce that, and they are worth keeping apart:

- the carry the mutation forces is the only one the code ever produces, so
  the instruction is a no-op and no input can reach a difference at all.
  Seven of the ten are of this kind.
- the carry does vary, but the difference cancels before it reaches
  anything observable. The three `sbcs` entries are of this kind. The
  borrow they force really does differ, on between 40% and 99.8% of
  sampled inputs depending on the limb, and the result is unchanged
  anyway.

#### How far these claims are established

The arguments below are proofs about the *arithmetic*, under the stated
preconditions. They are not proofs about the shipped instructions. There
are three layers, and only the first is actually proved:

1. **The arithmetic.** Each argument below is complete. `adc/19`, for
   instance, is a proof over the whole input domain rather than a spot
   check: the cross-term sum is a sum of products of non-negative terms,
   so it is monotone non-decreasing in each limb, so its maximum over the box
   `[0, 2^64)^3 x [0, 2^62]` is attained at the corner, and that corner
   evaluates to a 448-bit number, hence below the `2^448` at which a
   carry would reach limb 7.
2. **The instructions.** That this instruction sequence computes the
   arithmetic those proofs are about is established by differential
   testing, not by proof: `asm_aarch64.py` agrees with the mathematical
   result across the ranges exercised below, its own tests hold it against
   the CPU it models, and the Rust tests compare the real compiled backend
   against the portable implementation. A transcription error in the
   interpreter would invalidate the chain, and nothing here rules that
   out.
3. **The preconditions.** `rhs < p`, and `square` and `from_mont` being
   called only on canonical elements, are properties of the callers,
   established by reading the crate rather than enforced by a type.

`python3 asm_mutant_witnesses.py --check-invariants` narrows layer 2 the
cheap way. For every mutant in the tracked list it runs the routine over
the operand range that mutant's argument assumes, finds the inputs whose
carry differs from the value the mutation forces, and checks the mutated
routine still returns the same result on exactly those inputs. It samples,
so it refutes rather than proves; a violation, however, is conclusive, and
means either that a recorded argument is wrong or that the routine is
being called outside the range the argument assumes.

Closing layer 2 properly would need bit-precise symbolic execution of the
instruction sequence: the property is quantifier-free over bitvectors, so
in principle a solver can settle it, and there are tools aimed at exactly
this kind of carry chain in cryptographic assembly. That is a larger piece
of work than this directory, and has not been done.

Both Pasta moduli have `p[3] = 2^62`, hence `p < 2^255` and `2p < 2^256`,
which is what drives most of the arguments:

- `adc/15` (the final `adc x23,xzr,xzr` in `mul`). With `lhs < 2^256` and
  `rhs < p` the candidate is `(lhs*rhs + Q*p)/2^256 < 2p < 2^256`, so the
  fifth limb of the final accumulator is always zero. The per-round
  fifth limbs (`adc/3`, `adc/7`, `adc/11`) are a different matter: they do
  become one for a non-canonical `lhs`, and the vectors cover them.
- `adc/19` (the doubled cross-term carry into limb 7 of `sqr`). For a
  canonical `a` the top limb satisfies `a[3] <= 2^62`, which bounds twice
  the cross-term sum by `2^447`, a factor of two under the `2^448` where
  limb 7 begins. Relaxing `a[3]` to a full 64 bits raises that to `2^449`,
  so the carry becomes reachable and the mutant becomes killable. Note
  this bounds only the mutant's equivalence: the shipped instruction
  captures the carry either way, so the schoolbook square itself is
  correct for any input.
- `adc/21` (the 257th candidate bit in `sqr`). The reduced low half is at
  most `p` and the untouched high half is below `p`, so the sum is below
  `2p + 1 < 2^256`.
- `sbcs/9`, `sbcs/10`, `sbcs/11` (carry-clear only) and `csel/9` through
  `csel/12` (select-first only), all in `from_mont`. Reducing `v < p`
  yields `(v + Q*p)/R <= (R*p - 1)/R < p`, so the result is already
  canonical: the tentative subtraction always borrows and the `csel`
  always keeps the candidate. Forcing a borrow into limb 1, 2 or 3 can
  only change the final decision when the candidate's limb 3 equals
  `p[3]`, and that in turn forces limb 2 to zero and limb 1 to at most
  `p[1]`, which makes the borrow chain produce that same value anyway.
  The opposite direction, suppressing a borrow, is not equivalent,
  and those mutants are killed.

`from_mont`'s conditional subtraction is therefore dead code for every
input the crate can produce. It is cheap, constant-time, and guards the
routine against a non-canonical argument, so the sensible response is to
document it rather than remove it.
