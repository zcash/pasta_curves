import PastaCurves.Semantics

/-!
# x86-64 instruction semantics for the Pasta Montgomery routines

The subset of x86-64 that the crate's routines and inline blocks use. The definitions here
model the values and flags consumed by the blocks, not the entire architectural state.
`addc` supplies the result and unsigned carry of `add`, `adc`, `adcx`, and `adox`: the latter
uses OF as an unsigned carry chain, independently of CF. A transcription must keep these two
chains separate. `adcx` preserves OF, `adox` preserves CF, and `mulx` and moves preserve both.

Unlike AArch64's subtraction carry, x86-64 CF is set when subtraction borrows. `sub` passes
zero to `sbb`; `neg` sets CF exactly when its operand is nonzero. `cmovnc` replaces its
old destination only when CF is clear, without changing flags.

The low product of two-operand `imul` is `mulLo`, and the word results of immediate `shl`
and `shr` are `lsl` and `lsr`. Their flag writes must not be mistaken for preservation:
a generator may omit a flag result only when it checks that the result is overwritten before
being read. In particular, OF is undefined after the multi-bit shifts used by these routines.
The zeroing idiom `xor r, r` writes zero and clears both CF and OF, including when written to
a 32-bit subregister (which zero-extends to the full register).

ZF and SF are not kept as flags. The inversion's blocks read them only through `cmove` and
`cmovs`, right after the `test` or `add` that sets them, so a transcription passes the word that
instruction computed (`test`'s AND, `add`'s sum), and `cmovz` and `cmovs` read ZF and SF off it:
ZF is the word being zero, and SF its top bit. The generator checks that no other instruction
writes the flags, and that the word is not rewritten, in between. `bt` sets CF to one bit of its
operand, and `cmovb` reads it. One-operand `mul` multiplies RAX by its operand into RDX:RAX.
The shifts and bitwise operations reuse the shared words: `sar` is `asr`, `shrd dst, src, k` is
`extr src dst k`, `and`, `or`, and `xor` are `andw`, `orrw`, and `eorw`, `bts` is an `orrw`
with the bit's power of two, and `lea` with a base and an offset is `addw`.

Loads are modeled as reads of the corresponding input limb, not as memory operations. This
leaves pointer validity, register allocation, and the inline-assembly operand bindings in the
same external trust boundary as the AArch64 transcription.
-/

namespace PastaCurves.X86_64

/-- `sub` and `sbb`: the low word of `a - b - borrow` and CF, which is `1` on borrow.
For bounded operands and `borrow ≤ 1`, adding `2^64` keeps the natural subtraction from
truncating; its quotient is `1` exactly when the original subtraction did not borrow. -/
def sbb (a b borrow : Nat) : Nat × Nat :=
  ((a + regMod - b - borrow) % regMod, 1 - (a + regMod - b - borrow) / regMod)

/-- `neg`: the low word of `-a` and CF. -/
def neg (a : Nat) : Nat × Nat := ((sbb 0 a 0).1, if a = 0 then 0 else 1)

/-- `cmovnc dst, src`: keep `dst` on borrow, replace it with `src` when CF is clear. -/
def cmovnc (cf dst src : Nat) : Nat := if cf = 0 then src else dst

/-- `mulx hi, lo, src` in Intel syntax, with implicit multiplicand RDX.
The pair is in destination order, high word first; neither CF nor OF changes. -/
def mulx (rdx src : Nat) : Nat × Nat := (umulh rdx src, mulLo rdx src)

/-- One-operand `mul src`: RDX:RAX is RAX times `src`. The pair is `(RAX, RDX)`, low word first. -/
def mulq (rax src : Nat) : Nat × Nat := (mulLo rax src, umulh rax src)

/-- `bt a, n`: CF is bit `n` of `a`. -/
def bt (a n : Nat) : Nat := a / 2^n % 2

/-- `cmove dst, src` after an instruction whose result word is `w`: replace `dst` with `src` when
ZF is set, that is when `w` is zero. -/
def cmovz (w dst src : Nat) : Nat := if w = 0 then src else dst

/-- `cmovs dst, src` after an instruction whose result word is `w`: replace `dst` with `src` when
SF is set, that is when the top bit of `w` is. -/
def cmovs (w dst src : Nat) : Nat := if w < 2^63 then dst else src

/-- `cmovb dst, src`: replace `dst` with `src` when CF is set. -/
def cmovb (cf dst src : Nat) : Nat := if cf = 0 then dst else src

/-- Eight little-endian words, the output of the squaring block before Montgomery reduction. -/
structure WideLimbs where
  l0 : Nat
  l1 : Nat
  l2 : Nat
  l3 : Nat
  l4 : Nat
  l5 : Nat
  l6 : Nat
  l7 : Nat
  deriving DecidableEq, Repr

/-- The integer represented by the eight words. -/
def WideLimbs.toNat (x : WideLimbs) : Nat :=
  x.l0 + 2^64 * x.l1 + 2^128 * x.l2 + 2^192 * x.l3 +
    2^256 * x.l4 + 2^320 * x.l5 + 2^384 * x.l6 + 2^448 * x.l7

/-- Every word of the unreduced product is below `2^64`. -/
def WideLimbs.Bounded (x : WideLimbs) : Prop :=
  x.l0 < 2^64 ∧ x.l1 < 2^64 ∧ x.l2 < 2^64 ∧ x.l3 < 2^64 ∧
    x.l4 < 2^64 ∧ x.l5 < 2^64 ∧ x.l6 < 2^64 ∧ x.l7 < 2^64

-- Boundary cases distinguish x86-64's borrow convention from AArch64's carry convention.
example : sbb 0 0 0 = (0, 0) := by decide +kernel
example : sbb 0 0 1 = (2^64 - 1, 1) := by decide +kernel
example : sbb (2^64 - 1) (2^64 - 1) 1 = (2^64 - 1, 1) := by decide +kernel
example : neg 0 = (0, 0) := by decide +kernel
example : neg 1 = (2^64 - 1, 1) := by decide +kernel
example : mulx (2^64 - 1) 2 = (1, 2^64 - 2) := by decide +kernel
example : cmovnc 0 3 5 = 5 ∧ cmovnc 1 3 5 = 3 := by decide +kernel
example : mulq (2^64 - 1) 2 = (2^64 - 2, 1) := by decide +kernel
example : bt (2^63) 63 = 1 ∧ bt (2^63 - 1) 63 = 0 := by decide +kernel
example : cmovz 0 3 5 = 5 ∧ cmovz 1 3 5 = 3 := by decide +kernel
example : cmovs (2^63) 3 5 = 5 ∧ cmovs (2^63 - 1) 3 5 = 3 := by decide +kernel
example : cmovb 1 3 5 = 5 ∧ cmovb 0 3 5 = 3 := by decide +kernel

end PastaCurves.X86_64
