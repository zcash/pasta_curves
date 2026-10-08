import PastaCurves.Semantics

/-!
# AArch64 instruction semantics for the Pasta Montgomery routines

The subset of AArch64 that the crate's routines and inline blocks use. The Montgomery blocks use
`mul`, `umulh`, `adds`/`adcs`/`adc`, `subs`/`sbcs`, `lsl`/`lsr` by an immediate, `csel` on the `lo`
and `cs` conditions, and `mov`. The inversion blocks add the flag-setting `tst`, `cmp`, and `ccmp`,
the conditional `csel`, `cneg`, and `csetm` on the `ne`, `ge`, and `mi` conditions, the
multiply-accumulates `madd`, `msub`, and `mneg`, and the signed bitfield extract `sbfx`; the
flagless `add`, `sub`, and `neg`, the signed shift `asr`, `extr`, and the bitwise `and`, `orr`, and
`eor` are in `PastaCurves/Semantics.lean`, shared with the other architectures. Loads and stores are
not modelled as memory operations: the generated programs read their operand limbs where the
assembly loads them and return the limbs the assembly stores.

Registers hold natural numbers below `2^64`; a signed quantity is its two's-complement word, and
the signed operations are spelled out on that word (`asr`, `sbfx`, `negw`). The flags are
modelled in two forms. The carry chains of the Montgomery blocks carry only the carry `c`, as
the quotient of a sum by `2^64`, which is all that `adcs`, `sbcs`, and `csel` on `cs`/`lo` read.
The inversion blocks read other conditions, so their flag-setting instructions produce the four
flags `Flags`, and the conditional instructions after them take the `Flags`.

AArch64's carry convention for subtraction is the one modelled here: after `subs`/`sbcs` the
carry is set exactly when no borrow occurred, so `sbcs` subtracts `1 - carry` and `subs`
behaves as `sbcs` with the carry set. The `lo` condition of `csel` (also written `cc`) is
"carry clear", and `cs` is "carry set".
-/

namespace PastaCurves.AArch64

/-- `subs` and `sbcs`: the low 64 bits of `a - b - (1 - c)` and the carry-out, where a carry of
`1` means that no borrow occurred. `subs` passes carry-in `1`. The difference is formed as
`a + 2^64 - b - (1 - c)`, which is nonnegative for in-range operands, so the quotient by `2^64`
is `1` exactly when `a ≥ b + (1 - c)`. -/
def subc (a b c : Nat) : Nat × Nat :=
  ((a + regMod - b - (1 - c)) % regMod, (a + regMod - b - (1 - c)) / regMod)

/-- `csel d, x, y, lo`: `x` when the carry is clear, else `y`. -/
def cselLo (c x y : Nat) : Nat := if c = 0 then x else y

/-- `csel d, x, y, cs`: `x` when the carry is set, else `y`. -/
def cselCs (c x y : Nat) : Nat := if c = 0 then y else x

/-! ## The multiply-accumulates and the signed bitfield extract -/

/-- `madd d, a, b, c`: the low 64 bits of `a * b + c`. -/
def madd (a b c : Nat) : Nat := (a * b + c) % regMod

/-- `msub d, a, b, c`: the low 64 bits of `c - a * b`. -/
def msub (a b c : Nat) : Nat := (c + regMod - a * b % regMod) % regMod

/-- `mneg d, a, b`: the low 64 bits of `-(a * b)`. -/
def mneg (a b : Nat) : Nat := (regMod - a * b % regMod) % regMod

/-- `sbfx d, a, #lsb, #w`: the `w`-bit field of `a` at bit `lsb`, sign-extended. -/
def sbfx (a lsb w : Nat) : Nat :=
  let field := a / 2^lsb % 2^w
  if field < 2^(w - 1) then field else field + (regMod - 2^w)

/-! ## The four flags and the conditions on them -/

/-- The NZCV flags, each `0` or `1`. -/
structure Flags where
  /-- The result was negative as a signed word. -/
  n : Nat
  /-- The result was zero. -/
  z : Nat
  /-- Carry: for a subtraction, no borrow occurred. -/
  c : Nat
  /-- Signed overflow. -/
  v : Nat
  deriving DecidableEq, Repr

/-- The flags of `tst a, b`, from the result `r = a &&& b`: `N` is its top bit, `Z` its being
zero, and `C` and `V` are cleared. -/
def tstFlags (r : Nat) : Flags := ⟨r / 2^63, if r = 0 then 1 else 0, 0, 0⟩

/-- The flags of `cmp a, b` (`subs xzr, a, b`): from the difference `a - b` as a word, `N` its
top bit, `Z` its being zero, `C` that no borrow occurred, and `V` that the signed difference
overflowed, which happens exactly when the operands' signs differ and the result's sign differs
from `a`'s. -/
def cmpFlags (a b : Nat) : Flags :=
  let r := (a + regMod - b) % regMod
  ⟨r / 2^63, if r = 0 then 1 else 0, (a + regMod - b) / regMod,
    if a / 2^63 ≠ b / 2^63 ∧ r / 2^63 ≠ a / 2^63 then 1 else 0⟩

/-- The flags an immediate `#nzcv` of `ccmp` sets. -/
def immFlags (nzcv : Nat) : Flags := ⟨nzcv / 8 % 2, nzcv / 4 % 2, nzcv / 2 % 2, nzcv % 2⟩

/-- `ccmp a, b, #nzcv, ne`: the flags of `cmp a, b` when `Z` is clear, else the immediate's. -/
def ccmpNe (fl : Flags) (a b nzcv : Nat) : Flags :=
  if fl.z = 0 then cmpFlags a b else immFlags nzcv

/-- `csel d, x, y, ne`: `x` when `Z` is clear, else `y`. -/
def cselNe (fl : Flags) (x y : Nat) : Nat := if fl.z = 0 then x else y

/-- `csel d, x, y, ge`: `x` when `N = V`, else `y`. -/
def cselGe (fl : Flags) (x y : Nat) : Nat := if fl.n = fl.v then x else y

/-- `cneg d, a, ge`: the negation of `a` when `N = V`, else `a`. -/
def cnegGe (fl : Flags) (a : Nat) : Nat := if fl.n = fl.v then negw a else a

/-- `cneg d, a, mi`: the negation of `a` when `N` is set, else `a`. -/
def cnegMi (fl : Flags) (a : Nat) : Nat := if fl.n = 1 then negw a else a

/-- `csetm d, mi`: all ones when `N` is set, else zero. -/
def csetmMi (fl : Flags) : Nat := if fl.n = 1 then regMod - 1 else 0

end PastaCurves.AArch64
