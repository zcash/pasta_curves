import PastaCurves.Eisenstein.Ring

/-!
# The odd residue classes mod 8, and why eight points cover them

`glv_eisenstein`'s recoder is a width-3 NAF over `ℤ[ω]`: a nonzero column subtracts the digit in
the residual's class mod 8. Its look-up table `LUT` (`build_lut` in `src/glv_eisenstein.rs`)
writes, for each of the eight orbit representatives `REPS`, each rotation `ω^rot`, and each sign,
the digit `±ω^rot r` into the slot of its class. This module states that construction and proves,
by evaluation in the kernel, what the crate relies on:

- the 48 digits fall in 48 distinct classes (`digits_mod8_nodup`), so no write of `build_lut`
  overwrites another;
- those classes are exactly the odd ones, the classes not divisible by 2 (`mem_digits_iff_odd`), so
  every odd residual has a digit and no even one does;
- the action of the six units on the odd classes is free (`unit_action_free`), the fact that makes
  `48 / 6 = 8` representatives enough;
- every digit's coordinates are at most 5 in absolute value (`digit_bound`), the bound the
  residual analysis in `Ladder.lean` and the crate's `MAX_DIGITS` and `i128` headroom use.

The same facts are checked in `sage/glv_eisenstein.sage` by steps `[2]` and `[3]`, and the first two
by the crate's `lut_covers_exactly_the_odd_classes` and `orbit_action_is_free` tests.
-/

namespace PastaCurves.Eisenstein

open E

/-- The orbit representatives, the Rust `REPS` in its order, as coefficient pairs `(a, b)` meaning
`a + b ω`. -/
def reps : List (E ℤ) :=
  [⟨1, 0⟩, ⟨1, -1⟩, ⟨2, -1⟩, ⟨1, -2⟩, ⟨3, 0⟩, ⟨3, -1⟩, ⟨1, -3⟩, ⟨2, -3⟩]

/-- `ω^rot r`, negated when `neg`: the digit `build_lut` stores for `(rep, rot, neg)`. -/
def digitOf (r : E ℤ) (rot : ℕ) (neg : Bool) : E ℤ :=
  let d := rotate^[rot] r
  if neg then -d else d

/-- The 48 digits, in the order `build_lut` writes them: by representative, then rotation, then
sign. -/
def digits : List (E ℤ) :=
  reps.flatMap fun r => (List.range 3).flatMap fun rot => [false, true].map (digitOf r rot)

/-- Reduction mod 8, coordinate-wise: the class the recoder reads off a residual's low three bits. -/
def mod8 (x : E ℤ) : E (ZMod 8) := x.map (Int.castRingHom (ZMod 8))

/-- A class mod 8 is odd when it is not divisible by 2, that is, when some coordinate is odd
(2 being inert, `E.two_dvd_mul`). -/
def Odd8 (c : E (ZMod 8)) : Prop := ¬ (c.a.val % 2 = 0 ∧ c.b.val % 2 = 0)

instance (c : E (ZMod 8)) : Decidable (Odd8 c) := by unfold Odd8; infer_instance

/-- The digits, reduced mod 8, written out. -/
def digits8 : List (E (ZMod 8)) := digits.map mod8

theorem digits_length : digits.length = 48 := by decide

/-- The 48 digits lie in 48 distinct classes: `build_lut`'s writes never collide. -/
theorem digits_mod8_nodup : digits8.Nodup := by decide

/-- A class mod 8 holds a digit exactly when it is odd: every odd residual finds its digit in the
table, and the 16 even classes keep `ZERO_DIGIT`. -/
theorem mem_digits_iff_odd : ∀ c : E (ZMod 8), c ∈ digits8 ↔ Odd8 c := by
  rintro ⟨a, b⟩
  revert a b
  decide

/-- The six units act freely on the odd classes: only `1` fixes one. Hence every orbit has six
elements, the 48 odd classes fall into exactly eight orbits, and a table of eight points, recovered
in their other five forms by `φ` and negation, reaches every digit. -/
theorem unit_action_free :
    ∀ c : E (ZMod 8), Odd8 c → ∀ u ∈ E.units, mod8 u * c = c → u = 1 := by
  rintro ⟨a, b⟩
  revert a b
  decide

/-- Every digit's coordinates are at most 5 in absolute value. -/
theorem digit_bound : ∀ d ∈ digits, |d.a| ≤ 5 ∧ |d.b| ≤ 5 := by decide

/-- Every digit is a unit times a representative, the unit being `±ω^rot`: the factorisation the
packed digit `rep | rot << 3 | neg << 5` records. -/
theorem digitOf_eq_unit_mul (r : E ℤ) (rot : ℕ) (neg : Bool) (hrot : rot < 3) :
    ∃ u ∈ E.units, digitOf r rot neg = u * r := by
  interval_cases rot <;> cases neg
  · exact ⟨⟨1, 0⟩, by decide, by ext <;> simp [digitOf]⟩
  · exact ⟨⟨-1, 0⟩, by decide, by ext <;> simp [digitOf]⟩
  · exact ⟨⟨0, 1⟩, by decide, by ext <;> simp [digitOf, rotate]⟩
  · exact ⟨⟨0, -1⟩, by decide, by ext <;> simp [digitOf, rotate]; ring⟩
  · exact ⟨⟨-1, -1⟩, by decide, by ext <;> simp [digitOf, rotate] <;> ring⟩
  · exact ⟨⟨1, 1⟩, by decide, by ext <;> simp [digitOf, rotate]⟩

end PastaCurves.Eisenstein
