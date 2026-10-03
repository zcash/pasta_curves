import Mathlib.Data.List.Prime
import Mathlib.NumberTheory.LucasPrimality

/-!
# Primality by Pratt certificates

Lucas's test (`lucas_primality`) proves `p` prime from a witness `a` with
`a^(p-1) ≡ 1 (mod p)` and `a^((p-1)/q) ≢ 1 (mod p)` for every prime `q` that divides `p - 1`.
A Pratt certificate supplies the witness and the factorization of `p - 1`, whose primes are
certified in the same way. Here a certificate is a list of entries in which every factor of an
entry is certified by an earlier one, and `check` is a Boolean test of it that the kernel can
evaluate: the exponentiations are square-and-multiply, so that a 255-bit prime costs a few
hundred multiplications per exponent. `check_sound` proves every certified number prime.
-/

namespace PastaCurves.Pratt

/-- `a ^ n % m` by square-and-multiply over the `bits` low bits of `n`; `powMod_eq` says it is
exact when `n < 2^bits`. -/
def powMod (m : ℕ) : ℕ → ℕ → ℕ → ℕ
  | 0, _, _ => 1 % m
  | bits + 1, a, n =>
    let r := powMod m bits (a * a % m) (n / 2)
    if n % 2 = 1 then r * a % m else r

/-- Square-and-multiply computes the power when the exponent fits in `bits` bits. -/
theorem powMod_eq (m : ℕ) : ∀ (bits a n : ℕ), n < 2 ^ bits → powMod m bits a n = a ^ n % m
  | 0, a, n, h => by
    have : n = 0 := by simpa using h
    subst this
    simp [powMod]
  | bits + 1, a, n, h => by
    have hn : n / 2 < 2 ^ bits := by rw [pow_succ] at h; omega
    have ih := powMod_eq m bits (a * a % m) (n / 2) hn
    have key : (a * a % m) ^ (n / 2) % m = a ^ (2 * (n / 2)) % m := by
      rw [← Nat.pow_mod, pow_mul, sq]
    simp only [powMod]
    rw [ih, key]
    split_ifs with h2
    · conv_rhs => rw [← Nat.div_add_mod n 2, h2, pow_succ]
      rw [Nat.mod_mul_mod]
    · have h0 : n % 2 = 0 := by omega
      conv_rhs => rw [← Nat.div_add_mod n 2, h0, add_zero]

/-- One entry of a certificate: the number `p`, the witness `a`, and the factorization of
`p - 1` as pairs of a prime and its exponent. -/
structure Entry where
  /-- The number certified prime. -/
  p : ℕ
  /-- Lucas's witness. -/
  a : ℕ
  /-- The factorization of `p - 1`: each prime with its exponent. -/
  factors : List (ℕ × ℕ)

/-- Lucas's test for one entry, over exponents of `bits` bits, given the numbers certified
before it: the factors multiply to `p - 1` and are all certified, `a^(p-1) ≡ 1`, and
`a^((p-1)/q) ≢ 1` for each factor `q`. -/
def checkEntry (bits : ℕ) (proven : List ℕ) (e : Entry) : Bool :=
  2 ≤ e.p && e.p ≤ 2 ^ bits &&
    e.factors.all (fun f => decide (f.1 ∈ proven)) &&
    (e.factors.map fun f => f.1 ^ f.2).prod == e.p - 1 &&
    powMod e.p bits e.a (e.p - 1) == 1 &&
    e.factors.all (fun f => powMod e.p bits e.a ((e.p - 1) / f.1) != 1)

/-- A certificate's entries in order, each checked against the numbers certified before it. -/
def check (bits : ℕ) : List ℕ → List Entry → Bool
  | _, [] => true
  | proven, e :: es => checkEntry bits proven e && check bits (e.p :: proven) es

/-- `a ^ n ≡ 1 (mod p)` in `ZMod p` is `a ^ n % p = 1` in `ℕ`, for `2 ≤ p`. -/
theorem zmod_pow_eq_one_iff {p : ℕ} (hp : 2 ≤ p) (a n : ℕ) :
    ((a : ZMod p) ^ n = 1) ↔ a ^ n % p = 1 := by
  rw [← Nat.cast_pow, ← Nat.cast_one, ZMod.natCast_eq_natCast_iff',
    Nat.one_mod_eq_one.mpr (by omega)]

/-- An entry that passes Lucas's test, with its factors prime, is prime. -/
theorem checkEntry_sound {bits : ℕ} {proven : List ℕ} {e : Entry}
    (hproven : ∀ q ∈ proven, q.Prime) (h : checkEntry bits proven e = true) : e.p.Prime := by
  simp only [checkEntry, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, beq_iff_eq,
    bne_iff_ne, ne_eq] at h
  obtain ⟨⟨⟨⟨⟨h2, hbits⟩, hfp⟩, hprod⟩, hfull⟩, hpart⟩ := h
  have hlt : ∀ n, n ≤ e.p - 1 → n < 2 ^ bits := fun n hn => by omega
  refine lucas_primality e.p (e.a : ZMod e.p) ?_ ?_
  · rw [zmod_pow_eq_one_iff h2, ← powMod_eq _ _ _ _ (hlt _ le_rfl)]
    exact hfull
  · intro q hq hdvd
    rw [← hprod] at hdvd
    obtain ⟨x, hx, hqx⟩ := (Prime.dvd_prod_iff hq.prime).mp hdvd
    obtain ⟨f, hf, rfl⟩ := List.mem_map.mp hx
    have hf1 : f.1.Prime := hproven _ (hfp f hf)
    have hq_eq : q = f.1 := (Nat.prime_dvd_prime_iff_eq hq hf1).mp (hq.dvd_of_dvd_pow hqx)
    subst hq_eq
    rw [ne_eq, zmod_pow_eq_one_iff h2, ← powMod_eq _ _ _ _ (hlt _ (Nat.div_le_self _ _))]
    exact hpart f hf

/-- Every number that a checked certificate certifies is prime. -/
theorem check_sound {bits : ℕ} :
    ∀ {proven : List ℕ} {es : List Entry}, (∀ q ∈ proven, q.Prime) →
      check bits proven es = true → ∀ e ∈ es, e.p.Prime
  | _, [], _, _, _, he => by simp at he
  | proven, e :: es, hproven, h, e', he' => by
    simp only [check, Bool.and_eq_true] at h
    have hp := checkEntry_sound hproven h.1
    rcases List.mem_cons.mp he' with rfl | he'
    · exact hp
    · refine check_sound (proven := e.p :: proven) ?_ h.2 e' he'
      intro q hq
      rcases List.mem_cons.mp hq with rfl | hq
      · exact hp
      · exact hproven q hq

/-- The last number of a checked certificate is prime: the form in which the Pasta primes use
it. -/
theorem prime_of_check {bits : ℕ} {es : List Entry} (h : check bits [] es = true) {p : ℕ}
    (hp : p ∈ es.map Entry.p) : p.Prime := by
  obtain ⟨e, he, rfl⟩ := List.mem_map.mp hp
  exact check_sound (by simp) h e he

end PastaCurves.Pratt
