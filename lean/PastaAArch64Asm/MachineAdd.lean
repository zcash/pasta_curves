/-
Copyright (c) 2026 the pasta-asm contributors.
Released under the Apache License, Version 2.0, as described in the file LICENSE.
-/
import PastaAArch64Asm.Machine
import PastaAArch64Asm.Transcription

/-!
# `add`'s block on the machine (prototype)

The block as a program over the machine, for an allocation `σ` of its operands to registers,
and the theorem that, for every injective allocation, running it leaves memory, the stack
pointer, its input registers, and every register outside the allocation untouched, and leaves
`addMod` (the transcription) in its output registers.

PROTOTYPE: `addBlock` is written by hand here, one constructor per template line of `add` in
`src/asm/aarch64.rs`. In the real change it would be generated, like the transcription.
-/

namespace PastaAArch64Asm.Machine

/-- The operands of `add`'s block, by placeholder. -/
inductive AddOperand where
  | r0 | r1 | r2 | r3 | b0 | b1 | b2 | b3 | p0 | p1 | p3 | t0 | t1 | t2 | t3
  deriving DecidableEq

/-- `add`'s block for the allocation `σ`, one instruction per template line. -/
def addBlock (σ : AddOperand → Reg) : List Instr :=
  let r := fun o => Src.reg (σ o)
  [ .adds (σ .r0) (r .r0) (r .b0),                -- adds {r0}, {r0}, {b0}
    .adcs (σ .r1) (r .r1) (r .b1),                -- adcs {r1}, {r1}, {b1}
    .adcs (σ .r2) (r .r2) (r .b2),                -- adcs {r2}, {r2}, {b2}
    .adc (σ .r3) (r .r3) (r .b3),                 -- adc {r3}, {r3}, {b3}
    .subs (.reg (σ .t0)) (r .r0) (r .p0),         -- subs {t0}, {r0}, {p0}
    .sbcs (.reg (σ .t1)) (r .r1) (r .p1),         -- sbcs {t1}, {r1}, {p1}
    .sbcs (.reg (σ .t2)) (r .r2) .zero,           -- sbcs {t2}, {r2}, xzr
    .sbcs (.reg (σ .t3)) (r .r3) (r .p3),         -- sbcs {t3}, {r3}, {p3}
    .csel (σ .r0) (r .t0) (r .r0) .cs,            -- csel {r0}, {t0}, {r0}, cs
    .csel (σ .r1) (r .t1) (r .r1) .cs,            -- csel {r1}, {t1}, {r1}, cs
    .csel (σ .r2) (r .t2) (r .r2) .cs,            -- csel {r2}, {t2}, {r2}, cs
    .csel (σ .r3) (r .t3) (r .r3) .cs ]           -- csel {r3}, {t3}, {r3}, cs

theorem State.write_carry (s : State) (r : Reg) (v : Nat) : (s.write r v).carry = s.carry := rfl

theorem State.writeFlags_reg_carry (s : State) (r : Reg) (p : Nat × Nat) :
    (s.writeFlags (.reg r) p).reg = (s.write r p.1).reg ∧ (s.writeFlags (.reg r) p).carry = p.2 :=
  ⟨rfl, rfl⟩

/-- Running `add`'s block, for any injective allocation and any initial state whose input
registers hold the operands. -/
theorem addBlock_spec (σ : AddOperand → Reg) (hσ : Function.Injective σ)
    (lhs rhs modulus : Limbs) (s : State)
    (hr0 : s.reg (σ .r0) = lhs.l0) (hr1 : s.reg (σ .r1) = lhs.l1)
    (hr2 : s.reg (σ .r2) = lhs.l2) (hr3 : s.reg (σ .r3) = lhs.l3)
    (hb0 : s.reg (σ .b0) = rhs.l0) (hb1 : s.reg (σ .b1) = rhs.l1)
    (hb2 : s.reg (σ .b2) = rhs.l2) (hb3 : s.reg (σ .b3) = rhs.l3)
    (hp0 : s.reg (σ .p0) = modulus.l0) (hp1 : s.reg (σ .p1) = modulus.l1)
    (hp3 : s.reg (σ .p3) = modulus.l3) :
    let s' := run s (addBlock σ)
    s'.mem = s.mem ∧ s'.sp = s.sp ∧
      (∀ r, (∀ o, r ≠ σ o) → s'.reg r = s.reg r) ∧
      (s'.reg (σ .b0) = rhs.l0 ∧ s'.reg (σ .b1) = rhs.l1 ∧ s'.reg (σ .b2) = rhs.l2 ∧
        s'.reg (σ .b3) = rhs.l3 ∧ s'.reg (σ .p0) = modulus.l0 ∧
        s'.reg (σ .p1) = modulus.l1 ∧ s'.reg (σ .p3) = modulus.l3) ∧
      (⟨s'.reg (σ .r0), s'.reg (σ .r1), s'.reg (σ .r2), s'.reg (σ .r3)⟩ : Limbs) =
        addMod lhs rhs modulus := by
  intro s'
  refine ⟨run_mem _ _, run_sp _ _, ?_, ?_, ?_⟩
  · intro r hr
    apply run_reg_of_not_mem
    simp only [addBlock, List.flatMap_cons, List.flatMap_nil, Instr.writes, List.mem_append,
      List.mem_cons, List.not_mem_nil, or_false]
    intro h
    rcases h with h | h | h | h | h | h | h | h | h | h | h | h <;> exact hr _ h
  all_goals
    simp only [s', run, addBlock, List.foldl, step, State.writeFlags, State.writeDst,
      State.read, read_write_alloc hσ, State.write_carry, reduceCtorEq, if_false, if_true,
      addMod, hr0, hr1, hr2, hr3, hb0, hb1, hb2, hb3, hp0, hp1, hp3, and_self]

end PastaAArch64Asm.Machine
