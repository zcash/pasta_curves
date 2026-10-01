/-
Copyright (c) 2026 the pasta-asm contributors.
Released under the Apache License, Version 2.0, as described in the file LICENSE.
-/
import PastaAArch64Asm.Semantics

/-!
# An AArch64 machine state, and the blocks as programs over it

`Transcription.lean` models each inline block as a function from its operands to its outputs.
That is a statement about values only: it cannot say that a block leaves memory, the stack
pointer, its input registers, and the registers it was not given untouched, which is what the
block's operand declarations and `options(pure, nomem, nostack)` promise the compiler.

This module models the machine those promises are about: the 31 general-purpose registers, the
carry flag, memory, and the stack pointer. An instruction is data (`Instr`), executed by `step`,
and a block is a list of instructions over the physical registers that an allocation assigns to
its operands. The theorems about a block then hold for every allocation that keeps distinct
operands in distinct registers, which is what the compiler guarantees for operands declared
`in`, `out`, and `inout` (no `lateout`).

Only the carry flag is modelled among the NZCV flags: the blocks write the others but read only
the carry (`adcs`, `sbcs`, `adc`, and `csel` on `lo` and `cs`). Memory and the stack pointer are
part of the state although no instruction here touches them: that no instruction can is what
`run_mem` and `run_sp` state.
-/

namespace PastaAArch64Asm.Machine

/-- A general-purpose register, `x0` to `x30`. -/
abbrev Reg := Fin 31

/-- The machine state the blocks can observe or change. -/
structure State where
  /-- The general-purpose registers. -/
  reg : Reg → Nat
  /-- The carry flag, `0` or `1`. -/
  carry : Nat
  /-- Memory, by address. -/
  mem : Nat → Nat
  /-- The stack pointer. -/
  sp : Nat

/-- A source operand: a register, the zero register `xzr`, or an immediate. -/
inductive Src where
  | reg (r : Reg)
  | zero
  | imm (n : Nat)

/-- A destination of `subs`/`sbcs`: a register, or `xzr` to keep only the flags. -/
inductive Dst where
  | reg (r : Reg)
  | zero

/-- The conditions of `csel` that the blocks use. -/
inductive Cond where
  /-- `lo` (also written `cc`): carry clear. -/
  | lo
  /-- `cs` (also written `hs`): carry set. -/
  | cs

/-- The instructions of the blocks, one constructor per mnemonic, operands in assembly order. -/
inductive Instr where
  | mov (d : Reg) (a : Src)
  | mul (d : Reg) (a b : Src)
  | umulh (d : Reg) (a b : Src)
  | lsl (d : Reg) (a : Src) (k : Nat)
  | lsr (d : Reg) (a : Src) (k : Nat)
  | adds (d : Reg) (a b : Src)
  | adcs (d : Reg) (a b : Src)
  | adc (d : Reg) (a b : Src)
  | subs (d : Dst) (a b : Src)
  | sbcs (d : Dst) (a b : Src)
  | csel (d : Reg) (a b : Src) (c : Cond)

namespace State

/-- The value a source operand reads. -/
def read (s : State) : Src → Nat
  | .reg r => s.reg r
  | .zero => 0
  | .imm n => n

/-- Write register `r`. -/
def write (s : State) (r : Reg) (v : Nat) : State :=
  { s with reg := fun r' => if r' = r then v else s.reg r' }

/-- Write a destination; a write to `xzr` is discarded. -/
def writeDst (s : State) : Dst → Nat → State
  | .reg r, v => s.write r v
  | .zero, _ => s

/-- Write a destination and the carry: the result and carry-out of `addc` or `subc`. -/
def writeFlags (s : State) (d : Dst) (p : Nat × Nat) : State :=
  { s.writeDst d p.1 with carry := p.2 }

end State

/-- Execute one instruction, over the semantics of `PastaAArch64Asm.Semantics`. -/
def step (s : State) : Instr → State
  | .mov d a => s.write d (s.read a)
  | .mul d a b => s.write d (mulLo (s.read a) (s.read b))
  | .umulh d a b => s.write d (umulh (s.read a) (s.read b))
  | .lsl d a k => s.write d (PastaAArch64Asm.lsl (s.read a) k)
  | .lsr d a k => s.write d (PastaAArch64Asm.lsr (s.read a) k)
  | .adds d a b => s.writeFlags (.reg d) (addc (s.read a) (s.read b) 0)
  | .adcs d a b => s.writeFlags (.reg d) (addc (s.read a) (s.read b) s.carry)
  | .adc d a b => s.write d (addc (s.read a) (s.read b) s.carry).1
  | .subs d a b => s.writeFlags d (subc (s.read a) (s.read b) 1)
  | .sbcs d a b => s.writeFlags d (subc (s.read a) (s.read b) s.carry)
  | .csel d a b .lo => s.write d (cselLo s.carry (s.read a) (s.read b))
  | .csel d a b .cs => s.write d (cselCs s.carry (s.read a) (s.read b))

/-- Execute a block, first instruction first. -/
def run (s : State) (p : List Instr) : State := p.foldl step s

/-- The registers an instruction writes. -/
def Instr.writes : Instr → List Reg
  | .mov d _ | .mul d _ _ | .umulh d _ _ | .lsl d _ _ | .lsr d _ _ => [d]
  | .adds d _ _ | .adcs d _ _ | .adc d _ _ | .csel d _ _ _ => [d]
  | .subs (.reg d) _ _ | .sbcs (.reg d) _ _ => [d]
  | .subs .zero _ _ | .sbcs .zero _ _ => []

/-! ## What holds for every program: no memory, no stack, a register frame

The proofs go through the three ways a state is written (`write`, `writeDst`, `writeFlags`)
rather than unfolding `step` by `rfl`: checking `step` definitionally can unfold the `Nat`
arithmetic of `addc` and `subc` on the literal `2^64`, which does not terminate in practice. -/

namespace State

theorem write_mem (s : State) (r : Reg) (v : Nat) : (s.write r v).mem = s.mem := rfl
theorem write_sp (s : State) (r : Reg) (v : Nat) : (s.write r v).sp = s.sp := rfl

theorem writeFlags_mem (s : State) (d : Dst) (p : Nat × Nat) : (s.writeFlags d p).mem = s.mem := by
  cases d <;> rfl

theorem writeFlags_sp (s : State) (d : Dst) (p : Nat × Nat) : (s.writeFlags d p).sp = s.sp := by
  cases d <;> rfl

theorem write_reg_ne (s : State) {r r' : Reg} (v : Nat) (h : r' ≠ r) :
    (s.write r v).reg r' = s.reg r' := by
  simp only [write, if_neg h]

theorem writeFlags_reg_zero (s : State) (p : Nat × Nat) : (s.writeFlags .zero p).reg = s.reg := rfl

theorem writeFlags_reg_ne (s : State) {r r' : Reg} (p : Nat × Nat) (h : r' ≠ r) :
    (s.writeFlags (.reg r) p).reg r' = s.reg r' := write_reg_ne s p.1 h

end State

theorem step_mem (s : State) (i : Instr) : (step s i).mem = s.mem := by
  cases i with
  | csel _ _ _ c => cases c <;> exact State.write_mem ..
  | adds | adcs | subs | sbcs => exact State.writeFlags_mem ..
  | _ => exact State.write_mem ..

theorem step_sp (s : State) (i : Instr) : (step s i).sp = s.sp := by
  cases i with
  | csel _ _ _ c => cases c <;> exact State.write_sp ..
  | adds | adcs | subs | sbcs => exact State.writeFlags_sp ..
  | _ => exact State.write_sp ..

/-- No program in this instruction set changes memory: `nomem`, for every block. -/
theorem run_mem (s : State) (p : List Instr) : (run s p).mem = s.mem := by
  induction p generalizing s with
  | nil => rfl
  | cons i p ih => exact (ih (step s i)).trans (step_mem s i)

/-- No program in this instruction set changes the stack pointer: with `run_mem`, `nostack`. -/
theorem run_sp (s : State) (p : List Instr) : (run s p).sp = s.sp := by
  induction p generalizing s with
  | nil => rfl
  | cons i p ih => exact (ih (step s i)).trans (step_sp s i)

theorem step_reg_of_not_mem (s : State) (i : Instr) (r : Reg) (h : r ∉ i.writes) :
    (step s i).reg r = s.reg r := by
  cases i with
  | subs d _ _ | sbcs d _ _ =>
    cases d with
    | reg d => exact State.writeFlags_reg_ne _ _ (by simpa [Instr.writes] using h)
    | zero => exact congrFun (State.writeFlags_reg_zero _ _) r
  | adds d _ _ | adcs d _ _ =>
    exact State.writeFlags_reg_ne _ _ (by simpa [Instr.writes] using h)
  | csel d _ _ c =>
    cases c <;> exact State.write_reg_ne _ _ (by simpa [Instr.writes] using h)
  | _ => exact State.write_reg_ne _ _ (by simpa [Instr.writes] using h)

/-- A register that no instruction of the program writes keeps its value. -/
theorem run_reg_of_not_mem (s : State) (p : List Instr) (r : Reg)
    (h : r ∉ p.flatMap Instr.writes) : (run s p).reg r = s.reg r := by
  induction p generalizing s with
  | nil => rfl
  | cons i p ih =>
    simp only [List.flatMap_cons, List.mem_append, not_or] at h
    exact (ih (step s i) h.2).trans (step_reg_of_not_mem s i r h.1)

/-! ## Reading through writes, under an injective allocation -/

theorem read_write (s : State) (r r' : Reg) (v : Nat) :
    (s.write r v).reg r' = if r' = r then v else s.reg r' := rfl

/-- Under an injective allocation `σ`, a write to operand `x` is read back at operand `y` exactly
when `y = x`: the equality of registers becomes an equality of operand names, which `decide`
settles. -/
theorem read_write_alloc {α : Type} [DecidableEq α] {σ : α → Reg} (hσ : Function.Injective σ)
    (s : State) (x y : α) (v : Nat) :
    (s.write (σ x) v).reg (σ y) = if y = x then v else s.reg (σ y) := by
  simp only [read_write]
  by_cases h : y = x
  · simp [h]
  · simp [h, hσ.ne h]

end PastaAArch64Asm.Machine
