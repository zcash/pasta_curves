import PastaCurves.Leakage

/-!
# What an AArch64 instruction leaks

The leakage policy of the constant-time model for AArch64: the instructions as syntax, and what
each reveals as it runs. `Programs.lean` states the crate's blocks in this syntax, generated
from the same `asm!` blocks as `Transcription.lean`.

An instruction is syntax: a mnemonic and its operands as written. Its observation is the
instruction itself when its timing does not depend on its data, and otherwise the instruction
together with the values of its data sources (`leak`). The sources are its source registers,
the flags it reads, and the registers of a memory address.

Which timings do not depend on the data is a hardware fact, taken here as an assumption: the
data-independent-time instructions of the Arm architecture (the register `DIT`, `FEAT_DIT`,
Armv8.4). With `PSTATE.DIT` set, "the execution time of a data-independent-time sequence of code
must be independent of all data-independent-time values"; the list includes, among others, the
add/subtract (immediate, shifted register, and with carry: `ADC`, `ADCS`, `SBC`, `SBCS`), logical
(immediate and shifted register), bitfield (`SBFM`, `UBFM`), extract (`EXTR`), conditional
compare and conditional select (`CSEL`, `CSINV`, `CSNEG`), and three-source data-processing
(`MADD`, `MSUB`, `UMULH`) encodings, which with their aliases are every instruction the blocks
use but one. The exception is `mov` of an immediate, a move-wide `MOVZ`, which is not on the list
but reads no register and no flag, so it has no data to leak.

The policy also covers instructions the blocks do not use, so that it is not vacuous: a division
(not on the list), a load or store (whose address reaches the cache whatever its timing), and
the branches (whose condition is the control flow). `udiv_not_oblivious` and
`ldr_not_oblivious` check that the policy distinguishes them.

Caveats, not modelled: the guarantee holds when `PSTATE.DIT` is set, which the crate does not
do; a core without `FEAT_DIT` promises nothing; and the model is of the instructions only, not
of the microarchitectural state they share with other code.
-/

namespace PastaCurves.AArch64

/-- The mnemonics: those of the crate's blocks, and some that the blocks do not use, which the
policy classifies too. -/
inductive Mnemonic
  | mov | mul | umulh | madd | msub | mneg | lsl | lsr | asr | sbfx | extr
  | adds | adcs | adc | subs | sbcs | add | sub | neg | and | orr | eor
  | tst | cmp | ccmp | csel | cneg | csetm
  | udiv | sdiv | ldr | str | bcond | cbz
  deriving DecidableEq, Repr

/-- The condition codes of a conditional instruction. -/
inductive Cond
  | lo | cc | cs | hs | ne | ge | mi
  deriving DecidableEq, Repr

/-- An operand as written. -/
inductive Operand
  /-- A general-purpose register, by its name in the block. -/
  | reg (name : String)
  /-- The zero register `xzr`. -/
  | zero
  /-- An immediate. -/
  | imm (value : Nat)
  /-- A left shift of the preceding register operand by an immediate (`lsl #k`). -/
  | lsl (amount : Nat)
  /-- A condition code. -/
  | cond (c : Cond)
  /-- A memory operand, a base register and an immediate offset. -/
  | mem (base : String) (offset : Nat)
  deriving DecidableEq, Repr

/-- An instruction: its mnemonic and its operands as written. -/
structure Instr where
  op : Mnemonic
  operands : List Operand
  deriving DecidableEq, Repr

/-- The machine state, as far as the leakage looks at it: the value of each register, and of
the flags, as the pseudo-register `nzcv`. -/
abbrev Regs := String → Nat

/-- The instructions that read the flags. -/
def Mnemonic.readsFlags : Mnemonic → Bool
  | .adc | .adcs | .sbcs | .ccmp | .csel | .cneg | .csetm | .bcond => true
  | _ => false

/-- The instructions whose first operand is a source rather than a destination: those that set
only the flags, a store, and the branches. -/
def Mnemonic.firstIsSource : Mnemonic → Bool
  | .tst | .cmp | .ccmp | .str | .cbz | .bcond => true
  | _ => false

/-- A data source of an instruction. -/
inductive Source
  | reg (name : String)
  | flags
  deriving DecidableEq, Repr

/-- The registers an operand reads: a register, or a memory operand's base. -/
def Operand.regs : Operand → List Source
  | .reg r => [.reg r]
  | .mem base _ => [.reg base]
  | _ => []

/-- The data sources of an instruction: the registers of its operands but a destination, then
the flags if it reads them. -/
def Instr.sources (i : Instr) : List Source :=
  let operands := if i.op.firstIsSource then i.operands else i.operands.drop 1
  operands.flatMap Operand.regs ++ if i.op.readsFlags then [.flags] else []

/-- Whether an instruction is, in the form written, a data-independent-time instruction of the
Arm architecture. `mov` of a register is `ORR` and on the list; `mov` of an immediate is `MOVZ`,
which is not. A division is not on the list; a load or store is on it for its data but not for
its address; a branch's condition is its control flow. -/
def Instr.timingIndependent (i : Instr) : Bool :=
  match i.op, i.operands with
  | .mov, [_, .imm _] => false
  | .udiv, _ | .sdiv, _ | .ldr, _ | .str, _ | .bcond, _ | .cbz, _ => false
  | _, _ => true

/-- The value of a data source in a state. -/
def Source.value (s : Regs) : Source → Nat
  | .reg r => s r
  | .flags => s "nzcv"

/-- What an instruction reveals as it runs. -/
inductive Obs
  /-- The instruction, and nothing of its data. -/
  | instr (i : Instr)
  /-- The instruction and the values of its data sources. -/
  | data (i : Instr) (values : List Nat)
  deriving DecidableEq, Repr

/-- The leakage of an instruction: itself when its timing does not depend on its data, and
otherwise itself with the values of its data sources. -/
def leak (i : Instr) (s : Regs) : Obs :=
  if i.timingIndependent then .instr i else .data i (i.sources.map (Source.value s))

/-- The leakage model of AArch64 over a value semantics `step`, which is left arbitrary. -/
def model (step : Instr → Regs → Regs) : Leakage.LeakModel Instr Regs Obs := ⟨step, leak⟩

/-- The syntactic check: an instruction is oblivious if its timing does not depend on its data,
or if it has no data sources. -/
def Instr.oblivious (i : Instr) : Bool := i.timingIndependent || i.sources.isEmpty

/-- The syntactic check is sound: an instruction that passes it is oblivious in the leakage
model, whatever the value semantics. -/
theorem Instr.oblivious_sound (step : Instr → Regs → Regs) (i : Instr) (h : i.oblivious = true) :
    Leakage.Oblivious (model step) i := by
  intro s s'
  simp only [model, leak]
  unfold Instr.oblivious at h
  by_cases ht : i.timingIndependent = true
  · simp [ht]
  · have hs : i.sources = [] := by simpa [ht] using h
    simp [ht, hs]

/-- A program whose every instruction passes the check is constant-time. -/
theorem constantTime_of_all_oblivious (step : Instr → Regs → Regs) (prog : List Instr)
    (h : prog.all Instr.oblivious = true) : Leakage.ConstantTime (model step) prog :=
  Leakage.trace_eq_of_oblivious _ prog fun i hi =>
    Instr.oblivious_sound step i (List.all_eq_true.1 h i hi)

/-- The policy is not vacuous: a division of a register leaks it. -/
theorem udiv_not_oblivious (step : Instr → Regs → Regs) :
    ¬ Leakage.Oblivious (model step) ⟨.udiv, [.reg "q", .reg "a", .reg "b"]⟩ := by
  intro h
  have := h (fun _ => 0) (fun _ => 1)
  simp [model, leak, Instr.timingIndependent, Instr.sources, Mnemonic.firstIsSource,
    Mnemonic.readsFlags, Operand.regs, Source.value] at this

/-- The policy is not vacuous: a load leaks its address register. -/
theorem ldr_not_oblivious (step : Instr → Regs → Regs) :
    ¬ Leakage.Oblivious (model step) ⟨.ldr, [.reg "x", .mem "base" 8]⟩ := by
  intro h
  have := h (fun _ => 0) (fun _ => 1)
  simp [model, leak, Instr.timingIndependent, Instr.sources, Mnemonic.firstIsSource,
    Mnemonic.readsFlags, Operand.regs, Source.value] at this

end PastaCurves.AArch64
