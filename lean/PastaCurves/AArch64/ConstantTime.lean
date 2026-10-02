import PastaCurves.AArch64.Programs
import PastaCurves.AArch64.Compositions
import PastaCurves.Inversion.Schedule

/-!
# The AArch64 blocks and the inversion are constant-time

In the leakage model of `PastaCurves.Leakage`, under the AArch64 policy of
`PastaCurves.AArch64.Leakage`, and for every value semantics of the instructions:

- each of the ten blocks, as its instruction stream in `Programs.lean`, is constant-time: its
  trace does not depend on the registers and flags it starts from (`mulMont_constantTime` and
  the others). Every instruction is checked by the kernel against the policy, so the theorems
  hold because each instruction of each block is oblivious, and a block is straight-line;
- the inversion is constant-time (`invert_constantTime`): its trace, the traces of its block
  calls in the order the driver makes them, is the same for any two inputs, whatever registers
  each call starts from. The order of the calls is `invertSchedule` for every input
  (`invertM_logged`), and each called block is constant-time.

What this covers, and what it rests on:

- the instruction streams are the blocks' own, generated from `src/asm/aarch64.rs` by the
  transcription's front end and checked by `gen.py --check`;
- the hardware policy is an assumption, with the caveats stated in
  `PastaCurves/AArch64/Leakage.lean`: in particular, Arm guarantees data-independent timing for
  these instructions only with `PSTATE.DIT` set;
- the Rust between the blocks is modelled by `invertM` as source, not as compiled code. The
  theorem says that the source calls the blocks in a fixed order; that the compiled code also
  does, and that the code between the calls (the loop, and the sign word's multiply, add, and
  arithmetic shift) compiles without a data-dependent branch or address, is the compiler's part
  and not proved here. The blocks are `pure`, so the compiler may reorder or merge calls, but only
  by decisions made at compile time; `callsTrace_eq` holds for any list of calls, so any such
  fixed order leaks the same trace for every input;
- the portable Rust blocks, used where the assembly is not compiled, are not modelled.
-/

namespace PastaCurves.AArch64

/-- `mul`'s block is constant-time. -/
theorem mulMont_constantTime (step : Instr → Regs → Regs) :
    Leakage.ConstantTime (model step) mulMontProgram :=
  constantTime_of_all_oblivious step _ (by decide +kernel)

/-- `square`'s block is constant-time. -/
theorem sqrMont_constantTime (step : Instr → Regs → Regs) :
    Leakage.ConstantTime (model step) sqrMontProgram :=
  constantTime_of_all_oblivious step _ (by decide +kernel)

/-- `add`'s block is constant-time. -/
theorem addMod_constantTime (step : Instr → Regs → Regs) :
    Leakage.ConstantTime (model step) addModProgram :=
  constantTime_of_all_oblivious step _ (by decide +kernel)

/-- `sub`'s block is constant-time. -/
theorem subMod_constantTime (step : Instr → Regs → Regs) :
    Leakage.ConstantTime (model step) subModProgram :=
  constantTime_of_all_oblivious step _ (by decide +kernel)

/-- `divstep59`'s block is constant-time. -/
theorem divstep59Block_constantTime (step : Instr → Regs → Regs) :
    Leakage.ConstantTime (model step) divstep59BlockProgram :=
  constantTime_of_all_oblivious step _ (by decide +kernel)

/-- `sign_mag`'s block is constant-time. -/
theorem signMagBlock_constantTime (step : Instr → Regs → Regs) :
    Leakage.ConstantTime (model step) signMagBlockProgram :=
  constantTime_of_all_oblivious step _ (by decide +kernel)

/-- `fg_row`'s block is constant-time. -/
theorem fgRowBlock_constantTime (step : Instr → Regs → Regs) :
    Leakage.ConstantTime (model step) fgRowBlockProgram :=
  constantTime_of_all_oblivious step _ (by decide +kernel)

/-- `uv_row`'s block is constant-time. -/
theorem uvRowBlock_constantTime (step : Instr → Regs → Regs) :
    Leakage.ConstantTime (model step) uvRowBlockProgram :=
  constantTime_of_all_oblivious step _ (by decide +kernel)

/-- `amontred`'s block is constant-time. -/
theorem amontredBlock_constantTime (step : Instr → Regs → Regs) :
    Leakage.ConstantTime (model step) amontredBlockProgram :=
  constantTime_of_all_oblivious step _ (by decide +kernel)

/-- `cond_sub`'s block is constant-time. -/
theorem condSubBlock_constantTime (step : Instr → Regs → Regs) :
    Leakage.ConstantTime (model step) condSubBlockProgram :=
  constantTime_of_all_oblivious step _ (by decide +kernel)

/-- The instruction stream of each of the inversion's blocks. -/
def blockProgram : BlockName → List Instr
  | .divstep59 => divstep59BlockProgram
  | .signMag => signMagBlockProgram
  | .fgRow => fgRowBlockProgram
  | .uvRow => uvRowBlockProgram
  | .amontred => amontredBlockProgram
  | .condSub => condSubBlockProgram

/-- Each of the inversion's blocks is constant-time. -/
theorem blockProgram_constantTime (step : Instr → Regs → Regs) (b : BlockName) :
    Leakage.ConstantTime (model step) (blockProgram b) := by
  cases b
  · exact divstep59Block_constantTime step
  · exact signMagBlock_constantTime step
  · exact fgRowBlock_constantTime step
  · exact uvRowBlock_constantTime step
  · exact amontredBlock_constantTime step
  · exact condSubBlock_constantTime step

/-- The trace of a run of the inversion on `x`, its `k`-th block call starting from the
registers `states k`: the traces of the blocks it calls, in the order it calls them. -/
def invertTrace (step : Instr → Regs → Regs) (x modulus : Limbs) (inv : Nat) (v0 : Limbs)
    (states : Nat → Regs) : List Obs :=
  Leakage.callsTrace (model step) blockProgram
    ((invertM invertBlocks.logged x modulus inv v0).run []).2 states

/-- The inversion is constant-time: two runs, on any inputs and with any registers at each
block call, leak the same trace. -/
theorem invert_constantTime (step : Instr → Regs → Regs) (x x' modulus modulus' : Limbs)
    (inv inv' : Nat) (v0 v0' : Limbs) (states states' : Nat → Regs) :
    invertTrace step x modulus inv v0 states = invertTrace step x' modulus' inv' v0' states' := by
  unfold invertTrace
  rw [invertM_logged_log invertBlocks x x' modulus modulus' inv inv' v0 v0']
  exact Leakage.callsTrace_eq _ _ (blockProgram_constantTime step) _ _ _

/-- The run that `invert_constantTime` is about computes the inversion: logging the calls does
not change the result, which is `invert` over the AArch64 blocks. -/
theorem invertTrace_run_result (x modulus : Limbs) (inv : Nat) (v0 : Limbs) :
    ((invertM invertBlocks.logged x modulus inv v0).run []).1 =
      invert invertBlocks x modulus inv v0 := by
  rw [invertM_logged]

end PastaCurves.AArch64
