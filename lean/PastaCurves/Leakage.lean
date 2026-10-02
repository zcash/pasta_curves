import Mathlib.Data.List.Basic

/-!
# Leakage traces and constant time

The constant-time model of the formalization. A program is a list of instructions run on a
state; each instruction, as it runs, emits an observation, which is what an attacker who times
the execution, or watches its control flow and its memory addresses, learns from that
instruction. The observations of a run form its trace. A program is constant-time when its trace
does not depend on the state it starts from: every run, whatever its secret inputs, looks the
same.

This is the observational model of Barthe, Betarte, Campo, Luna, and Pichardie, "System-level
non-interference for constant-time cryptography" (CCS 2014), and of Almeida, Barbosa, Barthe,
Dupressoir, and Emmi, "Verifying constant-time implementations" (USENIX Security 2016): a
non-interference property of a leakage semantics. The architecture supplies what each instruction
leaks (`LeakModel.leak`); what it computes (`LeakModel.step`) is left arbitrary, so that the
theorems here hold for every value semantics.

The central fact is `trace_eq_of_oblivious`: if each instruction's observation is independent of
the state (the instruction is `Oblivious`), the trace of any program made of such instructions
is independent of the state, whatever the instructions compute. A straight-line program has no
control flow to leak, so its constant-timeness reduces to that of its instructions one by one.
-/

namespace PastaCurves.Leakage

/-- A leakage semantics over instructions `I`, states `S`, and observations `O`: what an
instruction does to the state, and what it reveals as it runs on a state. -/
structure LeakModel (I S O : Type) where
  /-- The effect of an instruction on the state. -/
  step : I → S → S
  /-- The observation an instruction emits when it runs on a state. -/
  leak : I → S → O

variable {I S O : Type}

/-- The observations of a run of `prog` from `s`, one per instruction, in order. -/
def trace (m : LeakModel I S O) : List I → S → List O
  | [], _ => []
  | i :: prog, s => m.leak i s :: trace m prog (m.step i s)

@[simp] theorem trace_nil (m : LeakModel I S O) (s : S) : trace m [] s = [] := rfl

@[simp] theorem trace_cons (m : LeakModel I S O) (i : I) (prog : List I) (s : S) :
    trace m (i :: prog) s = m.leak i s :: trace m prog (m.step i s) := rfl

/-- An instruction is oblivious when its observation does not depend on the state it runs on. -/
def Oblivious (m : LeakModel I S O) (i : I) : Prop := ∀ s s', m.leak i s = m.leak i s'

/-- A program is constant-time when its trace does not depend on the state it starts from. -/
def ConstantTime (m : LeakModel I S O) (prog : List I) : Prop :=
  ∀ s s', trace m prog s = trace m prog s'

/-- A program of oblivious instructions is constant-time, whatever the instructions compute. -/
theorem trace_eq_of_oblivious (m : LeakModel I S O) (prog : List I)
    (h : ∀ i ∈ prog, Oblivious m i) : ConstantTime m prog := by
  induction prog with
  | nil => intro s s'; rfl
  | cons i prog ih =>
    intro s s'
    simp only [trace_cons]
    rw [h i List.mem_cons_self s s',
      ih (fun j hj => h j (List.mem_cons_of_mem i hj)) (m.step i s) (m.step i s')]

/-- The trace of a sequence of calls to the programs named in `calls`, the `k`-th call starting
from the state `states k`. The states are left free: each call starts from whatever the code
between the calls (and the earlier calls) left, and the theorems quantify over it. -/
def callsTrace {B : Type} (m : LeakModel I S O) (program : B → List I) :
    List B → (Nat → S) → List O
  | [], _ => []
  | b :: calls, states => trace m (program b) (states 0) ++
      callsTrace m program calls (fun k => states (k + 1))

/-- A sequence of calls to constant-time programs leaks the same trace from any states. Together
with a schedule of calls that does not depend on the data, this is a composed program's
constant-timeness. -/
theorem callsTrace_eq {B : Type} (m : LeakModel I S O) (program : B → List I)
    (h : ∀ b, ConstantTime m (program b)) (calls : List B) (states states' : Nat → S) :
    callsTrace m program calls states = callsTrace m program calls states' := by
  induction calls generalizing states states' with
  | nil => rfl
  | cons b calls ih =>
    simp only [callsTrace]
    rw [h b (states 0) (states' 0), ih]

end PastaCurves.Leakage
