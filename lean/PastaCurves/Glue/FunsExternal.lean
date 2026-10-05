module
public import Aeneas
public import PastaCurves.Glue.Types
@[expose] public section
open Aeneas Aeneas.Std Result ControlFlow Error

/-!
# The external function that the translated glue calls

Written by hand, not generated. Beside `Types.lean` and `Funs.lean`, Aeneas emits a template that
declares each Rust function that the translation calls but does not translate as an axiom, to be
filled in under this file's name. The glue calls one: `core::hint::black_box`, through the operand
checks of its debug assertions. It is defined here, so the translation does not add any axiom.
-/

/-- `core::hint::black_box`: the identity on its value. In Rust it is a barrier to the optimizer,
which changes how the code is compiled but not what it computes. -/
@[rust_fun "core::hint::black_box"]
def core.hint.black_box {T : Type} (x : T) : Result T := ok x
