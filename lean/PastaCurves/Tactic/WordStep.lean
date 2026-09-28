
/-!
# The skeleton step of a block proof

A block theorem is proved from `hr : r = block args`, whose right-hand side unfolds to the
block's chain of `let`s, one per instruction. The generated skeleton walks that chain one
instruction at a time. For each instruction it extracts the instruction's bindings from `hr` as
locals, records each binding's defining equation, makes the locals opaque, and states the bound
that the instruction's result satisfies. `word_step` is that stanza as one tactic, so that a
skeleton line reads

```
word_step t := addw a b using addw_lt a b
```

The equation is stated by the generator rather than read from the `let`, because the generator
states the Montgomery instructions' equations in the `%`/`/` form that `omega` reads, which is
not the form the transcription spells them in.
-/

namespace PastaCurves

open Lean Lean.Parser.Tactic

/-- One binding of a `word_step`: the local `x`; with `:= v`, the equation `e_x : x = v`, proved
by `rfl` from the `let`; with `using p`, the bound `b_x : x < 2^64`, proved by `p` after rewriting
with `e_x`. -/
syntax wordStepItem := ident (" := " term)? (" using " term)?

/-- `word_step x₁ := v₁ using p₁, x₂ := v₂, …` is one instruction's stanza of a generated block
proof, on the hypothesis `hr` that every block theorem introduces: it extracts the named `let`s of
`hr` as locals (`extract_lets -merge +onlyGivenNames`), records each `e_xᵢ : xᵢ = vᵢ := rfl`,
makes the locals opaque (`clear_value`), and proves each `b_xᵢ : xᵢ < 2^64` by `rw [e_xᵢ]; exact
pᵢ`. A binding without `:=` (an instruction's (result, carry) pair, which nothing reads) gets no
equation, and one without `using` (a flag, a call, a value the annotations bound) gets no bound.
With `-clear`, the locals stay transparent, for the proofs whose hypothesis tail would be too
costly to re-check at every step. -/
syntax (name := wordStep) "word_step " optConfig wordStepItem,+ : tactic

/-- Whether the configuration `stx` sets the option `opt` to `false`, as `-opt`. -/
private partial def hasNegConfigItem (stx : Lean.Syntax) (opt : Lean.Name) : Bool :=
  if stx.getKind == ``Lean.Parser.Tactic.negConfigItem then stx[1].getId == opt
  else stx.getArgs.any (hasNegConfigItem · opt)

macro_rules
  | `(tactic| word_step $cfg:optConfig $items:wordStepItem,*) => do
    let clear := !hasNegConfigItem cfg.raw `clear
    let hr := Lean.mkIdent `hr
    let mut names : Array Lean.Ident := #[]
    let mut clearArgs : Array (Lean.TSyntax ``clearValueArg) := #[]
    let mut eqs : Array (Lean.TSyntax `tactic) := #[]
    let mut bounds : Array (Lean.TSyntax `tactic) := #[]
    for item in items.getElems do
      let x : Lean.Ident := ⟨item.raw[0]⟩
      let e := Lean.mkIdent (Lean.Name.mkSimple ("e_" ++ x.getId.toString))
      let b := Lean.mkIdent (Lean.Name.mkSimple ("b_" ++ x.getId.toString))
      names := names.push x
      clearArgs := clearArgs.push (← `(clearValueArg| $x:ident))
      let value := item.raw[1]
      if value.getNumArgs == 2 then
        let v : Lean.Term := ⟨value[1]⟩
        eqs := eqs.push (← `(tactic| have $e:ident : $x = $v := rfl))
      let bound := item.raw[2]
      if bound.getNumArgs == 2 then
        let p : Lean.Term := ⟨bound[1]⟩
        bounds := bounds.push
          (← `(tactic| have $b:ident : $x < 2^64 := by rw [$e:ident]; exact $p))
    let extractConfig ← `(optConfig| -merge +onlyGivenNames)
    let mut tacs : Array (Lean.TSyntax `tactic) := #[]
    tacs := tacs.push
      (← `(tactic| extract_lets $extractConfig:optConfig $names* at $hr:ident))
    tacs := tacs ++ eqs
    if clear then
      tacs := tacs.push (← `(tactic| clear_value $clearArgs*))
    tacs := tacs ++ bounds
    `(tactic| ($tacs;*))

end PastaCurves
