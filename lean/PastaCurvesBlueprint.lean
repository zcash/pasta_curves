import Architect
import PastaCurves

/-!
# The blueprint of the inversion's proof

LeanArchitect's `@[blueprint]` tags for the declarations of the inversion's proof, applied from
outside them, so the proofs do not import LeanArchitect. A node labelled like `lemma-6` is the
numbered result of `book/src/design/inversion.md` with that anchor, and carries the declarations
the book names for it under "In Lean:"; any other node is labelled by its first declaration.

The statements are macros, which `blueprint/sources.py` expands: `\bookquote{k}` and
`\booktitle{k}` into the book's text for the result `k`, `\leandoc{n}` into the docstring of the
declaration `n`, and `\touches{f:i, ...}` into links to the Rust items `i` of `src/asm/f.rs`.
LeanArchitect infers the edges and the proved status from the declarations themselves.
`blueprint/src/map.tex` sets the order of the nodes; see `blueprint/README.md`.
-/

/-! ## The divstep recurrence -/

attribute [blueprint (statement := /-- \leandoc{PastaCurves.Inversion.divstep} -/)]
  PastaCurves.Inversion.divstep
attribute [blueprint "PastaCurves.Inversion.divstep"] PastaCurves.Inversion.divsteps

attribute [blueprint (statement := /-- \leandoc{PastaCurves.Inversion.M} -/)]
  PastaCurves.Inversion.M
attribute [blueprint "PastaCurves.Inversion.M"] PastaCurves.Inversion.T

attribute [blueprint "lemma-1" (latexEnv := "lemma") (title := /-- \booktitle{lemma-1} -/)
  (statement := /-- \bookquote{lemma-1} -/)] PastaCurves.Inversion.M_spec

attribute [blueprint "lemma-2" (latexEnv := "lemma") (title := /-- \booktitle{lemma-2} -/)
  (statement := /-- \bookquote{lemma-2} -/)] PastaCurves.Inversion.divstep_local
attribute [blueprint "lemma-2" (latexEnv := "lemma")] PastaCurves.Inversion.divsteps_local

attribute [blueprint "lemma-3" (latexEnv := "lemma") (title := /-- \booktitle{lemma-3} -/)
  (statement := /-- \bookquote{lemma-3} -/)] PastaCurves.Inversion.M_rowSum_le
attribute [blueprint "lemma-3" (latexEnv := "lemma")] PastaCurves.Inversion.M_entry_range
attribute [blueprint "lemma-3" (latexEnv := "lemma")] PastaCurves.Inversion.divsteps_abs_le

attribute [blueprint "lemma-4" (latexEnv := "lemma") (title := /-- \booktitle{lemma-4} -/)
  (statement := /-- \bookquote{lemma-4} -/)] PastaCurves.Inversion.divsteps_gcd
attribute [blueprint "lemma-4" (latexEnv := "lemma")] PastaCurves.Inversion.divsteps_f_odd
attribute [blueprint "lemma-4" (latexEnv := "lemma")] PastaCurves.Inversion.f_natAbs_of_g_eq_zero
attribute [blueprint "lemma-4" (latexEnv := "lemma")] PastaCurves.Inversion.divsteps_of_g_zero

attribute [blueprint "lemma-4-prime" (latexEnv := "lemma")
  (title := /-- \booktitle{lemma-4-prime} -/) (statement := /-- \bookquote{lemma-4-prime} -/)]
  PastaCurves.Inversion.M_det
attribute [blueprint "lemma-4-prime" (latexEnv := "lemma")] PastaCurves.Inversion.M_inv_spec
attribute [blueprint "lemma-4-prime" (latexEnv := "lemma")] PastaCurves.Inversion.f_dvd_of_g_eq_zero

attribute [blueprint "theorem-5" (latexEnv := "theorem") (hasProof := true)
  (title := /-- \booktitle{theorem-5} -/)
  (statement := /-- \bookquote{theorem-5} -/)] PastaCurves.Inversion.TerminationBound
attribute [blueprint "theorem-5" (latexEnv := "theorem") (hasProof := true)]
  PastaCurves.Inversion.TerminationBound.ge
attribute [blueprint "theorem-5" (latexEnv := "theorem") (hasProof := true)]
  PastaCurves.Inversion.Hull.terminationBound_of_certified
attribute [blueprint "theorem-5" (latexEnv := "theorem") (hasProof := true)]
  PastaCurves.Inversion.Hull.terminationBound
attribute [blueprint "theorem-5" (latexEnv := "theorem") (hasProof := true)]
  PastaCurves.Inversion.Hull.terminationBound_256

/-! ## Divsteps on packed words -/

attribute [blueprint (statement := /-- \leandoc{PastaCurves.Inversion.packedStart}
  \touches{aarch64:divstep} -/)]
  PastaCurves.Inversion.packedStart
attribute [blueprint "PastaCurves.Inversion.packedStart"] PastaCurves.Inversion.unpack

attribute [blueprint "lemma-6" (latexEnv := "lemma") (title := /-- \booktitle{lemma-6} -/)
  (statement := /-- \bookquote{lemma-6} -/)] PastaCurves.Inversion.divsteps_packedStart
attribute [blueprint "lemma-6" (latexEnv := "lemma")]
  PastaCurves.Inversion.divsteps_packedStart_abs_lt

attribute [blueprint "lemma-6-prime" (latexEnv := "lemma")
  (title := /-- \booktitle{lemma-6-prime} -/) (statement := /-- \bookquote{lemma-6-prime} -/)]
  PastaCurves.Inversion.divsteps_packedStart_g_abs_lt

attribute [blueprint "lemma-7" (latexEnv := "lemma") (title := /-- \booktitle{lemma-7} -/)
  (statement := /-- \bookquote{lemma-7} -/)] PastaCurves.Inversion.unpack_spec

attribute [blueprint (statement := /-- \leandoc{PastaCurves.Inversion.divstep59}
  \touches{aarch64:divstep59} -/)]
  PastaCurves.Inversion.divstep59

-- The book also names `M_entry_range` here; it is on `lemma-3`, since a declaration has one node.
attribute [blueprint "corollary-8" (latexEnv := "corollary")
  (title := /-- \booktitle{corollary-8} -/) (statement := /-- \bookquote{corollary-8} -/)]
  PastaCurves.Inversion.divstep59_spec

/-! ## The round arithmetic -/

attribute [blueprint (statement := /-- \leandoc{PastaCurves.Inversion.updateFG}
  \touches{aarch64:fg_row} -/)]
  PastaCurves.Inversion.updateFG

attribute [blueprint "lemma-9" (latexEnv := "lemma") (title := /-- \booktitle{lemma-9} -/)
  (statement := /-- \bookquote{lemma-9} -/)] PastaCurves.Inversion.updateFG_spec

attribute [blueprint (statement := /-- \leandoc{PastaCurves.Inversion.amontredZ}
  \touches{aarch64:amontred} -/)]
  PastaCurves.Inversion.amontredZ
attribute [blueprint "PastaCurves.Inversion.amontredZ"] PastaCurves.Inversion.amontred

attribute [blueprint "lemma-10" (latexEnv := "lemma") (title := /-- \booktitle{lemma-10} -/)
  (statement := /-- \bookquote{lemma-10} -/)] PastaCurves.Inversion.amontredZ_spec
attribute [blueprint "lemma-10" (latexEnv := "lemma")] PastaCurves.Inversion.amontred_spec

/-! ## The invariant and the result -/

attribute [blueprint (statement := /-- \leandoc{PastaCurves.Inversion.montInvModel} -/)]
  PastaCurves.Inversion.montInvModel
attribute [blueprint "PastaCurves.Inversion.montInvModel"] PastaCurves.Inversion.startV

attribute [blueprint "lemma-11" (latexEnv := "lemma") (title := /-- \booktitle{lemma-11} -/)
  (statement := /-- \bookquote{lemma-11} -/)] PastaCurves.Inversion.rounds_invariant

-- The book also names `signWordOf` and `Signed5.toInt_emod` here. They are left on no node:
-- the model's definition uses the first, and Lemma 11's proof the second, so either one on this
-- node would make Lemma 11 and Theorem 12 use each other.
attribute [blueprint "theorem-12" (latexEnv := "theorem")
  (title := /-- \booktitle{theorem-12} -/) (statement := /-- \bookquote{theorem-12} -/)]
  PastaCurves.Inversion.montInv_spec
attribute [blueprint "theorem-12" (latexEnv := "theorem")] PastaCurves.Inversion.montInv_correct

/-! ## The composition over a backend's blocks -/

attribute [blueprint (statement := /-- \leandoc{PastaCurves.invert}
  \touches{inversion:invert} -/)]
  PastaCurves.invert
attribute [blueprint "PastaCurves.invert"] PastaCurves.InvertBlocks
attribute [blueprint "PastaCurves.invert"] PastaCurves.InvertBlocks.Spec

attribute [blueprint (statement := /-- \leandoc{PastaCurves.invert_eq_model} -/)]
  PastaCurves.invert_eq_model

attribute [blueprint (statement := /-- \leandoc{PastaCurves.invert_entry_spec} -/)]
  PastaCurves.invert_entry_spec

/-! ## The AArch64 blocks -/

attribute [blueprint (statement := /-- \leandoc{PastaCurves.AArch64.divstep59Block}
  \touches{aarch64:divstep59, aarch64:divstep} -/)]
  PastaCurves.AArch64.divstep59Block
attribute [blueprint (statement := /-- \leandoc{PastaCurves.AArch64.divstep59Block_spec} -/)]
  PastaCurves.AArch64.divstep59Block_spec

attribute [blueprint (statement := /-- \leandoc{PastaCurves.AArch64.signMagBlock}
  \touches{aarch64:sign_mag} -/)]
  PastaCurves.AArch64.signMagBlock
attribute [blueprint (statement := /-- \leandoc{PastaCurves.AArch64.signMagBlock_spec} -/)]
  PastaCurves.AArch64.signMagBlock_spec

attribute [blueprint (statement := /-- \leandoc{PastaCurves.AArch64.fgRowBlock}
  \touches{aarch64:fg_row} -/)]
  PastaCurves.AArch64.fgRowBlock
attribute [blueprint (statement := /-- \leandoc{PastaCurves.AArch64.fgRowBlock_spec} -/)]
  PastaCurves.AArch64.fgRowBlock_spec

attribute [blueprint (statement := /-- \leandoc{PastaCurves.AArch64.uvRowBlock}
  \touches{aarch64:uv_row} -/)]
  PastaCurves.AArch64.uvRowBlock
attribute [blueprint (statement := /-- \leandoc{PastaCurves.AArch64.uvRowBlock_spec} -/)]
  PastaCurves.AArch64.uvRowBlock_spec

attribute [blueprint (statement := /-- \leandoc{PastaCurves.AArch64.amontredBlock}
  \touches{aarch64:amontred} -/)]
  PastaCurves.AArch64.amontredBlock
attribute [blueprint (statement := /-- \leandoc{PastaCurves.AArch64.amontredBlock_spec} -/)]
  PastaCurves.AArch64.amontredBlock_spec

attribute [blueprint (statement := /-- \leandoc{PastaCurves.AArch64.condSubBlock}
  \touches{aarch64:cond_sub} -/)]
  PastaCurves.AArch64.condSubBlock
attribute [blueprint (statement := /-- \leandoc{PastaCurves.AArch64.condSubBlock_spec} -/)]
  PastaCurves.AArch64.condSubBlock_spec

attribute [blueprint (statement := /-- \leandoc{PastaCurves.AArch64.invertBlocks} -/)]
  PastaCurves.AArch64.invertBlocks
attribute [blueprint (statement := /-- \leandoc{PastaCurves.AArch64.invertBlocks_spec} -/)]
  PastaCurves.AArch64.invertBlocks_spec

attribute [blueprint (statement := /-- \leandoc{PastaCurves.AArch64.invert_entry_spec}
  \touches{entry:invert} -/)]
  PastaCurves.AArch64.invert_entry_spec
