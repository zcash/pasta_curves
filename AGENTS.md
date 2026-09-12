# `pasta_curves` — Agent Guidelines

> This file is read by AI coding agents (Claude Code, GitHub Copilot, Cursor, Devin, etc.).
> It provides project context and contribution policies.
>
> For the full contribution guide, see [CONTRIBUTING.md](CONTRIBUTING.md).

This crate provides an implementation of the Pasta elliptic curve cycle, **Pallas**
and **Vesta**, along with their base and scalar fields (`Fp`, `Fq`). The curves form a
cycle — the order of each is exactly the base field of the other — and are highly
2-adic, which makes them well-suited to recursive proof systems and PLONK-style
protocols.

This is low-level cryptographic code. Our priorities are **correctness, constant-time
behavior, and performance** — in that order. Field and curve arithmetic that handles
secret data must be constant-time; use the `subtle` crate's `CtOption` / `Choice` /
`ConditionallySelectable` rather than data-dependent branches. Variable-time code is
permitted only where the inputs are non-secret (e.g. the `glv` module's verifier-side
scalar multiplication), and must be documented as such.

Many downstream projects (halo2 and others) depend on this crate, so we prefer to "do
it right" the first time.

## Contribution Process

This crate follows the [zkcrypto RFC process](https://zkcrypto.github.io/rfcs/). Before
opening a PR for any **substantial** change (new public API, algorithmic changes,
changes to serialization or curve/field semantics), there should be an
[RFC](https://github.com/zkcrypto/rfcs) or a tracking issue with maintainer
acknowledgment. Trivial fixes (typos, doc clarifications, obvious bug fixes) do not need
an RFC.

If no prior discussion exists for a substantial change:

- Prefer opening an issue or RFC first and waiting for maintainer feedback.
- Keep changes focused — avoid unsolicited refactors or broad "improvement" PRs.

### Maintainer Check

If the `gh` CLI is authenticated, an agent can check the user's access level:

```bash
gh api repos/zcash/pasta_curves --jq '.permissions | .admin or .maintain or .push'
```

If this returns `true`, the user has write access and manages their own priorities.

### License & Contribution Terms

The crate is dual-licensed `MIT OR Apache-2.0`. Unless stated otherwise, any contribution
intentionally submitted for inclusion is dual-licensed as above, with no additional
terms. See `README.md`, `LICENSE-MIT`, and `LICENSE-APACHE`.

**Check the license of anything you bring in before building on it.** Before adding a
dependency, vendoring a file, or transcribing, porting, or generating code from someone
else's work, read that artifact's own license: its file header, its `LICENSE`, its manifest's
license field. A statement in a repository's README that all its code is dual-licensed does
not cover files that repository vendored from elsewhere. A port, a transcription, generated
code, or comments quoting a source are derivative works and inherit the source's license.
Anything not compatible with the crate's licensing must be raised with the maintainers before
work starts. For Rust dependencies, `cargo license --avoid-dev-deps` lists what ships.

### AI Disclosure

If AI tools were used in preparing a commit, the contributor MUST include a
`Co-Authored-By:` trailer identifying the AI system. The contributor is the sole
responsible author — "the AI generated it" is not a justification during review.

Example:
```
Co-Authored-By: Claude <noreply@anthropic.com>
```

## Code Conventions

- **Cryptographic constants are derived, not hand-transcribed.** The magic constants that
  appear in the source (curve parameters, GLV short-basis and Babai-rounding constants,
  boundary-scalar test witnesses, ...) are recomputed from the curve definitions by the
  SageMath scripts in `./sage`, which print them in the exact shape of the Rust code so
  the two can be diffed. When you add or change such a constant, add or update the
  corresponding derivation in `./sage` (see `sage/README.md`) rather than inlining an
  unexplained literal.

- **No issue or PR numbers in code comments, except in TODOs.** A comment must stand on
  its own, explaining the code in words. Do not annotate it with an issue/PR reference
  (`see #123`): when the patch already solves the problem the reference is stale noise.
  The exception is a `TODO`, where referencing the tracking issue is helpful and is
  removed when the issue is resolved.

- **Preserve constant-time behavior.** Do not introduce secret-dependent branches, array
  indexing, or early returns into arithmetic that may handle secret scalars or field
  elements. Return `CtOption` for fallible constant-time operations rather than `Option`
  or panicking.

## The assembly backends (`src/asm`)

The `asm` module provides assembly backends for the Pasta field arithmetic. It contains an
AArch64 backend and an x86-64 backend: Montgomery multiplication and squaring as inline `asm!`
blocks, modular addition and subtraction, and a repeated-squaring chain and conversion out of
Montgomery form composed from them. It is the one part of the crate that allows unsafe code. Its
priorities are those of the crate: **correctness, constant-time behaviour, and performance**, in
that order.

The routines are transcriptions of Supranational's Semolina v0.1.4 (see `src/asm/README.md`).
The instruction streams are the object of machine-checked correctness proofs, so a change to
an instruction is a change to a specification: keep the transcription, its documentation, and
the proofs in step, and do not "improve" the assembly in passing.

The module provides a backend on `target_arch = "aarch64"` and, in part, on
`target_arch = "x86_64"`: `add`, `sub`, and `from_mont` on every x86-64 target, and `mul` and
`square` on x86-64 with 64-bit pointers. Elsewhere the `asm` module is absent. Nothing is
assembled at build time, so no C toolchain is needed. On all of those, beside the crate's usual
checks:

```sh
cargo test asm::                # the backend's tests, with the debug assertions they check
cargo test --release asm::      # the same tests on the release code
```

A cfg-gated test that compiles out still reports success, so CI counts the `#[test]`
functions under `src/asm` and requires the run of the module's tests to report exactly that
many passed, in both profiles. CI also builds `core` from source on a nightly toolchain
instead of using the sysroot (`cargo +nightly build --release --no-default-features -Z
build-std=core,compiler_builtins --target aarch64-apple-darwin`), which proves that nothing in
the backend reaches for std.

Documentation is a synthetic cross-platform build: `cfg(doc)` retains APIs that are unavailable
on the rustdoc host, while `doc(cfg(...))` renders their real architecture requirements. Because
`doc(cfg)` is still unstable, docs.rs builds on nightly with `--cfg docsrs`. When adding another
backend, update these conditions and keep doc-only fallback bodies non-executable.

- **Preserve constant-time behaviour in the backend.** No secret-dependent branches or memory
  accesses in the blocks; the repeated-squaring loop branches only on its public count.
  Conditional reductions use `csel` after a full-width subtraction.
- **Operand contracts are stated on the entry points and checked by `debug_assert!`.** A
  change to a contract needs a change to the proofs that establish it.

## The Lean formalization (`lean`)

`lean/` is a Lake package (`PastaAArch64Asm`) that models the backend's routines formally and
contributes to assuring their correctness; see `lean/README.md` for the trust story, the
theorems and their caveats, and how those theorems are proven. Build it from that directory with
the elan-managed `lake` for its `lean-toolchain` (a `lake` of another Lean version corrupts the
shared `.lake` cache):

```sh
cd lean
lake exe cache get          # Mathlib's prebuilt oleans, once
lake build --wfail          # warnings fail the build, as in CI
cd .. && lean/scripts/check.sh   # regenerate the transcription and check the skeletons
```

- **`Transcription.lean` and `Vectors.lean` are generated** by `lean/scripts/gen.py` from the
  `asm!` blocks in `src/asm/aarch64.rs` and the vectors file. Never edit them by hand; change
  the generator or its inputs and regenerate. `Compositions.lean` is hand-written and mirrors
  the Rust of `sqr_n_mul` and `from_mont`; a change to either changes both.
- **In `Spec.lean`, the generated skeleton lines are not edited either.** Only the theorem
  statements and the `-- BEGIN ... -- END` annotation blocks are hand-written; `gen.py
  --check-spec` requires the rest to be the current skeleton. A change to a block regenerates
  the skeleton, and the annotations are then moved to their new places.
- **No `sorry`, no `native_decide`, no new axioms.** The nanoda re-check permits only the three
  standard axioms; concrete facts are checked by `decide +kernel`.

## Build & Test Commands

This is a **single crate** (not a workspace). It follows standard `cargo` practices.

**Important:** CI runs the test suite under two feature configurations — `--all-features`
and `--no-default-features`. Any change should build and pass tests under both, since
much of the crate is feature-gated (`alloc`, `sqrt-table`, `glv`, `serde`, etc.).

```sh
# Check without codegen (fastest iteration)
cargo check --all-features

# Build
cargo build --all-features

# Test (the two configurations CI gates on). Always use --release: this is
# arithmetic-heavy crypto code, and an unoptimized test run is dramatically
# slower — slow enough to paralyze local iteration. CI runs tests in release
# mode too, so this also matches the paths CI exercises.
cargo test --release --all-features
cargo test --release --no-default-features
```

`scripts/ci.sh` runs every check CI runs, these, the assembly backend's, and the formalization's,
in one go; a check whose tool is not installed is skipped with a note on how to install it.

### Toolchain note

`rust-toolchain.toml` pins the MSRV toolchain (currently **1.63.0**), whose old codegen
is markedly slower — often *much* slower — than a current stable, on top of the release
vs. debug gap above. For faster local iteration, run everything on a stable toolchain,
e.g. `cargo +stable test --release --all-features`. CI runs both MSRV (for the required
checks) and beta/stable lint passes.

### Cross-platform / target coverage

CI also verifies:

```sh
# 32-bit target (there have historically been 32-bit-specific bugs, e.g. sqrt)
cargo test --release --target i686-unknown-linux-gnu --all-features

# no_std targets build with default features off
cargo build --target thumbv6m-none-eabi   --no-default-features
cargo build --target wasm32-unknown-unknown --no-default-features
cargo build --target wasm32-wasi           --no-default-features
```

### Benchmarks & book

```sh
# Benchmarks (Criterion). Some require specific features:
cargo bench --all-features                 # all benches
cargo bench --features glv --bench glv     # glv bench requires the glv feature

# CI builds the benches to prevent bitrot:
cargo build --benches --all-features

# The mdBook design docs are tested against the built crate:
cargo build
mdbook test -L target/debug/deps book/
```

## Lint & Format

```sh
# Format (CI runs this with --check)
cargo fmt
cargo fmt -- --check

# Clippy — CI's required (MSRV) run uses -D warnings across all features and targets.
# A separate, non-blocking beta run surfaces newer lints informationally.
cargo clippy --all-features --all-targets -- -D warnings

# Intra-doc link validation (crate declares #![deny(rustdoc::broken_intra_doc_links)])
cargo doc --all-features --document-private-items
```

## Feature Flags

Defined in `Cargo.toml`; `default = ["bits", "sqrt-table"]`.

- `alloc` — enables heap-allocating functionality (`arithmetic::{CurveAffine, CurveExt}`,
  hash-to-curve, etc.); pulls in `blake2b_simd` and `group/alloc`. Many other features
  imply it.
- `bits` — enables the `ff/bits` integration (`PrimeFieldBits`). *(default)*
- `sqrt-table` — large precomputed tables (on the heap) that speed up square roots;
  implies `alloc`, pulls in `lazy_static`. *(default)*
- `glv` — variable-time GLV scalar multiplication for non-secret scalars (implies `alloc`).
- `deferred` — the `deferred` module: `DeferredField` and a wide `Product` accumulator
  for batching many field multiplications behind a single Montgomery reduction.
- `gpu` — exposes `ec-gpu`'s `GpuField` for `Fp`/`Fq` (implies `alloc`).
- `serde` — canonical byte-encoding Serde support (hex when the format is human-readable);
  pulls in `serde` and `hex`.
- `repr-c` — adds `repr(C)` to point structures to ease FFI use.
- `uninline-portable` — disables inlining of some functions; useful on tiny targets
  (e.g. Cortex-M0) where inlining hurts code size / performance.

The crate is `#![no_std]`; the `std`-using bits are limited to `#[cfg(test)]` and
dev-dependencies. The `docsrs` **cfg** flag (not a Cargo feature) is used only to gate
`doc(cfg(...))` annotations for docs.rs.

## Code Style

Standard Rust naming and formatting are enforced by `rustfmt` and `clippy`. Project
specifics:

### Fallibility & constant-time

- Fallible constant-time operations return `subtle::CtOption`, not `Option`/`Result`,
  and must not branch on secret data. Use `Choice` / `ConditionallySelectable` /
  `ConstantTimeEq` for selection and comparison.
- Panicking/aborting is avoided except where provably unreachable.

### Type Safety & Representation

- Field and point types (`Fp`, `Fq`, curve points) are opaque structs. `Fp`/`Fq` are
  `repr(transparent)` to enable FFI use while remaining opaque in Rust; do not expose
  their internal limb representation.
- Prefer immutability; use `mut` only where needed for performance.
- Keep the public surface intentional: `pub` items are part of the public API. Use
  `pub(crate)` for internal sharing, and gate test-only helpers behind `#[cfg(test)]`.

### Documentation

- All public API items MUST have `rustdoc` (`///`) comments; the crate uses
  `#![deny(missing_docs)]`. Crate-level docs use `//!` in `lib.rs`.
- The crate declares `#![deny(rustdoc::broken_intra_doc_links)]` — keep intra-doc links
  valid.
- KaTeX math in docs is supported via `katex-header.html` (see the `docs.rs` metadata in
  `Cargo.toml`).

### Serialization

- The `serde` feature serializes field elements and points to their **canonical byte
  encoding** (hex when the data format is human-readable). Serialization is
  security-critical: canonical encodings and their round-trip behavior must not change
  after release without a SemVer-appropriate version bump.

### Testing

- `proptest` is used (as a dev-dependency) for property tests, especially around parsing,
  arithmetic identities, and GLV decomposition; prefer it for rigorous coverage of new
  arithmetic. Strategies and property tests live inline in the relevant module's
  `#[cfg(test)]`.
- Watch for platform-specific behavior: keep tests passing on 32-bit and `no_std`
  targets, not just the host.

## Architecture

- **`no_std` by default.** `lib.rs` is `#![no_std]` with `extern crate alloc` behind the
  `alloc` feature; `std` appears only in tests.
- **Feature-gated modules.** `arithmetic`, `pallas`, `vesta` are always available;
  `deferred`, `glv`, hash-to-curve (`hashtocurve`), and `serde_impl` are compiled only
  when their features are enabled.
- **Macro-generated field/curve code.** `src/macros.rs` and the `fields/` and
  `arithmetic/` submodules generate the repetitive per-field and per-curve
  implementations; changes to arithmetic usually belong in the macro/shared code so both
  curves stay in sync.
- **Sage-derived constants.** Magic constants have companion derivations under `./sage`
  (see the Code Conventions section).

## Branching, SemVer & Releases

- **Merge-based workflow.** CI status checks run on trial-merges of PRs.
- **MSRV policy differs from some sibling crates:** per `README.md`, a change to the
  Minimum Supported Rust Version is shipped with a **minor** version bump, and is *not*
  treated as a breaking change. (This is the opposite of the librustzcash convention —
  do not carry that assumption over.)
- Otherwise follow Rust-flavored SemVer for the public API.
- New features and non-fix changes branch from `main`.

## Changelog & Commit Discipline

- `CHANGELOG.md` follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
  Record any public API change, bug fix, or user-visible semantic change under the
  `## [Unreleased]` section, in the same commit that makes the change.
- Entries describe only what a **user of the public API** needs to adapt to — not
  implementation details.
- **Never edit an entry under an already-released version heading** (`## [x.y.z] - DATE`).
  Those are the historical record of what shipped; new information goes under
  `[Unreleased]`.
- Commits should be discrete semantic changes (no WIP commits in final PR history). Use
  `git revise` / interactive rebase to keep PR history clean. A commit that alters public
  API updates its docs and changelog in the same commit.
- Commit messages: short title, body explaining the motivation for the change.

## CI Checks (all must pass)

The required aggregate check gates on: `test`, `test-32-bit`, `no-std`, and `bitrot`.
The individual jobs are:

- **`test`** — `cargo test --release` with `--all-features` and `--no-default-features`,
  on Ubuntu, Windows, and macOS; verifies the working directory is clean afterward.
- **`test-32-bit`** — the same feature matrix on `i686-unknown-linux-gnu`.
- **`no-std`** — builds `--no-default-features` for `thumbv6m-none-eabi`,
  `wasm32-unknown-unknown`, and `wasm32-wasi`.
- **`bitrot`** — builds the benchmarks (`--benches --all-features`).
- **`book`** — `mdbook test` against the built crate.
- **`doc-links`** — `cargo doc --all-features --document-private-items` (intra-doc links).
- **`fmt`** — `cargo fmt -- --check`.
- **Clippy** — `--all-features --all-targets -- -D warnings` on the MSRV toolchain
  (required, runs on PRs); a beta-toolchain run is informational only.
- **`codecov`** — coverage via `tarpaulin` (not a required check; depends on an external
  service).
