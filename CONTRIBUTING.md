# Contributing to `pasta_curves`

First off, thanks for taking the time to contribute! ❤️

All types of contributions are encouraged and valued. See the [Table of
Contents](#table-of-contents) for different ways to help and details about how
this project handles them. Please make sure to read the relevant section before
making your contribution. It will make it a lot easier for us maintainers and
smooth out the experience for all involved. The community looks forward to your
contributions. 🎉

> And if you like the project, but just don't have time to contribute, that's fine. There are other easy ways to support the project and show your appreciation, which we would also be very happy about:
> - Star the project on GitHub.
> - Post about the project.
> - Refer this project in your project's readme.
> - Mention the project at local meetups and tell your friends/colleagues.


## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [I Have a Question](#i-have-a-question)
- [I Want To Contribute](#i-want-to-contribute)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)
- [Styleguides](#styleguides)
- [Git Usage](#git-usage)
- [Coding Style](#coding-style)

## Code of Conduct

This project and everyone participating in it is governed by the
[Code of Conduct](https://github.com/zcash/zcash/blob/master/code_of_conduct.md). By
participating, you are expected to uphold this code. Please report unacceptable
behavior as documented in the code of conduct.


## I Have a Question

> If you want to ask a question, we assume that you have read the available documentation. API documentation for the crate is published to [docs.rs](https://docs.rs/pasta_curves), and design notes are published as the [Pasta book](https://zcash.github.io/pasta_curves/) and in the `book/` directory of this repository.

Before you ask a question, it is best to search for existing [Issues](/issues)
that might help you. In case you have found a suitable issue and still need
clarification, you can write your question in this issue. It is also advisable
to search the internet for answers first.

If you then still feel the need to ask a question and need clarification, we
recommend the following:

- Ask for help in the `#libraries` channel of the [Zcash R&D Discord](https://discordapp.com/channels/809218587167293450/876655911790321684).
  There are no bad questions, only insufficiently documented answers. If you're
  able to find an answer and it wasn't already in the docs, consider opening a
  pull request to add it to the documentation!
- You can also open an [Issue](/issues/new). If you do so:
  - Provide as much context as you can about what you're running into.
  - Provide project and platform versions depending on what seems relevant.

We will then attempt to triage the issue as soon as practical. Please be aware
that the maintainers of `pasta_curves` have a relatively heavy workload, so
this may take some time.


## I Want To Contribute

> ### Legal Notice
> When contributing to this project, you must agree that you have authored 100% of the content, that you have the necessary rights to the content and that the content you contribute may be provided under the project licenses.

### The RFC Process

`pasta_curves` follows the [zkcrypto RFC process](https://zkcrypto.github.io/rfcs/).
If you want to propose "substantial" changes to this crate — new public API,
algorithmic changes, or changes to serialization or curve/field semantics — please
[create an RFC](https://github.com/zkcrypto/rfcs) for wider discussion before
investing significant implementation effort. Smaller, well-scoped changes and bug
fixes can be proposed directly as issues and pull requests.

### Project Structure

`pasta_curves` is a single Rust crate that implements the Pasta elliptic curve
cycle (Pallas and Vesta) and their base and scalar fields (`Fp`, `Fq`). Please
refer to the [README](README.md) for an overview, and the published
[documentation](https://docs.rs/pasta_curves) for the API. The crate is
`#![no_std]`, with heap-allocating and optional functionality gated behind Cargo
feature flags (see `AGENTS.md` and `Cargo.toml` for the feature list).

This is low-level cryptographic code with many downstream dependents (such as
halo2). Correctness and constant-time behavior on secret data take priority over
convenience.

### Project Versioning

This crate follows [Semantic Versioning](https://semver.org/). If possible, it is
desirable for users to depend upon the latest released version. A detailed change
log is available in the [`CHANGELOG.md`](CHANGELOG.md) file.

Note that, unlike some related crates, a change to the **Minimum Supported Rust
Version (MSRV)** is released as a *minor* version bump and is **not** treated as a
SemVer-breaking change. This policy is stated in the [README](README.md).

### Reporting Bugs

#### Before Submitting a Bug Report

A good bug report shouldn't leave others needing to chase you up for more
information. Therefore, we ask you to investigate carefully, collect
information and describe the issue in detail in your report. Please complete
the following steps in advance to help us fix any potential bug as fast as
possible.

- Determine if your bug is really a bug and not an error on your side e.g.
  using incompatible environment components/versions or violating the
  documented preconditions for an operation.
- To see if other users have experienced (and potentially already solved) the
  same issue you are having, check if there is not already a bug report
  existing for your bug or error in the [bug tracker](issues?q=label%3Abug).
- Also make sure to search the internet to see if users outside of the GitHub
  community have discussed the issue. You can also ask about your problem in
  the [Zcash R&D Discord](https://discordapp.com/channels/809218587167293450/876655911790321684).
- Collect information about the problem:
  - OS, Platform and Version (Windows, Linux, macOS, x86, ARM). Note that this
    crate targets a range of platforms including 32-bit and `no_std` targets, so
    the target triple is often relevant.
  - Version of the compiler, and the crate feature flags you have enabled.
  - Your inputs and the resulting output, if revealing these values does not
    impact your privacy.
  - Can you reliably reproduce the issue? And can you also reproduce it with
    older versions?


#### How Do I Submit a Good Bug Report?

> You must never report security related issues, vulnerabilities or bugs including sensitive information to the issue tracker, or elsewhere in public. Issues that have implications for personal or network security should be reported as described at [https://z.cash/support/security/](https://z.cash/support/security/).


We use GitHub issues to track bugs and errors. If you run into an issue with
the project:

- Open an [Issue](/issues/new). (Since we can't be sure at this point whether
  the issue describes a bug or not, we ask you not to label the issue.)
- Explain the behavior you would expect and the actual behavior.
- Please provide as much context as possible and describe the **reproduction
  steps** that someone else can follow to recreate the issue on their own. This
  usually includes your code. For good bug reports you should isolate the
  problem and create a reduced test case.
- Provide the information you collected in the previous section.

Once it's filed:

- The project team will label the issue accordingly.
- Unless the issue is naturally hard to reproduce, a team member will try to
  reproduce the issue with your provided steps. If there are no reproduction
  steps or no obvious way to reproduce the issue, the team will ask you for
  those steps and mark the issue as `needs-repro`. Bugs with the `needs-repro`
  tag will not be addressed until they are reproduced.
- If the team is able to reproduce the issue, it will be assigned an
  appropriate category and fixed according to the criticality of the issue. If
  you're able to contribute a proposed fix, this will likely speed up the
  process, although be aware that fixes to cryptographic code will be
  considered in the context of correctness, constant-time behavior, and
  potential for unintentional misuse of the overall API; you should be prepared
  to alter your approach based on suggestions from the team and for your
  contributions to undergo multiple rounds of review.


### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion,
**including completely new features and minor improvements to existing
functionality**. Following these guidelines will help maintainers and the
community to understand your suggestion and find related suggestions.


#### Before Submitting an Enhancement

- Read the documentation of the latest version of the crate to find out if the
  functionality is already provided, potentially under a feature flag.
- Perform a [search](/issues) to see if the enhancement has already been
  suggested. If it has, add a comment to the existing issue instead of opening
  a new one.
- For substantial changes, consider whether the [zkcrypto RFC
  process](https://zkcrypto.github.io/rfcs/) is the more appropriate venue (see
  [The RFC Process](#the-rfc-process) above).
- Find out whether your idea fits with the scope and aims of the project. It's
  up to you to make a strong case to convince the project's developers of the
  merits of this feature. Keep in mind that this is a focused, low-level
  cryptographic library; functionality that is specific to a particular
  higher-level protocol usually belongs in a downstream crate rather than here.
- Note that, due to the practice of "airdrop farming", this project DOES NOT
  accept trivial PRs (spelling corrections, link fixes, minor style
  modifications, etc.) from unknown contributors. We appreciate problems of
  this sort being reported as issues, though.


#### How Do I Submit a Good Enhancement Suggestion?

Enhancement suggestions are tracked as [GitHub issues](/issues).

- Use a **clear and descriptive title** for the issue to identify the
  suggestion.
- Provide a **step-by-step description of the suggested enhancement** in as
  many details as possible.
- **Describe the current behavior** and **explain which behavior you expected
  to see instead** and why. At this point you can also tell which alternatives
  do not work for you.
- **Explain why this enhancement would be useful** to most users. You may also
  want to point out the other projects that solved the problem and which could
  serve as inspiration.


## Styleguides

### Git Usage

Individual releases are tagged with their version, e.g. `0.5.2`. Each tag points
to the Git commit at which that version was published (the commit that
incremented the crate's version).

#### Merge Workflow

This project uses a merge-based workflow.

We have a strong preference for preserving commit history. PRs are generally
merged to their target branch with merge commits. We do not use the
"rebase-merge" option in GitHub. We will avoid using the "squash-merge" option
in GitHub except on a case-by-case basis for PRs that do not have clean commit
histories.

New features and other changes should branch from `main`. If a change is a
SemVer-compatible bug fix that would be valuable to release against an
already-published version, note the relevant release tag in the top message of
the pull request; the maintainers may use this to prepare a point release.

If the contents of the target branch for a PR changes in a way that creates a
merge conflict in a PR (either explicit such that GitHub detects it and
prevents PR merging, or implicit such that CI detects it via test failures when
testing the merged state), the author should rebase the PR on top of the latest
state of the target branch, updating each commit as necessary to address the
conflicts.

In order to keep larger changes to a manageable size for review, we use Stacked PRs:

Each PR after the first branches from, and targets, the branch of the "parent"
PR. When an earlier PR changes, each subsequent PR's branch is rebased in
sequence on its "parent" PR's branch. We do not currently use specific tooling
to aid with PR stacking.

#### Branch History

- Commits should represent discrete semantic changes.
- We have a strong preference for a clean commit history. We will actively
  rebase PRs to squash changes (such as bugfixes or responses to review
  comments) into the relevant earlier commits on the PR branch. We recommend
  the use of the `git revise` tool to help maintain such a clean history within
  the context of a single PR.
- When a commit alters the public API, fixes a bug, or changes the underlying
  semantics of existing code, the commit MUST also modify the `CHANGELOG.md`
  file to clearly document the change.
- Updated or added members of the public API MUST include complete `rustdoc`
  documentation comments.
- It is acceptable and desirable to open pull requests in "Draft" status. Only
  once the pull request has passed CI checks should it be transitioned to
  "Ready For Review".
- There MUST NOT be "work in progress" commits as part of your history, with
  the following exceptions:
  - When making a change to a public API or a core semantic change, it is
    acceptable to make the essential change as a distinct commit, without the
    associated alterations that propagate the semantic change throughout the
    rest of the codebase. In such cases the commit message must CLEARLY DOCUMENT
    the partial nature of the work, and whether the commit is expected compile
    and/or for tests to pass, and what work remains to be done to complete the
    change.
  - If a pull request is fixing a bug, the bug SHOULD be demonstrated by the
    addition of a failing unit test in a distinct commit that precedes the
    commit(s) that fix the bug. Due to the complexity of creating some tests,
    additions or other changes to the test framework may be required. Please
    consult with the maintainers if substantial changes of this sort are
    needed, or if you are having difficulties reproducing the bug in a test.

#### Pull Request Review

Our rebase-heavy workflow for in-progress PRs can interact poorly with PR
review, because GitHub prevents reviewers from adding review comments to a
pre-rebase PR state and forces them to refresh their webpage (losing review
state).

To get around this GitHub UI limitation, the general process we follow is:

- Before a PR gets any review, PR authors rebase whenever they want.
- If anyone does not want the PR to be rebased (e.g. because they are actively
  reviewing it or because rebasing would make future reviews more difficult),
  they add the `S-please-do-not-rebase` label.
- While the PR author sees this label or while they know someone is reviewing
  the PR, they avoid rebasing or force-pushing.
- The PR author adjusts the branch as necessary to address any comments. They
  may always add new commits. If `S-please-do-not-rebase` is not present then
  they can also force-push or rebase previous commits. In any case they push
  the result to the branch.
- In cases where it is likely to aid reviewers, the PR author also posts a
  comment to the PR with a diff link between the previous branch tip and the
  new branch tip. When submitting a review for a PR, reviewers note the commit
  up to which the review covers; this aids PR authors in constructing these
  diff links.
- The PR author should mark each review comment that their update addresses as
  resolved using the GitHub UI. Reviewers will un-resolve comment threads to
  reopen them if they consider there to be a problem with the resolution.
- If the author would like to rebase the branch but `S-please-do-not-rebase` is
  present, they should ask the reviewer(s) through an external channel whether
  rebasing is okay. If everyone is agreed that it is no longer needed, they
  remove the label.
- PR authors try to separate target branch rebases from addressing comments. If
  a rebase is needed to fix a merge conflict, that rebase is performed and
  force-pushed first (and a comment created with the corresponding diff link).
  After that, the necessary commit alterations are made to address review
  comments, followed by a second force-push (with a separate diff link).
- If for whatever reason a particular PR becomes "too large" (for example, due
  to there not being a good way to split the contents down into stacked PRs),
  and significant review has started, then older commits in the PR will
  generally ossify. In that case we will add `S-please-do-not-rebase`
  permanently, and avoid rebasing the PR from then on. We will switch to
  merging the target branch (e.g. main) into the PR branch for merge conflict
  resolution, and commit changes in response to PR review as separate commits
  rather than updating the ossified earlier ones. Recent commits might still be
  okay to amend via force-push if they have not been reviewed yet, but if a PR
  is in this state then we generally tend to just eat the cost of the
  lower-value "addressed review comments" commits. This is a generally
  undesirable state for "leaf-level" change PRs, and we avoid it where
  possible.

If a PR author is non-responsive to review comments, the crate maintainers will
generally make the necessary changes to the PR ourselves. For PRs created from
user forks we can generally do this in the same PR. PRs from an organization
forks do not allow changes from maintainers (due to missing cross-organization
permissions); in this case (or if a user's PR has "allow maintainers to edit"
disabled), we will close the PR and open a new PR containing the commits from
the old PR.

#### Commit Messages

- Commit messages should have a short (preferably less than ~120 characters) title.
- The body of each commit message should include the motivation for the change,
  although for some simple cases (such as the application of suggested changes) this
  may be elided.
- When a commit has multiple authors, please add `Co-Authored-By:` metadata to
  the commit message to include everyone who is responsible for the contents of
  the commit; this is important for determining who has the most complete
  understanding of the changes. If AI tools were used in preparing a commit,
  the AI system MUST be identified via `Co-Authored-By:` metadata, and the
  contributor remains the sole responsible author of the change.
- When changes are requested in pull request review, it is desirable to apply
  those changes to the affected commit in order to avoid excessive noise in the
  commit history. The [git revise](https://github.com/mystor/git-revise) tool is
  **extremely** useful for this purpose. If a maintainer or other user uses the
  GitHub `suggestion` feature to suggest explicit code changes, it's usually
  best to accept those changes via the "Apply Suggested Changes" GitHub
  workflow, and then to amend the resulting commit to fix any related
  compilation, test, or lint errors; this ensures that correct co-author
  metadata is included in the commit.

### Coding Style

The `pasta_curves` authors hold our software to a high standard of quality. The
list of style requirements below is not comprehensive, but violation of any of
the following guidelines is likely to cause your pull request to be rejected or
changes to be required. The coding style in this repository has evolved over
time, and not all preexisting code follows this style; when modifications are
being made to existing code, it should be upgraded to reflect the recommended
style (although please ensure that you separate functional changes from
style-oriented refactoring in the Git commit history.)

Standard Rust naming and formatting are enforced by `rustfmt` and `clippy`;
see `AGENTS.md` for the exact build, test, lint, and CI commands.

#### Constant-Time Behavior

This is cryptographic code. Operations that may act on secret field elements or
scalars MUST be constant-time: they must not branch on, index by, or otherwise
leak secret data through timing.

- Use the [`subtle`](https://docs.rs/subtle) crate's `Choice`,
  `ConditionallySelectable`, and `ConstantTimeEq` for selection and comparison
  rather than native `if`/`==` on secret values.
- Fallible constant-time operations return `subtle::CtOption` rather than
  `Option`, `Result`, or a panic.
- Variable-time code is permitted only where the inputs are guaranteed to be
  non-secret (for example, the `glv` module's verifier-side scalar
  multiplication). Such code MUST be clearly documented as variable-time.

#### Type Safety

Type safety is of paramount importance. This has numerous implications,
including but not limited to the following:

- Invalid states should be made unrepresentable at the type level. `structs`
  should generally keep their internal members private or crate-private, and
  expose constructors and accessors that maintain invariants. The field and
  point types (`Fp`, `Fq`, and the curve points) are opaque structs; their
  internal limb representation is not part of the public API even where the
  types are `repr(transparent)` for FFI purposes.
- Use `enum`s liberally, and prefer custom `enum`s with semantically relevant
  variants over boolean arguments and return values.
- Avoid platform-specific integer sizing (i.e. `usize`) except when indexing
  into a Rust collection type that already requires such semantics. Be
  especially careful with integer widths and shifts, since this crate is tested
  on 32-bit as well as 64-bit targets.
- Prefer immutability; make data types immutable unless there is a strong
  reason to believe that values will need to be modified in-place for
  performance reasons.

#### Public API

The public API of `pasta_curves` is carefully curated. We rely on several
conventions to maintain the legibility of what is public in the API when
reviewing code:

- Any type or function annotated `pub` MUST be part of the public API; we do
  not permit publicly visible types in private modules (with the exception of
  those necessary for representing the "sealed trait" pattern, which we use
  when we want to prohibit third-party implementations of traits we define).
- Optional functionality is gated behind Cargo feature flags. When adding a
  feature-gated public item, annotate it with `#[cfg_attr(docsrs, doc(cfg(...)))]`
  so that its feature requirement is rendered in the docs.rs documentation.
- All public API items MUST carry complete `rustdoc` comments; the crate is
  built with `#![deny(missing_docs)]` and
  `#![deny(rustdoc::broken_intra_doc_links)]`.

#### Cryptographic Constants

The magic constants that appear in the source (curve parameters, GLV
short-basis and Babai-rounding constants, boundary-scalar test witnesses, and
so on) are recomputed from the curve definitions by the SageMath scripts in
the [`sage/`](sage) directory, which print them in the exact shape of the Rust
code so the two can be diffed. When you add or change such a constant, add or
update the corresponding derivation in `sage/` (see [`sage/README.md`](sage/README.md))
rather than inlining an unexplained literal.

#### Serialization

Serialization formats, and serialized data, must be treated with the utmost
care, as serialized data imposes an essentially permanent compatibility burden.

- Field elements and points have a canonical byte encoding, exposed through
  `ff::PrimeField` / `group::GroupEncoding` and (under the `serde` feature) via
  Serde, which serializes to that canonical encoding (as hexadecimal when the
  data format is human-readable).
- These canonical encodings, and their round-trip behavior, are
  serialization-critical: they must not change after a public release without a
  correspondingly appropriate version bump.

## Attribution
This guide is based on the template supplied by the
[CONTRIBUTING.md](https://contributing.md/) project.
