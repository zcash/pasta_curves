# Blueprint of the inversion's proof

A [leanblueprint](https://github.com/PatrickMassot/leanblueprint) blueprint of the Lean proof of
the constant-time inversion, published with the book under `blueprint/` and linked from its
[proof map](../book/src/design/proof-map.md) page. Its nodes are extracted from the Lean by
[LeanArchitect](https://github.com/hanwenzhu/LeanArchitect).

## Where each part comes from

- `lean/PastaCurvesBlueprint.lean` tags the proof's declarations with LeanArchitect's
  `@[blueprint]`, from outside them: the proofs do not import LeanArchitect, and the library is not
  part of `PastaCurves`. A node for a numbered result of the book is labelled by the result's
  anchor (`lemma-6`) and carries the declarations the book names for it under "In Lean:"; any
  other node is labelled by its first declaration.
- LeanArchitect (`lake build PastaCurvesBlueprint:blueprint`) extracts each node with its `\lean`
  names, its `\uses` edges, from the constants its statement and its proof use, and `\leanok`,
  when none of its declarations depends on `sorry`.
- The statements in the tags are macros. `sources.py` expands `\bookquote{k}` into the book's
  statement of the result `k`, up to its proof, with a link to it; `\booktitle{k}` into its title;
  `\leandoc{n}` into the docstring of the declaration `n`; and `\touches{f:i}` into a link to the
  Rust item `i` of `src/asm/f.rs`.
- `src/map.tex` orders the nodes into chapters: sections of the book page (`\bookchapter`) or
  Lean files (`\leanchapter`). `sources.py` writes `src/content.tex` from it, with each node
  inlined, since LeanArchitect inputs them by absolute paths that plasTeX cannot follow without TeX
  installed.
- leanblueprint links each Lean name to doc-gen4's search URL, `{dochome}/find/#doc/{name}`.
  There is no doc-gen4 build: `sources.py` writes a page there (`web/decls/find/index.html`) that
  redirects each name to its line in the source.

Every link is pinned to the commit being built. `sources.py` fails on an unknown book result, Lean
declaration, or Rust item, a missing docstring, a node that `map.tex` omits or repeats, a
declaration under a book result that the book does not name there, or a cycle among the nodes.

## Building

`sources.py` needs Python 3.11 or later. The plasTeX tooling is pinned, with hashes, in
`requirements.txt` (compiled from `requirements.in`); pygraphviz needs graphviz and its headers
(`apt-get install graphviz libgraphviz-dev`, or `brew install graphviz`). From the repository
root:

```sh
python3 -m venv .venv-blueprint
.venv-blueprint/bin/pip install --require-hashes -r blueprint/requirements.txt
PLASTEX=.venv-blueprint/bin/plastex blueprint/build.sh
python3 -m http.server --directory blueprint/web 8000
```

On macOS, set `CFLAGS="-I$(brew --prefix graphviz)/include"` and
`LDFLAGS="-L$(brew --prefix graphviz)/lib"` for the `pip install`.

`build.sh` runs the tests of `sources.py`, LeanArchitect (`SKIP_LAKE=1` reuses its last output),
and `sources.py` with the links pinned to the checked-out commit (or `REF`), then renders the
blueprint into `web/`, failing on any plasTeX warning. Open
`http://localhost:8000/dep_graph_document.html`: the graph is drawn by Graphviz compiled to
WebAssembly, which browsers do not load from a `file://` page. The book's CI runs the same steps
and copies `web/` into the book.
