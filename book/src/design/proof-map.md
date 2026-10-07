# Proof map

The dependency graph of the Lean proof of the [constant-time inversion](inversion.md):

- [Graph](../blueprint/dep_graph_document.html)
- [Nodes](../blueprint/index.html)

Each node quotes a result of the inversion page, or the docstring of a Lean declaration, and
links to the declarations in `lean/` and to the Rust in `src/asm/`. Its nodes and edges are
extracted from the Lean by LeanArchitect; see `blueprint/README.md`.
