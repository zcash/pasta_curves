#!/usr/bin/env python3
"""Assemble the blueprint's content from LeanArchitect's output, the book, and the sources.

The nodes, their Lean declarations, their `\\uses` edges, and their proved status are extracted
by LeanArchitect from `lean/PastaCurvesBlueprint.lean` (`lake build PastaCurvesBlueprint:blueprint`).
Their statements there are macros, which this script expands into quoted text:

* `\\bookquote{k}`: the statement of the result anchored `k` in `book/src/design/inversion.md`, up
  to its proof, followed by a link to it; `\\booktitle{k}`: its title.
* `\\leandoc{n}`: the docstring of the Lean declaration `n`.
* `\\touches{f:i, ...}`: links to the Rust items `i` of `src/asm/f.rs`.

`blueprint/src/map.tex` orders the nodes into chapters (`\\bookchapter{id}` for a section of the
book page, `\\leanchapter{path}` for a Lean file). The script writes, for one commit:

* `blueprint/src/content.tex`: `map.tex` with each `\\inputleannode{label}` replaced by the node
  LeanArchitect extracted, its macros expanded. LeanArchitect inputs each node from its own file by
  an absolute path, which plasTeX cannot follow without TeX installed, so the nodes are inlined.
* `blueprint/web/decls/find/index.html`: leanblueprint links each `\\lean{name}` to doc-gen4's
  search URL, `{dochome}/find/#doc/{name}`; with `\\dochome{decls}`, this page redirects each name
  to its source line instead, so no doc-gen4 build is needed.

Every link is pinned to the commit. The run fails on an unknown book result, Lean declaration, or
Rust item, a missing docstring, a node that `map.tex` omits or repeats, a declaration under a book
result that the book does not name for it, or a cycle among the nodes.

Run from anywhere; `--ref` defaults to the checked-out commit. Standard library only.
"""

import argparse
import html
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/zcash/pasta_curves"
LEAN_ROOT = ROOT / "lean"
RUST_ROOT = ROOT / "src" / "asm"
BLUEPRINT = ROOT / "blueprint"
BOOK_PAGE = ROOT / "book" / "src" / "design" / "inversion.md"
# The book page as published beside the blueprint, relative to the blueprint's pages.
BOOK_URL = "../design/inversion.html"
MAP = BLUEPRINT / "src" / "map.tex"
ARCHITECT_MODULE = (
    LEAN_ROOT / ".lake" / "build" / "blueprint" / "module" / "PastaCurvesBlueprint.tex"
)
OUT_CONTENT = BLUEPRINT / "src" / "content.tex"
OUT_FIND = BLUEPRINT / "web" / "decls" / "find" / "index.html"

# --- Lean ---------------------------------------------------------------------------------------

KEYWORDS = (
    "theorem",
    "lemma",
    "axiom",
    "def",
    "abbrev",
    "instance",
    "structure",
    "inductive",
    "class",
    "opaque",
)
MODIFIERS = r"(?:(?:private|protected|noncomputable|nonrec|partial|unsafe)\s+)*"
DECL = re.compile(
    r"^(?:@\[[^\]]*\]\s*)?" + MODIFIERS + r"(" + "|".join(KEYWORDS) + r")\s+([^\s:({\[]+)"
)
NAMESPACE = re.compile(r"^namespace\s+(\S+)")
SECTION = re.compile(r"^(?:noncomputable\s+)?section(?:\s+(\S+))?\s*$")
END = re.compile(r"^end(?:\s+(\S+))?\s*$")
STRUCT_FIELD = re.compile(r"^  ([A-Za-z_][\w']*)\s*:")


def strip_comments(text):
    """Replace comments (nested `/- -/` blocks, docstrings, and `--` lines) by spaces, keeping
    the line structure, so that line numbers still refer to the source."""
    out, i, depth, n = [], 0, 0, len(text)
    while i < n:
        if text.startswith("/-", i):
            depth += 1
            out.append("  ")
            i += 2
        elif depth and text.startswith("-/", i):
            depth -= 1
            out.append("  ")
            i += 2
        elif depth:
            out.append("\n" if text[i] == "\n" else " ")
            i += 1
        elif text.startswith("--", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def lean_declarations(root=LEAN_ROOT):
    """Map each declaration's full name to (path relative to the repository, line), from the
    source text: comments removed, namespaces tracked. Structure fields are included."""
    found = {}
    files = sorted((root / "PastaCurves").rglob("*.lean")) + [root / "PastaCurves.lean"]
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        scopes = []  # (kind, name): kind "ns" for a namespace, "sec" for a section
        structure = None
        for lineno, line in enumerate(strip_comments(path.read_text()).splitlines(), 1):
            prefix = ".".join(name for kind, name in scopes if kind == "ns")
            if m := NAMESPACE.match(line):
                scopes.extend(("ns", part) for part in m.group(1).split("."))
                structure = None
            elif m := SECTION.match(line):
                scopes.append(("sec", m.group(1) or ""))
                structure = None
            elif m := END.match(line):
                # `end A.B` closes the two scopes that `namespace A.B` opened; `end` closes one.
                for _ in range(len(m.group(1).split(".")) if m.group(1) else 1):
                    if scopes:
                        scopes.pop()
                structure = None
            elif m := DECL.match(line):
                name = m.group(2).removeprefix("_root_.")
                full = (
                    name if m.group(2).startswith("_root_.") or not prefix else f"{prefix}.{name}"
                )
                found.setdefault(full, (rel, lineno))
                structure = full if m.group(1) in ("structure", "class") else None
            elif structure and (m := STRUCT_FIELD.match(line)):
                found.setdefault(f"{structure}.{m.group(1)}", (rel, lineno))
            elif line and not line[0].isspace():
                structure = None
    return found


def lean_docstring(path, line):
    """The docstring of the declaration at `path:line`, or None: the `/-- ... -/` that ends just
    above it, past any attribute lines."""
    lines = (ROOT / path).read_text().splitlines()
    i = line - 2
    while i >= 0 and lines[i].lstrip().startswith("@["):
        i -= 1
    if i < 0 or not lines[i].rstrip().endswith("-/"):
        return None
    j = i
    while j >= 0 and "/--" not in lines[j]:
        j -= 1
    if j < 0:
        return None
    text = "\n".join(lines[j : i + 1])
    return text[text.index("/--") + 3 : text.rindex("-/")].strip()


# --- Rust ---------------------------------------------------------------------------------------

RUST_ITEM = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const\s+|unsafe\s+)*fn\s+(\w+)")
RUST_MACRO = re.compile(r"^\s*macro_rules!\s+(\w+)")


def rust_items(root=RUST_ROOT):
    """Map `file:item` (file stem, item name) to (path, line) for the functions and macros of
    `src/asm/`. Test modules are skipped, so an item name refers to the code that ships."""
    found = {}
    for path in sorted(root.glob("*.rs")):
        rel = path.relative_to(ROOT).as_posix()
        in_tests = False
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if re.match(r"^\s*mod tests\b", line):
                in_tests = True
            if in_tests:
                continue
            m = RUST_ITEM.match(line) or RUST_MACRO.match(line)
            if m:
                found.setdefault(f"{path.stem}:{m.group(1)}", (rel, lineno))
    return found


# --- The book -----------------------------------------------------------------------------------

# A numbered result of the book page: an anchor, then its bold title.
BOOK_RESULT = re.compile(
    r'^<a id="([a-z0-9-]+)"></a>\*\*((?:Lemma|Theorem|Corollary) [^*]+?)\.?\*\*', re.MULTILINE
)
NEXT_RESULT = re.compile(r'^<a id="[a-z0-9-]+"></a>\*\*|^#+ ', re.MULTILINE)
HEADING = re.compile(r"^#{1,6} (.+)$", re.MULTILINE)
IN_LEAN = "*In Lean:*"


def heading_id(text):
    """The id mdBook gives a heading: lowercase, spaces to hyphens, other punctuation dropped."""
    return re.sub(r"[^\w\- ]", "", text.strip().lower()).replace(" ", "-")


def book_sections(page=BOOK_PAGE):
    """Map each heading's id to its text."""
    return {heading_id(m.group(1)): m.group(1).strip() for m in HEADING.finditer(page.read_text())}


def book_results(page=BOOK_PAGE):
    """Map each numbered result of the book page, by its anchor, to (title, statement, Lean
    names). The statement is the result's text up to its `*Proof.*` or `*In Lean:*`, verbatim;
    the Lean names are the identifiers in backticks after `*In Lean:*`, without the file names."""
    text = page.read_text()
    found = {}
    for m in BOOK_RESULT.finditer(text):
        rest = text[m.end() :]
        nxt = NEXT_RESULT.search(rest)
        body = rest[: nxt.start()] if nxt else rest
        ends = [i for i in (body.find("*Proof.*"), body.find(IN_LEAN)) if i >= 0]
        statement = body[: min(ends)] if ends else body
        statement = statement.strip().removesuffix("\u220e").strip()
        lean_part = body[body.find(IN_LEAN) :] if IN_LEAN in body else ""
        names = [
            name
            for name in re.findall(r"`([^`]+)`", lean_part)
            if not name.endswith(".lean") and "/" not in name
        ]
        found[m.group(1)] = (m.group(2).strip(), statement, names)
    return found


# --- Markdown to LaTeX --------------------------------------------------------------------------

TEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "$": r"\$",
}
MARKDOWN = re.compile(
    r"(?P<display>\$\$.*?\$\$)|(?P<math>\$[^$]+\$)|(?P<code>`[^`]+`)"
    r"|(?P<link>\[[^\]]+\]\([^)]+\))|(?P<bold>\*\*.+?\*\*)|(?P<em>\*[^*\s][^*]*\*)",
    re.DOTALL,
)


def tex_escape(text):
    return "".join(TEX_SPECIAL.get(c, c) for c in text)


def tex_url(url):
    # `#` is TeX's parameter character; plasTeX keeps `\#` as `#` inside \href.
    return url.replace("#", r"\#")


def markdown_to_tex(text):
    """The Markdown of the book page and of Lean docstrings, as LaTeX: math is kept as written
    (the book's math is LaTeX), code becomes `\\texttt`, emphasis and links their LaTeX forms,
    and every other character is escaped. Paragraph breaks are kept."""
    out, pos = [], 0
    for m in MARKDOWN.finditer(text):
        out.append(tex_escape(text[pos : m.start()]))
        s = m.group(0)
        if m.group("display"):
            out.append("\\[" + s[2:-2].strip() + "\\]")
        elif m.group("math"):
            out.append(s)
        elif m.group("code"):
            out.append(r"\texttt{" + tex_escape(s[1:-1]) + "}")
        elif m.group("link"):
            label, target = re.match(r"\[([^\]]+)\]\(([^)]+)\)", s).groups()
            if target.endswith(".md") or ".md#" in target:
                target = "../design/" + target.replace(".md", ".html")
            out.append(r"\href{" + tex_url(target) + "}{" + markdown_to_tex(label) + "}")
        elif m.group("bold"):
            out.append(r"\textbf{" + markdown_to_tex(s[2:-2]) + "}")
        else:
            out.append(r"\emph{" + markdown_to_tex(s[1:-1]) + "}")
        pos = m.end()
    out.append(tex_escape(text[pos:]))
    return "".join(out)


# --- LeanArchitect's output ---------------------------------------------------------------------

NEW_NODE = re.compile(r"\\newleannode\{([^}]*)\}\{\\input\{([^}]*)\}\}")
LEAN_LIST = re.compile(r"\\lean\{([^}]*)\}")
USES = re.compile(r"\\uses\{([^}]*)\}")
LABEL = re.compile(r"\\label\{([^}]*)\}")


def architect_nodes(module=ARCHITECT_MODULE):
    """Map each node's label to its LaTeX, as LeanArchitect wrote it."""
    if not module.is_file():
        raise FileNotFoundError(
            f"{module.relative_to(ROOT)} is missing: run `lake build PastaCurvesBlueprint:blueprint` "
            "in `lean/` first"
        )
    nodes = {}
    for label, target in NEW_NODE.findall(module.read_text()):
        path = Path(target)
        if not path.is_absolute():
            path = module.parent / path
        nodes[label] = Path(f"{path}.tex").read_text()
    return nodes


def node_declarations(text):
    return [name.strip() for m in LEAN_LIST.finditer(text) for name in m.group(1).split(",")]


def node_uses(text):
    return {name.strip() for m in USES.finditer(text) for name in m.group(1).split(",")}


def find_cycle(edges):
    """A cycle in the graph `edges` (label to set of labels), as a list of labels, or None."""
    state = {}

    def visit(label, path):
        if state.get(label) == 1:
            return [*path[path.index(label) :], label]
        if state.get(label) == 2:
            return None
        state[label] = 1
        for target in sorted(edges.get(label, ())):
            cycle = visit(target, [*path, label])
            if cycle:
                return cycle
        state[label] = 2
        return None

    for label in sorted(edges):
        cycle = visit(label, [])
        if cycle:
            return cycle
    return None


# --- Expansion ----------------------------------------------------------------------------------

MACRO = re.compile(r"\\(bookquote|booktitle|leandoc|touches)\{([^}]*)\}")
MAP_LINE = re.compile(r"\\(bookchapter|leanchapter|inputleannode)\{([^}]*)\}")


def url(ref, path, line=None):
    return f"{REPO_URL}/blob/{ref}/{path}" + (f"#L{line}" if line else "")


class Expander:
    """Expands the quoting macros of LeanArchitect's nodes and of `map.tex`, collecting errors."""

    def __init__(self, ref, lean, rust, book, sections):
        self.ref, self.lean, self.rust = ref, lean, rust
        self.book, self.sections = book, sections
        self.errors = []

    def macro(self, m):
        kind, arg = m.group(1), m.group(2).strip()
        if kind in ("bookquote", "booktitle"):
            if arg not in self.book:
                self.errors.append(f"no result anchored {arg} in the book page")
                return ""
            title, statement, _ = self.book[arg]
            if kind == "booktitle":
                return markdown_to_tex(title)
            link = rf"\href{{{tex_url(BOOK_URL + '#' + arg)}}}{{{markdown_to_tex(title)}}}"
            return markdown_to_tex(statement) + "\n\n" + r"\noindent " + link
        if kind == "leandoc":
            doc = lean_docstring(*self.lean[arg]) if arg in self.lean else None
            if doc is None:
                self.errors.append(f"no docstring for the Lean declaration {arg}")
                return ""
            return markdown_to_tex(doc)
        links = []
        for key in (k.strip() for k in arg.split(",")):
            if key not in self.rust:
                self.errors.append(f"no Rust item {key}")
                continue
            path, line = self.rust[key]
            name = tex_escape(key.split(":", 1)[1])
            links.append(rf"\href{{{tex_url(url(self.ref, path, line))}}}{{\texttt{{{name}}}}}")
        return "\n\n" + r"\noindent\textbf{Touches:} " + ", ".join(links)

    def node(self, text):
        """A node's LaTeX with its macros expanded, its duplicate `\\uses` entries dropped, and
        LeanArchitect's `% at <local path>` comments removed."""
        text = re.sub(r"^% at .*\n", "", text, flags=re.MULTILINE)
        text = USES.sub(lambda m: r"\uses{" + ",".join(sorted(node_uses(m.group(0)))) + "}", text)
        return MACRO.sub(self.macro, text)

    def chapter(self, kind, arg):
        if kind == "bookchapter":
            if arg not in self.sections:
                self.errors.append(f"map.tex: no section {arg} in the book page")
                return ""
            heading = self.sections[arg]
            # plasTeX numbers the chapters itself; the book's heading carries its own number.
            title = markdown_to_tex(re.sub(r"^[0-9]+[.] ", "", heading))
            link = rf"\href{{{tex_url(BOOK_URL + '#' + arg)}}}{{{markdown_to_tex(heading)}}}"
            return f"\\chapter{{{title}}}\n\n{link}\n"
        if not (ROOT / arg).is_file():
            self.errors.append(f"map.tex: no file {arg}")
            return ""
        name = tex_escape(arg)
        return f"\\chapter{{\\texttt{{{name}}}}}\n\n\\href{{{url(self.ref, arg)}}}{{\\texttt{{{name}}}}}\n"


def check_book_nodes(nodes, book, lean):
    """Each declaration under a book result's node must be one the book names for that result."""
    errors = []
    for label, text in nodes.items():
        if label not in book:
            continue
        named = {full for full in lean for name in book[label][2] if full.endswith("." + name)}
        for name in node_declarations(text):
            if name not in named:
                errors.append(f"{name} is on the node {label}, but the book does not name it there")
    return errors


def assemble(ref):
    """`content.tex` and the errors found assembling it."""
    lean, rust = lean_declarations(), rust_items()
    book, sections = book_results(), book_sections()
    nodes = architect_nodes()
    expander = Expander(ref, lean, rust, book, sections)
    errors = check_book_nodes(nodes, book, lean)

    placed, out = (
        [],
        [
            "% GENERATED by blueprint/sources.py from blueprint/src/map.tex and LeanArchitect's output,",
            f"% with links pinned to {REPO_URL}/tree/{ref}; do not edit by hand.",
            "",
        ],
    )
    for line in MAP.read_text().splitlines():
        if line.lstrip().startswith("%") or not line.strip():
            continue
        m = MAP_LINE.fullmatch(line.strip())
        if not m:
            errors.append(f"map.tex: unexpected line {line.strip()!r}")
        elif m.group(1) == "inputleannode":
            label = m.group(2)
            if label not in nodes:
                errors.append(f"map.tex: LeanArchitect extracted no node {label}")
                continue
            placed.append(label)
            out += [expander.node(nodes[label]), ""]
        else:
            out.append(expander.chapter(m.group(1), m.group(2)))
    errors += [
        f"map.tex: the node {lb} appears twice"
        for lb in sorted(set(placed))
        if placed.count(lb) > 1
    ]
    errors += [f"map.tex: the node {lb} is missing" for lb in sorted(set(nodes) - set(placed))]

    edges = {}
    for text in nodes.values():
        labels = LABEL.findall(text)
        if labels:
            edges.setdefault(labels[0], set()).update(node_uses(text) - {labels[0]})
    cycle = find_cycle(edges)
    if cycle:
        errors.append("the nodes' uses form a cycle: " + " -> ".join(cycle))
    return "\n".join(out), errors + expander.errors


def render_find(ref, lean):
    table = {name: url(ref, path, line) for name, (path, line) in sorted(lean.items())}
    data = json.dumps(table, separators=(",", ":"), sort_keys=True)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lean declaration</title>
</head>
<body>
<p id="msg">Looking up the declaration...</p>
<script>
// Generated by blueprint/sources.py: redirects `#doc/<name>`, the doc-gen4 search URL that
// leanblueprint links to, to the declaration's source at {html.escape(ref)}.
var table = {data};
var name = decodeURIComponent(location.hash.replace(/^#doc\\//, ""));
if (Object.prototype.hasOwnProperty.call(table, name)) {{
  location.replace(table[name]);
}} else {{
  document.getElementById("msg").textContent = "No Lean declaration named " + name + ".";
}}
</script>
</body>
</html>
"""


def current_ref():
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ref", help="commit to pin the links to (default: HEAD)")
    args = parser.parse_args()

    ref = args.ref or current_ref()
    try:
        content, errors = assemble(ref)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        return 1
    OUT_CONTENT.write_text(content)
    OUT_FIND.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIND.write_text(render_find(ref, lean_declarations()))
    print(f"wrote {OUT_CONTENT.relative_to(ROOT)} and {OUT_FIND.relative_to(ROOT)} at {ref}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
