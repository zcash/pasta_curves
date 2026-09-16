#!/usr/bin/env python3
"""Check that the export roots reach every module of the package.

lean4export exports the given root modules and their transitive imports, so a project
module that no root imports would silently drop out of the nanoda re-check. This script
computes the import closure of the roots over the package's own `.lean` files and fails if
any project module lies outside it.

Usage, from the `lean/` directory: scripts/check_export_coverage.py <root-module>...
Exits 1 if a project module is unreachable from the roots.
"""

import re
import sys
from pathlib import Path

IMPORT = re.compile(r"^import\s+([A-Za-z0-9_.]+)", re.MULTILINE)


def module_name(path):
    return ".".join(path.with_suffix("").parts)


def main():
    roots = sys.argv[1:]
    if not roots:
        print("usage: check_export_coverage.py <root-module>...", file=sys.stderr)
        return 2
    files = [Path("PastaAsm.lean")]
    files += sorted(Path("PastaAsm").rglob("*.lean"))
    imports = {module_name(f): set(IMPORT.findall(f.read_text())) for f in files}
    reachable, todo = set(), list(roots)
    while todo:
        m = todo.pop()
        if m in reachable or m not in imports:
            continue
        reachable.add(m)
        todo.extend(imports[m])
    missing = sorted(set(imports) - reachable)
    if missing:
        print("VIOLATION: module(s) outside the export roots' import closure:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        return 1
    print(f"export coverage: all {len(imports)} project modules reachable from the roots")
    return 0


if __name__ == "__main__":
    sys.exit(main())
