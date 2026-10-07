#!/usr/bin/env bash
# Build the blueprint's web version into `blueprint/web/`.
#
# `lake build PastaCurvesBlueprint:blueprint` first has LeanArchitect extract the nodes, their
# edges, and their proved status from `lean/PastaCurvesBlueprint.lean`. `blueprint/sources.py`
# then assembles `blueprint/src/content.tex` from that output and `blueprint/src/map.tex`,
# expanding the quotes of the book and of the Lean docstrings, with every link pinned to REF
# (default: the checked-out commit). plasTeX with the leanblueprint plugin renders
# `blueprint/src/web.tex`. The tooling is pinned in `blueprint/requirements.txt`; graphviz and
# its headers must be installed for pygraphviz (see `blueprint/README.md`).
#
# PYTHON selects the interpreter for `sources.py` (default: `python3`), PLASTEX the plasTeX
# executable (default: `plastex` from PATH), and LAKE the lake executable (default: `lake`);
# SKIP_LAKE=1 reuses LeanArchitect's last output. Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python3}
PLASTEX=${PLASTEX:-plastex}
LAKE=${LAKE:-lake}
export PYTHONDONTWRITEBYTECODE=1

"$PYTHON" blueprint/test_sources.py
if [ "${SKIP_LAKE:-}" != 1 ]; then
  (cd lean && "$LAKE" build PastaCurvesBlueprint:blueprint)
fi
# plasTeX does not remove the pages of a previous build, so start from an empty output.
rm -rf blueprint/web
if [ -n "${REF:-}" ]; then
  "$PYTHON" blueprint/sources.py --ref "$REF"
else
  "$PYTHON" blueprint/sources.py
fi

# plasTeX reports a missing input or an undefined macro or environment as a WARNING and carries on,
# so the build would only fail later, if at all, far from the cause. Treat every warning as an
# error.
status=0
output=$(cd blueprint/src && "$PLASTEX" -c plastex.cfg web.tex 2>&1) || status=$?
echo "$output"
if [ "$status" -ne 0 ]; then
  echo "plasTeX failed with exit status $status" >&2
  exit "$status"
fi
if grep -q "WARNING" <<<"$output"; then
  echo "plasTeX reported warnings:" >&2
  grep "WARNING" <<<"$output" >&2
  exit 1
fi

# The graph is drawn by Graphviz compiled to WebAssembly, which browsers do not load from a
# `file://` page, so the blueprint has to be served over HTTP to be viewed.
echo "Blueprint built in blueprint/web/. To view it:"
echo "  python3 -m http.server --directory blueprint/web 8000"
echo "  then open http://localhost:8000/dep_graph_document.html"
