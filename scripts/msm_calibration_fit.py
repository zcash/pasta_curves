"""Fit the GLV MSM's machine-calibrated constants to timing grids.

Reads the grids `scripts/calibrate-msm.sh` writes (one CSV per machine and build), evaluates the
planner of `src/glv/msm/plan.rs` with the constants of `src/glv/msm/calibration.rs` on them, and
searches the constants that minimise the time lost against the fastest measured alternative.
Prints, per grid, the loss of the current and of the recommended constants, the cells where the
grids disagree, and the recommended constants in the shape of `calibration.rs`.

The planner is reimplemented here; `--check-routes` compares it with a routing snapshot written
by the Rust planner (see the body of the commit that introduced `Plan`), and must report no
difference before any recommendation is trusted.

Usage: python3 scripts/msm_calibration_fit.py [--this GRID] GRID...
       python3 scripts/msm_calibration_fit.py --check-routes SNAPSHOT
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import re
import statistics
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_RS = ROOT / "src/glv/msm/calibration.rs"
PLAN_RS = ROOT / "src/glv/msm/plan.rs"

# Facts of the algorithm and of the fallback, read from plan.rs: never fitted.
SCALAR_BITS = 256
HALF_BITS = 127
HALVES = 2
BUCKET_SUM_ADDITIONS = 2

# A loss above this counts as a bad cell in the reports.
BAD_LOSS = 0.05


def _int(expr: str) -> int:
    """Evaluates a Rust integer literal or `a << b`."""
    expr = expr.replace("_", "").strip()
    if "<<" in expr:
        left, right = expr.split("<<")
        return int(left) << int(right)
    return int(expr)


def _rust_consts(text: str) -> dict[str, int]:
    pattern = re.compile(r"const (\w+): usize = ([^;]+);")
    return {name: _int(value) for name, value in pattern.findall(text)}


def _rust_weights(text: str, name: str) -> Weights:
    match = re.search(
        rf"const {name}: PlannerWeights = PlannerWeights \{{\s*"
        r"point_visit: (\d+),\s*bucket_addition: (\d+),\s*doubling: (\d+),",
        text,
    )
    if match is None:
        raise ValueError(f"{name} not found")
    return Weights(*(int(group) for group in match.groups()))


@dataclass(frozen=True)
class Weights:
    """`PlannerWeights`: the cost of each operation, in fifths of a doubling."""

    point_visit: int
    bucket_addition: int
    doubling: int


@dataclass(frozen=True)
class Constants:
    """The values of `calibration.rs`, plus the policy and schedule the planner reads."""

    min_glv_multiexp_terms: int
    serial_straus_max_terms: int
    straus_leaf_terms: int
    min_parallel_straus_terms: int
    max_parallel_straus_terms: int
    serial_narrow_window_bits: int
    serial_wide_window_bits: int
    serial_wide_window_min_terms: int
    glv_weights: Weights
    generic_weights: Weights
    glv_extra_window_bits: int
    straus_pool_shares: int = 2
    thresholds: tuple[int, ...] = field(default_factory=tuple)


def read_constants() -> Constants:
    """The constants the crate currently has."""
    calibration = CALIBRATION_RS.read_text()
    plan = PLAN_RS.read_text()
    values = _rust_consts(calibration)
    plan_values = _rust_consts(plan)
    table = re.search(r"DEFAULT_MULTIEXP_WINDOW_THRESHOLDS: \[u32; \d+\] = \[([^\]]*)\]", plan)
    if table is None:
        raise ValueError("DEFAULT_MULTIEXP_WINDOW_THRESHOLDS not found")
    thresholds = tuple(_int(v) for v in table.group(1).split(",") if v.strip())
    return Constants(
        min_glv_multiexp_terms=values["MIN_GLV_MULTIEXP_TERMS"],
        serial_straus_max_terms=values["SERIAL_STRAUS_MAX_TERMS"],
        straus_leaf_terms=values["STRAUS_LEAF_TERMS"],
        min_parallel_straus_terms=values["MIN_PARALLEL_STRAUS_TERMS"],
        max_parallel_straus_terms=values["MAX_PARALLEL_STRAUS_TERMS"],
        serial_narrow_window_bits=values["SERIAL_NARROW_WINDOW_BITS"],
        serial_wide_window_bits=values["SERIAL_WIDE_WINDOW_BITS"],
        serial_wide_window_min_terms=values["SERIAL_WIDE_WINDOW_MIN_TERMS"],
        glv_weights=_rust_weights(calibration, "GLV_PLANNER_WEIGHTS"),
        generic_weights=_rust_weights(calibration, "GENERIC_PLANNER_WEIGHTS"),
        glv_extra_window_bits=values["GLV_EXTRA_WINDOW_BITS"],
        straus_pool_shares=plan_values["STRAUS_POOL_SHARES"],
        thresholds=thresholds,
    )


# ---- The planner, as in plan.rs ----


def default_window_bits(terms: int, c: Constants) -> int:
    """`default_multiexp_window_bits`: halo2's `best_multiexp` schedule."""
    if terms < 4:
        return 1
    if terms < 32:
        return 3
    index = next((i for i, upper in enumerate(c.thresholds) if terms <= upper), len(c.thresholds))
    return 4 + index


def operation_counts(
    terms: int, scalar_bits: int, window_bits: int, threads: int, horner: bool
) -> tuple[int, int, int]:
    """The (point visits, bucket additions, doublings) on the critical path that
    `estimated_signed_booth_work` weighs. With `horner`, the shifts form one serial chain after
    the windows, as the GLV backend runs them, instead of each worker shifting its own windows,
    as `best_multiexp` does."""
    windows = scalar_bits // window_bits + 1
    buckets = 1 << (window_bits - 1)
    if threads <= 1:
        return (
            terms * windows,
            buckets * windows * BUCKET_SUM_ADDITIONS,
            window_bits * (windows - 1),
        )
    workers = min(threads, windows)
    waves = -(-windows // workers)
    if horner:
        shifts = window_bits * (windows - 1)
    else:
        shifts = 0
        window = windows - 1
        for _ in range(waves):
            shifts += window_bits * window
            if window < workers:
                break
            window -= workers
    return terms * waves, buckets * BUCKET_SUM_ADDITIONS * waves, shifts


def weigh(counts: tuple[int, int, int], w: Weights) -> int:
    return counts[0] * w.point_visit + counts[1] * w.bucket_addition + counts[2] * w.doubling


Route = tuple[str, int]


def plan(terms: int, threads: int, c: Constants, horner: bool) -> Route:
    """`plan`: ("straus", 0), ("leaves", jobs), ("buckets", width) or ("decline", 0)."""
    if threads > 1 and c.min_parallel_straus_terms <= terms <= c.max_parallel_straus_terms:
        return ("leaves", min(-(-threads // c.straus_pool_shares), terms))
    straus_max = c.serial_straus_max_terms if threads <= 1 else c.straus_leaf_terms
    if terms <= straus_max:
        return ("straus", 0)
    if terms < c.min_glv_multiexp_terms or (terms == c.min_glv_multiexp_terms and threads <= 1):
        if threads > 1:
            return ("decline", 0)
        if terms < c.serial_wide_window_min_terms:
            return ("buckets", c.serial_narrow_window_bits)
        return ("buckets", c.serial_wide_window_bits)
    width = default_window_bits(terms, c)
    generic = weigh(operation_counts(terms, SCALAR_BITS, width, threads, False), c.generic_weights)
    best: tuple[int, int] | None = None
    for candidate in range(width, width + c.glv_extra_window_bits + 1):
        work = weigh(
            operation_counts(terms * HALVES, HALF_BITS, candidate, threads, horner),
            c.glv_weights,
        )
        if best is None or work < best[0]:
            best = (work, candidate)
    assert best is not None
    return ("buckets", best[1]) if best[0] <= generic else ("decline", 0)


def route_name(route: Route) -> str:
    kind, param = route
    return {
        "straus": "straus",
        "decline": "decline",
        "leaves": f"leaves {param}",
        "buckets": f"buckets {param}",
    }[kind]


def check_routes(snapshot: Path) -> int:
    """Compares this planner with a Rust routing snapshot; returns the number of differences."""
    c = read_constants()
    differences = 0
    for line in snapshot.read_text().splitlines():
        threads, terms, *rest = line.split()
        expected = " ".join(rest)
        got = route_name(plan(int(terms), int(threads), c, horner=False))
        if got != expected:
            differences += 1
            if differences <= 10:
                print(f"{threads} threads, {terms} terms: Rust {expected}, here {got}")
    return differences


# ---- Grids and losses ----


@dataclass
class Grid:
    """One machine's timing grid: medians per (threads, terms) cell and alternative."""

    name: str
    meta: dict[str, str]
    cells: dict[tuple[int, int], dict[Route, float]]


def read_grid(path: Path) -> Grid:
    meta: dict[str, str] = {}
    rows: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            key, _, value = line[1:].partition(":")
            meta[key.strip()] = value.strip()
        elif line.strip():
            rows.append(line)
    cells: dict[tuple[int, int], dict[Route, float]] = {}
    for row in csv.DictReader(rows):
        method = row["method"]
        kind = "decline" if method == "generic" else method
        param = 0 if kind in ("decline", "straus") else int(row["param"])
        cell = cells.setdefault((int(row["threads"]), int(row["terms"])), {})
        cell[(kind, param)] = float(row["median_ns"])
    return Grid(path.stem, meta, cells)


@dataclass
class Loss:
    """How a set of constants fares on one grid."""

    mean: float
    worst: float
    worst_cell: tuple[int, int]
    bad: int
    cells: int
    unmeasured: int

    def row(self) -> str:
        t, n = self.worst_cell
        return (
            f"mean {self.mean * 100:5.1f}%  worst {self.worst * 100:6.1f}% ({n} terms, {t} thr)"
            f"  over {BAD_LOSS * 100:.0f}%: {self.bad:3}/{self.cells}"
            + (f"  unmeasured choices: {self.unmeasured}" if self.unmeasured else "")
        )


def loss(grid: Grid, c: Constants, horner: bool) -> Loss:
    losses: list[float] = []
    worst, worst_cell, unmeasured = 0.0, (0, 0), 0
    for (threads, terms), times in grid.cells.items():
        choice = plan(terms, threads, c, horner)
        if choice not in times:
            unmeasured += 1
            continue
        value = times[choice] / min(times.values()) - 1
        losses.append(value)
        if value > worst:
            worst, worst_cell = value, (threads, terms)
    return Loss(
        statistics.fmean(losses) if losses else math.nan,
        worst,
        worst_cell,
        sum(value > BAD_LOSS for value in losses),
        len(losses),
        unmeasured,
    )


def objective(grids: list[Grid], c: Constants, horner: bool) -> float:
    """Each machine counts alike: the mean over grids of each grid's mean loss, plus a heavy
    penalty for choices no grid measured, so that the search stays where data exists."""
    total = 0.0
    for grid in grids:
        result = loss(grid, c, horner)
        total += (0.0 if math.isnan(result.mean) else result.mean) + result.unmeasured
    return total / len(grids)


# ---- Search ----


def _descend(
    grids: list[Grid],
    start: Constants,
    candidates: dict[str, Iterable[object]],
    horner: bool,
) -> Constants:
    """Coordinate descent over the given fields; ties keep the current value."""
    best, best_score = start, objective(grids, start, horner)
    improved = True
    while improved:
        improved = False
        for name, values in candidates.items():
            for value in values:
                trial = replace(best, **{name: value})
                score = objective(grids, trial, horner)
                if score < best_score - 1e-12:
                    best, best_score, improved = trial, score, True
    return best


def fit(grids: list[Grid], start: Constants, horner: bool) -> Constants:
    sizes = sorted({terms for grid in grids for (_, terms) in grid.cells})
    small = [n for n in sizes if n <= 512]
    weight_range = range(1, 21)
    weights = [Weights(v, b, 5) for v, b in itertools.product(weight_range, weight_range)]
    candidates: dict[str, Iterable[object]] = {
        "serial_straus_max_terms": [n for n in small if n < start.min_glv_multiexp_terms],
        "min_glv_multiexp_terms": [n for n in small if n >= 64],
        "serial_wide_window_min_terms": small,
        "serial_narrow_window_bits": range(3, 9),
        "serial_wide_window_bits": range(3, 9),
        "min_parallel_straus_terms": [n for n in small if n >= 2] + [3],
        "max_parallel_straus_terms": small,
        "glv_extra_window_bits": range(0, 4),
    }
    current = start
    for _ in range(3):
        current = _descend(grids, current, candidates, horner)
        current = _descend(
            grids, current, {"glv_weights": weights, "generic_weights": weights}, horner
        )
    return current


# ---- Reports ----


def rust_block(c: Constants) -> str:
    def weights(name: str, w: Weights) -> str:
        return (
            f"pub(super) const {name}: PlannerWeights = PlannerWeights {{\n"
            f"    point_visit: {w.point_visit},\n"
            f"    bucket_addition: {w.bucket_addition},\n"
            f"    doubling: {w.doubling},\n}};"
        )

    lines = [
        f"pub(super) const MIN_GLV_MULTIEXP_TERMS: usize = {c.min_glv_multiexp_terms};",
        f"pub(super) const SERIAL_STRAUS_MAX_TERMS: usize = {c.serial_straus_max_terms};",
        f"pub(super) const STRAUS_LEAF_TERMS: usize = {c.straus_leaf_terms};",
        f"pub(super) const MIN_PARALLEL_STRAUS_TERMS: usize = {c.min_parallel_straus_terms};",
        f"pub(super) const MAX_PARALLEL_STRAUS_TERMS: usize = {c.max_parallel_straus_terms};",
        f"pub(super) const SERIAL_NARROW_WINDOW_BITS: usize = {c.serial_narrow_window_bits};",
        f"pub(super) const SERIAL_WIDE_WINDOW_BITS: usize = {c.serial_wide_window_bits};",
        f"pub(super) const SERIAL_WIDE_WINDOW_MIN_TERMS: usize = {c.serial_wide_window_min_terms};",
        weights("GLV_PLANNER_WEIGHTS", c.glv_weights),
        weights("GENERIC_PLANNER_WEIGHTS", c.generic_weights),
        f"pub(super) const GLV_EXTRA_WINDOW_BITS: usize = {c.glv_extra_window_bits};",
    ]
    return "\n".join(lines)


def disagreements(grids: list[Grid]) -> list[str]:
    """Cells measured on several grids whose fastest alternative differs."""
    best: dict[tuple[int, int], dict[str, Route]] = {}
    for grid in grids:
        for cell, times in grid.cells.items():
            best.setdefault(cell, {})[grid.name] = min(times, key=times.__getitem__)
    lines = []
    for (threads, terms), winners in sorted(best.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if len(set(winners.values())) > 1:
            listed = ", ".join(f"{g}: {route_name(r)}" for g, r in winners.items())
            lines.append(f"  {terms} terms, {threads} threads: {listed}")
    return lines


def report(grids: list[Grid], this: str | None) -> None:
    current = read_constants()
    rows: list[tuple[str, Callable[[Grid], Loss]]] = [
        ("current constants, current model", lambda g: loss(g, current, False)),
    ]
    for horner in (False, True):
        fitted = fit(grids, current, horner)
        label = "recommended, " + ("Horner shift model" if horner else "current model")
        rows.append((label, lambda g, f=fitted, h=horner: loss(g, f, h)))
        print(f"\n== {label}\n{rust_block(fitted)}")
    print("\n== losses against the fastest measured alternative, per grid")
    for grid in grids:
        marker = " (this machine)" if grid.name == this else ""
        print(f"\n{grid.name}{marker}: {grid.meta.get('cpu', '?')}, {grid.meta.get('os', '?')}")
        for label, evaluate in rows:
            print(f"  {label:<34} {evaluate(grid).row()}")
    lines = disagreements(grids)
    print("\n== cells where the grids disagree on the fastest alternative")
    print("\n".join(lines) if lines else "  none")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("grids", nargs="*", type=Path)
    parser.add_argument("--this", type=Path, help="the grid this run measured")
    parser.add_argument("--check-routes", type=Path, help="a Rust routing snapshot")
    args = parser.parse_args()
    if args.check_routes is not None:
        differences = check_routes(args.check_routes)
        print(f"{differences} routes differ from the Rust planner")
        raise SystemExit(1 if differences else 0)
    paths = sorted(set(args.grids) | ({args.this} if args.this else set()))
    if not paths:
        parser.error("no grid given")
    grids = [read_grid(path) for path in paths]
    report(grids, args.this.stem if args.this else None)


if __name__ == "__main__":
    main()
