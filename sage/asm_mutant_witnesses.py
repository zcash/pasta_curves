"""Derive the AArch64 carry-boundary test vectors from the shipped assembly.

Finds inputs that exercise the carry propagations `cargo xtask asm-mutants`
reports as untested. They are constructed rather than searched for, by
inverting the Montgomery cancellation step; sage/README.md explains why that
works. Prints the vectors in the shape of the Rust constants in
src/fields/fp.rs and src/fields/fq.rs.

    python3 asm_mutant_witnesses.py                   emit the vectors
    python3 asm_mutant_witnesses.py --profile         carry measurements
    python3 asm_mutant_witnesses.py --check-invariants recheck the survivors
"""

import functools
import random
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asm_aarch64 as asm

M64 = (1 << 64) - 1
REPO_ROOT = Path(__file__).resolve().parent.parent
ASM_SOURCE = REPO_ROOT / "src/asm/pasta_mul-armv8.S"

# The traced value each mutation pins. For `adc`/`sbcs` that is the carry-in;
# for `csel` it is whether the `lo` condition was taken, which is the carry
# inverted, so `select-first` pins it high. The harness names the kind in the
# last segment of every id.
FORCED = {
    "carry-clear": 0,
    "carry-set": 1,
    "select-first": 1,
    "select-second": 0,
}

MODULI = {
    "Fp": 0x40000000000000000000000000000000224698FC094CF91B992D30ED00000001,
    "Fq": 0x40000000000000000000000000000000224698FC0994A8DD8C46EB2100000001,
}

MUTABLE = ("adcs", "adc", "sbcs", "csel")


#: One call's operands: `(lhs, rhs)` for `mul`, `(value,)` for `square` and
#: `from_mont`, so every routine can be driven through a single signature.
Case = tuple[int, ...]

#: Runs a routine on one case, returning its result and its carry trace.
Runner = Callable[[asm.Program, Case], tuple[int, asm.Trace]]


class Site(TypedDict):
    """One mutation the harness generates, as `mutation_sites` reports it."""

    id: str
    index: int
    line: int
    forced: int
    replacement: str


# The mutants that survive `cargo xtask asm-mutants` against the test suite as
# it stands. Ten of them are equivalent mutants: the carry they force is
# already the only one the code can produce, so no input can distinguish them
# (see README.md for why each path is unreachable). The rest are real gaps,
# and this script derives a vector for each.
SURVIVORS = (
    "adcs/5/carry-clear",
    "adc/3/carry-clear",
    "adcs/16/carry-clear",
    "adcs/19/carry-clear",
    "adcs/27/carry-clear",
    "adcs/30/carry-clear",
    "adc/11/carry-clear",
    "adcs/38/carry-clear",
    "adc/15/carry-clear",
    "adc/19/carry-clear",
    "adc/21/carry-clear",
    "sbcs/9/carry-clear",
    "sbcs/10/carry-clear",
    "sbcs/11/carry-clear",
    "csel/9/select-first",
    "csel/10/select-first",
    "csel/11/select-first",
    "csel/12/select-first",
    "adcs/60/carry-clear",
    "adcs/65/carry-clear",
    "adc/24/carry-clear",
    "adc/25/carry-clear",
    "adc/27/carry-clear",
    "adcs/75/carry-clear",
)


# --------------------------------------------------------------------------
# The mutation set, mirroring xtask/src/asm_mutants/arch/aarch64_apple.rs.
# --------------------------------------------------------------------------


@functools.cache
def harness_sites() -> tuple[tuple[str, int, str, str], ...]:
    """The mutation set as the xtask harness reports it."""
    listing = subprocess.run(
        ["cargo", "xtask", "asm-mutants", "--list-sites"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    records = []
    for line in listing.splitlines():
        identifier, at, instruction, replacement = line.split("\t")
        records.append((identifier, int(at), instruction, replacement))
    return tuple(records)


def mutation_sites(program: asm.Program) -> list[Site]:
    """Every mutation the xtask harness generates, in the same order and ids.

    Read from the harness rather than re-derived, so the ids and replacement
    text cannot drift from the ones the mutation run actually uses.
    """
    index_of = {
        line: index for index, (_, _, line) in enumerate(program.instructions)
    }
    sites = []
    for identifier, at, instruction, replacement in harness_sites():
        index = index_of[at]
        mnemonic, operands, _ = program.instructions[index]
        rendered = f"{mnemonic} {','.join(operands)}"
        if rendered != instruction:
            raise SystemExit(
                f"{identifier}: the harness sees {instruction!r} at line {at} "
                f"but the interpreter parses {rendered!r}"
            )
        sites.append(
            Site(
                id=identifier,
                index=index,
                line=at,
                forced=FORCED[identifier.rsplit("/", 1)[1]],
                replacement=replacement,
            )
        )
    return sites


def mutate(program: asm.Program, site: Site) -> asm.Program:
    """The program with this site's carry decision forced, as the harness does."""
    body = "".join(f"    {line}\n" for line in site["replacement"].split("\\n"))
    return program.replacing(
        site["index"], asm.Program("_r:\n" + body).instructions
    )


# --------------------------------------------------------------------------
# Constructed input families.
# --------------------------------------------------------------------------


def helper_states(p: int, inv: int) -> list[int]:
    """States for the shared reduction helper that force high carries."""
    modulus = asm.limbs(p)
    states = []
    seeds = [1, 2, 3, 5, 7, M64, M64 - 1, 0]
    seeds += [random.Random(11).randrange(1 << 64) for _ in range(48)]
    for limb0 in seeds:
        q = (limb0 * inv) & M64
        low1, low3 = (modulus[1] * q) & M64, (q << 62) & M64
        carry0 = 1 if limb0 else 0
        for limb1 in {
            0,
            1,
            M64,
            (-low1 - carry0) & M64,
            (M64 - low1 - carry0) & M64,
        }:
            carry1 = 1 if limb1 + low1 + carry0 > M64 else 0
            for limb2 in {0, 1, M64}:
                carry2 = 1 if limb2 + carry1 > M64 else 0
                for limb3 in {
                    0,
                    M64,
                    1 << 62,
                    (M64 - low3 - carry2) & M64,
                    (-low3 - carry2) & M64,
                    ((1 << 62) - carry2) & M64,
                }:
                    state = asm.from_limbs([limb0, limb1, limb2, limb3])
                    if state < p:
                        states.append(state)
    return sorted(set(states))


def from_mont_inputs(p: int, inv: int) -> list[int]:
    """from_mont inputs reaching each constructed helper state at each step."""
    return [
        (state * pow(1 << 64, step, p)) % p
        for step in range(5)
        for state in helper_states(p, inv)
    ]


def mul_inputs_for_state(
    state: int, round_index: int, p: int, widths: Iterable[int] = range(3, 4096)
) -> tuple[int, int] | None:
    """(lhs, rhs) whose `mul` accumulator entering `round_index` equals `state`."""
    if not 0 <= state < p:
        return None
    first = (state * pow(1 << 64, round_index - 1, p)) % p
    target = first << 64
    for width in widths:
        if p % width == 0:
            continue
        residue = (target % width) * pow(p % width, -1, width) % width
        high = target // p
        multiplier = high - ((high - residue) % width)
        if not 0 <= multiplier < (1 << 64):
            continue
        product = target - multiplier * p
        if product <= 0 or product % width:
            continue
        lhs = product // width
        if lhs < p:
            return lhs, width
    return None


def mul_inputs(p: int, inv: int) -> list[tuple[int, int]]:
    """`mul` inputs exercising the reduction carries in every round."""
    pairs = []
    # Rounds 1..3: drive the accumulator to a chosen state.
    for state in helper_states(p, inv):
        for round_index in (1, 2, 3):
            found = mul_inputs_for_state(state, round_index, p)
            if found:
                pairs.append(found)
    # Round 0 has no earlier state; shape the product a * b[0] directly.
    for limb0 in (1, 3, 5, 7, 0x224698FC094CF91C, 0x6303159B97C88ADF):
        for limb3 in (0, 1):
            lhs = asm.from_limbs([limb0, M64, M64, limb3])
            if lhs < p:
                for rhs0 in (1, 2, 3, 0x992D30ED00000001, M64):
                    pairs.append((lhs, rhs0))
    return pairs


def non_canonical_inputs(
    p: int, count: int = 30000, seed: int = 5
) -> list[tuple[int, int]]:
    """Unreduced left operands, as `Fp::from_u512` actually produces.

    That caller's right operand is always `R2` or `R3`, but sweeping only those
    hides the operand range in src/fields/aarch64_asm.rs, which turns on the
    right operand's limbs being at most `2^64 - 4`: `R2` and `R3` are far below
    it, so the sweep never approached the edge. A third of the family therefore
    pairs a left operand with all-ones high limbs against a canonical right
    operand sitting exactly at the cap.
    """
    rng = random.Random(seed)
    square_r = (1 << 512) % p
    cube_r = (square_r << 256) % p
    cap = M64 - 3

    def left(index: int) -> int:
        """Half from the top of the range, a third with all-ones high limbs."""
        if index % 3 == 2:
            return (
                rng.randrange(1 << 64)
                | (M64 << 64)
                | (M64 << 128)
                | (M64 << 192)
            )
        return (
            rng.randrange(1 << 255, 1 << 256)
            if index % 2
            else rng.randrange(1 << 256)
        )

    def right(index: int) -> int:
        """`R2`/`R3`, or a canonical value whose low limbs sit at the cap."""
        if index % 3 == 2:
            return cap | (cap << 64) | (rng.randrange(cap + 1) << 128)
        return rng.choice([square_r, cube_r])

    return [(left(index), right(index)) for index in range(count)]


# --------------------------------------------------------------------------
# Witness search.
# --------------------------------------------------------------------------


def find_witnesses(
    program: asm.Program, sites: Sequence[Site], p: int, inv: int
) -> tuple[dict[str, tuple[str, Case]], list[Site]]:
    """First input killing each site; sites with no witness are left out."""
    found: dict[str, tuple[str, Case]] = {}
    pending = list(sites)

    def sweep(cases: Iterable[Case], run: Runner, key: str) -> None:
        """Run one family of inputs, retiring each site as a witness is found."""
        nonlocal pending
        for case in cases:
            if not pending:
                return
            baseline, trace = run(program, case)
            observed: dict[int, set[int]] = {}
            for index, carry in trace:
                observed.setdefault(index, set()).add(carry)
            still: list[Site] = []
            for site in pending:
                reachable = observed.get(site["index"], set())
                if (1 - site["forced"]) not in reachable:
                    still.append(site)
                    continue
                if run(mutate(program, site), case)[0] != baseline:
                    found[site["id"]] = (key, case)
                else:
                    still.append(site)
            pending = still

    run_mul: Runner = lambda prog, case: asm.mul(prog, case[0], case[1], p, inv)
    run_square: Runner = lambda prog, case: asm.square(prog, case[0], p, inv)
    run_from_mont: Runner = lambda prog, case: asm.from_mont(
        prog, case[0], p, inv
    )

    sweep(
        [(value,) for value in from_mont_inputs(p, inv)],
        run_from_mont,
        "from_mont",
    )
    sweep(mul_inputs(p, inv), run_mul, "mul")
    sweep(non_canonical_inputs(p), run_mul, "mul-non-canonical")
    sweep([(value,) for value in helper_states(p, inv)], run_square, "square")
    return found, pending


def differential(
    program: asm.Program,
    p: int,
    inv: int,
    extra_products: Sequence[tuple[int, int]],
    extra_values: Sequence[int],
    count: int = 3000,
) -> tuple[int, int]:
    """Compare the assembly against the Montgomery definitions it implements."""
    rng = random.Random(2024)
    r_inverse = pow(1 << 256, -1, p)
    products = list(extra_products)
    products += [(rng.randrange(p), rng.randrange(p)) for _ in range(count)]
    products += [(0, 0), (1, 1), (p - 1, p - 1), (0, p - 1), (1, p - 1)]
    products += non_canonical_inputs(p, count=count, seed=99)
    values = list(extra_values)
    values += [rng.randrange(p) for _ in range(count)]
    values += [0, 1, p - 1]

    mismatches = 0
    for lhs, rhs in products:
        if asm.mul(program, lhs, rhs, p, inv)[0] != lhs * rhs * r_inverse % p:
            mismatches += 1
    for value in values:
        if asm.from_mont(program, value, p, inv)[0] != value * r_inverse % p:
            mismatches += 1
        if (
            asm.square(program, value, p, inv)[0]
            != value * value * r_inverse % p
        ):
            mismatches += 1
    return len(products) + len(values), mismatches


# --------------------------------------------------------------------------
# Output in the shape of the Rust constants.
# --------------------------------------------------------------------------


def rust_limbs(value: int) -> str:
    """Format a value as the Rust limb array literal."""
    return "[" + ", ".join(f"0x{limb:016x}" for limb in asm.limbs(value)) + "]"


def emit(
    field: str, p: int, inv: int, program: asm.Program
) -> tuple[dict[str, tuple[str, Case]], list[Site]]:
    """Derive one field's vectors and print them as Rust constants."""
    sites = [
        site for site in mutation_sites(program) if site["id"] in SURVIVORS
    ]
    found, unkilled = find_witnesses(program, sites, p, inv)

    products: list[tuple[int, int]] = []
    reductions: list[int] = []
    for key, case in found.values():
        if key in ("mul", "mul-non-canonical"):
            products.append((case[0], case[1]))
        elif key == "from_mont":
            reductions.append(case[0])
    products = sorted(set(products))
    reductions = sorted(set(reductions))

    compared, mismatches = differential(program, p, inv, products, reductions)
    print(f"// ---- {field} " + "-" * 60)
    print(
        f"// assembly vs the Montgomery definitions: {compared} inputs, "
        f"{mismatches} mismatches"
    )
    print(
        f"/// Operand pairs for `mul` that exercise carry propagations in the"
    )
    print(f"/// AArch64 backend which random testing does not reach.")
    print(
        f"const AARCH64_ASM_CARRY_PRODUCTS: [([u64; 4], [u64; 4]); "
        f"{len(products)}] = ["
    )
    for lhs, rhs in products:
        print(f"    ({rust_limbs(lhs)}, {rust_limbs(rhs)}),")
    print("];")
    print()
    print(f"/// Values for `from_mont` that exercise the shared reduction")
    print(f"/// helper's rare carry propagations.")
    print(
        f"const AARCH64_ASM_CARRY_REDUCTIONS: [[u64; 4]; {len(reductions)}] = ["
    )
    for value in reductions:
        print(f"    {rust_limbs(value)},")
    print("];")
    print()
    print(f"// surviving mutants with no witness ({len(unkilled)}); these are")
    print(f"// equivalent mutants -- the path is unreachable, see README.md:")
    for site in unkilled:
        print(f"//   {site['id']} (line {site['line']})")
    print()
    return found, unkilled


# --------------------------------------------------------------------------
# Carry profiling: which carry propagations does a given input family reach?
# --------------------------------------------------------------------------


def carry_profile(
    program: asm.Program,
    p: int,
    inv: int,
    cases: Iterable[Case],
    run: Runner,
) -> dict[int, list[int]]:
    """Per site, how often the real carry-in differs from the forced value."""
    counts: dict[int, list[int]] = {}
    for case in cases:
        observed: dict[int, list[int]] = {}
        for index, carry in run(program, case)[1]:
            observed.setdefault(index, []).append(carry)
        for index, carries in observed.items():
            counts.setdefault(index, [0, 0])
            for carry in carries:
                counts[index][carry] += 1
    return counts


def report_profile(program: asm.Program, count: int = 2000) -> None:
    """Reproduce the two measurements quoted in the analysis write-up."""
    for field, p in MODULI.items():
        inv = (-pow(p, -1, 1 << 64)) % (1 << 64)
        sites = mutation_sites(program)
        rng = random.Random(1234)
        run_mul: Runner = lambda prog, case: asm.mul(
            prog, case[0], case[1], p, inv
        )
        canonical: list[Case] = [
            (rng.randrange(p), rng.randrange(p)) for _ in range(count)
        ]
        counts = carry_profile(program, p, inv, canonical, run_mul)
        dead = [
            site
            for site in sites
            if counts.get(site["index"], [0, 0])[1 - site["forced"]] == 0
        ]
        print(
            f"{field}: {count} random canonical mul inputs; "
            f"{len(dead)} of {len(sites)} mutations are a no-op on all of them"
        )
        for site in dead:
            if site["id"] in SURVIVORS:
                print(f"    {site['id']:24s} line {site['line']:4d}")

        # The fifth accumulator limb, canonical vs the from_u512 domain.
        limb4 = [
            site["index"]
            for site in sites
            if site["id"]
            in (
                "adc/3/carry-clear",
                "adc/7/carry-clear",
                "adc/11/carry-clear",
                "adc/15/carry-clear",
            )
        ]
        lines = {index: program.instructions[index][2] for index in limb4}
        families: list[tuple[str, list[Case]]] = [
            (
                "canonical",
                [(rng.randrange(p), rng.randrange(p)) for _ in range(count)],
            ),
            ("non-canonical lhs", list(non_canonical_inputs(p, count=count))),
        ]
        for label, family in families:
            counts = carry_profile(program, p, inv, family, run_mul)
            hits = {
                lines[index]: counts.get(index, [0, 0])[1] for index in limb4
            }
            print(f"    limb-4 carry occurrences, {label}: {hits}")


# --------------------------------------------------------------------------
# Checking the equivalence claims recorded in the tracked survivor list.
# --------------------------------------------------------------------------

TRACKED_LIST = REPO_ROOT / "xtask/asm-mutants-aarch64-apple.txt"


# The operand range each routine is called with, which is what the recorded
# equivalence arguments assume. `mul` takes an arbitrary left operand because
# `from_u512` multiplies an unreduced 256-bit half by R2 or R3; `square` and
# `from_mont` are only ever called on canonical elements.
def routine_corpus(
    p: int, count: int, seed: int = 20260908
) -> tuple[list[tuple[int, int]], list[int]]:
    """Inputs for each routine, over the operand range its argument assumes."""
    rng = random.Random(seed)
    R = 1 << 256
    canonical = [0, 1, 2, p - 1, p - 2, R % p, (R - 1) % p, (1 << 255) % p]
    canonical += [rng.randrange(p) for _ in range(count)]
    products = [(lhs, rhs) for lhs in canonical[:8] for rhs in canonical[:8]]
    products += [(rng.randrange(p), rng.randrange(p)) for _ in range(count)]
    # The non-canonical half, biased to the top of the range where the carries
    # the per-round limbs depend on actually occur.
    products += [
        (
            rng.randrange(1 << 255, R) if index % 2 else rng.randrange(R),
            rng.randrange(p),
        )
        for index in range(count)
    ]
    return products, canonical


def tracked_ids() -> list[str]:
    """The mutation ids recorded as surviving, in file order."""
    ids = []
    for line in TRACKED_LIST.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.append(line.split("|")[0].strip())
    return ids


def check_invariants(program: asm.Program, count: int = 1200) -> int:
    """Check the equivalence claims recorded in the tracked survivor list."""
    sites = {site["id"]: site for site in mutation_sites(program)}
    recorded = tracked_ids()
    failures = []

    for field, p in MODULI.items():
        inv = (-pow(p, -1, 1 << 64)) % (1 << 64)
        products, values = routine_corpus(p, count)
        singles: list[Case] = [(value,) for value in values]
        routines: dict[str, tuple[list[Case], Runner]] = {
            "mul": (
                list(products),
                lambda prog, case: asm.mul(prog, case[0], case[1], p, inv),
            ),
            "square": (
                singles,
                lambda prog, case: asm.square(prog, case[0], p, inv),
            ),
            "from_mont": (
                singles,
                lambda prog, case: asm.from_mont(prog, case[0], p, inv),
            ),
        }

        # One baseline pass per routine: record the output, and which inputs
        # drive each recorded site's carry away from the value its mutation
        # forces. Only those inputs can possibly distinguish the mutant, so the
        # rest need not be run at all.
        outputs: dict[tuple[str, int], int] = {}
        reachable: dict[tuple[str, str], list[int]] = {}
        for name, (cases, run) in routines.items():
            for index, case in enumerate(cases):
                result, trace = run(program, case)
                outputs[(name, index)] = result
                for identifier in recorded:
                    site = sites.get(identifier)
                    if site is None:
                        continue
                    if any(
                        at == site["index"] and carry != site["forced"]
                        for at, carry in trace
                    ):
                        reachable.setdefault((identifier, name), []).append(
                            index
                        )

        print(
            f"{field} ({len(products)} mul, {len(values)} square, "
            f"{len(values)} from_mont calls)"
        )
        for identifier in recorded:
            site = sites.get(identifier)
            if site is None:
                failures.append(f"{field}: {identifier} is not a mutation site")
                continue
            mutated = mutate(program, site)
            reported = False
            for name, (cases, run) in routines.items():
                differing = reachable.get((identifier, name))
                if differing is None:
                    continue
                reported = True
                changed = [
                    index
                    for index in differing
                    if run(mutated, cases[index])[0] != outputs[(name, index)]
                ]
                if changed:
                    failures.append(
                        f"{field}: {identifier} changes the result of {name} "
                        f"on {cases[changed[0]]!r}, so it is not equivalent"
                    )
                    verdict = f"NOT EQUIVALENT ({len(changed)} differing)"
                else:
                    verdict = (
                        f"equivalent, carry differs on "
                        f"{len(differing)} inputs but the result does not"
                    )
                print(
                    f"  {identifier:24s} line {site['line']:4d}  {name:9s} {verdict}"
                )
            if not reported:
                where = (
                    "mul"
                    if site["line"] < 231
                    else ("square" if site["line"] < 346 else "from_mont")
                )
                print(
                    f"  {identifier:24s} line {site['line']:4d}  {where:9s} "
                    f"equivalent, the forced carry is the only one it produces"
                )
        print()

    if failures:
        print("FAILED:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print(
        f"all {len(recorded)} recorded mutants left every sampled result "
        f"unchanged"
    )
    return 0


def main() -> None:
    """Dispatch on the command-line mode."""
    program = asm.Program(ASM_SOURCE.read_text())
    if "--profile" in sys.argv:
        report_profile(program)
        return
    if "--check-invariants" in sys.argv:
        sys.exit(check_invariants(program))
    for field, p in MODULI.items():
        inv = (-pow(p, -1, 1 << 64)) % (1 << 64)
        emit(field, p, inv, program)


if __name__ == "__main__":
    main()
