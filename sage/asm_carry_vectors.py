"""Derive the carry-boundary test vectors of the assembly backends.

Montgomery multiplication, squaring, and conversion add multi-limb values limb by limb. A carry
that propagates through a limb that is exactly all ones, `x_j + y_j = 2^64 - 1` with a carry
arriving, occurs with probability about `2^-64` on random operands, so sampling never reaches
it. This script constructs operands that do, for both Pasta fields, and prints them in the shape
of the constants in `src/asm/tests.rs`.

It models the algorithm that both backends implement, not either backend's instructions: the
operand-scanning Montgomery multiplication with a five-limb accumulator, and a squaring that
reduces the low half of the full square before adding the high half. A boundary is a limb and a
step of that algorithm (`carry_boundaries`). The backends order the partial sums differently, so
a vector need not flip one particular instruction's carry on each; the assembly's correctness on
every input is proved in `lean/`, and these vectors exercise the rare carries on the hardware.

The operands are constructed rather than searched for. A Montgomery cancellation step is
invertible, so a chosen accumulator state can be reached by solving for the operands that lead
to it (`mul_for_state`); for squaring, the low half of `v^2` is chosen and `v` recovered as a
2-adic square root (`square_for_state`). Every vector is checked against exact arithmetic and
against its routine's contract, and only vectors reaching a boundary that a random sample of
operands does not are kept.

    python3 sage/asm_carry_vectors.py
"""

import random
import re
from pathlib import Path

M64 = (1 << 64) - 1
MODULI = {
    "FP": 0x40000000000000000000000000000000224698FC094CF91B992D30ED00000001,
    "FQ": 0x40000000000000000000000000000000224698FC0994A8DD8C46EB2100000001,
}


def crate_modulus(field: str) -> int:
    """The `MODULUS` limbs of `src/fields/fp.rs` or `fq.rs`, so a typo in `MODULI` cannot make
    the vectors miss their boundaries unnoticed (they would still be valid operands)."""
    name = field.capitalize()
    path = Path(__file__).resolve().parent.parent / f"src/fields/{field.lower()}.rs"
    pattern = rf"const MODULUS: {name} = {name}\(\[(.*?)\]\)"
    body = re.search(pattern, path.read_text(), re.DOTALL)
    return sum(int(limb, 16) << (64 * i) for i, limb in enumerate(re.findall(r"0x\w+", body[1])))


def limbs(x: int, n: int = 4) -> list[int]:
    return [(x >> (64 * i)) & M64 for i in range(n)]


def boundaries(x: int, y: int, n: int = 6) -> set[int]:
    """The limbs of `x + y` through which a carry propagates: `x_j + y_j = 2^64 - 1`, carry 1."""
    found, carry = set(), 0
    for j, (a, b) in enumerate(zip(limbs(x, n), limbs(y, n))):
        if carry and a + b == M64:
            found.add(j)
        carry = (a + b + carry) >> 64
    return found


def reduce_step(acc: int, p: int, inv: int, step: str, hits: set) -> int:
    """One Montgomery cancellation: add `q p` to clear the low limb, then drop that limb."""
    q = (acc & M64) * inv & M64
    hits.update((step, j) for j in boundaries(acc, q * p))
    return (acc + q * p) >> 64


def mul_boundaries(lhs: int, rhs: int, p: int, inv: int) -> tuple[int, set]:
    """`lhs rhs 2^-256 mod p` by four rounds of multiply-accumulate and cancellation."""
    acc, hits = 0, set()
    for i, b in enumerate(limbs(rhs)):
        hits.update((f"product {i}", j) for j in boundaries(acc, lhs * b))
        acc = reduce_step(acc + lhs * b, p, inv, f"reduction {i}", hits)
    return (acc - p if acc >= p else acc), hits


def square_boundaries(value: int, p: int, inv: int) -> tuple[int, set]:
    """`value^2 2^-256 mod p`: four cancellations on the square's low half, then its high half."""
    full, hits = value * value, set()
    acc = full & ((1 << 256) - 1)
    for i in range(4):
        acc = reduce_step(acc, p, inv, f"reduction {i}", hits)
    hits.update(("high half", j) for j in boundaries(acc, full >> 256))
    acc += full >> 256
    return (acc - p if acc >= p else acc), hits


def boundary_states(p: int, inv: int) -> list[int]:
    """Accumulators below `p` whose cancellation `+ q p` carries through all-ones limbs.

    The low limb fixes `q`, and so the limbs `w_j` of `q p`; each higher limb of the state is
    then `2^64 - 1 - w_j` (the sum is all ones, so an arriving carry propagates), `-w_j` (the
    sum wraps to zero), or a plain edge value."""
    seeds = [1, 2, 3, 5, 7, 9, 17, M64, M64 - 1, random.Random(11).randrange(1 << 64)]
    states = set()
    for limb0 in seeds:
        w = limbs((limb0 * inv & M64) * p)
        options = [{0, M64, (M64 - w[j]) & M64, -w[j] & M64} for j in (1, 2, 3)]
        for limb1 in options[0]:
            for limb2 in options[1]:
                for limb3 in options[2] | {1 << 62}:
                    state = limb0 | limb1 << 64 | limb2 << 128 | limb3 << 192
                    if state < p:
                        states.add(state)
    return sorted(states)


def mul_for_state(state: int, rounds: int, p: int) -> tuple[int, int] | None:
    """A canonical `lhs` and a one-limb `rhs` whose accumulator after `rounds` rounds is `state`.

    With `rhs` below `2^64`, rounds after the first add nothing but `q p`, so the accumulator
    after round `k` is that after round 1 times `2^(-64 (k - 1)) mod p`. Round 1 leaves
    `(lhs rhs + q p) / 2^64`; solving `lhs rhs = 2^64 s - q p` for a small `rhs` gives `lhs`."""
    target = (state * pow(1 << 64, rounds - 1, p) % p) << 64
    for rhs in range(3, 4096):
        if p % rhs == 0:
            continue
        q = target // p - ((target // p - (target % rhs) * pow(p % rhs, -1, rhs)) % rhs)
        if 0 <= q <= M64 and (target - q * p) % rhs == 0 and 0 < target - q * p < rhs * p:
            return (target - q * p) // rhs, rhs
    return None


def sqrt_2adic(x: int, bits: int = 256) -> int | None:
    """A square root of `x` modulo `2^bits`, for odd `x = 1 mod 8`, by Hensel lifting."""
    if x % 8 != 1:
        return None
    root = 1
    for k in range(3, bits):
        if (root * root - x) >> (k + 1) & 1:
            root += 1 << k
    return root % (1 << bits)


def square_for_state(state: int, p: int) -> int | None:
    """A canonical `v` whose square's low half is `state`, so squaring cancels from `state`."""
    root = sqrt_2adic(state)
    if root is None:
        return None
    half = 1 << 255
    candidates = (root, -root % (1 << 256), (root + half) % (1 << 256), (half - root) % (1 << 256))
    return next((v for v in candidates if v < p), None)


def from_u512_operands(p: int, count: int) -> list[tuple[int, int]]:
    """Unreduced left operands against `R^2`, `R^3`, and canonical right operands whose limbs
    1 to 3 sit at the contract's cap `2^64 - 3`: the multiplication's second contract, which
    `from_u512` uses, and where the accumulator's fifth limb is live."""
    rng, cap = random.Random(5), M64 - 2
    r2, r3 = (1 << 512) % p, (1 << 768) % p
    pairs = []
    for i in range(count):
        lhs = rng.randrange(1 << 64) | M64 << 64 | M64 << 128 | M64 << 192
        if i % 3 == 0:
            lhs = rng.randrange(1 << 255, 1 << 256)
        rhs = [r2, r3, cap | cap << 64 | rng.randrange(cap + 1) << 128][i % 3]
        if rhs < p:
            pairs.append((lhs, rhs))
    return pairs


def mul_contract(lhs: int, rhs: int, p: int) -> bool:
    """`src/asm/entry.rs`: a canonical `lhs`, or a canonical `rhs` with limbs 1 to 3 at most
    `2^64 - 3`."""
    return lhs < p or (rhs < p and all(limb <= M64 - 2 for limb in limbs(rhs)[1:]))


def select(candidates: list, run, sample: list) -> list:
    """Greedily keep the candidates reaching boundaries that neither `sample` nor an earlier
    choice reaches, most new boundaries first."""
    covered = set().union(*(run(*c)[1] for c in sample))
    chosen, pool = [], [(c, run(*c)[1]) for c in candidates]
    while True:
        best = max(pool, key=lambda item: len(item[1] - covered), default=None)
        if best is None or not best[1] - covered:
            return chosen
        chosen.append(best[0])
        covered |= best[1]


def derive(p: int) -> dict[str, list]:
    inv = -pow(p, -1, 1 << 64) % (1 << 64)
    r_inv = pow(1 << 256, -1, p)
    rng = random.Random(1)

    def mul(lhs: int, rhs: int) -> tuple[int, set]:
        assert mul_contract(lhs, rhs, p), (hex(lhs), hex(rhs))
        out, hits = mul_boundaries(lhs, rhs, p, inv)
        assert out == lhs * rhs * r_inv % p, (hex(lhs), hex(rhs))
        return out, hits

    def square(value: int) -> tuple[int, set]:
        assert value < p, hex(value)
        out, hits = square_boundaries(value, p, inv)
        assert out == value * value * r_inv % p, hex(value)
        return out, hits

    def from_mont(value: int) -> tuple[int, set]:
        return mul(value, 1)

    states = boundary_states(p, inv)
    products = [mul_for_state(s, k, p) for s in states for k in (1, 2, 3)]
    products = [pair for pair in products if pair] + from_u512_operands(p, 3000)
    conversions = [s * pow(1 << 64, k, p) % p for s in states for k in range(4)] + states
    # As for the conversion, a low half of `s 2^(64 k) mod p` cancels to `s` after `k` steps.
    squares = [v for v in (square_for_state(c, p) for c in conversions) if v is not None]

    canonical = [rng.randrange(p) for _ in range(400)]
    return {
        "PRODUCTS": select(products, mul, [(a, b) for a, b in zip(canonical, reversed(canonical))]),
        "CONVERSIONS": [
            v for (v,) in select([(v,) for v in conversions], from_mont, [(v,) for v in canonical])
        ],
        "SQUARES": [
            v for (v,) in select([(v,) for v in squares], square, [(v,) for v in canonical])
        ],
    }


def rust(value: int) -> str:
    return "[" + ", ".join(f"0x{limb:016x}" for limb in limbs(value)) + "]"


def main() -> None:
    for field, p in MODULI.items():
        assert p == crate_modulus(field), f"MODULI[{field!r}] differs from the crate's"
        vectors = derive(p)
        for kind, values in vectors.items():
            item = "(Limbs, Limbs)" if kind == "PRODUCTS" else "Limbs"
            print(f"const {field}_CARRY_{kind}: [{item}; {len(values)}] = [")
            for value in values:
                if kind == "PRODUCTS":
                    print(f"    (\n        {rust(value[0])},\n        {rust(value[1])},\n    ),")
                else:
                    print(f"    {rust(value)},")
            print("];")


if __name__ == "__main__":
    main()
