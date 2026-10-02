# Searches for inputs on which the constant-time inversion's half-delta
# divsteps run long, for the modulus f = p of each Pasta field, and prints
# them in the exact shape of the `slow_inversions` known answers in
# `src/test_fields.rs`, then as the Lean literals of
# `lean/PastaCurves/Inversion/LowerBound.lean`.
#
# Run from this directory:
#
#     uv run sage divstep_lower_bounds.sage
#
# The script is also plain Python (`python3 divstep_lower_bounds.sage`): it
# uses Python integers and a seeded `random.Random` only, so its output is
# deterministic and does not depend on the SageMath version. It takes a few
# seconds.
#
# The inversion runs 590 half-delta divsteps from (delta, f, g) =
# (1/2, p, x), the bound that Bernstein, Chen, Harrison, Huang, Maxwell,
# Wang, Wuille, and Yang prove for all 256-bit inputs ("Accelerating and
# verifying constant-time modular inversion", EUROCRYPT 2026, Theorem 1).
# For 255-bit inputs, as the Pasta moduli are, the same theorem gives 588.
# An input that needs n divsteps proves that no fixed count below n is
# correct for that modulus: a lower bound on the worst case, by witness.
# A search cannot give an upper bound; only the hull certificate does.
#
# Two searches:
#
# 1. Random restarts, then hill climbing on single-bit flips of g. This
#    finds the tail of the distribution of the step count on random inputs
#    (531 steps for Pallas, 530 for Vesta), and stalls there.
#
# 2. Grafting. The first n divsteps depend only on delta and the low n bits
#    of f and g, and the parity of g at step k is set by bit k of g given
#    the earlier bits. So for a fixed f, every sequence of n parities is the
#    trace of exactly one g mod 2^n. The search takes the parities of the
#    first n steps of the paper's 590-step witness pair (Section 4.4),
#    solves for the low n bits of g with f = p, and fills and hill-climbs
#    the high bits. The witness's slow start survives the change of f.

import random as pyrandom

MODULI = [
    ("FP", "pallas", int(0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001)),
    ("FQ", "vesta", int(0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001)),
]

# The 256-bit pair of the paper's Section 4.4, which needs exactly 590
# half-delta divsteps.
WITNESS_F = int(0xeec9f80577a885d22f8d37c1946187e26805ea27b26c5ae10aa38a02e2ea3157)
WITNESS_G = int(0xeb40350da50b11d23183ae8e88ffced0ad11263b6d62cde5e5dc1e934ef8229c)


def divstep(d, f, g):
    """One half-delta divstep on (d, f, g), with d = 2 delta, so integral."""
    if d > 0 and g & 1:
        return 2 - d, g, (g - f) >> 1
    return d + 2, f, (g + (g & 1) * f) >> 1


def steps(f, g):
    """The number of half-delta divsteps from (1/2, f, g) until g = 0."""
    d, n = 1, 0
    while g:
        d, f, g = divstep(d, f, g)
        n += 1
    return n


def flip(g, bit):
    return g - (1 << bit) if (g >> bit) & 1 else g + (1 << bit)


def hill_climb(rng, p, g, s, lo, hi):
    """Flips single bits of g in [lo, hi), in a random order, while that
    lengthens the run; returns the local maximum."""
    improved = True
    while improved:
        improved = False
        for bit in rng.sample(range(lo, hi), hi - lo):
            h = flip(g, bit)
            if 0 < h < p:
                t = steps(p, h)
                if t > s:
                    g, s, improved = h, t, True
    return s, g


def random_search():
    """Search 1: one generator for both fields, as first run."""
    rng = pyrandom.Random(int(1))
    found = {}
    for key, _, p in MODULI:
        best, best_g = 0, 0
        for _ in range(20000):
            g = rng.randrange(1, p)
            s = steps(p, g)
            if s > best:
                best, best_g = s, g
        found[key] = hill_climb(rng, p, best_g, best, 0, 254)
    return found


def parities(f, g, n):
    """The parities of g at the first n divsteps from (1/2, f, g)."""
    d, out = 1, []
    for _ in range(n):
        out.append(g & 1)
        d, f, g = divstep(d, f, g)
    return out


def g_for(f, par):
    """The g mod 2^n whose first n divsteps from (1/2, f, g) see the
    parities par: bit k is set so that g has parity par[k] at step k."""
    g = 0
    for k in range(len(par)):
        d, ff, gg = 1, f, g
        for _ in range(k):
            d, ff, gg = divstep(d, ff, gg)
        if (gg & 1) != par[k]:
            g |= 1 << k
    return g


def graft(p, n, seed):
    rng = pyrandom.Random(int(seed))
    low = g_for(p, parities(WITNESS_F, WITNESS_G, n))
    best, best_g = 0, 0
    for _ in range(400):
        g = low | (rng.getrandbits(int(255 - n)) << n)
        if 0 < g < p:
            s = steps(p, g)
            if s > best:
                best, best_g = s, g
    return hill_climb(rng, p, best_g, best, n, 255)


def graft_search():
    """Search 2, over graft lengths 196 to 248 and four seeds each."""
    found = {}
    for key, _, p in MODULI:
        s, _, g = max(
            (s, n, g)
            for n in range(196, 250, 2)
            for seed in range(4)
            for s, g in [graft(p, n, seed)]
        )
        found[key] = (s, g)
    return found


def limbs(x):
    return [(x >> (64 * i)) % 2**64 for i in range(4)]


def rust(key, rows):
    print(f"    // {key}")
    print("    slow_inversions: [")
    for s, g in rows:
        print("        (")
        print("            [")
        for limb in limbs(g):
            print(f"                0x{limb:016x},")
        print("            ],")
        print(f"            {s},")
        print("        ),")
    print("    ],")


def check(p, s, g):
    # The witness needs exactly s steps: g is nonzero after s - 1 of them.
    d, f, h = 1, p, g
    for _ in range(s - 1):
        d, f, h = divstep(d, f, h)
    assert 0 < g < p and h != 0 and divstep(d, f, h)[2] == 0


assert steps(WITNESS_F, WITNESS_G) == 590
first = random_search()
second = graft_search()
for key, name, p in MODULI:
    for s, g in (first[key], second[key]):
        check(p, s, g)
    rust(key, [first[key], second[key]])
print()
for key, name, p in MODULI:
    s, g = second[key]
    print(f"-- {name}: {s} steps")
    print(f"0x{g:064x}")
