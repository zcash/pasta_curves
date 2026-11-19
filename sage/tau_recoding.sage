# Derivation and verification of regular tau-adic recoding of Pasta GLV halves.
# Run from sage/: uv run --locked sage tau_recoding.sage
# The digit set and fixed-length method are from the paper's Sections 4 and 7;
# all constants and residue choices below are derived here without implementation sources.

from math import isqrt

# Fixed 16-bit long-division chunks used by Signed192. Derive each reciprocal
# and exhaust the complete dividend range of one chunk, not just random inputs.
for divisor in (3, 9):
    reciprocal = (2**33 + divisor - 1) // divisor
    assert reciprocal == {3: 0xaaaaaaab, 9: 0x38e38e39}[divisor]
    assert all((x * reciprocal) // 2**33 == x // divisor
               for x in range(divisor * 2**16))
    print('division by %s: reciprocal = 0x%x, shift = 33; all chunks verified' %
          (divisor, reciprocal))
from random import Random

p = 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001
q = 0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001
Fp, Fq = GF(p), GF(q)
Pallas, Vesta = EllipticCurve(Fp, [0, 5]), EllipticCurve(Fq, [0, 5])
Gp, Gv = Pallas(-1, 2), Vesta(-1, 2)
# The Pasta cycle gives the group orders; check these on the generators, without
# calling the (potentially expensive) general elliptic-curve order algorithm.
assert Gp != Pallas(0) and Gv != Vesta(0)
assert q * Gp == Pallas(0) and p * Gv == Vesta(0)
lam_pallas = int(Fq(5) ^ ((q - 1) // 3))
assert lam_pallas != 1 and (lam_pallas**2 + lam_pallas + 1) % q == 0
image = lam_pallas * Gp
beta_pallas = Fp(image[0]) / Fp(Gp[0])
lam_vesta = int(beta_pallas)
assert beta_pallas != 1 and beta_pallas**3 == 1
assert image == Pallas(beta_pallas * Gp[0], Gp[1])
assert lam_vesta * Gv == Vesta(Fq(lam_pallas) * Gv[0], Gv[1])

# tau = 1 - omega and tau^2 = 3*tau - 3. Pairs represent a + b*tau.
ZERO, ONE, TAU = (0, 0), (1, 0), (0, 1)
def add(x, y):
    return x[0] + y[0], x[1] + y[1]
def sub(x, y):
    return x[0] - y[0], x[1] - y[1]
def mul(x, y):
    a, b = x
    c, d = y
    return a*c - 3*b*d, a*d + b*c + 3*b*d
def norm(x):
    a, b = x
    return a*a + 3*a*b + 3*b*b

def div_tau(x):
    a, b = x
    assert a % 3 == 0
    return a+b, -a//3

def div_tau3(x):
    a, b = x
    assert a % 9 == 0 and b % 3 == 0
    y = div_tau(div_tau(div_tau(x)))
    assert y == ((a + 2*b)//3, -(2*a + 3*b)//9)
    assert 27*norm(y) == norm(x)
    return y

assert mul(TAU, TAU) == (-3, 3)
TAU3 = mul(mul(TAU, TAU), TAU)
assert TAU3 == (-9, 6) and norm(TAU3) == 27
OMEGA = sub(ONE, TAU)
ROTATIONS = [ONE, OMEGA, mul(OMEGA, OMEGA)]
assert mul(ROTATIONS[-1], OMEGA) == ONE
UNITS = {mul((s, 0), u) for s in (1, -1) for u in ROTATIONS}
assert len(UNITS) == 6 and all(norm(u) == 1 for u in UNITS)
# N(a+b*tau) = (a+b)^2 + (a+b)*(-b) + (-b)^2.
# It is 0 only at (0,0), since 4*N=(2*a+3*b)^2+3*b^2.
# N=2 is impossible because N mod 3 = a^2 mod 3 is either 0 or 1.
assert all(norm((a,b)) % 3 in (0,1) for a in range(-2,3) for b in range(-2,3))
assert { (a,b) for a in range(-2,3) for b in range(-2,3) if norm((a,b)) == 1 } == UNITS

BASES = [(1, 0), (2, 0), (1, 1), (1, -2), (2, 1), (1, 2)]
# Metadata: representative index, omega rotation (0..2), sign bit.
DIGIT_METADATA = {}
for index, base in enumerate(BASES):
    for rotation, unit in enumerate(ROTATIONS):
        for negative in (0, 1):
            digit = mul(mul((-1 if negative else 1, 0), unit), base)
            assert digit not in DIGIT_METADATA
            DIGIT_METADATA[digit] = (index, rotation, negative)
DIGITS = sorted(DIGIT_METADATA)
assert len(DIGITS) == 36 and {norm(d) for d in DIGITS} == {1, 4, 7, 13, 19}
assert max(map(norm, DIGITS)) == 19

def residue(x):
    return x[0] % 9, x[1] % 3

CLASSES = {(a, b): [] for a in range(9) if a % 3 for b in range(3)}
for d in DIGITS:
    assert residue(d) in CLASSES
    CLASSES[residue(d)].append(d)
assert len(CLASSES) == 18 and all(len(ds) == 2 for ds in CLASSES.values())

# The quotient's first coefficient is (a-d.a+2*(b-d.b))/3. Its residue
# modulo 3 depends only on (a mod 9,b mod 9), not on the full input.
# Choose the lexicographically first digit of smallest norm among candidates
# whose quotient is nonzero modulo tau. Exhaust every one of the 54 keys.
LUT = {}
for a9 in range(9):
    if a9 % 3 == 0:
        continue
    for b9 in range(9):
        candidates = CLASSES[(a9, b9 % 3)]
        valid = [d for d in candidates if div_tau3((a9-d[0], b9-d[1]))[0] % 3]
        assert valid, (a9, b9, candidates)
        chosen = min(valid, key=lambda d: (norm(d), d))
        LUT[(a9, b9)] = chosen
        assert residue(chosen) == (a9, b9 % 3)
        assert div_tau3((a9-chosen[0], b9-chosen[1]))[0] % 3
assert len(LUT) == 54
print('Digit table: each row has b mod 9 = 0..8; columns are (representative, omega rotation, negative)')
for a9 in (1, 2, 4, 5, 7, 8):
    print('a mod 9 = %s: %s' % (a9, [DIGIT_METADATA[LUT[(a9,b9)]] for b9 in range(9)]))
print('36 digits, 18 residue classes, 54 verified selection keys; maximum digit norm 19')
print('Maximum absolute digit coefficients:', tuple(max(abs(d[i]) for d in DIGITS) for i in range(2)))

# For x nondivisible by tau, the LUT makes (x-d)/tau^3 nondivisible by
# tau, so it cannot be zero. The triangle inequality, verified via the
# integer norm identity below, gives sqrt(N(next)) <=
# (sqrt(N(x))+sqrt(19))/sqrt(27). This also holds when x is a unit:
# (1+sqrt(19))^2/27 < 3, so the next nondivisible element is again a unit.
assert (1 + QQ(4360)/1000)**2 < 3*27
assert 4360**2 > 19*1000**2  # upper bound for sqrt(19)
assert 5196**2 < 27*1000**2  # lower bound for sqrt(27)
# For N(x)>=3, the conservative upper bound already implies N(next)<N(x):
# (sqrt(N)+sqrt(19))^2/27 < N, first at N=3 and then for all larger N.
# In rational arithmetic sqrt(19) < 4360/1000 < (5196/1000-1)*sqrt(3).
assert (QQ(4360)/1000)**2 < 3*(QQ(5196)/1000-1)**2

def step(cur):
    digit = LUT[(cur[0] % 9, cur[1] % 9)]
    delta = sub(cur, digit)
    nxt = div_tau3(delta)
    assert nxt[0] % 3
    nx, nd = norm(cur), norm(digit)
    # Cauchy-Schwarz in Z[omega]: |N(x-d)-N(x)-N(d)| <= 2 sqrt(N(x)N(d)).
    assert (norm(delta)-nx-nd)**2 <= 4*nx*nd
    if nx == 1:
        assert nxt in UNITS
    return digit, nxt

def recode(halves, steps):
    k1, k2 = halves
    start = (k1 + k2, -k2)
    correction = int(start[0] % 3 == 0)
    cur = add(start, (correction, 0))
    assert cur[0] % 3
    out = []
    for _ in range(steps):
        digit, cur = step(cur)
        out.append(digit)
    assert cur in UNITS, ('top not a unit', cur, halves)
    reconstructed = cur
    for digit in reversed(out):
        reconstructed = add(mul(TAU3, reconstructed), digit)
    assert reconstructed == add(start, (correction, 0))
    return out, cur, correction

# Reproduce the crate's Gauss-reduced basis and 384-bit Babai coefficients.
def iround(a, b):
    return (2*a+b) // (2*b)
def norm_plain(v):
    return v[0]*v[0]+v[1]*v[1]
def basis(n, lam):
    u, v = (n, 0), ((-lam) % n, 1)
    while True:
        if norm_plain(u) > norm_plain(v):
            u, v = v, u
        m = iround(u[0]*v[0] + u[1]*v[1], norm_plain(u))
        if m == 0:
            break
        v = (v[0]-m*u[0], v[1]-m*u[1])
    u = u if u[0] > 0 else (-u[0], -u[1])
    v = v if v[0] > 0 else (-v[0], -v[1])
    v1, v2 = (u, v) if u[1] < 0 else (v, u)
    assert v1[0]*v2[1]-v1[1]*v2[0] == n
    assert (v1[0]+v1[1]*lam) % n == (v2[0]+v2[1]*lam) % n == 0
    assert v1[0] > 0 and v1[1] < 0 and v2[0] > 0 and v2[1] > 0
    return v1, v2

def decompose(k, n, v1, v2, g1, g2):
    c1 = iround(g1*k, 2**384)
    c2 = iround(g2*k, 2**384)
    return k-c1*v1[0]-c2*v2[0], -c1*v1[1]-c2*v2[1]

# For 0 <= k < n, each rounded Babai coefficient differs from its exact
# real value by at most 1/2 + (n-1)/2^385. Triangle inequality on the
# resulting halves gives the bounds used below (even at rounding ties).
def half_bounds(n, v1, v2, g1, g2):
    eps = QQ(1)/2 + QQ(n-1)/2**385
    assert abs(g1*n - 2**384*v2[1])*2 <= n
    assert abs(g2*n + 2**384*v1[1])*2 <= n
    b1 = (v1[0]+v2[0])//2 + 1
    b2 = (-v1[1]+v2[1])//2 + 1
    assert eps*(v1[0]+v2[0]) < b1
    assert eps*(-v1[1]+v2[1]) < b2
    return b1, b2

# N(k1+k2-k2*tau) = k1^2+k1*k2+k2^2. For |ki|<=bi its
# maximum is b1^2+b1*b2+b2^2; the initial correction raises its
# square-root radius by at most 1. Maintain an outward rational radius
# R_{j+1} = (R_j + 4360/1000)/(5196/1000). Once R^2<3, the
# nondivisible next state must be one of the six norm-1 units.
def fixed_rounds(b1, b2):
    n0 = b1*b1+b1*b2+b2*b2
    radius = QQ(isqrt(int(n0))+2)
    assert radius > isqrt(int(n0))+1
    rounds = 0
    while radius*radius >= 3:
        radius = (1000*radius+4360)/5196
        rounds += 1
        assert rounds < 100
    return rounds, radius, n0

for name, E, G, n, lam in [('Pallas', Pallas, Gp, q, lam_pallas),
                             ('Vesta', Vesta, Gv, p, lam_vesta)]:
    v1, v2 = basis(n, lam)
    g1 = iround(2**384*v2[1], n)
    g2 = iround(-2**384*v1[1], n)
    assert max(g1,g2) < 2**320
    b1, b2 = half_bounds(n, v1, v2, g1, g2)
    assert b1 < 2**127 and b2 < 2**127
    rounds, radius, n0 = fixed_rounds(b1, b2)
    assert rounds == 54  # Matches the fixed-length Rust recoder.
    # Both N(x)<R0^2 and N(next)<R_next^2; for any pair,
    # 4*N(x)-a^2 = 3*(a+2*b)^2 and 4*N(x)-3*b^2 = (2*a+3*b)^2.
    r0 = isqrt(int(n0))+2
    coordinate_bound = 2*r0
    assert coordinate_bound < 2**129
    # Subtraction of a digit and the numerator 2*a+3*b used by div_tau3
    # also fit in signed 192-bit arithmetic throughout the recoding.
    assert 5*(coordinate_bound+max(max(abs(x) for x in d) for d in DIGITS)) < 2**191
    print('%s: |k1| <= %s; |k2| <= %s (384-bit rounded Babai)' % (name,b1,b2))
    print('%s: %s rounds, final radius < %.6f < sqrt(3); |a|,|b| < 2^129 (signed 130-bit coordinates)' %
          (name, rounds, float(radius)))
    print('%s: initial |a| <= %s, |b| <= %s; signed 192-bit intermediates suffice' %
          (name,b1+b2+1,b2))
    vectors = [0, 1, 2, 3, n-1, n//2, n//3, (1<<127)-1, 1<<127]
    vectors += [int((0x9E3779B97F4A7C15+i)**3+0x123456789ABCDEF) % n for i in range(2500)]
    rng = Random(int(0xA17D1C))
    vectors += [rng.randrange(int(n)) for _ in range(2500)]
    counts = {0: 0, 1: 0}
    for k in vectors:
        halves = decompose(k, n, v1, v2, g1, g2)
        k1, k2 = halves
        assert abs(k1) <= b1 and abs(k2) <= b2
        assert (k1+k2*lam-k) % n == 0
        ds, top, c = recode(halves, rounds)
        counts[c] += 1
        tau_mod = (1-lam) % n
        acc = (top[0]+top[1]*tau_mod) % n
        for d in reversed(ds):
            acc = (pow(tau_mod, 3, n)*acc+d[0]+d[1]*tau_mod) % n
        assert (acc-c-k) % n == 0
    assert pow(1-lam,2,n) == (-3*lam) % n
    assert pow(1-lam,3,n) == (6*(1-lam)-9) % n
    print('%s: %s scalar reconstructions, corrections %s; %s nonzero base-tau^3 digits and top unit' %
          (name, len(vectors), counts, rounds))

# Check small states, including each unit followed by further fixed steps.
for a in range(-40, 41):
    for b in range(-40, 41):
        if a % 3:
            cur = (a, b)
            for _ in range(7):
                old = norm(cur)
                _, cur = step(cur)
                if old > 1:
                    assert norm(cur) < old
print('Small-state norm descent, unit preservation, and nonzero-quotient checks passed')
