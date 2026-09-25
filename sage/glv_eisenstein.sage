# Verifies every parameter of the Eisenstein-integer GLV recoding in
# `src/glv_eisenstein.rs`, from the curve definitions alone, and prints the
# ones that appear literally in the Rust source so the two can be diffed.
#
# Run from this directory:
#
#     uv run sage glv_eisenstein.sage
#
# What is checked, in order:
#
#   1. The ring Z[w] = Z[X]/(X^2 + X + 1): that it is the ring of integers of
#      Q(sqrt(-3)), that 2 is INERT in it (so "even" means both coordinates
#      even and halving is exact), and that its unit group is mu_6.
#   2. That mu_6 acts FREELY on the 48 odd residue classes of Z[w]/8, so the
#      classes fall into exactly 48/6 = 8 orbits.
#   3. That the eight representatives `REPS` in the Rust source hit those
#      eight orbits, i.e. their 48 unit multiples are exactly the odd classes.
#   4. That the addition chain in `Table::window_proj` reaches those eight
#      representatives in seven additions -- symbolically in Z[w], and then as
#      real point arithmetic on Pallas and Vesta.
#   5. That `MAX_DIGITS = 130` is a true upper bound: the digit coefficients
#      are bounded by 5, the resulting contraction reaches the box
#      max(|a|,|b|) <= 12 in 124 columns, and every state in that box is
#      exhaustively verified to drain within 6 more.
#   6. The recoder end to end: on real scalars it reconstructs k modulo the
#      group order, and the resulting ladder computes k*P on the real curves.
#   7. The measured cost model quoted in the module docs (~38.4 additions over
#      ~126 columns, a density of 3/10).
#
# Everything is exact integer arithmetic plus Sage's own curve and number
# field machinery, so the output does not depend on the SageMath version.

import itertools

# The Pasta base/scalar field moduli. Fp is the Pallas base field and the
# Vesta scalar field; Fq is the Pallas scalar field and the Vesta base field.
p = 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001
q = 0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001

Fp = GF(p)
Fq = GF(q)
Pallas = EllipticCurve(Fp, [0, 5])
Vesta = EllipticCurve(Fq, [0, 5])
assert Pallas.order() == q and Vesta.order() == p
Gp = Pallas(-1, 2)
Gv = Vesta(-1, 2)

# Fq::ZETA / Fp::ZETA exactly as `src/fields/{fq,fp}.rs` define them: the
# element 5^((n-1)/3) of multiplicative order three. The base-field zeta is
# the one `CurveExt::endo` multiplies x by, pinned to the scalar-field zeta by
# the requirement phi(P) = lambda * P.
zeta_q = Fq(5) ^ ((q - 1) // 3)
zeta_p = Fp(5) ^ ((p - 1) // 3)
zeta_p_base = Fp((int(zeta_q) * Gp)[0]) / Fp(Gp[0])
zeta_q_base = Fq((int(zeta_p) * Gv)[0]) / Fq(Gv[0])


# ---------------------------------------------------------------------------
# 1. The ring Z[w], its inert 2, and its unit group mu_6
# ---------------------------------------------------------------------------

K.<w> = NumberField(x ^ 2 + x + 1)
OK = K.ring_of_integers()
assert w ^ 3 == 1 and w != 1, "w must be a primitive cube root of unity"
assert K.discriminant() == -3, "Q(w) is Q(sqrt(-3))"
assert K.order(w) == OK, "Z[w] must be the full ring of integers"

# 2 is INERT: it stays prime, and the quotient is the field F_4. This is what
# makes "even" mean both coordinates even, halving exact and coordinate-wise,
# and the whole width-w NAF theory carry over with 2^w read in the ring.
factors = K.ideal(2).factor()
assert len(factors) == 1 and factors[0][1] == 1, "2 must be inert in Z[w]"
assert K.ideal(2).residue_field().cardinality() == 4, "Z[w]/2 must be F_4"

# The unit group is mu_6 = {+-1, +-w, +-w^2}: exactly six elements, exactly the
# elements of norm one, and exactly the six automorphisms (x,y) -> (z^i x, +-y)
# of a j-invariant-0 curve.
UNITS = [K(1), w, w ^ 2, K(-1), -w, -(w ^ 2)]
assert len(set(UNITS)) == 6
assert all(u.norm() == 1 for u in UNITS)
assert K.unit_group().order() == 6, "unit group must be mu_6"
assert all(u * v in UNITS for u in UNITS for v in UNITS), "mu_6 must be closed"
print("[1] Z[w]: ring of integers of Q(sqrt(-3)), 2 inert (Z[w]/2 = F_4), units = mu_6")


# ---------------------------------------------------------------------------
# 2-3. The 48 odd classes mod 8, the free mu_6 action, and the eight reps
# ---------------------------------------------------------------------------

# Coefficient-pair arithmetic, matching the Rust representation exactly:
# (a, b) means a + b*w, and w^2 = -1 - w.
def emul(u, v):
    a, b = u
    c, d = v
    return (a * c - b * d, a * d + b * c - b * d)


def enorm(u):
    a, b = u
    return a * a - a * b + b * b


# `rotate` in the Rust source: (a + b w) * w = -b + (a - b) w.
def rotate(u):
    return (-u[1], u[0] - u[1])


UNIT_PAIRS = [(1, 0), (0, 1), (-1, -1), (-1, 0), (0, -1), (1, 1)]
assert [K(u[0]) + K(u[1]) * w for u in UNIT_PAIRS] == UNITS, "pair model of mu_6"
assert rotate((1, 0)) == (0, 1) and rotate(rotate(rotate((1, 0)))) == (1, 0)

M = 8  # the modulus 2^w for width w = 3


def red(u):
    return (u[0] % M, u[1] % M)


classes = [(a, b) for a in range(M) for b in range(M)]
odd = [c for c in classes if not (c[0] % 2 == 0 and c[1] % 2 == 0)]
assert len(classes) == 64 and len(odd) == 48

# FREENESS: every orbit has the full six elements. This is the fact that turns
# 48 digits into 8 stored points, and it is what makes the Rust table small.
for c in odd:
    assert len({red(emul(u, c)) for u in UNIT_PAIRS}) == 6, "action not free at %s" % (c,)
assert len(odd) // 6 == 8
print("[2] mu_6 acts freely on the 48 odd classes of Z[w]/8 -> exactly 8 orbits")

# The eight orbit representatives, as they appear in `REPS` in the Rust source.
REPS = [(1, 0), (1, -1), (2, -1), (1, -2), (3, 0), (3, -1), (1, -3), (2, -3)]

covered = {}
for i, r in enumerate(REPS):
    for j, u in enumerate(UNIT_PAIRS):
        c = red(emul(u, r))
        assert c not in covered, "collision: rep %d and rep %d" % (i, covered[c][0])
        covered[c] = (i, j)
assert set(covered) == set(odd), "REPS do not cover the odd classes exactly"
print("[3] the 8 reps x 6 units are exactly the 48 odd classes; norms %s"
      % [enorm(r) for r in REPS])

print()
print("// src/glv_eisenstein.rs")
print("const REPS: [(i8, i8); REP_COUNT] = [")
for r in REPS:
    print("    (%d, %d)," % r)
print("];")
print()


# ---------------------------------------------------------------------------
# 4. The addition chain of `Table::window_proj`
# ---------------------------------------------------------------------------

def window_chain(P, add, sub, neg, phi):
    """The chain in `Table::window_proj`, over any model of the mu_6 action.

    Seven additions (`add`/`sub`), plus applications of phi, which is one
    base-field multiplication on a curve point and free in Z[w]."""
    phi_p = phi(P)
    d1 = sub(P, phi_p)                    # add 1
    b = sub(d1, phi(d1))                  # add 2
    phi_b = phi(b)
    m3 = phi(phi_b)
    r3 = neg(phi_b)
    t3a = add(m3, phi_p)                  # add 3
    t3b = sub(phi_p, m3)                  # add 4
    t4a = add(phi_p, r3)                  # add 5
    t4b = sub(phi_p, r3)                  # add 6
    t19 = add(t4b, phi_p)                 # add 7
    # The eight representatives, plus the intermediates whose values the Rust
    # doc comment on `window_proj` claims.
    return [P, d1, phi(t4a), neg(phi(t3b)), neg(m3), neg(t3a),
            phi(phi(t4b)), phi(phi(t19))], (b, m3, r3)


# (a) symbolically in Z[w].
built, (b_, m3_, r3_) = window_chain(
    (1, 0),
    lambda u, v: (u[0] + v[0], u[1] + v[1]),
    lambda u, v: (u[0] - v[0], u[1] - v[1]),
    lambda u: (-u[0], -u[1]),
    rotate,
)
assert b_ == (0, -3), "b must be -3w"
assert m3_ == (-3, 0), "m3 must be -3"
assert r3_ == emul((3, 0), (-1, -1)), "r3 must be 3w^2"
assert built == REPS, "addition chain does not reach REPS"
print("[4a] the 7-addition chain reaches REPS symbolically in Z[w]")

# (b) as real point arithmetic, which additionally pins phi = multiplication
# by lambda and the base-field zeta that `CurveExt::endo` uses.
def check_curve(name, E, G, n, lam, zeta_base):
    def phi(P):
        if P.is_zero():
            return P
        return E(zeta_base * P[0], P[1])

    assert (lam * lam + lam + 1) % n == 0, "lambda must satisfy x^2+x+1"
    P = int(0xDEADBEEF) * G
    assert phi(P) == int(lam) * P, "phi(P) must be lambda * P on %s" % name

    pts, _ = window_chain(P, lambda A, B: A + B, lambda A, B: A - B,
                          lambda A: -A, phi)
    for r, R in zip(REPS, pts):
        assert R == (int(r[0]) + int(r[1]) * int(lam)) * P, \
            "%s: window entry %s is the wrong multiple" % (name, (r,))
    return phi


lam_pallas = int(zeta_q)   # Pallas scalars live in Fq
lam_vesta = int(zeta_p)    # Vesta scalars live in Fp
phi_pallas = check_curve("Pallas", Pallas, Gp, q, lam_pallas, zeta_p_base)
phi_vesta = check_curve("Vesta", Vesta, Gv, p, lam_vesta, zeta_q_base)
print("[4b] the same chain reaches the same multiples on Pallas and Vesta,")
print("     with phi(P) = lambda*P for lambda = ZETA and the base-field zeta")


# --- 4c. why the chain is safe in AFFINE coordinates --------------------------
#
# Table::batch runs that chain with affine additions, which fail when the two
# operands share an x. Every step adds u*P to v*P for Eisenstein u, v of norm at
# most 19, so a failure needs (u -+ v)P = O with u -+ v a nonzero element of tiny
# norm, which forces P = O. The sharp case is the first step, P - phi(P), whose
# denominator is (zeta - 1)x: it fails exactly at x = 0. A curve point with
# x = 0 would have phi(P) = P, hence (1 - w)P = O, hence be 3-torsion, and both
# groups have prime order prime to 3. So no such point exists, which is to say
# the curve constant b = 5 is a non-residue. Check all of that directly.

def check_affine_chain_safety(name, E, n, F):
    assert n % 3 != 0, "%s: group order must be prime to 3" % name
    assert not F(5).is_square(), "%s: a point with x = 0 would exist" % name
    assert E.count_points() == n

    # The differences u -+ v that each step's denominator is built from. A
    # failure would need one of them to annihilate P.
    w = (0, 1)
    phi_p, m3, r3 = w, (-3, 0), emul((3, 0), (-1, -1))
    steps = [
        ("d1 = P - phi(P)", (1, 0), phi_p),
        ("b  = d1 - phi(d1)", (1, -1), emul(w, (1, -1))),
        ("t3a = m3 + phi(P)", m3, phi_p),
        ("t3b = phi(P) - m3", phi_p, m3),
        ("t4a = phi(P) + r3", phi_p, r3),
        ("t4b = phi(P) - r3", phi_p, r3),
        ("t19 = t4b + phi(P)", (3, 4), phi_p),
    ]
    worst = 0
    for label, u, v in steps:
        for d in [(u[0] - v[0], u[1] - v[1]), (u[0] + v[0], u[1] + v[1])]:
            assert d != (0, 0), "%s: %s degenerates identically" % (name, label)
            worst = max(worst, enorm(d))
            # A nonzero Eisenstein integer of norm far below n cannot map to 0
            # in Z/n: its norm would have to be divisible by the prime n.
            assert enorm(d) % n != 0, "%s: %s can annihilate P" % (name, label)
    return worst


worst_p = check_affine_chain_safety("Pallas", Pallas, q, Fp)
worst_v = check_affine_chain_safety("Vesta", Vesta, p, Fq)
print("[4c] the affine chain has no exceptional case but P = O: b is a")
print("     non-residue (so no point has x = 0), the order is prime to 3,")
print("     and every step's operand difference has norm at most %d"
      % max(worst_p, worst_v))


# ---------------------------------------------------------------------------
# 5. MAX_DIGITS = 130
# ---------------------------------------------------------------------------

# The digit assigned to each odd class: the canonical orbit representative,
# rotated and negated. This is exactly what the Rust `LUT` holds.
DIGIT = {}
for i, r in enumerate(REPS):
    d = r
    for rot in range(3):
        for neg in (False, True):
            e = (-d[0], -d[1]) if neg else d
            DIGIT[red(e)] = e
        d = rotate(d)
assert len(DIGIT) == 48

D = max(max(abs(d[0]), abs(d[1])) for d in DIGIT.values())
assert D == 5, "digit coefficient bound changed"


def step(a, b):
    """One recoder column: halve, or subtract the digit and halve."""
    if a % 2 == 0 and b % 2 == 0:
        return a // 2, b // 2, None
    d = DIGIT[(a % M, b % M)]
    a -= d[0]
    b -= d[1]
    assert a % 8 == 0 and b % 8 == 0, "a digit must clear three bits"
    return a // 2, b // 2, d


# The box max(|a|,|b|) <= D is absorbing, since |x'| <= (|x| + D)/2.
for a in range(-D, D + 1):
    for b in range(-D, D + 1):
        a2, b2, _ = step(a, b)
        assert max(abs(a2), abs(b2)) <= D, "box |.| <= %d not absorbing" % D

# Exhaustively drain the larger box the Rust bound is stated over.
BOX = 12
remaining = {}


def drain(a, b, seen=()):
    if (a, b) == (0, 0):
        return 0
    if (a, b) in remaining:
        return remaining[(a, b)]
    assert (a, b) not in seen, "recoder cycles at %s" % ((a, b),)
    a2, b2, _ = step(a, b)
    r = 1 + drain(a2, b2, seen + ((a, b),))
    remaining[(a, b)] = r
    return r


TAIL = max(drain(a, b) for a in range(-BOX, BOX + 1) for b in range(-BOX, BOX + 1))
assert TAIL == 6, "tail length changed"

# |k1|, |k2| < 2^127, and |x'| <= floor((|x| + D)/2) each column.
m = 2 ^ 127 - 1
columns = 0
while m > BOX:
    m = (m + D) // 2
    columns += 1
MAX_DIGITS = columns + TAIL
assert MAX_DIGITS == 130, "MAX_DIGITS changed"
print("[5] digits bounded by %d; %d columns to reach the box |.| <= %d, which "
      "drains in %d more" % (D, columns, BOX, TAIL))
print("    => const MAX_DIGITS: usize = %d;" % MAX_DIGITS)


# ---------------------------------------------------------------------------
# 6-7. The recoder end to end, on real scalars and real curves
# ---------------------------------------------------------------------------

def iround(a, b):
    return (2 * a + b) // (2 * b)


def lagrange_gauss(u, v):
    while True:
        if u[0] ^ 2 + u[1] ^ 2 > v[0] ^ 2 + v[1] ^ 2:
            u, v = v, u
        m = iround(u[0] * v[0] + u[1] * v[1], u[0] ^ 2 + u[1] ^ 2)
        if m == 0:
            return u, v
        v = (v[0] - m * u[0], v[1] - m * u[1])


def short_basis(n, lam):
    """The same short basis `glv_constants.sage` derives for `GlvParams`."""
    u, v = lagrange_gauss((n, 0), ((-lam) % n, 1))
    u = u if u[0] > 0 else (-u[0], -u[1])
    v = v if v[0] > 0 else (-v[0], -v[1])
    return (u, v) if u[1] < 0 else (v, u)


def decompose(k, n, v1, v2):
    """Babai rounding, matching `decompose` in `src/glv.rs`."""
    det = v1[0] * v2[1] - v1[1] * v2[0]
    c1 = iround(v2[1] * k, det)
    c2 = iround(-v1[1] * k, det)
    return (k - c1 * v1[0] - c2 * v2[0], -c1 * v1[1] - c2 * v2[1])


def recode(a, b):
    out = []
    while a or b:
        a, b, d = step(a, b)
        out.append(d)
    return out


def check_recoder(name, E, G, n, lam):
    v1, v2 = short_basis(n, lam)
    # The Babai bound that keeps a decomposition half inside i128 even after a
    # digit is subtracted: |k1| <= (|v1.a| + |v2.a|)/2, likewise for b.
    for bound in [(abs(v1[0]) + abs(v2[0])) // 2 + 1,
                  (abs(v1[1]) + abs(v2[1])) // 2 + 1]:
        assert bound < 2 ^ 127 - D, "%s: GLV half can overflow i128" % name

    total_cols = total_adds = 0
    max_cols = max_adds = 0
    trials = 2000
    for i in range(trials):
        k = (int(0x9E3779B97F4A7C15 + i) ^ 2 + int(0x0123456789ABCDEF)) ^ 2 + i
        k %= n
        k1, k2 = decompose(k, n, v1, v2)
        assert abs(k1) < 2 ^ 127 and abs(k2) < 2 ^ 127
        assert (k1 + k2 * lam - k) % n == 0, "%s: decomposition lost k" % name

        digits = recode(k1, k2)
        assert len(digits) <= MAX_DIGITS, "%s: recoding exceeded MAX_DIGITS" % name

        # The digit string evaluates back to k, Horner-wise.
        acc = 0
        for d in reversed(digits):
            acc = 2 * acc
            if d is not None:
                acc += d[0] + d[1] * lam
        assert (acc - k) % n == 0, "%s: digit string does not evaluate to k" % name

        total_cols += len(digits)
        total_adds += sum(1 for d in digits if d is not None)
        max_cols = max(max_cols, len(digits))
        max_adds = max(max_adds, sum(1 for d in digits if d is not None))

    # And the ladder those digits drive really does compute k*P on the curve.
    for i in range(8):
        k = int(0x9E3779B97F4A7C15 + i) ^ 3 % n
        P = int(0xC0FFEE + i) * G
        acc = E(0)
        for d in reversed(recode(*decompose(k, n, v1, v2))):
            acc = 2 * acc
            if d is not None:
                acc = acc + (int(d[0]) + int(d[1]) * int(lam)) * P
        assert acc == int(k) * P, "%s: Eisenstein ladder != k*P" % name

    print("[6] %-6s ladder reproduces k*P; columns mean %.2f max %d; "
          "additions mean %.2f max %d"
          % (name, total_cols / trials, max_cols, total_adds / trials, max_adds))
    return total_adds / trials, total_cols / trials


adds_p, cols_p = check_recoder("Pallas", Pallas, Gp, q, lam_pallas)
adds_v, cols_v = check_recoder("Vesta", Vesta, Gv, p, lam_vesta)

# The cost model in the module docs: a nonzero digit forces three columns of
# divisibility, and the next is nonzero with probability 3/4, so the expected
# gap is 3 + (1/4)/(3/4) = 10/3 and the density is 3/10.
for adds, cols in [(adds_p, cols_p), (adds_v, cols_v)]:
    assert abs(adds / cols - 3 / 10) < 0.01, "density is off the 3/10 model"
    assert 37.5 < adds < 39.5, "mean additions off the documented ~38.4"
print("[7] density %.4f matches the predicted 3/10; mean additions ~%.1f "
      "against ~51.2 for the split wNAF-4 ladder in src/glv.rs"
      % (adds_p / cols_p, adds_p))
print()
print("all Eisenstein GLV parameters verified")
