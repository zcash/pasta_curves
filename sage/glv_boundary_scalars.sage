# Constructs the Babai-boundary witness scalars used by the
# `babai_boundary_*` / `native_vs_glv_boundary_*` regression tests in
# `src/glv.rs`, and prints them in the exact shape of the Rust source.
#
# Run from this directory:
#
#     uv run sage glv_boundary_scalars.sage
#
# The witnesses defend the Babai coefficient G2 against a corruption the
# rest of the suite provably cannot see: flipping bit 127 of G2 leaves
# the known-answer tests unmoved and shifts c2 = round(G2*k / 2^384) for
# only a small fraction of random scalars — and even where it does, the
# decomposition still reconstructs k; the observable failure is a
# decomposition half escaping its 2^127 bound. Each witness k is built so
# that, with the shipped constants, it decomposes like any other scalar,
# while under the bit flip c2 moves by one and |k2| = |c1*V1B_NEG -
# c2*V2B| lands in [2^127, 2^128).
#
# That places two joint conditions on the residues G1*k mod 2^384 and
# G2*k mod 2^384 (details below), each carving out a window a few bits
# wide, so a random search is wasteful and unauditable. Instead the
# witness is the Babai-nearest lattice point to an explicit target in the
# rank-3 lattice of joint residues — exact rational arithmetic and a
# hand-rolled reduction throughout, so the output does not depend on the
# SageMath version.

# --- The GLV constants, as derived in glv_constants.sage --------------------

p = 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001
q = 0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001

Fp = GF(p)
Fq = GF(q)
Pallas = EllipticCurve(Fp, [0, 5])
Vesta = EllipticCurve(Fq, [0, 5])
Gp = Pallas(-1, 2)

def iround(a, b):
    return (2 * a + b) // (2 * b)


zeta_q = Fq(5) ^ ((q - 1) // 3)
zeta_p = Fp((int(zeta_q) * Gp)[0]) / Fp(Gp[0])


def lagrange_gauss(u, v):
    while True:
        if u[0] ^ 2 + u[1] ^ 2 > v[0] ^ 2 + v[1] ^ 2:
            u, v = v, u
        m = iround(u[0] * v[0] + u[1] * v[1], u[0] ^ 2 + u[1] ^ 2)
        if m == 0:
            return u, v
        v = (v[0] - m * u[0], v[1] - m * u[1])


def glv_params(n, lam):
    u, v = lagrange_gauss((n, 0), ((-lam) % n, 1))
    u = u if u[0] > 0 else (-u[0], -u[1])
    v = v if v[0] > 0 else (-v[0], -v[1])
    if u[1] > 0:
        u, v = v, u
    (V1A, V1B_NEG), (V2A, V2B) = (u[0], -u[1]), v
    assert V1A * V2B + V1B_NEG * V2A == n
    return V1A, V1B_NEG, V2A, V2B, iround(2 ^ 384 * V2B, n), iround(2 ^ 384 * V1B_NEG, n)


# --- Babai's nearest-plane algorithm ----------------------------------------
#
# Textbook LLL (exact Gram-Schmidt over QQ) followed by nearest-plane
# rounding; on a rank-3 lattice both run instantly.

def gram_schmidt(B):
    Bs = [vector(QQ, b) for b in B]
    mu = [[QQ(0)] * len(B) for _ in B]
    for i in range(len(B)):
        for j in range(i):
            mu[i][j] = vector(QQ, B[i]).dot_product(Bs[j]) / Bs[j].dot_product(Bs[j])
            Bs[i] -= mu[i][j] * Bs[j]
    return Bs, mu


def lll(B):
    B = [vector(ZZ, b) for b in B]
    delta = QQ(99) / 100
    while True:
        Bs, mu = gram_schmidt(B)
        for i in range(1, len(B)):
            for j in range(i - 1, -1, -1):
                m = iround(mu[i][j].numerator(), mu[i][j].denominator())
                if m != 0:
                    B[i] -= m * B[j]
                    Bs, mu = gram_schmidt(B)
        swapped = False
        for i in range(len(B) - 1):
            l = Bs[i].dot_product(Bs[i])
            r = Bs[i + 1].dot_product(Bs[i + 1])
            if delta * l > r + mu[i + 1][i] ^ 2 * l:
                B[i], B[i + 1] = B[i + 1], B[i]
                swapped = True
                break
        if not swapped:
            return B


# The lattice vector nearest (in the Babai nearest-plane sense) to
# target `t` in the lattice spanned by the rows of `B`.
def babai_nearest(B, t):
    B = lll(B)
    Bs, _ = gram_schmidt(B)
    r = vector(QQ, t)
    for i in range(len(B) - 1, -1, -1):
        c = r.dot_product(Bs[i]) / Bs[i].dot_product(Bs[i])
        r -= iround(c.numerator(), c.denominator()) * vector(QQ, B[i])
    return vector(ZZ, t) - r


# --- The witness construction -----------------------------------------------
#
# Write f1 = frac(G1*k / 2^384) and f2 = frac(G2*k / 2^384), so that
# c1 = round(G1*k / 2^384) rounds down exactly when f1 < 1/2, and
# likewise c2. Flipping bit 127 of G2 (a zero bit on both curves) adds
# 2^127, growing G2*k / 2^384 by delta = k / 2^257; the witness needs:
#
#  * straddle: f2 in [1/2 - delta, 1/2), so the flip moves c2 up by one;
#  * escape: with c2 + 1, k2' = k2 - V2B must satisfy |k2'| >= 2^127,
#    i.e. k2 < V2B - 2^127 — and since k2 = f2*V2B - f1*V1B_NEG (for
#    f1, f2 < 1/2, up to a negligible fixed-point error), that needs f1
#    close below 1/2 while f2 sits close above its window's bottom;
#  * |k2'| < 2^128 and, with the shipped constants, |k1|, |k2| < 2^127:
#    automatic in this corner, and verified exactly below.
#
# Making delta (hence the f2 window) as large as possible wants k close
# below the group order n, so the target scalar is T = 24n/25, giving
# delta ~ 0.12. The joint residue targets then pin f1 just below 1/2 and
# f2 midway between the straddle bound 1/2 - delta and the escape bound
# ((V2B - 2^127) + f1*V1B_NEG) / V2B. A lattice point of
#
#     [ 2^176  G1     G2    ]           (2^176 balances the ~2^200 k-slack
#     [ 0      2^384  0     ]            against the ~2^376-wide residue
#     [ 0      0      2^384 ]            windows)
#
# near the scaled target (2^176*T, f1*2^384, f2*2^384) is exactly a
# scalar k near T whose residues G1*k, G2*k mod 2^384 land near those
# targets; Babai rounding on the reduced basis lands well inside both
# windows, and the conditions are then re-verified exactly.

def find_witness(n, lam):
    V1A, V1B_NEG, V2A, V2B, G1, G2 = glv_params(n, lam)
    assert (G2 >> 127) & 1 == 0  # the flip under test adds 2^127
    T = 24 * n // 25
    delta = QQ(T) / 2 ^ 257
    f1 = QQ(1) / 2 - QQ(1) / 2 ^ 10
    f2 = ((QQ(1) / 2 - delta) + QQ(V2B - 2 ^ 127 + f1 * V1B_NEG) / V2B) / 2
    assert QQ(1) / 2 - delta < f2 < QQ(1) / 2

    s = 2 ^ 176
    B = [[s, G1, G2], [0, 2 ^ 384, 0], [0, 0, 2 ^ 384]]
    t = [s * T, floor(f1 * 2 ^ 384), floor(f2 * 2 ^ 384)]
    k = ZZ(babai_nearest(B, t)[0] / s)

    # Exact re-verification of every property the Rust tests assert.
    assert 0 < k < n
    c1 = iround(G1 * k, 2 ^ 384)
    c2 = iround(G2 * k, 2 ^ 384)
    k1 = k - c1 * V1A - c2 * V2A
    k2 = c1 * V1B_NEG - c2 * V2B
    assert abs(k1) < 2 ^ 127 and abs(k2) < 2 ^ 127  # in bounds as shipped
    assert (k1 + k2 * lam - k) % n == 0  # reconstructs
    assert iround((G2 + 2 ^ 127) * k, 2 ^ 384) == c2 + 1  # straddles
    k2_bad = c1 * V1B_NEG - (c2 + 1) * V2B
    assert 2 ^ 127 <= abs(k2_bad) < 2 ^ 128  # escapes the half-width bound
    return k


def print_witness(name, n, lam):
    k = find_witness(n, lam)
    print("    const %s_BOUNDARY_SCALAR: [u64; 4] = [" % name)
    for i in range(4):
        print("        %#018x," % ((k >> (64 * i)) % 2 ^ 64))
    print("    ];")


print_witness("PALLAS", q, int(zeta_q))
print_witness("VESTA", p, int(zeta_p))
