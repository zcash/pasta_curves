# Derives the GLV constants declared in `src/glv.rs` (the `GlvParams`
# implementations for `pallas::Point` and `vesta::Point`) from the curve
# definitions alone, and prints them in the exact shape of the Rust
# source so the two can be diffed.
#
# Run from this directory:
#
#     uv run sage glv_constants.sage
#
# Everything below is exact integer/rational arithmetic and hand-rolled
# two-dimensional lattice reduction, so the output does not depend on
# the SageMath version.

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

# `round(a / b)` (half away from zero for positive operands), in exact
# integer arithmetic. This is the same rounding `round_mul_shift` in
# `src/glv.rs` implements by adding 2^383 and truncating.
def iround(a, b):
    return (2 * a + b) // (2 * b)


# --- The cube roots of unity ------------------------------------------------
#
# The crate pins Fq::ZETA to g^((q-1)/3) for the field's multiplicative
# generator g = 5 (`GENERATOR` in `src/fields/fq.rs`). Fp::ZETA is then
# the unique base-field cube root paired with it by the Pallas
# endomorphism: phi(x, y) = (zeta_p * x, y) must equal [zeta_q](x, y).
# The same pair, with the roles of the fields swapped, serves Vesta.

zeta_q = Fq(5) ^ ((q - 1) // 3)
assert zeta_q != 1 and zeta_q ^ 3 == 1

endo_image = int(zeta_q) * Gp
zeta_p = Fp(endo_image[0]) / Fp(Gp[0])
assert zeta_p != 1 and zeta_p ^ 3 == 1
assert endo_image == Pallas(zeta_p * Gp[0], Gp[1])

# The reciprocal pairing on Vesta, where zeta_p acts on scalars and
# zeta_q on coordinates.
assert int(zeta_p) * Gv == Vesta(zeta_q * Gv[0], Gv[1])


# --- The short lattice basis ------------------------------------------------
#
# GLV works in the rank-2 lattice L = {(a, b) : a + b*lambda = 0 (mod n)},
# where n is the group order and lambda the scalar-field cube root: any
# (a, b) in L with |a|, |b| about sqrt(n) rewrites k*P as a1*P + a2*phi(P)
# with half-width a1, a2. Lagrange-Gauss reduction of the obvious basis
# {(n, 0), (-lambda, 1)} yields a shortest basis.

def lagrange_gauss(u, v):
    while True:
        if u[0] ^ 2 + u[1] ^ 2 > v[0] ^ 2 + v[1] ^ 2:
            u, v = v, u
        m = iround(u[0] * v[0] + u[1] * v[1], u[0] ^ 2 + u[1] ^ 2)
        if m == 0:
            return u, v
        v = (v[0] - m * u[0], v[1] - m * u[1])


# The `GlvParams` constants for group order `n` and scalar-field cube
# root `lam`, as the tuple (V1A, V1B_NEG, V2A, V2B, G1, G2).
def glv_params(n, lam):
    u, v = lagrange_gauss((n, 0), ((-lam) % n, 1))
    # Canonicalize the reduced basis to the crate's sign convention:
    # v1 = (V1A, -V1B_NEG) and v2 = (V2A, V2B) with all four of V1A,
    # V1B_NEG, V2A, V2B positive, and det(v1, v2) = +n.
    u = u if u[0] > 0 else (-u[0], -u[1])
    v = v if v[0] > 0 else (-v[0], -v[1])
    if u[1] > 0:
        u, v = v, u
    (V1A, V1B_NEG), (V2A, V2B) = (u[0], -u[1]), v
    assert min(V1A, V1B_NEG, V2A, V2B) > 0
    assert (V1A - V1B_NEG * lam) % n == 0 and (V2A + V2B * lam) % n == 0
    assert V1A * V2B + V1B_NEG * V2A == n
    assert max(V1A, V1B_NEG, V2A, V2B) < 2 ^ 128  # each fits a u128

    # The Babai rounding coefficients: fixed-point approximations of
    # V2B/n and V1B_NEG/n with 384 fractional bits, so that
    # c1 = round(G1*k / 2^384) and c2 = round(G2*k / 2^384) recover the
    # nearest-lattice-point coefficients round(k*V2B/n), round(k*V1B_NEG/n)
    # for every scalar k (the crate's `decompose` tests check the
    # resulting halves stay in bounds over the whole field).
    G1 = iround(2 ^ 384 * V2B, n)
    G2 = iround(2 ^ 384 * V1B_NEG, n)
    assert max(G1, G2) < 2 ^ 320  # each fits five u64 limbs
    return V1A, V1B_NEG, V2A, V2B, G1, G2


def rust_limbs(x, count, indent):
    limbs = [(x >> (64 * i)) & (2 ^ 64 - 1) for i in range(count)]
    assert x == sum(l << (64 * i) for i, l in enumerate(limbs))
    return "\n".join("%s%#x," % (" " * indent, l) for l in limbs)


def print_impl(curve, n, lam):
    V1A, V1B_NEG, V2A, V2B, G1, G2 = glv_params(n, lam)
    print("impl GlvParams for %s::Point {" % curve)
    print("    const V1A: u128 = %#x;" % V1A)
    print("    const V1B_NEG: u128 = %#x;" % V1B_NEG)
    print("    const V2A: u128 = %#x;" % V2A)
    print("    const V2B: u128 = %#x;" % V2B)
    for name, g in [("G1", G1), ("G2", G2)]:
        print("    const %s: [u64; 5] = [" % name)
        print(rust_limbs(g, 5, 8))
        print("    ];")
    print("}")


print("// Pallas: group order q, lambda = Fq::ZETA = %#x" % int(zeta_q))
print_impl("pallas", q, int(zeta_q))
print()
print("// Vesta: group order p, lambda = Fp::ZETA = %#x" % int(zeta_p))
print_impl("vesta", p, int(zeta_p))
