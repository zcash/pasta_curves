# Derives the projective model of `src/curves/projective.rs` and the maps of
# `src/curves/maps.rs` that the radix-27 scalar multiplication uses, and checks
# them symbolically for every curve y^2 = x^3 + b, then on Pallas and Vesta:
#
# - the 3-isogeny psi_b maps E_b : Y^2 Z = X^3 + b Z^3 onto E_{-27b};
# - the isomorphism iota : (X : Y : Z) -> (U X : Y : U^3 Z), U = 3, maps
#   E_{729b} onto E_b;
# - iota . psi_{-27b} . psi_b is the multiplication by 3 on E_b;
# - the maps between Jacobian and projective coordinates send each model of
#   the curve onto the other, and are inverse up to the scaling of each model.
#
# It prints the constants of `src/curves/params.rs` and
# `src/curves/projective.rs` in the shape of the Rust source. Run from this directory:
#
#     uv run sage radix27_tripling.sage
#
# Everything is exact polynomial arithmetic over QQ(b) and exact finite-field
# arithmetic, so the output does not depend on the SageMath version.

R.<X, Y, Z, b> = QQ[]


def proj_eq(c):
    """The projective model of E_c : Y^2 Z = X^3 + c Z^3."""
    return Y^2 * Z - X^3 - c * Z^3


def jac_eq(c):
    """The Jacobian model of E_c : Y^2 = X^3 + c Z^6, weights (2, 3, 1)."""
    return Y^2 - X^3 - c * Z^6


def subs(f, t):
    return f(X=t[0], Y=t[1], Z=t[2])


def divides(g, f):
    """Whether g divides f in R, that is f vanishes on the zero set of g."""
    return f % g == 0


def psi(t, c):
    """The 3-isogeny psi_c of `Projective::isogeny_3_with`."""
    x, y, z = t
    v = y^2 - c * z^2
    return (x * (v + 4 * c * z^2), y * (v - 8 * c * z^2), z * v)


U = 3


def iota(t):
    """`Projective::iota`, the isomorphism E_{729b} -> E_b."""
    x, y, z = t
    return (U * x, y, U^3 * z)


P = (X, Y, Z)

# --- psi_b : E_b -> E_{-27b}, and iota : E_{729b} -> E_b ---------------------

assert divides(proj_eq(b), subs(proj_eq(-27 * b), psi(P, b)))
assert divides(proj_eq(-27 * b), subs(proj_eq(729 * b), psi(P, -27 * b)))
assert divides(proj_eq(729 * b), subs(proj_eq(b), iota(P)))
assert U^6 == 729

# --- iota . psi_{-27b} . psi_b = [3] ------------------------------------------
#
# Compare the affine coordinates of the composition with Sage's multiplication
# by 3 on y^2 = x^3 + b over QQ(b), modulo the curve equation.

K = FractionField(PolynomialRing(QQ, "bb"))
Ex = EllipticCurve(K, [0, K.gen()])
mx, my = Ex.multiplication_by_m(3)
T = mx.parent().ring()
x, y = T.gens()
three = iota(psi(psi(P, b), -27 * b))
affine = [c(X=x, Y=y, Z=1, b=K.gen()) for c in three]


def vanishes_on_curve(g):
    """Whether g lies in the ideal of y^2 - x^3 - b. That generator is monic in y,
    so dividing by it as a polynomial in y over QQ(b)[x] decides membership
    without a Groebner basis."""
    gy = g.polynomial(y)
    return gy % gy.parent()([-(x^3 + K.gen()), 0, 1]) == 0


def agrees_on_curve(num, den, f):
    """Whether num / den equals the rational function f on the curve."""
    return vanishes_on_curve(num * f.denominator() - f.numerator() * den)


assert agrees_on_curve(affine[0], affine[2], mx)
assert agrees_on_curve(affine[1], affine[2], my)

# --- Jacobian <-> projective --------------------------------------------------
#
# `Projective::from_jacobian` is (X : Y : Z) -> (X Z : Y : Z^3), the monomials of
# weighted degree 3; `to_jacobian` is (X : Y : Z) -> (X Z : Y Z^2 : Z).

from_jacobian = lambda t: (t[0] * t[2], t[1], t[2]^3)
to_jacobian = lambda t: (t[0] * t[2], t[1] * t[2]^2, t[2])

assert divides(jac_eq(b), subs(proj_eq(b), from_jacobian(P)))
assert divides(proj_eq(b), subs(jac_eq(b), to_jacobian(P)))
# Jacobian -> projective -> Jacobian scales by lambda = Z^2 with weights (2, 3, 1).
lam = Z^2
assert to_jacobian(from_jacobian(P)) == (lam^2 * X, lam^3 * Y, lam * Z)
# Projective -> Jacobian -> projective scales by Z^2 with weights (1, 1, 1).
assert from_jacobian(to_jacobian(P)) == (Z^2 * X, Z^2 * Y, Z^2 * Z)

# --- Pallas and Vesta ---------------------------------------------------------

p = 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001
q = 0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001
for F, order in [(GF(p), q), (GF(q), p)]:
    E = EllipticCurve(F, [0, 5])
    assert E.order() == order
    # The kernel's nonzero points (0, +-sqrt(5)) are not rational: no point of
    # order 3, as the order is prime.
    assert not F(5).is_square()
    set_random_seed(0)
    for _ in range(32):
        pt = E.random_point()
        t = (pt[0], pt[1], F(1))
        image = tuple(R.change_ring(F)(c)(X=t[0], Y=t[1], Z=t[2], b=5) for c in three)
        assert image[2] != 0
        assert E(image[0] / image[2], image[1] / image[2]) == 3 * pt
    # The identity maps to the identity, and no output is (0, 0, 0).
    o = tuple(R.change_ring(F)(c)(X=0, Y=1, Z=0, b=5) for c in three)
    assert o[0] == 0 and o[2] == 0 and o[1] != 0

# --- The constants of `src/curves/params.rs` and `src/curves/projective.rs` --

PASTA_B = 5
IMAGE_B = -27 * PASTA_B
IMAGE2_B = U^6 * PASTA_B
assert -27 * IMAGE_B == IMAGE2_B
print(f"pub(crate) const PASTA_B: i64 = {PASTA_B};")
print(f"const IMAGE_B: i64 = -27 * PASTA_B; // {IMAGE_B}")
print(f"const U: i64 = {U};")
print(f"const IMAGE2_B: i64 = U.pow(6) * PASTA_B; // {IMAGE2_B}")
print(f"// times_3b multiplies by {3 * PASTA_B}, times_u3 by {U^3}")
