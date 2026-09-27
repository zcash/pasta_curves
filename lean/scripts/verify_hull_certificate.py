#!/usr/bin/env python
"""Verify hull_certificate.json (schema hull-certificate/2) from the JSON alone,
in exact arithmetic (fractions.Fraction / int; no floats except in the log).

    python verify_hull_certificate.py [hull_certificate.json] [--results]

What is checked (see NOTES.md section 6 for the schema):

  P  polygons: integer coprime half-planes; consecutive half-planes meet in a
     corner; the corners form a strictly convex CCW polygon whose edge
     constraints are exactly the listed half-planes (so the polygon IS the
     intersection of the half-planes); H0 and H1 contain the origin strictly.
  F  inclusions: for each Farkas record, m > 0, n > 0, p >= 0, q > 0 and the
     three identities
        (m a_i + n a_j) s^k = q (a' m11 + b' m21)
        (m b_i + n b_j) s^k = q (a' m12 + b' m22)
         m c_i + n c_j + p  = q c'
     with (a', b', c') the t-th half-plane of the target, (a_i, ..), (a_j, ..)
     the i-th and j-th of the source, M = [[m11, m12], [m21, m22]]; one record
     per target half-plane, in order.  This is exactly HOL Light's generic_2.
  C  cross-check: corner-based inclusion M*corner / s^k in every target
     half-plane, for every corner of the source (independent of F).
  L  lattice endgame: the scales are fudge*s^(2i)/L and fudge*(2s^2)^i/L; the
     exclusion slacks are (bound*scale - box) > 0 with bound = abs_max + 1
     (so the enumeration is complete: |x| <= x_abs_max, |y| <= y_abs_max);
     the points listed are exactly all integer (x, y) with y != 0 in that
     range; each scaled point violates its separating H1 half-plane strictly;
     bigdelta: coefficient = fudge*2^m_min/L, 2 s^2 >= 1 and
     outer_y < coefficient * s^(2 m_min).
  I  constants: fuzziness = latticescale*stretch, 0 < s < 1, 2^4096 s^9437 <= 1,
     s <= fuzziness^4096, and the example 9437 b + 1 <= 4096 n,
     2^b s^n <= fuzziness for (b, n) = (256, 590).

If certificate-v1.json (the corner-list form) is present next to the
certificate, the derived corners are compared with it as well.
"""

import json
import math
import os
import sys
import time
from fractions import Fraction as Fr

HERE = os.path.dirname(os.path.abspath(__file__))


class Failure(Exception):
    pass


def check(cond, msg):
    if not cond:
        raise Failure(msg)


def fr(s):
    check(isinstance(s, str), f"number not given as a string: {s!r}")
    if "/" in s:
        p, q = s.split("/")
        return Fr(int(p), int(q))
    return Fr(int(s))


COUNT = {"farkas": 0, "identities": 0, "corner_edge_evals": 0, "lattice_points": 0, "max_bits": 0}


def bits(*vals):
    for v in vals:
        b = max(abs(v.numerator).bit_length(), v.denominator.bit_length())
        COUNT["max_bits"] = max(COUNT["max_bits"], b)


# ---- exact geometry ----------------------------------------------------------


def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points):
    P = sorted(set(points))
    lower = []
    for p in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(P):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def same_cyclic(A, B):
    if len(A) != len(B) or not A:
        return len(A) == len(B)
    try:
        k = B.index(A[0])
    except ValueError:
        return False
    return all(A[j] == B[(k + j) % len(B)] for j in range(len(A)))


def edges_of(H):
    out = []
    n = len(H)
    for i in range(n):
        (ax, ay), (bx, by) = H[i], H[(i + 1) % n]
        a, b, c = by - ay, ax - bx, ax * by - bx * ay
        g = math.lcm(a.denominator, b.denominator, c.denominator)
        a, b, c = int(a * g), int(b * g), int(c * g)
        g = math.gcd(a, b, c)
        out.append((a // g, b // g, c // g))
    return out


def corners_of(E):
    """corner j = intersection of half-plane boundaries j-1 and j."""
    out = []
    for j in range(len(E)):
        a1, b1, c1 = E[j - 1]
        a2, b2, c2 = E[j]
        det = a1 * b2 - a2 * b1
        check(det != 0, f"consecutive half-planes {j - 1}, {j} are parallel")
        out.append((Fr(c1 * b2 - c2 * b1, det), Fr(a1 * c2 - a2 * c1, det)))
    return out


# ---- the checks -------------------------------------------------------------


def check_polygons(c, log):
    polys = {}
    corners = {}
    for name, rows in c["polygons"].items():
        E = []
        for row in rows:
            check(len(row) == 3, f"{name}: half-plane with {len(row)} entries")
            a, b, cc = (fr(v) for v in row)
            check(
                all(v.denominator == 1 for v in (a, b, cc)),
                f"{name}: non-integer half-plane {row}",
            )
            a, b, cc = int(a), int(b), int(cc)
            check((a, b) != (0, 0), f"{name}: degenerate half-plane")
            check(math.gcd(a, b, cc) == 1, f"{name}: half-plane {row} not coprime")
            E.append((a, b, cc))
        H = corners_of(E)
        check(
            same_cyclic(convex_hull(H), H),
            f"{name}: corners are not a strictly convex CCW polygon",
        )
        check(
            edges_of(H) == E,
            f"{name}: edge constraints of the corners differ from the listed half-planes",
        )
        for x, y in H:
            for a, b, cc in E:
                check(
                    a * x + b * y <= cc,
                    f"{name}: a corner violates a half-plane (not the intersection)",
                )
        polys[name] = E
        corners[name] = H
        log(
            f"P   {name:<7} {len(E):3d} half-planes; corners derived, strictly convex, CCW, "
            "edges regenerate the list: ok"
        )
    for name in ("H0", "H1"):
        check(
            all(cc > 0 for _, _, cc in polys[name]),
            f"{name} does not contain the origin strictly",
        )
    log("P   origin strictly inside H0 and H1 (all c > 0): ok")
    return polys, corners


def check_inclusions(c, polys, corners, s, log):
    names = [inc["name"] for inc in c["inclusions"]]
    for inc in c["inclusions"]:
        name = inc["name"]
        S, T = polys[inc["source"]], polys[inc["target"]]
        k = inc["k"]
        check(isinstance(k, int) and k >= 0, f"{name}: bad k")
        M = [[fr(v) for v in row] for row in inc["matrix"]]
        check(len(M) == 2 and all(len(r) == 2 for r in M), f"{name}: matrix shape")
        (m11, m12), (m21, m22) = M
        check(m11 * m22 - m12 * m21 != 0, f"{name}: singular matrix")
        sk = s**k
        F = inc["farkas"]
        check(
            len(F) == len(T),
            f"{name}: {len(F)} Farkas records for {len(T)} target half-planes",
        )
        n_p0 = 0
        for t, rec in enumerate(F):
            i, j = rec["i"], rec["j"]
            check(
                isinstance(i, int) and isinstance(j, int) and 0 <= i < len(S) and 0 <= j < len(S),
                f"{name}[{t}]: source index out of range",
            )
            m, n, p, q = (fr(rec[key]) for key in ("m", "n", "p", "q"))
            check(m > 0 and n > 0, f"{name}[{t}]: m, n not positive")
            check(p >= 0, f"{name}[{t}]: p negative")
            check(q > 0, f"{name}[{t}]: q not positive")
            ai, bi, ci = S[i]
            aj, bj, cj = S[j]
            ap, bp, cp = T[t]
            lhs1, rhs1 = (m * ai + n * aj) * sk, q * (ap * m11 + bp * m21)
            lhs2, rhs2 = (m * bi + n * bj) * sk, q * (ap * m12 + bp * m22)
            lhs3, rhs3 = m * ci + n * cj + p, q * cp
            bits(lhs1, rhs1, lhs2, rhs2, lhs3, rhs3)
            check(lhs1 == rhs1, f"{name}[{t}]: x-identity fails")
            check(lhs2 == rhs2, f"{name}[{t}]: y-identity fails")
            check(lhs3 == rhs3, f"{name}[{t}]: constant identity fails")
            COUNT["farkas"] += 1
            COUNT["identities"] += 3
            n_p0 += p == 0
        # corner-based cross-check (independent of the multipliers)
        for x, y in corners[inc["source"]]:
            X, Y = (m11 * x + m12 * y) / sk, (m21 * x + m22 * y) / sk
            for a, b, cc in T:
                COUNT["corner_edge_evals"] += 1
                check(
                    a * X + b * Y <= cc,
                    f"{name}: corner ({x}, {y}) maps outside the target",
                )
        log(
            f"F/C {name:<14} {inc['source']:<5} -> {inc['target']:<6} k={k} "
            f"M={inc['matrix']!s:<40} {len(F):3d} Farkas records ok (p=0 in {n_p0}); "
            f"{len(corners[inc['source']])} corners x {len(T)} edges ok"
        )
    return names


def check_lattice(c, polys, s, L, log):
    lat = c["lattice"]
    H1 = polys["H1"]
    outer = polys["houter"]
    check(
        [(a == 0, b == 0) for a, b, _ in outer]
        == [(True, False), (False, True), (True, False), (False, True)],
        "houter is not a box",
    )
    oy = Fr(outer[0][2], -outer[0][1])
    ox = Fr(outer[1][2], outer[1][0])
    check(
        Fr(outer[2][2], outer[2][1]) == oy and Fr(outer[3][2], -outer[3][0]) == ox,
        "houter is not symmetric",
    )
    two_s2 = 2 * s * s
    for case in lat["cases"]:
        i = case["i"]
        fudge = fr(case["fudge"])
        sx, sy = fr(case["xscale"]), fr(case["yscale"])
        check(sx == fudge * s ** (2 * i) / L, f"lattice {i}: xscale != fudge s^(2i)/L")
        check(sy == fudge * two_s2**i / L, f"lattice {i}: yscale != fudge (2s^2)^i/L")
        xmax, ymax = case["x_abs_max"], case["y_abs_max"]
        # completeness of the enumeration: (xmax+1) * sx > ox, i.e. any integer x
        # with |x| sx <= ox has |x| <= xmax
        check(
            case["x_exclusion"]["bound"] == xmax + 1 and case["y_exclusion"]["bound"] == ymax + 1,
            f"lattice {i}: bounds",
        )
        slx, sly = fr(case["x_exclusion"]["slack"]), fr(case["y_exclusion"]["slack"])
        check(slx == (xmax + 1) * sx - ox and slx > 0, f"lattice {i}: x exclusion slack")
        check(sly == (ymax + 1) * sy - oy and sly > 0, f"lattice {i}: y exclusion slack")
        # (and the range is not larger than needed, so the count is the HOL count)
        check(xmax * sx <= ox and ymax * sy <= oy, f"lattice {i}: range larger than the box")
        expect = sorted(
            (x, y) for y in range(-ymax, ymax + 1) if y != 0 for x in range(-xmax, xmax + 1)
        )
        got = sorted((p["x"], p["y"]) for p in case["points"])
        check(
            got == expect,
            f"lattice {i}: points are not exactly the integer points with y != 0 in the range",
        )
        for p in case["points"]:
            X, Y = p["x"] * sx, p["y"] * sy
            a, b, cc = H1[p["separating_edge"]]
            check(
                a * X + b * Y > cc,
                f"lattice {i}: point ({p['x']}, {p['y']}) does not strictly violate edge "
                f"{p['separating_edge']}",
            )
            COUNT["lattice_points"] += 1
        log(
            f"L   i={i}: scales ({sx}, {sy}), |x|<={xmax}, |y|<={ymax}, {len(case['points'])} "
            "points with y!=0 each strictly outside H1 via its separating edge: ok"
        )
    bd = lat["bigdelta"]
    m_min = bd["m_min"]
    check(
        [case["i"] for case in lat["cases"]] == list(range(m_min)),
        "lattice cases do not cover 0..m_min-1",
    )
    fudge = fr(bd["fudge"])
    check(0 <= fudge <= 1, "bigdelta fudge not in [0,1] (h1_shrink needs it)")
    check(fr(bd["two_s2_ge_1"]["value"]) == two_s2 and two_s2 >= 1, "2 s^2 >= 1 fails")
    ineq = bd["inequality"]
    coef = fr(ineq["coefficient"])
    check(coef == fudge * 2**m_min / L and ineq["s_exponent"] == 2 * m_min, "bigdelta coefficient")
    val = coef * s ** ineq["s_exponent"]
    check(fr(ineq["value"]) == val and fr(ineq["outer_y"]) == oy, "bigdelta recorded values")
    check(oy < val, "bigdelta inequality outer_y < fudge (2s^2)^m_min / L fails")
    log(
        f"L   bigdelta: 2 s^2 = {float(two_s2):.6f} >= 1 and (32/33)(2 s^2)^{m_min} / L = "
        f"{float(val):.6f} > outer_y = {oy}: ok (so for i >= {m_min}, |y| < 1)"
    )
    return ox, oy


def check_constants(c, s, log):
    stretch, L, fuzz = fr(c["stretch"]), fr(c["latticescale"]), fr(c["fuzziness"])
    check(fuzz == L * stretch, "fuzziness != latticescale * stretch")
    check(0 < s < 1, "s not in (0,1)")
    ex = next(e for e in c["inequalities"] if e["name"] == "upow_bound")["exponents"]
    e2, es, off = ex["two"], ex["s"], ex["offset"]
    check(off == 1, "upow offset")
    t0 = time.perf_counter()
    num, den = s.numerator, s.denominator
    check(2**e2 * num**es <= den**es, f"2^{e2} s^{es} <= 1 fails")
    check(num * fuzz.denominator**e2 <= den * fuzz.numerator**e2, f"s <= fuzziness^{e2} fails")
    t1 = time.perf_counter() - t0
    b, n = c["example"]["b"], c["example"]["n"]
    check(es * b + off <= e2 * n, f"{es} b + {off} <= {e2} n fails for b={b}, n={n}")
    check(2**b * num**n <= fuzz * den**n, f"2^{b} s^{n} <= fuzziness fails")
    log(
        f"I   fuzziness = L*stretch = {fuzz}, 0 < s < 1, 2^{e2} s^{es} <= 1, "
        f"s <= fuzziness^{e2}: ok ({t1:.3f}s); example b={b} n={n}: ok"
    )


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "hull_certificate.json")
    with open(path) as f:
        c = json.load(f)
    check(c.get("schema") == "hull-certificate/2", f"unexpected schema {c.get('schema')!r}")

    def log(msg):
        print(msg)
        sys.stdout.flush()

    t0 = time.perf_counter()
    try:
        s = fr(c["s"])
        L = fr(c["latticescale"])
        polys, corners = check_polygons(c, log)
        names = check_inclusions(c, polys, corners, s, log)
        check(
            "init2stable" in names and "theoremouter" in names, "init2stable / theoremouter missing"
        )
        init = next(i for i in c["inclusions"] if i["name"] == "init2stable")
        check(
            init["source"] == "hinit"
            and init["target"] == "H1"
            and init["k"] == 0
            and init["matrix"] == [[c["stretch"], "0"], ["0", c["stretch"]]],
            "init2stable is not hinit -> H1 by stretch*I",
        )
        outer = next(i for i in c["inclusions"] if i["name"] == "theoremouter")
        check(
            outer["source"] == "H1"
            and outer["target"] == "houter"
            and outer["k"] == 0
            and outer["matrix"] == [["1", "0"], ["0", "1"]],
            "theoremouter is not H1 -> houter by I",
        )
        check_lattice(c, polys, s, L, log)
        check_constants(c, s, log)
        # optional: the corner lists of the v1 certificate (the Sage points)
        v1 = os.path.join(os.path.dirname(os.path.abspath(path)), "certificate-v1.json")
        if os.path.exists(v1):
            with open(v1) as f:
                old = json.load(f)
            for name in ("H0", "H1"):
                pts = [(fr(x), fr(y)) for x, y in old[name]]
                check(
                    pts == corners[name],
                    f"derived corners of {name} differ from certificate-v1.json",
                )
            log(
                "V   corners derived from the half-planes equal the Sage corner lists in "
                "certificate-v1.json: ok"
            )
    except Failure as e:
        log(f"VERIFICATION FAILED: {e}")
        sys.exit(1)
    dt = time.perf_counter() - t0
    summary = {
        "polygons": len(c["polygons"]),
        "half_planes": sum(len(p) for p in c["polygons"].values()),
        "inclusions": len(c["inclusions"]),
        "farkas_records": COUNT["farkas"],
        "identities": COUNT["identities"],
        "corner_edge_evaluations": COUNT["corner_edge_evals"],
        "lattice_points": COUNT["lattice_points"],
        "max_operand_bits": COUNT["max_bits"],
        "seconds": dt,
    }
    log(
        f"ALL CHECKS PASS: {summary['polygons']} polygons, {summary['half_planes']} half-planes, "
        f"{summary['inclusions']} inclusions, {summary['farkas_records']} Farkas records "
        f"({summary['identities']} identities), {summary['corner_edge_evaluations']} corner/edge "
        f"evaluations, {summary['lattice_points']} lattice points; largest operand "
        f"{summary['max_operand_bits']} bits; {summary['seconds']:.2f}s"
    )
    if "--results" in sys.argv:
        with open(os.path.join(HERE, "verify_results.json"), "w") as f:
            json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
