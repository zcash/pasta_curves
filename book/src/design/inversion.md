# Constant-time inversion

This page gives an algorithm that inverts an element of either Pasta field in constant time,
and the argument that it is correct. The algorithm is the "safegcd" method of Bernstein and
Yang, [Fast constant-time gcd computation and modular inversion](https://eprint.iacr.org/2019/266)
(TCHES 2019), in the half-delta form and the serial arrangement of Bernstein, Chen, Harrison,
Huang, Maxwell, Wang, Wuille, and Yang,
[Accelerating and verifying constant-time modular inversion](https://doi.org/10.1007/978-3-032-25336-1_21)
(EUROCRYPT 2026). That arrangement is the one implemented in `bignum_montinv_p256` of
[s2n-bignum](https://github.com/awslabs/s2n-bignum), for the P-256 prime, at commit
[`ec62054cc1864839d44b1acc6e6a3f9eff5b6e68`](https://github.com/awslabs/s2n-bignum/blob/ec62054cc1864839d44b1acc6e6a3f9eff5b6e68/arm/p256/bignum_montinv_p256.S);
here the Pasta primes take its place. The algorithm works on Montgomery residues: given the
Montgomery form of $X$, it returns the Montgomery form of $X^{-1}$.

The lemmas and theorems are numbered on this page, and the Lean development under
`lean/PastaCurves/Inversion/` cites these numbers in its docstrings; each result below ends by
naming the declarations that prove it, in files of that directory. Theorem 5 is Theorem 1 of
Bernstein et al. (2026); §5 describes the certificate from which the Lean development proves it.

Notation: $p$ is one of the two Pasta primes, so $p$ is odd and $2^{254} < p < 2^{255}$, and
$R = 2^{256}$. All congruences are modulo $p$ unless said otherwise. Powers of two are
invertible modulo $p$, so dividing by $2^k$ in a congruence is legitimate.

## The algorithm

Inputs: a canonical Montgomery residue $x < p$, one of the two Pasta primes as the modulus,
and $\mathit{inv} = -p^{-1} \bmod 2^{64}$. Output: the canonical Montgomery residue $z$ with
$x z \equiv 2^{512} \pmod{p}$, that is, the Montgomery form of the inverse; $x = 0$ gives
$z = 0$.

State: $f$ and $g$, signed integers of at most 256 bits in magnitude, kept as four words plus
a sign word; $u$ and $v$, unsigned values below $2^{256}$, four words; and $d = 2\delta$, a
small signed integer, starting at $d_0 = 1$ (the half-delta start). Initially $f = p$, $g = x$,
$u = 0$, and $v = 2^{562} \bmod p$.

There are ten rounds. Each round does the following:

1. $\mathrm{divstep59}(d, f \bmod 2^{64}, g \bmod 2^{64}) \to (d', M)$: the exact transition
   matrix of 59 divsteps, computed from the low words alone, as three packed batches of 20,
   20, and 19 steps in which the coefficients ride in the upper bits of the same two words,
   then two $2 \times 2$ integer products.
2. $\mathrm{updateFG}(M, f, g) \to ((m_{00} f + m_{01} g) / 2^{59}, (m_{10} f + m_{11} g) / 2^{59})$,
   with exact divisions, in five-word signed arithmetic.
3. $\mathrm{updateUV}(M, u, v) \to (\mathrm{amontred}(m_{00} u + m_{01} v), \mathrm{amontred}(m_{10} u + m_{11} v))$,
   where $\mathrm{amontred}(t)$ adds $2^{61} p$, performs one word of Montgomery reduction,
   and returns a value below $2^{256}$ that is congruent to $t / 2^{64}$. The extra five bits
   per round, 64 against 59, are why the start is $2^{562} = 2^{512 + 5 \cdot 10}$.

The tenth round computes only $u$, folds in the sign of the final $f$ (which is $\pm 1$), and
reduces strictly by one conditional subtraction. The invariant after round $i$ is
$(f, g) \equiv x \cdot 2^{5i - 562} \cdot (u, v)$. After ten rounds $g = 0$, $f = \pm 1$, and
$x \cdot (\pm u) \equiv 2^{512}$.

## 1. Divsteps

The state is $(d, f, g)$, with $d$ an odd integer ($d = 2\delta$, so
$\delta \in \mathbb{Z} + \frac{1}{2}$; the start is $d = 1$, that is $\delta = \frac{1}{2}$),
$f$ odd, and $g$ any integer.

$$
\mathrm{divstep}(d, f, g) = \begin{cases}
(2 - d,\ g,\ (g - f) / 2) & \text{if } d > 0 \text{ and } g \text{ is odd}, \\
(2 + d,\ f,\ (g + (g \bmod 2) f) / 2) & \text{otherwise.}
\end{cases}
$$

Both divisions are exact: in the first case $g - f$ is even because both are odd; in the
second, $g + (g \bmod 2) f$ is even in either parity of $g$. Write $(d_n, f_n, g_n)$ for the
state after $n$ steps from $(d_0, f_0, g_0)$.

**Lemma 1 (linearity).** There are integer matrices $T_i$ with
$(f_{i+1}, g_{i+1})^\top = \frac{1}{2} T_i (f_i, g_i)^\top$, namely
$T = \begin{bmatrix} 0 & 2 \\ -1 & 1 \end{bmatrix}$ in the swap case and
$T = \begin{bmatrix} 2 & 0 \\ b & 1 \end{bmatrix}$, $b = g_i \bmod 2$, otherwise. Hence with
$M_n = T_{n-1} \cdots T_0$,

$$
2^n (f_n, g_n)^\top = M_n (f_0, g_0)^\top.
$$

Write $M_n = \begin{bmatrix} u_n & v_n \\ q_n & r_n \end{bmatrix}$, so $M_0 = I$. *In Lean:*
`M_spec` (`Divstep.lean`).

**Lemma 2 (locality).** $d_n$ and $M_n$ depend only on $d_0$, $f_0 \bmod 2^n$, and
$g_0 \bmod 2^n$. *Proof.* Induction on $n$. The branch taken at step $i$ depends on $d_i$ and
$g_i \bmod 2$. If $(f_i, g_i) \equiv (f_i', g_i') \pmod{2^k}$ with $k \geq 1$, the same branch
is taken and $(f_{i+1}, g_{i+1}) \equiv (f_{i+1}', g_{i+1}') \pmod{2^{k-1}}$, because each new
component is half of a combination of the old ones. So $n$ steps from states congruent modulo
$2^n$ take the same branches, and the branches determine $d_n$ and $M_n$. ∎ *In Lean:*
`divstep_local` and `divsteps_local` (`Divstep.lean`).

**Lemma 3 (bounds).** For every $n$: $|u_n| + |v_n| \leq 2^n$ and $|q_n| + |r_n| \leq 2^n$;
moreover each entry lies in $(-2^n, 2^n]$; and $\max(|f_n|, |g_n|) \leq \max(|f_0|, |g_0|)$.
*Proof.* Row sums: in the swap case the new first row is $2 (q, r)$ and the new second row is
$(q - u, r - v)$; in the other case the new first row is $2 (u, v)$ and the new second row is
$(q + b u, r + b v)$. Each new row sum is at most twice the larger old row sum. The half-open
range: build $M_n$ from the left, $M_{n+1} = T_n M_n$, so with
$M_n = \begin{bmatrix} u & v \\ q & r \end{bmatrix}$ the new entries are $2q, 2r, q - u, r - v$
in the swap case and $2u, 2v, q + b u, r + b v$ otherwise. If every old entry lies in
$(-2^n, 2^n]$, then each new entry is twice an old one, or an old one plus or minus another, so
it lies in $(-2^{n+1}, 2^{n+1}]$; the strict lower bound and the closed upper bound both
propagate, and the base case is the identity. The $\max$ bound: $f_{i+1}$ is one of $f_i$ and
$g_i$, and $|g_{i+1}| \leq (|f_i| + |g_i|) / 2$. ∎ *In Lean:* `M_rowSum_le`, `M_entry_range`,
and `divsteps_abs_le` (`Divstep.lean`).

**Lemma 4 (gcd and the end state).** $\gcd(f_n, g_n) = \gcd(f_0, g_0)$, and $f_n$ is odd. If
$g_n = 0$ then $f_n = \pm \gcd(f_0, g_0)$. If $g_0 = 0$ then every step is the non-swap case
with $b = 0$, so $f_n = f_0$, $g_n = 0$, and
$M_n = \begin{bmatrix} 2^n & 0 \\ 0 & 1 \end{bmatrix}$. *In Lean:* `divsteps_gcd`,
`divsteps_f_odd`, `f_natAbs_of_g_eq_zero`, and `divsteps_of_g_zero` (`Divstep.lean`).

**Lemma 4′ (the adjugate).** Each $T_i$ has determinant $2$, so $\det M_n = 2^n$, and the
adjugate of Lemma 1 gives $f_0 = r_n f_n - v_n g_n$ and $g_0 = u_n g_n - q_n f_n$ exactly
(multiply the two identities of Lemma 1 by the cofactors and cancel $2^n$). Hence if $g_n = 0$
then $f_n$ divides both $f_0$ and $g_0$; with $\gcd(f_0, g_0) = 1$ that makes $f_n = \pm 1$.
This is the form that Theorem 12 uses; the gcd invariance of Lemma 4 is the classical statement
and is not needed separately. *In Lean:* `M_det`, `M_inv_spec`, and `f_dvd_of_g_eq_zero`
(`Divstep.lean`).

**Theorem 5 (termination; Bernstein et al. 2026, Theorem 1).** If $d_0 = 1$, $f_0$ is odd,
$0 \leq g_0 \leq f_0 < 2^b$, and $n \geq \lceil (9437 b + 1) / 4096 \rceil$, then $g_n = 0$. For
$b = 256$ this is $n = 590$. The paper proves it for all $b$, in HOL Light, and the Lean
development proves it for all $b$ from a certificate (§5). (The paper's hypothesis is
$f_0 \leq 2^b$; the two forms differ only at $b = 0$.) *In Lean:* stated at
$n = \lceil (9437 b + 1) / 4096 \rceil$ as `TerminationBound` (`Termination.lean`), extended to
every larger $n$ by `TerminationBound.ge`, and proved by `terminationBound_of_certified`
(`HullBound.lean`) and by `terminationBound` and `terminationBound_256` (`HullCert.lean`).

## 2. Divsteps on packed words

Fix a batch length $k \leq 20$ and starting values $f, g$ (only their low 20 bits will be
used). Define the packed words

$$
w_f = (f \bmod 2^{20}) - 2^{41} \cdot 1 - 2^{62} \cdot 0, \qquad
w_g = (g \bmod 2^{20}) - 2^{41} \cdot 0 - 2^{62} \cdot 1,
$$

as integers, and run the divstep recurrence on the pair $(w_f, w_g)$ with the same branch
rule, reading $g$'s parity from $w_g$ and using exact halving.

**Lemma 6 (packing).** After $j \leq k$ steps the packed words are

$$
w_f^{(j)} = \varphi_j - 2^{41-j} u_j - 2^{62-j} v_j, \qquad
w_g^{(j)} = \gamma_j - 2^{41-j} q_j - 2^{62-j} r_j,
$$

where $(\varphi_j, \gamma_j)$ is the state reached from $(f \bmod 2^{20}, g \bmod 2^{20})$ by
the true recurrence and $M_j = \begin{bmatrix} u_j & v_j \\ q_j & r_j \end{bmatrix}$ is the true
matrix (Lemma 1) for the same branches. The branches taken on the packed words are the true
branches, $|\varphi_j|, |\gamma_j| < 2^{20}$, and $|w^{(j)}| < 2^{63}$. *Proof.* Induction on
$j$. The coefficient terms are multiples of $2^{41-j} \geq 2^{21}$, so
$w_g^{(j)} \equiv \gamma_j \pmod{2^{21}}$, and the parity test on $w_g$ reads the parity of
$\gamma_j$, which by Lemma 2 is the parity of $g_j$; so the branch is the true one. The
recurrence is linear and the matrices update as in Lemma 1, so the packed combination before
halving is $(\gamma_j \mp \varphi_j) - 2^{41-j}(q_j \mp u_j) - 2^{62-j}(r_j \mp v_j)$ (or with
$b$), and every term is even (the true components are even by exactness, and the coefficient
terms carry the factor $2^{41-j}$ with $j < 41$), so exact halving gives the claimed form with
$j + 1$. Magnitudes: $|\varphi|, |\gamma| \leq 2^{20} - 1$ by Lemma 3's $\max$ bound applied to
the truncated start; $2^{41-j} |u_j| \leq 2^{41}$ and $2^{62-j} |v_j| \leq 2^{62}$ by Lemma 3,
so $|w| < 2^{20} + 2^{41} + 2^{62} < 2^{63}$. ∎ *In Lean:* `divsteps_packedStart` and
`divsteps_packedStart_abs_lt` (`Packed.lean`).

**Lemma 7 (unpacking).** From $w_f^{(k)}$, with
$t = -w_f^{(k)} = 2^{41-k} u_k + 2^{62-k} v_k - \varphi_k$ and
$|\varphi_k| < 2^{20} \leq 2^{40-k}$:
$\lfloor (t + 2^{40-k}) / 2^{41-k} \rfloor = u_k + 2^{21} v_k$ exactly, and then
$v_k = \lfloor (u_k + 2^{21} v_k + 2^{20} - 1) / 2^{21} \rfloor$ and
$u_k = (u_k + 2^{21} v_k) - 2^{21} v_k$, using $u_k \in (-2^{20}, 2^{20}]$ from Lemma 3.
Likewise for the second row from $w_g^{(k)}$. (The half-open range is what makes this well
defined: $(2^{20}, 0)$ and $(-2^{20}, 1)$ pack identically.) *In Lean:* `unpack_spec`
(`Packed.lean`).

**Corollary 8 ($\mathrm{divstep59}$).** With $M^{(1)}$ from 20 packed steps at $d$, $M^{(2)}$
from 20 packed steps at $d'$ on the state $2^{-20} M^{(1)} (f, g)$ (whose low 20 bits are
determined by the low 40 bits of $(f, g)$, so by the low words), and $M^{(3)}$ from 19 steps
likewise, the product $M^{(3)} M^{(2)} M^{(1)}$ is the true 59-step matrix $M_{59}$, and the
returned $d$ is $d_{59}$, by Lemma 2 applied three times. Entries of $M_{59}$ lie in
$(-2^{59}, 2^{59}]$, and those of the partial products in $(-2^{40}, 2^{40}]$, so all fit
signed 64-bit words, and the intermediate states' low words can be recomputed from the low
words of $(f, g)$ alone. *In Lean:* `divstep59_spec` (`Divstep59.lean`).

## 3. The round arithmetic

Throughout the rounds $|f|, |g| \leq p < 2^{255}$ (Lemma 3's $\max$ bound from $(p, x)$), and
$0 \leq u, v < 2^{256}$ (Lemma 10 below).

**Lemma 9 ($\mathrm{updateFG}$).** $m_{00} f + m_{01} g$ and $m_{10} f + m_{11} g$ are divisible
by $2^{59}$ (Lemma 1), and
$|m f + m' g| \leq (|m| + |m'|) \max(|f|, |g|) < 2^{59} \cdot 2^{255} = 2^{314}$, so both fit in
five signed words and the shifts are exact. *In Lean:* `updateFG_spec` (`Round.lean`).

**Lemma 10 ($\mathrm{amontred}$).** Let $t = m_{00} u + m_{01} v$ (or the second row), so
$|t| \leq (|m_{00}| + |m_{01}|) \max(u, v) < 2^{59} \cdot 2^{256} = 2^{315}$. Let
$s = t + 2^{61} p$. Then $s \geq 2^{61} \cdot 2^{254} - 2^{315} = 0$, and
$s < 2^{315} + 2^{61} \cdot 2^{255} = 2^{315} + 2^{316} < 2^{317}$. Let
$w = (s \cdot \mathit{inv}) \bmod 2^{64}$, where $\mathit{inv} = -p^{-1} \bmod 2^{64}$; then
$s + w p \equiv 0 \pmod{2^{64}}$ and

$$
t' = (s + w p) / 2^{64} < s / 2^{64} + p < 2^{251} + 2^{61} p / 2^{64} + p = 2^{251} + 9p/8.
$$

In integers, $8 t' < 2^{254} + 9p$, which is the form that the Lean statement uses. Since
$p > 2^{251} \cdot 8/7$, this is below $2p$; and $2^{251} + 9p/8 < 2^{256}$ since $p < 2^{255}$.
Also $t' \equiv t \cdot 2^{-64}$. So $\mathrm{amontred}$ returns a four-word value below
$2^{256}$ that is congruent to $t / 2^{64}$, and one conditional subtraction of $p$ makes it
canonical. (s2n-bignum's P-256 version needs a top-carry check inside $\mathrm{amontred}$; the
Pasta bound shows that none is needed.) *In Lean:* `amontredZ_spec` and `amontred_spec`
(`Round.lean`).

Note that the starting values $v = 2^{562} \bmod p < p < 2^{256}$ and $u = 0$ satisfy the
range, and Lemma 10 keeps it.

## 4. The invariant and the result

Let $(f_i, g_i)$ be the true state after $59 i$ divsteps from $(p, x)$ with $d_0 = 1$, and let
$(u_i, v_i)$ be the coefficient vector after $i$ rounds: $u_0 = 0$, $v_0 = 2^{562} \bmod p$,
and $(u_{i+1}, v_{i+1}) = (\mathrm{amontred}(m_{00} u_i + m_{01} v_i), \mathrm{amontred}(m_{10} u_i + m_{11} v_i))$
with $M = M_{59}$ of round $i + 1$.

**Lemma 11 (invariant).** $(f_i, g_i) \equiv x \cdot 2^{5i - 562} \cdot (u_i, v_i)$. *Proof.*
For $i = 0$: $(p, x) \equiv (0, x) = x \cdot 2^{-562} \cdot (0, 2^{562})$. Step: by Lemma 1,
$(f_{i+1}, g_{i+1}) = 2^{-59} M (f_i, g_i) \equiv 2^{-59} M \cdot x 2^{5i - 562} (u_i, v_i)$,
and by Lemma 10 $(u_{i+1}, v_{i+1}) \equiv 2^{-64} M (u_i, v_i)$, so the right side is
$x \cdot 2^{5i - 562 - 59 + 64} (u_{i+1}, v_{i+1}) = x \cdot 2^{5(i+1) - 562} (u_{i+1}, v_{i+1})$.
∎ *In Lean:* `rounds_invariant` (`Model.lean`).

**Theorem 12 (correctness).** Assume Theorem 5 for $b = 256$, and $0 < x < p$. After ten rounds
(590 divsteps) $g_{10} = 0$, so $f_{10} = \pm 1$ by Lemma 4′ ($\gcd(p, x) = 1$). By Lemma 11 with
$i = 10$, $f_{10} \equiv x \cdot 2^{-512} u_{10}$, hence
$x \cdot (f_{10} u_{10}) \equiv 2^{512}$. The last round computes $u_{10}$ with the sign of
$f_{10}$ folded into the matrix row, then reduces strictly; by Lemma 10 one conditional
subtraction gives the canonical $z \equiv f_{10} u_{10}$, and $x z \equiv R^2$. In Montgomery
terms, if $x = X R$ and $z = Z R$ then $X Z \equiv 1$. *In Lean:* `montInv_spec` (`Model.lean`),
with Theorem 5 as a hypothesis, discharged in `montInv_correct` (`Correctness.lean`), and with
the primality of $p$, which it needs for $\gcd(p, x) = 1$, from the Pratt certificates of
`lean/PastaCurves/Primality.lean`.

For $x = 0$: by Lemma 4 every matrix is $\begin{bmatrix} 2^{59} & 0 \\ 0 & 1 \end{bmatrix}$, so
$u$ stays $0$ through every round, and $z = 0$.

**The sign of $f_{10}$ from one word.** The sign is read from
$(m_{00} f_9 + m_{01} g_9) \bmod 2^{64}$, the low word of $2^{59} f_{10} = \pm 2^{59}$ before
the shift: as a signed 64-bit word, $+2^{59}$ has bit 63 clear and $-2^{59}$ has it set. This
uses only that the low word of the five-word product is the low word of the true integer,
which Lemma 9 gives.

## 5. The termination bound

The Lean development proves Theorem 5 from Bernstein's "hull light" certificate of 2023
([`hull-light-20230416.sage`](https://cr.yp.to/2023/hull-light-20230416.sage), public domain),
following Harrison's check and proof of it in HOL Light (the `Divstep/` directory of
`jrh13/hol-light`). Section 4.3 of the paper describes the same argument.

The certificate has two explicit 80-point rational hulls $S_{1/2}$ and $S_{-1/2}$; every other
$S_\delta$ is defined as a linear image of $S_{1/2}$, with a factor $33/32$ for
$|\delta| \geq 5/2$. It has a shrink factor $\lambda' = 30902639/41749730$, and it comes with
three exact checks:

* eight inclusions $M S_\delta \subseteq \lambda'^k S_{\delta'}$ for the step maps
  $M_{-1}(x, y) = (y, (y - x)/2)$, $M_1(x, y) = (x, (y + x)/2)$, and $M_0(x, y) = (x, y/2)$, each
  certified half-plane by half-plane with two Farkas multipliers; every other transition is an
  equality by definition or follows from convexity;
* the initial containment: the triangle $0 \leq y \leq x \leq 1$ scaled by $2753/4096$ lies in
  $\mathrm{Hull}\, S_{1/2}$, so $0 \leq g \leq f \leq 2^b$ gives
  $(f/H, g/H) \in \mathrm{Hull}\, S_{1/2}$ for $H = 2^b \cdot 4096/2753$;
* a lattice-point endgame: with $L = 3047/2048$ the scaled hulls contain no integer point with
  $y \neq 0$, checked through an outer box and the enumeration of the few candidate points, so
  that once $2^b \lambda'^n \leq L \cdot 2753/4096$ the state has $g_n = 0$.

The simpler endgame, $|x| < 1$ or $|y| < 1$ after $n$ steps, fails at $n = 590$ and first passes
at 591, so it is not used; for $b = 256$ and $n = 590$ the slack is about 5%. The triangle
covers only $g \leq f$. For the square, which would admit a non-canonical $x$ up to $2^{256}$,
the largest admissible scale is $5193/8192$; with it, 590 steps fail by 0.19% and 591 pass, so
the input stays canonical.

In Lean, the half-planes and the Farkas records are generated data (`HullData.lean`, generated
by `lean/scripts/gen_hull.py` from `lean/scripts/hull_certificate.json`); the inclusion checks
are evaluated by the kernel (`HullCert.lean`); and the argument over abstract regions, following
Harrison's structure, ends in `terminationBound_of_certified` (`HullBound.lean`), with the range
of $\delta$ handled by the definitional formula and lemmas rather than a finite table.
`lean/scripts/verify_hull_certificate.py` re-checks the certificate data independently, in exact
arithmetic: the ten inclusions (the eight above, the initial triangle, and the outer box), with
724 Farkas records, and the 16 lattice points.

## 6. What is not covered here

This page does not relate the algorithm to any implementation of it.
