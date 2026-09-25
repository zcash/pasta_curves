//! GLV scalar multiplication with **Eisenstein-integer recoding**.
//!
//! This is an alternative recoding for the same GLV split that
//! [`crate::glv`] performs, trading that module's two independent width-4
//! wNAF digit strings for a single *joint* digit string over the Eisenstein
//! integers. It cuts the ladder's point additions by roughly a quarter while
//! *shrinking* the precomputed table, and it guarantees at most one addition
//! per ladder column.
//!
//! # The structure
//!
//! GLV writes $k \equiv k_1 + k_2\lambda \pmod n$ with $|k_1|, |k_2| <
//! 2^{127}$, where $\lambda$ = `Scalar::ZETA` satisfies $\lambda^2 + \lambda +
//! 1 = 0$. The pair $(k_1, k_2)$ is therefore not two independent integers but
//! one element $k_1 + k_2\omega$ of the ring of **Eisenstein integers**
//!
//! $$\mathbb{Z}[\omega] = \mathbb{Z}\lbrack X \rbrack/(X^2 + X + 1),$$
//!
//! the ring of integers of $\mathbb{Q}(\sqrt{-3})$, mapped into the scalar
//! field by $\omega \mapsto \lambda$. It is a Euclidean domain under the norm
//! $N(a + b\omega) = a^2 - ab + b^2$, and two facts about it drive everything
//! below:
//!
//! 1. **2 is inert** in $\mathbb{Z}[\omega]$ (the quotient
//!    $\mathbb{Z}[\omega]/2 \cong \mathbb{F}_4$ is a field, since $X^2 + X + 1$
//!    is irreducible mod 2). So "even" means *both* coordinates even, halving
//!    is exact and coordinate-wise, and the whole width-$w$ NAF theory carries
//!    over verbatim with $2^w$ read in the ring.
//! 2. **The unit group is $\mu_6 = \{\pm 1, \pm\omega, \pm\omega^2\}$**, of
//!    order 6, and these six units are exactly the six automorphisms of a
//!    $j$-invariant-0 curve: $(x, y) \mapsto (\zeta^i x, \pm y)$. Each costs at
//!    most one base-field multiplication.
//!
//! Both facts are verified in `sage/glv_eisenstein.sage` by step `[1]`.
//!
//! # Why the table is small
//!
//! Taking $w = 3$, the digits are the odd residue classes of
//! $\mathbb{Z}[\omega]/8$: of the $8^2 = 64$ classes, the $4^2 = 16$ with both
//! coordinates even are the even ones, leaving **48 odd classes**. A
//! coordinate-wise recoding would need a stored point for each class up to
//! negation, i.e. 24 points.
//!
//! But $\mu_6$ acts on those 48 classes, and that action is *free* (verified
//! exhaustively by the `orbit_action_is_free` test), so the classes fall into
//! exactly $48 / 6 = 8$ orbits. Storing one representative per orbit and
//! recovering the other five by an automorphism gives a table of **8 points**
//! reaching all 48 digits. The representatives are
//!
//! $$1,\quad 1-\omega,\quad 2-\omega,\quad 1-2\omega,\quad 3,\quad 3-\omega,
//!   \quad 1-3\omega,\quad 2-3\omega,$$
//!
//! of norms 1, 3, 7, 7, 9, 13, 13, 19, and [`Table`] builds all eight from the
//! base point with **7 point additions** plus a few applications of the
//! endomorphism (see the `window_proj` addition chain).
//!
//! # Cost
//!
//! A nonzero digit forces the residual to be divisible by 8, so it is followed
//! by at least two zero columns; the third column after it is nonzero with
//! probability $3/4$ (at least one coordinate odd). The expected gap is
//! therefore $3 + \tfrac{1}{4}/\tfrac{3}{4} = \tfrac{10}{3}$ columns, a density
//! of $3/10$, and over the $\approx 128$ columns of a 127-bit half:
//!
//! | recoding | table build | ladder adds | total | stored |
//! |---|---|---|---|---|
//! | split wNAF-4 ([`crate::glv`]) | 4 adds | ~51.2 | ~55.2 | 8 x + 8 y |
//! | joint Eisenstein NAF (this module) | 7 adds | **~38.4** | **~45.4** | 24 x + 8 y |
//!
//! The table is larger in field elements but not in points: the eight orbit
//! representatives are what the chain builds, and the other sixteen entries
//! are their rotations, which share a $y$ and cost one multiplication each.
//! Over a batch, [`Table::batch`] runs the seven additions in affine
//! coordinates with a shared inversion, which makes that column cheaper than
//! the wNAF one rather than dearer.
//!
//! Doublings are unchanged at $\approx 125$. The `digit_statistics` test
//! measures the ladder figure over random scalars and pins it, and step
//! `[7]` of `sage/glv_eisenstein.sage` (`check_recoder`) measures the same
//! density against the real curves.
//!
//! # References
//!
//! - R. P. Gallant, R. J. Lambert, S. A. Vanstone, "Faster Point Multiplication
//!   on Elliptic Curves with Efficient Endomorphisms", CRYPTO 2001.
//!   <https://www.iacr.org/archive/crypto2001/21390189.pdf>
//! - The construction is verified in Lean 4 in
//!   <https://github.com/daira/CompElliptic> (`Rings/Eisenstein/*`), which
//!   mechanically checks the inertness of 2, the unit group $\mu_6$, the
//!   freeness of the action on the 48 odd classes, and the identification of
//!   $\mu_6$ with the six automorphisms of a $j = 0$ curve.
//! - `sage/glv_eisenstein.sage` regenerates and re-verifies every constant in
//!   this module against the real curves. Its checks are numbered `[1]` to
//!   `[7]`, and each item below whose correctness rests on one names the step
//!   and the function that establishes it.

use alloc::vec::Vec;

use group::CurveAffine as _;

use ff::{Field, WithSmallOrderMulGroup};

use crate::arithmetic::VartimeField;

use crate::glv::{GlvParams, decompose};

/// The orbit representatives $r_0, \dots, r_7$, as coefficient pairs
/// $(a, b)$ meaning $a + b\omega$.
///
/// These eight Eisenstein integers have pairwise distinct $\mu_6$-orbits in
/// $\mathbb{Z}[\omega]/8$, and their 48 unit multiples are exactly the 48 odd
/// classes, checked by the `lut_covers_exactly_the_odd_classes` test. They are
/// what [`Table::window_proj`]'s addition chain produces, checked by
/// the `window_matches_representatives` test.
///
/// Verified in `sage/glv_eisenstein.sage` by step `[3]`, which rebuilds the
/// orbits from the ring helpers `red` and `rotate`.
const REPS: [(i8, i8); REP_COUNT] = [
    (1, 0),
    (1, -1),
    (2, -1),
    (1, -2),
    (3, 0),
    (3, -1),
    (1, -3),
    (2, -3),
];

/// Number of orbit representatives, $48 / |\mu_6| = 8$.
const REP_COUNT: usize = 8;

/// Number of distinct rotations $\omega^i$; the remaining three units are
/// their negations, and negating a point is free.
const ROTATIONS: usize = 3;

/// Multiplies an Eisenstein integer by $\omega$:
/// $(a + b\omega)\omega = a\omega + b\omega^2 = -b + (a - b)\omega$,
/// using $\omega^2 = -1 - \omega$.
const fn rotate(d: (i8, i8)) -> (i8, i8) {
    (-d.1, d.0 - d.1)
}

/// Bit of a packed digit carrying the unit's sign; negating a point is free.
const NEG_BIT: u8 = 1 << 5;

/// The packed digit standing for "this column is zero".
///
/// Real digits pack as `rep | rot << 3 | neg << 5` and so never exceed
/// `7 | 2 << 3 | 1 << 5` = 55.
const ZERO_DIGIT: u8 = 0xFF;

/// One entry of the residue-class lookup table.
#[derive(Clone, Copy)]
struct Lut {
    /// Packed digit, or [`ZERO_DIGIT`] for the even classes (which the
    /// recoder handles by its parity branch and never looks up).
    digit: u8,
    /// The digit's own coefficients, subtracted from the residual.
    da: i8,
    db: i8,
}

/// The 48 odd classes of $\mathbb{Z}[\omega]/8$, indexed by
/// `(a mod 8) * 8 + (b mod 8)`.
///
/// Derived at compile time from [`REPS`] and the unit action, so the eight
/// representatives are the single source of truth; the 16 even classes keep
/// [`ZERO_DIGIT`]. That the 48 writes land in 48 *distinct* slots is the
/// freeness of the action, asserted by the
/// `orbit_action_is_free` test.
///
/// Verified in `sage/glv_eisenstein.sage` by step `[2]`.
const LUT: [Lut; 64] = build_lut();

const fn build_lut() -> [Lut; 64] {
    let mut lut = [Lut {
        digit: ZERO_DIGIT,
        da: 0,
        db: 0,
    }; 64];
    let mut rep = 0;
    while rep < REP_COUNT {
        let mut d = REPS[rep];
        let mut rot = 0;
        while rot < ROTATIONS {
            let mut neg = 0;
            while neg < 2 {
                let e = if neg == 1 { (-d.0, -d.1) } else { d };
                // `& 7` is the low three bits, which for two's complement is
                // the least non-negative residue mod 8.
                let idx = ((e.0 & 7) as usize) * 8 + ((e.1 & 7) as usize);
                lut[idx] = Lut {
                    digit: (rep as u8) | ((rot as u8) << 3) | ((neg as u8) << 5),
                    da: e.0,
                    db: e.1,
                };
                neg += 1;
            }
            d = rotate(d);
            rot += 1;
        }
        rep += 1;
    }
    lut
}

/// Upper bound on the number of Eisenstein-NAF columns of one decomposition.
///
/// The digits have coefficients bounded by 5 in absolute value, so a column
/// maps a residual coordinate $x$ to $(x - d)/2$ with $|(x - d)/2| \le
/// \lfloor (|x| + 5)/2 \rfloor$. Iterating that bound from the $2^{127}$ that
/// [`decompose`] guarantees reaches the box $\max(|a|, |b|) \le 12$ in 124
/// columns, and every state in that box is exhaustively verified to
/// terminate within 6 more. The `digit_statistics` test also observes the
/// realised maximum.
///
/// Verified in `sage/glv_eisenstein.sage` by step `[5]`, whose `step` and
/// `drain` functions iterate that bound and exhaust the box.
const MAX_DIGITS: usize = 130;

/// Batch size from which [`Table::batch_mul`] switches to the batch-affine
/// ladder.
///
/// Below it the projective ladder is cheaper, because a batch-affine ladder
/// spends 164 field inversions however few points share them. See
/// [`Table::batch_mul`] for the cost model.
///
/// The value tracks the cost of [`VartimeField::invert_vartime`], the safegcd
/// inversion, at about 63 multiplications on the Pasta fields. The two ladders
/// measure level at a batch of 64 and the affine one is clearly ahead by 96,
/// which is the value taken here. Against the constant-time Fermat inversion,
/// at about 417 multiplications, the crossover would instead be near 420.
pub const BATCH_AFFINE_THRESHOLD: usize = 96;

/// The Eisenstein window for one base point: the eight orbit representatives
/// $r_j P$, each in its three rotations $\omega^i r_j P$, in affine
/// coordinates, stored as 24 $x$ plus 8 shared $y$ (1 KiB) because
/// $\varphi$ fixes $y$. Only the 8 representatives are normalized; each
/// rotation is then one multiplication.
///
/// Build one with [`Table::new`], or many sharing a single field inversion
/// with [`Table::batch`].
#[derive(Clone, Copy, Debug)]
pub struct Table<C: GlvParams> {
    /// $x(\omega^{\mathrm{rot}}\, r_{\mathrm{rep}}\, P)$, at
    /// `rep * ROTATIONS + rot`.
    xs: [C::Base; REP_COUNT * ROTATIONS],
    /// $y(r_{\mathrm{rep}}\, P)$, shared by that representative's three
    /// rotations because $\varphi$ leaves $y$ alone.
    ys: [C::Base; REP_COUNT],
}

impl<C: GlvParams> Table<C> {
    /// Builds the window for a single point: 7 point additions, one field
    /// inversion, and the rotations. Amortize the inversion over many points
    /// with [`Table::batch`].
    pub fn new(p: &C) -> Self {
        let proj = Self::window_proj(p);
        let mut affine = [C::AffineExt::identity(); REP_COUNT];
        C::batch_normalize(&proj, &mut affine);
        Self::from_window(&affine)
    }

    /// Builds [`Table`]s for a batch of points, running the whole
    /// seven-addition chain in *affine* coordinates with one field inversion
    /// shared across the batch per step.
    ///
    /// A projective chain pays about 16 field multiplications per addition and
    /// then a normalization of all `8 * n` representatives. Batched affine
    /// pays about 6 per addition, counting the roughly three multiplications
    /// Montgomery's trick costs each lane, and normalizes only the `n` base
    /// points. The extra seven inversions are shared by the whole batch, so
    /// they vanish against even a modest `n`.
    ///
    /// Identity inputs produce identity tables and may be mixed with
    /// non-identity points in the same batch.
    ///
    /// # Exceptional cases
    ///
    /// An affine addition fails when the operands share an `x`. Every step of
    /// this chain adds two multiples `u P` and `v P` with `u, v` Eisenstein
    /// integers of norm at most 19, so failure needs `(u -+ v) P = O` with
    /// `u -+ v` a nonzero element of norm well below the group order, which
    /// forces `P = O`. The first step is the sharp case: `P - \varphi(P)` has
    /// denominator `(\zeta - 1) x`, so it fails exactly when `x = 0`, and a
    /// curve point with `x = 0` would satisfy `\varphi(P) = P`, hence
    /// `(1 - \omega) P = O`, hence be 3-torsion; both Pasta groups have prime
    /// order not divisible by 3, so no such point exists (equivalently, 5 is a
    /// non-residue in both base fields, which `no_point_has_zero_x` checks).
    ///
    /// Lanes that do fail, which is to say identity inputs, fall back to the
    /// projective chain.
    ///
    /// Verified in `sage/glv_eisenstein.sage` by step `[4c]`
    /// (`check_affine_chain_safety`), which checks the non-residue argument
    /// and bounds every step's operand difference by norm 19.
    pub fn batch(points: &[C]) -> Vec<Table<C>> {
        let n = points.len();
        if n == 0 {
            return Vec::new();
        }
        if n < AFFINE_CHAIN_THRESHOLD {
            return Self::batch_projective(points);
        }

        // One inversion brings the base points affine; the chain then stays
        // there.
        let mut base = alloc::vec![C::AffineExt::identity(); n];
        C::batch_normalize(points, &mut base);

        let zero = C::Base::ZERO;
        let buf = || alloc::vec![zero; n];
        let (mut px, mut py) = (buf(), buf());
        let mut live = alloc::vec![true; n];
        for (i, a) in base.iter().enumerate() {
            let (x, y) = C::affine_xy(a);
            px[i] = x;
            py[i] = y;
            live[i] = !bool::from(a.is_identity());
        }

        let mut adder = LaneAdder {
            den: buf(),
            scratch: buf(),
            live: &mut live,
        };
        // phi fixes y, so a rotation is one multiplication on x alone.
        let mut phix = buf();
        for (o, x) in phix.iter_mut().zip(px.iter()) {
            *o = *x * C::Base::ZETA;
        }

        // d1 = P - phi(P)
        let (mut d1x, mut d1y) = (buf(), buf());
        adder.add((&px, &py), (&phix, &py), true, (&mut d1x, &mut d1y));

        // b = d1 - phi(d1)
        let mut phid1x = buf();
        for (o, x) in phid1x.iter_mut().zip(d1x.iter()) {
            *o = *x * C::Base::ZETA;
        }
        let (mut bx, mut by) = (buf(), buf());
        adder.add((&d1x, &d1y), (&phid1x, &d1y), true, (&mut bx, &mut by));

        // r3 = -phi(b) = (zeta bx, -by) and m3 = phi(phi(b)) = (zeta^2 bx, by)
        let (mut r3x, mut m3x) = (buf(), buf());
        for i in 0..n {
            r3x[i] = bx[i] * C::Base::ZETA;
            m3x[i] = r3x[i] * C::Base::ZETA;
        }

        // The four additions against phi(P).
        let (mut t3ax, mut t3ay) = (buf(), buf());
        adder.add((&m3x, &by), (&phix, &py), false, (&mut t3ax, &mut t3ay));
        let (mut t3bx, mut t3by) = (buf(), buf());
        adder.add((&phix, &py), (&m3x, &by), true, (&mut t3bx, &mut t3by));
        let (mut t4ax, mut t4ay) = (buf(), buf());
        adder.add((&phix, &py), (&r3x, &by), true, (&mut t4ax, &mut t4ay));
        let (mut t4bx, mut t4by) = (buf(), buf());
        adder.add((&phix, &py), (&r3x, &by), false, (&mut t4bx, &mut t4by));

        // t19 = t4b + phi(P)
        let (mut t19x, mut t19y) = (buf(), buf());
        adder.add((&t4bx, &t4by), (&phix, &py), false, (&mut t19x, &mut t19y));

        // The representatives, read off exactly as `window_proj` does.
        let mut tables = alloc::vec![
            Table {
                xs: [zero; REP_COUNT * ROTATIONS],
                ys: [zero; REP_COUNT],
            };
            n
        ];
        let mut exceptional = Vec::new();
        for (i, t) in tables.iter_mut().enumerate() {
            if !live[i] {
                exceptional.push(i);
                continue;
            }
            let z = C::Base::ZETA;
            let reps = [
                (px[i], py[i]),             // 1
                (d1x[i], d1y[i]),           // 1 - w
                (t4ax[i] * z, t4ay[i]),     // 2 - w   = phi(t4a)
                (t3bx[i] * z, -t3by[i]),    // 1 - 2w  = -phi(t3b)
                (m3x[i], -by[i]),           // 3       = -m3
                (t3ax[i], -t3ay[i]),        // 3 - w   = -t3a
                (t4bx[i] * z * z, t4by[i]), // 1 - 3w  = phi^2(t4b)
                (t19x[i] * z * z, t19y[i]), // 2 - 3w  = phi^2(t19)
            ];
            for (j, (x, y)) in reps.into_iter().enumerate() {
                t.ys[j] = y;
                t.xs[j * ROTATIONS] = x;
                for r in 1..ROTATIONS {
                    t.xs[j * ROTATIONS + r] = t.xs[j * ROTATIONS + r - 1] * z;
                }
            }
        }
        if !exceptional.is_empty() {
            let fallback: Vec<C> = exceptional.iter().map(|&i| points[i]).collect();
            for (&i, t) in exceptional.iter().zip(Self::batch_projective(&fallback)) {
                tables[i] = t;
            }
        }
        tables
    }

    /// The projective chain, one shared normalization of all `8 * n`
    /// representatives. Used for batches too small to amortize the affine
    /// chain's seven inversions, and for the exceptional lanes above.
    fn batch_projective(points: &[C]) -> Vec<Table<C>> {
        let n = points.len();
        if n == 0 {
            return Vec::new();
        }
        let mut proj = Vec::with_capacity(n * REP_COUNT);
        for p in points {
            proj.extend_from_slice(&Self::window_proj(p));
        }
        let mut affine = alloc::vec![C::AffineExt::identity(); n * REP_COUNT];
        C::batch_normalize(&proj, &mut affine);
        affine
            .chunks_exact(REP_COUNT)
            .map(Self::from_window)
            .collect()
    }

    /// Fills the 24 slots from the 8 normalized representatives.
    ///
    /// The other five units act as $(x, y) \mapsto (\zeta^i x, \pm y)$, and
    /// negation is applied in the ladder, so a rotation costs exactly one
    /// base-field multiplication and no second normalization. The identity is
    /// stored as $(0, 0)$, a fixed point of $x \mapsto \zeta x$, so identity
    /// tables rotate to themselves.
    fn from_window(w: &[C::AffineExt]) -> Self {
        let mut xs = [C::Base::ZERO; REP_COUNT * ROTATIONS];
        let mut ys = [C::Base::ZERO; REP_COUNT];
        for (j, a) in w.iter().enumerate() {
            let (x, y) = C::affine_xy(a);
            ys[j] = y;
            xs[j * ROTATIONS] = x;
            for i in 1..ROTATIONS {
                xs[j * ROTATIONS + i] = xs[j * ROTATIONS + i - 1] * C::Base::ZETA;
            }
        }
        Table { xs, ys }
    }

    /// The 24 projective window entries for one point.
    ///
    /// The addition chain reaches all eight orbit representatives in **seven
    /// additions**, using $\varphi$ (one base-field multiplication, and
    /// $\varphi(P) = \omega P$) to move between them for free wherever
    /// possible:
    ///
    /// ```text
    ///   phi_p = phi(P)            = w P
    ///   d1    = P - phi_p         = (1 - w) P          add 1
    ///   b     = d1 - phi(d1)      = -3w P              add 2
    ///   phi_b = phi(b)            = -3w^2 P
    ///   m3    = phi(phi_b)        = -3 P
    ///   r3    = -phi_b            = 3w^2 P
    ///   t3a   = m3 + phi_p        = (-3 + w) P         add 3
    ///   t3b   = phi_p - m3        = (3 + w) P          add 4
    ///   t4a   = phi_p + r3        = (-3 - 2w) P        add 5
    ///   t4b   = phi_p - r3        = (3 + 4w) P         add 6
    ///   t19   = t4b + phi_p       = (3 + 5w) P         add 7
    /// ```
    ///
    /// with the representatives then read off as `P`, `d1`, `phi(t4a)`,
    /// `-phi(t3b)`, `-m3`, `-t3a`, `phi^2(t4b)`, `phi^2(t19)`. Every identity
    /// in that chain is re-derived symbolically in
    /// the `window_matches_representatives` test.
    ///
    /// Verified in `sage/glv_eisenstein.sage` by step `[4a]`
    /// (`window_chain`, symbolically in $\mathbb{Z}[\omega]$) and step
    /// `[4b]` (`check_curve`, the same chain as point arithmetic on Pallas
    /// and Vesta).
    fn window_proj(p: &C) -> [C; REP_COUNT] {
        let phi_p = p.endo();
        let d1 = *p - phi_p;
        let b = d1 - d1.endo();
        let phi_b = b.endo();
        let m3 = phi_b.endo();
        let r3 = -phi_b;
        let t3a = m3 + phi_p;
        let t3b = phi_p - m3;
        let t4a = phi_p + r3;
        let t4b = phi_p - r3;
        let t19 = t4b + phi_p;

        [
            *p,                // 1
            d1,                // 1 - w
            t4a.endo(),        // 2 - w
            -t3b.endo(),       // 1 - 2w
            -m3,               // 3
            -t3a,              // 3 - w
            t4b.endo().endo(), // 1 - 3w
            t19.endo().endo(), // 2 - 3w
        ]
    }

    /// The base point P back as a projective point.
    #[cfg(test)]
    fn point(&self) -> C {
        C::from(C::affine_from_xy_unchecked(self.xs[0], self.ys[0]))
    }

    /// `k * P` for the P encoded by this table, recoding `k` on the spot.
    ///
    /// When one scalar meets many tables, recode once with [`Recoded::new`]
    /// and use [`Table::mul_recoded`] instead.
    pub fn mul(&self, k: &C::ScalarExt) -> C {
        self.mul_recoded(&Recoded::new(k))
    }

    /// `k * P` for the P encoded by this table, via the joint Eisenstein-NAF
    /// ladder. Identical to `P * k` (tested).
    ///
    /// Exactly one point addition per nonzero column, and one doubling per
    /// column after the first.
    pub fn mul_recoded(&self, k: &Recoded<C>) -> C {
        let mut acc = C::identity();
        for i in (0..k.len).rev() {
            // `acc` is still the identity on the first iteration; skip the
            // wasted doubling.
            if i + 1 < k.len {
                acc = acc.double();
            }
            let d = k.digits[i];
            if d != ZERO_DIGIT {
                acc += self.digit_point(d);
            }
        }
        acc
    }

    /// The affine coordinates of the point a packed nonzero digit names, with
    /// the digit's sign folded into `y`.
    ///
    /// Two array reads and a conditional negation: the unit action is already
    /// baked into the layout, and $\varphi$ fixes $y$.
    fn digit_xy(&self, d: u8) -> (C::Base, C::Base) {
        let rep = (d & 7) as usize;
        let x = self.xs[rep * ROTATIONS + ((d >> 3) & 3) as usize];
        let y = self.ys[rep];
        if d & NEG_BIT != 0 { (x, -y) } else { (x, y) }
    }

    /// The same point, as an affine point.
    fn digit_point(&self, d: u8) -> C::AffineExt {
        let (x, y) = self.digit_xy(d);
        C::affine_from_xy_unchecked(x, y)
    }

    /// Multiplies a whole batch of points by **one** recoded scalar, choosing
    /// the cheaper ladder for the batch size.
    ///
    /// The results come back affine, which is what a key-agreement KDF needs
    /// anyway. Identity inputs are handled and mix freely with ordinary
    /// points.
    ///
    /// # Which ladder, and why it depends on the size
    ///
    /// Counting a squaring as a multiplication, the per-point ladder costs are
    ///
    /// - projective (Jacobian): $125 \times 7 + 38.4 \times 11 \approx 1297$
    ///   multiplications;
    /// - batch affine: $87.6 \times 7 + 38.4 \times 11 \approx 1035$, plus
    ///   $164\,I/n$ for a batch of $n$, where $I$ is the cost of one field
    ///   inversion, since the 164 inversions of a ladder are shared.
    ///
    /// So on ladder work alone the affine form wins once $n > 164 I / 262
    /// \approx 0.63\,I$. The inversion used is
    /// [`VartimeField::invert_vartime`], the safegcd one, at $I \approx 63$
    /// multiplications, which puts that near 40; the measured end-to-end
    /// crossover is a little higher, level at 64 and clear by 96, because
    /// building a table is 7 projective additions either way and the affine
    /// ladder does nothing for it.
    ///
    /// $I$ is what sets the crossover, and it is the whole reason this path
    /// is worth having: against the constant-time Fermat inversion, at
    /// $I \approx 417$, the same code does not break even until a batch of
    /// about 420.
    ///
    /// Measured end to end against the split-wNAF GLV ladder on Pallas
    /// (recode, one table per point, ladder, affine out), taking whichever
    /// arm this function dispatches to: 9% faster at a batch of 16, 11% at
    /// 64, 13% at 128, 15% at 256 and 16% at 512. The affine arm is the one
    /// winning from 128 up.
    ///
    /// [`BATCH_AFFINE_THRESHOLD`] records the measured crossover, and
    /// [`Table::batch_mul_affine`] forces the affine ladder regardless.
    pub fn batch_mul(tables: &[Self], k: &Recoded<C>) -> Vec<C::AffineExt> {
        if tables.len() >= BATCH_AFFINE_THRESHOLD {
            return Self::batch_mul_affine(tables, k);
        }
        let proj: Vec<C> = tables.iter().map(|t| t.mul_recoded(k)).collect();
        let mut aff = alloc::vec![C::AffineExt::identity(); proj.len()];
        C::batch_normalize(&proj, &mut aff);
        aff
    }

    /// Multiplies a whole batch of points by **one** recoded scalar, always
    /// on the batch-affine ladder.
    ///
    /// Prefer [`Table::batch_mul`], which picks this only when the batch is
    /// large enough to pay for it (see [`BATCH_AFFINE_THRESHOLD`]). Call this
    /// directly if you know the field's inversion is cheap.
    ///
    /// Every accumulator is kept in *affine* coordinates, with a single field
    /// inversion shared across the batch at each ladder column.
    ///
    /// This is the shape wallet scanning has: one incoming viewing key against
    /// many ephemeral keys. Because the scalar is shared, every lane executes
    /// the same column at the same time, so the inversion that makes affine
    /// arithmetic impractical for a single multiplication is amortized over
    /// the batch by Montgomery's trick and costs about three multiplications
    /// per lane.
    ///
    /// That changes which recoding is best. A column here is either a
    /// doubling, which is roughly break-even against Jacobian, or a *fused*
    /// $2A + D$, which is markedly cheaper, so what matters is minimizing
    /// additions and never needing two in one column. Both are exactly what
    /// the Eisenstein recoding delivers.
    ///
    /// The fused column uses the Eisentrager-Lauter-Montgomery trick: reading
    /// $2A + D$ as $(A + D) + A$ lets the intermediate point's
    /// $y$-coordinate go uncomputed, saving a multiplication, and replaces a
    /// separate doubling and addition with two slopes.
    ///
    /// The results come back affine, which for key agreement is what the KDF
    /// needs anyway, so the usual final normalization is already paid for.
    ///
    /// Identity inputs are handled, and mix freely with ordinary points.
    pub fn batch_mul_affine(tables: &[Self], k: &Recoded<C>) -> Vec<C::AffineExt> {
        let n = tables.len();
        let mut out = alloc::vec![C::AffineExt::identity(); n];
        if n == 0 || k.len == 0 {
            return out;
        }

        let zero = C::Base::ZERO;
        let (mut xs, mut ys) = (alloc::vec![zero; n], alloc::vec![zero; n]);
        let (mut x2s, mut y2s) = (alloc::vec![zero; n], alloc::vec![zero; n]);
        let (mut x3s, mut lam) = (alloc::vec![zero; n], alloc::vec![zero; n]);
        let (mut den, mut scratch) = (alloc::vec![zero; n], alloc::vec![zero; n]);
        // A lane leaves the affine path when a denominator vanishes, which
        // means its accumulator is the identity (see `batch_mul` in the tests
        // and the exception analysis in the module docs).
        let mut live = alloc::vec![true; n];

        // The top column always carries a nonzero digit, so the accumulator
        // starts at that digit's point rather than at the identity, which
        // affine coordinates cannot represent.
        let top = k.digits[k.len - 1];
        for (i, t) in tables.iter().enumerate() {
            let (x, y) = t.digit_xy(top);
            xs[i] = x;
            ys[i] = y;
        }

        for col in (0..k.len - 1).rev() {
            let d = k.digits[col];
            if d == ZERO_DIGIT {
                // Doubling: lambda = 3x^2 / 2y.
                for i in 0..n {
                    den[i] = if live[i] { ys[i].double() } else { zero };
                }
                batch_invert(&mut den, &mut scratch);
                for i in 0..n {
                    if !live[i] || bool::from(den[i].is_zero()) {
                        live[i] = false;
                        continue;
                    }
                    let xx = xs[i].square();
                    let l = (xx.double() + xx) * den[i];
                    let x = l.square() - xs[i].double();
                    ys[i] = l * (xs[i] - x) - ys[i];
                    xs[i] = x;
                }
            } else {
                // Fused 2A + D = (A + D) + A. First slope, and the x of the
                // intermediate A + D; its y is never formed.
                for i in 0..n {
                    let (x2, y2) = tables[i].digit_xy(d);
                    x2s[i] = x2;
                    y2s[i] = y2;
                    den[i] = if live[i] { x2 - xs[i] } else { zero };
                }
                batch_invert(&mut den, &mut scratch);
                for i in 0..n {
                    if !live[i] || bool::from(den[i].is_zero()) {
                        live[i] = false;
                        continue;
                    }
                    let l1 = (y2s[i] - ys[i]) * den[i];
                    x3s[i] = l1.square() - xs[i] - x2s[i];
                    lam[i] = l1;
                }
                // Second slope, lambda2 = -lambda1 - 2*y1 / (x3 - x1).
                for i in 0..n {
                    den[i] = if live[i] { x3s[i] - xs[i] } else { zero };
                }
                batch_invert(&mut den, &mut scratch);
                for i in 0..n {
                    if !live[i] || bool::from(den[i].is_zero()) {
                        live[i] = false;
                        continue;
                    }
                    let l2 = -lam[i] - ys[i].double() * den[i];
                    let x4 = l2.square() - xs[i] - x3s[i];
                    ys[i] = l2 * (xs[i] - x4) - ys[i];
                    xs[i] = x4;
                }
            }
        }

        let mut exceptional = Vec::new();
        for i in 0..n {
            if live[i] {
                out[i] = C::affine_from_xy_unchecked(xs[i], ys[i]);
            } else {
                exceptional.push(i);
            }
        }
        if !exceptional.is_empty() {
            // Rerun those lanes on the projective ladder, which has no
            // exceptional cases, and normalize them together.
            let proj: Vec<C> = exceptional
                .iter()
                .map(|&i| tables[i].mul_recoded(k))
                .collect();
            let mut aff = alloc::vec![C::AffineExt::identity(); proj.len()];
            C::batch_normalize(&proj, &mut aff);
            for (&i, a) in exceptional.iter().zip(aff.iter()) {
                out[i] = *a;
            }
        }
        out
    }
}

/// A scalar in GLV-decomposed, Eisenstein-NAF-recoded form, ready for
/// [`Table::mul_recoded`].
///
/// Building this once per scalar hoists the decomposition and recoding out of
/// a loop that multiplies the same scalar against many tables, the wallet
/// trial-decryption shape, where one incoming viewing key meets a batch of
/// ephemeral keys.
#[derive(Clone, Debug)]
pub struct Recoded<C: GlvParams> {
    digits: [u8; MAX_DIGITS],
    /// Columns in use; `digits` is [`ZERO_DIGIT`] beyond it.
    len: usize,
    _curve: core::marker::PhantomData<C>,
}

impl<C: GlvParams> Recoded<C> {
    /// Decomposes `k` and recodes the resulting Eisenstein integer as a joint
    /// width-3 NAF.
    pub fn new(k: &C::ScalarExt) -> Self {
        let ((neg1, a1), (neg2, a2)) = decompose::<C>(k);
        let (digits, len) = recode(signed(neg1, a1), signed(neg2, a2));
        Recoded {
            digits,
            len,
            _curve: core::marker::PhantomData,
        }
    }

    /// The number of columns, i.e. the ladder length.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the recoding is empty, which happens exactly for `k == 0`.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The number of nonzero columns, i.e. the number of point additions
    /// [`Table::mul_recoded`] will perform.
    #[cfg(test)]
    fn additions(&self) -> usize {
        self.digits[..self.len]
            .iter()
            .filter(|&&d| d != ZERO_DIGIT)
            .count()
    }
}

/// Reassembles a [`decompose`] half into a signed integer.
///
/// `decompose` bounds the magnitude below `2^127`, so the cast is exact; the
/// tighter Babai bound `(V1A + V2A) / 2 < 2^126.6` (checked for both curves by
/// the
/// `half_bounds` tests) is what leaves room for the digit
/// subtraction in [`recode`] to stay inside `i128`.
fn signed(negative: bool, magnitude: u128) -> i128 {
    debug_assert!(magnitude >> 127 == 0, "GLV half exceeds 127 bits");
    let m = magnitude as i128;
    if negative { -m } else { m }
}

/// Width-3 Eisenstein NAF of `a + b*omega`, lowest column first.
///
/// Each column either halves an even residual (both coordinates even, since 2
/// is inert) and emits [`ZERO_DIGIT`], or subtracts the orbit-canonical
/// representative of the residual's class mod 8, which makes the difference
/// divisible by 8, and then halves.
///
/// Verified in `sage/glv_eisenstein.sage` by step `[6]`: `recode` is the
/// same recoding, and `check_recoder` runs the resulting ladder on both
/// curves and checks it against `k * P`.
fn recode(mut a: i128, mut b: i128) -> ([u8; MAX_DIGITS], usize) {
    let mut digits = [ZERO_DIGIT; MAX_DIGITS];
    let mut n = 0;
    while a != 0 || b != 0 {
        debug_assert!(n < MAX_DIGITS, "Eisenstein NAF longer than its bound");
        if (a | b) & 1 == 0 {
            // Even in Z[omega]: halving is exact and coordinate-wise.
            digits[n] = ZERO_DIGIT;
        } else {
            let e = LUT[((a & 7) as usize) * 8 + ((b & 7) as usize)];
            debug_assert!(e.digit != ZERO_DIGIT, "odd class must have a digit");
            digits[n] = e.digit;
            a -= i128::from(e.da);
            b -= i128::from(e.db);
            debug_assert!(a & 7 == 0 && b & 7 == 0, "digit must clear three bits");
        }
        // Arithmetic shift: exact division by two for even values of either
        // sign, and both coordinates are even here.
        a >>= 1;
        b >>= 1;
        n += 1;
    }
    (digits, n)
}

/// A batch of affine points held as parallel `(x, y)` coordinate arrays.
type Lanes<'a, F> = (&'a [F], &'a [F]);

/// The same, writable.
type LanesMut<'a, F> = (&'a mut [F], &'a mut [F]);

/// Batch size from which [`Table::batch`] builds its window with the affine
/// chain rather than the projective one.
///
/// The affine chain spends seven field inversions per batch whatever its size,
/// against the projective chain's one, and saves roughly ten multiplications
/// per addition per lane. With the safegcd inversion that pays from a handful
/// of points; the value is set well clear of the break-even.
const AFFINE_CHAIN_THRESHOLD: usize = 8;

/// Lane-wise affine point addition sharing one field inversion per step.
struct LaneAdder<'a, F> {
    den: Vec<F>,
    scratch: Vec<F>,
    /// Cleared for a lane whose addition is exceptional; see [`Table::batch`].
    live: &'a mut [bool],
}

impl<F: Field + VartimeField> LaneAdder<'_, F> {
    /// `out = p + q` lane-wise, or `p - q` when `neg_q` is set, with one
    /// inversion for the whole batch.
    fn add(&mut self, p: Lanes<'_, F>, q: Lanes<'_, F>, neg_q: bool, out: LanesMut<'_, F>) {
        let (px, py) = p;
        let (qx, qy) = q;
        let (ox, oy) = out;
        for (i, d) in self.den.iter_mut().enumerate() {
            *d = if self.live[i] { qx[i] - px[i] } else { F::ZERO };
        }
        batch_invert(&mut self.den, &mut self.scratch);
        for i in 0..px.len() {
            if !self.live[i] || bool::from(self.den[i].is_zero()) {
                self.live[i] = false;
                continue;
            }
            let qyi = if neg_q { -qy[i] } else { qy[i] };
            let l = (qyi - py[i]) * self.den[i];
            let x = l.square() - px[i] - qx[i];
            oy[i] = l * (px[i] - x) - py[i];
            ox[i] = x;
        }
    }
}

/// Inverts every nonzero element of `v` in place with a single field
/// inversion (Montgomery's trick), leaving zeros as zero.
///
/// Skipping zeros rather than letting one poison the running product is what
/// lets the batch ladder detect an exceptional lane: after the call, a zero is
/// exactly a denominator that was zero going in.
///
/// The one inversion is [`VartimeField::invert_vartime`], the safegcd path,
/// which is about six times faster than the constant-time Fermat
/// exponentiation and is what makes the batch-affine ladder pay off at
/// ordinary batch sizes. It leaks nothing this module does not already leak:
/// the recoding's digit pattern is what drives the branch structure, and it
/// is a function of the same scalar.
fn batch_invert<F: Field + VartimeField>(v: &mut [F], scratch: &mut [F]) {
    debug_assert!(scratch.len() >= v.len());
    let mut acc = F::ONE;
    for (e, s) in v.iter().zip(scratch.iter_mut()) {
        *s = acc;
        if !bool::from(e.is_zero()) {
            acc *= e;
        }
    }
    // A product of nonzero field elements (or the empty product), so nonzero.
    acc = acc
        .invert_vartime()
        .expect("product of nonzero field elements");
    for (e, s) in v.iter_mut().zip(scratch.iter()).rev() {
        if !bool::from(e.is_zero()) {
            let inv = acc * s;
            acc *= *e;
            *e = inv;
        }
    }
}

/// One-shot `k * p` through the Eisenstein recoding: variable-time in `k`,
/// identical in value to `p * k` (including `p` = identity).
///
/// For repeated multiplications against the same point or the same scalar,
/// use [`Table`] / [`Recoded`] directly to reuse the precomputation.
pub fn mul<C: GlvParams>(p: &C, k: &C::ScalarExt) -> C {
    if bool::from(p.is_identity()) {
        // k*O = O. Identity tables work, but building one still costs a
        // field inversion; short-circuit.
        return C::identity();
    }
    Table::new(p).mul(k)
}

/// `k * p` for every `p`, sharing one field inversion across the batch at
/// every ladder column and returning affine results.
///
/// This is [`Table::batch_mul`] with the per-point tables and the recoding
/// built for you: it is the one call wallet scanning wants, since the tables
/// also share a single inversion. It picks the ladder by batch size, so it is
/// never slower than the projective path. Identity inputs are handled.
pub fn batch_mul<C: GlvParams>(points: &[C], k: &C::ScalarExt) -> Vec<C::AffineExt> {
    Table::batch_mul(&Table::batch(points), &Recoded::new(k))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ff::{Field, PrimeField, WithSmallOrderMulGroup};

    /// Multiplies an Eisenstein integer by a unit $\pm\omega^{\mathrm{rot}}$,
    /// in plain integer arithmetic, the test-side mirror of [`rotate`].
    fn unit_mul(d: (i32, i32), rot: usize, neg: bool) -> (i32, i32) {
        let mut d = d;
        for _ in 0..rot {
            d = (-d.1, d.0 - d.1);
        }
        if neg { (-d.0, -d.1) } else { d }
    }

    /// The 48 odd residue classes of `Z[omega]/8`, as `(a, b)` in `0..8`.
    fn odd_classes() -> impl Iterator<Item = (i32, i32)> {
        (0..8)
            .flat_map(|a| (0..8).map(move |b| (a, b)))
            .filter(|(a, b)| (a | b) & 1 == 1)
    }

    /// $a + b\omega$ evaluated in the scalar field, i.e. $a + b\lambda$.
    fn to_scalar<F: WithSmallOrderMulGroup<3>>(a: i32, b: i32) -> F {
        let f = |v: i32| {
            let m = F::from(v.unsigned_abs() as u64);
            if v < 0 { -m } else { m }
        };
        f(a) + f(b) * F::ZETA
    }

    /// The count is the arithmetic content of the whole design: 64 classes
    /// mod 8, 16 even, 48 odd, and $48 = 8 \times 6$.
    #[test]
    fn lut_covers_exactly_the_odd_classes() {
        assert_eq!(odd_classes().count(), 48);
        assert_eq!(REP_COUNT * ROTATIONS * 2, 48, "8 reps x 6 units");

        for (a, b) in odd_classes() {
            let e = LUT[(a as usize) * 8 + b as usize];
            assert_ne!(e.digit, ZERO_DIGIT, "odd class ({a}, {b}) has no digit");
        }
        // The even classes are untouched: the recoder's parity branch owns them.
        for a in (0..8).step_by(2) {
            for b in (0..8).step_by(2) {
                let e = LUT[(a as usize) * 8 + b as usize];
                assert_eq!(e.digit, ZERO_DIGIT, "even class ({a}, {b}) has a digit");
                assert_eq!((e.da, e.db), (0, 0));
            }
        }
    }

    /// The mu_6 action on the 48 odd classes is FREE: every orbit has the full
    /// six elements, so `48 / 6 = 8` orbits exist and eight stored points
    /// suffice. This is the fact that makes the table small, and it is what
    /// the Lean development (`Rings/Eisenstein/Orbits.lean`) proves.
    #[test]
    fn orbit_action_is_free() {
        for c in odd_classes() {
            let orbit: alloc::collections::BTreeSet<(i32, i32)> = (0..ROTATIONS)
                .flat_map(|rot| [false, true].map(move |neg| unit_mul(c, rot, neg)))
                .map(|(a, b)| (a.rem_euclid(8), b.rem_euclid(8)))
                .collect();
            assert_eq!(orbit.len(), 6, "orbit of {c:?} is not free");
        }
    }

    /// Every LUT entry is (a) congruent to its own index mod 8, so subtracting
    /// it clears three bits, and (b) literally the unit multiple of the
    /// representative that its packed `(rep, rot, neg)` names.
    #[test]
    fn lut_entries_are_consistent() {
        let mut seen = alloc::collections::BTreeSet::new();
        for (a, b) in odd_classes() {
            let e = LUT[(a as usize) * 8 + b as usize];
            assert_eq!(
                (i32::from(e.da).rem_euclid(8), i32::from(e.db).rem_euclid(8)),
                (a, b),
                "digit is not congruent to its class"
            );

            let rep = (e.digit & 7) as usize;
            let rot = ((e.digit >> 3) & 3) as usize;
            let neg = e.digit & (1 << 5) != 0;
            assert!(rep < REP_COUNT && rot < ROTATIONS);
            let r = REPS[rep];
            assert_eq!(
                unit_mul((i32::from(r.0), i32::from(r.1)), rot, neg),
                (i32::from(e.da), i32::from(e.db)),
                "packed digit does not name its own value"
            );
            assert!(seen.insert(e.digit), "packed digit {} reused", e.digit);
        }
        assert_eq!(seen.len(), 48);
    }

    /// The bound behind [`MAX_DIGITS`]: digit coefficients never exceed 5, so
    /// the residual obeys `|x'| <= (|x| + 5) / 2`.
    #[test]
    fn digit_coefficients_are_bounded() {
        let d = LUT
            .iter()
            .filter(|e| e.digit != ZERO_DIGIT)
            .map(|e| e.da.abs().max(e.db.abs()))
            .max()
            .expect("48 digits");
        assert_eq!(d, 5, "MAX_DIGITS is derived from this bound");

        // Iterate |x| <- (|x| + 5)/2 from 2^127 down into the box |x| <= 12,
        // which `sage/glv_eisenstein.sage` verifies drains in 6 more columns.
        let mut m: u128 = (1 << 127) - 1;
        let mut n = 0usize;
        while m > 12 {
            m = (m + 5) / 2;
            n += 1;
        }
        assert_eq!(n + 6, MAX_DIGITS, "MAX_DIGITS must match its derivation");
    }

    /// The addition chain in [`Table::window_proj`], re-derived symbolically in
    /// `Z[omega]`, produces exactly [`REPS`], in seven additions.
    #[test]
    fn window_matches_representatives() {
        type E = (i32, i32);
        let add = |p: E, q: E| (p.0 + q.0, p.1 + q.1);
        let sub = |p: E, q: E| (p.0 - q.0, p.1 - q.1);
        let neg = |p: E| (-p.0, -p.1);
        let phi = |p: E| unit_mul(p, 1, false);

        let p: E = (1, 0);
        let phi_p = phi(p);
        let d1 = sub(p, phi_p);
        let b = sub(d1, phi(d1));
        let phi_b = phi(b);
        let m3 = phi(phi_b);
        let r3 = neg(phi_b);
        let t3a = add(m3, phi_p);
        let t3b = sub(phi_p, m3);
        let t4a = add(phi_p, r3);
        let t4b = sub(phi_p, r3);
        let t19 = add(t4b, phi_p);

        // The chain's intermediate claims, as documented on `window_proj`.
        assert_eq!(b, (0, -3), "b must be -3w");
        assert_eq!(m3, (-3, 0), "m3 must be -3");
        assert_eq!(r3, (-3, -3), "r3 must be 3w^2");

        let built = [
            p,
            d1,
            phi(t4a),
            neg(phi(t3b)),
            neg(m3),
            neg(t3a),
            phi(phi(t4b)),
            phi(phi(t19)),
        ];
        let want: Vec<E> = REPS
            .iter()
            .map(|r| (i32::from(r.0), i32::from(r.1)))
            .collect();
        assert_eq!(built.to_vec(), want, "addition chain must reach the reps");
    }

    /// The Babai bound leaves room inside `i128` for the digit subtraction in
    /// [`recode`]: `|k1| <= (V1A + V2A)/2` and `|k2| <= (V1B_NEG + V2B)/2`,
    /// both comfortably under `2^127 - 5`.
    fn halves_leave_room_for_a_digit<C: GlvParams>() {
        let headroom = i128::MAX as u128; // 2^127 - 1
        for bound in [
            (C::V1A / 2) + (C::V2A / 2) + 1,
            (C::V1B_NEG / 2) + (C::V2B / 2) + 1,
        ] {
            assert!(bound < headroom - 5, "GLV half can overflow i128");
        }
    }

    /// Deterministic full-width scalars (matching `glv`'s own test suite).
    fn scalars<F: PrimeField>(n: u64) -> impl Iterator<Item = F> {
        (0..n).map(|i| {
            (F::from(0x9E37_79B9_7F4A_7C15u64 + i).square() + F::from(0x0123_4567_89AB_CDEFu64))
                .square()
                + F::from(i)
        })
    }

    /// The recoding is an exact representation: evaluating the digit string
    /// Horner-wise in the scalar field returns `k`.
    fn recoding_reconstructs<C: GlvParams>() {
        let check = |k: C::ScalarExt| {
            let r = Recoded::<C>::new(&k);
            let mut acc = C::ScalarExt::ZERO;
            for i in (0..r.len).rev() {
                acc = acc.double();
                let d = r.digits[i];
                if d != ZERO_DIGIT {
                    let rep = (d & 7) as usize;
                    let rot = ((d >> 3) & 3) as usize;
                    let neg = d & (1 << 5) != 0;
                    let rp = REPS[rep];
                    let (a, b) = unit_mul((i32::from(rp.0), i32::from(rp.1)), rot, neg);
                    acc += to_scalar::<C::ScalarExt>(a, b);
                }
            }
            assert_eq!(acc, k, "digit string must evaluate back to k");
        };
        check(C::ScalarExt::ZERO);
        check(C::ScalarExt::ONE);
        check(-C::ScalarExt::ONE);
        check(C::ScalarExt::ZETA);
        check(-C::ScalarExt::ZETA);
        for k in scalars::<C::ScalarExt>(500) {
            check(k);
        }
        assert!(Recoded::<C>::new(&C::ScalarExt::ZERO).is_empty());
    }

    /// Each table slot holds the scalar multiple it claims:
    /// `entries[rep][rot] == (omega^rot * r_rep) * P`.
    fn table_entries_are_the_right_multiples<C: GlvParams>() {
        let p = C::generator() * C::ScalarExt::from(0xDEAD_BEEFu64);
        let table = Table::new(&p);
        for (rep, r) in REPS.iter().enumerate() {
            for rot in 0..ROTATIONS {
                for neg in [false, true] {
                    let (a, b) = unit_mul((i32::from(r.0), i32::from(r.1)), rot, neg);
                    let d = (rep as u8) | ((rot as u8) << 3) | (u8::from(neg) << 5);
                    assert_eq!(
                        C::from(table.digit_point(d)),
                        p * to_scalar::<C::ScalarExt>(a, b),
                        "table slot (rep {rep}, rot {rot}, neg {neg}) is wrong"
                    );
                }
            }
        }
    }

    /// Table-based multiplication matches the group's native `Mul`, and the
    /// split-wNAF GLV path.
    fn table_mul_matches_group_mul<C: GlvParams>() {
        let g = C::generator();
        for (i, k) in scalars::<C::ScalarExt>(48).enumerate() {
            let p = g * (k + C::ScalarExt::from(i as u64 + 1));
            let table = Table::new(&p);
            for k2 in scalars::<C::ScalarExt>(4) {
                assert_eq!(table.mul(&k2), p * k2, "table mul must match group mul");
                assert_eq!(table.mul(&k2), p.mul_glv(&k2), "must match wNAF GLV");
            }
            // The small scalars are where a recoding's tail goes wrong.
            for small in [0u64, 1, 2, 3, 7, 8, 15, 16, 255, u64::MAX] {
                let k2 = C::ScalarExt::from(small);
                assert_eq!(table.mul(&k2), p * k2, "table mul must match at {small}");
                assert_eq!(table.mul(&-k2), p * -k2, "table mul must match at -{small}");
            }
        }
    }

    /// One-shot [`mul`] matches the native operator, including on the identity.
    fn mul_matches_operator<C: GlvParams>() {
        let g = C::generator();
        for k in scalars::<C::ScalarExt>(48) {
            let p = g * (k + C::ScalarExt::ONE);
            assert_eq!(mul(&p, &k), p * k, "mul must match operator");
        }
        assert_eq!(
            mul(&C::identity(), &C::ScalarExt::from(7)),
            C::identity(),
            "k * O must be O"
        );
    }

    /// The batched table build equals the solo build, point by point.
    fn batch_tables_equal_solo<C: GlvParams>() {
        let g = C::generator();
        let points: Vec<C> = scalars::<C::ScalarExt>(16)
            .map(|k| g * (k + C::ScalarExt::ONE))
            .collect();
        let batched = Table::batch(&points);
        assert_eq!(batched.len(), points.len());
        let k = C::ScalarExt::from(0xDEAD_BEEFu64);
        for (p, table) in points.iter().zip(batched.iter()) {
            let solo = Table::new(p);
            assert_eq!(table.point(), solo.point());
            assert_eq!(
                table.mul(&k),
                solo.mul(&k),
                "batched table must act like solo"
            );
            assert_eq!(table.mul(&k), *p * k);
        }
        assert!(Table::<C>::batch(&[]).is_empty());
    }

    /// Identity tables work both alone and alongside non-identity points.
    fn identity_tables<C: GlvParams>() {
        let identity = C::identity();
        let generator = C::generator();
        let k = C::ScalarExt::from(0xDEAD_BEEFu64);

        let solo = Table::new(&identity);
        assert_eq!(solo.point(), identity);
        assert_eq!(solo.mul(&k), identity);

        let batched = Table::batch(&[identity, generator]);
        assert_eq!(batched.len(), 2);
        assert_eq!(batched[0].mul(&k), identity);
        assert_eq!(batched[1].mul(&k), generator * k);
    }

    /// A reused [`Recoded`] gives the same products as recoding per call.
    fn recoded_reuse_matches_fresh<C: GlvParams>() {
        let g = C::generator();
        let k = scalars::<C::ScalarExt>(1).next().expect("one scalar");
        let recoded = Recoded::<C>::new(&k);
        for k2 in scalars::<C::ScalarExt>(16) {
            let p = g * (k2 + C::ScalarExt::ONE);
            let table = Table::new(&p);
            assert_eq!(
                table.mul_recoded(&recoded),
                table.mul(&k),
                "hoisted recoding must match fresh"
            );
        }
    }

    /// Pins the cost model in the module docs: a mean of ~38.4 additions over
    /// ~126 columns (density 3/10), and lengths inside [`MAX_DIGITS`].
    fn digit_statistics<C: GlvParams>() {
        const N: u64 = 4000;
        let (mut cols, mut adds, mut max_cols, mut max_adds) = (0usize, 0usize, 0usize, 0usize);
        for k in scalars::<C::ScalarExt>(N) {
            let r = Recoded::<C>::new(&k);
            cols += r.len();
            adds += r.additions();
            max_cols = max_cols.max(r.len());
            max_adds = max_adds.max(r.additions());
        }
        let mean_cols = cols as f64 / N as f64;
        let mean_adds = adds as f64 / N as f64;
        std::println!(
            "columns mean {mean_cols:.2} max {max_cols}; \
             additions mean {mean_adds:.2} max {max_adds}"
        );
        assert!(max_cols <= MAX_DIGITS, "recoding exceeded its bound");
        assert!(
            (37.5..39.5).contains(&mean_adds),
            "mean additions {mean_adds} is off the 3/10 density model"
        );
        assert!(
            (124.0..128.0).contains(&mean_cols),
            "mean columns {mean_cols} is off model"
        );
        // Comfortably beats the split-wNAF ladder's ~51.2.
        assert!(mean_adds < 45.0);
    }

    /// No curve point has `x = 0`, which is what makes the first step of the
    /// affine table chain (`P - phi(P)`, denominator `(zeta - 1) x`) safe for
    /// every point but the identity.
    ///
    /// Such a point would satisfy `y^2 = b` and `phi(P) = P`, so `(1 - w)P =
    /// O` and `P` would be 3-torsion; the group has prime order not divisible
    /// by 3. Equivalently `b` is a non-residue, which is what is checked here.
    fn no_point_has_zero_x<C: GlvParams>() {
        assert!(
            bool::from(C::b().sqrt().is_none()),
            "a point with x = 0 would exist"
        );
        // And the order really is prime to 3, the other half of the argument.
        assert_ne!(C::ScalarExt::ZETA, C::ScalarExt::ONE);
    }

    /// The batched affine window chain agrees with the projective one, point
    /// for point, including identity lanes that take the fallback.
    ///
    /// This is the invariant the seven fused affine additions have to keep;
    /// the batch must be at least [`AFFINE_CHAIN_THRESHOLD`] for the affine
    /// chain to be the one under test.
    fn affine_chain_matches_projective<C: GlvParams>() {
        let g = C::generator();
        let mut points: Vec<C> = scalars::<C::ScalarExt>(20)
            .map(|k| g * (k + C::ScalarExt::from(3)))
            .collect();
        // Identity lanes, at the ends and in the middle, must take the
        // exceptional path and still come out right.
        points[0] = C::identity();
        points[9] = C::identity();
        points[19] = C::identity();
        assert!(points.len() >= AFFINE_CHAIN_THRESHOLD);

        let affine = Table::batch(&points);
        let projective = Table::<C>::batch_projective(&points);
        assert_eq!(affine.len(), points.len());
        let k = C::ScalarExt::from(0x5EED_5EEDu64);
        for ((p, a), q) in points.iter().zip(affine.iter()).zip(projective.iter()) {
            assert_eq!(a.point(), q.point(), "chains disagree on the base point");
            for rep in 0..REP_COUNT {
                for rot in 0..ROTATIONS {
                    for neg in [false, true] {
                        let d = (rep as u8) | ((rot as u8) << 3) | (u8::from(neg) << 5);
                        assert_eq!(
                            a.digit_xy(d),
                            q.digit_xy(d),
                            "chains disagree on digit (rep {rep}, rot {rot}, neg {neg})"
                        );
                    }
                }
            }
            assert_eq!(a.mul(&k), *p * k, "affine-chain table must multiply right");
        }
    }

    /// The batch affine ladder agrees with the group operator on every lane,
    /// across batch sizes, scalar shapes and identity inputs.
    fn batch_mul_matches_operator<C: GlvParams>() {
        let g = C::generator();

        // Empty batch, and the zero scalar (an empty recoding).
        assert!(batch_mul::<C>(&[], &C::ScalarExt::ONE).is_empty());
        for p in [C::identity(), g] {
            assert_eq!(
                batch_mul(&[p], &C::ScalarExt::ZERO),
                alloc::vec![C::AffineExt::identity()],
            );
        }

        let points: Vec<C> = scalars::<C::ScalarExt>(37)
            .map(|k| g * (k + C::ScalarExt::ONE))
            .collect();

        // Scalar shapes that stress the recoding's ends: a one-column ladder,
        // tiny values, negatives, lambda itself, and full-width scalars.
        let mut ks: Vec<C::ScalarExt> = alloc::vec![
            C::ScalarExt::ONE,
            -C::ScalarExt::ONE,
            C::ScalarExt::from(2),
            C::ScalarExt::from(3),
            C::ScalarExt::from(8),
            C::ScalarExt::from(u64::MAX),
            C::ScalarExt::ZETA,
            -C::ScalarExt::ZETA,
        ];
        ks.extend(scalars::<C::ScalarExt>(6));

        for k in &ks {
            let recoded = Recoded::<C>::new(k);
            for size in [1usize, 2, 3, 7, 37] {
                let batch = &points[..size];
                let tables = Table::batch(batch);
                // Both ladders, explicitly: `batch_mul` dispatches on size and
                // would otherwise never reach the affine one here.
                for got in [
                    batch_mul(batch, k),
                    Table::batch_mul_affine(&tables, &recoded),
                    Table::batch_mul(&tables, &recoded),
                ] {
                    assert_eq!(got.len(), size);
                    for (p, a) in batch.iter().zip(got.iter()) {
                        assert_eq!(C::from(*a), *p * k, "batch lane must match p * k");
                    }
                }
            }
        }

        // The exceptional path: identity inputs, alone and mixed in, must fall
        // back to the projective ladder and still be right.
        let k = scalars::<C::ScalarExt>(1).next().expect("one scalar");
        let mixed = alloc::vec![C::identity(), points[0], C::identity(), points[1]];
        let recoded = Recoded::<C>::new(&k);
        let tables = Table::batch(&mixed);
        for got in [
            batch_mul(&mixed, &k),
            Table::batch_mul_affine(&tables, &recoded),
        ] {
            for (p, a) in mixed.iter().zip(got.iter()) {
                assert_eq!(C::from(*a), *p * k, "identity lanes must still be right");
            }
            assert!(bool::from(got[0].is_identity()) && bool::from(got[2].is_identity()));
        }
        // An all-identity batch drives every lane down the exceptional path.
        let all_identity = alloc::vec![C::identity(); 5];
        for a in batch_mul(&all_identity, &k) {
            assert!(bool::from(a.is_identity()));
        }
    }

    /// [`Table::batch_mul`] agrees with the per-point projective ladder, which
    /// is the invariant the fused affine formulas have to preserve.
    fn batch_mul_matches_projective_ladder<C: GlvParams>() {
        let g = C::generator();
        let points: Vec<C> = scalars::<C::ScalarExt>(24)
            .map(|k| g * (k + C::ScalarExt::from(7)))
            .collect();
        let tables = Table::batch(&points);
        for k in scalars::<C::ScalarExt>(8) {
            let recoded = Recoded::<C>::new(&k);
            let batched = Table::batch_mul_affine(&tables, &recoded);
            for (t, a) in tables.iter().zip(batched.iter()) {
                assert_eq!(
                    C::from(*a),
                    t.mul_recoded(&recoded),
                    "batch affine ladder must match the projective one"
                );
            }
        }
    }

    /// The two arms of [`Table::batch_mul`] agree at the size where it
    /// switches between them, so the dispatch is a pure performance choice.
    fn batch_mul_dispatch_arms_agree<C: GlvParams>() {
        let g = C::generator();
        let k = scalars::<C::ScalarExt>(1).next().expect("one scalar");
        let recoded = Recoded::<C>::new(&k);
        let points: Vec<C> = (0..BATCH_AFFINE_THRESHOLD + 2)
            .map(|i| g * (C::ScalarExt::from(i as u64).square() + C::ScalarExt::from(11)))
            .collect();
        for size in [BATCH_AFFINE_THRESHOLD - 1, BATCH_AFFINE_THRESHOLD] {
            let tables = Table::batch(&points[..size]);
            assert_eq!(
                Table::batch_mul(&tables, &recoded),
                Table::batch_mul_affine(&tables, &recoded),
                "the two ladders must agree at the dispatch boundary"
            );
        }
    }

    /// `batch_invert` inverts the nonzero entries and leaves zeros alone,
    /// which is what lets the ladder spot an exceptional lane.
    fn batch_invert_skips_zeros<C: GlvParams>() {
        let mut v: Vec<C::Base> = (0..16u64).map(C::Base::from).collect();
        v.push(C::Base::ZERO);
        let original = v.clone();
        let mut scratch = alloc::vec![C::Base::ZERO; v.len()];
        batch_invert(&mut v, &mut scratch);
        for (o, i) in original.iter().zip(v.iter()) {
            if bool::from(o.is_zero()) {
                assert!(bool::from(i.is_zero()), "zero must stay zero");
            } else {
                assert_eq!(*o * *i, C::Base::ONE, "must be the inverse");
            }
        }
        // All-zero and empty inputs must not panic.
        let mut z = alloc::vec![C::Base::ZERO; 4];
        batch_invert(&mut z, &mut scratch);
        assert!(z.iter().all(|e| bool::from(e.is_zero())));
        batch_invert(&mut [] as &mut [C::Base], &mut scratch);
    }

    macro_rules! eisenstein_tests {
        ($mod_name:ident, $curve:ty) => {
            mod $mod_name {
                use super::*;

                #[test]
                fn half_bounds() {
                    halves_leave_room_for_a_digit::<$curve>();
                }
                #[test]
                fn reconstructs() {
                    recoding_reconstructs::<$curve>();
                }
                #[test]
                fn table_entries() {
                    table_entries_are_the_right_multiples::<$curve>();
                }
                #[test]
                fn table_mul() {
                    table_mul_matches_group_mul::<$curve>();
                }
                #[test]
                fn one_shot() {
                    mul_matches_operator::<$curve>();
                }
                #[test]
                fn batch_build() {
                    batch_tables_equal_solo::<$curve>();
                }
                #[test]
                fn zero_x() {
                    no_point_has_zero_x::<$curve>();
                }
                #[test]
                fn affine_chain() {
                    affine_chain_matches_projective::<$curve>();
                }
                #[test]
                fn identity_table() {
                    identity_tables::<$curve>();
                }
                #[test]
                fn recoded_reuse() {
                    recoded_reuse_matches_fresh::<$curve>();
                }
                #[test]
                fn statistics() {
                    digit_statistics::<$curve>();
                }
                #[test]
                fn batch_ladder() {
                    batch_mul_matches_operator::<$curve>();
                }
                #[test]
                fn batch_ladder_vs_projective() {
                    batch_mul_matches_projective_ladder::<$curve>();
                }
                #[test]
                fn batch_dispatch() {
                    batch_mul_dispatch_arms_agree::<$curve>();
                }
                #[test]
                fn batch_inversion() {
                    batch_invert_skips_zeros::<$curve>();
                }
            }
        };
    }

    eisenstein_tests!(pallas_tests, crate::pallas::Point);
    eisenstein_tests!(vesta_tests, crate::vesta::Point);
}
