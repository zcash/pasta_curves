//! The Jacobian coordinates of the curves, generated for each curve by
//! `new_jacobian_impl!` from `new_curve_impl!`.

#[cfg(feature = "alloc")]
macro_rules! impl_batch_mul_same_scalar_vartime {
    (glv, $name:ident) => {
        #[cfg(feature = "glv")]
        fn try_multiexp_vartime(
            scalars: &[Self::ScalarExt],
            bases: &[Self::AffineExt],
        ) -> Option<Self> {
            crate::glv::msm::try_multiexp::<$name>(scalars, bases)
        }

        /// Prepares positioned tables of about 76 KiB per base, from which a
        /// sum needs no doublings.
        #[cfg(feature = "glv")]
        fn try_prepare_fixed_base_sum(
            bases: &[Self::AffineExt],
        ) -> Option<alloc::boxed::Box<dyn crate::arithmetic::PreparedFixedBaseSum<Self>>> {
            Some(crate::glv::fixed_base::prepare_fixed_base_sum::<$name>(
                bases,
            ))
        }

        #[cfg(feature = "glv")]
        fn batch_mul_same_scalar_vartime(
            points: &[Self::AffineExt],
            scalar: &Self::ScalarExt,
            output: &mut [Self],
        ) {
            assert_eq!(points.len(), output.len());
            if points.is_empty() {
                return;
            }
            for (point, output) in points.iter().zip(output.iter_mut()) {
                *output = point.to_curve();
            }
            let scalar = crate::glv::Decomposed::<$name>::new(scalar);
            let tables = crate::glv::Table::batch(output);
            for (output, table) in output.iter_mut().zip(tables) {
                *output = table.mul_decomposed(&scalar);
            }
        }
    };
    (native, $name:ident) => {};
}

/// Generates the Jacobian point type `$name` of a curve whose affine point
/// type is `$name_affine`, and the impls whose receiver is the Jacobian type.
macro_rules! new_jacobian_impl {
    (($($privacy:tt)*), $name:ident, $name_affine:ident, $iso:ident, $base:ident,
     $scalar:ident, $curve_id:literal, $a_raw:expr, $b_raw:expr, $curve_type:ident,
     $same_scalar_mul:ident) => {
        /// A point in Jacobian coordinates.
        ///
        /// The coordinates are a class $(X : Y : Z)$ of the weighted projective plane with
        /// weights $(2, 3, 1)$: $(X : Y : Z)$ and $(\lambda^2 X : \lambda^3 Y : \lambda Z)$
        /// are the same point for every nonzero $\lambda$. The point lies on
        ///
        /// $$Y^2 = X^3 + a X Z^4 + b Z^6,$$
        ///
        /// and is the affine point $(X / Z^2, Y / Z^3)$ when $Z \neq 0$. Every triple with
        /// $Z = 0$ represents the identity, which [`group::Group::identity`] returns as
        /// $(0, 0, 0)$.
        #[derive(Copy, Clone, Debug)]
        #[cfg_attr(feature = "repr-c", repr(C))]
        $($privacy)* struct $name {
            x: $base,
            y: $base,
            z: $base,
        }

        impl $name {
            // Without `alloc`, only the curves without a generator use it, in `double`.
            #[cfg_attr(not(feature = "alloc"), allow(dead_code))]
            const fn curve_constant_a() -> $base {
                $base::from_raw($a_raw)
            }

            const fn curve_constant_b() -> $base {
                $base::from_raw($b_raw)
            }

            #[cfg(feature = "alloc")]
            fn is_identity_vartime(&self) -> bool {
                self.z.is_zero_vartime()
            }
        }

        impl group::Group for $name {
            type Scalar = $scalar;

            fn try_random<R: TryRng + ?Sized>(rng: &mut R) -> Result<Self, R::Error> {
                loop {
                    let x = $base::try_random(rng)?;
                    let ysign = (rng.try_next_u32()? % 2) as u8;

                    let x3 = x.sqr_n_mul_runtime(1, &x);
                    let y = (x3 + $name::curve_constant_b()).sqrt();
                    if let Some(y) = Option::<$base>::from(y) {
                        let sign = y.is_odd().unwrap_u8();
                        let y = if ysign ^ sign == 0 { y } else { -y };

                        let p = $name_affine {
                            x,
                            y,
                        };
                        break Ok(p.to_curve());
                    }
                }
            }

            impl_jacobian_curve_specific!($name, $base, $curve_type);

            fn identity() -> Self {
                Self {
                    x: $base::zero(),
                    y: $base::zero(),
                    z: $base::zero(),
                }
            }

            fn is_identity(&self) -> Choice {
                self.z.is_zero()
            }
        }

        #[cfg(feature = "alloc")]
        #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
        impl group::WnafGroup for $name {
            fn recommended_wnaf_for_num_scalars(num_scalars: usize) -> usize {
                // Copied from bls12_381::g1, should be updated.
                const RECOMMENDATIONS: [usize; 12] =
                    [1, 3, 7, 20, 43, 120, 273, 563, 1630, 3128, 7933, 62569];

                let mut ret = 4;
                for r in &RECOMMENDATIONS {
                    if num_scalars > *r {
                        ret += 1;
                    } else {
                        break;
                    }
                }

                ret
            }
        }

        #[cfg(feature = "alloc")]
        #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
        impl CurveExt for $name {
            type ScalarExt = $scalar;
            type Base = $base;
            type AffineExt = $name_affine;

            const CURVE_ID: &'static str = $curve_id;

            impl_jacobian_curve_ext!($name, $iso, $base, $curve_type);

            fn a() -> Self::Base {
                $name::curve_constant_a()
            }

            fn b() -> Self::Base {
                $name::curve_constant_b()
            }

            fn new_jacobian(x: Self::Base, y: Self::Base, z: Self::Base) -> CtOption<Self> {
                let p = $name { x, y, z };
                CtOption::new(p, p.is_on_curve())
            }

            fn jacobian_coordinates(&self) -> ($base, $base, $base) {
               (self.x, self.y, self.z)
            }

            fn is_on_curve(&self) -> Choice {
                // Y^2 = X^3 + AX(Z^4) + b(Z^6)
                // Y^2 - (X^2 + A(Z^4))X = b(Z^6)

                let z2 = Field::square(&self.z);
                let z4 = Field::square(&z2);
                let z6 = z4 * z2;
                (Field::square(&self.y) - (Field::square(&self.x) + $name::curve_constant_a() * z4) * self.x)
                    .ct_eq(&(z6 * $name::curve_constant_b()))
                    | self.z.is_zero()
            }

            fn to_affine_vartime(&self) -> Self::Affine {
                let zinv = self.z.invert_vartime().unwrap_or($base::zero());
                if zinv.is_zero_vartime() {
                    $name_affine::identity()
                } else {
                    let zinv2 = Field::square(&zinv);
                    let x = self.x * zinv2;
                    let zinv3 = zinv2 * zinv;
                    let y = self.y * zinv3;

                    $name_affine {
                        x,
                        y,
                    }
                }
            }

            fn batch_normalize_vartime(p: &[Self], q: &mut [Self::Affine]) {
                assert_eq!(p.len(), q.len());

                // Two chains hide multiplication latency for larger batches.
                // Below this length, the two-lane variant's fixed overhead
                // and lane bookkeeping outweigh the dependency-chain split.
                const TWO_LANE_MIN: usize = 32;

                if p.len() < TWO_LANE_MIN {
                    let Some((first_p, p)) = p.split_first() else {
                        return;
                    };
                    let (first_q, q) = q.split_first_mut().unwrap();
                    let first_skip = first_p.is_identity_vartime();
                    let mut acc = if first_skip { $base::one() } else { first_p.z };
                    for (p, q) in p.iter().zip(q.iter_mut()) {
                        // We use the `x` field of $name_affine to store the
                        // product of previous z-coordinates seen.
                        q.x = acc;

                        // We will end up skipping all identities in p
                        if !p.is_identity_vartime() {
                            acc *= p.z;
                        }
                    }

                    // This is the inverse, as all z-coordinates are nonzero and
                    // the ones that are not are skipped.
                    acc = acc.invert_vartime().unwrap();

                    for (p, q) in p.iter().rev().zip(q.iter_mut().rev()) {
                        if p.is_identity_vartime() {
                            *q = $name_affine::identity();
                            continue;
                        }

                        // Compute tmp = 1/z
                        let tmp = q.x * acc;

                        // Cancel out z-coordinate in denominator of `acc`
                        acc *= p.z;

                        // Set the coordinates to the correct value
                        let tmp2 = Field::square(&tmp);
                        let tmp3 = tmp2 * tmp;

                        q.x = p.x * tmp2;
                        q.y = p.y * tmp3;
                    }

                    // The first prefix is one, and the accumulator is dead
                    // after producing its inverse, so both multiplications
                    // at this endpoint can be omitted.
                    if first_skip {
                        *first_q = $name_affine::identity();
                    } else {
                        let tmp2 = Field::square(&acc);
                        let tmp3 = tmp2 * acc;
                        first_q.x = first_p.x * tmp2;
                        first_q.y = first_p.y * tmp3;
                    }
                    return;
                }

                // Even- and odd-indexed points form independent prefix-product
                // and back-substitution chains, sharing a single field inversion.
                let (first_p, p) = p.split_at(2);
                let (first_q, q) = q.split_at_mut(2);
                let first_skip = [
                    first_p[0].is_identity_vartime(),
                    first_p[1].is_identity_vartime(),
                ];
                let mut acc = [
                    if first_skip[0] { $base::one() } else { first_p[0].z },
                    if first_skip[1] { $base::one() } else { first_p[1].z },
                ];
                for (i, (p, q)) in p.iter().zip(q.iter_mut()).enumerate() {
                    // Store each lane's prefix in the output's unused x coordinate.
                    q.x = acc[i & 1];
                    if !p.is_identity_vartime() {
                        acc[i & 1] *= p.z;
                    }
                }

                // Join the lane products, invert once, and recover the inverse
                // of each lane's product.
                let inverse = (acc[0] * acc[1]).invert_vartime().unwrap();
                let mut acc = [inverse * acc[1], inverse * acc[0]];

                for (i, (p, q)) in p.iter().zip(q.iter_mut()).enumerate().rev() {
                    let lane = i & 1;
                    if p.is_identity_vartime() {
                        *q = $name_affine::identity();
                        continue;
                    }
                    let tmp = q.x * acc[lane];
                    acc[lane] *= p.z;
                    let tmp2 = Field::square(&tmp);
                    let tmp3 = tmp2 * tmp;
                    q.x = p.x * tmp2;
                    q.y = p.y * tmp3;
                }

                for lane in 0..2 {
                    // The first prefix in each lane is one, and its final
                    // accumulator update is dead.
                    if first_skip[lane] {
                        first_q[lane] = $name_affine::identity();
                    } else {
                        let tmp2 = Field::square(&acc[lane]);
                        let tmp3 = tmp2 * acc[lane];
                        first_q[lane].x = first_p[lane].x * tmp2;
                        first_q[lane].y = first_p[lane].y * tmp3;
                    }
                }
            }

            impl_batch_mul_same_scalar_vartime!($same_scalar_mul, $name);
        }

        #[cfg(feature = "alloc")]
        impl CurveExtUnchecked for $name {
            fn new_jacobian_unchecked(x: Self::Base, y: Self::Base, z: Self::Base) -> Self {
                let point = $name { x, y, z };
                debug_assert!(bool::from(point.is_on_curve()));
                point
            }
        }

        impl PrimeGroup for $name {}

        impl CofactorGroup for $name {
            type Subgroup = $name;

            fn clear_cofactor(&self) -> Self {
                // This is a prime-order group, with a cofactor of 1.
                *self
            }

            fn into_subgroup(self) -> CtOption<Self::Subgroup> {
                // Nothing to do here.
                CtOption::new(self, 1.into())
            }

            fn is_torsion_free(&self) -> Choice {
                // Shortcut: all points in a prime-order group are torsion free.
                1.into()
            }
        }

        impl PrimeCurve for $name {}

        impl CofactorCurve for $name {}

        impl GroupEncoding for $name {
            type Repr = [u8; 32];

            fn from_bytes(bytes: &Self::Repr) -> CtOption<Self> {
                $name_affine::from_bytes(bytes).map(Self::from)
            }

            fn from_bytes_unchecked(bytes: &Self::Repr) -> CtOption<Self> {
                // We can't avoid curve checks when parsing a compressed encoding.
                $name_affine::from_bytes(bytes).map(Self::from)
            }

            fn to_bytes(&self) -> Self::Repr {
                $name_affine::from(self).to_bytes()
            }
        }

        impl Default for $name {
            fn default() -> $name {
                $name::identity()
            }
        }

        /// Zeroizing a point overwrites it with the identity, which is the `Default` value.
        #[cfg(feature = "zeroize")]
        #[cfg_attr(docsrs, doc(cfg(feature = "zeroize")))]
        impl zeroize::DefaultIsZeroes for $name {}

        impl ConstantTimeEq for $name {
            fn ct_eq(&self, other: &Self) -> Choice {
                // Is (xz^2, yz^3, z) equal to (x'z'^2, yz'^3, z') when converted to affine?

                let z = Field::square(&other.z);
                let x1 = self.x * z;
                let z = z * other.z;
                let y1 = self.y * z;
                let z = Field::square(&self.z);
                let x2 = other.x * z;
                let z = z * self.z;
                let y2 = other.y * z;

                let self_is_zero = self.is_identity();
                let other_is_zero = other.is_identity();

                (self_is_zero & other_is_zero) // Both point at infinity
                            | ((!self_is_zero) & (!other_is_zero) & x1.ct_eq(&x2) & y1.ct_eq(&y2))
                // Neither point at infinity, coordinates are the same
            }
        }

        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.ct_eq(other).into()
            }
        }

        impl cmp::Eq for $name {}

        impl ConditionallySelectable for $name {
            fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
                $name {
                    x: $base::conditional_select(&a.x, &b.x, choice),
                    y: $base::conditional_select(&a.y, &b.y, choice),
                    z: $base::conditional_select(&a.z, &b.z, choice),
                }
            }
        }

        impl<'a> Neg for &'a $name {
            type Output = $name;

            fn neg(self) -> $name {
                $name {
                    x: self.x,
                    y: -self.y,
                    z: self.z,
                }
            }
        }

        impl Neg for $name {
            type Output = $name;

            fn neg(self) -> $name {
                -&self
            }
        }

        impl<T> Sum<T> for $name
        where
            T: core::borrow::Borrow<$name>,
        {
            fn sum<I>(iter: I) -> Self
            where
                I: Iterator<Item = T>,
            {
                iter.fold(Self::identity(), |acc, item| acc + item.borrow())
            }
        }

        impl<'a, 'b> Add<&'a $name> for &'b $name {
            type Output = $name;

            fn add(self, rhs: &'a $name) -> $name {
                if bool::from(self.is_identity()) {
                    *rhs
                } else if bool::from(rhs.is_identity()) {
                    *self
                } else {
                    let z1z1 = Field::square(&self.z);
                    let z2z2 = Field::square(&rhs.z);
                    let u1 = self.x * z2z2;
                    let u2 = rhs.x * z1z1;
                    let s1 = self.y * z2z2 * rhs.z;
                    let s2 = rhs.y * z1z1 * self.z;

                    if u1 == u2 {
                        if s1 == s2 {
                            self.double()
                        } else {
                            $name::identity()
                        }
                    } else {
                        let h = u2 - u1;
                        let i = Field::square(&(h + h));
                        let j = h * i;
                        let r = s2 - s1;
                        let r = r + r;
                        let v = u1 * i;
                        let x3 = Field::square(&r) - j - v - v;
                        let s1 = s1 * j;
                        let s1 = s1 + s1;
                        let y3 = r * (v - x3) - s1;
                        let z3 = Field::square(&(self.z + rhs.z)) - z1z1 - z2z2;
                        let z3 = z3 * h;

                        $name {
                            x: x3, y: y3, z: z3
                        }
                    }
                }
            }
        }

        impl<'a, 'b> Add<&'a $name_affine> for &'b $name {
            type Output = $name;

            fn add(self, rhs: &'a $name_affine) -> $name {
                if bool::from(self.is_identity()) {
                    rhs.to_curve()
                } else if bool::from(rhs.is_identity()) {
                    *self
                } else {
                    let z1z1 = Field::square(&self.z);
                    let u2 = rhs.x * z1z1;
                    let s2 = rhs.y * z1z1 * self.z;

                    if self.x == u2 {
                        if self.y == s2 {
                            self.double()
                        } else {
                            $name::identity()
                        }
                    } else {
                        let h = u2 - self.x;
                        let hh = Field::square(&h);
                        let i = hh + hh;
                        let i = i + i;
                        let j = h * i;
                        let r = s2 - self.y;
                        let r = r + r;
                        let v = self.x * i;
                        let x3 = Field::square(&r) - j - v - v;
                        let j = self.y * j;
                        let j = j + j;
                        let y3 = r * (v - x3) - j;
                        let z3 = Field::square(&(self.z + h)) - z1z1 - hh;

                        $name {
                            x: x3, y: y3, z: z3
                        }
                    }
                }
            }
        }

        impl<'a, 'b> Sub<&'a $name> for &'b $name {
            type Output = $name;

            fn sub(self, other: &'a $name) -> $name {
                self + (-other)
            }
        }

        impl<'a, 'b> Sub<&'a $name_affine> for &'b $name {
            type Output = $name;

            fn sub(self, other: &'a $name_affine) -> $name {
                self + (-other)
            }
        }

        #[allow(clippy::suspicious_arithmetic_impl)]
        impl<'a, 'b> Mul<&'b $scalar> for &'a $name {
            type Output = $name;

            fn mul(self, other: &'b $scalar) -> Self::Output {
                if_homogeneous_mul!(
                    mul_curve_specific!($name, $base, $curve_type, jacobian, self, other),
                    mul_curve_specific!($name, $base, general, jacobian, self, other),
                )
            }
        }

        impl_binops_additive!($name, $name);
        impl_binops_additive!($name, $name_affine);
        impl_binops_multiplicative!($name, $scalar);
    };
}

macro_rules! impl_jacobian_curve_specific {
    ($name:ident, $base:ident, special_a0_b5) => {
        fn generator() -> Self {
            // NOTE: This is specific to b = 5

            const NEGATIVE_ONE: $base = $base::neg(&$base::one());
            const TWO: $base = $base::from_raw([2, 0, 0, 0]);

            Self {
                x: NEGATIVE_ONE,
                y: TWO,
                z: $base::one(),
            }
        }

        fn double(&self) -> Self {
            // http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
            //
            // There are no points of order 2.

            let a = Field::square(&self.x);
            let b = Field::square(&self.y);
            let c = Field::square(&b);
            let d = self.x + b;
            let d = Field::square(&d);
            let d = d - a - c;
            let d = d + d;
            let e = a + a + a;
            let f = Field::square(&e);
            let z3 = self.z * self.y;
            let z3 = z3 + z3;
            let x3 = f - (d + d);
            let c = c + c;
            let c = c + c;
            let c = c + c;
            let y3 = e * (d - x3) - c;

            let tmp = $name {
                x: x3,
                y: y3,
                z: z3,
            };

            $name::conditional_select(&tmp, &$name::identity(), self.is_identity())
        }
    };
    ($name:ident, $base:ident, general) => {
        /// Unimplemented: there is no standard generator for this curve.
        fn generator() -> Self {
            unimplemented!()
        }

        fn double(&self) -> Self {
            // http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian.html#doubling-dbl-2007-bl
            //
            // There are no points of order 2.

            let xx = Field::square(&self.x);
            let yy = Field::square(&self.y);
            let a = Field::square(&yy);
            let zz = Field::square(&self.z);
            let s = Field::double(&(Field::square(&(self.x + yy)) - xx - a));
            let m = Field::double(&xx) + xx + $name::curve_constant_a() * Field::square(&zz);
            let x3 = Field::square(&m) - Field::double(&s);
            let a = Field::double(&a);
            let a = Field::double(&a);
            let a = Field::double(&a);
            let y3 = m * (s - x3) - a;
            let z3 = Field::square(&(self.y + self.z)) - yy - zz;

            let tmp = $name {
                x: x3,
                y: y3,
                z: z3,
            };

            $name::conditional_select(&tmp, &$name::identity(), self.is_identity())
        }
    };
}

#[cfg(feature = "alloc")]
macro_rules! impl_jacobian_curve_ext {
    ($name:ident, $iso:ident, $base:ident, special_a0_b5) => {
        fn hash_to_curve<'a>(domain_prefix: &'a str) -> Box<dyn Fn(&[u8]) -> Self + 'a> {
            use super::hashtocurve;

            Box::new(move |message| {
                let mut us = [Field::ZERO; 2];
                hashtocurve::hash_to_field($name::CURVE_ID, domain_prefix, message, &mut us);
                let q0 = hashtocurve::map_to_curve_simple_swu::<$base, $name, $iso>(
                    &us[0],
                    $name::THETA,
                    $name::Z,
                );
                let q1 = hashtocurve::map_to_curve_simple_swu::<$base, $name, $iso>(
                    &us[1],
                    $name::THETA,
                    $name::Z,
                );
                let r = q0 + &q1;
                debug_assert!(bool::from(r.is_on_curve()));
                hashtocurve::iso_map::<$base, $name, $iso>(&r, &$name::ISOGENY_CONSTANTS)
            })
        }

        /// Apply the curve endomorphism by multiplying the x-coordinate
        /// by an element of multiplicative order 3.
        fn endo(&self) -> Self {
            $name {
                x: self.x * $base::ZETA,
                y: self.y,
                z: self.z,
            }
        }
    };
    ($name:ident, $iso:ident, $base:ident, general) => {
        /// Unimplemented: hashing to this curve is not supported
        fn hash_to_curve<'a>(_domain_prefix: &'a str) -> Box<dyn Fn(&[u8]) -> Self + 'a> {
            unimplemented!()
        }

        /// Unimplemented: no endomorphism is supported for this curve.
        fn endo(&self) -> Self {
            unimplemented!()
        }
    };
}
