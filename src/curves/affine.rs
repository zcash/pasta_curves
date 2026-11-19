//! The affine chart of the curves, generated for each curve by
//! `new_affine_impl!` from `new_curve_impl!`.

/// Generates the affine point type `$name_affine` of a curve whose Jacobian
/// point type is `$name`, and the impls whose receiver is the affine type.
macro_rules! new_affine_impl {
    (($($privacy:tt)*), $name:ident, $name_affine:ident, $base:ident, $scalar:ident,
     $curve_type:ident, $unchecked_ctor:ident) => {
        /// Represents a point in the affine coordinate space (or the point at
        /// infinity).
        #[derive(Copy, Clone)]
        #[cfg_attr(feature = "repr-c", repr(C))]
        $($privacy)* struct $name_affine {
            x: $base,
            y: $base,
        }

        impl_affine_unchecked_ctor!($name_affine, $base, $unchecked_ctor);

        impl fmt::Debug for $name_affine {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
                if self.is_identity().into() {
                    write!(f, "Infinity")
                } else {
                    write!(f, "({:?}, {:?})", self.x, self.y)
                }
            }
        }

        impl_binops_additive_specify_output!($name_affine, $name_affine, $name);
        impl_binops_additive_specify_output!($name_affine, $name, $name);
        impl_binops_multiplicative_mixed!($name_affine, $scalar, $name);

        impl<'a> Neg for &'a $name_affine {
            type Output = $name_affine;

            fn neg(self) -> $name_affine {
                $name_affine {
                    x: self.x,
                    y: -self.y,
                }
            }
        }

        impl Neg for $name_affine {
            type Output = $name_affine;

            fn neg(self) -> $name_affine {
                -&self
            }
        }

        impl<'a, 'b> Add<&'a $name> for &'b $name_affine {
            type Output = $name;

            fn add(self, rhs: &'a $name) -> $name {
                rhs + self
            }
        }

        impl<'a, 'b> Add<&'a $name_affine> for &'b $name_affine {
            type Output = $name;

            fn add(self, rhs: &'a $name_affine) -> $name {
                if bool::from(self.is_identity()) {
                    rhs.to_curve()
                } else if bool::from(rhs.is_identity()) {
                    self.to_curve()
                } else {
                    if self.x == rhs.x {
                        if self.y == rhs.y {
                            self.to_curve().double()
                        } else {
                            $name::identity()
                        }
                    } else {
                        let h = rhs.x - self.x;
                        let hh = Field::square(&h);
                        let i = hh + hh;
                        let i = i + i;
                        let j = h * i;
                        let r = rhs.y - self.y;
                        let r = r + r;
                        let v = self.x * i;
                        let x3 = Field::square(&r) - j - v - v;
                        let j = self.y * j;
                        let j = j + j;
                        let y3 = r * (v - x3) - j;
                        let z3 = h + h;

                        $name {
                            x: x3, y: y3, z: z3
                        }
                    }
                }
            }
        }

        impl<'a, 'b> Sub<&'a $name_affine> for &'b $name_affine {
            type Output = $name;

            fn sub(self, other: &'a $name_affine) -> $name {
                self + (-other)
            }
        }

        impl<'a, 'b> Sub<&'a $name> for &'b $name_affine {
            type Output = $name;

            fn sub(self, other: &'a $name) -> $name {
                self + (-other)
            }
        }

        #[allow(clippy::suspicious_arithmetic_impl)]
        impl<'a, 'b> Mul<&'b $scalar> for &'a $name_affine {
            type Output = $name;

            fn mul(self, other: &'b $scalar) -> Self::Output {
                if_homogeneous_mul!(
                    mul_curve_specific!($name, $base, $curve_type, affine, self, other),
                    mul_curve_specific!($name, $base, general, affine, self, other),
                )
            }
        }

        impl group::CurveAffine for $name_affine {
            type Curve = $name;
            type Scalar = $scalar;

            impl_affine_curve_specific!($name, $base, $curve_type);

            fn identity() -> Self {
                Self {
                    x: $base::zero(),
                    y: $base::zero(),
                }
            }

            fn is_identity(&self) -> Choice {
                self.x.is_zero() & self.y.is_zero()
            }

            fn to_curve(&self) -> Self::Curve {
                $name {
                    x: self.x,
                    y: self.y,
                    z: $base::conditional_select(&$base::one(), &$base::zero(), self.is_identity()),
                }
            }
        }

        #[cfg(all(feature = "sqrt-table", feature = "alloc"))]
        impl $name_affine {
            /// Decodes compressed public points with interleaved square-root chains.
            ///
            /// Every result matches [`GroupEncoding::from_bytes`] converted to an
            /// `Option`, including noncanonical encodings and the identity.
            ///
            /// Inputs must be public: decoding and square-root table lookups are
            /// variable-time.
            fn batch_from_bytes_vartime(items: &[[u8; 32]]) -> alloc::vec::Vec<Option<Self>> {
                let mut out = alloc::vec::Vec::with_capacity(items.len());
                let mut chunks = items.chunks_exact(8);
                for chunk in &mut chunks {
                    Self::decode_chunk::<8>(chunk, &mut out);
                }
                let rest = chunks.remainder();
                match rest.len() {
                    0 => {}
                    1 => out.push(Self::from_bytes(&rest[0]).into()),
                    2 => Self::decode_chunk::<2>(rest, &mut out),
                    3 => Self::decode_chunk::<3>(rest, &mut out),
                    4 => Self::decode_chunk::<4>(rest, &mut out),
                    5 => Self::decode_chunk::<5>(rest, &mut out),
                    6 => Self::decode_chunk::<6>(rest, &mut out),
                    _ => Self::decode_chunk::<7>(rest, &mut out),
                }
                out
            }

            fn decode_chunk<const K: usize>(
                chunk: &[[u8; 32]],
                out: &mut alloc::vec::Vec<Option<Self>>,
            ) {
                debug_assert!(chunk.len() <= K);
                let mut xs = [$base::ZERO; K];
                let mut x_ok = [Choice::from(0u8); K];
                let mut ysigns = [Choice::from(0u8); K];
                for (k, bytes) in chunk.iter().enumerate() {
                    let mut tmp = *bytes;
                    ysigns[k] = Choice::from(tmp[31] >> 7);
                    tmp[31] &= 0b0111_1111;
                    let x = $base::from_repr(tmp);
                    x_ok[k] = x.is_some();
                    xs[k] = x.unwrap_or($base::ZERO);
                }
                let rhs: [$base; K] = core::array::from_fn(|k| {
                    xs[k].sqr_n_mul_runtime(1, &xs[k]) + $name::curve_constant_b()
                });
                let roots = $base::sqrt_alt_lanes(&rhs);
                for k in 0..chunk.len() {
                    let (is_square, y) = roots[k];
                    let (x, ysign) = (xs[k], ysigns[k]);
                    let y = $base::conditional_select(&y, &-y, ysign ^ y.is_odd());
                    let is_identity = x.is_zero() & !ysign;
                    let point = Self::conditional_select(
                        &$name_affine { x, y },
                        &Self::identity(),
                        is_identity,
                    );
                    out.push(CtOption::new(point, x_ok[k] & (is_identity | is_square)).into());
                }
            }


        }

        impl GroupEncoding for $name_affine {
            type Repr = [u8; 32];

            fn from_bytes(bytes: &[u8; 32]) -> CtOption<Self> {
                let mut tmp = *bytes;
                let ysign = Choice::from(tmp[31] >> 7);
                tmp[31] &= 0b0111_1111;

                $base::from_repr(tmp).and_then(|x| {
                    CtOption::new(Self::identity(), x.is_zero() & (!ysign)).or_else(|| {
                        let x3 = x.sqr_n_mul_runtime(1, &x);
                        (x3 + $name::curve_constant_b()).sqrt().and_then(|y| {
                            let sign = y.is_odd();

                            let y = $base::conditional_select(&y, &-y, ysign ^ sign);

                            CtOption::new(
                                $name_affine {
                                    x,
                                    y,
                                },
                                Choice::from(1u8),
                            )
                        })
                    })
                })
            }

            fn from_bytes_unchecked(bytes: &Self::Repr) -> CtOption<Self> {
                // We can't avoid curve checks when parsing a compressed encoding.
                Self::from_bytes(bytes)
            }

            fn to_bytes(&self) -> [u8; 32] {
                // TODO: not constant time
                if bool::from(self.is_identity()) {
                    [0; 32]
                } else {
                    let (x, y) = (self.x, self.y);
                    let sign = y.is_odd().unwrap_u8() << 7;
                    let mut xbytes = x.to_repr();
                    xbytes[31] |= sign;
                    xbytes
                }
            }
        }

        #[cfg(feature = "alloc")]
        #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
        impl CurveAffine for $name_affine {
            type ScalarExt = $scalar;
            type Base = $base;
            type CurveExt = $name;

            #[cfg(feature = "sqrt-table")]
            fn batch_decode_vartime(bytes: &[Self::Repr]) -> alloc::vec::Vec<Option<Self>> {
                Self::batch_from_bytes_vartime(bytes)
            }

            fn is_on_curve(&self) -> Choice {
                // y^2 - x^3 - ax ?= b
                (Field::square(&self.y) - (Field::square(&self.x) + &$name::curve_constant_a()) * self.x).ct_eq(&$name::curve_constant_b())
                    | self.is_identity()
            }

            fn coordinates(&self) -> CtOption<Coordinates<Self>> {
                CtOption::new(Coordinates { x: self.x, y: self.y }, !self.is_identity())
            }

            fn from_xy(x: Self::Base, y: Self::Base) -> CtOption<Self> {
                let p = $name_affine {
                    x, y,
                };
                CtOption::new(p, p.is_on_curve())
            }

            fn a() -> Self::Base {
                $name::curve_constant_a()
            }

            fn b() -> Self::Base {
                $name::curve_constant_b()
            }
        }

        impl Default for $name_affine {
            fn default() -> $name_affine {
                $name_affine::identity()
            }
        }

        /// Zeroizing a point overwrites it with the identity, which is the `Default` value.
        #[cfg(feature = "zeroize")]
        #[cfg_attr(docsrs, doc(cfg(feature = "zeroize")))]
        impl zeroize::DefaultIsZeroes for $name_affine {}

        impl ConstantTimeEq for $name_affine {
            fn ct_eq(&self, other: &Self) -> Choice {
                self.x.ct_eq(&other.x) & self.y.ct_eq(&other.y)
            }
        }

        impl PartialEq for $name_affine {
            fn eq(&self, other: &Self) -> bool {
                self.ct_eq(other).into()
            }
        }

        impl cmp::Eq for $name_affine {}

        impl ConditionallySelectable for $name_affine {
            fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
                $name_affine {
                    x: $base::conditional_select(&a.x, &b.x, choice),
                    y: $base::conditional_select(&a.y, &b.y, choice),
                }
            }
        }

        #[cfg(feature = "gpu")]
        impl ec_gpu::GpuName for $name_affine {
            fn name() -> alloc::string::String {
                ec_gpu::name!()
            }
        }
    };
}

macro_rules! impl_affine_unchecked_ctor {
    // Only the public curves get the constructor and its inverse `raw_xy`: on
    // the crate-private isogenous curves nothing would call them, and they
    // would be dead code.
    ($name_affine:ident, $base:ident, with_unchecked_ctor) => {
        impl $name_affine {
            /// Constructs an affine point from coordinates without checking that
            /// it lies on the curve.
            ///
            /// This is intended for protocol constants and precomputed tables
            /// whose coordinates were validated when they were generated, where
            /// the check performed by `arithmetic::CurveAffine::from_xy` is pure
            /// overhead, and where a `const` context is needed. For untrusted
            /// input, use `from_xy` instead.
            ///
            /// `(0, 0)` is the representation of the identity, exactly as for
            /// `from_xy` and [`group::CurveAffine::identity`].
            ///
            /// Passing coordinates that are not on the curve does not cause
            /// memory unsafety, but every operation on the result is then
            /// meaningless, and so are the security properties of any protocol
            /// built on it. Callers should test their constants with
            /// `arithmetic::CurveAffine::is_on_curve`.
            pub const fn from_xy_unchecked(x: $base, y: $base) -> Self {
                $name_affine { x, y }
            }

            /// The stored coordinates, the identity read as `(0, 0)`: the
            /// inverse of `from_xy_unchecked`.
            #[cfg(feature = "glv")]
            pub(crate) const fn raw_xy(&self) -> ($base, $base) {
                (self.x, self.y)
            }
        }
    };
    ($name_affine:ident, $base:ident, without_unchecked_ctor) => {};
}

macro_rules! impl_affine_curve_specific {
    ($name:ident, $base:ident, special_a0_b5) => {
        fn generator() -> Self {
            // NOTE: This is specific to b = 5

            const NEGATIVE_ONE: $base = $base::neg(&$base::from_raw([1, 0, 0, 0]));
            const TWO: $base = $base::from_raw([2, 0, 0, 0]);

            Self {
                x: NEGATIVE_ONE,
                y: TWO,
            }
        }
    };
    ($name:ident, $base:ident, general) => {
        /// Unimplemented: there is no standard generator for this curve.
        fn generator() -> Self {
            unimplemented!()
        }
    };
}
