//! The maps between the curves' models, generated for each curve by
//! `new_maps_impl!` from `new_curve_impl!`, and between the public point types
//! and the projective model.

/// Generates the conversions between the Jacobian point type `$name` and the
/// affine point type `$name_affine` of a curve.
macro_rules! new_maps_impl {
    ($name:ident, $name_affine:ident, $base:ident) => {
        impl group::Curve for $name {
            type Affine = $name_affine;

            fn batch_normalize(p: &[Self], q: &mut [Self::Affine]) {
                crate::inversion::batch::invert(p, q);
            }

            fn to_affine(&self) -> Self::Affine {
                let zinv = self.z.invert().unwrap_or($base::zero());
                let zinv2 = Field::square(&zinv);
                let x = self.x * zinv2;
                let zinv3 = zinv2 * zinv;
                let y = self.y * zinv3;

                let tmp = $name_affine { x, y };

                $name_affine::conditional_select(&tmp, &$name_affine::identity(), zinv.is_zero())
            }
        }

        impl<'a> From<&'a $name_affine> for $name {
            fn from(p: &'a $name_affine) -> $name {
                p.to_curve()
            }
        }

        /// A batch normalization as a batch inversion of the points' $Z$
        /// coordinates: each affine output keeps its prefix product in its $x$
        /// coordinate, and turns its inverse $1 / Z$ into $(X / Z^2, Y / Z^3)$,
        /// or the identity when $Z = 0$.
        impl crate::inversion::batch::Slot<$base> for $name_affine {
            type Input = $name;

            fn value(point: &$name) -> $base {
                point.z
            }

            fn scratch(&mut self) -> &mut $base {
                // The x coordinate is free until the slot is put.
                &mut self.x
            }

            fn put(&mut self, point: &$name, zinv: $base, is_identity: Choice) {
                let zinv2 = Field::square(&zinv);
                let zinv3 = zinv2 * zinv;
                self.x = point.x * zinv2;
                self.y = point.y * zinv3;
                *self =
                    $name_affine::conditional_select(self, &$name_affine::identity(), is_identity);
            }

            fn put_vartime(&mut self, point: &$name, zinv: Option<$base>) {
                let Some(zinv) = zinv else {
                    *self = $name_affine::identity();
                    return;
                };
                let zinv2 = Field::square(&zinv);
                let zinv3 = zinv2 * zinv;
                self.x = point.x * zinv2;
                self.y = point.y * zinv3;
            }
        }

        impl From<$name_affine> for $name {
            fn from(p: $name_affine) -> $name {
                p.to_curve()
            }
        }

        impl<'a> From<&'a $name> for $name_affine {
            fn from(p: &'a $name) -> $name_affine {
                p.to_affine()
            }
        }

        impl From<$name> for $name_affine {
            fn from(p: $name) -> $name_affine {
                p.to_affine()
            }
        }
    };
}

// On 32-bit ARM the scalar multiplication keeps the Jacobian ladder, so only the
// tests use these conversions there.
#[cfg_attr(target_arch = "arm", allow(dead_code))]
mod projective {
    use ff::Field;
    use subtle::{Choice, ConditionallySelectable};

    use group::CurveAffine as _;

    use super::super::params::PASTA_B;
    use super::super::projective::Projective;
    use super::super::{Ep, EpAffine, Eq, EqAffine};
    use crate::{Fp, Fq};

    /// The typed maps between the public point types of Pallas or Vesta and the
    /// projective model, through the coordinate maps below.
    macro_rules! projective_maps {
        ($point:ident, $affine:ident, $base:ident) => {
            /// Jacobian to projective, $(X : Y : Z) \mapsto (X Z : Y : Z^3)$.
            impl From<&$point> for Projective<$base, PASTA_B> {
                fn from(p: &$point) -> Self {
                    Self::from_jacobian(p.x, p.y, p.z)
                }
            }

            /// Affine to projective, $(x, y) \mapsto (x : y : 1)$.
            impl From<&$affine> for Projective<$base, PASTA_B> {
                fn from(p: &$affine) -> Self {
                    Self::from_affine(p.x, p.y, p.is_identity())
                }
            }

            /// Projective to Jacobian, $(X : Y : Z) \mapsto (X Z : Y Z^2 : Z)$.
            impl From<Projective<$base, PASTA_B>> for $point {
                fn from(p: Projective<$base, PASTA_B>) -> Self {
                    let (x, y, z) = p.to_jacobian();
                    $point { x, y, z }
                }
            }
        };
    }

    projective_maps!(Ep, EpAffine, Fp);
    projective_maps!(Eq, EqAffine, Fq);

    impl<F: Field, const B: i64> Projective<F, B> {
        pub(crate) fn from_jacobian(x: F, y: F, z: F) -> Self {
            let z2 = z.square();
            let p = Self {
                x: x * z,
                y,
                z: z2 * z,
            };
            Self::conditional_select(&p, &Self::identity(), z.is_zero())
        }

        pub(crate) fn from_affine(x: F, y: F, infinity: Choice) -> Self {
            let p = Self { x, y, z: F::ONE };
            Self::conditional_select(&p, &Self::identity(), infinity)
        }

        pub(crate) fn to_jacobian(self) -> (F, F, F) {
            let z2 = self.z.square();
            let x = self.x * self.z;
            let y = self.y * z2;
            let zero = F::ZERO;
            (
                F::conditional_select(&x, &zero, self.z.is_zero()),
                F::conditional_select(&y, &zero, self.z.is_zero()),
                self.z,
            )
        }
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests;
