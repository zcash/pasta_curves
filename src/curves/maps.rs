//! The maps between the curves' models, generated for each curve by
//! `new_maps_impl!` from `new_curve_impl!`.

/// Generates the conversions between the Jacobian point type `$name` and the
/// affine point type `$name_affine` of a curve.
macro_rules! new_maps_impl {
    ($name:ident, $name_affine:ident, $base:ident) => {
        impl group::Curve for $name {
            type Affine = $name_affine;

            fn batch_normalize(p: &[Self], q: &mut [Self::Affine]) {
                assert_eq!(p.len(), q.len());

                // Two chains hide multiplication latency for larger batches.
                // Below this length, the two-lane variant's fixed overhead
                // and lane bookkeeping outweigh the dependency-chain split.
                const TWO_LANE_MIN: usize = 32;

                // Only the public batch length selects the single- or two-chain
                // path. Identity handling and inversion remain constant-time.
                if p.len() < TWO_LANE_MIN {
                    let Some((first_p, p)) = p.split_first() else {
                        return;
                    };
                    let (first_q, q) = q.split_first_mut().unwrap();
                    let first_skip = first_p.is_identity();
                    let mut acc = $base::conditional_select(&first_p.z, &$base::one(), first_skip);
                    for (p, q) in p.iter().zip(q.iter_mut()) {
                        // We use the `x` field of $name_affine to store the
                        // product of previous z-coordinates seen.
                        q.x = acc;

                        // We will end up skipping all identities in p
                        acc = $base::conditional_select(&(acc * p.z), &acc, p.is_identity());
                    }

                    // This is the inverse, as all z-coordinates are nonzero and
                    // the ones that are not are skipped.
                    acc = acc.invert().unwrap();

                    for (p, q) in p.iter().rev().zip(q.iter_mut().rev()) {
                        let skip = p.is_identity();

                        // Compute tmp = 1/z
                        let tmp = q.x * acc;

                        // Cancel out z-coordinate in denominator of `acc`
                        acc = $base::conditional_select(&(acc * p.z), &acc, skip);

                        // Set the coordinates to the correct value
                        let tmp2 = Field::square(&tmp);
                        let tmp3 = tmp2 * tmp;

                        q.x = p.x * tmp2;
                        q.y = p.y * tmp3;

                        *q = $name_affine::conditional_select(q, &$name_affine::identity(), skip);
                    }

                    // The first prefix is one, and the accumulator is dead
                    // after producing its inverse, so both multiplications
                    // at this endpoint can be omitted.
                    let tmp2 = Field::square(&acc);
                    let tmp3 = tmp2 * acc;
                    first_q.x = first_p.x * tmp2;
                    first_q.y = first_p.y * tmp3;
                    *first_q = $name_affine::conditional_select(
                        first_q,
                        &$name_affine::identity(),
                        first_skip,
                    );
                    return;
                }

                // Even- and odd-indexed points form independent prefix-product
                // and back-substitution chains, sharing a single field inversion.
                let (first_p, p) = p.split_at(2);
                let (first_q, q) = q.split_at_mut(2);
                let first_skip = [first_p[0].is_identity(), first_p[1].is_identity()];
                let mut acc = [
                    $base::conditional_select(&first_p[0].z, &$base::one(), first_skip[0]),
                    $base::conditional_select(&first_p[1].z, &$base::one(), first_skip[1]),
                ];
                for (i, (p, q)) in p.iter().zip(q.iter_mut()).enumerate() {
                    let lane = i & 1;
                    q.x = acc[lane];
                    acc[lane] =
                        $base::conditional_select(&(acc[lane] * p.z), &acc[lane], p.is_identity());
                }

                // Join the lane products, invert once, and recover the inverse
                // of each lane's product.
                let inverse = (acc[0] * acc[1]).invert().unwrap();
                let mut acc = [inverse * acc[1], inverse * acc[0]];

                for (i, (p, q)) in p.iter().zip(q.iter_mut()).enumerate().rev() {
                    let lane = i & 1;
                    let skip = p.is_identity();
                    let tmp = q.x * acc[lane];
                    acc[lane] = $base::conditional_select(&(acc[lane] * p.z), &acc[lane], skip);
                    let tmp2 = Field::square(&tmp);
                    let tmp3 = tmp2 * tmp;
                    q.x = p.x * tmp2;
                    q.y = p.y * tmp3;
                    *q = $name_affine::conditional_select(q, &$name_affine::identity(), skip);
                }

                for lane in 0..2 {
                    // The first prefix in each lane is one, and its final
                    // accumulator update is dead.
                    let tmp2 = Field::square(&acc[lane]);
                    let tmp3 = tmp2 * acc[lane];
                    first_q[lane].x = first_p[lane].x * tmp2;
                    first_q[lane].y = first_p[lane].y * tmp3;
                    first_q[lane] = $name_affine::conditional_select(
                        &first_q[lane],
                        &$name_affine::identity(),
                        first_skip[lane],
                    );
                }
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

// Keep this in sync with if_homogeneous_mul!, as for the projective module.
#[cfg(any(not(target_arch = "arm"), test))]
mod projective {
    use ff::Field;
    use subtle::{Choice, ConditionallySelectable};

    use super::super::projective::Projective;

    impl<F: Field> Projective<F> {
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
