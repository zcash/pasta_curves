//! The maps between the curves' models, generated for each curve by
//! `new_maps_impl!` from `new_curve_impl!`, and the maps of hashing to Pallas
//! and Vesta: the constants of the simplified SWU map to the isogenous curves,
//! and the 3-isogenies from those curves.

use super::{Ep, Eq, Fp, Fq};

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

impl Ep {
    /// Constants used for computing the isogeny from IsoEp to Ep.
    pub const ISOGENY_CONSTANTS: [Fp; 13] = [
        Fp::from_raw([
            0x775f6034aaaaaaab,
            0x4081775473d8375b,
            0xe38e38e38e38e38e,
            0x0e38e38e38e38e38,
        ]),
        Fp::from_raw([
            0x8cf863b02814fb76,
            0x0f93b82ee4b99495,
            0x267c7ffa51cf412a,
            0x3509afd51872d88e,
        ]),
        Fp::from_raw([
            0x0eb64faef37ea4f7,
            0x380af066cfeb6d69,
            0x98c7d7ac3d98fd13,
            0x17329b9ec5253753,
        ]),
        Fp::from_raw([
            0xeebec06955555580,
            0x8102eea8e7b06eb6,
            0xc71c71c71c71c71c,
            0x1c71c71c71c71c71,
        ]),
        Fp::from_raw([
            0xc47f2ab668bcd71f,
            0x9c434ac1c96b6980,
            0x5a607fcce0494a79,
            0x1d572e7ddc099cff,
        ]),
        Fp::from_raw([
            0x2aa3af1eae5b6604,
            0xb4abf9fb9a1fc81c,
            0x1d13bf2a7f22b105,
            0x325669becaecd5d1,
        ]),
        Fp::from_raw([
            0x5ad985b5e38e38e4,
            0x7642b01ad461bad2,
            0x4bda12f684bda12f,
            0x1a12f684bda12f68,
        ]),
        Fp::from_raw([
            0xc67c31d8140a7dbb,
            0x07c9dc17725cca4a,
            0x133e3ffd28e7a095,
            0x1a84d7ea8c396c47,
        ]),
        Fp::from_raw([
            0x02e2be87d225b234,
            0x1765e924f7459378,
            0x303216cce1db9ff1,
            0x3fb98ff0d2ddcadd,
        ]),
        Fp::from_raw([
            0x93e53ab371c71c4f,
            0x0ac03e8e134eb3e4,
            0x7b425ed097b425ed,
            0x025ed097b425ed09,
        ]),
        Fp::from_raw([
            0x5a28279b1d1b42ae,
            0x5941a3a4a97aa1b3,
            0x0790bfb3506defb6,
            0x0c02c5bcca0e6b7f,
        ]),
        Fp::from_raw([
            0x4d90ab820b12320a,
            0xd976bbfabbc5661d,
            0x573b3d7f7d681310,
            0x17033d3c60c68173,
        ]),
        Fp::from_raw([
            0x992d30ecfffffde5,
            0x224698fc094cf91b,
            0x0000000000000000,
            0x4000000000000000,
        ]),
    ];

    // The constants of the simplified SWU map from the field to the isogenous
    // curve, which the hash to the curve composes with the isogeny above.
    /// Z = -13
    pub const Z: Fp = Fp::from_raw([
        0x992d30ecfffffff4,
        0x224698fc094cf91b,
        0x0000000000000000,
        0x4000000000000000,
    ]);

    /// `(F::ROOT_OF_UNITY.invert().unwrap() * z).sqrt().unwrap()`
    pub const THETA: Fp = Fp::from_raw([
        0xca330bcc09ac318e,
        0x51f64fc4dc888857,
        0x4647aef782d5cdc8,
        0x0f7bdb65814179b4,
    ]);
}

impl Eq {
    /// Constants used for computing the isogeny from IsoEq to Eq.
    pub const ISOGENY_CONSTANTS: [Fq; 13] = [
        Fq::from_raw([
            0x43cd42c800000001,
            0x0205dd51cfa0961a,
            0x8e38e38e38e38e39,
            0x38e38e38e38e38e3,
        ]),
        Fq::from_raw([
            0x8b95c6aaf703bcc5,
            0x216b8861ec72bd5d,
            0xacecf10f5f7c09a2,
            0x1d935247b4473d17,
        ]),
        Fq::from_raw([
            0xaeac67bbeb586a3d,
            0xd59d03d23b39cb11,
            0xed7ee4a9cdf78f8f,
            0x18760c7f7a9ad20d,
        ]),
        Fq::from_raw([
            0xfb539a6f0000002b,
            0xe1c521a795ac8356,
            0x1c71c71c71c71c71,
            0x31c71c71c71c71c7,
        ]),
        Fq::from_raw([
            0xb7284f7eaf21a2e9,
            0xa3ad678129b604d3,
            0x1454798a5b5c56b2,
            0x0a2de485568125d5,
        ]),
        Fq::from_raw([
            0xf169c187d2533465,
            0x30cd6d53df49d235,
            0x0c621de8b91c242a,
            0x14735171ee542778,
        ]),
        Fq::from_raw([
            0x6bef1642aaaaaaab,
            0x5601f4709a8adcb3,
            0xda12f684bda12f68,
            0x12f684bda12f684b,
        ]),
        Fq::from_raw([
            0x8bee58e5fb81de63,
            0x21d910aefb03b31d,
            0xd6767887afbe04d1,
            0x2ec9a923da239e8b,
        ]),
        Fq::from_raw([
            0x4986913ab4443034,
            0x97a3ca5c24e9ea63,
            0x66d1466e9de10e64,
            0x19b0d87e16e25788,
        ]),
        Fq::from_raw([
            0x8f64842c55555533,
            0x8bc32d36fb21a6a3,
            0x425ed097b425ed09,
            0x1ed097b425ed097b,
        ]),
        Fq::from_raw([
            0x58dfecce86b2745e,
            0x06a767bfc35b5bac,
            0x9e7eb64f890a820c,
            0x2f44d6c801c1b8bf,
        ]),
        Fq::from_raw([
            0xd43d449776f99d2f,
            0x926847fb9ddd76a1,
            0x252659ba2b546c7e,
            0x3d59f455cafc7668,
        ]),
        Fq::from_raw([
            0x8c46eb20fffffde5,
            0x224698fc0994a8dd,
            0x0000000000000000,
            0x4000000000000000,
        ]),
    ];

    // The constants of the simplified SWU map from the field to the isogenous
    // curve, which the hash to the curve composes with the isogeny above.
    /// Z = -13
    pub const Z: Fq = Fq::from_raw([
        0x8c46eb20fffffff4,
        0x224698fc0994a8dd,
        0x0000000000000000,
        0x4000000000000000,
    ]);

    /// `(F::ROOT_OF_UNITY.invert().unwrap() * z).sqrt().unwrap()`
    pub const THETA: Fq = Fq::from_raw([
        0x632cae9872df1b5d,
        0x38578ccadf03ac27,
        0x53c3808d9e2f2357,
        0x2b3483a1ee9a382f,
    ]);
}

#[cfg(all(test, feature = "alloc"))]
mod tests;
