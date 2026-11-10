use super::*;
use crate::Fp;
use ff::Field;
use rand::SeedableRng;

const P: [u64; 4] = [0x992d30ed00000001, 0x224698fc094cf91b, 0, 1 << 62];

fn lanes(v: [Fp; 4]) -> [[u64; 4]; 4] {
    v.map(|x| x.0)
}

#[test]
fn building_blocks_match_scalar() {
    let mut rng = rand_xorshift::XorShiftRng::from_seed([3; 16]);
    for _ in 0..200 {
        let a: [Fp; 4] = core::array::from_fn(|_| Fp::random(&mut rng));
        let b: [Fp; 4] = core::array::from_fn(|_| Fp::random(&mut rng));
        // SAFETY: test runs only where pasta_curves_avx2 guarantees AVX2.
        unsafe {
            let m = mul::<{ P[0] }, { P[1] }, { P[2] }, { P[3] }>(
                &shl5(&pack(&lanes(a))),
                &pack(&lanes(b)),
            );
            let got = unpack_wide(&m);
            for l in 0..4 {
                assert_eq!(reduce_wide(&got[l], &P), (a[l] * b[l]).0, "mul lane {l}");
            }
            let k2 = kp(&pasta_limbs29(&P), 2);
            let s = sub(&pack(&lanes(a)), &pack(&lanes(b)), &k2);
            let got = unpack_wide(&s);
            for l in 0..4 {
                assert_eq!(reduce_wide(&got[l], &P), (a[l] - b[l]).0, "sub lane {l}");
            }
            let got = unpack_wide(&shl5(&pack(&lanes(a))));
            for l in 0..4 {
                let thirty_two = Fp::ONE.double().double().double().double().double();
                assert_eq!(
                    reduce_wide(&got[l], &P),
                    (a[l] * thirty_two).0,
                    "shl5 lane {l}"
                );
            }
        }
    }
}

/// `k p + r` as a 320-bit integer.
fn k_p_plus(k: u64, p: &[u64; 4], r: &[u64; 4]) -> [u64; 5] {
    let mut out = [0u64; 5];
    let mut carry = 0u128;
    for i in 0..4 {
        let t = u128::from(k) * u128::from(p[i]) + u128::from(r[i]) + carry;
        out[i] = t as u64;
        carry = t >> 64;
    }
    out[4] = carry as u64;
    out
}

/// Normalized radix-2^29 limbs of `x < 2^261`.
fn limbs29(x: &[u64; 5]) -> [u64; L] {
    core::array::from_fn(|i| {
        let bit = i * BITS as usize;
        let (w, s) = (bit / 64, bit % 64);
        let mut v = x[w] >> s;
        if s + BITS as usize > 64 && w + 1 < 5 {
            v |= x[w + 1] << (64 - s);
        }
        v & MASK
    })
}

fn vector(lanes: [[u64; L]; 4]) -> V {
    // SAFETY: tests run only where `pasta_curves_avx2` guarantees AVX2.
    V(core::array::from_fn(|i| unsafe {
        _mm256_set_epi64x(
            lanes[3][i] as i64,
            lanes[2][i] as i64,
            lanes[1][i] as i64,
            lanes[0][i] as i64,
        )
    }))
}

/// The largest normalized radix-2^29 value below `bound`: every limb at
/// its maximum except limb 8, which is as large as the bound allows.
fn max_limbs_below(bound: &[u64; 5]) -> [u64; L] {
    let top = limbs29(bound)[8];
    let mut limbs = [MASK; L];
    limbs[8] = top - 1;
    limbs
}

fn integer(limbs: &[u64; L]) -> [u64; 5] {
    let mut out = [0u64; 5];
    for (i, &limb) in limbs.iter().enumerate() {
        let bit = i * BITS as usize;
        let (w, s) = (bit / 64, bit % 64);
        out[w] |= limb << s;
        if s > 0 && w + 1 < 5 {
            out[w + 1] |= limb >> (64 - s);
        }
    }
    out
}

macro_rules! boundary_tests {
    ($name:ident, $field:ty, $modulus:expr) => {
        mod $name {
            use super::*;

            const P: [u64; 4] = $modulus;

            fn reduce(x: &[u64; 5]) -> $field {
                <$field>::from_chord_limbs(reduce_wide(x, &P))
            }

            // Interpret an integer independently of the AVX2 reducers.
            fn integer_as_field(x: &[u64; 5]) -> $field {
                let mut acc = <$field>::ZERO;
                for word in x.iter().rev() {
                    for bit in (0..64).rev() {
                        acc = acc.double();
                        if word >> bit & 1 != 0 {
                            acc += <$field>::ONE;
                        }
                    }
                }
                acc
            }

            #[test]
            fn canonical_accepts_top_limb_equal_to_2_pow_25() {
                // 2^257 is below 8p, but its top limb equals 2^25.
                let input = [0u64, 0, 0, 0, 2];
                let v = limbs29(&input);
                assert_eq!(v[L - 1], 1 << 25);
                // SAFETY: AVX2 is guaranteed (see `vector`).
                let got = unsafe {
                    store4(&canonical(&vector([v; 4]), &pasta_limbs29(&P)))
                };
                for lane in got {
                    let raw = lane.to_limbs();
                    assert!(raw.iter().rev().cmp(P.iter().rev()).is_lt());
                    assert_eq!(
                        integer_as_field(&[raw[0], raw[1], raw[2], raw[3], 0]),
                        integer_as_field(&input),
                    );
                }
            }

            #[test]
            fn full_radix_mul_and_reducers_match_independent_oracle() {
                use rand::Rng;
                let mut rng = rand_xorshift::XorShiftRng::from_seed([0xa7; 16]);
                let exponent = L as u64 * BITS as u64;
                let inv_r = <$field>::from(2).pow_vartime([exponent]).invert().unwrap();
                let p29 = pasta_limbs29(&P);
                for sample in 0..1024 {
                    let inputs: [[[u64; L]; 4]; 2] = core::array::from_fn(|side| {
                        core::array::from_fn(|lane| {
                            core::array::from_fn(|limb| match sample {
                                0 => MASK,
                                1 => 0,
                                2 => u64::from(limb == (lane + side) % L),
                                3 => if (limb + lane + side) % 2 == 0 { MASK } else { 0 },
                                _ => rng.next_u64() & MASK,
                            })
                        })
                    });
                    // SAFETY: AVX2 is guaranteed (see `vector`).
                    let product = unsafe {
                        unpack_wide(&mul::<{ P[0] }, { P[1] }, { P[2] }, { P[3] }>(
                            &vector(inputs[0]), &vector(inputs[1]),
                        ))
                    };
                    for (lane, got) in product.iter().enumerate() {
                        let a = integer_as_field(&integer(&inputs[0][lane]));
                        let b = integer_as_field(&integer(&inputs[1][lane]));
                        assert_eq!(integer_as_field(got), a * b * inv_r);
                    }

                    let wide: [[u64; 5]; 4] = core::array::from_fn(|_| {
                        let mut x = core::array::from_fn(|_| rng.next_u64());
                        x[4] &= 15;
                        x
                    });
                    for x in wide {
                        let got = reduce_wide(&x, &P);
                        assert!(got.iter().rev().cmp(P.iter().rev()).is_lt());
                        assert_eq!(
                            integer_as_field(&[got[0], got[1], got[2], got[3], 0]),
                            integer_as_field(&x),
                        );
                    }
                    let wide: [[u64; 5]; 4] = core::array::from_fn(|lane| {
                        let r = <$field>::random(&mut rng).chord_limbs();
                        k_p_plus(((sample + lane) % 8) as u64, &P, &r)
                    });
                    // SAFETY: AVX2 is guaranteed (see `vector`).
                    let got = unsafe {
                        store4(&canonical(&vector(wide.map(|x| limbs29(&x))), &p29))
                    };
                    for (got, x) in got.into_iter().zip(wide) {
                        let got = got.to_limbs();
                        assert!(got.iter().rev().cmp(P.iter().rev()).is_lt());
                        assert_eq!(
                            integer_as_field(&[got[0], got[1], got[2], got[3], 0]),
                            integer_as_field(&x),
                        );
                    }
                }
            }

            #[test]
            fn extreme_coordinates_long_prefixes_and_output_order() {
                // Exercise 8192 full lane groups followed by a short tail.
                const LONG_PREFIX_ADDITIONS: usize = LANES * 8192 + 1;

                let mut minus1 = P;
                minus1[0] -= 1;
                let mut minus2 = P;
                minus2[0] -= 2;
                let edges = [
                    <$field>::ZERO,
                    <$field>::ONE,
                    -<$field>::ONE,
                    <$field>::from_chord_limbs([1, 0, 0, 0]),
                    <$field>::from_chord_limbs([2, 0, 0, 0]),
                    <$field>::from_chord_limbs(minus1),
                    <$field>::from_chord_limbs(minus2),
                ];
                // One scratch across every call, as the reducer reuses
                // it across levels: stale chunks must not leak.
                let mut scratch = ChordScratch::default();
                for n in [
                    0usize, 1, 7, 8, 9, 31, 32, 33, 255, 256, 257,
                    LONG_PREFIX_ADDITIONS,
                    257, 33, 9, 8, 7, 1, 0,
                ] {
                    for seed in [0usize, 1, 6] {
                        let mut flat = Vec::new();
                        let mut expected = Vec::new();
                        for i in 0..n {
                            let lx = edges[(i * 5 + seed) % edges.len()];
                            let mut rx = edges[(i * 3 + seed + 1) % edges.len()];
                            if lx == rx {
                                rx += <$field>::ONE;
                            }
                            let ly = edges[(i * 11 + seed) % edges.len()];
                            let ry = edges[(i * 13 + 2) % edges.len()];
                            let slope = (ry - ly) * (rx - lx).invert().unwrap();
                            let x = slope.square() - lx - rx;
                            let y = slope * (lx - x) - ly;
                            expected.push(Point29 {
                                x: Limbs29::from_limbs(&x.chord_limbs()),
                                y: Limbs29::from_limbs(&y.chord_limbs()),
                            });
                            for (x, y) in [(lx, ly), (rx, ry)] {
                                flat.push(Point29 {
                                    x: Limbs29::from_limbs(&x.chord_limbs()),
                                    y: Limbs29::from_limbs(&y.chord_limbs()),
                                });
                            }
                        }
                        let sentinel = flat.first().copied().unwrap_or_default();
                        let mut actual = alloc::vec![sentinel; n + 3];
                        <$field>::chord_level(
                            n,
                            |i| (2 * i, 2 * i + 1, n - 1 - i),
                            &flat,
                            &mut actual,
                            &mut scratch,
                        )
                        .unwrap();
                        for i in 0..n {
                            assert_eq!(
                                actual[n - 1 - i], expected[i],
                                "n={n}, seed={seed}, i={i}",
                            );
                        }
                        assert!(actual[n..].iter().all(|v| *v == sentinel));
                        // Put an exceptional pair in each lane near the
                        // end, after long prefix chains have been built.
                        if n == LONG_PREFIX_ADDITIONS {
                            for lane in 0..LANES {
                                let mut bad = flat.clone();
                                let i = n - 1 - lane;
                                bad[2 * i + 1].x = bad[2 * i].x;
                                let mut untouched = alloc::vec![sentinel; n + 3];
                                assert!(
                                    <$field>::chord_level(
                                        n,
                                        |i| (2 * i, 2 * i + 1, n - 1 - i),
                                        &bad,
                                        &mut untouched,
                                        &mut scratch,
                                    )
                                    .is_none(),
                                );
                                assert!(untouched.iter().all(|v| *v == sentinel));
                            }
                        }
                    }
                }
            }

            /// [`canonical`] across its whole input range: every multiple
            /// of `p` up to `7p` plus residues at both ends, and the
            /// all-maximum limb pattern just below `8p`.
            #[test]
            fn canonical_up_to_8p() {
                let p29 = pasta_limbs29(&P);
                let mut minus = P;
                minus[0] -= 1;
                let mut minus2 = P;
                minus2[0] -= 2;
                let residues = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], minus, minus2];
                for k in 0..8 {
                    for r in &residues {
                        let v = limbs29(&k_p_plus(k, &P, r));
                        // SAFETY: AVX2 is guaranteed (see `vector`).
                        let got = unsafe { store4(&canonical(&vector([v; 4]), &p29)) };
                        let want = limbs29(&[r[0], r[1], r[2], r[3], 0]);
                        for lane in got {
                            assert_eq!(lane.0.map(u64::from), want, "{k} p + r");
                        }
                    }
                }
                let top = max_limbs_below(&k_p_plus(8, &P, &[0; 4]));
                // SAFETY: AVX2 is guaranteed (see `vector`).
                let got = unsafe { store4(&canonical(&vector([top; 4]), &p29)) };
                let want = reduce_wide(&integer(&top), &P);
                assert_eq!(got[0].to_limbs(), want);
            }

            /// [`sub`] with the largest subtrahend each multiple of `p`
            /// must dominate, from zero and from a large minuend: the
            /// result is exactly `a + c p - b`, normalized.
            #[test]
            fn sub_at_dominance_limit() {
                let p29 = pasta_limbs29(&P);
                for c in [2u64, 3, 4, 5] {
                    let k = kp(&p29, c);
                    let mut b = [MASK; L];
                    b[8] = c * (1 << 22) - 2;
                    for a in [[0u64; L], max_limbs_below(&k_p_plus(96, &P, &[0; 4]))] {
                        // SAFETY: AVX2 is guaranteed (see `vector`).
                        let got =
                            unsafe { lanes_of(&sub(&vector([a; 4]), &vector([b; 4]), &k)) };
                        let got: [u64; L] = core::array::from_fn(|i| got[i][0]);
                        assert!(got[..8].iter().all(|&limb| limb <= MASK), "c = {c}");
                        // a + c p - b, as integers.
                        let (a, b) = (integer(&a), integer(&b));
                        let cp = k_p_plus(c, &P, &[0; 4]);
                        let mut want = [0u64; 5];
                        let mut carry = 0i128;
                        for i in 0..5 {
                            let t =
                                i128::from(a[i]) + i128::from(cp[i]) - i128::from(b[i]) + carry;
                            want[i] = t as u64;
                            carry = t >> 64;
                        }
                        assert_eq!(carry, 0, "c p dominates b");
                        assert_eq!(integer(&got), want, "c = {c}");
                    }
                }
            }

            /// [`mul`] at the largest operands `chord_level` feeds it
            /// (scaled differences below `96p` times products below
            /// `4.01p`, checked up to `5p`, and the all-maximum limb
            /// pattern of each bound) is the
            /// exact Montgomery product `a b / 2^261 mod p`.
            #[test]
            fn mul_at_kernel_bounds() {
                let inv32 = <$field>::from(32).invert().unwrap();
                let mut minus = P;
                minus[0] -= 1;
                let operands = |k: u64| {
                    [
                        limbs29(&k_p_plus(k - 1, &P, &minus)),
                        max_limbs_below(&k_p_plus(k, &P, &[0; 4])),
                    ]
                };
                for a in operands(96) {
                    for b in operands(5).into_iter().chain(operands(36)) {
                        // SAFETY: AVX2 is guaranteed (see `vector`).
                        let got = unsafe {
                            unpack_wide(&mul::<{ P[0] }, { P[1] }, { P[2] }, { P[3] }>(
                                &vector([a; 4]),
                                &vector([b; 4]),
                            ))
                        };
                        let want = reduce(&integer(&a)) * reduce(&integer(&b)) * inv32;
                        assert_eq!(reduce(&got[0]), want);
                    }
                }
            }
        }
    };
}

boundary_tests!(
    pallas_bounds,
    crate::Fp,
    [0x992d30ed00000001, 0x224698fc094cf91b, 0, 1 << 62]
);
boundary_tests!(
    vesta_bounds,
    crate::Fq,
    [0x8c46eb2100000001, 0x224698fc0994a8dd, 0, 1 << 62]
);
