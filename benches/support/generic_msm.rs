// Ported from halo2_proofs 0.4.1, src/arithmetic.rs (MIT OR Apache-2.0,
// Copyright the Electric Coin Company and contributors): `Bucket`,
// `booth_digit`, `booth_window_count`, `BoothBuckets` and `best_multiexp`.
// Kept as close to the source as possible.
use super::*;

#[derive(Clone, Copy)]
enum Bucket<C: CurveAffine> {
    None,
    Affine(C),
    Projective(C::CurveExt),
}

impl<C: CurveAffine> Bucket<C> {
    fn add_assign(&mut self, other: C) {
        *self = match *self {
            Bucket::None => Bucket::Affine(other),
            Bucket::Affine(a) => Bucket::Projective(a + other),
            Bucket::Projective(mut a) => {
                a += other;
                Bucket::Projective(a)
            }
        }
    }

    fn add(self, mut other: C::CurveExt) -> C::CurveExt {
        match self {
            Bucket::None => other,
            Bucket::Affine(a) => {
                other += a;
                other
            }
            Bucket::Projective(a) => other + a,
        }
    }
}

struct BoothDigit {
    magnitude: usize,
    negative: bool,
}

fn booth_digit(bytes: &[u8], window_bits: usize, window: usize) -> BoothDigit {
    let window_start = window * window_bits;
    let byte_start = window_start / 8;
    let mut encoded = [0; 8];
    if byte_start < bytes.len() {
        for (slot, byte) in encoded.iter_mut().zip(&bytes[byte_start..]) {
            *slot = *byte;
        }
    }

    let bit_offset = window_start - byte_start * 8;
    let radix = 1usize << window_bits;
    let value = ((u64::from_le_bytes(encoded) >> bit_offset) as usize) & (radix - 1);
    let overlap = if window_start == 0 {
        0
    } else {
        let bit = window_start - 1;
        bytes
            .get(bit / 8)
            .map_or(0, |byte| usize::from((byte >> (bit % 8)) & 1))
    };

    if value < radix / 2 {
        BoothDigit {
            magnitude: value + overlap,
            negative: false,
        }
    } else {
        let magnitude = radix - value - overlap;
        BoothDigit {
            magnitude,
            negative: magnitude != 0,
        }
    }
}

fn booth_window_count<R: AsRef<[u8]>>(reprs: &[R], window_bits: usize) -> usize {
    let repr_bytes = reprs
        .iter()
        .map(|repr| repr.as_ref().len())
        .max()
        .unwrap_or(0);
    repr_bytes * 8 / window_bits + 1
}

#[derive(Clone)]
struct BoothBuckets<C: CurveAffine> {
    c: usize,
    coeffs: Vec<Bucket<C>>,
}

impl<C: CurveAffine> BoothBuckets<C> {
    fn new(c: usize) -> Self {
        Self {
            c,
            coeffs: vec![Bucket::None; 1 << (c - 1)],
        }
    }

    fn sum(
        &mut self,
        coeffs: &[<C::ScalarExt as PrimeField>::Repr],
        bases: &[C],
        i: usize,
    ) -> C::CurveExt {
        for (coeff, base) in coeffs.iter().zip(bases.iter()) {
            let digit = booth_digit(coeff.as_ref(), self.c, i);
            if digit.magnitude != 0 {
                let base = if digit.negative { -*base } else { *base };
                self.coeffs[digit.magnitude - 1].add_assign(base);
            }
        }
        let mut acc = C::CurveExt::identity();
        let mut sum = C::CurveExt::identity();
        self.coeffs.iter().rev().for_each(|b| {
            sum = b.add(sum);
            acc += sum;
        });
        acc
    }
}

/// halo2_proofs 0.4.1 `best_multiexp`, including its parallel branch.
pub(super) fn best_multiexp<C: CurveAffine>(coeffs: &[C::ScalarExt], bases: &[C]) -> C::CurveExt {
    assert_eq!(coeffs.len(), bases.len());
    let coeffs = coeffs.iter().map(PrimeField::to_repr).collect::<Vec<_>>();
    let c = if bases.len() < 4 {
        1
    } else if bases.len() < 32 {
        3
    } else {
        (f64::from(bases.len() as u32)).ln().ceil() as usize
    };
    let mut multi_buckets: Vec<BoothBuckets<C>> =
        vec![BoothBuckets::new(c); booth_window_count(&coeffs, c)];
    #[cfg(feature = "multicore")]
    {
        let num_threads = maybe_rayon::current_num_threads();
        if num_threads > 1 && coeffs.len() > num_threads {
            return multi_buckets
                .par_iter_mut()
                .enumerate()
                .rev()
                .map(|(i, buckets)| {
                    let mut acc = buckets.sum(&coeffs, bases, i);
                    (0..c * i).for_each(|_| acc = acc.double());
                    acc
                })
                .reduce(C::CurveExt::identity, |a, b| a + b);
        }
    }
    multi_buckets
        .iter_mut()
        .enumerate()
        .rev()
        .map(|(i, buckets)| buckets.sum(&coeffs, bases, i))
        .fold(C::CurveExt::identity(), |mut sum, bucket| {
            (0..c).for_each(|_| sum = sum.double());
            sum + bucket
        })
}
