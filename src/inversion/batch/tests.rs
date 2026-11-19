use std::vec;
use std::vec::Vec;

use ff::Field;
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;
use subtle::Choice;

use super::{Slot, TWO_LANE_MIN, invert, invert_vartime};
use crate::arithmetic::VartimeField;
use crate::{Fp, Fq};

/// The slot of one field element, which records its result and how often it
/// received one.
#[derive(Clone, Default)]
struct Recorded<F> {
    scratch: F,
    inverse: Option<F>,
    puts: usize,
}

impl<F: Field> Slot<F> for Recorded<F> {
    type Input = F;

    fn value(input: &F) -> F {
        *input
    }

    fn scratch(&mut self) -> &mut F {
        &mut self.scratch
    }

    fn put(&mut self, _: &F, inverse: F, is_zero: Choice) {
        self.inverse = (!bool::from(is_zero)).then_some(inverse);
        self.puts += 1;
    }

    fn put_vartime(&mut self, _: &F, inverse: Option<F>) {
        self.inverse = inverse;
        self.puts += 1;
    }
}

/// Both inversions give each nonzero element its inverse, flag each zero one,
/// and fill every slot exactly once.
fn check<F: VartimeField + core::fmt::Debug>(values: &[F]) {
    let expected: Vec<Option<F>> = values.iter().map(|x| x.invert().into()).collect();

    for vartime in [false, true] {
        let mut slots = vec![Recorded::<F>::default(); values.len()];
        if vartime {
            invert_vartime(values, &mut slots);
        } else {
            invert(values, &mut slots);
        }
        let inverses: Vec<Option<F>> = slots.iter().map(|slot| slot.inverse).collect();
        assert_eq!(inverses, expected, "vartime: {vartime}");
        assert!(
            slots.iter().all(|slot| slot.puts == 1),
            "vartime: {vartime}"
        );
    }
}

/// Lengths around the switch to two chains, with zeros placed so that each
/// chain, its first or its last element, or both chains at once, are zero.
fn check_field<F: VartimeField + core::fmt::Debug>() {
    let mut rng = XorShiftRng::from_seed([
        0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc,
        0xe5,
    ]);
    let lengths = [
        0,
        1,
        2,
        3,
        7,
        16,
        TWO_LANE_MIN - 1,
        TWO_LANE_MIN,
        TWO_LANE_MIN + 1,
        63,
        64,
        65,
        128,
        129,
    ];
    for len in lengths {
        check(&vec![F::ZERO; len]);
        check(&vec![F::ONE; len]);
        check(&vec![-F::ONE; len]);
        let values: Vec<F> = (0..len).map(|_| F::random(&mut rng)).collect();
        check(&values);

        for stride in [2, 3] {
            for offset in 0..stride {
                let mixed: Vec<F> = values
                    .iter()
                    .enumerate()
                    .map(|(i, x)| if i % stride == offset { F::ZERO } else { *x })
                    .collect();
                check(&mixed);
            }
        }
        if len > 0 {
            for zero in [0, len / 2, len - 1] {
                let mut one_zero = values.clone();
                one_zero[zero] = F::ZERO;
                check(&one_zero);
            }
        }
    }
}

#[test]
fn inverts_each_nonzero_element() {
    check_field::<Fp>();
    check_field::<Fq>();
}
