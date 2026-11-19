//! Batch inversion without allocation: Montgomery's trick, which inverts $n$
//! field elements with one field inversion and $3(n - 1)$ multiplications.
//!
//! A forward pass records the prefix products, the product of all of them is
//! inverted once, and a backward pass peels one element off that inverse at a
//! time. Zero elements are skipped: they do not enter the products, and their
//! slot receives an unspecified value flagged as zero.
//!
//! The inversion runs over a slice of inputs and a slice of output slots of the
//! same length, through [`Slot`]: each input supplies its element, and each slot
//! keeps its prefix product and consumes its inverse as it is produced. This
//! lets the batch normalizations store the prefixes in their affine output and
//! scale the coordinates in the backward pass, without a buffer. The loops zip
//! the two slices, so they index nothing and the compiler checks no bounds.
//!
//! From [`TWO_LANE_MIN`] elements on, the even- and odd-indexed elements form
//! two independent chains of products that share the one inversion, which hides
//! the latency of the multiplications. Only the public length selects the path.

use ff::Field;
use subtle::Choice;

use crate::arithmetic::VartimeField;

/// The length from which the two chains pay: below it, the fixed cost of the
/// second chain and its bookkeeping outweigh the split.
const TWO_LANE_MIN: usize = 32;

/// An output slot of a batch inversion, for an input of type `Self::Input`.
pub(crate) trait Slot<F> {
    /// The inputs, each of which supplies one element.
    type Input;

    /// The element of `input` to invert.
    fn value(input: &Self::Input) -> F;

    /// Where the slot keeps its prefix product, written in the forward pass and
    /// read back once, before [`Slot::put`] for the same slot.
    fn scratch(&mut self) -> &mut F;

    /// Receives the inverse of the element of `input`, once per slot, in the
    /// constant-time inversion. When `is_zero` is set, the element is zero and
    /// `inverse` is unspecified.
    fn put(&mut self, input: &Self::Input, inverse: F, is_zero: Choice);

    /// Receives the inverse of the element of `input`, once per slot, in the
    /// variable-time inversion; `None` when the element is zero.
    fn put_vartime(&mut self, input: &Self::Input, inverse: Option<F>);
}

/// Inverts the element of every input, into the slot of the same index, in
/// constant time except for the length.
///
/// # Panics
///
/// If `inputs` and `slots` differ in length.
pub(crate) fn invert<F: Field, S: Slot<F>>(inputs: &[S::Input], slots: &mut [S]) {
    assert_eq!(inputs.len(), slots.len());

    if inputs.len() < TWO_LANE_MIN {
        let Some((first_input, inputs)) = inputs.split_first() else {
            return;
        };
        let (first_slot, slots) = slots.split_first_mut().unwrap();
        let first = S::value(first_input);
        let first_zero = first.is_zero();
        let mut acc = F::conditional_select(&first, &F::ONE, first_zero);
        for (input, slot) in inputs.iter().zip(slots.iter_mut()) {
            *slot.scratch() = acc;
            let value = S::value(input);
            acc = F::conditional_select(&(acc * value), &acc, value.is_zero());
        }

        // The product of the nonzero elements is nonzero.
        acc = acc.invert().unwrap();

        for (input, slot) in inputs.iter().rev().zip(slots.iter_mut().rev()) {
            let value = S::value(input);
            let is_zero = value.is_zero();
            let inverse = *slot.scratch() * acc;
            // Cancel the element out of the accumulator.
            acc = F::conditional_select(&(acc * value), &acc, is_zero);
            slot.put(input, inverse, is_zero);
        }

        // The first prefix is one, so the accumulator is the first inverse.
        first_slot.put(first_input, acc, first_zero);
        return;
    }

    let (first_inputs, inputs) = inputs.split_at(2);
    let (first_slots, slots) = slots.split_at_mut(2);
    let first = [S::value(&first_inputs[0]), S::value(&first_inputs[1])];
    let first_zero = [first[0].is_zero(), first[1].is_zero()];
    let mut acc = [
        F::conditional_select(&first[0], &F::ONE, first_zero[0]),
        F::conditional_select(&first[1], &F::ONE, first_zero[1]),
    ];
    for (i, (input, slot)) in inputs.iter().zip(slots.iter_mut()).enumerate() {
        let lane = i & 1;
        *slot.scratch() = acc[lane];
        let value = S::value(input);
        acc[lane] = F::conditional_select(&(acc[lane] * value), &acc[lane], value.is_zero());
    }

    // Join the lane products, invert once, and recover the inverse of each
    // lane's product.
    let inverse = (acc[0] * acc[1]).invert().unwrap();
    let mut acc = [inverse * acc[1], inverse * acc[0]];

    for (i, (input, slot)) in inputs.iter().zip(slots.iter_mut()).enumerate().rev() {
        let lane = i & 1;
        let value = S::value(input);
        let is_zero = value.is_zero();
        let inverse = *slot.scratch() * acc[lane];
        acc[lane] = F::conditional_select(&(acc[lane] * value), &acc[lane], is_zero);
        slot.put(input, inverse, is_zero);
    }

    // The first prefix in each lane is one.
    for lane in 0..2 {
        first_slots[lane].put(&first_inputs[lane], acc[lane], first_zero[lane]);
    }
}

/// Inverts the element of every input, into the slot of the same index, in
/// variable time.
///
/// # Panics
///
/// If `inputs` and `slots` differ in length.
pub(crate) fn invert_vartime<F: VartimeField, S: Slot<F>>(inputs: &[S::Input], slots: &mut [S]) {
    assert_eq!(inputs.len(), slots.len());

    if inputs.len() < TWO_LANE_MIN {
        let Some((first_input, inputs)) = inputs.split_first() else {
            return;
        };
        let (first_slot, slots) = slots.split_first_mut().unwrap();
        let first = S::value(first_input);
        let first_zero = first.is_zero_vartime();
        let mut acc = if first_zero { F::ONE } else { first };
        for (input, slot) in inputs.iter().zip(slots.iter_mut()) {
            *slot.scratch() = acc;
            let value = S::value(input);
            if !value.is_zero_vartime() {
                acc *= value;
            }
        }

        acc = acc.invert_vartime().unwrap();

        for (input, slot) in inputs.iter().rev().zip(slots.iter_mut().rev()) {
            let value = S::value(input);
            if value.is_zero_vartime() {
                slot.put_vartime(input, None);
                continue;
            }
            let inverse = *slot.scratch() * acc;
            acc *= value;
            slot.put_vartime(input, Some(inverse));
        }

        first_slot.put_vartime(first_input, (!first_zero).then_some(acc));
        return;
    }

    let (first_inputs, inputs) = inputs.split_at(2);
    let (first_slots, slots) = slots.split_at_mut(2);
    let first = [S::value(&first_inputs[0]), S::value(&first_inputs[1])];
    let first_zero = [first[0].is_zero_vartime(), first[1].is_zero_vartime()];
    let mut acc = [
        if first_zero[0] { F::ONE } else { first[0] },
        if first_zero[1] { F::ONE } else { first[1] },
    ];
    for (i, (input, slot)) in inputs.iter().zip(slots.iter_mut()).enumerate() {
        let lane = i & 1;
        *slot.scratch() = acc[lane];
        let value = S::value(input);
        if !value.is_zero_vartime() {
            acc[lane] *= value;
        }
    }

    let inverse = (acc[0] * acc[1]).invert_vartime().unwrap();
    let mut acc = [inverse * acc[1], inverse * acc[0]];

    for (i, (input, slot)) in inputs.iter().zip(slots.iter_mut()).enumerate().rev() {
        let lane = i & 1;
        let value = S::value(input);
        if value.is_zero_vartime() {
            slot.put_vartime(input, None);
            continue;
        }
        let inverse = *slot.scratch() * acc[lane];
        acc[lane] *= value;
        slot.put_vartime(input, Some(inverse));
    }

    for lane in 0..2 {
        first_slots[lane].put_vartime(
            &first_inputs[lane],
            (!first_zero[lane]).then_some(acc[lane]),
        );
    }
}

#[cfg(test)]
mod tests;
