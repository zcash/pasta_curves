//! Batched affine point addition.
//!
//! The affine chord law is a partial operation, defined when the operands'
//! `x` differ. A level of additions shares one field inversion (Montgomery's
//! trick); a level that meets an exceptional pair is retried with the
//! complete case analysis, which the batch makes possible by writing nothing
//! until its inversion succeeds.

use alloc::vec::Vec;

use crate::arithmetic::VartimeField;

#[derive(Clone, Copy)]
pub(super) struct AffinePoint<F> {
    pub(super) x: F,
    pub(super) y: F,
}

// The left operand lives in its `points[output]` result slot. The prefix pass
// scales `numerator` in place, so the backward pass can recover the slope
// directly without keeping a separate inversion scratch field.
pub(super) struct PendingAffineAddition<F> {
    pub(super) output: usize,
    pub(super) x_sum: F,
    pub(super) numerator: F,
    pub(super) denominator: F,
}

/// Independent multiplication lanes used by batch inversion.
pub(super) const BATCH_INVERSION_LANES: usize = 2;

/// Batch-inverts affine denominators and immediately finishes the additions.
///
/// Each `output` identifies the slot containing the addition's left operand.
/// The outputs must be distinct. Returns `None` if the product of the
/// denominators is zero. In that case, this function has not written to
/// `points`: the prefix pass changes only disposable pending numerators, and
/// the output pass starts only after the product is successfully inverted.
/// The reducer relies on this failure-atomic behavior to retry the level.
pub(super) fn batch_invert_and_add<F>(
    additions: &mut [PendingAffineAddition<F>],
    points: &mut [AffinePoint<F>],
) -> Option<()>
where
    F: VartimeField,
{
    let Some((first, additions)) = additions.split_first_mut() else {
        return Some(());
    };

    if additions.is_empty() {
        let denominator_inverse = first.denominator.invert_vartime()?;
        let left = points[first.output];
        let slope = first.numerator * denominator_inverse;
        let x = slope.square() - first.x_sum;
        let y = slope * (left.x - x) - left.y;
        points[first.output] = AffinePoint { x, y };
        return Some(());
    }

    let (second, additions) = additions.split_first_mut().unwrap();

    // Compute two prefix lanes in lockstep. This retains one field inversion
    // for the entire batch while exposing independent multiplication chains.
    // Seeding from the first denominator in each lane removes the initial
    // multiplication by one. Pre-scale each numerator by its denominator's
    // prefix product so the backward pass recovers the slope directly. This
    // moves one multiplication off the dependent path before each square
    // without changing the total multiplication count.
    let mut lane_products = [first.denominator, second.denominator];
    for pair in additions.chunks_mut(BATCH_INVERSION_LANES) {
        for (addition, product) in pair.iter_mut().zip(&mut lane_products) {
            addition.numerator *= *product;
            *product *= addition.denominator;
        }
    }

    // Invert both lane products with one inversion. A field has no zero
    // divisors, so this product is zero exactly when at least one affine
    // denominator is zero. No output point has been written yet; `?` therefore
    // makes failure atomic with respect to `points`. Writes to
    // pending numerators are discarded by the caller on failure.
    let product = lane_products[0] * lane_products[1];
    // This MSM is already variable-time with respect to scalar digits; batch
    // inversion does not provide a constant-time guarantee.
    let product_inverse = product.invert_vartime()?;
    let mut lane_inverses = [
        lane_products[1] * product_inverse,
        lane_products[0] * product_inverse,
    ];

    // If the last pair is incomplete, consume its first lane before walking
    // the complete pairs backward.
    let complete_pairs = additions.len() / BATCH_INVERSION_LANES;
    if additions.len() % BATCH_INVERSION_LANES != 0 {
        let addition = &additions[additions.len() - 1];
        let denominator = addition.denominator;
        let slope = addition.numerator * lane_inverses[0];
        lane_inverses[0] *= denominator;

        let left = points[addition.output];
        let x = slope.square() - addition.x_sum;
        let y = slope * (left.x - x) - left.y;
        points[addition.output] = AffinePoint { x, y };
    }
    for pair in additions[..complete_pairs * BATCH_INVERSION_LANES]
        .chunks(BATCH_INVERSION_LANES)
        .rev()
    {
        let first = &pair[0];
        let second = &pair[1];
        let first_slope = first.numerator * lane_inverses[0];
        let second_slope = second.numerator * lane_inverses[1];
        lane_inverses[0] *= first.denominator;
        lane_inverses[1] *= second.denominator;

        // Complete each affine chord addition while its pending record is
        // already resident. Keep two independent field-operation lanes.
        let first_left = points[first.output];
        let second_left = points[second.output];
        let first_x = first_slope.square() - first.x_sum;
        let second_x = second_slope.square() - second.x_sum;
        let first_y = first_slope * (first_left.x - first_x) - first_left.y;
        let second_y = second_slope * (second_left.x - second_x) - second_left.y;
        points[first.output] = AffinePoint {
            x: first_x,
            y: first_y,
        };
        points[second.output] = AffinePoint {
            x: second_x,
            y: second_y,
        };
    }

    let first_left = points[first.output];
    let second_left = points[second.output];
    let first_slope = first.numerator * lane_inverses[0];
    let second_slope = second.numerator * lane_inverses[1];
    let first_x = first_slope.square() - first.x_sum;
    let second_x = second_slope.square() - second.x_sum;
    let first_y = first_slope * (first_left.x - first_x) - first_left.y;
    let second_y = second_slope * (second_left.x - second_x) - second_left.y;
    points[first.output] = AffinePoint {
        x: first_x,
        y: first_y,
    };
    points[second.output] = AffinePoint {
        x: second_x,
        y: second_y,
    };
    Some(())
}

/// Reduce buckets with one Montgomery inversion per tree level. Inputs are
/// valid nonidentity points on a Pasta curve (a = 0); offsets partition them.
/// The incomplete pass writes nothing until inversion succeeds. On a zero
/// denominator, retry the unchanged level with complete formulas. Successful
/// results are then compacted in place, retaining the input allocation.
pub(super) fn reduce_affine_buckets<F>(
    points: Vec<AffinePoint<F>>,
    offsets: Vec<usize>,
) -> Option<Vec<Option<AffinePoint<F>>>>
where
    F: VartimeField,
{
    reduce_affine_buckets_inner::<F, false>(points, offsets)
}

pub(super) fn reduce_affine_buckets_inner<F, const COMPLETE: bool>(
    mut points: Vec<AffinePoint<F>>,
    mut offsets: Vec<usize>,
) -> Option<Vec<Option<AffinePoint<F>>>>
where
    F: VartimeField,
{
    debug_assert!(!offsets.is_empty());
    let mut next_offsets = Vec::with_capacity(offsets.len());
    let mut pending = Vec::with_capacity(points.len() / 2);

    while offsets.windows(2).any(|range| range[1] - range[0] > 1) {
        pending.clear();
        for range in offsets.windows(2) {
            for input in (range[0]..range[1] - (range[1] - range[0]) % 2).step_by(2) {
                let left = points[input];
                let right = points[input + 1];
                let dx = right.x - left.x;
                let (numerator, denominator) = if COMPLETE && dx.is_zero_vartime() {
                    // Equal x means equal or opposite points. Omit inverse
                    // pairs, including order-two points, from the next level.
                    if !(right.y - left.y).is_zero_vartime() || left.y.is_zero_vartime() {
                        continue;
                    }
                    let square = left.x.square();
                    (square.double() + square, left.y.double())
                } else {
                    (right.y - left.y, dx)
                };
                pending.push(PendingAffineAddition {
                    output: input,
                    x_sum: left.x + right.x,
                    numerator,
                    denominator,
                });
            }
        }

        if batch_invert_and_add(&mut pending, &mut points).is_none() {
            if !COMPLETE {
                return reduce_affine_buckets_inner::<F, true>(points, offsets);
            }
            // Complete formulas have nonzero denominators for valid inputs.
            // Decline rather than returning a partial MSM if that guard fails.
            return None;
        }

        // Pending outputs identify surviving pairs; cancelled pairs have no
        // record. A carried odd point survives unchanged. The write index is
        // never ahead of the read index, so compaction cannot clobber a result.
        next_offsets.clear();
        next_offsets.push(0);
        let mut additions = pending.iter().peekable();
        let mut output = 0;
        for range in offsets.windows(2) {
            let end_pairs = range[1] - (range[1] - range[0]) % 2;
            for input in (range[0]..end_pairs).step_by(2) {
                if additions
                    .peek()
                    .is_some_and(|addition| addition.output == input)
                {
                    points[output] = points[input];
                    output += 1;
                    additions.next();
                }
            }
            if end_pairs != range[1] {
                points[output] = points[end_pairs];
                output += 1;
            }
            next_offsets.push(output);
        }
        points.truncate(output);
        core::mem::swap(&mut offsets, &mut next_offsets);
    }

    Some(
        offsets
            .windows(2)
            .map(|range| (range[0] != range[1]).then(|| points[range[0]]))
            .collect(),
    )
}
