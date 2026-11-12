# Derives `DEFAULT_MULTIEXP_WINDOW_THRESHOLDS` in `src/glv/msm.rs` and
# prints it in the exact shape of the Rust source so the two can be diffed.
#
# Run from this directory:
#
#     uv run sage msm_window_thresholds.sage
#
# The table encodes the generic MSM's window schedule, ceil(ln(n)) bits for
# n terms, without floating point: for an integer n and k >= 1,
#
#     ceil(ln(n)) <= k  <=>  ln(n) <= k  <=>  n <= e^k  <=>  n <= floor(e^k),
#
# so the window width is 4 plus the index of the first entry at least n.
# Every floor is computed with interval arithmetic and checked to be exact,
# so the output does not depend on the SageMath version.

FIRST_K = 4
LAST_K = 22
U32_MAX = 2^32 - 1

def floor_exp(k):
    # e^k is irrational for k >= 1, so a fine enough interval around it
    # contains no integer and pins its floor.
    R = RealIntervalField(256)
    interval = R(k).exp()
    lower, upper = interval.lower().floor(), interval.upper().floor()
    assert lower == upper, "interval too wide for e^%d" % k
    return Integer(lower)

thresholds = [floor_exp(k) for k in range(FIRST_K, LAST_K + 1)]

# The schedule the table stands for: floor(e^k) is the largest n whose
# window is at most k bits, and the next integer needs one more bit.
for k, n in zip(range(FIRST_K, LAST_K + 1), thresholds):
    assert ceil(log(RealIntervalField(256)(n)).upper()) == k
    assert ceil(log(RealIntervalField(256)(n + 1)).lower()) == k + 1

# The table is indexed by u32 term counts and stops where they run out:
# its last entry fits, and the next one would not.
assert thresholds[-1] <= U32_MAX
assert floor_exp(LAST_K + 1) > U32_MAX

def rust_literal(n):
    digits = str(n)
    if n < 1000:
        return digits
    groups = []
    while digits:
        groups.insert(0, digits[-3:])
        digits = digits[:-3]
    return "_".join(groups)

print("const DEFAULT_MULTIEXP_WINDOW_THRESHOLDS: [u32; %d] = [" % len(thresholds))
for n in thresholds:
    print("    %s," % rust_literal(n))
print("];")
