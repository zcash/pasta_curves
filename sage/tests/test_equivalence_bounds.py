"""The arithmetic the equivalent-mutant arguments rest on, made executable.

sage/README.md calls these layer 1: statements about the arithmetic, complete
over the whole input domain rather than sampled. Each is the maximum of a
function that is monotone in every limb, so evaluating one corner of the box
settles it. The prose arguments are in xtask/asm-mutants-aarch64-apple.txt,
one above each entry; these tests are the same arguments as code, so a moduli
change that invalidates one fails here instead of going unnoticed.

Layer 2, that the shipped instructions compute this arithmetic, is not in
reach of these tests and is not claimed by them.
"""

import random
import re
from pathlib import Path

import pytest

from asm_mutant_witnesses import MODULI, REPO_ROOT

M64 = (1 << 64) - 1
R = 1 << 256
FIELDS = sorted(MODULI)


def rust_modulus(field: str) -> int:
    """Read the `MODULUS` limbs back out of the field's Rust source."""
    source = Path(REPO_ROOT / f"src/fields/{field.lower()}.rs").read_text()
    body = re.search(
        rf"const MODULUS: {field} = {field}\(\[(.*?)\]\)", source, re.S
    )
    assert body is not None, f"no MODULUS in {field.lower()}.rs"
    limbs = [int(limb, 16) for limb in re.findall(r"0x[0-9a-fA-F]+", body[1])]
    return sum(limb << (64 * index) for index, limb in enumerate(limbs))


@pytest.mark.parametrize("field", FIELDS)
def test_moduli_match_the_crate(field: str) -> None:
    """The copy of `p` here is the one the crate actually uses."""
    assert MODULI[field] == rust_modulus(field)


@pytest.mark.parametrize("field", FIELDS)
def test_top_limb_is_two_to_the_62(field: str) -> None:
    """`p[3] = 2^62`, hence `p < 2^255` and `2p < 2^256`."""
    modulus = MODULI[field]
    assert modulus >> 192 == 1 << 62
    assert 2 * modulus < R


@pytest.mark.parametrize("field", FIELDS)
def test_mul_candidate_never_sets_the_fifth_limb(field: str) -> None:
    """adc/15: with `lhs < 2^256` and `rhs < p` the candidate stays below `2p`."""
    modulus = MODULI[field]
    # `(lhs * rhs + q * p) / R` at its corner: lhs = R - 1, rhs = p - 1, q < R.
    candidate = ((R - 1) * (modulus - 1) + (R - 1) * modulus) // R
    assert candidate < 2 * modulus < R


@pytest.mark.parametrize("field", FIELDS)
def test_doubled_cross_terms_never_reach_limb_7(field: str) -> None:
    """adc/19: twice a canonical value's cross-term sum stays below `2^448`."""
    modulus = MODULI[field]
    corner = [M64, M64, M64, modulus >> 192]
    assert 2 * cross_terms(corner) < 1 << 448

    # The corner is the maximum only because the sum is monotone in each limb.
    draw = random.Random(0)
    for _ in range(200):
        value = draw.randrange(modulus)
        limbs = [(value >> (64 * index)) & M64 for index in range(4)]
        assert cross_terms(limbs) <= cross_terms(corner)


@pytest.mark.parametrize("field", FIELDS)
def test_square_candidate_never_sets_the_257th_bit(field: str) -> None:
    """adc/21: the reduced low half is at most `p`, the high half below it."""
    assert 2 * MODULI[field] + 1 < R


def cross_terms(limbs: list[int]) -> int:
    """`sum(a[i] * a[j] * 2^(64(i+j)))` over `i < j`, the doubled half of a^2."""
    return sum(
        limbs[i] * limbs[j] << (64 * (i + j))
        for i in range(4)
        for j in range(i + 1, 4)
    )
