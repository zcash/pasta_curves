"""The operand range in `src/fields/aarch64_asm.rs`, as an executable check.

The contract is that `rhs` is canonical, and that an unreduced `lhs` is only
allowed against an `rhs` whose every limb is at most `2^64 - 4`. It was stated
the wrong way round before, as `rhs < p` sufficing for any `lhs`, which the
sweep in `asm_mutant_witnesses.py` never contradicted because it only ever
pairs an unreduced `lhs` with `R2` or `R3`, whose limbs happen to be small.

So both halves are pinned here: the families inside the range agree with the
Montgomery definition, and the one just outside it does not. The second is a
statement about behaviour outside the contract, not a bug the crate can reach;
`from_u512` is the only caller passing an unreduced `lhs` and it uses `R2`/`R3`.
"""

import random

import pytest

import asm_aarch64 as asm
from asm_mutant_witnesses import ASM_SOURCE, MODULI

M64 = (1 << 64) - 1
R = 1 << 256
CAP = M64 - 3  # the largest rhs limb an unreduced lhs may meet
FIELDS = sorted(MODULI)


@pytest.fixture(scope="module")
def program() -> asm.Program:
    """The shipped assembly, parsed once for every case below."""
    return asm.Program(ASM_SOURCE.read_text())


def limbs(value: int) -> list[int]:
    """Little-endian 64-bit limbs of a 256-bit value."""
    return [(value >> (64 * index)) & M64 for index in range(4)]


def from_limbs(values: list[int]) -> int:
    """The integer those limbs denote."""
    return sum(limb << (64 * index) for index, limb in enumerate(values))


def montgomery(program: asm.Program, field: str, lhs: int, rhs: int) -> bool:
    """Whether the backend's `mul` matches `lhs * rhs * R^-1 mod p`."""
    modulus = MODULI[field]
    inverse = pow(-modulus, -1, 1 << 64)
    got, _ = asm.mul(program, lhs, rhs, modulus, inverse)
    return got == lhs * rhs * pow(R, -1, modulus) % modulus


@pytest.mark.parametrize("field", FIELDS)
def test_r2_and_r3_limbs_are_capped(field: str) -> None:
    """What keeps `from_u512`'s unreduced left operand inside the range."""
    modulus = MODULI[field]
    for constant in ((1 << 512) % modulus, (1 << 768) % modulus):
        assert all(limb <= CAP for limb in limbs(constant))


@pytest.mark.parametrize("field", FIELDS)
def test_canonical_operands_always_agree(
    program: asm.Program, field: str
) -> None:
    """The unconditional half of the contract."""
    modulus = MODULI[field]
    draw = random.Random(11)
    for _ in range(60):
        assert montgomery(
            program, field, draw.randrange(modulus), draw.randrange(modulus)
        )


@pytest.mark.parametrize("field", FIELDS)
def test_unreduced_lhs_agrees_against_a_capped_rhs(
    program: asm.Program, field: str
) -> None:
    """An unreduced left operand is safe exactly when the right one is capped."""
    modulus = MODULI[field]
    draw = random.Random(12)
    for _ in range(60):
        rhs = from_limbs([draw.randrange(CAP + 1) for _ in range(4)])
        if rhs >= modulus:
            continue
        assert montgomery(program, field, draw.randrange(R), rhs)


@pytest.mark.parametrize("field", FIELDS)
def test_unreduced_lhs_diverges_against_an_uncapped_rhs(
    program: asm.Program, field: str
) -> None:
    """Just outside the range the backend is wrong, not merely non-canonical."""
    lhs = from_limbs([0, M64, M64, M64])
    rhs = from_limbs([M64, M64, 0, 0])
    assert rhs < MODULI[field], "the witness needs a canonical rhs"
    assert not montgomery(program, field, lhs, rhs)
