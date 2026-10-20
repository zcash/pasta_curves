# Derives the top-limb shifts and checks the lazy accumulator bound in
# src/fields/portable.rs. Run: uv run sage lazy_squaring.sage

moduli = [
    0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001,
    0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001,
]
word_bits = 64
R = 2^(4 * word_bits)
max_count = 2^32 - 1
shifts = []
for m in moduli:
    limbs = [(m >> (word_bits * j)) % 2^word_bits for j in range(4)]
    assert limbs[2] == 0
    shift = limbs[3].nbits() - 1
    assert limbs[3] == 2^shift
    shifts.append(shift)

    e = QQ(m) / R - QQ(1) / 4
    assert 0 < e < QQ(1) / 2^130
    assert QQ(m) / R < QQ(1) / 3
    # For every 0 <= i <= max_count, i + 2 >= 2 and the denominator
    # 4*(i+3)^2*(i+4) is at most its value at max_count.
    assert e < QQ(2) / (4 * (max_count + 3)^2 * (max_count + 4))

# The inductive gap before adding e is exactly the bound used above.
i = SR.var('i')
b = QQ(1) / 2 - 1 / (2 * (i + 3))
next_b = QQ(1) / 2 - 1 / (2 * (i + 4))
assert (next_b - b^2 - QQ(1) / 4 - (i + 2) / (4 * (i + 3)^2 * (i + 4))).simplify_full() == 0
assert shifts[0] == shifts[1]
print(f"const PASTA_MODULUS_TOP_SHIFT: u32 = {shifts[0]};")
print("const PASTA_MODULUS_TOP_OVERFLOW_SHIFT: u32 = u64::BITS - PASTA_MODULUS_TOP_SHIFT;")
