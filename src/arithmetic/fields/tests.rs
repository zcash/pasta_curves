use super::{SqrtTableHelpers, SqrtTables};
use subtle::Choice;

impl<F: SqrtTableHelpers> SqrtTables<F> {
    // Pre-port square-root correction, with the eager exponent supplied by the field tests.
    pub(crate) fn sqrt_alt_eager(&self, u: &F, v: &F) -> (Choice, F) {
        let uv = *u * v;
        let res = self.sqrt_common_eager(&uv, v);
        let sq = res.square();
        let is_square = (sq - u).is_zero();
        let is_nonsquare = (sq - F::ROOT_OF_UNITY * u).is_zero();
        assert!(bool::from(u.is_zero() | (is_square ^ is_nonsquare)));
        (is_square, res)
    }

    fn sqrt_common_eager(&self, uv: &F, v: &F) -> F {
        let sqr = |x: F, i: u32| (0..i).fold(x, |x, _| x.square());
        let inv = |x: F| self.inv[self.hasher.hash(&x)] as usize;

        let x3 = *uv * v;
        let x2 = sqr(x3, 8);
        let x1 = sqr(x2, 8);
        let x0 = sqr(x1, 8);

        // i = 0, 1
        let mut t_ = inv(x0); // = t >> 16
        // 1 == x0 * ROOT_OF_UNITY^(t_ << 24)
        assert!(t_ < 0x100);
        let alpha = x1 * self.g2[t_];

        // i = 2
        t_ += inv(alpha) << 8; // = t >> 8
        // 1 == x1 * ROOT_OF_UNITY^(t_ << 16)
        assert!(t_ < 0x10000);
        let alpha = x2 * self.g1[t_ & 0xFF] * self.g2[t_ >> 8];

        // i = 3
        t_ += inv(alpha) << 16; // = t
        // 1 == x2 * ROOT_OF_UNITY^(t_ << 8)
        assert!(t_ < 0x1000000);
        let alpha = x3 * self.g0[t_ & 0xFF] * self.g1[(t_ >> 8) & 0xFF] * self.g2[t_ >> 16];

        t_ += inv(alpha) << 24; // = t << 1
        // 1 == x3 * ROOT_OF_UNITY^t_
        t_ = (((t_ as u64) + 1) >> 1) as usize;
        assert!(t_ <= 0x80000000);

        *uv * self.g0[t_ & 0xFF]
            * self.g1[(t_ >> 8) & 0xFF]
            * self.g2[(t_ >> 16) & 0xFF]
            * self.g3[t_ >> 24]
    }
}
