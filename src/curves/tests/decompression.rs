use super::{CurveAffine, EpAffine, EqAffine};

#[test]
fn decompression_preserves_identity_and_invalid_encodings() {
    fn check<C: CurveAffine>() {
        let identity = C::identity().to_bytes();
        assert!(identity.as_ref().iter().all(|byte| *byte == 0));
        assert!(bool::from(C::from_bytes(&identity).unwrap().is_identity()));

        let mut signed_zero = identity;
        signed_zero.as_mut()[31] = 0x80;
        assert!(bool::from(C::from_bytes(&signed_zero).is_none()));

        let mut noncanonical = C::Repr::default();
        noncanonical.as_mut().fill(0xff);
        assert!(bool::from(C::from_bytes(&noncanonical).is_none()));
        assert!(bool::from(C::from_bytes_unchecked(&noncanonical).is_none()));

        let generator = C::generator();
        assert_eq!(C::from_bytes(&generator.to_bytes()).unwrap(), generator);
    }
    check::<EpAffine>();
    check::<EqAffine>();
}
