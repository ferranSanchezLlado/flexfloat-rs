use std::cmp::Ordering;

use crate::bitarray::BitArray;
use crate::flexfloat::FlexFloat;

impl<B1, B2> PartialEq<FlexFloat<B1>> for FlexFloat<B2>
where
    B1: BitArray,
    B2: BitArray,
{
    fn eq(&self, other: &FlexFloat<B1>) -> bool {
        if self.is_nan() || other.is_nan() {
            return false;
        }
        self.sign == other.sign
            && self.exponent.to_bigint() == other.exponent.to_bigint()
            && self.fraction.to_biguint() == other.fraction.to_biguint()
    }
}

impl<B1, B2> PartialOrd<FlexFloat<B1>> for FlexFloat<B2>
where
    B1: BitArray,
    B2: BitArray,
{
    fn partial_cmp(&self, other: &FlexFloat<B1>) -> Option<Ordering> {
        if self.is_nan() || other.is_nan() {
            return None;
        }

        let sign = self.sign.cmp(&other.sign);
        let exponent = self.exponent.to_bigint().cmp(&other.exponent.to_bigint());
        let fraction = self.fraction.to_biguint().cmp(&other.fraction.to_biguint());

        match (sign, exponent, fraction) {
            (Ordering::Equal, Ordering::Equal, Ordering::Equal) => Some(Ordering::Equal),
            (Ordering::Less, _, _) => Some(Ordering::Greater),
            (Ordering::Equal, Ordering::Greater, _) => Some(Ordering::Greater),
            (Ordering::Equal, Ordering::Equal, Ordering::Greater) => Some(Ordering::Greater),
            _ => Some(Ordering::Less),
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;
    use rstest::rstest;

    use super::*;
    use crate::tests::*;

    #[rstest]
    fn test_partial_eq(mut rng: impl Rng, n_experiments: usize) {
        let a = FlexFloat::from(1.5);
        let b = FlexFloat::from(2.5);
        assert_eq!(a, a);
        assert_ne!(a, b);

        let nan = FlexFloat::nan();
        assert!(nan != nan);

        for _ in 0..n_experiments {
            let a: f64 = rng.random();
            let b: f64 = rng.random();
            let expected1 = a == b;
            let expected2 = a == a;

            let fa = FlexFloat::from(a);
            let fb = FlexFloat::from(b);
            let result1 = fa == fb;
            let result2 = fa == fa;

            assert_eq!(result1, expected1);
            assert_eq!(result2, expected2);
        }
    }

    #[rstest]
    #[allow(clippy::bool_comparison)]
    fn test_partial_ord(mut rng: impl Rng, n_experiments: usize) {
        let a = FlexFloat::from(1.5);
        let b = FlexFloat::from(2.5);
        assert!(a < b,);
        assert!((a >= b) == false);

        let nan = FlexFloat::nan();
        assert!((nan > nan) == false);

        for _ in 0..n_experiments {
            let a: f64 = rng.random();
            let b: f64 = rng.random();
            let expected1 = a < b;
            let expected2 = a >= a;

            let fa = FlexFloat::from(a);
            let fb = FlexFloat::from(b);
            let result1 = fa < fb;
            let result2 = fa >= fa;

            assert_eq!(result1, expected1);
            assert_eq!(result2, expected2);
        }
    }
}
