use crate::bitarray::{BitArray, DefaultBitArray};
use crate::flexfloat::FlexFloat;

impl<B: BitArray> FlexFloat<B> {
    pub fn from_f64(value: f64) -> Self {
        let bits = B::from_f64(value);

        Self {
            sign: *bits.get(63).unwrap(),
            exponent: bits.get_range(52..63).unwrap(),
            fraction: bits.get_range(0..52).unwrap(),
        }
    }

    pub fn to_f64(&self) -> Option<f64> {
        if self.exponent.len() < 11 || self.fraction.len() < 52 {
            return None;
        }

        let mut bits: Vec<bool> = vec![false; 64];
        bits[63] = self.sign;
        bits[52..63].copy_from_slice(&self.exponent.to_bits()[0..11]);
        bits[0..52].copy_from_slice(&self.fraction.to_bits()[0..52]);
        Some(
            B::from_bits(bits.as_slice())
                .to_float()
                .expect("Underlaying BitArray cannot convert to f64"),
        )
    }
}

impl From<f64> for FlexFloat<DefaultBitArray> {
    fn from(value: f64) -> Self {
        Self::from_f64(value)
    }
}

impl From<FlexFloat<DefaultBitArray>> for f64 {
    fn from(value: FlexFloat<DefaultBitArray>) -> Self {
        value.to_f64().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use rand::Rng;
    use rstest::rstest;

    use crate::tests::n_experiments;

    use super::*;
    use crate::tests::*;

    fn test_from_f64(mut rng: impl Rng, n_experiments: usize) {
        let special_values = [
            (0.0, false, vec![false; 11], vec![false; 52]),
            (-0.0, true, vec![false; 11], vec![false; 52]),
            (f64::INFINITY, false, vec![true; 11], vec![false; 52]),
            (f64::NEG_INFINITY, true, vec![true; 11], vec![false; 52]),
        ];

        for (value, sign, exponent, fraction) in special_values {
            let ff = FlexFloat::from(value);
            assert_eq!(ff.sign, sign);
            assert_eq!(ff.exponent.to_bits(), exponent);
            assert_eq!(ff.fraction.to_bits(), fraction);
        }

        let ff = FlexFloat::from(f64::NAN);
        assert!(ff.is_nan());

        for _ in 0..n_experiments {
            let value: f64 = rng.random();
            let ff = FlexFloat::from(value);
            assert_eq!(value.is_sign_negative(), ff.sign);
            assert_eq!(value.is_nan(), ff.is_nan());
        }
    }

    fn test_to_f64(mut rng: impl Rng, n_experiments: usize) {
        let special_values = [
            0.0,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MIN_POSITIVE,
            f64::NAN,
            f64::MAX,
            f64::MIN,
        ];

        for value in special_values {
            let ff = FlexFloat::from(value);
            let converted_value: f64 = ff.into();
            if value.is_nan() {
                assert!(converted_value.is_nan());
            } else {
                assert_eq!(value, converted_value);
            }
        }

        let ff = FlexFloat {
            sign: false,
            exponent: DefaultBitArray::ones(11),
            fraction: DefaultBitArray::from_bits(&[true; 52]),
        };
        let converted_value: f64 = ff.into();
        assert!(converted_value.is_nan());

        for _ in 0..n_experiments {
            let value: f64 = rng.random();
            let ff = FlexFloat::from(value);
            let converted_value: f64 = ff.into();
            if value.is_nan() {
                assert!(converted_value.is_nan());
            } else {
                assert_eq!(value, converted_value);
            }
        }
    }

    #[rstest]
    fn test_conversion(mut rng: impl Rng, n_experiments: usize) {
        test_from_f64(&mut rng, n_experiments);
        test_to_f64(&mut rng, n_experiments);
    }
}
