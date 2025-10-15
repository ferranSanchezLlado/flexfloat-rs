use std::ops::{Add, Neg, Sub};

use num_bigint::BigInt;
use num_traits::sign::Signed;

use crate::bitarray::BitArray;
use crate::flexfloat::FlexFloat;

impl<B: BitArray> FlexFloat<B> {
    pub fn abs(&self) -> Self
    where
        B: Clone,
    {
        Self {
            sign: false,
            exponent: self.exponent.clone(),
            fraction: self.fraction.clone(),
        }
    }
}

impl<B: BitArray> Neg for FlexFloat<B> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            sign: !self.sign,
            exponent: self.exponent,
            fraction: self.fraction,
        }
    }
}

impl<B: BitArray> FlexFloat<B> {
    fn grow_exponent_bits(exp: &BigInt, current_len: usize) -> usize {
        let mut exponent_length = current_len;
        loop {
            let half = 1 << (exponent_length - 1);
            let min_exponent = BigInt::from(-half);
            let max_exponent = BigInt::from(half - 1);

            if &min_exponent <= exp && exp <= &max_exponent {
                break;
            }
            exponent_length += 1;
        }

        exponent_length
    }
}

impl<B: BitArray> Add for FlexFloat<B> {
    type Output = Self;

    fn add(mut self, mut rhs: Self) -> Self::Output {
        if self.sign != rhs.sign {
            return self - (-rhs);
        }

        // OBJECTIVE: Add two FlexFloat instances together.
        // https://www.sciencedirect.com/topics/computer-science/floating-point-addition
        // and: https://cse.hkust.edu.hk/~cktang/cs180/notes/lec21.pdf
        //
        // Steps:
        // 0. Handle special cases (NaN, Infinity).
        // 1. Extract exponent and fraction bits.
        // 2. Prepend leading 1 to form the mantissa.
        // 3. Compare exponents.
        // 4. Shift smaller mantissa if necessary.
        // 5. Add mantissas.
        // 6. Normalize mantissa and adjust exponent if necessary.
        // 7. Grow exponent if necessary. (no limit on size)
        // 8. Round result.
        // 9. Return new FlexFloat instance.

        // 0. Handle special cases (NaN, Infinity).
        match (self.is_nan(), rhs.is_nan()) {
            (true, _) => return self,
            (_, true) => return rhs,
            _ => {}
        }

        match (self.is_infinity(), rhs.is_infinity(), self.sign == rhs.sign) {
            (true, true, true) => return self, // inf + inf = inf
            (true, true, false) => return FlexFloat::new_nan(), // inf + -inf = NaN
            (true, false, _) => return self,   // inf + x = inf
            (false, true, _) => return rhs,    // x + inf = inf
            _ => {}
        }

        // 1. Extract exponent and fraction bits.
        // 2. Prepend leading 1 to form the mantissa.
        // 3. Compare exponents.
        let mut exp_self = self.exponent.to_bigint() + 1_u8;
        let mut exp_rhs = rhs.exponent.to_bigint() + 1_u8;

        // Compare to make sure self has the larger exponent.
        if exp_self < exp_rhs {
            std::mem::swap(&mut self, &mut rhs);
            std::mem::swap(&mut exp_self, &mut exp_rhs);
        }

        let mant_self = self.fraction.append_bool_in_place(true);
        let mant_rhs = rhs.fraction.append_bool_in_place(true);

        // 4. Shift smaller mantissa if necessary.
        let exp_diff = exp_self.clone() - exp_rhs.clone();
        assert!(
            exp_diff >= BigInt::ZERO,
            "Self exponent should be larger/equal"
        );
        let mant_rhs = mant_rhs.shift(exp_diff.try_into().expect("Exponent difference too large"));

        assert_eq!(mant_self.len(), 53, "Mantissa length should be 53 bits");
        assert_eq!(mant_rhs.len(), 53, "Mantissa length should be 53 bits");

        // 5. Add mantissas.
        let mut mantissa_result = B::zeros(53);
        let mut carry: u8 = 0;
        for i in 0..53 {
            let self_bit = *mant_self.get(i).unwrap() as u8;
            let rhs_bit = *mant_rhs.get(i).unwrap() as u8;
            let sum = self_bit + rhs_bit + carry;

            *mantissa_result.get_mut(i).unwrap() = !sum.is_multiple_of(2);
            carry = sum / 2;
            assert!(carry <= 1);
        }

        // 6. Normalize mantissa and adjust exponent if necessary.
        if carry > 0 {
            mantissa_result = mantissa_result.shift(1);
            exp_self += 1_u8;
        }

        // 7. Grow exponent if necessary. (no limit on size)
        let exp_result_length = Self::grow_exponent_bits(&exp_self, self.exponent.len());

        let exponent_result = B::from_bigint(&(exp_self - 1_u8), exp_result_length)
            .expect("Exponent lenght should have grown");
        let fraction_result = B::from_bits(&mantissa_result.to_bits()[..52]);
        FlexFloat {
            sign: self.sign,
            exponent: exponent_result,
            fraction: fraction_result,
        }
    }
}

impl<B: BitArray> Sub for FlexFloat<B> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        if self.sign != rhs.sign {
            return self - (-rhs);
        }

        // OBJECTIVE: Subtract two FlexFloat instances together.
        // https://www.sciencedirect.com/topics/computer-science/floating-point-addition
        // and: https://cse.hkust.edu.hk/~cktang/cs180/notes/lec21.pdf
        //
        // Steps:
        // 0. Handle special cases (NaN, Infinity).
        // 1. Extract exponent and fraction bits.
        // 2. Prepend leading 1 to form the mantissa.
        // 3. Compare exponents.
        // 4. Shift smaller mantissa if necessary.
        // 5. Subtract mantissas.
        // 6. Normalize mantissa and adjust exponent if necessary.
        // 7. Grow exponent if necessary. (no limit on size)
        // 8. Round result.
        // 9. Return new FlexFloat instance.

        // 0. Handle special cases (NaN, Infinity).
        match (self.is_nan(), rhs.is_nan()) {
            (true, _) => return self,
            (_, true) => return rhs,
            _ => {}
        }

        match (self.is_infinity(), rhs.is_infinity(), self.sign == rhs.sign) {
            (true, true, true) => return FlexFloat::new_nan(), // inf - inf = Nan
            (true, true, false) => return self,                // inf - -inf = inf
            (true, false, _) => return self,                   // inf - x = inf
            (false, true, _) => return -rhs,                   // x - inf = -inf
            _ => {}
        }

        // 1. Extract exponent and fraction bits.
        // 2. Prepend leading 1 to form the mantissa.
        // 3. Compare exponents.
        let exp_self = self.exponent.to_bigint() + 1_u8;
        let exp_rhs = rhs.exponent.to_bigint() + 1_u8;

        let mut mant_self = self.fraction.append_bool_in_place(true);
        let mut mant_rhs = rhs.fraction.append_bool_in_place(true);

        // 4. Shift smaller mantissa if necessary.
        let exp_diff = exp_self.clone() - exp_rhs.clone();

        let mut exp = match exp_diff.is_positive() {
            true => {
                mant_rhs =
                    mant_rhs.shift(exp_diff.try_into().expect("Exponent difference too large"));
                exp_self
            }
            false => {
                mant_self = mant_self.shift(
                    exp_diff
                        .abs()
                        .try_into()
                        .expect("Exponent difference too large"),
                );
                exp_rhs
            }
        };

        assert_eq!(mant_self.len(), 53, "Mantissa length should be 53 bits");
        assert_eq!(mant_rhs.len(), 53, "Mantissa length should be 53 bits");

        // 5. Subtract mantissas.
        let mant_self = if self.sign { mant_self.to_comp2() } else { mant_self };
        let mant_rhs = if !rhs.sign { mant_rhs.to_comp2() } else { mant_rhs };
        
        let mant_self = mant_self.append_bool_in_place(self.sign);
        let mant_rhs = mant_rhs.append_bool_in_place(!rhs.sign);

        let mut mantissa_result = B::zeros(54);

        let mut carry: u8 = 0;
        for i in 0..54 {
            let self_bit = *mant_self.get(i).unwrap() as u8;
            let rhs_bit = *mant_rhs.get(i).unwrap() as u8;
            let sum = self_bit + rhs_bit + carry;

            *mantissa_result.get_mut(i).unwrap() = !sum.is_multiple_of(2);
            carry = sum / 2;
            assert!(carry <= 1);
        }

        let mut mantissa_result = mantissa_result.to_comp2();
        let sign = !*mantissa_result.get(53).unwrap();

        // 6. Normalize mantissa and adjust exponent if necessary.
        let position = mantissa_result
            .to_bits()
            .into_iter()
            .rev()
            .skip(1)
            .position(|el| el);
        match position {
            None => return Self::new_zero_with_sign(sign),
            Some(i) => {
                mantissa_result = mantissa_result.shift(-(i as isize));
                exp -= i;
            }
        }

        // 7. Grow exponent if necessary. (no limit on size)
        let exp_result_length = Self::grow_exponent_bits(&exp, self.exponent.len());

        let exponent_result = B::from_bigint(&(exp - 1_u8), exp_result_length)
            .expect("Exponent lenght should have grown");
        let fraction_result = B::from_bits(&mantissa_result.to_bits()[..52]);
        FlexFloat {
            sign: sign,
            exponent: exponent_result,
            fraction: fraction_result,
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;
    use rstest::rstest;

    use super::*;
    use crate::tests::*;

    fn assert_almost_eq_within(a: f64, b: f64, epsilon: f64, message: &str) {
        assert!(
            (a - b).abs() <= epsilon,
            "{}: {} and {} differ more than {}",
            message,
            a,
            b,
            epsilon
        );
    }

    fn assert_almost_eq(a: f64, b: f64, message: &str) {
        assert_almost_eq_within(a, b, 1e-10, message);
    }

    #[rstest]
    fn test_add(mut rng: impl Rng, n_experiments: usize) {
        // Test random basic case
        let a = FlexFloat::from(1.5);
        let b = FlexFloat::from(2.25);
        let c = a + b;
        assert_eq!(c.to_f64(), Some(3.75));

        // Test overflow case
        let a = FlexFloat::from(f64::MAX);
        let b = FlexFloat::from(f64::MAX / 2.0);
        let c = a + b;
        assert!(!c.is_infinity(), "Result should not overflow");
        assert!(c.exponent.len() > 11, "Exponent should have grown");

        // Test wierd edge case
        let a = FlexFloat::from(f64::MAX);
        let b = FlexFloat::from(f64::MIN_POSITIVE);
        let c = a + b;
        assert_eq!(c.to_f64(), Some(f64::MAX));

        for _ in 0..n_experiments {
            let a: f64 = rng.random();
            let b: f64 = rng.random();
            let expected = a + b;

            let fa = FlexFloat::from(a);
            let fb = FlexFloat::from(b);
            let fc = fa + fb;

            if expected.is_infinite() {
                // TODO: check fc is larger tha f64::MAX)
                continue;
            }

            if expected.is_nan() {
                assert!(fc.is_nan());
                continue;
            }

            let result = fc.to_f64().expect("Result should fit in f64");
            assert_almost_eq(
                result,
                expected,
                format!("Failed on {} + {}", a, b).as_str(),
            );
        }
    }

    #[rstest]
    fn test_sub(mut rng: impl Rng, n_experiments: usize) {
        // Test random basic case
        let a = FlexFloat::from(1.5);
        let b = FlexFloat::from(2.25);
        let c = a - b;
        assert_eq!(c.to_f64(), Some(-0.75));

        // Test overflow case
        // let a = FlexFloat::from(-f64::MAX);
        // let b = FlexFloat::from(f64::MAX / 2.0);
        // let c = a - b;
        // assert!(!c.is_infinity(), "Result should not overflow");
        // assert!(c.exponent.len() > 11, "Exponent should have grown");

        // Test wierd edge case
        let a = FlexFloat::from(f64::MAX);
        let b = FlexFloat::from(f64::MIN_POSITIVE);
        let c = a - b;
        assert_eq!(c.to_f64(), Some(f64::MAX));

        for _ in 0..n_experiments {
            let a: f64 = rng.random();
            let b: f64 = rng.random();
            let expected = a - b;

            let fa = FlexFloat::from(a);
            let fb = FlexFloat::from(b);
            let fc = fa - fb;

            if expected.is_infinite() {
                // TODO: check fc is larger tha f64::MAX)
                continue;
            }

            if expected.is_nan() {
                assert!(fc.is_nan());
                continue;
            }

            let result = fc.to_f64().expect("Result should fit in f64");
            assert_almost_eq(
                result,
                expected,
                format!("Failed on {} - {}", a, b).as_str(),
            );
        }
    }
}
