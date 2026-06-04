use core::fmt;
use core::fmt::Alignment;

use num_bigint::{BigInt, BigUint};
use num_traits::{One, Signed, Zero};

use crate::bitarray::BitArray;
use crate::flexfloat::order::{Direction, adjacent};
use crate::prelude::BitArrayAccess;

use super::FlexFloat;

impl<Exp: BitArray, Frac: BitArray> FlexFloat<Exp, Frac> {
    /// Returns the shortest fixed-point decimal representation that round-trips to this value.
    pub fn to_decimal_string(&self) -> String {
        self.format_special()
            .unwrap_or_else(|| self.format_signless_fixed(None))
            .with_sign(self.sign)
    }

    fn format_special(&self) -> Option<String> {
        if self.is_nan() {
            Some(String::from("NaN"))
        } else if self.is_infinite() {
            Some(String::from("inf").with_sign(self.sign))
        } else if self.is_zero() {
            Some(String::from("0").with_sign(self.sign))
        } else {
            None
        }
    }

    fn format_signless_fixed(&self, precision: Option<usize>) -> String {
        if self.is_zero() {
            return format_zero_fixed(precision);
        }

        let abs = self.abs();
        let exact = ExactDecimal::from_flexfloat(&abs);
        match precision {
            Some(precision) => exact.render_fixed_with_precision(precision),
            None => {
                let shortest = exact.shortest_roundtrippable(&RoundingBounds::from_flexfloat(&abs));
                render_fixed(&shortest.digits, shortest.exp10)
            }
        }
    }

    fn format_signless_scientific(&self, precision: Option<usize>, exponent_char: char) -> String {
        if self.is_zero() {
            return format_zero_scientific(precision, exponent_char);
        }

        let abs = self.abs();
        let exact = ExactDecimal::from_flexfloat(&abs);
        match precision {
            Some(precision) => exact.render_scientific_with_precision(precision, exponent_char),
            None => {
                let shortest = exact.shortest_roundtrippable(&RoundingBounds::from_flexfloat(&abs));
                render_scientific(
                    &shortest.digits,
                    scientific_exponent(&shortest.digits, shortest.exp10),
                    exponent_char,
                )
            }
        }
    }

    fn fmt_non_special(&self, f: &mut fmt::Formatter<'_>, body: String) -> fmt::Result {
        f.pad_integral(!self.sign, "", &body)
    }

    fn fmt_with(
        &self,
        f: &mut fmt::Formatter<'_>,
        formatter: impl FnOnce(&Self) -> String,
    ) -> fmt::Result {
        if self.is_nan() {
            return f.write_str(&pad_special(f, "NaN"));
        }
        if self.is_infinite() {
            return f.pad_integral(!self.sign, "", "inf");
        }

        self.fmt_non_special(f, formatter(self))
    }
}

impl<Exp: BitArray, Frac: BitArray> fmt::Display for FlexFloat<Exp, Frac> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let precision = f.precision();
        self.fmt_with(f, |value| value.format_signless_fixed(precision))
    }
}

impl<Exp: BitArray, Frac: BitArray> fmt::LowerExp for FlexFloat<Exp, Frac> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let precision = f.precision();
        self.fmt_with(f, |value| value.format_signless_scientific(precision, 'e'))
    }
}

impl<Exp: BitArray, Frac: BitArray> fmt::UpperExp for FlexFloat<Exp, Frac> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let precision = f.precision();
        self.fmt_with(f, |value| value.format_signless_scientific(precision, 'E'))
    }
}

struct ExactDecimal {
    digits: String,
    exp10: i64,
}

impl ExactDecimal {
    fn from_flexfloat<Exp: BitArray, Frac: BitArray>(value: &FlexFloat<Exp, Frac>) -> Self {
        let (mantissa, exponent) = normalized_mantissa_and_exponent(value);
        let fraction_bits = BigInt::from(value.fraction.len());
        let binary_scale = exponent - fraction_bits;

        let (digits, exp10) = if binary_scale >= BigInt::zero() {
            let shift = usize::try_from(binary_scale).expect("non-negative exponent fits usize");
            ((mantissa << shift).to_str_radix(10), 0)
        } else {
            let scale =
                usize::try_from(-binary_scale).expect("negative exponent magnitude fits usize");
            let scaled = mantissa * pow5(scale);
            (
                scaled.to_str_radix(10),
                -i64::try_from(scale).expect("scale fits i64"),
            )
        };

        let mut exact = Self { digits, exp10 };
        exact.normalize();
        exact
    }

    fn normalize(&mut self) {
        while self.digits.len() > 1 && self.digits.ends_with('0') {
            self.digits.pop();
            self.exp10 += 1;
        }
    }

    fn shortest_roundtrippable(&self, bounds: &RoundingBounds) -> ShortenedDecimal {
        let scientific_exp = scientific_exponent(&self.digits, self.exp10);
        let exact = Rational::from_decimal_digits(&self.digits, self.exp10);

        for keep in 1..=self.digits.len() {
            let base_exp10 =
                scientific_exp - i64::try_from(keep).expect("digit length fits i64") + 1;
            if let Some(shortest) = bounds.shortest_at_precision(&exact, keep, base_exp10) {
                return shortest;
            }
            if let Some(shortest) = bounds.shortest_at_precision(&exact, keep, base_exp10 + 1) {
                return shortest;
            }
        }

        ShortenedDecimal {
            digits: self.digits.clone(),
            exp10: self.exp10,
        }
    }

    fn render_fixed_with_precision(&self, precision: usize) -> String {
        let coefficient = rounded_fixed_coefficient(&self.digits, self.exp10, precision);
        format_fixed_from_coefficient(&coefficient, precision)
    }

    fn render_scientific_with_precision(&self, precision: usize, exponent_char: char) -> String {
        let significant_digits = precision + 1;
        let mut exponent = scientific_exponent(&self.digits, self.exp10);
        let coefficient = rounded_significant_digits(&self.digits, significant_digits);

        let coefficient = if coefficient.len() > significant_digits {
            exponent += 1;
            coefficient[..significant_digits].to_string()
        } else if coefficient.len() < significant_digits {
            let mut padded = coefficient;
            padded.push_str(&"0".repeat(significant_digits - padded.len()));
            padded
        } else {
            coefficient
        };

        render_scientific(&coefficient, exponent, exponent_char)
    }
}

struct ShortenedDecimal {
    digits: String,
    exp10: i64,
}

impl ShortenedDecimal {
    fn normalize(&mut self) {
        while self.digits.len() > 1 && self.digits.ends_with('0') {
            self.digits.pop();
            self.exp10 += 1;
        }
    }
}

trait DecimalSign {
    fn with_sign(self, negative: bool) -> String;
}

impl DecimalSign for String {
    fn with_sign(mut self, negative: bool) -> String {
        if negative {
            self.insert(0, '-');
        }
        self
    }
}

struct RoundingBounds {
    lower: Rational,
    upper: Rational,
    inclusive: bool,
}

impl RoundingBounds {
    fn from_flexfloat<Exp: BitArray, Frac: BitArray>(value: &FlexFloat<Exp, Frac>) -> Self {
        let current = Rational::from_flexfloat(value);
        let previous = Rational::from_flexfloat(&adjacent(value, Direction::Down));
        let next = Rational::from_flexfloat(&adjacent(value, Direction::Up));

        Self {
            lower: Rational::midpoint(&previous, &current),
            upper: Rational::midpoint(&current, &next),
            inclusive: significand_is_even(value),
        }
    }

    fn shortest_at_precision(
        &self,
        exact: &Rational,
        digits: usize,
        exp10: i64,
    ) -> Option<ShortenedDecimal> {
        let min_digits = if digits == 1 {
            BigInt::one()
        } else {
            BigInt::from(pow10(digits - 1))
        };
        let max_digits = BigInt::from(pow10(digits)) - 1u8;

        let lower = self.lower_integer_bound(exp10);
        let upper = self.upper_integer_bound(exp10);
        let lower = lower.max(min_digits);
        let upper = upper.min(max_digits);

        if lower > upper {
            return None;
        }

        let chosen = choose_closest_integer(exact, exp10, &lower, &upper);
        let mut shortened = ShortenedDecimal {
            digits: chosen.to_str_radix(10),
            exp10,
        };
        shortened.normalize();
        Some(shortened)
    }

    fn lower_integer_bound(&self, exp10: i64) -> BigInt {
        scaled_integer_bound(&self.lower, exp10, self.inclusive, BoundDirection::Lower)
    }

    fn upper_integer_bound(&self, exp10: i64) -> BigInt {
        scaled_integer_bound(&self.upper, exp10, self.inclusive, BoundDirection::Upper)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Rational {
    numerator: BigInt,
    denominator: BigInt,
}

impl Rational {
    fn new(numerator: BigInt, denominator: BigInt) -> Self {
        debug_assert!(!denominator.is_zero());
        debug_assert!(denominator.is_positive());
        Self {
            numerator,
            denominator,
        }
    }

    fn from_flexfloat<Exp: BitArray, Frac: BitArray>(value: &FlexFloat<Exp, Frac>) -> Self {
        let (mantissa, exponent) = normalized_mantissa_and_exponent(value);
        let mut numerator = BigInt::from(mantissa);
        let mut denominator = BigInt::one();
        let binary_scale = exponent - BigInt::from(value.fraction.len());

        if binary_scale >= BigInt::zero() {
            let shift = usize::try_from(binary_scale).expect("non-negative exponent fits usize");
            numerator <<= shift;
        } else {
            let shift =
                usize::try_from(-binary_scale).expect("negative exponent magnitude fits usize");
            denominator <<= shift;
        }

        if value.sign {
            numerator = -numerator;
        }

        Self::new(numerator, denominator)
    }

    fn from_decimal_digits(digits: &str, exp10: i64) -> Self {
        let numerator = BigInt::parse_bytes(digits.as_bytes(), 10).expect("valid decimal digits");
        if exp10 >= 0 {
            let scale = pow10(usize::try_from(exp10).expect("exp10 fits usize"));
            Self::new(numerator * BigInt::from(scale), BigInt::one())
        } else {
            let scale =
                pow10(usize::try_from(-exp10).expect("negative exp10 magnitude fits usize"));
            Self::new(numerator, BigInt::from(scale))
        }
    }

    fn midpoint(lhs: &Self, rhs: &Self) -> Self {
        let numerator = &lhs.numerator * &rhs.denominator + &rhs.numerator * &lhs.denominator;
        let denominator = (&lhs.denominator * &rhs.denominator) << 1usize;
        Self::new(numerator, denominator)
    }

    fn scaled_by_pow10(&self, power: i64) -> Self {
        if power >= 0 {
            let scale = BigInt::from(pow10(usize::try_from(power).expect("power fits usize")));
            Self::new(&self.numerator * scale, self.denominator.clone())
        } else {
            let scale = BigInt::from(pow10(
                usize::try_from(-power).expect("negative power magnitude fits usize"),
            ));
            Self::new(self.numerator.clone(), &self.denominator * scale)
        }
    }
}

impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Rational {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        (&self.numerator * &other.denominator).cmp(&(&other.numerator * &self.denominator))
    }
}

#[derive(Clone, Copy)]
enum BoundDirection {
    Lower,
    Upper,
}

fn normalized_mantissa_and_exponent<Exp: BitArray, Frac: BitArray>(
    value: &FlexFloat<Exp, Frac>,
) -> (BigUint, BigInt) {
    let stored_exponent = value.exponent.to_bigint();
    let fraction = value.fraction.to_biguint();
    if value.exponent.is_zeros() {
        (fraction, BigInt::from(-1022))
    } else {
        (
            fraction + (BigUint::one() << value.fraction.len()),
            stored_exponent + 1_u8,
        )
    }
}

fn scientific_exponent(digits: &str, exp10: i64) -> i64 {
    i64::try_from(digits.len()).expect("digit length fits i64") + exp10 - 1
}

fn pad_special(f: &fmt::Formatter<'_>, value: &str) -> String {
    let Some(width) = f.width() else {
        return value.to_string();
    };
    if value.len() >= width {
        return value.to_string();
    }

    let fill = if f.sign_aware_zero_pad() {
        '0'
    } else {
        f.fill()
    };
    let pad_len = width - value.len();
    let align = f.align().unwrap_or(Alignment::Right);

    match align {
        Alignment::Left => format!("{value}{}", fill.to_string().repeat(pad_len)),
        Alignment::Center => {
            let left = pad_len / 2;
            let right = pad_len - left;
            format!(
                "{}{}{}",
                fill.to_string().repeat(left),
                value,
                fill.to_string().repeat(right)
            )
        }
        Alignment::Right => format!("{}{value}", fill.to_string().repeat(pad_len)),
    }
}

fn render_fixed(digits: &str, exp10: i64) -> String {
    if exp10 >= 0 {
        let zeros = "0".repeat(usize::try_from(exp10).expect("positive exponent fits usize"));
        return format!("{digits}{zeros}");
    }

    let point = i64::try_from(digits.len()).expect("digit length fits i64") + exp10;
    if point > 0 {
        let point = usize::try_from(point).expect("positive point fits usize");
        format!("{}.{}", &digits[..point], &digits[point..])
    } else {
        let zeros = "0".repeat(usize::try_from(-point).expect("padding fits usize"));
        format!("0.{zeros}{digits}")
    }
}

fn render_scientific(digits: &str, exponent: i64, exponent_char: char) -> String {
    if digits.len() == 1 {
        format!("{digits}{exponent_char}{exponent}")
    } else {
        format!("{}.{}{exponent_char}{exponent}", &digits[..1], &digits[1..])
    }
}

fn format_zero_fixed(precision: Option<usize>) -> String {
    format_zero(precision, None)
}

fn format_zero_scientific(precision: Option<usize>, exponent_char: char) -> String {
    format_zero(precision, Some(exponent_char))
}

fn format_zero(precision: Option<usize>, exponent_char: Option<char>) -> String {
    match precision {
        Some(0) | None => match exponent_char {
            Some(exponent_char) => format!("0{exponent_char}0"),
            None => String::from("0"),
        },
        Some(precision) => match exponent_char {
            Some(exponent_char) => format!("0.{}{exponent_char}0", "0".repeat(precision)),
            None => format!("0.{}", "0".repeat(precision)),
        },
    }
}

fn rounded_fixed_coefficient(digits: &str, exp10: i64, precision: usize) -> String {
    let target_exp10 = -i64::try_from(precision).expect("precision fits i64");
    if exp10 >= target_exp10 {
        let zeros = usize::try_from(exp10 - target_exp10).expect("padding fits usize");
        return format!("{digits}{}", "0".repeat(zeros));
    }

    let remove = usize::try_from(target_exp10 - exp10).expect("rounding distance fits usize");
    round_decimal_quotient(digits, remove)
}

fn rounded_significant_digits(digits: &str, significant_digits: usize) -> String {
    if digits.len() <= significant_digits {
        return digits.to_string();
    }

    round_decimal_digits(digits, digits.len() - significant_digits)
}

fn round_decimal_digits(digits: &str, remove: usize) -> String {
    round_decimal(digits, remove, 0)
}

fn round_decimal_quotient(digits: &str, divisor_power: usize) -> String {
    round_decimal(
        digits,
        divisor_power,
        divisor_power.saturating_sub(digits.len()),
    )
}

fn round_decimal(digits: &str, remove: usize, leading_zeroes: usize) -> String {
    if remove == 0 {
        return digits.to_string();
    }

    if remove >= digits.len() {
        let removed = if leading_zeroes == 0 {
            digits.as_bytes().to_vec()
        } else {
            let mut removed = vec![b'0'; leading_zeroes];
            removed.extend_from_slice(digits.as_bytes());
            removed
        };

        let should_round_up = should_round_up(
            None,
            removed.first().copied().unwrap_or(b'0'),
            &removed[1..],
        );
        return if should_round_up {
            String::from("1")
        } else {
            String::from("0")
        };
    }

    let split_at = digits.len() - remove;
    let kept = &digits.as_bytes()[..split_at];
    let removed = &digits.as_bytes()[split_at..];

    let mut rounded = kept.to_vec();
    if should_round_up(kept.last().copied(), removed[0], &removed[1..]) {
        add_one_decimal(&mut rounded);
    }

    String::from_utf8(rounded).expect("rounded decimal digits are valid utf-8")
}

fn should_round_up(last_kept: Option<u8>, first_removed: u8, rest_removed: &[u8]) -> bool {
    match first_removed.cmp(&b'5') {
        core::cmp::Ordering::Greater => true,
        core::cmp::Ordering::Less => false,
        core::cmp::Ordering::Equal => {
            if rest_removed.iter().any(|digit| *digit != b'0') {
                true
            } else {
                last_kept.is_some_and(|digit| (digit - b'0') % 2 == 1)
            }
        }
    }
}

fn add_one_decimal(digits: &mut Vec<u8>) {
    for digit in digits.iter_mut().rev() {
        if *digit < b'9' {
            *digit += 1;
            return;
        }
        *digit = b'0';
    }
    digits.insert(0, b'1');
}

fn scaled_integer_bound(
    rational: &Rational,
    exp10: i64,
    inclusive: bool,
    direction: BoundDirection,
) -> BigInt {
    let scaled = rational.scaled_by_pow10(-exp10);
    match (direction, inclusive) {
        (BoundDirection::Lower, true) => ceil_div_positive(&scaled.numerator, &scaled.denominator),
        (BoundDirection::Lower, false) => {
            floor_div_positive(&scaled.numerator, &scaled.denominator) + 1u8
        }
        (BoundDirection::Upper, true) => floor_div_positive(&scaled.numerator, &scaled.denominator),
        (BoundDirection::Upper, false) => {
            ceil_div_positive(&scaled.numerator, &scaled.denominator) - 1u8
        }
    }
}

fn choose_closest_integer(exact: &Rational, exp10: i64, lower: &BigInt, upper: &BigInt) -> BigInt {
    let scaled = exact.scaled_by_pow10(-exp10);
    let rounded = round_rational_to_even_integer(&scaled);
    if rounded < *lower {
        lower.clone()
    } else if rounded > *upper {
        upper.clone()
    } else {
        rounded
    }
}

fn floor_div_positive(numerator: &BigInt, denominator: &BigInt) -> BigInt {
    debug_assert!(denominator.is_positive());
    numerator / denominator
}

fn ceil_div_positive(numerator: &BigInt, denominator: &BigInt) -> BigInt {
    debug_assert!(denominator.is_positive());
    let quotient = numerator / denominator;
    let remainder = numerator % denominator;
    if remainder.is_zero() {
        quotient
    } else {
        quotient + 1u8
    }
}

fn round_rational_to_even_integer(value: &Rational) -> BigInt {
    let quotient = &value.numerator / &value.denominator;
    let remainder = &value.numerator % &value.denominator;
    let doubled = &remainder << 1usize;

    if doubled < value.denominator {
        quotient
    } else if doubled > value.denominator {
        quotient + 1u8
    } else if (&quotient & BigInt::one()).is_zero() {
        quotient
    } else {
        quotient + 1u8
    }
}

fn significand_is_even<Exp, Frac: BitArrayAccess>(value: &FlexFloat<Exp, Frac>) -> bool {
    value.fraction.get(0).is_none_or(|bit| !bit)
}

fn format_fixed_from_coefficient(coefficient: &str, precision: usize) -> String {
    if precision == 0 {
        return coefficient.to_string();
    }

    if coefficient.len() > precision {
        let point = coefficient.len() - precision;
        format!("{}.{}", &coefficient[..point], &coefficient[point..])
    } else {
        format!(
            "0.{}{}",
            "0".repeat(precision - coefficient.len()),
            coefficient
        )
    }
}

fn pow5(power: usize) -> BigUint {
    BigUint::from(5u8).pow(u32::try_from(power).expect("power fits u32"))
}

fn pow10(power: usize) -> BigUint {
    BigUint::from(10u8).pow(u32::try_from(power).expect("power fits u32"))
}

#[cfg(test)]
mod tests {
    use core::str::FromStr;

    use rand::Rng;
    use rstest::rstest;

    use crate::FlexFloat;
    use crate::test_support::{n_experiments, random_f64, rng};

    #[test]
    fn test_display_special_values() {
        assert_eq!(FlexFloat::nan().to_string(), "NaN");
        assert_eq!(FlexFloat::pos_infinity().to_string(), "inf");
        assert_eq!(FlexFloat::neg_infinity().to_string(), "-inf");
        assert_eq!(FlexFloat::zero().to_string(), "0");
        assert_eq!(FlexFloat::zero_with_sign(true).to_string(), "-0");
    }

    #[test]
    fn test_display_known_values() {
        let cases = [
            (1.0, "1"),
            (1.5, "1.5"),
            (0.125, "0.125"),
            (10.0, "10"),
            (1000000.0, "1000000"),
            (1e20, "100000000000000000000"),
            (1e-5, "0.00001"),
            (core::f64::consts::PI, "3.141592653589793"),
        ];

        for (value, expected) in cases {
            let ff = FlexFloat::from(value);
            assert_eq!(ff.to_string(), expected, "value={value}");
        }
    }

    #[rstest]
    fn test_display_roundtrips_f64(mut rng: impl Rng, n_experiments: usize) {
        for _ in 0..(n_experiments / 100) {
            let value = random_f64(&mut rng);
            let ff = FlexFloat::from(value);
            let rendered = ff.to_string();

            if value == 0.0 && value.is_sign_negative() {
                assert_eq!(rendered, "-0");
                continue;
            }

            let reparsed = f64::from_str(&rendered).unwrap();
            assert_eq!(
                reparsed.to_bits(),
                value.to_bits(),
                "value={value}, rendered={rendered}"
            );
        }
    }

    #[rstest]
    fn test_display_matches_f64(mut rng: impl Rng, n_experiments: usize) {
        for _ in 0..(n_experiments / 100) {
            let value = random_f64(&mut rng);
            let ff = FlexFloat::from(value);
            assert_eq!(format!("{ff}"), format!("{value}"), "value={value}");
        }
    }

    #[test]
    fn test_display_precision_matches_f64() {
        let values = [
            0.0,
            -0.0,
            0.1,
            0.3,
            1.0,
            1.5,
            2.5,
            3.5,
            1.25,
            1.35,
            1.45,
            core::f64::consts::PI,
            1e20,
            1e-20,
        ];

        for precision in [0usize, 1, 2, 3, 6, 17, 20, 55] {
            for value in values {
                let ff = FlexFloat::from(value);
                assert_eq!(
                    format!("{:.*}", precision, ff),
                    format!("{:.*}", precision, value),
                    "value={value}, precision={precision}"
                );
            }
        }
    }

    #[test]
    fn test_lower_exp_matches_f64() {
        let values = [
            0.0,
            -0.0,
            0.1,
            1.0,
            1.5,
            123.456,
            1e20,
            1e-20,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];

        for value in values {
            let ff = FlexFloat::from(value);
            assert_eq!(format!("{ff:e}"), format!("{value:e}"), "value={value:?}");
        }
    }

    #[test]
    fn test_exp_precision_matches_f64() {
        let values = [
            0.0,
            -0.0,
            0.1,
            1.0,
            1.5,
            2.5,
            core::f64::consts::PI,
            123.456,
            1e20,
            1e-20,
        ];

        for precision in [0usize, 1, 2, 3, 6, 17] {
            for value in values {
                let ff = FlexFloat::from(value);
                assert_eq!(
                    format!("{:.*e}", precision, ff),
                    format!("{:.*e}", precision, value),
                    "value={value}, precision={precision}, fmt=e"
                );
                assert_eq!(
                    format!("{:.*E}", precision, ff),
                    format!("{:.*E}", precision, value),
                    "value={value}, precision={precision}, fmt=E"
                );
            }
        }
    }

    #[test]
    fn test_formatter_flags_match_f64() {
        let values = [
            0.0,
            -0.0,
            1.0,
            -2.5,
            core::f64::consts::PI,
            1e20,
            f64::INFINITY,
            f64::NAN,
        ];

        for value in values {
            let ff = FlexFloat::from(value);

            assert_eq!(
                format!("{ff:+}"),
                format!("{value:+}"),
                "value={value:?}, fmt=+"
            );
            assert_eq!(
                format!("{ff:10}"),
                format!("{value:10}"),
                "value={value:?}, fmt=10"
            );
            assert_eq!(
                format!("{ff:<10}"),
                format!("{value:<10}"),
                "value={value:?}, fmt=<10"
            );
            assert_eq!(
                format!("{ff:>10}"),
                format!("{value:>10}"),
                "value={value:?}, fmt=>10"
            );
            assert_eq!(
                format!("{ff:^10}"),
                format!("{value:^10}"),
                "value={value:?}, fmt=^10"
            );
            assert_eq!(
                format!("{ff:010}"),
                format!("{value:010}"),
                "value={value:?}, fmt=010"
            );
            assert_eq!(
                format!("{ff:+010.3}"),
                format!("{value:+010.3}"),
                "value={value:?}, fmt=+010.3"
            );
            assert_eq!(
                format!("{ff:+010.3e}"),
                format!("{value:+010.3e}"),
                "value={value:?}, fmt=+010.3e"
            );
            assert_eq!(
                format!("{ff:+010.3E}"),
                format!("{value:+010.3E}"),
                "value={value:?}, fmt=+010.3E"
            );
        }
    }

    #[test]
    fn test_shortest_boundary_cases_match_f64() {
        let values = [
            1e-20,
            1e20,
            f64::MIN_POSITIVE,
            f64::from_bits(1),
            2.2250738585072014e-308,
        ];

        for value in values {
            let ff = FlexFloat::from(value);
            assert_eq!(format!("{ff}"), format!("{value}"), "value={value}");
            assert_eq!(format!("{ff:e}"), format!("{value:e}"), "value={value}");
        }
    }
}
