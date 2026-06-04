//! Error types for FlexFloat conversion operations.

use core::fmt;

/// Error returned when a `FlexFloat` cannot be converted to `f64`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlexFloatToF64Error {
    /// The exponent field is wider than 11 bits and the value lies outside `f64` range.
    ExponentOverflow,
    /// The value is so small it would underflow to zero in `f64`.
    ExponentUnderflow,
    /// The exponent field is not exactly 11 bits wide and the value cannot be represented.
    UnsupportedWidth {
        /// The actual exponent bit width of this `FlexFloat`.
        exponent_bits: usize,
    },
}

impl fmt::Display for FlexFloatToF64Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FlexFloatToF64Error::ExponentOverflow => {
                write!(f, "FlexFloat value overflows f64 range")
            }
            FlexFloatToF64Error::ExponentUnderflow => {
                write!(f, "FlexFloat value underflows to zero in f64")
            }
            FlexFloatToF64Error::UnsupportedWidth { exponent_bits } => write!(
                f,
                "FlexFloat has {exponent_bits}-bit exponent; f64 requires exactly 11 bits"
            ),
        }
    }
}

/// Error returned when a `FlexFloat` cannot be converted to `BigInt`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlexFloatToIntError {
    /// The value is NaN.
    NotANumber,
    /// The value is infinite.
    Infinite,
    /// The value has a non-zero fractional part.
    NotAnInteger,
}

impl fmt::Display for FlexFloatToIntError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FlexFloatToIntError::NotANumber => write!(f, "cannot convert NaN to integer"),
            FlexFloatToIntError::Infinite => write!(f, "cannot convert infinity to integer"),
            FlexFloatToIntError::NotAnInteger => {
                write!(
                    f,
                    "FlexFloat has a fractional part and cannot be converted to integer"
                )
            }
        }
    }
}

/// Unified conversion error (kept for convenience).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConversionError {
    /// Wraps a `FlexFloat → f64` error.
    ToF64(FlexFloatToF64Error),
    /// Wraps a `FlexFloat → BigInt` error.
    ToInt(FlexFloatToIntError),
}

impl fmt::Display for ConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConversionError::ToF64(e) => e.fmt(f),
            ConversionError::ToInt(e) => e.fmt(f),
        }
    }
}

impl From<FlexFloatToF64Error> for ConversionError {
    fn from(e: FlexFloatToF64Error) -> Self {
        ConversionError::ToF64(e)
    }
}

impl From<FlexFloatToIntError> for ConversionError {
    fn from(e: FlexFloatToIntError) -> Self {
        ConversionError::ToInt(e)
    }
}
