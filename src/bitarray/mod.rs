#[cfg(feature = "bigint")]
use num_bigint::{BigInt, BigUint};
use std::cmp::Ordering;
use std::hint::unreachable_unchecked;
use std::ops::{Add, Index, IndexMut, Range};

pub mod boolean_list;

pub use boolean_list::BoolBitArray;

pub type DefaultBitArray = BoolBitArray;

/// LE byte order
pub trait BitArray {
    fn from_bytes(bytes: &[u8], n_bits: usize) -> Self
    where
        Self: Sized;
    fn from_bits(bits: &[bool]) -> Self
    where
        Self: Sized,
    {
        let n_bits = bits.len();
        let n_bytes = n_bits.div_ceil(8);
        let mut bytes = vec![0u8; n_bytes];

        for (i, &bit) in bits.iter().enumerate() {
            if bit {
                let byte_index = i / 8;
                let bit_index = i % 8;
                bytes[byte_index] |= 1 << bit_index;
            }
        }

        Self::from_bytes(&bytes, n_bits)
    }
    fn from_f64(value: f64) -> Self
    where
        Self: Sized,
    {
        let bits = value.to_bits();
        Self::from_bytes(&bits.to_le_bytes(), 64)
    }
    #[cfg(feature = "bigint")]
    fn from_biguint(value: &BigUint) -> Self
    where
        Self: Sized,
    {
        let bytes = value.to_bytes_le();
        let n_bits = bytes.len() * 8;
        Self::from_bytes(&bytes, n_bits)
    }
    #[cfg(feature = "bigint")]
    fn from_bigint(value: &BigInt, n_bits: usize) -> Option<Self>
    where
        Self: Sized,
    {
        let half = BigInt::from(1u8) << (n_bits - 1);
        let max_value = &half - 1u8;
        let min_value = -&half;

        if value < &min_value || value > &max_value {
            return None;
        }

        let unsigned_value = (value + half).to_biguint()?;
        let bytes = unsigned_value.to_bytes_le();
        Some(Self::from_bytes(&bytes, n_bits))
    }

    fn zeros(n_bits: usize) -> Self
    where
        Self: Sized,
    {
        let n_bytes = n_bits.div_ceil(8);
        let bytes = vec![0u8; n_bytes];

        Self::from_bytes(&bytes, n_bits)
    }
    fn ones(n_bits: usize) -> Self
    where
        Self: Sized,
    {
        let n_bytes = n_bits.div_ceil(8);
        let mut bytes = vec![0xffu8; n_bytes];

        let last_num_bits = n_bits % 8;
        if last_num_bits > 0 {
            bytes[n_bytes - 1] = (1 << last_num_bits) - 1;
        }

        Self::from_bytes(&bytes, n_bits)
    }

    fn to_bytes(&self) -> Vec<u8>;
    fn to_bits(&self) -> Vec<bool> {
        self.iter_bits().collect()
    }
    fn to_float(&self) -> Option<f64> {
        if self.len() != 64 {
            return None;
        }
        let bytes: [u8; 8] = self.to_bytes().try_into().ok()?;
        Some(f64::from_bits(u64::from_le_bytes(bytes)))
    }
    fn to_bits_string(&self) -> String {
        self.iter_bits()
            .map(|b| if b { '1' } else { '0' })
            .collect()
    }
    #[cfg(feature = "bigint")]
    fn to_biguint(&self) -> BigUint {
        BigUint::from_bytes_le(&self.to_bytes())
    }
    #[cfg(feature = "bigint")]
    fn to_bigint(&self) -> BigInt {
        let n_bits = self.len();
        let half = BigInt::from(1u8) << (n_bits - 1);
        let unsigned = BigInt::from(self.to_biguint());
        unsigned - half
    }

    fn iter_bits(&self) -> impl Iterator<Item = bool>;

    fn len(&self) -> usize;
    // TODO: https://doc.rust-lang.org/std/slice/trait.SliceIndex.html
    fn get(&self, index: usize) -> Option<&bool>;
    fn get_mut(&mut self, index: usize) -> Option<&mut bool>;

    fn get_range(&self, range: Range<usize>) -> Option<Self>
    where
        Self: Sized,
    {
        if range.end > self.len() || range.start > range.end {
            return None;
        }
        let bits: Vec<bool> = range.map(|i| *self.get(i).unwrap()).collect();
        Some(Self::from_bits(&bits))
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn append_bool_in_place(mut self, value: bool) -> Self
    where
        Self: Sized,
    {
        let mut bits = self.to_bits();
        bits.push(value);
        Self::from_bits(&bits)
    }

    fn shift_with_fill(self, shift: isize, fill: bool) -> Self
    where
        Self: Sized,
    {
        if shift == 0 {
            return self;
        }

        let bits = self.to_bits();
        let shift_abs = usize::min(shift.abs() as usize, bits.len());
        let empty_bits = vec![fill; shift_abs];
        match shift.cmp(&0) {
            Ordering::Less => {
                let mut new_bits = empty_bits;
                new_bits.extend_from_slice(&bits[..bits.len() - shift_abs]);
                Self::from_bits(&new_bits)
            }
            Ordering::Greater => {
                let mut new_bits = bits[shift_abs..].to_vec();
                new_bits.extend_from_slice(&empty_bits);
                Self::from_bits(&new_bits)
            }
            // UNSAFE: This branch is unreachable because zero is handled at the top
            Ordering::Equal => unsafe { unreachable_unchecked() },
        }
    }

    fn shift(self, shift: isize) -> Self
    where
        Self: Sized,
    {
        self.shift_with_fill(shift, false)
    }
}

macro_rules! impl_index {
    ($t:ty) => {
        impl Index<usize> for $t {
            type Output = bool;

            fn index(&self, index: usize) -> &Self::Output {
                self.get(index).expect("Index out of bounds")
            }
        }

        impl IndexMut<usize> for $t {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                self.get_mut(index).expect("Index out of bounds")
            }
        }
    };
    () => {};
}

impl_index!(BoolBitArray);

#[cfg(test)]
mod tests {
    use num_bigint::{BigInt, BigUint};
    pub use rand::Rng;

    pub fn random_bits(rng: &mut impl Rng, len: usize) -> Vec<bool> {
        (0..len).map(|_| rng.random_bool(0.5)).collect()
    }

    pub fn random_bytes(rng: &mut impl Rng, len: usize) -> Vec<u8> {
        (0..len).map(|_| rng.random()).collect()
    }

    pub fn random_bits_string(rng: &mut impl Rng, len: usize) -> String {
        (0..len)
            .map(|_| if rng.random_bool(0.5) { '1' } else { '0' })
            .collect()
    }

    pub fn random_biguint(rng: &mut impl Rng, n_bits: usize) -> BigUint {
        let n_bytes = n_bits.div_ceil(8);
        let mut bytes = vec![0u8; n_bytes];
        rng.fill(&mut bytes[..]);
        let last_num_bits = n_bits % 8;
        if last_num_bits > 0 {
            bytes[n_bytes - 1] &= (1 << last_num_bits) - 1;
        }
        BigUint::from_bytes_le(&bytes)
    }

    pub fn random_bigint(rng: &mut impl Rng, n_bits: usize) -> BigInt {
        let uint = random_biguint(rng, n_bits - 1);
        if rng.random_bool(0.5) {
            BigInt::from(uint)
        } else {
            -BigInt::from(uint)
        }
    }

    pub fn string_to_bits(s: &str) -> Vec<bool> {
        s.chars().map(|c| c == '1').collect()
    }

    pub fn string_to_bytes(s: &str) -> Vec<u8> {
        let mut bytes = Vec::new();
        for bits in s.as_bytes().chunks(8) {
            let string = std::str::from_utf8(bits).unwrap();
            let mut byte = u8::from_str_radix(string, 2).unwrap();
            byte = byte.reverse_bits() >> (8 - bits.len());
            bytes.push(byte);
        }
        bytes
    }
}
