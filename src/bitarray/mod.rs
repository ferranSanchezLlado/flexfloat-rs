use std::ops::{Index, IndexMut};

pub mod boolean_list;

pub use boolean_list::BoolBitArray;

/// LE byte order
pub trait BitArray {
    fn from_bytes(bits: &[u8], n_bits: usize) -> Self
    where
        Self: Sized;
    fn from_bits(bits: &[bool]) -> Self
    where
        Self: Sized,
    {
        let n_bits = bits.len();
        let bytes = (n_bits + 7) / 8;
        let mut byte_vec = vec![0u8; bytes];

        for (i, &bit) in bits.iter().enumerate() {
            if bit {
                let byte_index = i / 8;
                let bit_index = i % 8;
                byte_vec[byte_index] |= 1 << bit_index;
            }
        }

        Self::from_bytes(&byte_vec, n_bits)
    }
    fn from_float(value: f64) -> Self
    where
        Self: Sized,
    {
        let bits = value.to_bits();
        Self::from_bytes(&bits.to_le_bytes(), 64)
    }

    fn zeros(n_bits: usize) -> Self
    where
        Self: Sized,
    {
        let bytes = (n_bits + 7) / 8;
        let bits = vec![0u8; bytes];

        Self::from_bytes(&bits, n_bits)
    }
    fn ones(n_bits: usize) -> Self
    where
        Self: Sized,
    {
        let bytes = (n_bits + 7) / 8;
        let mut bits = vec![0xffu8; bytes];

        let last_num_bits = n_bits % 8;
        if last_num_bits > 0 {
            bits[bytes - 1] = (1 << last_num_bits) - 1;
        }

        Self::from_bytes(&bits, n_bits)
    }

    fn to_bytes(&self) -> Vec<u8>;
    fn to_bits(&self) -> Vec<bool> {
        let n_bits = self.len();
        let mut bool_vec = Vec::with_capacity(n_bits);

        for i in 0..n_bits {
            if let Some(bit) = self.get(i) {
                bool_vec.push(*bit);
            }
        }

        bool_vec
    }
    fn to_float(&self) -> Option<f64> {
        if self.len() != 64 {
            return None;
        }
        let bytes: [u8; 8] = self.to_bytes().try_into().ok()?;
        Some(f64::from_bits(u64::from_le_bytes(bytes)))
    }

    fn len(&self) -> usize;
    // TODO: https://doc.rust-lang.org/std/slice/trait.SliceIndex.html
    fn get(&self, index: usize) -> Option<&bool>;
    fn get_mut(&mut self, index: usize) -> Option<&mut bool>;
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
    pub use crate::tests::*;
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
