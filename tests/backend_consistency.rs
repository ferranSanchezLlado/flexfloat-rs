//! Property-based tests verifying that operations on UsizeBitArray
//! produce results identical to BoolBitArray.

use flexfloat::bitarray::usize_list::UsizeBitArray;
use flexfloat::prelude::*;
use num_bigint::BigUint;
use proptest::prelude::*;

fn arb_bits(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<bool>> {
    prop::collection::vec(any::<bool>(), min_len..=max_len)
}

fn trim(mut v: Vec<bool>) -> Vec<bool> {
    while matches!(v.last(), Some(false)) {
        v.pop();
    }
    v
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        ..ProptestConfig::default()
    })]

    #[test]
    fn add_consistent(a_bits in arb_bits(1, 128), b_bits in arb_bits(1, 128)) {
        let a_bool = BoolBitArray::from_bits(&a_bits);
        let b_bool = BoolBitArray::from_bits(&b_bits);
        let a_usize = UsizeBitArray::from_bits(&a_bits);
        let b_usize = UsizeBitArray::from_bits(&b_bits);

        let result_bool = (a_bool + b_bool).to_bits();
        let result_usize = (a_usize + b_usize).to_bits();
        prop_assert_eq!(trim(result_bool), trim(result_usize));
    }

    #[test]
    fn sub_consistent(a_bits in arb_bits(1, 128), b_bits in arb_bits(1, 128)) {
        let a_value = BoolBitArray::from_bits(&a_bits).to_biguint();
        let b_value = BoolBitArray::from_bits(&b_bits).to_biguint();
        let (a_safe, b_safe): (BigUint, BigUint) = if a_value >= b_value {
            (a_value, b_value)
        } else {
            (b_value, a_value)
        };

        let a_safe = BoolBitArray::from_biguint(&a_safe).to_bits();
        let b_padded = BoolBitArray::from_biguint(&b_safe).to_bits();

        let a_bool = BoolBitArray::from_bits(&a_safe);
        let b_bool = BoolBitArray::from_bits(&b_padded);
        let a_usize = UsizeBitArray::from_bits(&a_safe);
        let b_usize = UsizeBitArray::from_bits(&b_padded);

        let result_bool = (a_bool - b_bool).to_bits();
        let result_usize = (a_usize - b_usize).to_bits();
        prop_assert_eq!(trim(result_bool), trim(result_usize));
    }

    #[test]
    fn mul_consistent(a_bits in arb_bits(1, 64), b_bits in arb_bits(1, 64)) {
        let a_bool = BoolBitArray::from_bits(&a_bits);
        let b_bool = BoolBitArray::from_bits(&b_bits);
        let a_usize = UsizeBitArray::from_bits(&a_bits);
        let b_usize = UsizeBitArray::from_bits(&b_bits);

        let result_bool = (a_bool * b_bool).to_bits();
        let result_usize = (a_usize * b_usize).to_bits();
        prop_assert_eq!(trim(result_bool), trim(result_usize));
    }

    #[test]
    fn div_consistent(a_bits in arb_bits(1, 64), b_bits in arb_bits(1, 64)) {
        prop_assume!(b_bits.iter().any(|&b| b));

        let a_bool = BoolBitArray::from_bits(&a_bits);
        let b_bool = BoolBitArray::from_bits(&b_bits);
        let a_usize = UsizeBitArray::from_bits(&a_bits);
        let b_usize = UsizeBitArray::from_bits(&b_bits);

        let result_bool = (a_bool / b_bool).to_bits();
        let result_usize = (a_usize / b_usize).to_bits();
        prop_assert_eq!(trim(result_bool), trim(result_usize));
    }

    #[test]
    fn arithmetic_length_contract(a_bits in arb_bits(1, 128), b_bits in arb_bits(1, 128)) {
        let a_bool = BoolBitArray::from_bits(&a_bits);
        let b_bool = BoolBitArray::from_bits(&b_bits);
        let a_uint = a_bool.to_biguint();
        let b_uint = b_bool.to_biguint();

        let add_result = BoolBitArray::from_bits(&a_bits) + BoolBitArray::from_bits(&b_bits);
        let add_expected_bits = (&a_uint + &b_uint).bits() as usize;
        prop_assert!(add_result.len() >= add_expected_bits);

        let mul_result = BoolBitArray::from_bits(&a_bits) * BoolBitArray::from_bits(&b_bits);
        let mul_expected_bits = (&a_uint * &b_uint).bits() as usize;
        prop_assert!(mul_result.len() >= mul_expected_bits);

        prop_assume!(b_uint != BigUint::from(0u8));
        let div_result = BoolBitArray::from_bits(&a_bits) / BoolBitArray::from_bits(&b_bits);
        let div_expected_bits = (&a_uint / &b_uint).bits() as usize;
        prop_assert!(div_result.len() >= div_expected_bits);

        if a_uint >= b_uint {
            let sub_result = BoolBitArray::from_bits(&a_bits) - BoolBitArray::from_bits(&b_bits);
            let sub_expected_bits = (&a_uint - &b_uint).bits() as usize;
            prop_assert!(sub_result.len() >= sub_expected_bits);
        }
    }
}
