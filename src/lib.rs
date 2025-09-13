pub mod bitarray;

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use rand::{Rng, SeedableRng, rngs::StdRng};
    use rstest::fixture;

    static SEED: OnceLock<u64> = OnceLock::new();

    #[fixture]
    pub fn n_experiments() -> usize {
        10000
    }

    #[fixture]
    pub fn rng(n_experiments: usize) -> impl Rng {
        let seed = *SEED.get_or_init(|| rand::rng().random());
        println!("Seed: {} (for {} experiments)", seed, n_experiments);
        StdRng::seed_from_u64(seed)
    }
}
