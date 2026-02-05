use rand::Rng;

/// Normalize an array into a probability simplex.
///
/// - Negative / non-finite entries are clamped to `0.0`.
/// - If the total mass is `<= 0.0`, this returns a uniform distribution.
pub fn normalize<const CATS: usize>(mut p: [f64; CATS]) -> [f64; CATS] {
    for x in &mut p {
        if !x.is_finite() || *x < 0.0 {
            *x = 0.0;
        }
    }
    let s: f64 = p.iter().sum();
    if s <= 0.0 {
        let u = 1.0 / (CATS as f64).max(1.0);
        return [u; CATS];
    }
    for x in &mut p {
        *x /= s;
    }
    p
}

/// Sample a categorical index according to `p`.
///
/// This is robust to small floating-point error: if the CDF undershoots `1.0`, it returns the last
/// index.
pub fn sample_cat<R: Rng, const CATS: usize>(rng: &mut R, p: [f64; CATS]) -> usize {
    let r: f64 = rng.random();
    let mut cdf = 0.0;
    for (i, &pi) in p.iter().enumerate() {
        cdf += pi;
        if r < cdf {
            return i;
        }
    }
    CATS.saturating_sub(1)
}
