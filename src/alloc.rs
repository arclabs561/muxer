//! Allocation helpers (probabilities, softmax).
//!
//! These are deterministic utilities for turning scores into a probability
//! distribution in a stable (reproducible) way.

use std::collections::BTreeMap;

/// Compute a stable softmax distribution over a map of scores.
///
/// - `temperature` controls sharpness (must be finite and > 0).
/// - Uses the standard max-trick for numerical stability.
/// - Returns a distribution that sums to 1 (or empty if input is empty).
pub fn softmax_map(scores: &BTreeMap<String, f64>, temperature: f64) -> BTreeMap<String, f64> {
    if scores.is_empty() {
        return BTreeMap::new();
    }
    let t = if temperature.is_finite() && temperature > 0.0 {
        temperature
    } else {
        1.0
    };

    let max_score = scores.values().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut out: BTreeMap<String, f64> = BTreeMap::new();
    let mut denom = 0.0;
    for (k, &v) in scores.iter() {
        let x = ((v - max_score) / t).exp();
        denom += x;
        out.insert(k.clone(), x);
    }
    if denom <= 0.0 || !denom.is_finite() {
        // Degenerate fallback: uniform.
        let n = scores.len() as f64;
        return scores
            .keys()
            .map(|k| (k.clone(), 1.0 / n))
            .collect::<BTreeMap<_, _>>();
    }

    for v in out.values_mut() {
        *v /= denom;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one() {
        let mut m = BTreeMap::new();
        m.insert("a".to_string(), 0.0);
        m.insert("b".to_string(), 1.0);
        m.insert("c".to_string(), -2.0);
        let p = softmax_map(&m, 1.0);
        let s: f64 = p.values().sum();
        assert!((s - 1.0).abs() < 1e-9, "sum={}", s);
    }
}
