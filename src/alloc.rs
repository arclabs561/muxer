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
    use proptest::prelude::*;

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

    proptest! {
        #[test]
        fn softmax_map_is_a_distribution(
            // Keep bounded: small maps, bounded magnitudes.
            kvs in proptest::collection::vec(("[a-z]{1,8}", -1.0e6f64..1.0e6f64), 0..20),
            temperature in prop_oneof![Just(f64::NAN), Just(0.0), Just(-1.0), 1.0e-6f64..1.0e6f64],
        ) {
            let mut m: BTreeMap<String, f64> = BTreeMap::new();
            for (k, v) in kvs {
                m.insert(k, v);
            }
            let p = softmax_map(&m, temperature);

            // Deterministic.
            let p2 = softmax_map(&m, temperature);
            prop_assert_eq!(&p, &p2);

            if m.is_empty() {
                prop_assert!(p.is_empty());
            } else {
                // Same key set.
                prop_assert_eq!(p.len(), m.len());
                for k in m.keys() {
                    prop_assert!(p.contains_key(k));
                }

                // Valid distribution.
                let sum: f64 = p.values().sum();
                prop_assert!((sum - 1.0).abs() < 1e-9, "sum={}", sum);
                for &v in p.values() {
                    prop_assert!(v.is_finite());
                    prop_assert!(v >= 0.0);
                    prop_assert!(v <= 1.0);
                }
            }
        }
    }
}
