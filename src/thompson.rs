//! Thompson sampling bandit for arm selection.
//!
//! This is useful when you have a scalar reward in `[0, 1]` per call and want an
//! adaptive policy that balances exploration and exploitation.
//!
//! Notes:
//! - This policy is **seedable** so selection can be reproducible in tests.
//! - Default construction uses a fixed seed (deterministic by default).

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Beta, Distribution};
use std::collections::BTreeMap;

use crate::alloc::softmax_map;

/// Configuration for Thompson sampling.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ThompsonConfig {
    /// Prior alpha (must be > 0).
    pub alpha0: f64,
    /// Prior beta (must be > 0).
    pub beta0: f64,
    /// Optional per-arm priors (alpha, beta). If present, overrides (alpha0, beta0).
    pub priors: BTreeMap<String, (f64, f64)>,
}

impl Default for ThompsonConfig {
    fn default() -> Self {
        Self {
            alpha0: 1.0,
            beta0: 1.0,
            priors: BTreeMap::new(),
        }
    }
}

/// Beta posterior state for one arm.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BetaStats {
    pub alpha: f64,
    pub beta: f64,
    pub uses: u64,
}

impl BetaStats {
    pub fn expected_value(&self) -> f64 {
        let denom = self.alpha + self.beta;
        if denom <= 0.0 {
            0.5
        } else {
            self.alpha / denom
        }
    }
}

/// Seedable Thompson-sampling bandit.
#[derive(Debug, Clone)]
pub struct ThompsonSampling {
    cfg: ThompsonConfig,
    stats: BTreeMap<String, BetaStats>,
    rng: StdRng,
}

impl ThompsonSampling {
    /// Create a Thompson-sampling bandit with a deterministic fixed seed (0).
    pub fn new(cfg: ThompsonConfig) -> Self {
        Self::with_seed(cfg, 0)
    }

    /// Create a Thompson-sampling bandit with a fixed seed (reproducible).
    pub fn with_seed(cfg: ThompsonConfig, seed: u64) -> Self {
        Self {
            cfg,
            stats: BTreeMap::new(),
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Access the per-arm Beta stats.
    pub fn stats(&self) -> &BTreeMap<String, BetaStats> {
        &self.stats
    }

    /// Deterministic allocation over arms based on posterior means (softmax).
    ///
    /// This is often useful for traffic-splitting systems that want probabilities
    /// rather than an argmax choice.
    pub fn allocation_mean_softmax(
        &mut self,
        arms_in_order: &[String],
        temperature: f64,
    ) -> BTreeMap<String, f64> {
        for a in arms_in_order {
            self.get_or_create_stats(a);
        }
        let mut scores: BTreeMap<String, f64> = BTreeMap::new();
        for a in arms_in_order {
            let s = self.stats.get(a).copied().unwrap_or(BetaStats {
                alpha: self.cfg.alpha0,
                beta: self.cfg.beta0,
                uses: 0,
            });
            scores.insert(a.clone(), s.expected_value());
        }
        softmax_map(&scores, temperature)
    }

    /// Reset all learned state.
    pub fn reset(&mut self) {
        self.stats.clear();
    }

    fn prior_for(&self, arm: &str) -> (f64, f64) {
        if let Some(&(a, b)) = self.cfg.priors.get(arm) {
            (a, b)
        } else {
            (self.cfg.alpha0, self.cfg.beta0)
        }
    }

    fn get_or_create_stats(&mut self, arm: &str) -> &mut BetaStats {
        // Avoid borrowing `self` inside the `entry` closure (borrowck footgun).
        let (a, b) = self.prior_for(arm);
        self.stats
            .entry(arm.to_string())
            .or_insert_with(|| BetaStats {
                alpha: if a.is_finite() && a > 0.0 { a } else { 1.0 },
                beta: if b.is_finite() && b > 0.0 { b } else { 1.0 },
                uses: 0,
            })
    }

    fn sample_beta(&mut self, alpha: f64, beta: f64) -> f64 {
        if !(alpha.is_finite() && beta.is_finite()) || alpha <= 0.0 || beta <= 0.0 {
            return 0.5;
        }
        match Beta::new(alpha, beta) {
            Ok(dist) => dist.sample(&mut self.rng),
            Err(_) => 0.5,
        }
    }

    /// Select an arm.
    ///
    /// Policy:
    /// - Explore: return the first arm (stable order) that has `uses == 0`.
    /// - Otherwise: sample from each arm’s Beta posterior and choose the max.
    /// - Tie-break: lexicographic arm name.
    pub fn select<'a>(&mut self, arms_in_order: &'a [String]) -> Option<&'a String> {
        for a in arms_in_order {
            let s = *self.get_or_create_stats(a);
            if s.uses == 0 {
                return Some(a);
            }
        }

        let mut best: Option<&'a String> = None;
        let mut best_sample = f64::NEG_INFINITY;
        for a in arms_in_order {
            let s = *self.get_or_create_stats(a);
            let x = self.sample_beta(s.alpha, s.beta);
            if x > best_sample
                || ((x - best_sample).abs() <= 1e-12 && best.map(|b| a < b).unwrap_or(true))
            {
                best_sample = x;
                best = Some(a);
            }
        }
        best
    }

    /// Update the chosen arm with a bounded reward in `[0, 1]`.
    ///
    /// Interprets reward as a fractional “success”:
    /// - `alpha += reward`
    /// - `beta += 1 - reward`
    pub fn update_reward(&mut self, arm: &str, reward01: f64) {
        let r = reward01.clamp(0.0, 1.0);
        let s = self.get_or_create_stats(arm);
        s.alpha += r;
        s.beta += 1.0 - r;
        s.uses = s.uses.saturating_add(1);
    }
}

impl Default for ThompsonSampling {
    fn default() -> Self {
        Self::new(ThompsonConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explores_each_arm_once_in_order() {
        let mut ts = ThompsonSampling::with_seed(ThompsonConfig::default(), 123);
        let arms = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        assert_eq!(ts.select(&arms).unwrap(), "a");
        ts.update_reward("a", 1.0);
        assert_eq!(ts.select(&arms).unwrap(), "b");
        ts.update_reward("b", 1.0);
        assert_eq!(ts.select(&arms).unwrap(), "c");
    }

    #[test]
    fn deterministic_first_choice_given_same_seed_and_state() {
        let cfg = ThompsonConfig::default();
        let arms = vec!["a".to_string(), "b".to_string()];
        let mut t1 = ThompsonSampling::with_seed(cfg.clone(), 42);
        let mut t2 = ThompsonSampling::with_seed(cfg, 42);

        // Put both into the same state.
        t1.update_reward("a", 1.0);
        t1.update_reward("b", 0.0);
        t2.update_reward("a", 1.0);
        t2.update_reward("b", 0.0);

        assert_eq!(t1.select(&arms), t2.select(&arms));
    }

    #[test]
    fn update_reward_moves_expected_value() {
        let mut ts = ThompsonSampling::default();
        let arms = vec!["a".to_string()];
        // Ensure stats exist.
        ts.select(&arms);
        let before = ts.stats().get("a").unwrap().expected_value();
        for _ in 0..10 {
            ts.update_reward("a", 1.0);
        }
        let after = ts.stats().get("a").unwrap().expected_value();
        assert!(after > before);
    }
}
