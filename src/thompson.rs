//! Thompson sampling bandit for arm selection.
//!
//! This is useful when you have a scalar reward in `[0, 1]` per call and want an
//! adaptive policy that balances exploration and exploitation.
//!
//! Notes:
//! - This policy is **seedable** so selection can be reproducible in tests.
//! - Default construction uses a fixed seed (deterministic by default).

use rand::rngs::StdRng;
use rand::Rng;
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
    /// Exponential decay factor in `(0, 1]` applied on each update.
    ///
    /// - `1.0` means no decay (no forgetting).
    /// - Smaller values forget older observations faster (useful for non-stationarity).
    pub decay: f64,
    /// Optional per-arm priors (alpha, beta). If present, overrides (alpha0, beta0).
    pub priors: BTreeMap<String, (f64, f64)>,
}

impl Default for ThompsonConfig {
    fn default() -> Self {
        Self {
            alpha0: 1.0,
            beta0: 1.0,
            decay: 1.0,
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

    /// Select an arm by sampling from the mean-based softmax allocation, returning
    /// both the chosen arm and the allocation used.
    ///
    /// This is a pragmatic “traffic splitting” selector:
    /// - The returned map is a probability distribution over arms.
    /// - The chosen arm is sampled from that distribution (seedable RNG).
    /// - It still explores each arm once in stable order.
    ///
    /// # Example
    ///
    /// ```rust
    /// use muxer::{ThompsonConfig, ThompsonSampling};
    ///
    /// let arms = vec!["a".to_string(), "b".to_string()];
    /// let mut ts = ThompsonSampling::with_seed(ThompsonConfig { decay: 0.99, ..ThompsonConfig::default() }, 0);
    /// let (chosen, probs) = ts.select_softmax_mean_with_probs(&arms, 0.5).unwrap();
    /// ts.update_reward(chosen, 1.0);
    /// let s: f64 = probs.values().sum();
    /// assert!((s - 1.0).abs() < 1e-9);
    /// ```
    pub fn select_softmax_mean_with_probs<'a>(
        &mut self,
        arms_in_order: &'a [String],
        temperature: f64,
    ) -> Option<(&'a String, BTreeMap<String, f64>)> {
        if arms_in_order.is_empty() {
            return None;
        }

        // Explore first (stable).
        for a in arms_in_order {
            let s = *self.get_or_create_stats(a);
            if s.uses == 0 {
                let probs = self.allocation_mean_softmax(arms_in_order, temperature);
                return Some((a, probs));
            }
        }

        let probs = self.allocation_mean_softmax(arms_in_order, temperature);
        // Sample from the returned distribution in `arms_in_order` order.
        let r: f64 = self.rng.random();
        let mut cdf = 0.0;
        for a in arms_in_order {
            cdf += probs.get(a).copied().unwrap_or(0.0);
            if r < cdf {
                return Some((a, probs));
            }
        }
        Some((arms_in_order.last().unwrap(), probs))
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
        let (a0, b0) = self.prior_for(arm);
        let decay = if self.cfg.decay.is_finite() && self.cfg.decay > 0.0 && self.cfg.decay <= 1.0 {
            self.cfg.decay
        } else {
            1.0
        };
        let s = self.get_or_create_stats(arm);
        // Decay toward prior, then add this observation.
        s.alpha = a0 + decay * (s.alpha - a0) + r;
        s.beta = b0 + decay * (s.beta - b0) + (1.0 - r);
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
    use proptest::prelude::*;

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

    proptest! {
        #[test]
        fn thompson_allocation_is_a_distribution_and_deterministic(
            seed in any::<u64>(),
            decay in 0.01f64..1.0f64,
            temperature in prop_oneof![Just(f64::NAN), Just(0.0), Just(-1.0), 1.0e-6f64..1.0e6f64],
            rewards in proptest::collection::vec(0.0f64..1.0f64, 0..100),
        ) {
            let cfg = ThompsonConfig { decay, ..ThompsonConfig::default() };
            let arms = vec!["a".to_string(), "b".to_string(), "c".to_string()];

            let mut t1 = ThompsonSampling::with_seed(cfg.clone(), seed);
            let mut t2 = ThompsonSampling::with_seed(cfg, seed);

            for (i, r) in rewards.iter().enumerate() {
                let (c1, p1) = t1.select_softmax_mean_with_probs(&arms, temperature).unwrap();
                let (c2, p2) = t2.select_softmax_mean_with_probs(&arms, temperature).unwrap();
                prop_assert_eq!(c1, c2, "step={}", i);
                prop_assert_eq!(p1.clone(), p2, "step={}", i);

                let s: f64 = p1.values().sum();
                prop_assert!((s - 1.0).abs() < 1e-9, "sum={}", s);
                for v in p1.values() {
                    prop_assert!(v.is_finite());
                    prop_assert!(*v >= 0.0 && *v <= 1.0);
                }

                t1.update_reward(c1, *r);
                t2.update_reward(c2, *r);
            }
        }

        #[test]
        fn thompson_decay_stays_finite_under_many_updates(
            seed in any::<u64>(),
            decay in 0.01f64..1.0f64,
            steps in 0usize..500,
            rewards in proptest::collection::vec(0.0f64..1.0f64, 0..500),
        ) {
            let cfg = ThompsonConfig { decay, ..ThompsonConfig::default() };
            let mut ts = ThompsonSampling::with_seed(cfg, seed);
            let arms = vec!["a".to_string(), "b".to_string()];

            for i in 0..steps {
                let a = ts.select(&arms).unwrap().clone();
                let r = rewards.get(i).copied().unwrap_or(0.5);
                ts.update_reward(&a, r);
            }

            for s in ts.stats().values() {
                prop_assert!(s.alpha.is_finite());
                prop_assert!(s.beta.is_finite());
                prop_assert!(s.alpha > 0.0);
                prop_assert!(s.beta > 0.0);
            }
        }
    }
}
