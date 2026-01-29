//! EXP3-IX (adversarial bandit) for arm selection.
//!
//! This policy is useful when rewards can be adversarial / highly non-stationary.
//! It is **seedable** so it can be reproducible in tests. Like other policies
//! in this crate, default construction is deterministic by default.

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::collections::BTreeMap;

/// Configuration for EXP3-IX.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Exp3IxConfig {
    /// Pseudo time horizon used to set learning rate.
    pub horizon: usize,
    /// Optional confidence parameter \(delta \in (0, 1)\) used to set learning rate.
    pub confidence_delta: Option<f64>,
    /// Seed for the internal RNG (used only after initial exploration).
    pub seed: u64,
    /// Exponential decay factor in `(0, 1]` applied to accumulated losses each update.
    ///
    /// - `1.0` means no decay (no forgetting).
    /// - Smaller values forget older losses faster (useful for non-stationarity).
    pub decay: f64,
}

impl Default for Exp3IxConfig {
    fn default() -> Self {
        Self {
            horizon: 1_000,
            confidence_delta: None,
            seed: 0,
            decay: 1.0,
        }
    }
}

/// Seedable EXP3-IX bandit.
#[derive(Debug, Clone)]
pub struct Exp3Ix {
    cfg: Exp3IxConfig,
    gamma: f64,
    learning_rate: f64,
    rng: StdRng,

    // Per-arm state (aligned to `arms_in_order` indices).
    arms: Vec<String>,
    uses: Vec<u64>,
    cum_loss_hat: Vec<f64>,
    probs: Vec<f64>,
}

impl Exp3Ix {
    /// Create a new EXP3-IX instance with deterministic defaults.
    pub fn new(cfg: Exp3IxConfig) -> Self {
        Self::with_seed(cfg, cfg.seed)
    }

    /// Create with an explicit seed.
    pub fn with_seed(mut cfg: Exp3IxConfig, seed: u64) -> Self {
        cfg.seed = seed;
        Self {
            cfg,
            gamma: 0.0,
            learning_rate: 0.0,
            rng: StdRng::seed_from_u64(seed),
            arms: Vec::new(),
            uses: Vec::new(),
            cum_loss_hat: Vec::new(),
            probs: Vec::new(),
        }
    }

    fn ensure_arms(&mut self, arms_in_order: &[String]) {
        if self.arms == arms_in_order {
            return;
        }
        self.arms = arms_in_order.to_vec();
        let k = self.arms.len().max(1);
        self.uses = vec![0; k];
        self.cum_loss_hat = vec![0.0; k];
        self.probs = vec![1.0 / (k as f64); k];

        // Set learning rate + implicit exploration like the reference implementation.
        let kf = k as f64;
        let horizon = (self.cfg.horizon.max(1)) as f64;
        let nk = kf * horizon;
        let lr = match self.cfg.confidence_delta {
            Some(delta) if delta.is_finite() && delta > 0.0 && delta < 1.0 => {
                ((kf.ln() + ((kf + 1.0) / delta).ln()) / nk).sqrt()
            }
            _ => (2.0 * (kf + 1.0).ln() / nk).sqrt(),
        };
        self.learning_rate = lr;
        self.gamma = 0.5 * lr;
    }

    fn recompute_probs(&mut self) {
        if self.arms.is_empty() {
            self.probs.clear();
            return;
        }
        // weights_i = exp(-eta * Lhat_i), normalized
        let eta = self.learning_rate;
        let min_l = self
            .cum_loss_hat
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let mut w: Vec<f64> = Vec::with_capacity(self.cum_loss_hat.len());
        let mut denom = 0.0;
        for &l in &self.cum_loss_hat {
            let x = (-eta * (l - min_l)).exp();
            denom += x;
            w.push(x);
        }
        if denom <= 0.0 || !denom.is_finite() {
            let k = self.cum_loss_hat.len() as f64;
            self.probs = vec![1.0 / k; self.cum_loss_hat.len()];
            return;
        }
        for (i, wi) in w.into_iter().enumerate() {
            self.probs[i] = wi / denom;
        }
    }

    /// Current selection probabilities (aligned to `arms_in_order`).
    pub fn probabilities(&mut self, arms_in_order: &[String]) -> BTreeMap<String, f64> {
        self.ensure_arms(arms_in_order);
        let mut out = BTreeMap::new();
        for (i, a) in self.arms.iter().enumerate() {
            out.insert(a.clone(), self.probs.get(i).copied().unwrap_or(0.0));
        }
        out
    }

    /// Select an arm and return the probabilities used for selection.
    pub fn select_with_probs<'a>(
        &mut self,
        arms_in_order: &'a [String],
    ) -> Option<(&'a String, BTreeMap<String, f64>)> {
        self.ensure_arms(arms_in_order);
        if self.arms.is_empty() {
            return None;
        }

        // Always return probabilities as of this decision.
        let probs = self.probabilities(arms_in_order);

        // Explore first.
        for (i, a) in arms_in_order.iter().enumerate() {
            if self.uses.get(i).copied().unwrap_or(0) == 0 {
                return Some((a, probs));
            }
        }

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

    /// Select an arm.
    ///
    /// Policy:
    /// - Explore each arm once in stable order.
    /// - Otherwise sample from the current EXP3-IX distribution (seeded RNG).
    pub fn select<'a>(&mut self, arms_in_order: &'a [String]) -> Option<&'a String> {
        self.ensure_arms(arms_in_order);
        if self.arms.is_empty() {
            return None;
        }

        for (i, a) in arms_in_order.iter().enumerate() {
            if self.uses.get(i).copied().unwrap_or(0) == 0 {
                return Some(a);
            }
        }

        let r: f64 = self.rng.random();
        let mut cdf = 0.0;
        for (i, p) in self.probs.iter().enumerate() {
            cdf += *p;
            if r < cdf {
                return arms_in_order.get(i);
            }
        }
        // Numerical fallback.
        arms_in_order.last()
    }

    /// Update EXP3-IX with a bounded reward in `[0, 1]`.
    pub fn update_reward(&mut self, arm: &str, reward01: f64) {
        if self.arms.is_empty() {
            // No-op if not initialized.
            return;
        }
        let Some(idx) = self.arms.iter().position(|a| a == arm) else {
            return;
        };
        let decay = if self.cfg.decay.is_finite() && self.cfg.decay > 0.0 && self.cfg.decay <= 1.0 {
            self.cfg.decay
        } else {
            1.0
        };
        if decay < 1.0 {
            for x in &mut self.cum_loss_hat {
                *x *= decay;
            }
        }
        let r = reward01.clamp(0.0, 1.0);
        let loss = 1.0 - r;
        let p = self.probs.get(idx).copied().unwrap_or(0.0);
        let denom = p + self.gamma;
        let loss_hat = if denom > 0.0 { loss / denom } else { loss };

        self.cum_loss_hat[idx] += loss_hat;
        self.uses[idx] = self.uses[idx].saturating_add(1);
        self.recompute_probs();
    }
}

impl Default for Exp3Ix {
    fn default() -> Self {
        Self::new(Exp3IxConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn explores_each_arm_once_in_order() {
        let mut ex = Exp3Ix::with_seed(Exp3IxConfig::default(), 123);
        let arms = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        assert_eq!(ex.select(&arms).unwrap(), "a");
        ex.update_reward("a", 1.0);
        assert_eq!(ex.select(&arms).unwrap(), "b");
        ex.update_reward("b", 1.0);
        assert_eq!(ex.select(&arms).unwrap(), "c");
    }

    #[test]
    fn probabilities_sum_to_one() {
        let mut ex = Exp3Ix::default();
        let arms = vec!["a".to_string(), "b".to_string()];
        let p = ex.probabilities(&arms);
        let s: f64 = p.values().sum();
        assert!((s - 1.0).abs() < 1e-9, "sum={}", s);
    }

    #[test]
    fn deterministic_given_same_seed_and_updates() {
        let cfg = Exp3IxConfig {
            horizon: 100,
            confidence_delta: None,
            seed: 7,
            decay: 1.0,
        };
        let arms = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mut e1 = Exp3Ix::new(cfg);
        let mut e2 = Exp3Ix::new(cfg);

        // Initialize and match state.
        e1.select(&arms);
        e2.select(&arms);
        e1.update_reward("a", 0.2);
        e2.update_reward("a", 0.2);
        e1.update_reward("b", 0.9);
        e2.update_reward("b", 0.9);

        // After exploration, RNG-driven choice should match with same seed.
        // Consume exploration for both.
        for _ in 0..3 {
            let a1 = e1.select(&arms).unwrap().clone();
            let a2 = e2.select(&arms).unwrap().clone();
            assert_eq!(a1, a2);
            e1.update_reward(&a1, 0.5);
            e2.update_reward(&a2, 0.5);
        }
    }

    proptest! {
        #[test]
        fn exp3ix_probs_are_well_formed_and_choice_is_member(
            seed in any::<u64>(),
            horizon in 1usize..5000,
            decay in 0.01f64..1.0f64,
            steps in 0usize..200,
            // reward stream (bounded)
            rewards in proptest::collection::vec(0.0f64..1.0f64, 0..200),
        ) {
            let cfg = Exp3IxConfig {
                seed,
                horizon,
                confidence_delta: None,
                decay,
            };
            let mut ex = Exp3Ix::new(cfg);
            let arms = vec!["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string()];

            let mut i = 0usize;
            while i < steps {
                let (chosen, probs) = ex.select_with_probs(&arms).unwrap();
                // choice is a member
                prop_assert!(arms.iter().any(|a| a == chosen));

                // probs is a distribution
                let s: f64 = probs.values().sum();
                prop_assert!((s - 1.0).abs() < 1e-9, "sum={}", s);
                for v in probs.values() {
                    prop_assert!(v.is_finite());
                    prop_assert!(*v >= 0.0 && *v <= 1.0);
                }

                let r = rewards.get(i).copied().unwrap_or(0.5);
                ex.update_reward(chosen, r);
                i += 1;
            }
        }

        #[test]
        fn exp3ix_is_deterministic_with_seed_for_select_with_probs(
            seed in any::<u64>(),
            decay in 0.1f64..1.0f64,
            rewards in proptest::collection::vec(0.0f64..1.0f64, 0..100),
        ) {
            let cfg = Exp3IxConfig { seed, horizon: 1000, confidence_delta: None, decay };
            let arms = vec!["a".to_string(), "b".to_string(), "c".to_string()];
            let mut e1 = Exp3Ix::new(cfg);
            let mut e2 = Exp3Ix::new(cfg);

            for (i, r) in rewards.iter().enumerate() {
                let (c1, p1) = e1.select_with_probs(&arms).unwrap();
                let (c2, p2) = e2.select_with_probs(&arms).unwrap();
                prop_assert_eq!(c1, c2, "step={}", i);
                prop_assert_eq!(p1, p2, "step={}", i);
                e1.update_reward(c1, *r);
                e2.update_reward(c2, *r);
            }
        }
    }
}
