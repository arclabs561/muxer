//! Boltzmann (softmax-temperature) bandit policy via drawset's Gumbel-max trick.
//!
//! At each `decide`, the policy samples one arm with probability proportional
//! to `exp(mean_reward[i] / temperature)`. Concretely, it uses the Gumbel-max
//! reparameterisation: `argmax_i (logit_i + g_i)` where `g_i ~ Gumbel(0, 1)`.
//! For `temperature -> 0` the policy is greedy; for `temperature -> infinity`
//! it is uniform.
//!
//! Compared with [`crate::ThompsonSampling`] (Beta posteriors over Bernoulli
//! rewards) and [`crate::Exp3Ix`] (importance-weighted loss updates), Boltzmann is
//! the simplest stateless-given-stats baseline. It works for any real-valued
//! reward and integrates with `drawset` for the Gumbel-noise RNG.
//!
//! Available behind the `boltzmann` feature.

use std::collections::BTreeMap;

use crate::policy::BanditPolicy;
use crate::{Decision, DecisionNote, DecisionPolicy};

/// Configuration for [`BoltzmannPolicy`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BoltzmannConfig {
    /// Softmax temperature in `(0, +inf)`. Lower = more exploitation.
    /// Defaults to `1.0` (standard softmax over raw rewards).
    pub temperature: f64,
    /// Per-arm reward to use before any updates land. Defaults to `0.0`,
    /// which produces uniform sampling for the cold-start window.
    pub initial_reward: f64,
    /// Optional clamp on the absolute value of any single reward update.
    /// `None` = no clamp. Useful when downstream rewards can spike.
    pub reward_clip: Option<f64>,
}

impl Default for BoltzmannConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            initial_reward: 0.0,
            reward_clip: None,
        }
    }
}

impl BoltzmannConfig {
    /// Validate hyperparameters. Panics on misuse.
    pub fn validate(&self) {
        assert!(
            self.temperature.is_finite() && self.temperature > 0.0,
            "BoltzmannConfig::temperature must be finite and > 0, got {}",
            self.temperature
        );
        if let Some(clip) = self.reward_clip {
            assert!(
                clip.is_finite() && clip > 0.0,
                "BoltzmannConfig::reward_clip must be finite and > 0 if set, got {}",
                clip
            );
        }
    }
}

/// Boltzmann (softmax) bandit policy.
///
/// Maintains per-arm running mean reward and samples each decision via the
/// Gumbel-max trick (delegated to [`drawset::gumbel_max_sample`]).
///
/// # Example
///
/// ```rust
/// use muxer::{BanditPolicy, BoltzmannPolicy, BoltzmannConfig};
///
/// let mut policy = BoltzmannPolicy::new(BoltzmannConfig {
///     temperature: 0.5, // sharper than default softmax
///     ..Default::default()
/// });
/// let arms = vec!["a".to_string(), "b".to_string()];
/// for _ in 0..20 {
///     policy.update_reward("a", 1.0);
///     policy.update_reward("b", 0.0);
/// }
/// // After training, decide() will heavily favor "a".
/// let _d = policy.decide(&arms);
/// ```
#[derive(Debug, Clone)]
pub struct BoltzmannPolicy {
    config: BoltzmannConfig,
    // BTreeMap<arm_name, (sum_reward, count)>. Stable iteration order =
    // reproducible logits vector ordering.
    stats: BTreeMap<String, (f64, u64)>,
}

#[cfg(feature = "serde")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BoltzmannCheckpoint {
    pub(crate) temperature: u64,
    pub(crate) initial_reward: u64,
    pub(crate) reward_clip: Option<u64>,
    pub(crate) stats: Vec<(String, u64, u64)>,
}

impl BoltzmannPolicy {
    #[cfg(feature = "serde")]
    pub(crate) fn checkpoint_state(&self) -> Result<BoltzmannCheckpoint, crate::PolicyError> {
        if !self.config.temperature.is_finite()
            || self.config.temperature <= 0.0
            || !self.config.initial_reward.is_finite()
            || self
                .config
                .reward_clip
                .is_some_and(|clip| !clip.is_finite() || clip <= 0.0)
            || self
                .stats
                .iter()
                .any(|(arm, (sum, count))| arm.is_empty() || !sum.is_finite() || *count == 0)
        {
            return Err(crate::PolicyError::new(
                "Boltzmann checkpoint state is invalid",
            ));
        }
        Ok(BoltzmannCheckpoint {
            temperature: self.config.temperature.to_bits(),
            initial_reward: self.config.initial_reward.to_bits(),
            reward_clip: self.config.reward_clip.map(f64::to_bits),
            stats: self
                .stats
                .iter()
                .map(|(arm, (sum, count))| (arm.clone(), sum.to_bits(), *count))
                .collect(),
        })
    }
    #[cfg(feature = "serde")]
    pub(crate) fn from_checkpoint_state(
        state: BoltzmannCheckpoint,
    ) -> Result<Self, crate::PolicyError> {
        let config = BoltzmannConfig {
            temperature: f64::from_bits(state.temperature),
            initial_reward: f64::from_bits(state.initial_reward),
            reward_clip: state.reward_clip.map(f64::from_bits),
        };
        if !config.temperature.is_finite()
            || config.temperature <= 0.0
            || !config.initial_reward.is_finite()
            || config
                .reward_clip
                .is_some_and(|clip| !clip.is_finite() || clip <= 0.0)
        {
            return Err(crate::PolicyError::new(
                "Boltzmann checkpoint config is invalid",
            ));
        }
        let mut stats = BTreeMap::new();
        for (arm, sum, count) in state.stats {
            let sum = f64::from_bits(sum);
            if arm.is_empty()
                || !sum.is_finite()
                || count == 0
                || stats.insert(arm, (sum, count)).is_some()
            {
                return Err(crate::PolicyError::new(
                    "Boltzmann checkpoint stats are invalid",
                ));
            }
        }
        Ok(Self { config, stats })
    }
    /// Construct with explicit config.
    pub fn new(config: BoltzmannConfig) -> Self {
        config.validate();
        Self {
            config,
            stats: BTreeMap::new(),
        }
    }

    /// Mean observed reward for `arm`, or `initial_reward` if no updates.
    pub fn mean_reward(&self, arm: &str) -> f64 {
        match self.stats.get(arm) {
            Some(&(sum, n)) if n > 0 => sum / n as f64,
            _ => self.config.initial_reward,
        }
    }

    /// Per-arm softmax probabilities for the given arm list, computed from
    /// the same logits used for sampling. Useful for debug / monitoring.
    pub fn probs(&self, arms: &[String]) -> BTreeMap<String, f64> {
        let inv_t = 1.0 / self.config.temperature;
        let logits: Vec<f64> = arms.iter().map(|a| self.mean_reward(a) * inv_t).collect();
        let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|&l| (l - max).exp()).collect();
        let z: f64 = exps.iter().sum();
        arms.iter()
            .zip(exps.iter())
            .map(|(a, &e)| (a.clone(), e / z))
            .collect()
    }
}

impl BanditPolicy for BoltzmannPolicy {
    fn decide(&mut self, arms: &[String]) -> Option<Decision> {
        if arms.is_empty() {
            return None;
        }
        let inv_t = 1.0 / self.config.temperature;
        // drawset::gumbel_max_sample takes &[f32]. Down-cast logits.
        let logits: Vec<f32> = arms
            .iter()
            .map(|a| (self.mean_reward(a) * inv_t) as f32)
            .collect();
        let idx = drawset::gumbel_max_sample(&logits);
        let chosen = arms[idx].clone();
        let probs = self.probs(arms);
        Some(Decision {
            policy: DecisionPolicy::Boltzmann,
            chosen,
            probs: Some(probs),
            notes: vec![DecisionNote::SampledFromDistribution],
        })
    }

    fn update_reward(&mut self, arm: &str, reward: f64) {
        if !reward.is_finite() {
            return;
        }
        let r = match self.config.reward_clip {
            Some(clip) => reward.clamp(-clip, clip),
            None => reward,
        };
        let entry = self.stats.entry(arm.to_string()).or_insert((0.0, 0));
        entry.0 += r;
        entry.1 += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_arms_returns_none() {
        let mut p = BoltzmannPolicy::new(BoltzmannConfig::default());
        assert!(p.decide(&[]).is_none());
    }

    #[test]
    fn non_finite_reward_is_ignored() {
        let mut policy = BoltzmannPolicy::new(BoltzmannConfig::default());
        let before = policy.mean_reward("a");
        policy.update_reward("a", f64::NAN);
        assert_eq!(policy.mean_reward("a"), before);
    }

    #[test]
    fn cold_start_roughly_uniform() {
        // With no reward history all arms have logit = initial / T = 0;
        // Gumbel noise dominates, sample is roughly uniform.
        let mut p = BoltzmannPolicy::new(BoltzmannConfig::default());
        let arms = vec!["a".into(), "b".into(), "c".into()];
        let mut counts = std::collections::HashMap::new();
        for _ in 0..600 {
            if let Some(d) = p.decide(&arms) {
                *counts.entry(d.chosen).or_insert(0u32) += 1;
            }
        }
        for arm in &arms {
            let n = counts.get(arm).copied().unwrap_or(0);
            assert!(n > 60, "arm {} only selected {} times", arm, n);
        }
    }

    #[test]
    fn high_reward_arm_dominates_at_low_temperature() {
        let mut p = BoltzmannPolicy::new(BoltzmannConfig {
            temperature: 0.1, // sharp softmax
            ..Default::default()
        });
        let arms = vec!["a".into(), "b".into()];
        for _ in 0..50 {
            p.update_reward("a", 1.0);
            p.update_reward("b", 0.0);
        }
        let mut a_wins = 0u32;
        for _ in 0..500 {
            if let Some(d) = p.decide(&arms) {
                if d.chosen == "a" {
                    a_wins += 1;
                }
            }
        }
        assert!(
            a_wins > 450,
            "expected high-reward arm to dominate, got {}/500",
            a_wins
        );
    }

    #[test]
    fn high_temperature_close_to_uniform() {
        let mut p = BoltzmannPolicy::new(BoltzmannConfig {
            temperature: 100.0,
            ..Default::default()
        });
        let arms = vec!["a".into(), "b".into()];
        for _ in 0..50 {
            p.update_reward("a", 1.0);
            p.update_reward("b", 0.0);
        }
        let mut a_wins = 0u32;
        for _ in 0..1000 {
            if let Some(d) = p.decide(&arms) {
                if d.chosen == "a" {
                    a_wins += 1;
                }
            }
        }
        assert!(
            (350..650).contains(&a_wins),
            "expected near-uniform at high T, got {}/1000",
            a_wins
        );
    }

    #[test]
    fn reward_clip_bounds_updates() {
        let mut p = BoltzmannPolicy::new(BoltzmannConfig {
            reward_clip: Some(1.0),
            ..Default::default()
        });
        p.update_reward("a", 100.0);
        assert!((p.mean_reward("a") - 1.0).abs() < 1e-12);
        p.update_reward("a", -100.0);
        assert!(p.mean_reward("a").abs() < 1e-12);
    }

    #[test]
    fn probs_sum_to_one() {
        let mut p = BoltzmannPolicy::new(BoltzmannConfig::default());
        let arms = vec!["a".into(), "b".into(), "c".into()];
        p.update_reward("a", 1.0);
        p.update_reward("b", 0.5);
        p.update_reward("c", 0.0);
        let probs = p.probs(&arms);
        let total: f64 = probs.values().sum();
        assert!((total - 1.0).abs() < 1e-9);
        // Highest-reward arm gets the largest probability.
        assert!(probs["a"] > probs["b"]);
        assert!(probs["b"] > probs["c"]);
    }

    #[test]
    fn decision_envelope_carries_probs_and_note() {
        let mut p = BoltzmannPolicy::new(BoltzmannConfig::default());
        let arms = vec!["a".into(), "b".into()];
        let d = p.decide(&arms).expect("decision");
        assert_eq!(d.policy, DecisionPolicy::Boltzmann);
        assert!(d.probs.is_some());
        assert!(d.notes.contains(&DecisionNote::SampledFromDistribution));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn checkpoint_round_trip_preserves_config_bits_and_rejects_invalid_stats() {
        let mut policy = BoltzmannPolicy::new(BoltzmannConfig {
            temperature: 0.125,
            initial_reward: -0.25,
            reward_clip: Some(2.5),
        });
        policy.update_reward("a", 1.0);
        let state = policy.checkpoint_state().unwrap();
        let restored = BoltzmannPolicy::from_checkpoint_state(state.clone()).unwrap();
        assert_eq!(restored.config.temperature.to_bits(), 0.125f64.to_bits());
        assert_eq!(
            restored.config.initial_reward.to_bits(),
            (-0.25f64).to_bits()
        );
        assert_eq!(
            restored.config.reward_clip.map(f64::to_bits),
            Some(2.5f64.to_bits())
        );
        assert_eq!(restored.stats, policy.stats);

        let mut zero_count = state.clone();
        zero_count.stats[0].2 = 0;
        assert!(BoltzmannPolicy::from_checkpoint_state(zero_count).is_err());

        let mut invalid_temperature = state;
        invalid_temperature.temperature = f64::NAN.to_bits();
        assert!(BoltzmannPolicy::from_checkpoint_state(invalid_temperature).is_err());
    }

    #[test]
    #[should_panic(expected = "temperature must be finite and > 0")]
    fn zero_temperature_rejected() {
        let _ = BoltzmannPolicy::new(BoltzmannConfig {
            temperature: 0.0,
            ..Default::default()
        });
    }

    #[test]
    #[should_panic(expected = "temperature must be finite and > 0")]
    fn negative_temperature_rejected() {
        let _ = BoltzmannPolicy::new(BoltzmannConfig {
            temperature: -1.0,
            ..Default::default()
        });
    }
}
