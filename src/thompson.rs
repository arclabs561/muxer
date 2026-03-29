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
use crate::{Decision, DecisionNote, DecisionPolicy};

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
    /// Alpha parameter of the Beta posterior.
    pub alpha: f64,
    /// Beta parameter of the Beta posterior.
    pub beta: f64,
    /// Number of reward updates applied to this arm.
    pub uses: u64,
}

impl BetaStats {
    /// Posterior mean: `alpha / (alpha + beta)`.
    pub fn expected_value(&self) -> f64 {
        let denom = self.alpha + self.beta;
        if denom <= 0.0 {
            0.5
        } else {
            self.alpha / denom
        }
    }

    /// Posterior variance: `ab / ((a+b)^2 (a+b+1))`.
    ///
    /// This quantifies how uncertain the posterior is about this arm's success rate.
    /// Useful as a convergence diagnostic (shrinks toward zero as observations accumulate)
    /// and as a staleness signal (if it stops shrinking under decay, the environment
    /// may be non-stationary).
    ///
    /// Note: under adaptive sampling, this is a Bayesian credible-interval width, not
    /// a frequentist confidence interval. The selected arm's posterior mean is
    /// systematically optimistic due to selection bias (Shin, Ramdas & Rinaldo 2019).
    pub fn variance(&self) -> f64 {
        let ab = self.alpha + self.beta;
        if ab <= 0.0 || !ab.is_finite() {
            return 0.25; // maximal variance for uniform
        }
        let v = (self.alpha * self.beta) / (ab * ab * (ab + 1.0));
        if v.is_finite() {
            v
        } else {
            0.25
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

    /// Per-arm probability of being the best arm, estimated by Monte Carlo.
    ///
    /// Draws `n_samples` from each arm's Beta posterior and counts how often each
    /// arm's sample is the highest. Returns a distribution over arms that sums to 1.
    ///
    /// This is the standard A/B testing "probability of being best" metric
    /// (Garivier & Kaufmann 2016). It answers "should I commit to this arm?" --
    /// when `P(best) > 0.95` for some arm, further exploration has diminishing returns.
    ///
    /// **RNG caveat**: this method draws `n_samples * K` from the internal RNG,
    /// advancing its state. Calling `probability_best` between `select()` calls
    /// changes the subsequent selection sequence. If you need stable selection
    /// determinism, call this only at diagnostic checkpoints, not in the hot path.
    ///
    /// Note: under adaptive sampling, the posteriors themselves carry selection bias
    /// (Shin et al 2019), so these probabilities are calibrated only approximately.
    /// They are diagnostic, not frequentist guarantees.
    ///
    /// # Example
    ///
    /// ```rust
    /// use muxer::{ThompsonConfig, ThompsonSampling};
    ///
    /// let arms = vec!["a".to_string(), "b".to_string()];
    /// let mut ts = ThompsonSampling::with_seed(ThompsonConfig::default(), 42);
    /// for _ in 0..50 {
    ///     ts.update_reward("a", 0.9);
    ///     ts.update_reward("b", 0.3);
    /// }
    /// let pbb = ts.probability_best(&arms, 10_000);
    /// assert!(pbb["a"] > 0.9);
    /// ```
    pub fn probability_best(
        &mut self,
        arms_in_order: &[String],
        n_samples: u32,
    ) -> BTreeMap<String, f64> {
        let n = n_samples.max(1) as usize;
        for a in arms_in_order {
            self.get_or_create_stats(a);
        }

        // Collect per-arm (alpha, beta) for sampling.
        let params: Vec<(f64, f64)> = arms_in_order
            .iter()
            .map(|a| {
                let s = self.stats.get(a).copied().unwrap_or(BetaStats {
                    alpha: self.cfg.alpha0,
                    beta: self.cfg.beta0,
                    uses: 0,
                });
                (s.alpha, s.beta)
            })
            .collect();

        let k = params.len();
        if k == 0 {
            return BTreeMap::new();
        }

        let mut wins = vec![0u64; k];
        for _ in 0..n {
            let mut best_val = f64::NEG_INFINITY;
            let mut best_idx = 0;
            for (i, &(a, b)) in params.iter().enumerate() {
                let sample = self.sample_beta(a, b);
                // Strict improvement only; ties go to the first arm (lower index).
                if sample > best_val + 1e-15 {
                    best_val = sample;
                    best_idx = i;
                }
            }
            wins[best_idx] += 1;
        }

        let mut out = BTreeMap::new();
        let nf = n as f64;
        for (i, a) in arms_in_order.iter().enumerate() {
            out.insert(a.clone(), wins[i] as f64 / nf);
        }
        out
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
        let d = self.decide_softmax_mean(arms_in_order, temperature)?;
        let chosen = arms_in_order
            .iter()
            .find(|a| a.as_str() == d.chosen.as_str())?;
        Some((chosen, d.probs.unwrap_or_default()))
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

    /// Select via mean-softmax sampling and return a unified `Decision`.
    ///
    /// Notes:
    /// - Always includes `probs` (the mean-softmax allocation).
    /// - Records explore-first vs sampling and numerical fallback.
    pub fn decide_softmax_mean(
        &mut self,
        arms_in_order: &[String],
        temperature: f64,
    ) -> Option<Decision> {
        if arms_in_order.is_empty() {
            return None;
        }

        // Ensure stats exist before we compute probs (so probs map contains all arms).
        for a in arms_in_order {
            self.get_or_create_stats(a);
        }
        let probs = self.allocation_mean_softmax(arms_in_order, temperature);

        // Explore first (stable order).
        for a in arms_in_order {
            let s = *self.get_or_create_stats(a);
            if s.uses == 0 {
                return Some(Decision {
                    policy: DecisionPolicy::Thompson,
                    chosen: a.clone(),
                    probs: Some(probs),
                    notes: vec![DecisionNote::ExploreFirst],
                });
            }
        }

        // Sample from the returned distribution in `arms_in_order` order.
        let r: f64 = self.rng.random();
        let mut cdf = 0.0;
        for a in arms_in_order {
            cdf += probs.get(a).copied().unwrap_or(0.0);
            if r < cdf {
                return Some(Decision {
                    policy: DecisionPolicy::Thompson,
                    chosen: a.clone(),
                    probs: Some(probs),
                    notes: vec![DecisionNote::SampledFromDistribution],
                });
            }
        }

        let last = arms_in_order.last()?.clone();
        Some(Decision {
            policy: DecisionPolicy::Thompson,
            chosen: last,
            probs: Some(probs),
            notes: vec![
                DecisionNote::SampledFromDistribution,
                DecisionNote::NumericalFallbackToLastArm,
            ],
        })
    }

    /// Select via Thompson posterior sampling and return a unified `Decision`.
    ///
    /// Notes:
    /// - Does not include `probs` (this method samples per-arm posteriors and chooses the max).
    /// - Records explore-first vs posterior sampling.
    pub fn decide(&mut self, arms_in_order: &[String]) -> Option<Decision> {
        if arms_in_order.is_empty() {
            return None;
        }

        for a in arms_in_order {
            let s = *self.get_or_create_stats(a);
            if s.uses == 0 {
                return Some(Decision {
                    policy: DecisionPolicy::Thompson,
                    chosen: a.clone(),
                    probs: None,
                    notes: vec![DecisionNote::ExploreFirst],
                });
            }
        }

        let chosen = self.select(arms_in_order)?.clone();
        Some(Decision {
            policy: DecisionPolicy::Thompson,
            chosen,
            probs: None,
            notes: vec![DecisionNote::SampledPosteriorMax],
        })
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

/// Serializable Thompson-sampling state snapshot (for persistence across process restarts).
///
/// Mirrors `Exp3IxState` and `LinUcbState` in design: intentionally excludes RNG state
/// (callers manage seeds externally) and stores only the sufficient statistics needed to
/// resume learning.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ThompsonState {
    /// Per-arm Beta posterior parameters.
    pub arms: BTreeMap<String, BetaStats>,
}

impl ThompsonSampling {
    /// Capture a persistence snapshot of the current Thompson-sampling state.
    ///
    /// This includes per-arm Beta parameters (alpha, beta, uses) so that a caller
    /// can serialize, store, and later restore the policy state across process restarts.
    /// RNG state is excluded; callers manage seeds externally.
    pub fn snapshot(&self) -> ThompsonState {
        ThompsonState {
            arms: self.stats.clone(),
        }
    }

    /// Restore a previously snapshotted Thompson-sampling state.
    ///
    /// Arms not present in the snapshot are initialized fresh on next use.
    /// Arms in the snapshot with non-finite or non-positive parameters are skipped.
    pub fn restore(&mut self, st: ThompsonState) {
        for (name, bs) in st.arms {
            if bs.alpha.is_finite() && bs.alpha > 0.0 && bs.beta.is_finite() && bs.beta > 0.0 {
                self.stats.insert(name, bs);
            }
        }
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
    fn snapshot_restore_round_trip() {
        let cfg = ThompsonConfig {
            decay: 0.98,
            ..ThompsonConfig::default()
        };
        let arms = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mut ts = ThompsonSampling::with_seed(cfg.clone(), 42);

        // Train for a while.
        for _ in 0..30 {
            let chosen = ts.select(&arms).unwrap().clone();
            let r = if chosen == "a" { 0.9 } else { 0.2 };
            ts.update_reward(&chosen, r);
        }

        let snap = ts.snapshot();
        assert_eq!(snap.arms.len(), 3);

        // Restore into a fresh instance (same seed for RNG consistency).
        let mut ts2 = ThompsonSampling::with_seed(cfg, 42);
        ts2.restore(snap);

        // Beta stats must match.
        for a in &arms {
            let s1 = ts.stats().get(a).unwrap();
            let s2 = ts2.stats().get(a).unwrap();
            assert!(
                (s1.alpha - s2.alpha).abs() < 1e-12,
                "alpha mismatch for {a}"
            );
            assert!((s1.beta - s2.beta).abs() < 1e-12, "beta mismatch for {a}");
            assert_eq!(s1.uses, s2.uses);
        }

        // Mean-softmax allocation must match.
        let p1 = ts.allocation_mean_softmax(&arms, 0.3);
        let p2 = ts2.allocation_mean_softmax(&arms, 0.3);
        for a in &arms {
            assert!((p1[a] - p2[a]).abs() < 1e-12, "allocation mismatch for {a}");
        }
    }

    #[test]
    fn snapshot_restore_skips_corrupted_arms() {
        let cfg = ThompsonConfig::default();
        let mut ts = ThompsonSampling::with_seed(cfg, 0);

        // Restore with one bad arm (negative alpha).
        let mut bad_state = ThompsonState {
            arms: BTreeMap::new(),
        };
        bad_state.arms.insert(
            "good".to_string(),
            BetaStats {
                alpha: 5.0,
                beta: 3.0,
                uses: 8,
            },
        );
        bad_state.arms.insert(
            "bad".to_string(),
            BetaStats {
                alpha: -1.0,
                beta: 3.0,
                uses: 4,
            },
        );
        ts.restore(bad_state);

        assert!(ts.stats().contains_key("good"));
        assert!(!ts.stats().contains_key("bad"));
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

    #[test]
    fn variance_known_values() {
        // Beta(1,1) = Uniform: variance = 1/(4*3) = 1/12
        let uniform = BetaStats {
            alpha: 1.0,
            beta: 1.0,
            uses: 0,
        };
        assert!((uniform.variance() - 1.0 / 12.0).abs() < 1e-12);

        // Beta(10, 10): var = 100 / (400 * 21) = 100/8400
        let sym = BetaStats {
            alpha: 10.0,
            beta: 10.0,
            uses: 20,
        };
        assert!((sym.variance() - 100.0 / 8400.0).abs() < 1e-12);

        // Variance shrinks with more evidence.
        let tight = BetaStats {
            alpha: 100.0,
            beta: 100.0,
            uses: 200,
        };
        assert!(tight.variance() < sym.variance());
    }

    #[test]
    fn probability_best_concentrates_on_better_arm() {
        let arms = vec!["a".to_string(), "b".to_string()];
        let mut ts = ThompsonSampling::with_seed(ThompsonConfig::default(), 42);
        // Train arm "a" with high reward, "b" with low.
        for _ in 0..100 {
            ts.update_reward("a", 0.9);
            ts.update_reward("b", 0.2);
        }
        let pbb = ts.probability_best(&arms, 10_000);
        assert!(pbb["a"] > 0.95, "pbb[a]={}", pbb["a"]);
        assert!(pbb["b"] < 0.05, "pbb[b]={}", pbb["b"]);
        // Sums to 1.
        let s: f64 = pbb.values().sum();
        assert!((s - 1.0).abs() < 1e-9);
    }

    #[test]
    fn probability_best_uniform_prior_is_roughly_equal() {
        let arms = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mut ts = ThompsonSampling::with_seed(ThompsonConfig::default(), 0);
        // No updates -- all arms have uniform prior Beta(1,1).
        let pbb = ts.probability_best(&arms, 30_000);
        for a in &arms {
            // Each arm should win roughly 1/3 of the time.
            assert!(
                (pbb[a] - 1.0 / 3.0).abs() < 0.05,
                "pbb[{}]={}, expected ~0.333",
                a,
                pbb[a]
            );
        }
    }

    #[test]
    fn probability_best_deterministic_with_same_seed() {
        let arms = vec!["a".to_string(), "b".to_string()];
        let mut t1 = ThompsonSampling::with_seed(ThompsonConfig::default(), 99);
        let mut t2 = ThompsonSampling::with_seed(ThompsonConfig::default(), 99);
        for _ in 0..20 {
            t1.update_reward("a", 0.7);
            t1.update_reward("b", 0.4);
            t2.update_reward("a", 0.7);
            t2.update_reward("b", 0.4);
        }
        let p1 = t1.probability_best(&arms, 5_000);
        let p2 = t2.probability_best(&arms, 5_000);
        assert_eq!(p1, p2);
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
                // Variance must be finite and in [0, 0.25].
                let v = s.variance();
                prop_assert!(v.is_finite(), "variance not finite: {}", v);
                prop_assert!((0.0..=0.25).contains(&v), "variance out of [0, 0.25]: {}", v);
            }
        }

        #[test]
        fn probability_best_is_a_distribution(
            seed in any::<u64>(),
            decay in 0.01f64..1.0f64,
            rewards in proptest::collection::vec(0.0f64..1.0f64, 0..50),
        ) {
            let cfg = ThompsonConfig { decay, ..ThompsonConfig::default() };
            let mut ts = ThompsonSampling::with_seed(cfg, seed);
            let arms = vec!["a".to_string(), "b".to_string(), "c".to_string()];

            for (i, r) in rewards.iter().enumerate() {
                let a = &arms[i % arms.len()];
                ts.update_reward(a, *r);
            }

            let pbb = ts.probability_best(&arms, 1_000);
            let s: f64 = pbb.values().sum();
            prop_assert!((s - 1.0).abs() < 1e-9, "pbb sum={}", s);
            for v in pbb.values() {
                prop_assert!(v.is_finite());
                prop_assert!(*v >= 0.0 && *v <= 1.0);
            }
        }
    }
}
