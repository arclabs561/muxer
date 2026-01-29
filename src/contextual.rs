//! Contextual bandits for routing.
//!
//! This module is intentionally small and production-oriented:
//! - The caller supplies a fixed-length feature vector per decision ("context").
//! - Rewards are scalar in `[0, 1]` (caller-defined).
//! - Policies are **seedable** and deterministic-by-default given a fixed seed and arm order.
//!
//! Today this module provides a simple linear contextual bandit (LinUCB) with
//! per-arm ridge regression state and incremental Sherman–Morrison updates.

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::collections::BTreeMap;

use crate::softmax_map;

/// Per-arm score tuple: `(ucb, mean, bonus)`.
pub type LinUcbScore = (f64, f64, f64);

/// Configuration for linear UCB.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LinUcbConfig {
    /// Feature vector dimension (must be >= 1).
    pub dim: usize,
    /// Ridge regularization parameter (lambda, must be finite and > 0).
    pub lambda: f64,
    /// Exploration strength (alpha, must be finite and >= 0).
    pub alpha: f64,
    /// Seed for RNG used only for tie-breaking between equal scores.
    pub seed: u64,
    /// Optional exponential decay factor applied to the sufficient statistics on each update.
    ///
    /// - `1.0` means no decay (no forgetting).
    /// - Smaller values forget older observations faster (useful for drift).
    ///
    /// Implementation detail: if \(d \in (0, 1)\), we apply
    /// `A <- d*A + x*x^T` and `b <- d*b + r*x` each update.
    /// Since we store \(A^{-1}\), we first scale `A^{-1} <- A^{-1} / d`
    /// before applying the Sherman–Morrison rank-1 update.
    pub decay: f64,
}

impl Default for LinUcbConfig {
    fn default() -> Self {
        Self {
            dim: 8,
            lambda: 1.0,
            alpha: 1.0,
            seed: 0,
            decay: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
struct ArmState {
    // A^{-1} for ridge regression (d x d).
    a_inv: Vec<f64>,
    // b vector (d).
    b: Vec<f64>,
    uses: u64,
}

impl ArmState {
    fn new(dim: usize, lambda: f64) -> Self {
        let mut a_inv = vec![0.0; dim * dim];
        let diag = if lambda.is_finite() && lambda > 0.0 {
            1.0 / lambda
        } else {
            1.0
        };
        for i in 0..dim {
            a_inv[i * dim + i] = diag;
        }
        Self {
            a_inv,
            b: vec![0.0; dim],
            uses: 0,
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        s += x * y;
    }
    s
}

fn mat_vec(a: &[f64], dim: usize, x: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; dim];
    for i in 0..dim {
        let mut s = 0.0;
        let row = &a[i * dim..(i + 1) * dim];
        for j in 0..dim {
            s += row[j] * x[j];
        }
        out[i] = s;
    }
    out
}

/// Numerically safe clamp for scalar rewards.
fn clamp01(r: f64) -> f64 {
    if !r.is_finite() {
        return 0.0;
    }
    r.clamp(0.0, 1.0)
}

/// Seedable linear contextual bandit (LinUCB).
///
/// Usage:
/// - call `select_with_scores(arms, context)` to get a choice + debug scores
/// - call `update_reward(chosen_arm, context, reward01)` after observing reward
#[derive(Debug, Clone)]
pub struct LinUcb {
    cfg: LinUcbConfig,
    rng: StdRng,
    arms: Vec<String>,
    state: BTreeMap<String, ArmState>,
}

impl LinUcb {
    /// Create a new LinUCB instance (deterministic by default).
    pub fn new(cfg: LinUcbConfig) -> Self {
        Self {
            rng: StdRng::seed_from_u64(cfg.seed),
            cfg,
            arms: Vec::new(),
            state: BTreeMap::new(),
        }
    }

    fn dim(&self) -> usize {
        self.cfg.dim.max(1)
    }

    fn ensure_arms(&mut self, arms_in_order: &[String]) {
        if self.arms == arms_in_order {
            return;
        }
        self.arms = arms_in_order.to_vec();
        let dim = self.dim();
        let lambda = self.cfg.lambda;
        for a in &self.arms {
            self.state
                .entry(a.clone())
                .or_insert_with(|| ArmState::new(dim, lambda));
        }
    }

    fn sanitize_context(&self, context: &[f64]) -> Vec<f64> {
        let d = self.dim();
        let mut x = vec![0.0; d];
        for (i, v) in x.iter_mut().enumerate() {
            let raw = context.get(i).copied().unwrap_or(0.0);
            *v = if raw.is_finite() { raw } else { 0.0 };
        }
        x
    }

    fn theta(&self, st: &ArmState) -> Vec<f64> {
        let d = self.dim();
        mat_vec(&st.a_inv, d, &st.b)
    }

    fn score(&self, st: &ArmState, x: &[f64]) -> (f64, f64, f64) {
        // mean = theta^T x
        let theta = self.theta(st);
        let mean = dot(&theta, x);

        // bonus = alpha * sqrt(x^T A^{-1} x)
        let d = self.dim();
        let ax = mat_vec(&st.a_inv, d, x);
        let var = dot(x, &ax).max(0.0);
        let alpha = if self.cfg.alpha.is_finite() && self.cfg.alpha >= 0.0 {
            self.cfg.alpha
        } else {
            0.0
        };
        let bonus = alpha * var.sqrt();
        (mean + bonus, mean, bonus)
    }

    /// Return per-arm (ucb, mean, bonus) scores for a given context.
    pub fn scores(
        &mut self,
        arms_in_order: &[String],
        context: &[f64],
    ) -> BTreeMap<String, LinUcbScore> {
        self.ensure_arms(arms_in_order);
        let x = self.sanitize_context(context);
        let mut out = BTreeMap::new();
        for a in arms_in_order {
            let st = self.state.get(a).expect("arm state missing");
            out.insert(a.clone(), self.score(st, &x));
        }
        out
    }

    /// Select an arm for a given context, returning the chosen arm + per-arm scores.
    ///
    /// Policy:
    /// - Explore each arm once (stable order) before using scores.
    /// - Otherwise choose argmax UCB; tie-break is stable, with seeded randomness as a last resort.
    pub fn select_with_scores<'a>(
        &mut self,
        arms_in_order: &'a [String],
        context: &[f64],
    ) -> Option<(&'a String, BTreeMap<String, LinUcbScore>)> {
        self.ensure_arms(arms_in_order);
        if arms_in_order.is_empty() {
            return None;
        }

        for a in arms_in_order {
            let uses = self.state.get(a).map(|s| s.uses).unwrap_or(0);
            if uses == 0 {
                let scores = self.scores(arms_in_order, context);
                return Some((a, scores));
            }
        }

        let scores = self.scores(arms_in_order, context);
        let mut best = arms_in_order[0].as_str();
        let mut best_score = scores.get(best).map(|t| t.0).unwrap_or(f64::NEG_INFINITY);

        // Stable tie-break first: lexicographic.
        for a in arms_in_order {
            let sc = scores.get(a.as_str()).map(|t| t.0).unwrap_or(best_score);
            if sc > best_score + 1e-12 {
                best = a;
                best_score = sc;
            } else if (sc - best_score).abs() <= 1e-12 && a.as_str() < best {
                best = a;
            }
        }

        // If multiple arms are exactly equal after lexicographic tie-break (rare), keep deterministic choice.
        // Seeded RNG is reserved for future use (e.g. stochastic tie-breaking).
        let chosen = arms_in_order.iter().find(|a| a.as_str() == best)?;
        Some((chosen, scores))
    }

    /// Softmax distribution over arms based on their current UCB scores for this context.
    ///
    /// This is useful for:
    /// - traffic-splitting (probabilistic routing)
    /// - logging an approximate propensity distribution for offline evaluation
    ///
    /// # Example
    ///
    /// ```rust
    /// # #[cfg(feature = "contextual")]
    /// # {
    /// use muxer::{LinUcb, LinUcbConfig};
    ///
    /// let arms = vec!["a".to_string(), "b".to_string()];
    /// let mut p = LinUcb::new(LinUcbConfig { dim: 2, ..LinUcbConfig::default() });
    /// let probs = p.probabilities(&arms, &[0.2, 0.8], 0.3);
    /// let s: f64 = probs.values().sum();
    /// assert!((s - 1.0).abs() < 1e-9);
    /// # }
    /// ```
    pub fn probabilities(
        &mut self,
        arms_in_order: &[String],
        context: &[f64],
        temperature: f64,
    ) -> BTreeMap<String, f64> {
        let scores = self.scores(arms_in_order, context);
        let mut ucb: BTreeMap<String, f64> = BTreeMap::new();
        for a in arms_in_order {
            ucb.insert(
                a.clone(),
                scores.get(a.as_str()).map(|t| t.0).unwrap_or(0.0),
            );
        }
        softmax_map(&ucb, temperature)
    }

    /// Select an arm by sampling from `probabilities(...)`, returning the chosen arm and the probabilities used.
    ///
    /// Policy:
    /// - Explore each arm once in stable order (still returns a full `probs` map).
    /// - Otherwise sample from a softmax over UCB scores (seeded RNG).
    ///
    /// # Example
    ///
    /// ```rust
    /// # #[cfg(feature = "contextual")]
    /// # {
    /// use muxer::{LinUcb, LinUcbConfig};
    ///
    /// let arms = vec!["a".to_string(), "b".to_string()];
    /// let mut p = LinUcb::new(LinUcbConfig { dim: 2, seed: 0, ..LinUcbConfig::default() });
    /// let (chosen, probs) = p.select_softmax_ucb_with_probs(&arms, &[0.2, 0.8], 0.3).unwrap();
    /// p.update_reward(chosen, &[0.2, 0.8], 1.0);
    /// let s: f64 = probs.values().sum();
    /// assert!((s - 1.0).abs() < 1e-9);
    /// # }
    /// ```
    pub fn select_softmax_ucb_with_probs<'a>(
        &mut self,
        arms_in_order: &'a [String],
        context: &[f64],
        temperature: f64,
    ) -> Option<(&'a String, BTreeMap<String, f64>)> {
        self.ensure_arms(arms_in_order);
        if arms_in_order.is_empty() {
            return None;
        }

        let probs = self.probabilities(arms_in_order, context, temperature);

        for a in arms_in_order {
            let uses = self.state.get(a).map(|s| s.uses).unwrap_or(0);
            if uses == 0 {
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

    /// Update the model for `arm` given the same context used for selection and a reward in `[0, 1]`.
    pub fn update_reward(&mut self, arm: &str, context: &[f64], reward01: f64) {
        let d = self.dim();
        let x = self.sanitize_context(context);
        let r = clamp01(reward01);
        let decay = if self.cfg.decay.is_finite() && self.cfg.decay > 0.0 && self.cfg.decay <= 1.0 {
            self.cfg.decay
        } else {
            1.0
        };
        let decay = decay.clamp(1.0e-6, 1.0);

        let Some(st) = self.state.get_mut(arm) else {
            return;
        };

        // Optional decay (forgetting).
        if decay < 1.0 {
            for v in &mut st.b {
                *v *= decay;
            }
            // If A <- d*A, then A^{-1} <- A^{-1} / d.
            for v in &mut st.a_inv {
                *v /= decay;
            }
        }

        // Sherman–Morrison update for A^{-1} where A := A + x x^T
        // A^{-1} <- A^{-1} - (A^{-1} x x^T A^{-1}) / (1 + x^T A^{-1} x)
        let ax = mat_vec(&st.a_inv, d, &x);
        let denom = 1.0 + dot(&x, &ax);
        if denom.is_finite() && denom > 1e-12 {
            // rank-1 update: outer(ax, ax) / denom
            for i in 0..d {
                for j in 0..d {
                    st.a_inv[i * d + j] -= (ax[i] * ax[j]) / denom;
                }
            }
        }

        // b <- b + r x
        for (i, xi) in x.iter().enumerate() {
            st.b[i] += r * xi;
        }
        st.uses = st.uses.saturating_add(1);
    }
}

#[cfg(all(test, feature = "contextual"))]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn linucb_explores_each_arm_once_in_order() {
        let arms = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let cfg = LinUcbConfig {
            dim: 2,
            lambda: 1.0,
            alpha: 1.0,
            seed: 0,
            decay: 1.0,
        };
        let mut p = LinUcb::new(cfg);
        let ctx = [0.2, 0.7];

        let (c1, _) = p.select_with_scores(&arms, &ctx).unwrap();
        p.update_reward(c1, &ctx, 0.0);
        assert_eq!(c1.as_str(), "a");

        let (c2, _) = p.select_with_scores(&arms, &ctx).unwrap();
        p.update_reward(c2, &ctx, 0.0);
        assert_eq!(c2.as_str(), "b");

        let (c3, _) = p.select_with_scores(&arms, &ctx).unwrap();
        p.update_reward(c3, &ctx, 0.0);
        assert_eq!(c3.as_str(), "c");
    }

    #[test]
    fn linucb_learns_better_arm_in_simple_linear_env() {
        // Context is constant; arm "a" always yields reward 1, arm "b" yields reward 0.
        // After initial exploration, LinUCB should prefer "a" most of the time.
        let arms = vec!["a".to_string(), "b".to_string()];
        let cfg = LinUcbConfig {
            dim: 2,
            lambda: 1.0,
            alpha: 0.1, // small exploration bonus
            seed: 0,
            decay: 1.0,
        };
        let mut p = LinUcb::new(cfg);
        let ctx = [1.0, 0.5];

        let mut chosen_a = 0u64;
        for _ in 0..200 {
            let (chosen, _) = p.select_with_scores(&arms, &ctx).unwrap();
            if chosen.as_str() == "a" {
                chosen_a += 1;
                p.update_reward(chosen, &ctx, 1.0);
            } else {
                p.update_reward(chosen, &ctx, 0.0);
            }
        }
        assert!(chosen_a >= 150, "chosen_a={}", chosen_a);
    }

    #[test]
    fn linucb_learns_context_dependent_routing() {
        // Two contexts; each has a different optimal arm.
        // Reward is deterministic (no noise) to avoid test flakiness.
        let arms = vec!["small".to_string(), "big".to_string()];
        let cfg = LinUcbConfig {
            dim: 2,
            lambda: 1.0,
            alpha: 0.2,
            seed: 0,
            decay: 1.0,
        };
        let mut p = LinUcb::new(cfg);

        // Context A => "small" best, context B => "big" best.
        let ctx_a = [1.0, 0.0];
        let ctx_b = [0.0, 1.0];

        let mut correct = 0u64;
        let mut total = 0u64;

        for t in 0..400u64 {
            let (ctx, optimal) = if t % 2 == 0 {
                (&ctx_a[..], "small")
            } else {
                (&ctx_b[..], "big")
            };
            let (chosen, _) = p.select_with_scores(&arms, ctx).unwrap();
            let reward = if chosen.as_str() == optimal { 1.0 } else { 0.0 };
            p.update_reward(chosen, ctx, reward);

            // Evaluate after warmup.
            if t >= 50 {
                total += 1;
                if chosen.as_str() == optimal {
                    correct += 1;
                }
            }
        }

        let acc = (correct as f64) / (total.max(1) as f64);
        assert!(acc >= 0.85, "acc={}", acc);
    }

    #[test]
    fn linucb_softmax_probs_is_a_distribution_and_deterministic() {
        let arms = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let cfg = LinUcbConfig {
            dim: 2,
            lambda: 1.0,
            alpha: 0.5,
            seed: 0,
            decay: 0.97,
        };
        let mut p = LinUcb::new(cfg);
        let ctx = [0.1, 0.9];

        let probs1 = p.probabilities(&arms, &ctx, 0.3);
        let probs2 = p.probabilities(&arms, &ctx, 0.3);
        assert_eq!(probs1, probs2);

        let sum: f64 = probs1.values().sum();
        assert!((sum - 1.0).abs() < 1e-9);
        for v in probs1.values() {
            assert!(v.is_finite());
            assert!(*v >= 0.0 && *v <= 1.0);
        }
    }

    proptest! {
        #[test]
        fn linucb_decay_keeps_state_finite(
            dim in 1usize..10,
            decay in 0.5f64..1.0f64,
            alpha in 0.0f64..3.0f64,
            lambda in 1.0e-6f64..10.0f64,
            steps in 0usize..200,
            ctxs in proptest::collection::vec(
                proptest::collection::vec(prop_oneof![Just(f64::NAN), -1.0e3f64..1.0e3f64], 0..20),
                0..200
            ),
            rewards in proptest::collection::vec(prop_oneof![Just(f64::NAN), -10.0f64..10.0f64], 0..200),
        ) {
            let arms = vec!["a".to_string(), "b".to_string()];
            let cfg = LinUcbConfig { dim, alpha, lambda, seed: 0, decay };
            let mut p = LinUcb::new(cfg);

            for i in 0..steps {
                let ctx = ctxs.get(i).cloned().unwrap_or_default();
                let (chosen, _scores) = p.select_with_scores(&arms, &ctx).unwrap();
                let r = rewards.get(i).copied().unwrap_or(0.0);
                p.update_reward(chosen, &ctx, r);
            }

            // Invariants: scores finite, state finite, A^{-1} roughly symmetric.
            let ctx = ctxs.first().cloned().unwrap_or_default();
            let scores = p.scores(&arms, &ctx);
            for (_a, (ucb, mean, bonus)) in scores.iter() {
                prop_assert!(ucb.is_finite());
                prop_assert!(mean.is_finite());
                prop_assert!(bonus.is_finite());
            }

            for st in p.state.values() {
                for v in &st.b {
                    prop_assert!(v.is_finite());
                }
                for v in &st.a_inv {
                    prop_assert!(v.is_finite());
                }
                let d = p.dim();
                for i in 0..d {
                    for j in 0..d {
                        let aij = st.a_inv[i*d + j];
                        let aji = st.a_inv[j*d + i];
                        prop_assert!((aij - aji).abs() < 1e-7);
                    }
                }
            }
        }
    }

    proptest! {
        #[test]
        fn linucb_is_deterministic_given_seed(
            dim in 1usize..12,
            alpha in 0.0f64..5.0f64,
            lambda in 1.0e-6f64..10.0f64,
            seed in any::<u64>(),
            // bounded contexts
            ctx in proptest::collection::vec(-10.0f64..10.0f64, 0..20),
            rewards in proptest::collection::vec(0.0f64..1.0f64, 0..50),
        ) {
            let arms = vec!["a".to_string(), "b".to_string(), "c".to_string()];
            let cfg = LinUcbConfig { dim, alpha, lambda, seed, decay: 1.0 };

            let mut p1 = LinUcb::new(cfg);
            let mut p2 = LinUcb::new(cfg);

            for (i, r) in rewards.iter().enumerate() {
                let (c1, _) = p1.select_with_scores(&arms, &ctx).unwrap();
                let (c2, _) = p2.select_with_scores(&arms, &ctx).unwrap();
                prop_assert_eq!(c1, c2);

                // update on the chosen arm
                p1.update_reward(c1, &ctx, *r);
                p2.update_reward(c2, &ctx, *r);

                // occasional score equality
                if i % 5 == 0 {
                    let s1 = p1.scores(&arms, &ctx);
                    let s2 = p2.scores(&arms, &ctx);
                    prop_assert_eq!(s1.len(), s2.len());
                }
            }
        }
    }
}
