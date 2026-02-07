//! Contextual bandits for routing.
//!
//! This module enables the **contextual regime** where the three routing objectives
//! (exploitation, estimation, detection) genuinely diverge.
//!
//! ## Why context matters: the design measure perspective
//!
//! In the non-contextual regime, the policy's design measure is just an allocation
//! vector `(n_1, ..., n_K)` -- observation counts per arm.  All three objectives
//! (regret, estimation, detection) are monotone in these counts, and their sensitivity
//! functions are colinear under static schedules.  The Pareto front collapses.
//!
//! With context features, the design measure becomes a **function** `p_a(x, t)` --
//! the probability of pulling arm `a` at context `x` at time `t`.  Each objective
//! depends on a different projection of this function:
//!
//! - **Regret** depends on the gap-weighted marginal (concentrate near decision boundaries).
//! - **Estimation** depends on the leverage function (spread to extremes, D-optimal).
//! - **Detection** depends on uniform coverage (spread uniformly -- changes could be anywhere).
//!
//! These are linearly independent sensitivity functions in the static-design limit.
//! For adaptive policies, the feasible set of design measures is constrained by the
//! information structure (causality: action at t depends on filtration at t-1), so
//! the achievable Pareto front may be a strict subset of the unconstrained one.
//!
//! **Local vs. global caveat**: the gradient test (sensitivity-function rank) provides
//! **local** redundancy information at a specific design point.  The global Pareto
//! front can have varying dimension -- lower-dimensional patches where objectives
//! happen to align may coexist with full-dimensional regions where they diverge
//! (Zhen et al. 2018 call this "partial redundancy").  In practice, `LinUcb`
//! traverses the design space adaptively, so it encounters both regimes.
//!
//! Still, the core insight holds: context creates genuine degrees of freedom that
//! the non-contextual regime lacks.
//!
//! ## Empirical gradient rank
//!
//! The gradient test can be computed empirically by extracting the per-arm theta
//! vectors from LinUCB and checking their matrix rank.  When feature variation is
//! insufficient (e.g. all evaluations use the same language/domain), the theta
//! vectors collapse to rank 1 (proportional) even with 8-dimensional features --
//! the non-contextual collapse reasserts itself because the design measure was
//! never forced to vary along the feature dimensions.  Genuine contextual
//! divergence requires **training data that spans multiple feature regimes** (e.g.
//! both biomedical and social media text, both English and multilingual datasets).
//! This is a practical prerequisite, not a theoretical one: the degrees of freedom
//! exist in the feature space but must be *exercised* by the data generating process.
//!
//! ## Connection to Information-Directed Sampling
//!
//! The two-objective (regret, information-gain) tradeoff is already formalized by
//! **Information-Directed Sampling** (IDS), which minimizes the information ratio
//! Gamma_t = Delta_t^2 / g_t per round.  `LinUcb` can be viewed as addressing a
//! related but distinct problem: it navigates a **three**-objective landscape
//! (regret, estimation, detection) where the detection objective is not reducible
//! to the information-gain direction.  Whether a "monitoring-augmented information
//! ratio" can extend IDS to this setting is an open question; the objective manifold
//! framework suggests the extension is non-trivial only in the contextual case
//! (where the detection sensitivity function is linearly independent of the
//! information-gain direction).
//!
//! ## Practical implications
//!
//! - **Without context**: `select_mab` / `Exp3Ix` are sufficient; monitoring comes
//!   approximately "for free" with any reasonable exploration schedule.
//! - **With context**: `LinUcb` enables transfer learning across conditions (e.g.
//!   "biomedical text prefers backend X") without maintaining separate per-facet
//!   histories.  Monitoring may require extra budget beyond what regret-optimal
//!   sampling provides -- this is the price of breaking the non-contextual collapse.
//!
//! ## Design
//!
//! - The caller supplies a fixed-length feature vector per decision ("context").
//! - Rewards are scalar in `[0, 1]` (caller-defined).
//! - Policies are **seedable** and deterministic-by-default given a fixed seed and arm order.
//!
//! Implementation: linear contextual bandit (LinUCB) with per-arm ridge regression
//! state and incremental Sherman-Morrison updates.

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::collections::BTreeMap;

use crate::softmax_map;
use crate::{Decision, DecisionNote, DecisionPolicy};

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

    /// Select (argmax UCB) and return a unified `Decision`.
    ///
    /// Notes:
    /// - Does not include `probs` (this method is deterministic argmax).
    /// - Records explore-first vs deterministic choice.
    ///
    /// Explore-first detection: `select_with_scores` already returns an unseen arm
    /// (uses == 0) in stable order before using UCB scores.  We re-read the state
    /// here to tag the decision correctly; this is intentionally a read-after-select
    /// (not a second decision), so the tag always agrees with the choice.
    pub fn decide(&mut self, arms_in_order: &[String], context: &[f64]) -> Option<Decision> {
        let (chosen, _scores) = self.select_with_scores(arms_in_order, context)?;
        let explore_first = self
            .state
            .get(chosen.as_str())
            .map(|s| s.uses == 0)
            .unwrap_or(false);

        Some(Decision {
            policy: DecisionPolicy::LinUcb,
            chosen: chosen.clone(),
            probs: None,
            notes: if explore_first {
                vec![DecisionNote::ExploreFirst]
            } else {
                vec![DecisionNote::DeterministicChoice]
            },
        })
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

    /// Select via softmax over UCB scores and return a unified `Decision`.
    ///
    /// Notes:
    /// - Always includes `probs` (the softmax allocation over UCB for this context).
    /// - Records explore-first vs sampling and numerical fallback.
    pub fn decide_softmax_ucb(
        &mut self,
        arms_in_order: &[String],
        context: &[f64],
        temperature: f64,
    ) -> Option<Decision> {
        self.ensure_arms(arms_in_order);
        if arms_in_order.is_empty() {
            return None;
        }

        let probs = self.probabilities(arms_in_order, context, temperature);

        // Explore first (stable order) while still returning full probs.
        for a in arms_in_order {
            let uses = self.state.get(a).map(|s| s.uses).unwrap_or(0);
            if uses == 0 {
                return Some(Decision {
                    policy: DecisionPolicy::LinUcb,
                    chosen: a.clone(),
                    probs: Some(probs),
                    notes: vec![DecisionNote::ExploreFirst],
                });
            }
        }

        let r: f64 = self.rng.random();
        let mut cdf = 0.0;
        for a in arms_in_order {
            cdf += probs.get(a).copied().unwrap_or(0.0);
            if r < cdf {
                return Some(Decision {
                    policy: DecisionPolicy::LinUcb,
                    chosen: a.clone(),
                    probs: Some(probs),
                    notes: vec![DecisionNote::SampledFromDistribution],
                });
            }
        }

        let last = arms_in_order.last()?.clone();
        Some(Decision {
            policy: DecisionPolicy::LinUcb,
            chosen: last,
            probs: Some(probs),
            notes: vec![
                DecisionNote::SampledFromDistribution,
                DecisionNote::NumericalFallbackToLastArm,
            ],
        })
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

    /// Return per-arm theta vectors (A_inv @ b) for sensitivity analysis.
    ///
    /// Each arm's theta is its learned response function: `E[reward | context x] = theta^T x`.
    /// The matrix of theta vectors (arms x dim) can be fed to `pare::sensitivity::analyze_redundancy`
    /// to compute the gradient rank and identify which arms have genuinely different
    /// context-dependent behavior.
    pub fn theta_vectors(&mut self, arms_in_order: &[String]) -> BTreeMap<String, Vec<f64>> {
        self.ensure_arms(arms_in_order);
        let mut out = BTreeMap::new();
        for a in arms_in_order {
            if let Some(st) = self.state.get(a) {
                out.insert(a.clone(), self.theta(st));
            }
        }
        out
    }

    /// Capture a persistence snapshot of the current LinUCB state.
    ///
    /// This includes per-arm sufficient statistics (A_inv, b, uses) so that
    /// a caller can serialize, store, and later restore the policy state across
    /// process restarts.
    pub fn snapshot(&self) -> LinUcbState {
        let mut arms = BTreeMap::new();
        for (name, st) in &self.state {
            arms.insert(
                name.clone(),
                LinUcbArmState {
                    a_inv: st.a_inv.clone(),
                    b: st.b.clone(),
                    uses: st.uses,
                },
            );
        }
        LinUcbState {
            dim: self.dim(),
            arms,
        }
    }

    /// Restore a previously snapshotted LinUCB state.
    ///
    /// Arms not present in the snapshot are initialized fresh.  Arms in the
    /// snapshot but not in the current arm set are ignored.  Dimension mismatches
    /// cause affected arms to be re-initialized.
    pub fn restore(&mut self, st: LinUcbState) {
        let dim = self.dim();
        for (name, arm_st) in st.arms {
            if arm_st.a_inv.len() != dim * dim || arm_st.b.len() != dim {
                continue; // dimension mismatch, skip
            }
            if !arm_st.a_inv.iter().all(|v| v.is_finite())
                || !arm_st.b.iter().all(|v| v.is_finite())
            {
                continue; // corrupted, skip
            }
            self.state.insert(
                name,
                ArmState {
                    a_inv: arm_st.a_inv,
                    b: arm_st.b,
                    uses: arm_st.uses,
                },
            );
        }
    }
}

/// Serializable per-arm state for LinUCB persistence.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LinUcbArmState {
    /// Inverse design matrix (d x d, row-major).
    pub a_inv: Vec<f64>,
    /// Reward-weighted feature sum (d).
    pub b: Vec<f64>,
    /// Number of updates for this arm.
    pub uses: u64,
}

/// Serializable LinUCB state snapshot (for persistence across process restarts).
///
/// Mirrors `Exp3IxState` in design: intentionally excludes RNG state (callers
/// manage seeds externally) and stores only the sufficient statistics needed to
/// resume learning.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LinUcbState {
    /// Feature dimension used when this state was captured.
    pub dim: usize,
    /// Per-arm sufficient statistics.
    pub arms: BTreeMap<String, LinUcbArmState>,
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

    #[test]
    fn linucb_snapshot_restore_round_trip() {
        let arms = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let cfg = LinUcbConfig {
            dim: 3,
            lambda: 1.0,
            alpha: 0.5,
            seed: 42,
            decay: 0.98,
        };

        // Train for a while.
        let mut p1 = LinUcb::new(cfg);
        let ctx1 = [1.0, 0.0, 0.5];
        let ctx2 = [0.0, 1.0, 0.2];
        for t in 0..50u64 {
            let ctx = if t % 2 == 0 { &ctx1[..] } else { &ctx2[..] };
            let (chosen, _) = p1.select_with_scores(&arms, ctx).unwrap();
            let r = if chosen.as_str() == "a" { 0.9 } else { 0.3 };
            p1.update_reward(chosen, ctx, r);
        }

        // Snapshot.
        let snap = p1.snapshot();
        assert_eq!(snap.dim, 3);
        assert_eq!(snap.arms.len(), 3);

        // Restore into a fresh instance.
        let mut p2 = LinUcb::new(cfg);
        p2.ensure_arms(&arms);
        p2.restore(snap);

        // Scores must match after restore.
        for ctx in [&ctx1[..], &ctx2[..]] {
            let s1 = p1.scores(&arms, ctx);
            let s2 = p2.scores(&arms, ctx);
            for a in &arms {
                let (ucb1, mean1, bonus1) = s1[a];
                let (ucb2, mean2, bonus2) = s2[a];
                assert!(
                    (ucb1 - ucb2).abs() < 1e-12,
                    "ucb mismatch for {a}: {ucb1} vs {ucb2}"
                );
                assert!(
                    (mean1 - mean2).abs() < 1e-12,
                    "mean mismatch for {a}: {mean1} vs {mean2}"
                );
                assert!(
                    (bonus1 - bonus2).abs() < 1e-12,
                    "bonus mismatch for {a}: {bonus1} vs {bonus2}"
                );
            }
        }

        // Selections must match after restore.
        let test_ctx = [0.5, 0.5, 0.5];
        let (c1, _) = p1.select_with_scores(&arms, &test_ctx).unwrap();
        let (c2, _) = p2.select_with_scores(&arms, &test_ctx).unwrap();
        assert_eq!(c1, c2);
    }

    proptest! {
        #[test]
        fn linucb_snapshot_restore_preserves_scores(
            dim in 1usize..6,
            alpha in 0.0f64..3.0f64,
            lambda in 1.0e-3f64..5.0f64,
            seed in any::<u64>(),
            ctx in proptest::collection::vec(-5.0f64..5.0f64, 0..12),
            rewards in proptest::collection::vec(0.0f64..1.0f64, 5..30),
        ) {
            let arms = vec!["a".to_string(), "b".to_string()];
            let cfg = LinUcbConfig { dim, alpha, lambda, seed, decay: 1.0 };

            let mut p1 = LinUcb::new(cfg);
            for (i, r) in rewards.iter().enumerate() {
                let (chosen, _) = p1.select_with_scores(&arms, &ctx).unwrap();
                p1.update_reward(chosen, &ctx, *r);
                let _ = i;
            }

            let snap = p1.snapshot();
            let mut p2 = LinUcb::new(cfg);
            p2.ensure_arms(&arms);
            p2.restore(snap);

            let s1 = p1.scores(&arms, &ctx);
            let s2 = p2.scores(&arms, &ctx);
            for a in &arms {
                let (u1, m1, b1) = s1[a];
                let (u2, m2, b2) = s2[a];
                prop_assert!((u1 - u2).abs() < 1e-9, "ucb: {} vs {}", u1, u2);
                prop_assert!((m1 - m2).abs() < 1e-9, "mean: {} vs {}", m1, m2);
                prop_assert!((b1 - b2).abs() < 1e-9, "bonus: {} vs {}", b1, b2);
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
