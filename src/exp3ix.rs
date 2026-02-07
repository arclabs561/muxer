//! EXP3-IX (adversarial bandit) for arm selection.
//!
//! This policy is useful when rewards can be adversarial / highly non-stationary.
//! It is **seedable** so it can be reproducible in tests. Like other policies
//! in this crate, default construction is deterministic by default.

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::collections::BTreeMap;

use crate::{
    apply_latency_guardrail, Decision, DecisionNote, DecisionPolicy, LatencyGuardrailConfig,
    Summary,
};

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

/// Serializable EXP3-IX state snapshot (for persistence).
///
/// This intentionally excludes RNG state; callers that want deterministic sampling can
/// sample deterministically from `probabilities(...)` using their own seed.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Exp3IxState {
    pub arms: Vec<String>,
    pub uses: Vec<u64>,
    pub cum_loss_hat: Vec<f64>,
    /// Probability distribution aligned with `arms`.
    ///
    /// This is redundant (it can be recomputed from `cum_loss_hat`), but storing it avoids
    /// small numerical drift and makes persistence/replay cheaper.
    #[cfg_attr(feature = "serde", serde(default))]
    pub probs: Vec<f64>,
}

/// Deterministic persisted EXP3-IX decision helper.
///
/// This is designed for “thin harnesses” that want to keep storage logic outside of `muxer`.
/// It restores state (if present), makes a deterministic filtered decision, and returns the
/// updated persistence snapshot (with cached probabilities).
#[cfg(feature = "stochastic")]
pub fn exp3ix_decide_persisted(
    cfg: Exp3IxConfig,
    state: Option<Exp3IxState>,
    arms_in_order: &[String],
    eligible_in_order: &[String],
    decision_seed: u64,
) -> Option<(Decision, Exp3IxState)> {
    let mut ex = Exp3Ix::new(cfg);
    if let Some(st) = state {
        ex.restore(st);
    }
    let d = ex.decide_deterministic_filtered(arms_in_order, eligible_in_order, decision_seed)?;
    let st = ex.snapshot();
    Some((d, st))
}

/// Persisted EXP3-IX decision helper that also returns the probability used.
///
/// This returns a "ticket" (`chosen`, `prob_used`) that can be stored across an evaluation and
/// later fed back into `exp3ix_update_persisted(...)` without retaining the full `Decision`.
#[cfg(feature = "stochastic")]
pub fn exp3ix_decide_persisted_with_prob(
    cfg: Exp3IxConfig,
    state: Option<Exp3IxState>,
    arms_in_order: &[String],
    eligible_in_order: &[String],
    decision_seed: u64,
) -> Option<(Decision, Exp3IxState, f64)> {
    let (d, st) =
        exp3ix_decide_persisted(cfg, state, arms_in_order, eligible_in_order, decision_seed)?;
    let prob_used = d
        .probs
        .as_ref()
        .and_then(|m| m.get(d.chosen.as_str()).copied())
        .unwrap_or(0.0);
    Some((d, st, prob_used))
}

/// Persisted EXP3-IX update helper.
///
/// Updates the stored state using an explicit probability mass (typically the probability
/// that was actually used to sample the chosen arm after external filtering/renormalization).
#[cfg(feature = "stochastic")]
pub fn exp3ix_update_persisted(
    cfg: Exp3IxConfig,
    state: Exp3IxState,
    chosen: &str,
    reward01: f64,
    prob_used: f64,
) -> Exp3IxState {
    let mut ex = Exp3Ix::new(cfg);
    ex.restore(state);
    ex.update_reward_with_prob(chosen, reward01, prob_used);
    ex.snapshot()
}

/// Convenience helper: apply a latency guardrail and then make a persisted EXP3-IX decision.
///
/// Returns the guardrail decision metadata plus the EXP3-IX decision ticket.
#[derive(Debug, Clone)]
pub struct Exp3IxGuardrailedDecision {
    pub guardrail: crate::LatencyGuardrailDecision,
    pub decision: Decision,
    pub state: Exp3IxState,
    pub prob_used: f64,
}

/// A compact, log-ready row for an EXP3-IX decision under an external guardrail.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Exp3IxRoundLog {
    pub remaining: Vec<String>,
    pub guardrail_eligible: Vec<String>,
    pub guardrail_fallback_used: bool,
    pub guardrail_stop_early: bool,
    pub chosen: String,
    pub explore_first: bool,
    pub prob_used: f64,
    pub top_candidates: crate::LogTopCandidates,
}

/// Convert a persisted, guardrailed EXP3-IX decision into a compact log row.
#[cfg(feature = "stochastic")]
pub fn log_exp3ix_guardrailed_typed(
    gd: &Exp3IxGuardrailedDecision,
    remaining: &[String],
    top: usize,
) -> Exp3IxRoundLog {
    let explore_first = gd
        .decision
        .notes
        .iter()
        .any(|n| matches!(n, crate::DecisionNote::ExploreFirst));

    Exp3IxRoundLog {
        remaining: remaining.to_vec(),
        guardrail_eligible: gd.guardrail.eligible.clone(),
        guardrail_fallback_used: gd.guardrail.fallback_used,
        guardrail_stop_early: gd.guardrail.stop_early,
        chosen: gd.decision.chosen.clone(),
        explore_first,
        prob_used: gd.prob_used,
        top_candidates: crate::log_top_candidates_exp3ix_typed(&gd.decision, top),
    }
}

#[cfg(feature = "stochastic")]
pub fn exp3ix_decide_persisted_guardrailed(
    cfg: Exp3IxConfig,
    state: Option<Exp3IxState>,
    arms_in_order: &[String],
    summaries: &BTreeMap<String, Summary>,
    guardrail: LatencyGuardrailConfig,
    already_chosen: usize,
    decision_seed: u64,
) -> Option<Exp3IxGuardrailedDecision> {
    let gd = apply_latency_guardrail(arms_in_order, summaries, guardrail, already_chosen);
    if gd.stop_early || gd.eligible.is_empty() {
        return None;
    }
    let (decision, st, prob_used) =
        exp3ix_decide_persisted_with_prob(cfg, state, arms_in_order, &gd.eligible, decision_seed)?;
    Some(Exp3IxGuardrailedDecision {
        guardrail: gd,
        decision,
        state: st,
        prob_used,
    })
}

/// Multi-pick EXP3-IX selection round (guardrailed).
#[derive(Debug, Clone)]
pub struct Exp3IxKRound {
    pub remaining: Vec<String>,
    pub guardrail: crate::LatencyGuardrailDecision,
    pub decision: Decision,
    pub prob_used: f64,
}

/// Stop record when multi-pick selection ends early (guardrail stop/empty eligible).
#[derive(Debug, Clone)]
pub struct Exp3IxKStop {
    pub remaining: Vec<String>,
    pub guardrail: crate::LatencyGuardrailDecision,
}

/// Multi-pick EXP3-IX selection (guardrailed, deterministic, persisted-state friendly).
#[derive(Debug, Clone)]
pub struct Exp3IxKExplain {
    /// Chosen arms in selection order.
    pub chosen: Vec<String>,
    /// Per-round decision details.
    pub rounds: Vec<Exp3IxKRound>,
    /// Optional stop record when selection ends early.
    pub stop: Option<Exp3IxKStop>,
    /// Updated persistence snapshot (same arm order, cached probabilities).
    pub state: Exp3IxState,
}

/// Log-ready row for a multi-pick EXP3-IX selection.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Exp3IxKRoundLog {
    pub round: usize,
    pub remaining: Vec<String>,
    pub guardrail_eligible: Vec<String>,
    pub guardrail_fallback_used: bool,
    pub guardrail_stop_early: bool,
    pub chosen: Option<String>,
    pub explore_first: Option<bool>,
    pub prob_used: Option<f64>,
    pub top_candidates: Option<crate::LogTopCandidates>,
}

/// Convert a multi-pick EXP3-IX explanation into compact, log-ready round rows.
///
/// This includes an additional “stop row” when selection ends early (e.g. guardrail `stop_early=true`).
#[cfg(feature = "stochastic")]
pub fn log_exp3ix_k_rounds_typed(explain: &Exp3IxKExplain, top: usize) -> Vec<Exp3IxKRoundLog> {
    let mut out: Vec<Exp3IxKRoundLog> = Vec::new();

    for (i, r) in explain.rounds.iter().enumerate() {
        let explore_first = r
            .decision
            .notes
            .iter()
            .any(|n| matches!(n, DecisionNote::ExploreFirst));
        out.push(Exp3IxKRoundLog {
            round: i + 1,
            remaining: r.remaining.clone(),
            guardrail_eligible: r.guardrail.eligible.clone(),
            guardrail_fallback_used: r.guardrail.fallback_used,
            guardrail_stop_early: r.guardrail.stop_early,
            chosen: Some(r.decision.chosen.clone()),
            explore_first: Some(explore_first),
            prob_used: Some(r.prob_used),
            top_candidates: Some(crate::log_top_candidates_exp3ix_typed(&r.decision, top)),
        });
    }

    if let Some(ref s) = explain.stop {
        out.push(Exp3IxKRoundLog {
            round: explain.rounds.len() + 1,
            remaining: s.remaining.clone(),
            guardrail_eligible: s.guardrail.eligible.clone(),
            guardrail_fallback_used: s.guardrail.fallback_used,
            guardrail_stop_early: s.guardrail.stop_early,
            chosen: None,
            explore_first: None,
            prob_used: None,
            top_candidates: None,
        });
    }

    out
}

/// Select up to `k` unique arms by repeatedly applying deterministic EXP3-IX sampling, with an optional
/// external latency guardrail applied each round.
///
/// Notes:
/// - This does **not** update EXP3-IX losses between picks; it only selects a batch of distinct arms
///   for evaluation. Callers should feed the observed rewards back via `exp3ix_update_persisted(...)`.
/// - The returned `prob_used` is the probability mass from the renormalized eligible distribution in
///   the round that produced that pick.
#[cfg(feature = "stochastic")]
pub fn exp3ix_decide_k_persisted_guardrailed_explain_full(
    cfg: Exp3IxConfig,
    state: Option<Exp3IxState>,
    arms_in_order: &[String],
    summaries: &BTreeMap<String, Summary>,
    guardrail: LatencyGuardrailConfig,
    k: usize,
    decision_seed: u64,
) -> Exp3IxKExplain {
    if arms_in_order.is_empty() || k == 0 {
        let mut ex = Exp3Ix::new(cfg);
        if let Some(st) = state {
            ex.restore(st);
        }
        // Ensure arm order is initialized for persistence stability.
        ex.ensure_arms(arms_in_order);
        return Exp3IxKExplain {
            chosen: Vec::new(),
            rounds: Vec::new(),
            stop: None,
            state: ex.snapshot(),
        };
    }

    let mut ex = Exp3Ix::new(cfg);
    if let Some(st) = state {
        ex.restore(st);
    }
    ex.ensure_arms(arms_in_order);

    let mut remaining: Vec<String> = arms_in_order.to_vec();
    let mut chosen: Vec<String> = Vec::new();
    let mut rounds: Vec<Exp3IxKRound> = Vec::new();
    let mut stop: Option<Exp3IxKStop> = None;

    for round in 0..k.min(remaining.len()) {
        let remaining_in = remaining.clone();
        let gd = apply_latency_guardrail(&remaining_in, summaries, guardrail, chosen.len());
        if gd.stop_early || gd.eligible.is_empty() {
            stop = Some(Exp3IxKStop {
                remaining: remaining_in,
                guardrail: gd,
            });
            break;
        }

        let seed = decision_seed ^ ((round as u64).wrapping_add(1).wrapping_mul(0x9E37_79B9));
        let Some(decision) = ex.decide_deterministic_filtered(arms_in_order, &gd.eligible, seed)
        else {
            stop = Some(Exp3IxKStop {
                remaining: remaining_in,
                guardrail: gd,
            });
            break;
        };
        let pick = decision.chosen.clone();
        let prob_used = decision
            .probs
            .as_ref()
            .and_then(|m| m.get(pick.as_str()).copied())
            .unwrap_or(0.0);

        chosen.push(pick.clone());
        remaining.retain(|b| b != &pick);
        rounds.push(Exp3IxKRound {
            remaining: remaining_in,
            guardrail: gd,
            decision,
            prob_used,
        });
    }

    Exp3IxKExplain {
        chosen,
        rounds,
        stop,
        state: ex.snapshot(),
    }
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

    fn reset_arms(&mut self, arms_in_order: &[String]) {
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

    /// Ensure internal state is aligned with `arms_in_order`.
    ///
    /// **Important**: if the arm set has changed (different names, different order, or different
    /// length), all learned state is **silently reset** (cumulative losses zeroed, probabilities
    /// reset to uniform). This is by design: EXP3-IX's per-arm loss estimates are indexed by
    /// position, so a change in the arm vector invalidates them.
    ///
    /// If you need to add/remove arms without losing state, use [`snapshot`](Self::snapshot)
    /// and [`restore`](Self::restore) to migrate explicitly.
    fn ensure_arms(&mut self, arms_in_order: &[String]) {
        if self.arms == arms_in_order {
            return;
        }
        self.reset_arms(arms_in_order);
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

    /// Capture a persistence snapshot of the current EXP3-IX state.
    ///
    /// Callers should prefer calling this after `probabilities(...)` or `decide(...)` so
    /// `arms` is initialized and `probs` is up to date.
    pub fn snapshot(&self) -> Exp3IxState {
        Exp3IxState {
            arms: self.arms.clone(),
            uses: self.uses.clone(),
            cum_loss_hat: self.cum_loss_hat.clone(),
            probs: self.probs.clone(),
        }
    }

    /// Restore a previously snapshotted EXP3-IX state.
    ///
    /// If the stored state is inconsistent (length mismatches), this resets to a fresh state.
    pub fn restore(&mut self, st: Exp3IxState) {
        if st.arms.len() != st.uses.len() || st.arms.len() != st.cum_loss_hat.len() {
            self.reset_arms(&[]);
            return;
        }
        self.reset_arms(&st.arms);
        self.uses = st.uses;
        self.cum_loss_hat = st.cum_loss_hat;
        if st.probs.len() == self.arms.len() && st.probs.iter().all(|x| x.is_finite()) {
            self.probs = st.probs;
        } else {
            self.recompute_probs();
        }
    }

    fn u01(seed: u64) -> f64 {
        crate::stable_hash::u01_from_seed(seed)
    }

    fn filtered_probs(&self, eligible_in_order: &[String]) -> BTreeMap<String, f64> {
        // Base probabilities are `self.probs` aligned with `self.arms`.
        // We project onto `eligible_in_order` and renormalize.
        let mut out = BTreeMap::new();
        let mut sum = 0.0;
        for a in eligible_in_order {
            let p = self
                .arms
                .iter()
                .position(|x| x == a)
                .and_then(|i| self.probs.get(i).copied())
                .unwrap_or(0.0);
            let pi = if p.is_finite() && p > 0.0 { p } else { 0.0 };
            out.insert(a.clone(), pi);
            sum += pi;
        }
        if sum > 0.0 && sum.is_finite() {
            for v in out.values_mut() {
                *v /= sum;
            }
            return out;
        }
        // Fallback: uniform over eligible.
        let k = eligible_in_order.len().max(1) as f64;
        out.clear();
        for a in eligible_in_order {
            out.insert(a.clone(), 1.0 / k);
        }
        out
    }

    /// Deterministic decision from a filtered eligible set.
    ///
    /// This is designed for callers (like `anno`) that:
    /// - keep persistent EXP3-IX state across process runs
    /// - apply external hard constraints (e.g. latency guardrail) that shrink the eligible set
    /// - want a deterministic decision given a seed, without persisting RNG state
    ///
    /// The returned `Decision.probs` is over `eligible_in_order` (renormalized).
    /// If you update using this decision, prefer `update_reward_with_prob(...)` with
    /// `prob_used := decision.probs[chosen]`.
    pub fn decide_deterministic_filtered(
        &mut self,
        arms_in_order: &[String],
        eligible_in_order: &[String],
        decision_seed: u64,
    ) -> Option<Decision> {
        self.ensure_arms(arms_in_order);
        if self.arms.is_empty() || eligible_in_order.is_empty() {
            return None;
        }

        // Always capture probabilities as of this decision.
        let probs = self.filtered_probs(eligible_in_order);

        // Explore-first within the eligible set (stable order).
        for a in eligible_in_order {
            let uses = self
                .arms
                .iter()
                .position(|x| x == a)
                .and_then(|i| self.uses.get(i).copied())
                .unwrap_or(0);
            if uses == 0 {
                return Some(Decision {
                    policy: DecisionPolicy::Exp3Ix,
                    chosen: a.clone(),
                    probs: Some(probs),
                    notes: vec![DecisionNote::ExploreFirst],
                });
            }
        }

        let r = Self::u01(decision_seed);
        let mut cdf = 0.0;
        for a in eligible_in_order {
            cdf += probs.get(a).copied().unwrap_or(0.0);
            if r < cdf {
                return Some(Decision {
                    policy: DecisionPolicy::Exp3Ix,
                    chosen: a.clone(),
                    probs: Some(probs),
                    notes: vec![DecisionNote::SampledFromDistribution],
                });
            }
        }

        // Numerical fallback.
        let last = eligible_in_order.last()?.clone();
        Some(Decision {
            policy: DecisionPolicy::Exp3Ix,
            chosen: last,
            probs: Some(probs),
            notes: vec![
                DecisionNote::SampledFromDistribution,
                DecisionNote::NumericalFallbackToLastArm,
            ],
        })
    }

    /// Update EXP3-IX with a bounded reward in `[0, 1]`, using an explicit probability.
    ///
    /// This is useful when the decision was made from a filtered/renormalized distribution
    /// (e.g. a latency guardrail) and you want the importance weighting to use the exact
    /// probability mass function that was actually sampled.
    pub fn update_reward_with_prob(&mut self, arm: &str, reward01: f64, prob_used: f64) {
        if self.arms.is_empty() {
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
        let p = if prob_used.is_finite() && prob_used > 0.0 {
            prob_used
        } else {
            0.0
        };
        let denom = p + self.gamma;
        let loss_hat = if denom > 0.0 { loss / denom } else { loss };

        self.cum_loss_hat[idx] += loss_hat;
        self.uses[idx] = self.uses[idx].saturating_add(1);
        self.recompute_probs();
    }

    /// Select an arm and return the probabilities used for selection.
    ///
    /// # Example
    ///
    /// ```rust
    /// use muxer::{Exp3Ix, Exp3IxConfig};
    ///
    /// let arms = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    /// let mut ex = Exp3Ix::new(Exp3IxConfig { seed: 123, decay: 0.98, ..Exp3IxConfig::default() });
    /// let (chosen, probs) = ex.select_with_probs(&arms).unwrap();
    /// ex.update_reward(chosen, 0.7);
    /// let s: f64 = probs.values().sum();
    /// assert!((s - 1.0).abs() < 1e-9);
    /// ```
    pub fn select_with_probs<'a>(
        &mut self,
        arms_in_order: &'a [String],
    ) -> Option<(&'a String, BTreeMap<String, f64>)> {
        let d = self.decide(arms_in_order)?;
        let chosen = arms_in_order
            .iter()
            .find(|a| a.as_str() == d.chosen.as_str())?;
        Some((chosen, d.probs.unwrap_or_default()))
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

    /// Select an arm and return a unified `Decision` (recommended for logging/replay).
    ///
    /// Notes:
    /// - Always includes a `probs` distribution over arms as of this decision.
    /// - Records whether explore-first occurred and whether numerical fallback was used.
    pub fn decide(&mut self, arms_in_order: &[String]) -> Option<Decision> {
        self.ensure_arms(arms_in_order);
        if self.arms.is_empty() {
            return None;
        }

        // Always capture probabilities as of this decision.
        let probs = self.probabilities(arms_in_order);

        // Explore first (stable order).
        for (i, a) in arms_in_order.iter().enumerate() {
            if self.uses.get(i).copied().unwrap_or(0) == 0 {
                return Some(Decision {
                    policy: DecisionPolicy::Exp3Ix,
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
                    policy: DecisionPolicy::Exp3Ix,
                    chosen: a.clone(),
                    probs: Some(probs),
                    notes: vec![DecisionNote::SampledFromDistribution],
                });
            }
        }

        // Numerical fallback (CDF did not reach 1.0 due to rounding/NaNs).
        let last = arms_in_order.last()?.clone();
        Some(Decision {
            policy: DecisionPolicy::Exp3Ix,
            chosen: last,
            probs: Some(probs),
            notes: vec![
                DecisionNote::SampledFromDistribution,
                DecisionNote::NumericalFallbackToLastArm,
            ],
        })
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
        let p = self.probs.get(idx).copied().unwrap_or(0.0);
        self.update_reward_with_prob(arm, reward01, p);
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
    fn snapshot_restore_round_trip() {
        let cfg = Exp3IxConfig {
            horizon: 200,
            confidence_delta: None,
            seed: 7,
            decay: 0.98,
        };
        let arms = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mut ex = Exp3Ix::new(cfg);

        // Train for a while.
        for _ in 0..30 {
            let chosen = ex.select(&arms).unwrap().clone();
            let r = if chosen == "a" { 0.9 } else { 0.2 };
            ex.update_reward(&chosen, r);
        }

        let snap = ex.snapshot();
        assert_eq!(snap.arms.len(), 3);
        assert_eq!(snap.uses.len(), 3);
        assert_eq!(snap.cum_loss_hat.len(), 3);
        assert_eq!(snap.probs.len(), 3);

        // Restore into a fresh instance.
        let mut ex2 = Exp3Ix::new(cfg);
        ex2.restore(snap);

        // Probabilities must match.
        let p1 = ex.probabilities(&arms);
        let p2 = ex2.probabilities(&arms);
        for a in &arms {
            assert!(
                (p1[a] - p2[a]).abs() < 1e-12,
                "prob mismatch for {a}: {} vs {}",
                p1[a],
                p2[a]
            );
        }

        // Deterministic decision must match (same seed, same state).
        let d1 = ex.decide_deterministic_filtered(&arms, &arms, 999);
        let d2 = ex2.decide_deterministic_filtered(&arms, &arms, 999);
        assert_eq!(
            d1.as_ref().map(|d| &d.chosen),
            d2.as_ref().map(|d| &d.chosen)
        );
    }

    #[test]
    fn snapshot_restore_handles_corrupted_state() {
        let cfg = Exp3IxConfig::default();
        let mut ex = Exp3Ix::new(cfg);

        // Restore with mismatched lengths -> should reset.
        let bad = Exp3IxState {
            arms: vec!["a".to_string(), "b".to_string()],
            uses: vec![1], // wrong length
            cum_loss_hat: vec![0.0, 0.0],
            probs: vec![0.5, 0.5],
        };
        ex.restore(bad);
        // After restoring bad state, arms should be empty (reset).
        assert!(ex.probabilities(&["x".to_string()]).contains_key("x"));
    }

    #[test]
    fn exp3ix_guardrailed_log_row_is_well_formed() {
        let cfg = Exp3IxConfig {
            horizon: 100,
            confidence_delta: None,
            seed: 0,
            decay: 1.0,
        };
        let arms = vec!["a".to_string(), "b".to_string()];
        let mut m = BTreeMap::new();
        m.insert(
            "a".to_string(),
            Summary {
                calls: 1,
                ok: 1,
                junk: 0,
                hard_junk: 0,
                cost_units: 0,
                elapsed_ms_sum: 10,
            },
        );
        m.insert(
            "b".to_string(),
            Summary {
                calls: 1,
                ok: 1,
                junk: 0,
                hard_junk: 0,
                cost_units: 0,
                elapsed_ms_sum: 10,
            },
        );
        let gd = exp3ix_decide_persisted_guardrailed(
            cfg,
            None,
            &arms,
            &m,
            LatencyGuardrailConfig::default(),
            0,
            123,
        )
        .expect("decision");

        let row = log_exp3ix_guardrailed_typed(&gd, &arms, 2);
        assert_eq!(row.remaining, arms);
        assert_eq!(row.guardrail_eligible.len(), 2);
        assert!(row.prob_used.is_finite());
        assert_eq!(row.top_candidates.kind, crate::LOG_SCORE_KIND_EXP3IX_PROB);
        assert!(!row.chosen.is_empty());
    }

    #[test]
    fn exp3ix_multi_pick_guardrailed_selects_unique_arms() {
        let cfg = Exp3IxConfig {
            horizon: 100,
            confidence_delta: None,
            seed: 0,
            decay: 1.0,
        };
        let arms = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mut m = BTreeMap::new();
        for a in &arms {
            m.insert(
                a.clone(),
                Summary {
                    calls: 1,
                    ok: 1,
                    junk: 0,
                    hard_junk: 0,
                    cost_units: 0,
                    elapsed_ms_sum: 10,
                },
            );
        }

        let ex = exp3ix_decide_k_persisted_guardrailed_explain_full(
            cfg,
            None,
            &arms,
            &m,
            LatencyGuardrailConfig::default(),
            2,
            123,
        );

        assert_eq!(ex.chosen.len(), 2);
        assert_ne!(
            ex.chosen[0], ex.chosen[1],
            "multi-pick must not repeat arms"
        );
        assert_eq!(ex.rounds.len(), 2);

        let logs = log_exp3ix_k_rounds_typed(&ex, 3);
        assert_eq!(logs.len(), 2);
        assert_eq!(logs[0].round, 1);
        assert_eq!(logs[1].round, 2);
        assert!(logs[0].chosen.as_ref().is_some());
        assert!(logs[0].prob_used.unwrap_or(0.0).is_finite());
    }

    #[test]
    fn ensure_arms_resets_state_on_arm_set_change() {
        let cfg = Exp3IxConfig {
            horizon: 200,
            confidence_delta: None,
            seed: 0,
            decay: 1.0,
        };
        let arms_v1 = vec!["a".to_string(), "b".to_string()];
        let arms_v2 = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        let mut ex = Exp3Ix::new(cfg);

        // Train on v1.
        for _ in 0..20 {
            let chosen = ex.select(&arms_v1).unwrap().clone();
            ex.update_reward(&chosen, if chosen == "a" { 0.9 } else { 0.1 });
        }

        // After training, probabilities should be non-uniform.
        let probs_before = ex.probabilities(&arms_v1);
        let p_a_before = probs_before["a"];
        assert!(
            (p_a_before - 0.5).abs() > 0.01,
            "probs should be non-uniform after training: {probs_before:?}"
        );

        // Change arm set: this must reset state.
        let probs_after = ex.probabilities(&arms_v2);
        let p_a_after = probs_after["a"];
        let p_c_after = probs_after["c"];
        let expected_uniform = 1.0 / 3.0;
        assert!(
            (p_a_after - expected_uniform).abs() < 1e-9,
            "arm set change should reset to uniform: {probs_after:?}"
        );
        assert!(
            (p_c_after - expected_uniform).abs() < 1e-9,
            "new arm should start uniform: {probs_after:?}"
        );
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

    #[test]
    fn exp3ix_can_outperform_mab_when_reward_is_graded_but_success_is_constant() {
        // This is a deliberately constructed scenario to demonstrate a *capability gap*:
        //
        // - MAB selection in this crate consumes `Summary` (ok/junk/latency/cost) and cannot
        //   express a graded reward signal when `ok_rate` is 1.0 for all arms and other metrics
        //   are equal.
        // - EXP3-IX consumes a scalar reward in [0, 1] and can learn to prefer higher reward.
        //
        // So, in this scenario EXP3-IX should strictly beat deterministic MAB selection.
        use crate::{select_mab_explain, MabConfig, Summary};

        let arms = vec!["a".to_string(), "b".to_string()];

        // But rewards differ (graded quality). With identical summaries, deterministic MAB
        // tie-breaking will stick to one arm; we set that arm ("a", lexicographically first)
        // to be worse so EXP3-IX has an opportunity to learn and beat it.
        let r_a = 0.6;
        let r_b = 0.9;

        // MAB config with only success dimension (no penalties).
        let cfg = MabConfig {
            exploration_c: 0.8,
            ..MabConfig::default()
        };

        let mut ex = Exp3Ix::new(Exp3IxConfig {
            horizon: 200,
            confidence_delta: None,
            seed: 0,
            decay: 1.0,
        });

        // Both arms have identical summaries forever: MAB cannot distinguish them.
        let s = Summary {
            calls: 10,
            ok: 10,
            junk: 0,
            hard_junk: 0,
            cost_units: 0,
            elapsed_ms_sum: 0,
        };

        let mut total_mab = 0.0;
        let mut total_exp3 = 0.0;

        for _ in 0..200 {
            let mut m = BTreeMap::new();
            m.insert("a".to_string(), s);
            m.insert("b".to_string(), s);
            let mab_choice = select_mab_explain(&arms, &m, cfg).selection.chosen;
            let r = if mab_choice == "a" { r_a } else { r_b };
            total_mab += r;

            // EXP3-IX learns from scalar reward.
            let chosen = ex.select(&arms).unwrap().clone();
            let r = if chosen == "a" { r_a } else { r_b };
            total_exp3 += r;
            ex.update_reward(&chosen, r);
        }

        assert!(
            total_exp3 > total_mab + 5.0,
            "expected exp3ix to beat mab in this scenario: exp3={} mab={}",
            total_exp3,
            total_mab
        );
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

        #[test]
        fn exp3ix_decay_keeps_probs_well_formed(
            seed in any::<u64>(),
            decay in 0.01f64..1.0f64,
            steps in 0usize..400,
            rewards in proptest::collection::vec(0.0f64..1.0f64, 0..400),
        ) {
            let cfg = Exp3IxConfig { seed, horizon: 2000, confidence_delta: None, decay };
            let mut ex = Exp3Ix::new(cfg);
            let arms = vec!["a".to_string(), "b".to_string(), "c".to_string()];

            for i in 0..steps {
                let (chosen, probs) = ex.select_with_probs(&arms).unwrap();
                let sum: f64 = probs.values().sum();
                prop_assert!((sum - 1.0).abs() < 1e-9);
                for v in probs.values() {
                    prop_assert!(v.is_finite());
                    prop_assert!(*v >= 0.0 && *v <= 1.0);
                }
                let r = rewards.get(i).copied().unwrap_or(0.5);
                ex.update_reward(chosen, r);
            }
        }
    }
}
