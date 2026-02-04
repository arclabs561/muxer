//! `muxer`: deterministic, multi-objective bandit-style routing primitives.
//!
//! This crate is designed for “provider routing” style problems:
//! you have a small set of arms (providers) and repeated calls that produce
//! outcomes (ok/fail, rate limit, cost, latency, “junk”).
//! outcomes (ok/fail, cost, latency, “junk”).
//!
//! Goals:
//! - **Deterministic by default**: same stats + config → same choice.
//! - **Non-stationarity friendly**: prefer **sliding-window** summaries over lifetime averages.
//! - **Multi-objective**: choose on a Pareto frontier, then deterministic scalarization.
//!
//! Included policies:
//! - `select_mab`: deterministic Pareto + scalarization selection from windowed summaries.
//! - `ThompsonSampling`: seedable Thompson sampling for scalar rewards in `[0, 1]` (with optional decay).
//! - `Exp3Ix`: seedable EXP3-IX for adversarial / fast-shifting scalar rewards in `[0, 1]` (with optional decay).
//! - `softmax_map`: stable score→probability helper for traffic splitting.
//! - (feature `contextual`) `LinUcb`: contextual bandit (linear UCB) for per-request routing using feature vectors.
//!
//! Non-goals:
//! - This is not a full bandit “platform” (no storage, OPE pipelines, dashboards, etc.).
//! - The `contextual` feature is intentionally a small, pragmatic policy module, not a full framework.

#![forbid(unsafe_code)]

use pare::{Direction, ParetoFrontier};
use std::collections::{BTreeMap, VecDeque};

mod decision;
pub use decision::*;

mod alloc;
pub use alloc::*;

mod guardrail;
pub use guardrail::*;

#[cfg(feature = "stochastic")]
mod exp3ix;
#[cfg(feature = "stochastic")]
pub use exp3ix::*;

#[cfg(feature = "stochastic")]
mod thompson;
#[cfg(feature = "stochastic")]
pub use thompson::*;

#[cfg(feature = "contextual")]
mod contextual;
#[cfg(feature = "contextual")]
pub use contextual::*;

mod sticky;
pub use sticky::*;

mod stable_hash;
pub use stable_hash::*;

mod novelty;
pub use novelty::*;

mod prior;
pub use prior::*;

mod worst_first;
pub use worst_first::*;

mod harness;
pub use harness::*;

/// Per-round details for multi-pick MAB selection with an external latency guardrail.
#[derive(Debug, Clone)]
pub struct MabKRound {
    /// Remaining arms at the start of the round.
    pub remaining: Vec<String>,
    /// Result of applying the latency guardrail to `remaining` for this round.
    pub guardrail: LatencyGuardrailDecision,
    /// MAB selection decision for this round (run on `guardrail.eligible`).
    pub mab: MabSelectionDecision,
}

/// Stop information for multi-pick MAB selection.
#[derive(Debug, Clone)]
pub struct MabKStop {
    /// Remaining arms at the stop point.
    pub remaining: Vec<String>,
    /// Result of applying the latency guardrail to `remaining` at the stop point.
    pub guardrail: LatencyGuardrailDecision,
}

/// Output of selecting up to `k` unique arms via repeated MAB selection, including a stop record
/// when the loop terminates early due to guardrail/emptiness.
#[derive(Debug, Clone)]
pub struct MabKExplain {
    /// Chosen arms (unique, in pick order).
    pub chosen: Vec<String>,
    /// Per-round details (one entry per chosen arm).
    pub rounds: Vec<MabKRound>,
    /// Present when the loop stopped early (e.g. guardrail `stop_early=true`).
    pub stop: Option<MabKStop>,
}

/// Output of selecting up to `k` unique arms via repeated MAB selection.
#[derive(Debug, Clone)]
pub struct MabKSelection {
    /// Chosen arms (unique, in pick order).
    pub chosen: Vec<String>,
    /// Per-round details (one entry per chosen arm).
    pub rounds: Vec<MabKRound>,
}

/// A compact, log-ready row for a single round of a multi-pick MAB selection.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MabKRoundLog {
    pub round: usize,
    pub remaining: Vec<String>,
    pub guardrail_eligible: Vec<String>,
    pub guardrail_fallback_used: bool,
    pub guardrail_stop_early: bool,
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub chosen: Option<String>,
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub explore_first: Option<bool>,
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub constraints_fallback_used: Option<bool>,
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub constraints_eligible_arms: Option<Vec<String>>,
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub top_candidates: Option<LogTopCandidates>,
}

/// Small “log-ready” top-candidate row, used to avoid duplicating score/prob formatting in harnesses.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LogTopCandidate {
    pub arm: String,
    /// Score used for sorting (meaning depends on policy).
    pub score: f64,
    /// Optional call count (present for MAB).
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub calls: Option<u64>,
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub ok_rate: Option<f64>,
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub junk_rate: Option<f64>,
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub hard_junk_rate: Option<f64>,
}

/// A typed top-candidates payload for logging.
///
/// `kind` describes the meaning of `rows[*].score`, e.g. `"mab_scalar"` or `"exp3ix_prob"`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LogTopCandidates {
    pub kind: String,
    pub rows: Vec<LogTopCandidate>,
}

pub const LOG_SCORE_KIND_MAB_SCALAR: &str = "mab_scalar";
pub const LOG_SCORE_KIND_EXP3IX_PROB: &str = "exp3ix_prob";
pub const MUXER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Extract top candidate rows from a MAB decision using the same scalarization as selection.
pub fn log_top_candidates_mab(decision: &MabSelectionDecision, top: usize) -> Vec<LogTopCandidate> {
    let cfg = decision.selection.config;
    let mut rows: Vec<(f64, &CandidateDebug)> = decision
        .selection
        .candidates
        .iter()
        .map(|c| {
            let score = c.objective_success
                - cfg.cost_weight * c.mean_cost_units
                - cfg.latency_weight * c.mean_elapsed_ms
                - cfg.hard_junk_weight * c.hard_junk_rate
                - cfg.junk_weight * c.soft_junk_rate;
            (score, c)
        })
        .collect();
    rows.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.name.cmp(&b.1.name)));
    rows.into_iter()
        .take(top.max(1))
        .map(|(score, c)| LogTopCandidate {
            arm: c.name.clone(),
            score,
            calls: Some(c.calls),
            ok_rate: Some(c.ok_rate),
            junk_rate: Some(c.junk_rate),
            hard_junk_rate: Some(c.hard_junk_rate),
        })
        .collect()
}

/// Extract top probability rows from an EXP3-IX decision.
pub fn log_top_candidates_exp3ix(decision: &Decision, top: usize) -> Vec<LogTopCandidate> {
    let Some(ref probs) = decision.probs else {
        return Vec::new();
    };
    let mut rows: Vec<(f64, String)> = probs.iter().map(|(k, v)| (*v, k.clone())).collect();
    rows.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    rows.into_iter()
        .take(top.max(1))
        .map(|(p, arm)| LogTopCandidate {
            arm,
            score: p,
            calls: None,
            ok_rate: None,
            junk_rate: None,
            hard_junk_rate: None,
        })
        .collect()
}

/// Extract top-candidates payload for MAB with a self-describing `kind`.
pub fn log_top_candidates_mab_typed(
    decision: &MabSelectionDecision,
    top: usize,
) -> LogTopCandidates {
    LogTopCandidates {
        kind: LOG_SCORE_KIND_MAB_SCALAR.to_string(),
        rows: log_top_candidates_mab(decision, top),
    }
}

/// Extract top-candidates payload for EXP3-IX with a self-describing `kind`.
pub fn log_top_candidates_exp3ix_typed(decision: &Decision, top: usize) -> LogTopCandidates {
    LogTopCandidates {
        kind: LOG_SCORE_KIND_EXP3IX_PROB.to_string(),
        rows: log_top_candidates_exp3ix(decision, top),
    }
}

/// A single observed outcome for an arm.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Outcome {
    /// Whether the request succeeded for this arm.
    pub ok: bool,
    /// Whether the downstream result was judged “low value” (e.g. blocked, empty extraction).
    pub junk: bool,
    /// Whether the junk was “hard” (e.g. JS/auth wall) vs “soft” (low-signal extraction).
    pub hard_junk: bool,
    /// Provider-specific cost units for this call (caller-defined).
    pub cost_units: u64,
    /// Total elapsed time for this call, in milliseconds.
    pub elapsed_ms: u64,
}

/// Sliding-window statistics for an arm.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Window {
    cap: usize,
    buf: VecDeque<Outcome>,
}

impl Window {
    /// Create an empty window with capacity `cap` (minimum 1).
    pub fn new(cap: usize) -> Self {
        Self {
            cap: cap.max(1),
            buf: VecDeque::new(),
        }
    }

    /// Maximum number of outcomes retained.
    pub fn cap(&self) -> usize {
        self.cap
    }

    /// Number of outcomes currently retained.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Whether the window has no outcomes.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Push a new outcome, evicting the oldest if at capacity.
    pub fn push(&mut self, o: Outcome) {
        if self.buf.len() == self.cap {
            self.buf.pop_front();
        }
        self.buf.push_back(o);
    }

    /// Best-effort: mutate the most recent outcome.
    ///
    /// This is useful when an outcome's "junk" label is only known after downstream processing
    /// (e.g. search succeeded, but extraction later looked low-signal).
    pub fn set_last_junk(&mut self, junk: bool) {
        if let Some(last) = self.buf.back_mut() {
            last.junk = junk;
        }
    }

    /// Best-effort: set “junk” and whether it is “hard junk” for the most recent outcome.
    pub fn set_last_junk_level(&mut self, junk: bool, hard_junk: bool) {
        if let Some(last) = self.buf.back_mut() {
            last.junk = junk;
            last.hard_junk = hard_junk && junk;
        }
    }

    /// Summarize the current window as counts and sums.
    pub fn summary(&self) -> Summary {
        let n = self.buf.len() as u64;
        if n == 0 {
            return Summary::default();
        }
        let mut ok = 0u64;
        let mut junk = 0u64;
        let mut hard_junk = 0u64;
        let mut cost_units = 0u64;
        let mut elapsed_ms_sum = 0u64;
        for o in &self.buf {
            ok += o.ok as u64;
            junk += o.junk as u64;
            hard_junk += o.hard_junk as u64;
            cost_units = cost_units.saturating_add(o.cost_units);
            elapsed_ms_sum = elapsed_ms_sum.saturating_add(o.elapsed_ms);
        }
        Summary {
            calls: n,
            ok,
            junk,
            hard_junk,
            cost_units,
            elapsed_ms_sum,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
/// Aggregate counts/sums over a window of outcomes.
pub struct Summary {
    /// Number of calls observed.
    pub calls: u64,
    /// Number of successful calls.
    pub ok: u64,
    /// Number of calls judged “junk”.
    pub junk: u64,
    /// Number of calls judged “hard junk”.
    pub hard_junk: u64,
    /// Sum of `cost_units` over calls.
    pub cost_units: u64,
    /// Sum of `elapsed_ms` over calls.
    pub elapsed_ms_sum: u64,
}

impl Summary {
    /// Fraction of calls that succeeded.
    pub fn ok_rate(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            (self.ok as f64) / (self.calls as f64)
        }
    }

    /// Fraction of calls that were judged “junk”.
    pub fn junk_rate(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            (self.junk as f64) / (self.calls as f64)
        }
    }

    /// Mean `cost_units` per call.
    pub fn mean_cost_units(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            (self.cost_units as f64) / (self.calls as f64)
        }
    }

    /// Mean `elapsed_ms` per call.
    pub fn mean_elapsed_ms(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            (self.elapsed_ms_sum as f64) / (self.calls as f64)
        }
    }

    /// Fraction of calls that were judged “hard junk”.
    pub fn hard_junk_rate(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            (self.hard_junk as f64) / (self.calls as f64)
        }
    }

    /// Fraction of calls that were judged “soft junk” (junk but not hard junk).
    pub fn soft_junk_rate(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            let soft = self.junk.saturating_sub(self.hard_junk);
            (soft as f64) / (self.calls as f64)
        }
    }
}

/// Configuration knobs for deterministic MAB-style selection.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MabConfig {
    /// UCB exploration coefficient.
    pub exploration_c: f64,
    /// Penalty weight for mean cost_units (0 disables).
    pub cost_weight: f64,
    /// Penalty weight for mean latency in ms (0 disables).
    pub latency_weight: f64,
    /// Penalty weight for junk_rate (0 disables).
    pub junk_weight: f64,
    /// Penalty weight for hard_junk_rate (0 disables).
    pub hard_junk_weight: f64,
    /// Optional constraint: discard arms whose windowed junk_rate exceeds this.
    pub max_junk_rate: Option<f64>,
    /// Optional constraint: discard arms whose windowed hard_junk_rate exceeds this.
    pub max_hard_junk_rate: Option<f64>,
    /// Optional constraint: discard arms whose windowed mean_cost_units exceeds this.
    pub max_mean_cost_units: Option<f64>,
}

impl Default for MabConfig {
    fn default() -> Self {
        Self {
            exploration_c: 0.7,
            cost_weight: 0.0,
            latency_weight: 0.0,
            junk_weight: 0.0,
            hard_junk_weight: 0.0,
            max_junk_rate: None,
            max_hard_junk_rate: None,
            max_mean_cost_units: None,
        }
    }
}

/// Debug row for one candidate arm.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CandidateDebug {
    /// Arm name.
    pub name: String,
    /// Calls observed in the summary.
    pub calls: u64,
    /// Successes observed in the summary.
    pub ok: u64,
    /// Junk outcomes observed in the summary.
    pub junk: u64,
    /// Hard junk outcomes observed in the summary.
    pub hard_junk: u64,
    /// Success rate.
    pub ok_rate: f64,
    /// Junk rate.
    pub junk_rate: f64,
    /// Hard junk rate.
    pub hard_junk_rate: f64,
    /// Soft junk rate.
    pub soft_junk_rate: f64,
    /// Mean cost units per call.
    pub mean_cost_units: f64,
    /// Mean latency (ms) per call.
    pub mean_elapsed_ms: f64,
    /// UCB exploration term.
    pub ucb: f64,
    /// Scalar objective used for the frontier’s success dimension (junk is a separate objective).
    pub objective_success: f64,
}

/// Output of `select_mab` (chosen arm + debugging context).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Selection {
    /// The selected arm.
    pub chosen: String,
    /// Arm names on the Pareto frontier (in the frontier’s iteration order).
    pub frontier: Vec<String>,
    /// Candidate debug rows (in the input arm order).
    pub candidates: Vec<CandidateDebug>,
    /// The config used to compute this selection.
    pub config: MabConfig,
}

/// Additional metadata for a deterministic `select_mab` decision.
///
/// This exists because production routers typically need more than "which arm":
/// they also need to know whether constraints eliminated all arms (fallback) and
/// whether the decision was explore-first.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MabSelectionDecision {
    /// The base deterministic selection output (chosen arm + debug rows).
    pub selection: Selection,
    /// Arms that were eligible after applying constraints.
    ///
    /// If constraints eliminated all arms, this is equal to the original `arms_in_order`.
    pub eligible_arms: Vec<String>,
    /// True if constraints eliminated all arms and the selector fell back to the original set.
    pub constraints_fallback_used: bool,
    /// True if the selector chose an arm due to explore-first (some arm had `calls == 0`).
    pub explore_first: bool,
}

/// Deterministic selection:
/// - Explore each arm at least once (in stable order).
/// - Then:
///   - build a Pareto frontier over:
///     - maximize success (ok_rate, plus UCB)
///     - minimize mean cost_units
///     - minimize mean latency
///     - minimize junk_rate
///   - pick the best scalarized point (with stable tie-break)
///
/// # Example
///
/// ```rust
/// use muxer::{select_mab, MabConfig, Summary};
/// use std::collections::BTreeMap;
///
/// let arms = vec!["a".to_string(), "b".to_string()];
/// let mut summaries = BTreeMap::new();
/// summaries.insert(
///     "a".to_string(),
///     Summary { calls: 10, ok: 9, junk: 0, hard_junk: 0, cost_units: 10, elapsed_ms_sum: 900 }
/// );
/// summaries.insert(
///     "b".to_string(),
///     Summary { calls: 10, ok: 9, junk: 2, hard_junk: 0, cost_units: 10, elapsed_ms_sum: 900 }
/// );
///
/// let sel = select_mab(&arms, &summaries, MabConfig::default());
/// assert_eq!(sel.chosen, "a");
/// ```
pub fn select_mab(
    arms_in_order: &[String],
    summaries: &BTreeMap<String, Summary>,
    cfg: MabConfig,
) -> Selection {
    select_mab_explain(arms_in_order, summaries, cfg).selection
}

/// Like `select_mab`, but also returns metadata about constraints and explore-first behavior.
pub fn select_mab_explain(
    arms_in_order: &[String],
    summaries: &BTreeMap<String, Summary>,
    cfg: MabConfig,
) -> MabSelectionDecision {
    // Apply hard constraints (BwK-ish “anytime” gating).
    // If constraints filter everything, fall back to the original arm set (never return empty).
    let mut eligible: Vec<String> = Vec::new();
    for a in arms_in_order {
        let s = summaries.get(a).copied().unwrap_or_default();
        let ok = cfg
            .max_junk_rate
            .map(|thr| s.junk_rate() <= thr)
            .unwrap_or(true)
            && cfg
                .max_hard_junk_rate
                .map(|thr| s.hard_junk_rate() <= thr)
                .unwrap_or(true)
            && cfg
                .max_mean_cost_units
                .map(|thr| s.mean_cost_units() <= thr)
                .unwrap_or(true);
        if ok {
            eligible.push(a.clone());
        }
    }
    let constraints_fallback_used = eligible.is_empty();
    let eligible_arms: Vec<String> = if constraints_fallback_used {
        arms_in_order.to_vec()
    } else {
        eligible.clone()
    };
    let arms_in_order: &[String] = if constraints_fallback_used {
        arms_in_order
    } else {
        &eligible
    };

    // Explore first.
    for a in arms_in_order {
        if summaries.get(a).copied().unwrap_or_default().calls == 0 {
            let sel = Selection {
                chosen: a.clone(),
                frontier: vec![a.clone()],
                candidates: vec![CandidateDebug {
                    name: a.clone(),
                    calls: 0,
                    ok: 0,
                    junk: 0,
                    hard_junk: 0,
                    ok_rate: 0.0,
                    junk_rate: 0.0,
                    hard_junk_rate: 0.0,
                    soft_junk_rate: 0.0,
                    mean_cost_units: 0.0,
                    mean_elapsed_ms: 0.0,
                    ucb: 0.0,
                    objective_success: 0.0,
                }],
                config: cfg,
            };
            return MabSelectionDecision {
                selection: sel,
                eligible_arms,
                constraints_fallback_used,
                explore_first: true,
            };
        }
    }

    // Only count calls for the arms actually being considered (important if `summaries`
    // contains additional entries beyond `arms_in_order`).
    let total_calls: f64 = arms_in_order
        .iter()
        .map(|a| summaries.get(a).copied().unwrap_or_default().calls as f64)
        .sum::<f64>()
        .max(1.0);

    // Pareto frontier (maximize space):
    // - objective_success: maximize
    // - costs/latency/junk: minimize => negate
    //
    // We keep the frontier computation separate from scalarization so callers can
    // inspect "who is on the frontier" deterministically.
    let mut frontier_points: Vec<Vec<f64>> = Vec::new();
    let mut frontier_names_in_order: Vec<String> = Vec::new();

    let mut candidates = Vec::new();
    for a in arms_in_order {
        let s = summaries.get(a).copied().unwrap_or_default();
        let n = (s.calls as f64).max(1.0);
        let ok_rate = s.ok_rate();
        let junk_rate = s.junk_rate();
        let hard_junk_rate = s.hard_junk_rate();
        let soft_junk_rate = s.soft_junk_rate();

        let ucb = cfg.exploration_c * ((total_calls.ln() / n).sqrt());
        let objective_success = ok_rate + ucb;

        let mean_cost = s.mean_cost_units();
        let mean_lat = s.mean_elapsed_ms();

        candidates.push(CandidateDebug {
            name: a.clone(),
            calls: s.calls,
            ok: s.ok,
            junk: s.junk,
            hard_junk: s.hard_junk,
            ok_rate,
            junk_rate,
            hard_junk_rate,
            soft_junk_rate,
            mean_cost_units: mean_cost,
            mean_elapsed_ms: mean_lat,
            ucb,
            objective_success,
        });

        frontier_points.push(vec![
            objective_success,
            -mean_cost,
            -mean_lat,
            -hard_junk_rate,
            -soft_junk_rate,
        ]);
        frontier_names_in_order.push(a.clone());
    }

    let mut frontier = ParetoFrontier::new(vec![Direction::Maximize; 5]);
    for (i, vals) in frontier_points.iter().enumerate() {
        frontier.push(vals.clone(), i);
    }
    let mut frontier_indices: Vec<usize> = if frontier.is_empty() {
        (0..frontier_points.len()).collect()
    } else {
        frontier.points().iter().map(|p| p.data).collect()
    };
    // Ensure stable ordering regardless of frontier internals.
    frontier_indices.sort_unstable();

    let frontier_names: Vec<String> = frontier_indices
        .iter()
        .filter_map(|&i| frontier_names_in_order.get(i).cloned())
        .collect();

    // Deterministic scalarization among frontier points.
    let weights: [f64; 5] = [
        1.0,
        cfg.cost_weight.max(0.0),
        cfg.latency_weight.max(0.0),
        cfg.hard_junk_weight.max(0.0),
        cfg.junk_weight.max(0.0),
    ];
    let mut best_name = frontier_names
        .first()
        .cloned()
        .unwrap_or_else(|| arms_in_order.first().cloned().unwrap_or_default());
    let mut best_score = f64::NEG_INFINITY;
    for &idx in &frontier_indices {
        let Some(c) = candidates.get(idx) else {
            continue;
        };
        // Maximize:
        //   objective_success - w_cost * cost - w_lat * latency - w_hard * hard_junk - w_soft * soft_junk
        let s = c.objective_success
            - weights[1] * c.mean_cost_units
            - weights[2] * c.mean_elapsed_ms
            - weights[3] * c.hard_junk_rate
            - weights[4] * c.soft_junk_rate;
        if s > best_score || ((s - best_score).abs() <= 1e-12 && c.name < best_name) {
            best_score = s;
            best_name = c.name.clone();
        }
    }

    let sel = Selection {
        chosen: best_name,
        frontier: frontier_names,
        candidates,
        config: cfg,
    };
    MabSelectionDecision {
        selection: sel,
        eligible_arms,
        constraints_fallback_used,
        explore_first: false,
    }
}

/// Unified decision envelope for deterministic MAB selection.
///
/// This is a convenience wrapper around `select_mab_explain` that returns a `Decision`
/// suitable for consistent logging/replay across policies.
pub fn select_mab_decide(
    arms_in_order: &[String],
    summaries: &BTreeMap<String, Summary>,
    cfg: MabConfig,
) -> Decision {
    let d = select_mab_explain(arms_in_order, summaries, cfg);
    let mut notes = vec![DecisionNote::Constraints {
        eligible_arms: d.eligible_arms.clone(),
        fallback_used: d.constraints_fallback_used,
    }];
    if d.explore_first {
        notes.push(DecisionNote::ExploreFirst);
    } else {
        notes.push(DecisionNote::DeterministicChoice);
    }
    Decision {
        policy: DecisionPolicy::Mab,
        chosen: d.selection.chosen.clone(),
        probs: None,
        notes,
    }
}

/// Select up to `k` unique arms by repeatedly applying `select_mab_explain`, with an optional
/// external latency guardrail applied each round.
///
/// This is intended for “thin harnesses” that want consistent guardrail semantics without
/// re-implementing the multi-pick loop.
///
/// The `summaries_for` callback is invoked once per round with the *current* remaining arm set.
pub fn select_mab_k_guardrailed_explain<F>(
    arms_in_order: &[String],
    mut summaries_for: F,
    cfg: MabConfig,
    guardrail: LatencyGuardrailConfig,
    k: usize,
) -> MabKSelection
where
    F: FnMut(&[String]) -> BTreeMap<String, Summary>,
{
    if arms_in_order.is_empty() || k == 0 {
        return MabKSelection {
            chosen: Vec::new(),
            rounds: Vec::new(),
        };
    }

    let mut remaining: Vec<String> = arms_in_order.to_vec();
    let mut chosen: Vec<String> = Vec::new();
    let mut rounds: Vec<MabKRound> = Vec::new();

    for _round in 0..k.min(remaining.len()) {
        let remaining_in = remaining.clone();
        let summaries = summaries_for(&remaining_in);

        let gd = apply_latency_guardrail(&remaining_in, &summaries, guardrail, chosen.len());
        if gd.stop_early {
            break;
        }
        if gd.eligible.is_empty() {
            break;
        }

        let d = select_mab_explain(&gd.eligible, &summaries, cfg);
        let pick = d.selection.chosen.clone();
        chosen.push(pick.clone());
        remaining.retain(|b| b != &pick);

        rounds.push(MabKRound {
            remaining: remaining_in,
            guardrail: gd,
            mab: d,
        });
    }

    MabKSelection { chosen, rounds }
}

/// Like `select_mab_k_guardrailed_explain`, but also returns a stop record when the selection
/// ends early due to guardrail semantics or an empty eligible set.
pub fn select_mab_k_guardrailed_explain_full<F>(
    arms_in_order: &[String],
    mut summaries_for: F,
    cfg: MabConfig,
    guardrail: LatencyGuardrailConfig,
    k: usize,
) -> MabKExplain
where
    F: FnMut(&[String]) -> BTreeMap<String, Summary>,
{
    if arms_in_order.is_empty() || k == 0 {
        return MabKExplain {
            chosen: Vec::new(),
            rounds: Vec::new(),
            stop: None,
        };
    }

    let mut remaining: Vec<String> = arms_in_order.to_vec();
    let mut chosen: Vec<String> = Vec::new();
    let mut rounds: Vec<MabKRound> = Vec::new();
    let mut stop: Option<MabKStop> = None;

    for _round in 0..k.min(remaining.len()) {
        let remaining_in = remaining.clone();
        let summaries = summaries_for(&remaining_in);

        let gd = apply_latency_guardrail(&remaining_in, &summaries, guardrail, chosen.len());
        if gd.stop_early || gd.eligible.is_empty() {
            stop = Some(MabKStop {
                remaining: remaining_in,
                guardrail: gd,
            });
            break;
        }

        let d = select_mab_explain(&gd.eligible, &summaries, cfg);
        let pick = d.selection.chosen.clone();
        chosen.push(pick.clone());
        remaining.retain(|b| b != &pick);

        rounds.push(MabKRound {
            remaining: remaining_in,
            guardrail: gd,
            mab: d,
        });
    }

    MabKExplain {
        chosen,
        rounds,
        stop,
    }
}

/// Convert a multi-pick MAB explanation into compact, log-ready round rows.
///
/// This includes an additional “stop row” when selection ends early (e.g. guardrail `stop_early=true`).
pub fn log_mab_k_rounds_typed(explain: &MabKExplain, top: usize) -> Vec<MabKRoundLog> {
    let mut out: Vec<MabKRoundLog> = Vec::new();

    for (i, r) in explain.rounds.iter().enumerate() {
        let d = &r.mab;
        out.push(MabKRoundLog {
            round: i + 1,
            remaining: r.remaining.clone(),
            guardrail_eligible: r.guardrail.eligible.clone(),
            guardrail_fallback_used: r.guardrail.fallback_used,
            guardrail_stop_early: r.guardrail.stop_early,
            chosen: Some(d.selection.chosen.clone()),
            explore_first: Some(d.explore_first),
            constraints_fallback_used: Some(d.constraints_fallback_used),
            constraints_eligible_arms: Some(d.eligible_arms.clone()),
            top_candidates: Some(log_top_candidates_mab_typed(d, top)),
        });
    }

    if let Some(ref s) = explain.stop {
        out.push(MabKRoundLog {
            round: explain.rounds.len() + 1,
            remaining: s.remaining.clone(),
            guardrail_eligible: s.guardrail.eligible.clone(),
            guardrail_fallback_used: s.guardrail.fallback_used,
            guardrail_stop_early: s.guardrail.stop_early,
            chosen: None,
            explore_first: None,
            constraints_fallback_used: None,
            constraints_eligible_arms: None,
            top_candidates: None,
        });
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn s(
        calls: u64,
        ok: u64,
        junk: u64,
        hard_junk: u64,
        cost_units: u64,
        elapsed_ms_sum: u64,
    ) -> Summary {
        Summary {
            calls,
            ok,
            junk,
            hard_junk,
            cost_units,
            elapsed_ms_sum,
        }
    }

    #[test]
    fn select_mab_is_deterministic_and_prefers_lower_junk_all_else_equal() {
        let arms = vec!["a".to_string(), "b".to_string()];
        let mut m = BTreeMap::new();
        // Same ok/cost/lat, but different junk.
        m.insert("a".to_string(), s(10, 9, 5, 0, 10, 1000));
        m.insert("b".to_string(), s(10, 9, 0, 0, 10, 1000));

        let sel1 = select_mab(&arms, &m, MabConfig::default());
        let sel2 = select_mab(&arms, &m, MabConfig::default());
        assert_eq!(sel1.chosen, "b");
        assert_eq!(sel1.chosen, sel2.chosen);
    }

    #[test]
    fn constraints_filter_arms_but_never_return_empty() {
        let arms = vec!["a".to_string(), "b".to_string()];
        let mut m = BTreeMap::new();
        m.insert("a".to_string(), s(10, 9, 9, 0, 10, 1000));
        m.insert("b".to_string(), s(10, 9, 9, 0, 10, 1000));

        let cfg = MabConfig {
            max_junk_rate: Some(0.1),
            ..MabConfig::default()
        };
        let sel = select_mab(&arms, &m, cfg);
        assert!(!sel.chosen.is_empty());
        assert!(sel.frontier.iter().any(|x| x == &sel.chosen));
    }

    #[test]
    fn constraints_can_exclude_high_hard_junk_arm() {
        let arms = vec!["a".to_string(), "b".to_string()];
        let mut m = BTreeMap::new();
        m.insert("a".to_string(), s(10, 9, 1, 1, 10, 1000));
        m.insert("b".to_string(), s(10, 9, 1, 0, 10, 1000));

        let cfg = MabConfig {
            max_hard_junk_rate: Some(0.05),
            ..MabConfig::default()
        };
        let sel = select_mab(&arms, &m, cfg);
        assert_eq!(sel.chosen, "b");
    }

    #[test]
    fn mab_k_full_includes_guardrail_stop_record() {
        let arms = vec!["a".to_string(), "b".to_string()];
        let cfg = MabConfig::default();
        let guardrail = LatencyGuardrailConfig {
            max_mean_ms: Some(10.0),
            require_measured: false,
            allow_fewer: true,
        };

        // Both arms are "slow" so guardrail will:
        // - fallback on round 1 (already_chosen=0)
        // - stop early on round 2 (already_chosen>0)
        let mut all = BTreeMap::new();
        all.insert(
            "a".to_string(),
            Summary {
                calls: 1,
                ok: 1,
                junk: 0,
                hard_junk: 0,
                cost_units: 0,
                elapsed_ms_sum: 1000,
            },
        );
        all.insert(
            "b".to_string(),
            Summary {
                calls: 1,
                ok: 1,
                junk: 0,
                hard_junk: 0,
                cost_units: 0,
                elapsed_ms_sum: 1000,
            },
        );

        let ex = select_mab_k_guardrailed_explain_full(
            &arms,
            |_remaining| all.clone(),
            cfg,
            guardrail,
            2,
        );
        assert_eq!(ex.chosen.len(), 1);
        assert!(ex.stop.is_some());
        assert!(ex.stop.as_ref().unwrap().guardrail.stop_early);

        let logs = log_mab_k_rounds_typed(&ex, 3);
        assert_eq!(logs.len(), 2);
        assert!(logs[1].guardrail_stop_early);
        assert!(logs[1].chosen.is_none());
    }

    proptest! {
        #[test]
        fn select_mab_never_panics_and_returns_member_of_arms(
            // Keep this intentionally small/bounded to avoid slow tests.
            calls_a in 0u64..50,
            calls_b in 0u64..50,
            ok_a in 0u64..50,
            ok_b in 0u64..50,
            junk_a in 0u64..50,
            junk_b in 0u64..50,
            hard_a in 0u64..50,
            hard_b in 0u64..50,
            cost_a in 0u64..500,
            cost_b in 0u64..500,
            lat_a in 0u64..50_000,
            lat_b in 0u64..50_000,
        ) {
            let arms = vec!["a".to_string(), "b".to_string()];
            let mut m = BTreeMap::new();

            // Sanitize counts so they never exceed calls.
            let sa = s(
                calls_a,
                ok_a.min(calls_a),
                junk_a.min(calls_a),
                hard_a.min(junk_a.min(calls_a)),
                cost_a,
                lat_a,
            );
            let sb = s(
                calls_b,
                ok_b.min(calls_b),
                junk_b.min(calls_b),
                hard_b.min(junk_b.min(calls_b)),
                cost_b,
                lat_b,
            );
            m.insert("a".to_string(), sa);
            m.insert("b".to_string(), sb);

            let cfg = MabConfig {
                exploration_c: 0.7,
                cost_weight: 0.0,
                latency_weight: 0.0,
                junk_weight: 0.0,
                hard_junk_weight: 0.0,
                max_junk_rate: None,
                max_hard_junk_rate: None,
                max_mean_cost_units: None,
            };

            let sel = select_mab(&arms, &m, cfg);
            prop_assert!(sel.chosen == "a" || sel.chosen == "b");
            prop_assert!(sel.frontier.iter().any(|x| x == &sel.chosen));

            // Determinism: same input -> same output.
            let sel2 = select_mab(&arms, &m, cfg);
            prop_assert_eq!(sel.chosen, sel2.chosen);
        }

        #[test]
        fn select_mab_ignores_summaries_for_unknown_arms(
            calls_a in 1u64..50,
            calls_b in 1u64..50,
            ok_a in 0u64..50,
            ok_b in 0u64..50,
            junk_a in 0u64..50,
            junk_b in 0u64..50,
            hard_a in 0u64..50,
            hard_b in 0u64..50,
            cost_a in 0u64..500,
            cost_b in 0u64..500,
            lat_a in 0u64..50_000,
            lat_b in 0u64..50_000,
            extra_calls in 0u64..50,
            extra_ok in 0u64..50,
            extra_junk in 0u64..50,
            extra_hard in 0u64..50,
            extra_cost in 0u64..500,
            extra_lat in 0u64..50_000,
        ) {
            let arms = vec!["a".to_string(), "b".to_string()];
            let mut m = BTreeMap::new();

            let sa = s(
                calls_a,
                ok_a.min(calls_a),
                junk_a.min(calls_a),
                hard_a.min(junk_a.min(calls_a)),
                cost_a,
                lat_a,
            );
            let sb = s(
                calls_b,
                ok_b.min(calls_b),
                junk_b.min(calls_b),
                hard_b.min(junk_b.min(calls_b)),
                cost_b,
                lat_b,
            );
            m.insert("a".to_string(), sa);
            m.insert("b".to_string(), sb);

            let cfg = MabConfig::default();
            let sel1 = select_mab(&arms, &m, cfg);

            // Add an irrelevant arm to the summaries map.
            let sx = s(
                extra_calls,
                extra_ok.min(extra_calls),
                extra_junk.min(extra_calls),
                extra_hard.min(extra_junk.min(extra_calls)),
                extra_cost,
                extra_lat,
            );
            m.insert("zzz-extra".to_string(), sx);

            let sel2 = select_mab(&arms, &m, cfg);
            prop_assert_eq!(sel1.chosen, sel2.chosen);
        }

        #[test]
        fn select_mab_explores_first_zero_call_arm(
            // Ensure we have at least one unexplored arm.
            calls_a in 0u64..10,
            calls_b in 0u64..10,
            calls_c in 0u64..10,
        ) {
            let arms = vec!["a".to_string(), "b".to_string(), "c".to_string()];
            let mut m = BTreeMap::new();
            m.insert("a".to_string(), s(calls_a, 0, 0, 0, 0, 0));
            m.insert("b".to_string(), s(calls_b, 0, 0, 0, 0, 0));
            m.insert("c".to_string(), s(calls_c, 0, 0, 0, 0, 0));

            // Find first index with calls == 0.
            let expected = if calls_a == 0 {
                "a"
            } else if calls_b == 0 {
                "b"
            } else if calls_c == 0 {
                "c"
            } else {
                // No zero-call arms -> skip (property vacuously holds).
                return Ok(());
            };

            let sel = select_mab(&arms, &m, MabConfig::default());
            prop_assert_eq!(sel.chosen, expected);
        }
    }

    #[test]
    fn sticky_mab_respects_min_dwell() {
        let arms = vec!["a".to_string(), "b".to_string()];
        let cfg = MabConfig::default();

        let mut sticky = StickyMab::new(StickyConfig {
            min_dwell: 3,
            min_switch_margin: 0.0,
        });

        // First: pick "a".
        let mut m1 = BTreeMap::new();
        m1.insert("a".to_string(), s(10, 10, 0, 0, 0, 0));
        m1.insert("b".to_string(), s(10, 5, 0, 0, 0, 0));
        let e1 = sticky.apply_mab(select_mab_explain(&arms, &m1, cfg));
        assert_eq!(e1.selection.chosen, "a");
        assert_eq!(sticky.dwell(), 1);

        // Now "b" is better, but dwell gate should keep "a" for 2 more decisions.
        let mut m2 = BTreeMap::new();
        m2.insert("a".to_string(), s(10, 5, 0, 0, 0, 0));
        m2.insert("b".to_string(), s(10, 10, 0, 0, 0, 0));

        let e2 = sticky.apply_mab(select_mab_explain(&arms, &m2, cfg));
        assert_eq!(e2.selection.chosen, "a");
        let e3 = sticky.apply_mab(select_mab_explain(&arms, &m2, cfg));
        assert_eq!(e3.selection.chosen, "a");

        // Next decision: allowed to switch.
        let e4 = sticky.apply_mab(select_mab_explain(&arms, &m2, cfg));
        assert_eq!(e4.selection.chosen, "b");
        assert_eq!(sticky.dwell(), 1);
    }

    #[test]
    fn sticky_mab_respects_min_switch_margin() {
        // Construct a base Selection directly so we can control scalar scores precisely.
        let cfg = MabConfig::default();
        let mut sticky = StickyMab::new(StickyConfig {
            min_dwell: 0,
            min_switch_margin: 0.5,
        });

        let mk = |chosen: &str, a_score: f64, b_score: f64| -> MabSelectionDecision {
            MabSelectionDecision {
                selection: Selection {
                    chosen: chosen.to_string(),
                    frontier: vec!["a".to_string(), "b".to_string()],
                    candidates: vec![
                        CandidateDebug {
                            name: "a".to_string(),
                            calls: 10,
                            ok: 0,
                            junk: 0,
                            hard_junk: 0,
                            ok_rate: 0.0,
                            junk_rate: 0.0,
                            hard_junk_rate: 0.0,
                            soft_junk_rate: 0.0,
                            mean_cost_units: 0.0,
                            mean_elapsed_ms: 0.0,
                            ucb: 0.0,
                            objective_success: a_score,
                        },
                        CandidateDebug {
                            name: "b".to_string(),
                            calls: 10,
                            ok: 0,
                            junk: 0,
                            hard_junk: 0,
                            ok_rate: 0.0,
                            junk_rate: 0.0,
                            hard_junk_rate: 0.0,
                            soft_junk_rate: 0.0,
                            mean_cost_units: 0.0,
                            mean_elapsed_ms: 0.0,
                            ucb: 0.0,
                            objective_success: b_score,
                        },
                    ],
                    config: cfg,
                },
                eligible_arms: vec!["a".to_string(), "b".to_string()],
                constraints_fallback_used: false,
                explore_first: false,
            }
        };

        // Start on "a".
        let e1 = sticky.apply_mab(mk("a", 1.0, 1.0));
        assert_eq!(e1.selection.chosen, "a");
        assert_eq!(sticky.previous(), Some("a"));

        // Candidate "b" is only slightly better: margin < 0.5 => keep "a".
        let e2 = sticky.apply_mab(mk("b", 1.0, 1.4));
        assert_eq!(e2.selection.chosen, "a");

        // Candidate "b" is much better: margin >= 0.5 => switch to "b".
        let e3 = sticky.apply_mab(mk("b", 1.0, 1.7));
        assert_eq!(e3.selection.chosen, "b");
        assert_eq!(sticky.previous(), Some("b"));
    }

    #[test]
    fn sticky_mab_follows_base_choice_if_previous_missing_from_candidates() {
        let cfg = MabConfig::default();
        let mut sticky = StickyMab::new(StickyConfig {
            min_dwell: 10,
            min_switch_margin: 100.0,
        });

        // Set a previous arm that won't appear.
        sticky.apply_mab(MabSelectionDecision {
            selection: Selection {
                chosen: "old".to_string(),
                frontier: vec!["old".to_string()],
                candidates: vec![CandidateDebug {
                    name: "old".to_string(),
                    calls: 0,
                    ok: 0,
                    junk: 0,
                    hard_junk: 0,
                    ok_rate: 0.0,
                    junk_rate: 0.0,
                    hard_junk_rate: 0.0,
                    soft_junk_rate: 0.0,
                    mean_cost_units: 0.0,
                    mean_elapsed_ms: 0.0,
                    ucb: 0.0,
                    objective_success: 0.0,
                }],
                config: cfg,
            },
            eligible_arms: vec!["old".to_string()],
            constraints_fallback_used: false,
            explore_first: true,
        });
        assert_eq!(sticky.previous(), Some("old"));

        // Now candidates don't include "old" => stickiness must not force an unavailable arm.
        let base = Selection {
            chosen: "a".to_string(),
            frontier: vec!["a".to_string()],
            candidates: vec![CandidateDebug {
                name: "a".to_string(),
                calls: 10,
                ok: 0,
                junk: 0,
                hard_junk: 0,
                ok_rate: 0.0,
                junk_rate: 0.0,
                hard_junk_rate: 0.0,
                soft_junk_rate: 0.0,
                mean_cost_units: 0.0,
                mean_elapsed_ms: 0.0,
                ucb: 0.0,
                objective_success: 0.0,
            }],
            config: cfg,
        };
        let e = sticky.apply_mab(MabSelectionDecision {
            selection: base,
            eligible_arms: vec!["a".to_string()],
            constraints_fallback_used: false,
            explore_first: false,
        });
        assert_eq!(e.selection.chosen, "a");
        assert_eq!(sticky.previous(), Some("a"));
    }

    #[test]
    fn select_mab_chosen_satisfies_constraints_when_eligible_exists() {
        // If at least one arm passes constraints, we should never return a violating arm.
        let arms = vec!["a".to_string(), "b".to_string()];
        let mut m = BTreeMap::new();

        // Arm "a" violates junk constraint, arm "b" is fine.
        m.insert("a".to_string(), s(100, 90, 80, 0, 10, 1000));
        m.insert("b".to_string(), s(100, 90, 0, 0, 10, 1000));

        let cfg = MabConfig {
            max_junk_rate: Some(0.1),
            ..MabConfig::default()
        };

        let sel = select_mab(&arms, &m, cfg);
        assert_eq!(sel.chosen, "b");

        // Sanity: chosen meets constraints.
        let s = m.get(&sel.chosen).copied().unwrap_or_default();
        assert!(s.junk_rate() <= 0.1);
    }
}
