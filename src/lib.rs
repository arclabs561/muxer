//! `muxer`: deterministic, multi-objective bandit-style routing primitives.
//!
//! Designed for “arm selection” problems: you have a small set of arms
//! (model versions, inference endpoints, backends, data sources — anything
//! you choose between repeatedly) and calls where you evaluate quality after
//! the fact.  You define what constitutes ok, degraded, or broken for your
//! domain; `muxer` tracks the rates and routes accordingly.
//!
//! An [`Outcome`] carries three caller-defined quality fields:
//! - `ok`: the call produced a usable result.
//! - `junk`: quality was below your threshold (`hard_junk=true` implies `junk=true`).
//! - `hard_junk`: the call failed entirely — a harsher subset of junk, penalized separately.
//! - `quality_score: Option<f64>`: optional continuous quality signal `[0,1]` (higher = better).
//!   Supplements the binary flags without changing routing semantics.
//!
//! Plus `cost_units` (caller-defined cost proxy) and `elapsed_ms` (wall-clock time).
//!
//! **Goals:**
//! - **Deterministic by default**: same stats + config → same choice.
//! - **Non-stationarity friendly**: sliding-window summaries, not lifetime averages.
//! - **Multi-objective**: Pareto frontier over ok/junk/cost/latency, then scalarization.
//! - **Small K**: designed for 2–10 arms; not intended for K in the hundreds.
//!
//! **Selection policies:**
//! - [`select_mab`] / [`select_mab_explain`] / [`select_mab_monitored_explain`]:
//!   deterministic Pareto + scalarization.  [`MabConfig::quality_weight`] > 0
//!   incorporates the `quality_score` gradient into `objective_success`.
//! - [`ThompsonSampling`]: seedable Thompson sampling for scalar rewards in `[0, 1]`.
//! - [`Exp3Ix`]: seedable EXP3-IX for adversarial / fast-shifting rewards.
//! - [`BanditPolicy`] (feature `stochastic`): common `decide`/`update_reward` trait
//!   for `ThompsonSampling` and `Exp3Ix` — enables generic policy code.
//! - [`softmax_map`]: stable score → probability helper for traffic splitting.
//! - (feature `contextual`) [`LinUcb`]: linear contextual bandit.
//!
//! **Operational primitives:**
//! - [`TriageSession`]: detect-then-triage — per-arm CUSUM detection wired to
//!   per-cell (`arm × context-bin`) investigation.
//! - [`WorstFirstConfig`] / [`worst_first_pick_k`]: post-detection investigation routing.
//! - [`CoverageConfig`] / [`coverage_pick_under_sampled`]: maintenance sampling floor.
//! - [`LatencyGuardrailConfig`] / [`apply_latency_guardrail`]: hard pre-filter by mean latency.
//! - [`PipelineOrder`] / [`policy_plan_generic`] / [`policy_fill_generic`]: harness glue.
//!
//! **Non-goals:**
//! - Not a full bandit platform (no storage, OPE pipelines, dashboards).
//! - `contextual` is intentionally a small, pragmatic policy module.
//!
//!
//! # The Three Objectives and the Objective Manifold
//!
//! Every routing decision simultaneously serves three purposes:
//!
//! 1. **Exploitation** (regret minimization): route to the best arm now.
//! 2. **Estimation** (learning): understand how each arm performs across conditions.
//! 3. **Detection/triage** (monitoring): notice when an arm breaks, then investigate.
//!
//! These map to the modules of this crate:
//!
//! - **Selection** (`select_mab`, `Exp3Ix`, `ThompsonSampling`): objectives 1+2.
//! - **Monitoring** (`monitor`): objective 3 -- drift/catKL/CUSUM are the detection
//!   surface of the design measure.
//! - **Triage** (`worst_first`): active investigation after detection fires --
//!   prioritize historically broken arms to characterize the change.
//!
//! ## The non-contextual collapse (static schedules)
//!
//! For **fixed (non-adaptive) allocation schedules** with K arms and Gaussian rewards,
//! estimation error (MSE ~ 1/n) and **average** detection delay
//! (D_avg ~ C*T / (n * delta^2)) are both monotone-decreasing in the suboptimal-arm
//! pull count n, with proportional gradients everywhere:
//!
//! ```text
//!   D_avg = (2 * C * sigma^2 * ln(1/alpha) / delta^2) * MSE
//! ```
//!
//! This proportionality is structural, not approximate: both functionals care only
//! about "how many observations at this cell", and their sensitivity functions are
//! scalar multiples of each other.  The three-way Pareto surface collapses to a
//! **one-dimensional curve** parameterized by n.  This yields the product identity
//! `R_T * D_avg = C * Delta * T / delta^2` for the restricted class of uniform
//! schedules.
//!
//! **Categorical note.** The formula above uses Gaussian notation (`sigma^2`, mean
//! shift `delta`).  `muxer` operates on categorical outcomes (ok/junk/hard_junk),
//! where detection delay scales as `h / KL(p1 || p0)` in sample time rather than
//! `2*b*sigma^2 / delta^2`.  The proportionality structure — MSE and average
//! detection delay both `O(1/n_k)` — holds identically; only the constants change.
//! See [`pare::sensitivity`][pare] for the general form.
//!
//! **Caveat: adaptive policies.** This clean proportionality holds exactly only for
//! static (non-adaptive) schedules.  Adaptive policies (UCB, Thompson Sampling) break
//! it in at least two ways: (1) they allocate the suboptimal arm in bursts during
//! exploration phases rather than uniformly, worsening worst-case detection delay
//! relative to uniform spacing without changing total n; and (2) data-dependent
//! allocation introduces selection bias, so the effective sample size for estimation
//! is no longer simply n.  For adaptive policies, the product identity becomes a
//! **lower bound** (with constants that absorb regularity conditions), not an equality.
//!
//! Practically, `muxer` operates at small K (2-10 arms) and moderate T (hundreds to
//! low thousands of windowed observations).  At these scales, the asymptotic
//! impossibility results may not bind: all three objectives can often be simultaneously
//! satisfied at acceptable levels.  The sliding-window design further blunts the
//! static/adaptive distinction, since the effective horizon is the window size, not T.
//!
//! ## The contextual revival -- and its subtlety
//!
//! In the **contextual** regime (per-request feature vectors via `LinUcb`), the design
//! measure gains spatial dimensions, but objectives do **not** automatically diverge.
//! The mechanism controlling collapse vs. revival is **average-case vs. worst-case
//! aggregation**, not "contextual vs. non-contextual" per se:
//!
//! - **Average detection delay** has sensitivity `s(x) ~ -1/p_a(x)^2`, which is
//!   proportional to nonparametric IMSE sensitivity everywhere.  Average detection
//!   is **structurally redundant with estimation** even in contextual settings.
//!   Adding contexts does not break this proportionality.
//!
//! - **Worst-case detection delay** (`D_max = max_j D_j`) concentrates its sensitivity
//!   on the **bottleneck cell** -- the (arm, covariate) pair with the fewest observations.
//!   This is a point mass, linearly independent from both the regret sensitivity (a
//!   ramp near decision boundaries) and the estimation sensitivity (D-optimal / extremal).
//!   Worst-case detection is genuinely independent from regret and estimation.
//!
//! In the non-contextual case (one cell), average and worst-case are identical, so the
//! distinction is moot.  In the contextual case (many cells), they diverge: average
//! detection remains redundant with estimation, but worst-case detection introduces a
//! genuinely new objective axis.
//!
//! Concretely, each objective wants a different sampling distribution:
//!
//! - **Regret-optimal**: concentrate near decision boundaries.
//! - **Estimation-optimal**: spread to extremes (D-optimal experimental design).
//! - **Detection-optimal (worst-case)**: ensure no cell is starved (space-filling).
//!
//! `LinUcb` exists to break the non-contextual collapse: it learns per-request routing
//! without maintaining separate per-facet histories, at the cost of requiring explicit
//! monitoring budget beyond what regret-optimal sampling provides.
//!
//! ## Saturation principle
//!
//! The number of genuinely independent objectives is bounded by the effective dimension
//! of the design space:
//!
//! ```text
//!   dim(Pareto front) <= min(m - 1, D_eff)
//! ```
//!
//! where `m` is the number of named objectives and `D_eff` is the design dimension
//! (K-1 for non-contextual, ~K*M for M covariate cells, infinite for continuous
//! covariates).  Adding objectives beyond `D_eff + 1` cannot create new tradeoffs.
//!
//! The formal algebraic rank can overstate the practical number of tradeoffs.  The
//! **effective Pareto dimension** is better measured by the singular value spectrum of
//! the Jacobian of objectives with respect to design variables: a K=3, M=9 setup with
//! 8 named objectives achieves formal rank 8 but effective dimension ~3-4 (the top 2
//! singular values carry >99% of the Frobenius norm).  See `pare::sensitivity` for
//! computational tools.
//!
//! ## Design measure perspective
//!
//! The fundamental object is not "which objectives matter" but the **design measure**:
//! the joint distribution over (arm, covariate, time) that the policy induces.  Given
//! the design measure, every objective is computable.
//!
//! Note: the design measure in an adaptive setting is a random object (it depends on
//! the realized trajectory), not a fixed distribution.  Reasoning about it requires
//! either working with the expected design measure (which loses adaptivity) or a
//! conditional analysis that respects the filtration (which is harder and may break
//! clean proportionality results).  See Hadad et al. (arXiv:1911.02768) for the
//! observed vs. expected Fisher information distinction in adaptive experiments.
//!
//! ## Related work
//!
//! **Non-stationary bandits and sliding windows.**
//! Garivier & Moulines (2008, arXiv:0805.3415) established the Sliding-Window UCB
//! (SW-UCB) algorithm, which achieves `O(sqrt(Υ_T * K * T * log T))` regret for
//! piecewise-stationary environments with `Υ_T` changepoints.  This is the theoretical
//! foundation for `muxer`'s sliding-window approach.  Optimal window size is
//! `O(sqrt(T / Υ_T))`; in practice `muxer` uses a fixed caller-chosen cap.
//!
//! **Non-stationary bandits with explicit detection.**
//! Besson, Kaufmann, Maillard & Seznec (2019, arXiv:1902.01575, JMLR 2023) introduced
//! GLR-klUCB: a parameter-free algorithm combining kl-UCB with a Bernoulli Generalized
//! Likelihood Ratio Test.  It is the closest published analog to `muxer`'s monitored
//! selection path.  Key difference: GLR-klUCB **restarts** all arm statistics on
//! detection; `muxer` instead switches to worst-first routing to investigate the flagged
//! arm, preserving history.  See `TriageSession` and `WorstFirstConfig`.
//!
//! **Bandit Quickest Changepoint Detection (BQCD).**
//! Gopalan, Saligrama & Lakshminarayanan (2021, arXiv:2107.10492) established the BQCD
//! lower bound: any algorithm with mean-time-to-false-alarm `m` must suffer expected
//! detection delay `Ω(log(m) / D*)`, where `D*` is the maximum KL divergence between
//! post-change and pre-change distributions across arms.  Their ε-GCD algorithm achieves
//! this with exploration rate `ε = Θ(1/√T)`.  `CoverageConfig`'s minimum sampling-rate
//! floor is the practical lever for approaching this bound.
//!
//! **Sampling-constrained detection.**
//! Zhang & Mei (2020, arXiv:2009.11891) directly analyze changepoint detection under
//! sampling-rate constraints, confirming that detection delay in wall time scales as
//! `h / (KL(p1 || p0) * rate_k)` — the formal basis for the two-clocks approximation.
//!
//! **Regret–BAI Pareto frontier.**
//! Zhong, Cheung & Tan (2021, arXiv:2110.08627) formally prove the Pareto tradeoff
//! between regret minimization (RM) and best-arm identification (BAI): achieving
//! `O(log T)` regret and `O(log T)` BAI error simultaneously is impossible.  The
//! product-identity formulation in `muxer`'s docs is the static-schedule special case.
//!
//! **Piecewise-stationary multi-objective bandits.**
//! Rezaei Balef & Maghsudi (2023, arXiv:2302.05257) study the combination of
//! non-stationarity and multi-objective rewards directly.  `muxer` operates in this
//! space — without claiming worst-case-optimal bounds — by combining sliding-window
//! summaries with Pareto scalarization and explicit monitoring.
//!
//! **Window-limited CUSUM.**
//! Xie, Moustakides & Xie (2022, arXiv:2206.06777) connect windowed observation to
//! CUSUM optimality, providing theoretical grounding for `MonitoredWindow`'s split
//! between baseline and recent windows.
//!
//! **Multi-objective bandit frameworks.**
//! - **Constrained / safe bandits** (BwK): `muxer`'s `max_junk_rate`, `max_drift`, etc.
//!   are BwK-style anytime constraints.  (Badanidiyuru, Kleinberg & Slivkins 2013,
//!   FOCS; arXiv:1305.2545)
//! - **Pareto bandits** (Drugan & Nowé, IJCNN 2013): `muxer`'s `pare`-based frontier
//!   is the selection-time analogue.
//! - **Information-Directed Sampling** (Russo & Van Roy 2014, arXiv:1403.5556):
//!   scalarizes regret/information via the information ratio.  The three-objective
//!   extension is non-trivial only in the contextual worst-case-detection regime.
//! - **Multi-objective RL**: `muxer`'s deterministic post-Pareto scalarization is a
//!   pragmatic instance of this literature.
//! - **Adaptive experiment design** (Hadad et al. 2021, arXiv:1911.02768): the
//!   observed vs. expected Fisher information distinction applies to `muxer`'s
//!   adaptive design-measure analysis.

#![forbid(unsafe_code)]

use pare::{Direction, ParetoFrontier};
use std::collections::{BTreeMap, VecDeque};

/// Epsilon used for floating-point tie-breaking in selection scoring.
///
/// This avoids exact equality comparisons on f64 scores and provides a stable
/// threshold across all selection paths (Pareto scalarization, UCB, etc.).
const TIEBREAK_EPS: f64 = 1e-12;

mod decision;
pub use decision::*;

mod policy;
#[cfg(feature = "stochastic")]
pub use policy::BanditPolicy;

mod alloc;
pub use alloc::*;

mod utils;
pub use utils::*;

mod control;
pub use control::*;

mod router;
pub use router::*;

mod guardrail;
pub use guardrail::*;

pub mod monitor;

mod coverage;
pub use coverage::*;

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

mod triage;
pub use triage::*;

pub use monitor::{
    DriftConfig, DriftDecision, DriftMetric, MonitoredWindow, RateBoundMode, UncertaintyConfig,
    ThresholdCalibration, calibrate_threshold_from_max_scores,
};
#[cfg(feature = "stochastic")]
pub use monitor::{calibrate_cusum_threshold, simulate_cusum_null_max_scores};

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
    /// Mean quality score (present when `Outcome::quality_score` has been set).
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub mean_quality_score: Option<f64>,
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
            let drift = c.drift_score.unwrap_or(0.0);
            let catkl = c.catkl_score.unwrap_or(0.0);
            let cusum = c.cusum_score.unwrap_or(0.0);
            let quality_bonus = cfg.quality_weight * c.mean_quality_score.unwrap_or(0.0);
            let score = c.objective_success
                - cfg.cost_weight * c.mean_cost_units
                - cfg.latency_weight * c.mean_elapsed_ms
                - cfg.hard_junk_weight * c.hard_junk_rate
                - cfg.junk_weight * c.soft_junk_rate
                - cfg.drift_weight * drift
                - cfg.catkl_weight * catkl
                - cfg.cusum_weight * cusum
                + quality_bonus;
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
            mean_quality_score: c.mean_quality_score,
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
            mean_quality_score: None,
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
    /// Optional continuous quality signal in `[0.0, 1.0]` (higher = better).
    ///
    /// Supplements the binary `junk`/`hard_junk` flags with a gradient signal.
    /// A response scoring `0.58` on a `0.60` threshold is functionally different
    /// from a `0.0` response; this field preserves that distinction without
    /// changing the binary routing logic.
    ///
    /// `None` (the default) means "not measured". Set via
    /// [`Window::set_last_quality_score`] when scoring completes after the call.
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Option::is_none"))]
    pub quality_score: Option<f64>,
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

    /// Iterate over outcomes in the window (oldest to newest).
    pub fn iter(&self) -> impl Iterator<Item = &Outcome> + '_ {
        self.buf.iter()
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

    /// Best-effort: set the continuous quality score for the most recent outcome.
    ///
    /// Call this after downstream scoring completes (same pattern as
    /// [`set_last_junk_level`]).  The value is clamped to `[0.0, 1.0]`.
    pub fn set_last_quality_score(&mut self, score: f64) {
        if let Some(last) = self.buf.back_mut() {
            last.quality_score = Some(score.clamp(0.0, 1.0));
        }
    }

    /// Mean quality score over outcomes that have one, or `None` if none have been set.
    ///
    /// Provides a gradient signal complementary to the binary ok/junk rates.
    pub fn mean_quality_score(&self) -> Option<f64> {
        let mut sum = 0.0_f64;
        let mut count = 0u64;
        for o in &self.buf {
            if let Some(q) = o.quality_score {
                sum += q;
                count += 1;
            }
        }
        if count == 0 { None } else { Some(sum / count as f64) }
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
        // Compute mean quality score from outcomes that have it set.
        let mut quality_sum = 0.0_f64;
        let mut quality_count = 0u64;
        for o in &self.buf {
            if let Some(q) = o.quality_score {
                quality_sum += q;
                quality_count += 1;
            }
        }
        let mean_quality_score = if quality_count > 0 {
            Some(quality_sum / quality_count as f64)
        } else {
            None
        };

        Summary {
            calls: n,
            ok,
            junk,
            hard_junk,
            cost_units,
            elapsed_ms_sum,
            mean_quality_score,
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
    /// Mean quality score over outcomes that had `quality_score` set, or `None`.
    ///
    /// Populated by [`Window::summary`].  When constructing `Summary` directly
    /// (not via `Window`), set this to reflect a pre-computed quality estimate.
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Option::is_none"))]
    pub mean_quality_score: Option<f64>,
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
    /// Bonus weight for mean quality score (`quality_score` field on `Outcome`).
    ///
    /// When non-zero and quality scores have been recorded, adds
    /// `quality_weight * mean_quality_score` to `objective_success`.
    /// Higher quality scores push an arm toward selection; 0 disables.
    pub quality_weight: f64,
    /// Optional constraint: discard arms whose windowed junk_rate exceeds this.
    pub max_junk_rate: Option<f64>,
    /// Optional constraint: discard arms whose windowed hard_junk_rate exceeds this.
    pub max_hard_junk_rate: Option<f64>,
    /// Optional constraint: discard arms whose windowed mean_cost_units exceeds this.
    pub max_mean_cost_units: Option<f64>,

    /// Optional drift guard: discard arms whose drift score exceeds this.
    ///
    /// Only applies in monitored selection APIs (e.g. `select_mab_monitored_*`).
    pub max_drift: Option<f64>,

    /// Drift metric used when applying `max_drift`.
    ///
    /// Only applies in monitored selection APIs.
    pub drift_metric: DriftMetric,

    /// Penalty weight for drift (0 disables). Larger means "avoid arms that recently changed."
    ///
    /// Only applies in monitored selection APIs.
    pub drift_weight: f64,

    /// Rate uncertainty configuration for Wilson bounds.
    ///
    /// Only applies in monitored selection APIs.
    pub uncertainty: UncertaintyConfig,

    /// Optional categorical KL guard: discard arms whose `S = n_recent * KL(q_recent || p0_baseline)`
    /// exceeds this threshold.
    ///
    /// Only applies in monitored selection APIs.
    pub max_catkl: Option<f64>,

    /// Dirichlet smoothing pseudo-count (alpha) used for categorical KL monitoring.
    ///
    /// Only applies in monitored selection APIs.
    pub catkl_alpha: f64,

    /// Minimum baseline samples required before categorical KL monitoring applies.
    ///
    /// Only applies in monitored selection APIs.
    pub catkl_min_baseline: u64,

    /// Minimum recent samples required before categorical KL monitoring applies.
    ///
    /// Only applies in monitored selection APIs.
    pub catkl_min_recent: u64,

    /// Penalty weight for categorical KL score (0 disables).
    ///
    /// Only applies in monitored selection APIs.
    pub catkl_weight: f64,

    /// Optional categorical CUSUM guard: discard arms whose CUSUM score over the recent window
    /// exceeds this threshold.
    ///
    /// Only applies in monitored selection APIs.
    pub max_cusum: Option<f64>,

    /// Dirichlet smoothing pseudo-count (alpha) used when building categorical CUSUM `p0/p1`.
    ///
    /// Only applies in monitored selection APIs.
    pub cusum_alpha: f64,

    /// Minimum baseline samples required before categorical CUSUM monitoring applies.
    ///
    /// Only applies in monitored selection APIs.
    pub cusum_min_baseline: u64,

    /// Minimum recent samples required before categorical CUSUM monitoring applies.
    ///
    /// Only applies in monitored selection APIs.
    pub cusum_min_recent: u64,

    /// Alternative distribution `p1` for categorical CUSUM over `muxer`'s 4 outcome-categories:
    /// `[ok_clean, ok_soft_junk, ok_hard_junk, fail]`.
    ///
    /// If `None`, a conservative default is used that biases toward `hard_junk`/`fail`.
    ///
    /// Only applies in monitored selection APIs.
    pub cusum_alt_p: Option<[f64; 4]>,

    /// Penalty weight for categorical CUSUM score (0 disables).
    ///
    /// Only applies in monitored selection APIs.
    pub cusum_weight: f64,
}

impl Default for MabConfig {
    fn default() -> Self {
        Self {
            exploration_c: 0.7,
            cost_weight: 0.0,
            latency_weight: 0.0,
            junk_weight: 0.0,
            hard_junk_weight: 0.0,
            quality_weight: 0.0,
            max_junk_rate: None,
            max_hard_junk_rate: None,
            max_mean_cost_units: None,
            max_drift: None,
            drift_metric: DriftMetric::default(),
            drift_weight: 0.0,
            uncertainty: UncertaintyConfig::default(),
            max_catkl: None,
            catkl_alpha: 1e-3,
            catkl_min_baseline: 40,
            catkl_min_recent: 20,
            catkl_weight: 0.0,
            max_cusum: None,
            cusum_alpha: 1e-3,
            cusum_min_baseline: 40,
            cusum_min_recent: 20,
            cusum_alt_p: None,
            cusum_weight: 0.0,
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

    /// Optional drift score for this arm (present for monitored selection).
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub drift_score: Option<f64>,

    /// Optional categorical KL score for this arm (present for monitored selection).
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub catkl_score: Option<f64>,

    /// Optional categorical CUSUM score for this arm (present for monitored selection).
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub cusum_score: Option<f64>,

    /// Wilson half-width for ok_rate (present when uncertainty bounds are enabled).
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub ok_half_width: Option<f64>,

    /// Wilson half-width for junk_rate (present when uncertainty bounds are enabled).
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub junk_half_width: Option<f64>,

    /// Wilson half-width for hard_junk_rate (present when uncertainty bounds are enabled).
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub hard_junk_half_width: Option<f64>,

    /// Mean quality score for this arm (when `Outcome::quality_score` has been set).
    ///
    /// Present when any outcome in the summary window had `quality_score` set.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub mean_quality_score: Option<f64>,
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

    /// Drift guard outcome (only present for monitored selection).
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub drift_guard: Option<DriftGuardDecision>,

    /// Categorical KL guard outcome (only present for monitored selection).
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub catkl_guard: Option<CatKlGuardDecision>,

    /// Categorical CUSUM guard outcome (only present for monitored selection).
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub cusum_guard: Option<CusumGuardDecision>,
}

/// Output of applying a drift guard to a candidate set.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DriftGuardDecision {
    /// Arms eligible after applying drift guard.
    pub eligible_arms: Vec<String>,
    /// Whether we fell back to the full input set because drift guard would have eliminated all arms.
    pub fallback_used: bool,
    /// Drift metric used.
    pub metric: DriftMetric,
    /// The max drift threshold used.
    pub max_drift: f64,
}

/// Output of applying a categorical KL guard to a candidate set.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CatKlGuardDecision {
    /// Arms eligible after applying catKL guard.
    pub eligible_arms: Vec<String>,
    /// Whether we fell back to the full input set because the guard would have eliminated all arms.
    pub fallback_used: bool,
    /// The threshold on `n_recent * KL(q_recent || p0_baseline)`.
    pub max_catkl: f64,
    /// Dirichlet smoothing pseudo-count.
    pub alpha: f64,
    /// Minimum baseline samples required.
    pub min_baseline: u64,
    /// Minimum recent samples required.
    pub min_recent: u64,
}

/// Output of applying a categorical CUSUM guard to a candidate set.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CusumGuardDecision {
    /// Arms eligible after applying CUSUM guard.
    pub eligible_arms: Vec<String>,
    /// Whether we fell back to the full input set because the guard would have eliminated all arms.
    pub fallback_used: bool,
    /// CUSUM threshold used.
    pub max_cusum: f64,
    /// Dirichlet smoothing pseudo-count.
    pub alpha: f64,
    /// Minimum baseline samples required.
    pub min_baseline: u64,
    /// Minimum recent samples required.
    pub min_recent: u64,
    /// Alternative distribution used for CUSUM.
    pub alt_p: [f64; 4],
}

fn apply_base_constraints(
    arms_in_order: &[String],
    summaries: &BTreeMap<String, Summary>,
    cfg: &MabConfig,
) -> (Vec<String>, bool) {
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
        eligible
    };
    (eligible_arms, constraints_fallback_used)
}

fn explore_first_decision(
    chosen: String,
    eligible_arms: Vec<String>,
    constraints_fallback_used: bool,
    cfg: MabConfig,
) -> MabSelectionDecision {
    let sel = Selection {
        chosen: chosen.clone(),
        frontier: vec![chosen.clone()],
        candidates: vec![CandidateDebug {
            name: chosen,
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
            drift_score: None,
            catkl_score: None,
            cusum_score: None,
            ok_half_width: None,
            junk_half_width: None,
            hard_junk_half_width: None,
            mean_quality_score: None,
        }],
        config: cfg,
    };
    MabSelectionDecision {
        selection: sel,
        eligible_arms,
        constraints_fallback_used,
        explore_first: true,
        drift_guard: None,
        catkl_guard: None,
        cusum_guard: None,
    }
}

fn choose_from_frontier<FScore>(
    dims: usize,
    candidates: &[CandidateDebug],
    frontier_points: &[Vec<f64>],
    frontier_names_in_order: &[String],
    fallback_first: Option<&String>,
    score: FScore,
) -> (String, Vec<String>)
where
    FScore: Fn(&CandidateDebug) -> f64,
{
    let mut frontier = ParetoFrontier::new(vec![Direction::Maximize; dims]);
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

    let mut best_name = frontier_names
        .first()
        .cloned()
        .unwrap_or_else(|| fallback_first.cloned().unwrap_or_default());
    let mut best_score = f64::NEG_INFINITY;
    for &idx in &frontier_indices {
        let Some(c) = candidates.get(idx) else {
            continue;
        };
        let s = score(c);
        if s > best_score || ((s - best_score).abs() <= TIEBREAK_EPS && c.name < best_name) {
            best_score = s;
            best_name = c.name.clone();
        }
    }

    (best_name, frontier_names)
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
///     Summary { calls: 10, ok: 9, junk: 0, hard_junk: 0, cost_units: 10, elapsed_ms_sum: 900, mean_quality_score: None }
/// );
/// summaries.insert(
///     "b".to_string(),
///     Summary { calls: 10, ok: 9, junk: 2, hard_junk: 0, cost_units: 10, elapsed_ms_sum: 900, mean_quality_score: None }
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
    let (eligible_arms, constraints_fallback_used) =
        apply_base_constraints(arms_in_order, summaries, &cfg);
    let arms_in_order: &[String] = &eligible_arms;

    // Explore first.
    let explore_choice: Option<String> = arms_in_order
        .iter()
        .find(|a| summaries.get(*a).copied().unwrap_or_default().calls == 0)
        .cloned();
    if let Some(chosen) = explore_choice {
        return explore_first_decision(chosen, eligible_arms, constraints_fallback_used, cfg);
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
        let quality_bonus = if cfg.quality_weight > 0.0 {
            cfg.quality_weight * s.mean_quality_score.unwrap_or(0.0)
        } else {
            0.0
        };
        let objective_success = ok_rate + ucb + quality_bonus;

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
            drift_score: None,
            catkl_score: None,
            cusum_score: None,
            ok_half_width: None,
            junk_half_width: None,
            hard_junk_half_width: None,
            mean_quality_score: s.mean_quality_score,
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

    // Deterministic scalarization among frontier points.
    let weights: [f64; 5] = [
        1.0,
        cfg.cost_weight.max(0.0),
        cfg.latency_weight.max(0.0),
        cfg.hard_junk_weight.max(0.0),
        cfg.junk_weight.max(0.0),
    ];
    let (best_name, frontier_names) = choose_from_frontier(
        5,
        &candidates,
        &frontier_points,
        &frontier_names_in_order,
        arms_in_order.first(),
        |c| {
            // Maximize:
            //   objective_success - w_cost * cost - w_lat * latency - w_hard * hard_junk - w_soft * soft_junk
            c.objective_success
                - weights[1] * c.mean_cost_units
                - weights[2] * c.mean_elapsed_ms
                - weights[3] * c.hard_junk_rate
                - weights[4] * c.soft_junk_rate
        },
    );

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
        drift_guard: None,
        catkl_guard: None,
        cusum_guard: None,
    }
}

/// Monitored deterministic selection (baseline vs recent drift + uncertainty-aware rates).
///
/// This is intended for production routers that already maintain `MonitoredWindow`s per arm.
///
/// Semantics:
/// - Uses `monitored[*].recent_summary()` for the base stats (ok/junk/cost/latency).
/// - Optionally applies a drift guardrail (`cfg.max_drift`) against baseline-vs-recent drift.
/// - Optionally penalizes drift (`cfg.drift_weight`) as an additional objective.
/// - Optionally applies categorical KL monitoring as an objective/guard (`cfg.max_catkl`, `cfg.catkl_weight`).
/// - Optionally adjusts rates using Wilson bounds (`cfg.uncertainty`).
///
/// Like `select_mab_explain`, this never returns an empty choice set: if drift filtering would
/// eliminate all arms, it falls back to the unfiltered eligible set.
///
/// This is a convenience wrapper: it builds summaries from `monitored[*].recent_summary()` and
/// delegates to [`select_mab_monitored_explain_with_summaries`].
pub fn select_mab_monitored_explain(
    arms_in_order: &[String],
    monitored: &BTreeMap<String, MonitoredWindow>,
    drift_cfg: DriftConfig,
    cfg: MabConfig,
) -> MabSelectionDecision {
    let summaries: BTreeMap<String, Summary> = monitored
        .iter()
        .map(|(k, w)| (k.clone(), w.recent_summary()))
        .collect();
    select_mab_monitored_explain_with_summaries(
        arms_in_order,
        &summaries,
        monitored,
        drift_cfg,
        cfg,
    )
}

/// Like `select_mab_monitored_explain`, but uses caller-provided summaries for the base objectives
/// (e.g. prior-smoothed rates), while still computing monitoring scores from `monitored`.
///
/// This is useful for harnesses that want:
/// - selection on a smoothed/aggregated summary, but
/// - drift/catKL/CUSUM computed from raw baseline/recent windows.
pub fn select_mab_monitored_explain_with_summaries(
    arms_in_order: &[String],
    summaries: &BTreeMap<String, Summary>,
    monitored: &BTreeMap<String, MonitoredWindow>,
    drift_cfg: DriftConfig,
    cfg: MabConfig,
) -> MabSelectionDecision {
    // Apply base hard constraints first (same semantics as `select_mab_explain`).
    let (eligible_arms, constraints_fallback_used) =
        apply_base_constraints(arms_in_order, summaries, &cfg);
    let arms_in_order: &[String] = &eligible_arms;

    // Explore first (stable order).
    let explore_choice: Option<String> = arms_in_order
        .iter()
        .find(|a| summaries.get(*a).copied().unwrap_or_default().calls == 0)
        .cloned();
    if let Some(chosen) = explore_choice {
        return explore_first_decision(chosen, eligible_arms, constraints_fallback_used, cfg);
    }

    // Apply drift guard (optional) over the constraint-eligible set.
    let max_drift = cfg
        .max_drift
        .and_then(|x| (x.is_finite() && x >= 0.0).then_some(x));
    let mut eligible_after_drift = arms_in_order.to_vec();
    let mut drift_guard: Option<DriftGuardDecision> = None;
    if let Some(thr) = max_drift {
        let mut kept: Vec<String> = Vec::new();
        for a in arms_in_order {
            let Some(w) = monitored.get(a) else {
                kept.push(a.clone());
                continue;
            };
            let d = monitor::drift_between_windows(
                w.baseline(),
                w.recent(),
                DriftConfig {
                    metric: cfg.drift_metric,
                    ..drift_cfg
                },
            );
            let violates = d.as_ref().map(|x| x.score > thr).unwrap_or(false);
            if !violates {
                kept.push(a.clone());
            }
        }
        let fallback_used = kept.is_empty();
        let eligible_arms = if fallback_used {
            arms_in_order.to_vec()
        } else {
            kept
        };
        drift_guard = Some(DriftGuardDecision {
            eligible_arms: eligible_arms.clone(),
            fallback_used,
            metric: cfg.drift_metric,
            max_drift: thr,
        });
        eligible_after_drift = eligible_arms;
    }

    // Apply categorical KL guard (optional) over the drift-eligible set.
    let max_catkl = cfg
        .max_catkl
        .and_then(|x| (x.is_finite() && x >= 0.0).then_some(x));
    let catkl_alpha = if cfg.catkl_alpha.is_finite() && cfg.catkl_alpha > 0.0 {
        cfg.catkl_alpha
    } else {
        1e-3
    };
    let mut eligible_after_catkl = eligible_after_drift.clone();
    let mut catkl_guard: Option<CatKlGuardDecision> = None;
    if let Some(thr) = max_catkl {
        let mut kept: Vec<String> = Vec::new();
        for a in &eligible_after_drift {
            let Some(w) = monitored.get(a) else {
                kept.push(a.clone());
                continue;
            };
            let s = monitor::catkl_score_between_windows(
                w.baseline(),
                w.recent(),
                catkl_alpha,
                drift_cfg.tol,
                cfg.catkl_min_baseline,
                cfg.catkl_min_recent,
            );
            let violates = s.map(|x| x > thr).unwrap_or(false);
            if !violates {
                kept.push(a.clone());
            }
        }
        let fallback_used = kept.is_empty();
        let eligible_arms = if fallback_used {
            eligible_after_drift.clone()
        } else {
            kept
        };
        catkl_guard = Some(CatKlGuardDecision {
            eligible_arms: eligible_arms.clone(),
            fallback_used,
            max_catkl: thr,
            alpha: catkl_alpha,
            min_baseline: cfg.catkl_min_baseline,
            min_recent: cfg.catkl_min_recent,
        });
        eligible_after_catkl = eligible_arms;
    }

    // Apply categorical CUSUM guard (optional) over the catKL-eligible set.
    let max_cusum = cfg
        .max_cusum
        .and_then(|x| (x.is_finite() && x >= 0.0).then_some(x));
    let cusum_alpha = if cfg.cusum_alpha.is_finite() && cfg.cusum_alpha > 0.0 {
        cfg.cusum_alpha
    } else {
        1e-3
    };
    let cusum_alt_p = cfg.cusum_alt_p.unwrap_or([0.05, 0.05, 0.45, 0.45]);
    let mut eligible_after_cusum = eligible_after_catkl.clone();
    let mut cusum_guard: Option<CusumGuardDecision> = None;
    if let Some(thr) = max_cusum {
        let mut kept: Vec<String> = Vec::new();
        for a in &eligible_after_catkl {
            let Some(w) = monitored.get(a) else {
                kept.push(a.clone());
                continue;
            };
            let s = monitor::cusum_score_between_windows(
                w.baseline(),
                w.recent(),
                cusum_alpha,
                drift_cfg.tol,
                cfg.cusum_min_baseline,
                cfg.cusum_min_recent,
                Some(cusum_alt_p),
            );
            let violates = s.map(|x| x > thr).unwrap_or(false);
            if !violates {
                kept.push(a.clone());
            }
        }
        let fallback_used = kept.is_empty();
        let eligible_arms = if fallback_used {
            eligible_after_catkl.clone()
        } else {
            kept
        };
        cusum_guard = Some(CusumGuardDecision {
            eligible_arms: eligible_arms.clone(),
            fallback_used,
            max_cusum: thr,
            alpha: cusum_alpha,
            min_baseline: cfg.cusum_min_baseline,
            min_recent: cfg.cusum_min_recent,
            alt_p: cusum_alt_p,
        });
        eligible_after_cusum = eligible_arms;
    }

    // Only count calls for the arms actually being considered.
    let total_calls: f64 = eligible_after_cusum
        .iter()
        .map(|a| summaries.get(a).copied().unwrap_or_default().calls as f64)
        .sum::<f64>()
        .max(1.0);

    // Monitored Pareto frontier includes drift + catKL + CUSUM as extra objectives (minimize each).
    let mut frontier_points: Vec<Vec<f64>> = Vec::new();
    let mut frontier_names_in_order: Vec<String> = Vec::new();
    let mut candidates: Vec<CandidateDebug> = Vec::new();

    for a in &eligible_after_cusum {
        let s = summaries.get(a).copied().unwrap_or_default();
        let n = (s.calls as f64).max(1.0);

        // Uncertainty-aware rates (Wilson).
        let z = cfg.uncertainty.z;
        let soft = s.junk.saturating_sub(s.hard_junk);
        let (ok_rate_used, ok_half) =
            monitor::apply_rate_bound(s.ok, s.calls, z, cfg.uncertainty.ok_mode);
        let (hard_used, hard_half) =
            monitor::apply_rate_bound(s.hard_junk, s.calls, z, cfg.uncertainty.hard_junk_mode);
        let (soft_used, soft_half) =
            monitor::apply_rate_bound(soft, s.calls, z, cfg.uncertainty.junk_mode);

        let junk_total_used = (soft_used + hard_used).clamp(0.0, 1.0);

        // Monitoring scores from raw windows (optional).
        let drift_score = monitored.get(a).and_then(|w| {
            monitor::drift_between_windows(
                w.baseline(),
                w.recent(),
                DriftConfig {
                    metric: cfg.drift_metric,
                    ..drift_cfg
                },
            )
            .map(|x| x.score)
        });
        let drift_used = drift_score.unwrap_or(0.0);

        let catkl_score = monitored.get(a).and_then(|w| {
            monitor::catkl_score_between_windows(
                w.baseline(),
                w.recent(),
                catkl_alpha,
                drift_cfg.tol,
                cfg.catkl_min_baseline,
                cfg.catkl_min_recent,
            )
        });
        let catkl_used = catkl_score.unwrap_or(0.0);

        let cusum_score = monitored.get(a).and_then(|w| {
            monitor::cusum_score_between_windows(
                w.baseline(),
                w.recent(),
                cusum_alpha,
                drift_cfg.tol,
                cfg.cusum_min_baseline,
                cfg.cusum_min_recent,
                Some(cusum_alt_p),
            )
        });
        let cusum_used = cusum_score.unwrap_or(0.0);

        let ucb = cfg.exploration_c * ((total_calls.ln() / n).sqrt());
        let quality_bonus = if cfg.quality_weight > 0.0 {
            cfg.quality_weight * s.mean_quality_score.unwrap_or(0.0)
        } else {
            0.0
        };
        let objective_success = ok_rate_used + ucb + quality_bonus;

        let mean_cost = s.mean_cost_units();
        let mean_lat = s.mean_elapsed_ms();

        candidates.push(CandidateDebug {
            name: a.clone(),
            calls: s.calls,
            ok: s.ok,
            junk: s.junk,
            hard_junk: s.hard_junk,
            ok_rate: ok_rate_used,
            junk_rate: junk_total_used,
            hard_junk_rate: hard_used,
            soft_junk_rate: soft_used,
            mean_cost_units: mean_cost,
            mean_elapsed_ms: mean_lat,
            ucb,
            objective_success,
            drift_score,
            catkl_score,
            cusum_score,
            ok_half_width: Some(ok_half),
            junk_half_width: Some(soft_half),
            hard_junk_half_width: Some(hard_half),
            mean_quality_score: s.mean_quality_score,
        });

        frontier_points.push(vec![
            objective_success,
            -mean_cost,
            -mean_lat,
            -hard_used,
            -soft_used,
            -drift_used,
            -catkl_used,
            -cusum_used,
        ]);
        frontier_names_in_order.push(a.clone());
    }
    let weights: [f64; 8] = [
        1.0,
        cfg.cost_weight.max(0.0),
        cfg.latency_weight.max(0.0),
        cfg.hard_junk_weight.max(0.0),
        cfg.junk_weight.max(0.0),
        cfg.drift_weight.max(0.0),
        cfg.catkl_weight.max(0.0),
        cfg.cusum_weight.max(0.0),
    ];
    let (best_name, frontier_names) = choose_from_frontier(
        8,
        &candidates,
        &frontier_points,
        &frontier_names_in_order,
        eligible_after_cusum.first(),
        |c| {
            let drift = c.drift_score.unwrap_or(0.0);
            let catkl = c.catkl_score.unwrap_or(0.0);
            let cusum = c.cusum_score.unwrap_or(0.0);
            c.objective_success
                - weights[1] * c.mean_cost_units
                - weights[2] * c.mean_elapsed_ms
                - weights[3] * c.hard_junk_rate
                - weights[4] * c.soft_junk_rate
                - weights[5] * drift
                - weights[6] * catkl
                - weights[7] * cusum
        },
    );

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
        drift_guard,
        catkl_guard,
        cusum_guard,
    }
}

/// Multi-pick variant of monitored selection using caller-provided summaries and monitored windows.
pub fn select_mab_k_guardrailed_monitored_explain_full<FS, FM>(
    arms_in_order: &[String],
    mut summaries_for: FS,
    mut monitored_for: FM,
    drift_cfg: DriftConfig,
    cfg: MabConfig,
    guardrail: LatencyGuardrailConfig,
    k: usize,
) -> MabKExplain
where
    FS: FnMut(&[String]) -> BTreeMap<String, Summary>,
    FM: FnMut(&[String]) -> BTreeMap<String, MonitoredWindow>,
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
        let monitored = monitored_for(&remaining_in);

        let gd = apply_latency_guardrail(&remaining_in, &summaries, guardrail, chosen.len());
        if gd.stop_early || gd.eligible.is_empty() {
            stop = Some(MabKStop {
                remaining: remaining_in,
                guardrail: gd,
            });
            break;
        }

        let d = select_mab_monitored_explain_with_summaries(
            &gd.eligible,
            &summaries,
            &monitored,
            drift_cfg,
            cfg,
        );
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

/// Unified decision envelope for monitored deterministic MAB selection.
pub fn select_mab_monitored_decide(
    arms_in_order: &[String],
    monitored: &BTreeMap<String, MonitoredWindow>,
    drift_cfg: DriftConfig,
    cfg: MabConfig,
) -> Decision {
    let d = select_mab_monitored_explain(arms_in_order, monitored, drift_cfg, cfg);

    let mut notes = vec![DecisionNote::Constraints {
        eligible_arms: d.eligible_arms.clone(),
        fallback_used: d.constraints_fallback_used,
    }];
    if let Some(ref dg) = d.drift_guard {
        notes.push(DecisionNote::DriftGuard {
            eligible_arms: dg.eligible_arms.clone(),
            fallback_used: dg.fallback_used,
            metric: dg.metric,
            max_drift: dg.max_drift,
        });
    }
    if let Some(ref cg) = d.catkl_guard {
        notes.push(DecisionNote::CatKlGuard {
            eligible_arms: cg.eligible_arms.clone(),
            fallback_used: cg.fallback_used,
            max_catkl: cg.max_catkl,
            alpha: cg.alpha,
            min_baseline: cg.min_baseline,
            min_recent: cg.min_recent,
        });
    }
    if let Some(ref ug) = d.cusum_guard {
        notes.push(DecisionNote::CusumGuard {
            eligible_arms: ug.eligible_arms.clone(),
            fallback_used: ug.fallback_used,
            max_cusum: ug.max_cusum,
            alpha: ug.alpha,
            min_baseline: ug.min_baseline,
            min_recent: ug.min_recent,
            alt_p: ug.alt_p,
        });
    }
    if d.explore_first {
        notes.push(DecisionNote::ExploreFirst);
    } else {
        notes.push(DecisionNote::DeterministicChoice);
    }

    // Attach chosen-arm diagnostics (if present).
    let chosen_row = d
        .selection
        .candidates
        .iter()
        .find(|c| c.name == d.selection.chosen);
    if let Some(c) = chosen_row {
        notes.push(DecisionNote::Diagnostics {
            drift_score: c.drift_score,
            catkl_score: c.catkl_score,
            cusum_score: c.cusum_score,
            ok_half_width: c.ok_half_width,
            junk_half_width: c.junk_half_width,
            hard_junk_half_width: c.hard_junk_half_width,
            mean_quality_score: c.mean_quality_score,
        });
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
            mean_quality_score: None,
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
                mean_quality_score: None,
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
                mean_quality_score: None,
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
                ..MabConfig::default()
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
                            drift_score: None,
                            catkl_score: None,
                            cusum_score: None,
                            ok_half_width: None,
                            junk_half_width: None,
                            hard_junk_half_width: None,
                            mean_quality_score: None,
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
                            drift_score: None,
                            catkl_score: None,
                            cusum_score: None,
                            ok_half_width: None,
                            junk_half_width: None,
                            hard_junk_half_width: None,
                            mean_quality_score: None,
                        },
                    ],
                    config: cfg,
                },
                eligible_arms: vec!["a".to_string(), "b".to_string()],
                constraints_fallback_used: false,
                explore_first: false,
                drift_guard: None,
                catkl_guard: None,
                cusum_guard: None,
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
                    drift_score: None,
                    catkl_score: None,
                    cusum_score: None,
                    ok_half_width: None,
                    junk_half_width: None,
                    hard_junk_half_width: None,
                    mean_quality_score: None,
                }],
                config: cfg,
            },
            eligible_arms: vec!["old".to_string()],
            constraints_fallback_used: false,
            explore_first: true,
            drift_guard: None,
            catkl_guard: None,
            cusum_guard: None,
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
                drift_score: None,
                catkl_score: None,
                cusum_score: None,
                ok_half_width: None,
                junk_half_width: None,
                hard_junk_half_width: None,
                mean_quality_score: None,
            }],
            config: cfg,
        };
        let e = sticky.apply_mab(MabSelectionDecision {
            selection: base,
            eligible_arms: vec!["a".to_string()],
            constraints_fallback_used: false,
            explore_first: false,
            drift_guard: None,
            catkl_guard: None,
            cusum_guard: None,
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
