//! `muxer`: deterministic, multi-objective routing primitives for
//! piecewise-stationary multi-armed bandit problems.
//!
//! Given `K` arms (model versions, inference endpoints, backends, or any
//! discrete action set selected repeatedly), the agent observes
//! vector-valued outcomes per call and selects the next arm.  Reward
//! distributions are piecewise-stationary: they may change at unknown
//! times, and the agent must detect and adapt to these changes.
//!
//! An [`Outcome`] carries four caller-defined quality fields:
//! - `ok`: the operation produced a usable result according to the caller.
//! - `junk`: the result was below the caller's quality threshold (`hard_junk=true` implies
//!   `junk=true`).
//! - `hard_junk`: a severe subset of junk, tracked separately from ordinary degradation.
//! - `quality_score: Option<f64>`: optional continuous quality signal `[0,1]` (higher = better).
//!   Supplements the binary flags without changing routing semantics.
//!
//! Plus `cost_units` (caller-defined cost proxy) and `elapsed_ms` (wall-clock time).
//!
//! **Goals:**
//! - **Deterministic by default**: same stats + config → same choice.
//! - **Non-stationarity friendly**: sliding-window summaries, not lifetime averages.
//! - **Multi-objective**: Pareto frontier over configurable [`Objective`] dimensions,
//!   then scalarization.  [`default_objectives`] provides ok/junk/cost/latency/quality.
//! - **Small K**: designed for 2–10 arms; not intended for K in the hundreds.
//!
//! **Selection policies:**
//! - [`select_mab`] / [`select_mab_explain`] / [`select_mab_monitored_explain`]:
//!   deterministic Pareto + scalarization over [`MabConfig::objectives`].
//!   Each [`Objective`] defines an extraction, direction, and scalarization weight.
//! - [`ThompsonSampling`]: seedable Thompson sampling for scalar rewards in `[0, 1]`.
//! - [`Exp3Ix`]: seedable EXP3-IX for adversarial / fast-shifting rewards.
//! - [`BanditPolicy`] (feature `stochastic`): common `decide`/`update_reward` trait
//!   for `ThompsonSampling` and `Exp3Ix` — enables generic policy code.
//! - [`softmax_map`]: stable score → probability helper for traffic splitting.
//! - [`CandidateAssessment`] / [`select_candidate_assessments`]: domain-neutral
//!   Pareto selection over caller-defined metric vectors.
//! - (feature `contextual`) `LinUcb`: linear contextual bandit.
//!
//! **Operational primitives:**
//! - [`TriageSession`]: detect-then-triage — per-arm CUSUM detection wired to
//!   per-cell (`arm × context-bin`) investigation.
//! - [`WorstFirstConfig`] / [`worst_first_pick_k`]: post-detection investigation routing.
//! - [`CoverageConfig`] / [`coverage_pick_under_sampled`]: maintenance sampling floor.
//! - [`LatencyGuardrailConfig`]: hard pre-filter by mean latency.
//! - [`PipelineOrder`] / [`policy_plan_generic`] / [`policy_fill_generic`]: harness glue.
//! - [`ObservationId`]: caller-owned identity for out-of-order delayed labels.
//! - [`LoggedReward`] / [`ips_value`] / [`self_normalized_ips_value`]: scalar
//!   off-policy evaluation over logged rewards and propensities.
//!
//! **Non-goals:**
//! - Not a full bandit platform (no storage, full OPE pipelines, dashboards).
//! - `contextual` is intentionally a small, pragmatic policy module.
//!
//! # Statistical scope
//!
//! `muxer` composes several routing and monitoring primitives. Results proved for an
//! individual bandit or detector do not automatically transfer to this composition.
//! In particular, the crate does not claim a regret bound, a detection-delay bound,
//! or a system-wide false-alarm guarantee for [`Router`].
//!
//! [`MonitoredWindow`] compares two overlapping rolling windows, not a frozen
//! historical reference. Its scores are empirical signals whose sensitivity depends
//! on the capacity ratio. Wilson bounds are fixed-sample score adjustments, not
//! time-uniform confidence sequences under adaptive sampling.
//! [`DisjointMonitoredWindow`] is available when the descriptive comparison must
//! keep the baseline and recent samples separate.
//!
//! The `max_*` configuration fields are empirical routing filters. Some selection
//! paths deliberately fall back when every arm is filtered, so they are not safety
//! constraints or resource budgets. When legal, capability, or readiness eligibility
//! varies per request, the caller determines that set and passes it to
//! [`Router::select_from`], which keeps every selection stage inside it.
//!
#![forbid(unsafe_code)]
#![warn(missing_docs)]

use pare::Direction;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Caller-owned identity for one observed execution.
///
/// IDs let delayed labels update the observation that produced them instead of
/// relying on the latest observation for an arm. The caller must not reuse an ID
/// while the corresponding observation may still be retained by a [`Window`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ObservationId(u64);

impl ObservationId {
    /// Construct an observation ID from a caller-owned value.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Return the caller-owned value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

fn finite_or_zero(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

fn finite_scalar_weight(weight: f64, mut values: impl Iterator<Item = f64>) -> f64 {
    let weight = finite_or_zero(weight);
    if values.any(|value| !(weight * value).is_finite()) {
        0.0
    } else {
        weight
    }
}

mod decision;
pub use decision::{Decision, DecisionNote, DecisionPolicy};

mod policy;
#[cfg(feature = "stochastic")]
pub use policy::BanditPolicy;

mod alloc;
pub use alloc::softmax_map;

mod assessment;
pub use assessment::{
    select_candidate_assessments, CandidateAssessment, CandidateAssessmentDebug,
    CandidateAssessmentSelection, MetricObjective, MetricObjectiveValue,
};

mod utils;
pub use utils::suggested_window_cap;

mod ope;
pub use ope::{ips_value, self_normalized_ips_value, LoggedReward, OpeError};

mod control;
pub use control::{pick_control_arms, split_control_budget, ControlConfig};

mod router;
pub use router::{Router, RouterConfig, RouterDecision, RouterMode, RouterSnapshot};

mod guardrail;
pub use guardrail::LatencyGuardrailConfig;

pub mod monitor;

mod coverage;
pub use coverage::{coverage_pick_under_sampled, coverage_pick_under_sampled_idx, CoverageConfig};

#[cfg(feature = "stochastic")]
mod exp3ix;
#[cfg(feature = "stochastic")]
pub use exp3ix::{Exp3Ix, Exp3IxConfig, Exp3IxState};

#[cfg(feature = "stochastic")]
mod thompson;
#[cfg(feature = "stochastic")]
pub use thompson::{BetaStats, ThompsonConfig, ThompsonSampling, ThompsonState};

#[cfg(feature = "boltzmann")]
mod boltzmann;
#[cfg(feature = "boltzmann")]
pub use boltzmann::{BoltzmannConfig, BoltzmannPolicy};

#[cfg(feature = "contextual")]
mod contextual;
#[cfg(feature = "contextual")]
pub use contextual::{LinUcb, LinUcbArmState, LinUcbConfig, LinUcbScore, LinUcbState};

mod sticky;
pub use sticky::{StickyConfig, StickyMab};

mod stable_hash;
pub use stable_hash::stable_hash64;
pub(crate) use stable_hash::stable_hash64_u64;

mod novelty;
pub use novelty::novelty_pick_unseen;
pub(crate) use novelty::pick_random_subset;

mod prior;
pub use prior::apply_prior_counts_to_summary;

mod worst_first;
pub use worst_first::{
    context_bin, contextual_worst_first_pick_k, contextual_worst_first_pick_one,
    worst_first_pick_k, worst_first_pick_one, ContextBinConfig, ContextualCell,
    ContextualCoverageTracker, WorstFirstConfig,
};

mod harness;
pub use harness::{
    guardrail_filter_observed, guardrail_filter_observed_elapsed, policy_fill_generic,
    policy_fill_k_observed_guardrail_first_with_coverage, policy_fill_k_observed_with_coverage,
    policy_plan_generic, select_k_without_replacement_by, PipelineOrder, PolicyFill, PolicyPlan,
};
#[cfg(feature = "contextual")]
pub use harness::{policy_fill_k_contextual, ContextualPolicyFill};

mod triage;
pub use triage::{OutcomeIdx, TriageSession, TriageSessionConfig};

#[cfg(feature = "stochastic")]
pub use monitor::{calibrate_cusum_threshold, simulate_cusum_null_max_scores};
pub use monitor::{
    calibrate_threshold_from_max_scores, DisjointMonitoredWindow, DriftConfig, DriftMetric,
    MonitoredWindow, RateBoundMode, ThresholdCalibration, UncertaintyConfig,
};

/// A single observed outcome for an arm.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[non_exhaustive]
pub struct Outcome {
    /// Whether the request succeeded for this arm.
    pub ok: bool,
    /// Whether the result was below the caller's quality threshold.
    pub junk: bool,
    /// Whether the junk belongs to the caller's severe category.
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
    /// `None` (the default) means "not measured". Prefer recording a finalized
    /// outcome. [`Window::set_last_quality_score`] is only safe when no later
    /// outcome has entered that window.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub quality_score: Option<f64>,
}

impl Outcome {
    fn normalized(mut self) -> Self {
        self.junk |= self.hard_junk;
        self.quality_score = self
            .quality_score
            .filter(|score| score.is_finite())
            .map(|score| score.clamp(0.0, 1.0));
        self
    }

    /// Create an outcome with the `hard_junk => junk` invariant enforced.
    ///
    /// If `hard_junk` is true, `junk` is forced to true regardless of the
    /// passed value.  This prevents the silent bug where `soft_junk_rate`
    /// (`junk_rate - hard_junk_rate`) saturates to zero.
    pub fn new(ok: bool, junk: bool, hard_junk: bool, cost_units: u64, elapsed_ms: u64) -> Self {
        Self {
            ok,
            junk: junk || hard_junk,
            hard_junk,
            cost_units,
            elapsed_ms,
            quality_score: None,
        }
    }

    /// Create a successful outcome (ok=true, no junk).
    ///
    /// This is the most common case: the call succeeded with no quality issues.
    pub fn success(cost_units: u64, elapsed_ms: u64) -> Self {
        Self {
            ok: true,
            junk: false,
            hard_junk: false,
            cost_units,
            elapsed_ms,
            quality_score: None,
        }
    }

    /// Create a failed outcome (ok=false, hard_junk=true, junk=true).
    ///
    /// Use for complete failures: errors, timeouts, parse failures.
    pub fn failure(cost_units: u64, elapsed_ms: u64) -> Self {
        Self {
            ok: false,
            junk: true,
            hard_junk: true,
            cost_units,
            elapsed_ms,
            quality_score: None,
        }
    }

    /// Create a degraded-but-ok outcome (ok=true, junk=true, hard_junk=false).
    ///
    /// Use for soft quality failures: the call succeeded but the result
    /// was below the caller's quality threshold.
    pub fn degraded(cost_units: u64, elapsed_ms: u64) -> Self {
        Self {
            ok: true,
            junk: true,
            hard_junk: false,
            cost_units,
            elapsed_ms,
            quality_score: None,
        }
    }

    /// Create an outcome with a quality score, enforcing `hard_junk => junk`.
    /// Non-finite quality scores are treated as unmeasured.
    pub fn with_quality(
        ok: bool,
        junk: bool,
        hard_junk: bool,
        cost_units: u64,
        elapsed_ms: u64,
        quality_score: f64,
    ) -> Self {
        Self {
            ok,
            junk: junk || hard_junk,
            hard_junk,
            cost_units,
            elapsed_ms,
            quality_score: Some(quality_score),
        }
        .normalized()
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Outcome {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Raw {
            ok: bool,
            junk: bool,
            hard_junk: bool,
            cost_units: u64,
            elapsed_ms: u64,
            #[serde(default)]
            quality_score: Option<f64>,
        }
        let raw = Raw::deserialize(deserializer)?;
        Ok(Outcome {
            ok: raw.ok,
            junk: raw.junk || raw.hard_junk,
            hard_junk: raw.hard_junk,
            cost_units: raw.cost_units,
            elapsed_ms: raw.elapsed_ms,
            quality_score: raw.quality_score,
        }
        .normalized())
    }
}

/// Sliding-window statistics for an arm.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Window {
    cap: usize,
    buf: VecDeque<Outcome>,
    #[cfg_attr(feature = "serde", serde(default))]
    ids: VecDeque<Option<ObservationId>>,
}

impl Window {
    /// Create an empty window with capacity `cap` (minimum 1).
    pub fn new(cap: usize) -> Self {
        Self {
            cap: cap.max(1),
            buf: VecDeque::new(),
            ids: VecDeque::new(),
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
        self.push_with_optional_id(None, o);
    }

    /// Push a new outcome with caller-owned identity.
    ///
    /// The ID is retained alongside the outcome until that outcome leaves the
    /// window. Use [`Window::set_quality_score_for_id`] or
    /// [`Window::set_junk_level_for_id`] when a label arrives later.
    pub fn push_with_id(&mut self, id: ObservationId, o: Outcome) {
        self.push_with_optional_id(Some(id), o);
    }

    pub(crate) fn push_with_optional_id(&mut self, id: Option<ObservationId>, o: Outcome) {
        self.align_ids();
        if self.buf.len() == self.cap {
            self.buf.pop_front();
            self.ids.pop_front();
        }
        self.buf.push_back(o.normalized());
        self.ids.push_back(id);
    }

    pub(crate) fn pop_front(&mut self) -> Option<(Option<ObservationId>, Outcome)> {
        self.align_ids();
        let outcome = self.buf.pop_front()?;
        let id = self.ids.pop_front().unwrap_or(None);
        Some((id, outcome))
    }

    fn align_ids(&mut self) {
        while self.ids.len() < self.buf.len() {
            self.ids.push_front(None);
        }
        while self.ids.len() > self.buf.len() {
            self.ids.pop_front();
        }
    }

    pub(crate) fn contains_id(&mut self, id: ObservationId) -> bool {
        self.align_ids();
        self.ids.iter().any(|candidate| *candidate == Some(id))
    }

    /// Best-effort: set “junk” and whether it is “hard junk” for the most recent outcome.
    ///
    /// This is correct only if no later outcome has entered the window.
    pub fn set_last_junk_level(&mut self, junk: bool, hard_junk: bool) {
        if let Some(last) = self.buf.back_mut() {
            last.junk = junk;
            last.hard_junk = hard_junk && junk;
        }
    }

    /// Update junk labels for the identified outcome.
    ///
    /// Returns `true` when the ID is retained by this window.
    pub fn set_junk_level_for_id(
        &mut self,
        id: ObservationId,
        junk: bool,
        hard_junk: bool,
    ) -> bool {
        self.align_ids();
        for (candidate, outcome) in self.ids.iter().zip(self.buf.iter_mut()) {
            if *candidate == Some(id) {
                outcome.junk = junk;
                outcome.hard_junk = hard_junk && junk;
                return true;
            }
        }
        false
    }

    /// Best-effort: set the continuous quality score for the most recent outcome.
    ///
    /// Call this after downstream scoring completes (same pattern as
    /// [`Window::set_last_junk_level`]).  Finite values are clamped to
    /// `[0.0, 1.0]`; non-finite values are treated as unmeasured. This is
    /// correct only if no later outcome has entered the window.
    pub fn set_last_quality_score(&mut self, score: f64) {
        if let Some(last) = self.buf.back_mut() {
            last.quality_score = score.is_finite().then(|| score.clamp(0.0, 1.0));
        }
    }

    /// Update the quality score for the identified outcome.
    ///
    /// Returns `true` when the ID is retained by this window.
    pub fn set_quality_score_for_id(&mut self, id: ObservationId, score: f64) -> bool {
        self.align_ids();
        for (candidate, outcome) in self.ids.iter().zip(self.buf.iter_mut()) {
            if *candidate == Some(id) {
                outcome.quality_score = score.is_finite().then(|| score.clamp(0.0, 1.0));
                return true;
            }
        }
        false
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
        if count == 0 {
            None
        } else {
            Some(sum / count as f64)
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
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
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

/// How to extract an objective value from a [`Summary`].
///
/// Built-in extractors cover the standard `Outcome` fields. [`Extract::Custom`]
/// resolves only a fixed [`Objective::value`] shared by all candidates.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Extract {
    /// `ok_rate + UCB exploration bonus` (requires `exploration_c` from config).
    OkRateUcb,
    /// `Summary::mean_cost_units()`.
    MeanCost,
    /// `Summary::mean_elapsed_ms()`.
    MeanLatency,
    /// `Summary::hard_junk_rate()`.
    HardJunkRate,
    /// `Summary::soft_junk_rate()` (`junk_rate - hard_junk_rate`).
    SoftJunkRate,
    /// `Summary::mean_quality_score` (0.0 when absent).
    MeanQuality,
    /// Fixed caller-defined objective. The shared value can be set via
    /// [`Objective::value`]; extraction returns 0.0 as fallback.
    Custom,
}

impl Extract {
    /// Extract the raw value from a summary.
    ///
    /// `ucb` is the pre-computed UCB term; only used by `OkRateUcb`.
    /// For `Custom`, returns 0.0 (the caller should set `Objective::value` instead).
    #[must_use]
    pub fn apply(self, s: &Summary, ucb: f64) -> f64 {
        match self {
            Self::OkRateUcb => s.ok_rate() + ucb,
            Self::MeanCost => s.mean_cost_units(),
            Self::MeanLatency => s.mean_elapsed_ms(),
            Self::HardJunkRate => s.hard_junk_rate(),
            Self::SoftJunkRate => s.soft_junk_rate(),
            Self::MeanQuality => s
                .mean_quality_score
                .filter(|value| value.is_finite())
                .map(|value| value.clamp(0.0, 1.0))
                .unwrap_or(0.0),
            Self::Custom => 0.0,
        }
    }
}

/// A single objective dimension for Pareto selection.
///
/// Each objective contributes one axis to the Pareto frontier and one
/// term to the scalarized tiebreaker.  `direction` controls the frontier;
/// `weight` controls scalarization (higher weight = more influence).
///
/// [`Objective::value`] is one fixed override shared by every candidate in a
/// selection call. The current API cannot represent an arm-specific custom metric.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Objective {
    /// How to extract this objective from a `Summary`.
    pub extract: Extract,
    /// Whether higher values are better (`Maximize`) or worse (`Minimize`).
    pub direction: Direction,
    /// Scalarization weight (0 disables this objective in the tiebreaker
    /// but it still participates in the Pareto frontier).
    pub weight: f64,
    /// Pre-computed value override. When `Some`, this same value is used
    /// instead of extracting from every candidate's `Summary`.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub value: Option<f64>,
}

impl Objective {
    /// Create an objective that maximizes the extracted value.
    #[must_use]
    pub fn maximize(extract: Extract, weight: f64) -> Self {
        Self {
            extract,
            direction: Direction::Maximize,
            weight,
            value: None,
        }
    }

    /// Create an objective that minimizes the extracted value.
    #[must_use]
    pub fn minimize(extract: Extract, weight: f64) -> Self {
        Self {
            extract,
            direction: Direction::Minimize,
            weight,
            value: None,
        }
    }

    /// Create a caller-defined objective with a fixed value.
    ///
    /// The value applies to every arm in a selection call, so it cannot affect
    /// ranking by itself. Arm-specific custom metrics are not supported by the
    /// current selection API.
    #[must_use]
    pub fn custom(direction: Direction, weight: f64, value: f64) -> Self {
        Self {
            extract: Extract::Custom,
            direction,
            weight,
            value: Some(value),
        }
    }

    /// Override the extracted value with a pre-computed one.
    #[must_use]
    pub fn with_value(mut self, v: f64) -> Self {
        self.value = Some(v);
        self
    }

    /// Resolve the value: use the override if present, otherwise extract from summary.
    /// Non-finite values are resolved as `0.0` so unchecked caller input cannot
    /// poison scalarization or the Pareto frontier.
    #[must_use]
    pub fn resolve(&self, s: &Summary, ucb: f64) -> f64 {
        finite_or_zero(self.value.unwrap_or_else(|| self.extract.apply(s, ucb)))
    }

    /// Value oriented for Pareto (always maximize): negates for `Minimize` objectives.
    #[must_use]
    pub fn pareto_value(&self, s: &Summary, ucb: f64) -> f64 {
        let v = self.resolve(s, ucb);
        match self.direction {
            Direction::Maximize => v,
            Direction::Minimize => -v,
        }
    }

    /// Signed scalarization contribution (higher is always better).
    #[must_use]
    pub fn scalar_contribution(&self, s: &Summary, ucb: f64) -> f64 {
        let v = self.resolve(s, ucb);
        let weight = finite_or_zero(self.weight);
        finite_or_zero(match self.direction {
            Direction::Maximize => weight * v,
            Direction::Minimize => -(weight * v),
        })
    }
}

/// The default objective set for deterministic MAB selection.
///
/// Reproduces the pre-0.5 hardcoded behavior:
/// - Maximize `ok_rate + UCB` (weight 1.0)
/// - Minimize mean cost (scalar weight 0.0; remains a Pareto axis)
/// - Minimize mean latency (scalar weight 0.0; remains a Pareto axis)
/// - Minimize hard junk rate (scalar weight 0.0; remains a Pareto axis)
/// - Minimize soft junk rate (scalar weight 0.0; remains a Pareto axis)
/// - Maximize mean quality (scalar weight 0.0; remains a Pareto axis)
#[must_use]
pub fn default_objectives() -> Vec<Objective> {
    vec![
        Objective::maximize(Extract::OkRateUcb, 1.0),
        Objective::minimize(Extract::MeanCost, 0.0),
        Objective::minimize(Extract::MeanLatency, 0.0),
        Objective::minimize(Extract::HardJunkRate, 0.0),
        Objective::minimize(Extract::SoftJunkRate, 0.0),
        Objective::maximize(Extract::MeanQuality, 0.0),
    ]
}

/// Configuration knobs for deterministic MAB-style selection.
///
/// Contains the objective list, exploration coefficient, and empirical filters
/// used by all selection paths ([`select_mab`], [`select_mab_explain`],
/// [`select_mab_decide`]).
///
/// For monitored selection APIs (`select_mab_monitored_*`), use [`MonitoredMabConfig`]
/// which wraps this with additional monitoring-specific guards.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MabConfig {
    /// UCB exploration coefficient.
    pub exploration_c: f64,
    /// Objective dimensions for the Pareto frontier and scalarization.
    ///
    /// Each objective defines an axis (extract + direction) and a
    /// scalarization weight.  The default set ([`default_objectives`])
    /// reproduces pre-0.5 behavior.
    pub objectives: Vec<Objective>,
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
            objectives: default_objectives(),
            max_junk_rate: None,
            max_hard_junk_rate: None,
            max_mean_cost_units: None,
        }
    }
}

impl MabConfig {
    /// Set the weight for an objective matching the given extractor.
    /// If not found, the objective is not added (no-op).
    pub fn set_weight(&mut self, extract: Extract, weight: f64) {
        if let Some(obj) = self.objectives.iter_mut().find(|o| o.extract == extract) {
            obj.weight = weight;
        }
    }

    /// Builder: set cost weight.
    #[must_use]
    pub fn with_cost_weight(mut self, w: f64) -> Self {
        self.set_weight(Extract::MeanCost, w);
        self
    }

    /// Builder: set latency weight.
    #[must_use]
    pub fn with_latency_weight(mut self, w: f64) -> Self {
        self.set_weight(Extract::MeanLatency, w);
        self
    }

    /// Builder: set soft junk weight.
    #[must_use]
    pub fn with_junk_weight(mut self, w: f64) -> Self {
        self.set_weight(Extract::SoftJunkRate, w);
        self
    }

    /// Builder: set hard junk weight.
    #[must_use]
    pub fn with_hard_junk_weight(mut self, w: f64) -> Self {
        self.set_weight(Extract::HardJunkRate, w);
        self
    }

    /// Builder: set quality weight.
    #[must_use]
    pub fn with_quality_weight(mut self, w: f64) -> Self {
        self.set_weight(Extract::MeanQuality, w);
        self
    }

    /// Builder: set objectives directly (replaces default set).
    #[must_use]
    pub fn with_objectives(mut self, objectives: Vec<Objective>) -> Self {
        self.objectives = objectives;
        self
    }
}

/// Extended configuration for monitored MAB selection.
///
/// Includes all base [`MabConfig`] fields plus monitoring-specific guards
/// (drift, categorical KL, CUSUM) that only apply in `select_mab_monitored_*` APIs.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MonitoredMabConfig {
    /// Base selection configuration.
    pub base: MabConfig,
    /// Optional drift guard: discard arms whose drift score exceeds this.
    pub max_drift: Option<f64>,
    /// Drift metric used when applying `max_drift`.
    pub drift_metric: DriftMetric,
    /// Penalty weight for drift (0 disables).
    pub drift_weight: f64,
    /// Rate uncertainty configuration for Wilson bounds.
    pub uncertainty: UncertaintyConfig,
    /// Optional categorical KL guard threshold.
    pub max_catkl: Option<f64>,
    /// Dirichlet smoothing pseudo-count for categorical KL.
    pub catkl_alpha: f64,
    /// Minimum baseline samples for categorical KL.
    pub catkl_min_baseline: u64,
    /// Minimum recent samples for categorical KL.
    pub catkl_min_recent: u64,
    /// Penalty weight for categorical KL score (0 disables).
    pub catkl_weight: f64,
    /// Optional categorical CUSUM guard threshold.
    pub max_cusum: Option<f64>,
    /// Dirichlet smoothing pseudo-count for CUSUM.
    pub cusum_alpha: f64,
    /// Minimum baseline samples for CUSUM.
    pub cusum_min_baseline: u64,
    /// Minimum recent samples for CUSUM.
    pub cusum_min_recent: u64,
    /// Alternative distribution for CUSUM.
    pub cusum_alt_p: Option<[f64; 4]>,
    /// Penalty weight for CUSUM score (0 disables).
    pub cusum_weight: f64,
}

impl Default for MonitoredMabConfig {
    fn default() -> Self {
        Self {
            base: MabConfig::default(),
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

impl From<MabConfig> for MonitoredMabConfig {
    fn from(base: MabConfig) -> Self {
        Self {
            base,
            ..Self::default()
        }
    }
}

/// A resolved objective value for one candidate arm.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ObjectiveValue {
    /// The extractor that produced this value.
    pub extract: Extract,
    /// The resolved value (after applying overrides / uncertainty bounds).
    pub value: f64,
    /// The Pareto-oriented value (negated for `Minimize` objectives).
    pub pareto_value: f64,
    /// Scalarization contribution (signed, higher is better).
    pub scalar_contribution: f64,
}

/// Debug row for one candidate arm.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CandidateDebug {
    /// Arm name.
    pub name: String,
    /// Summary snapshot for this arm.
    pub summary: Summary,
    /// UCB exploration term.
    pub ucb: f64,
    /// Per-objective resolved values (in the same order as `MabConfig::objectives`).
    pub objective_values: Vec<ObjectiveValue>,
    /// Total scalarized score (sum of all `scalar_contribution` values).
    pub score: f64,

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
/// This exists because callers may need more than "which arm": they may also
/// need to know whether constraints eliminated all arms (fallback) and
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
    // Apply empirical filters.
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
    let zero_summary = Summary::default();
    let obj_values: Vec<ObjectiveValue> = cfg
        .objectives
        .iter()
        .map(|obj| ObjectiveValue {
            extract: obj.extract,
            value: 0.0,
            pareto_value: 0.0,
            scalar_contribution: 0.0,
        })
        .collect();
    let sel = Selection {
        chosen: chosen.clone(),
        frontier: vec![chosen.clone()],
        candidates: vec![CandidateDebug {
            name: chosen,
            summary: zero_summary,
            ucb: 0.0,
            objective_values: obj_values,
            score: 0.0,
            drift_score: None,
            catkl_score: None,
            cusum_score: None,
            ok_half_width: None,
            junk_half_width: None,
            hard_junk_half_width: None,
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

    let total_calls: f64 = arms_in_order
        .iter()
        .map(|a| summaries.get(a).copied().unwrap_or_default().calls as f64)
        .sum::<f64>()
        .max(1.0);

    let mut candidates = Vec::new();

    for a in arms_in_order {
        let s = summaries.get(a).copied().unwrap_or_default();
        let n = (s.calls as f64).max(1.0);
        let ucb = finite_or_zero(cfg.exploration_c) * ((total_calls.ln() / n).sqrt());

        let obj_values: Vec<ObjectiveValue> = cfg
            .objectives
            .iter()
            .map(|obj| {
                let value = obj.resolve(&s, ucb);
                ObjectiveValue {
                    extract: obj.extract,
                    value,
                    pareto_value: obj.pareto_value(&s, ucb),
                    scalar_contribution: obj.scalar_contribution(&s, ucb),
                }
            })
            .collect();

        let score: f64 = obj_values.iter().map(|o| o.scalar_contribution).sum();

        candidates.push(CandidateDebug {
            name: a.clone(),
            summary: s,
            ucb,
            objective_values: obj_values,
            score,
            drift_score: None,
            catkl_score: None,
            cusum_score: None,
            ok_half_width: None,
            junk_half_width: None,
            hard_junk_half_width: None,
        });
    }

    let mut assessment_names = BTreeSet::new();
    let assessments: Vec<CandidateAssessment> = candidates
        .iter()
        .filter(|candidate| assessment_names.insert(candidate.name.clone()))
        .map(|candidate| {
            CandidateAssessment::new(
                candidate.name.clone(),
                candidate.summary.calls,
                candidate
                    .objective_values
                    .iter()
                    .map(|value| value.value)
                    .collect(),
            )
        })
        .collect();
    let metric_objectives: Vec<MetricObjective> = cfg
        .objectives
        .iter()
        .enumerate()
        .map(|(metric, objective)| MetricObjective {
            metric,
            direction: objective.direction,
            weight: finite_scalar_weight(
                objective.weight,
                candidates
                    .iter()
                    .map(|candidate| candidate.objective_values[metric].value),
            ),
        })
        .collect();
    let generic = select_candidate_assessments(&assessments, &metric_objectives)
        .expect("quality summary adapter produces valid candidate assessments");
    let best_name = generic
        .chosen
        .unwrap_or_else(|| arms_in_order.first().cloned().unwrap_or_default());
    let frontier_names = generic.frontier;

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
/// This is intended for callers that already maintain `MonitoredWindow`s per arm.
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
    cfg: MonitoredMabConfig,
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

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct MonitoringScores {
    drift_score: Option<f64>,
    catkl_score: Option<f64>,
    cusum_score: Option<f64>,
}

fn nonnegative_threshold(x: Option<f64>) -> Option<f64> {
    x.and_then(|x| (x.is_finite() && x >= 0.0).then_some(x))
}

fn monitored_catkl_alpha(cfg: &MonitoredMabConfig) -> f64 {
    if cfg.catkl_alpha.is_finite() && cfg.catkl_alpha > 0.0 {
        cfg.catkl_alpha
    } else {
        1e-3
    }
}

fn monitored_cusum_alpha(cfg: &MonitoredMabConfig) -> f64 {
    if cfg.cusum_alpha.is_finite() && cfg.cusum_alpha > 0.0 {
        cfg.cusum_alpha
    } else {
        1e-3
    }
}

fn monitored_cusum_alt_p(cfg: &MonitoredMabConfig) -> [f64; 4] {
    cfg.cusum_alt_p.unwrap_or([0.05, 0.05, 0.45, 0.45])
}

pub(crate) fn monitoring_scores_for_arms(
    arms_in_order: &[String],
    monitored: &BTreeMap<String, MonitoredWindow>,
    drift_cfg: DriftConfig,
    cfg: &MonitoredMabConfig,
) -> BTreeMap<String, MonitoringScores> {
    let catkl_alpha = monitored_catkl_alpha(cfg);
    let cusum_alpha = monitored_cusum_alpha(cfg);
    let cusum_alt_p = monitored_cusum_alt_p(cfg);

    arms_in_order
        .iter()
        .filter_map(|a| {
            let w = monitored.get(a)?;
            let drift_score = monitor::drift_between_windows(
                w.baseline(),
                w.recent(),
                DriftConfig {
                    metric: cfg.drift_metric,
                    ..drift_cfg
                },
            )
            .map(|x| x.score);
            let catkl_score = monitor::catkl_score_between_windows(
                w.baseline(),
                w.recent(),
                catkl_alpha,
                drift_cfg.tol,
                cfg.catkl_min_baseline,
                cfg.catkl_min_recent,
            );
            let cusum_score = monitor::cusum_score_between_windows(
                w.baseline(),
                w.recent(),
                cusum_alpha,
                drift_cfg.tol,
                cfg.cusum_min_baseline,
                cfg.cusum_min_recent,
                Some(cusum_alt_p),
            );
            Some((
                a.clone(),
                MonitoringScores {
                    drift_score,
                    catkl_score,
                    cusum_score,
                },
            ))
        })
        .collect()
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
    cfg: MonitoredMabConfig,
) -> MabSelectionDecision {
    let monitoring_scores = monitoring_scores_for_arms(arms_in_order, monitored, drift_cfg, &cfg);
    select_mab_monitored_explain_with_summaries_and_scores(
        arms_in_order,
        summaries,
        &monitoring_scores,
        &cfg,
    )
}

pub(crate) fn select_mab_monitored_explain_with_summaries_and_scores(
    arms_in_order: &[String],
    summaries: &BTreeMap<String, Summary>,
    monitoring_scores: &BTreeMap<String, MonitoringScores>,
    cfg: &MonitoredMabConfig,
) -> MabSelectionDecision {
    let base = &cfg.base;

    // Apply base empirical filters first (same semantics as `select_mab_explain`).
    let (eligible_arms, constraints_fallback_used) =
        apply_base_constraints(arms_in_order, summaries, base);
    let arms_in_order: &[String] = &eligible_arms;

    // Explore first (stable order).
    let explore_choice: Option<String> = arms_in_order
        .iter()
        .find(|a| summaries.get(*a).copied().unwrap_or_default().calls == 0)
        .cloned();
    if let Some(chosen) = explore_choice {
        return explore_first_decision(
            chosen,
            eligible_arms,
            constraints_fallback_used,
            base.clone(),
        );
    }

    // Apply drift guard (optional) over the constraint-eligible set.
    let max_drift = nonnegative_threshold(cfg.max_drift);
    let max_catkl = nonnegative_threshold(cfg.max_catkl);
    let catkl_alpha = monitored_catkl_alpha(cfg);
    let max_cusum = nonnegative_threshold(cfg.max_cusum);
    let cusum_alpha = monitored_cusum_alpha(cfg);
    let cusum_alt_p = monitored_cusum_alt_p(cfg);
    let mut eligible_after_drift = arms_in_order.to_vec();
    let mut drift_guard: Option<DriftGuardDecision> = None;
    if let Some(thr) = max_drift {
        let mut kept: Vec<String> = Vec::new();
        for a in arms_in_order {
            let violates = monitoring_scores
                .get(a)
                .and_then(|s| s.drift_score)
                .map(|x| x > thr)
                .unwrap_or(false);
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
    let mut eligible_after_catkl = eligible_after_drift.clone();
    let mut catkl_guard: Option<CatKlGuardDecision> = None;
    if let Some(thr) = max_catkl {
        let mut kept: Vec<String> = Vec::new();
        for a in &eligible_after_drift {
            let violates = monitoring_scores
                .get(a)
                .and_then(|s| s.catkl_score)
                .map(|x| x > thr)
                .unwrap_or(false);
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
    let mut eligible_after_cusum = eligible_after_catkl.clone();
    let mut cusum_guard: Option<CusumGuardDecision> = None;
    if let Some(thr) = max_cusum {
        let mut kept: Vec<String> = Vec::new();
        for a in &eligible_after_catkl {
            let violates = monitoring_scores
                .get(a)
                .and_then(|s| s.cusum_score)
                .map(|x| x > thr)
                .unwrap_or(false);
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

    // Monitored Pareto frontier: base objectives (with Wilson-bounded overrides)
    // plus monitoring objectives whose weights are positive.
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

        let scores = monitoring_scores.get(a).copied().unwrap_or_default();
        let drift_score = scores.drift_score;
        let catkl_score = scores.catkl_score;
        let cusum_score = scores.cusum_score;

        let ucb = finite_or_zero(base.exploration_c) * ((total_calls.ln() / n).sqrt());

        // Build base objective values with Wilson-bounded overrides.
        let mut obj_values: Vec<ObjectiveValue> = base
            .objectives
            .iter()
            .map(|obj| {
                // Override rate-based extractors with Wilson-bounded values.
                let value = finite_or_zero(match obj.extract {
                    Extract::OkRateUcb => ok_rate_used + ucb,
                    Extract::HardJunkRate => hard_used,
                    Extract::SoftJunkRate => soft_used,
                    _ => obj.resolve(&s, ucb),
                });
                let pv = match obj.direction {
                    Direction::Maximize => value,
                    Direction::Minimize => -value,
                };
                let weight = finite_or_zero(obj.weight);
                let sc = finite_or_zero(match obj.direction {
                    Direction::Maximize => weight * value,
                    Direction::Minimize => -(weight * value),
                });
                ObjectiveValue {
                    extract: obj.extract,
                    value,
                    pareto_value: pv,
                    scalar_contribution: sc,
                }
            })
            .collect();

        // Append monitoring objectives with pre-computed values.
        let monitored_values = [
            (cfg.drift_weight, finite_or_zero(drift_score.unwrap_or(0.0))),
            (cfg.catkl_weight, finite_or_zero(catkl_score.unwrap_or(0.0))),
            (cfg.cusum_weight, finite_or_zero(cusum_score.unwrap_or(0.0))),
        ];
        for (weight, value) in monitored_values {
            if !weight.is_finite() || weight <= 0.0 {
                continue;
            }
            obj_values.push(ObjectiveValue {
                extract: Extract::Custom,
                value,
                pareto_value: -value,
                scalar_contribution: -(weight * value),
            });
        }

        let score: f64 = obj_values.iter().map(|o| o.scalar_contribution).sum();

        candidates.push(CandidateDebug {
            name: a.clone(),
            summary: s,
            ucb,
            objective_values: obj_values,
            score,
            drift_score,
            catkl_score,
            cusum_score,
            ok_half_width: Some(ok_half),
            junk_half_width: Some(soft_half),
            hard_junk_half_width: Some(hard_half),
        });
    }

    let mut assessment_names = BTreeSet::new();
    let assessments: Vec<CandidateAssessment> = candidates
        .iter()
        .filter(|candidate| assessment_names.insert(candidate.name.clone()))
        .map(|candidate| {
            CandidateAssessment::new(
                candidate.name.clone(),
                candidate.summary.calls,
                candidate
                    .objective_values
                    .iter()
                    .map(|value| value.value)
                    .collect(),
            )
        })
        .collect();
    let mut metric_objectives: Vec<MetricObjective> = base
        .objectives
        .iter()
        .enumerate()
        .map(|(metric, objective)| MetricObjective {
            metric,
            direction: objective.direction,
            weight: finite_scalar_weight(
                objective.weight,
                candidates
                    .iter()
                    .map(|candidate| candidate.objective_values[metric].value),
            ),
        })
        .collect();
    let mut next_metric = metric_objectives.len();
    for weight in [cfg.drift_weight, cfg.catkl_weight, cfg.cusum_weight] {
        if weight.is_finite() && weight > 0.0 {
            metric_objectives.push(MetricObjective::minimize(next_metric, weight));
            next_metric += 1;
        }
    }
    let generic = select_candidate_assessments(&assessments, &metric_objectives)
        .expect("monitored adapter produces valid candidate assessments");
    let best_name = generic
        .chosen
        .unwrap_or_else(|| eligible_after_cusum.first().cloned().unwrap_or_default());
    let frontier_names = generic.frontier;

    let sel = Selection {
        chosen: best_name,
        frontier: frontier_names,
        candidates,
        config: base.clone(),
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

/// Unified decision envelope for deterministic MAB selection.
///
/// This is a convenience wrapper around `select_mab_explain` that returns a `Decision`
/// suitable for consistent diagnostics across policies.
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
    cfg: MonitoredMabConfig,
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
            mean_quality_score: c.summary.mean_quality_score,
        });
    }

    Decision {
        policy: DecisionPolicy::Mab,
        chosen: d.selection.chosen.clone(),
        probs: None,
        notes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn mk_test_candidate(name: &str, score: f64) -> CandidateDebug {
        CandidateDebug {
            name: name.to_string(),
            summary: Summary::default(),
            ucb: 0.0,
            objective_values: vec![],
            score,
            drift_score: None,
            catkl_score: None,
            cusum_score: None,
            ok_half_width: None,
            junk_half_width: None,
            hard_junk_half_width: None,
        }
    }

    fn mk_test_candidate_with_calls(name: &str, calls: u64, score: f64) -> CandidateDebug {
        CandidateDebug {
            name: name.to_string(),
            summary: Summary {
                calls,
                ..Summary::default()
            },
            ucb: 0.0,
            objective_values: vec![],
            score,
            drift_score: None,
            catkl_score: None,
            cusum_score: None,
            ok_half_width: None,
            junk_half_width: None,
            hard_junk_half_width: None,
        }
    }

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
    fn preference_reweighting_flips_route_without_recomputing_stats() {
        // BaRP's "one policy, many trade-offs" property (Bandit-feedback
        // Routing with Preferences, arXiv:2510.07429): the operator dials the
        // performance/cost trade-off at inference time without retraining.
        //
        // Here the per-arm `Summary` IS the learned state and the objective
        // weight vector IS the preference. The same summaries, scored under a
        // quality-leaning vs a cost-leaning weight vector, must produce
        // different routes -- with no change to the arm statistics.
        let arms = vec!["premium".to_string(), "budget".to_string()];
        let mut m = BTreeMap::new();
        // premium: higher ok-rate (0.95), higher mean cost (10/call).
        m.insert("premium".to_string(), s(20, 19, 0, 0, 200, 2000));
        // budget: lower ok-rate (0.75), lower mean cost (1/call).
        m.insert("budget".to_string(), s(20, 15, 0, 0, 20, 2000));

        // exploration_c = 0 removes the UCB bonus so the flip is attributable
        // to the preference weights alone, not to exploration tie-breaks.
        let quality_pref = MabConfig {
            exploration_c: 0.0,
            objectives: vec![
                Objective::maximize(Extract::OkRateUcb, 1.0),
                Objective::minimize(Extract::MeanCost, 0.01),
            ],
            ..MabConfig::default()
        };
        let cost_pref = MabConfig {
            exploration_c: 0.0,
            objectives: vec![
                Objective::maximize(Extract::OkRateUcb, 1.0),
                Objective::minimize(Extract::MeanCost, 1.0),
            ],
            ..MabConfig::default()
        };

        let q = select_mab(&arms, &m, quality_pref);
        let c = select_mab(&arms, &m, cost_pref);
        assert_eq!(
            q.chosen, "premium",
            "quality-leaning preference routes to premium"
        );
        assert_eq!(
            c.chosen, "budget",
            "cost-leaning preference routes to budget"
        );
        // Both arms sit on the Pareto frontier (neither dominates the other),
        // so it is the scalarization weights -- the preference -- that decide.
        assert!(q.frontier.iter().any(|a| a == "premium"));
        assert!(q.frontier.iter().any(|a| a == "budget"));
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
    fn zero_monitoring_weights_do_not_add_pareto_axes() {
        let arms = vec!["a".to_string(), "b".to_string()];
        let summaries = BTreeMap::from([
            ("a".to_string(), s(10, 9, 0, 0, 10, 1000)),
            ("b".to_string(), s(10, 9, 0, 0, 10, 1000)),
        ]);
        let scores = BTreeMap::from([
            (
                "a".to_string(),
                MonitoringScores {
                    drift_score: Some(10.0),
                    ..MonitoringScores::default()
                },
            ),
            (
                "b".to_string(),
                MonitoringScores {
                    drift_score: Some(0.0),
                    ..MonitoringScores::default()
                },
            ),
        ]);

        let disabled = MonitoredMabConfig::default();
        let base_len = disabled.base.objectives.len();
        let decision = select_mab_monitored_explain_with_summaries_and_scores(
            &arms, &summaries, &scores, &disabled,
        );
        assert_eq!(decision.selection.frontier, arms);
        assert!(decision
            .selection
            .candidates
            .iter()
            .all(|candidate| candidate.objective_values.len() == base_len));

        let enabled = MonitoredMabConfig {
            drift_weight: 1.0,
            ..MonitoredMabConfig::default()
        };
        let decision = select_mab_monitored_explain_with_summaries_and_scores(
            &arms, &summaries, &scores, &enabled,
        );
        assert_eq!(decision.selection.chosen, "b");
        assert_eq!(decision.selection.frontier, vec!["b".to_string()]);
    }

    #[test]
    fn non_finite_objective_inputs_do_not_reach_pareto_frontier() {
        let arms = vec!["a".to_string(), "b".to_string()];
        let summaries = BTreeMap::from([
            (
                "a".to_string(),
                Summary {
                    calls: 1,
                    ok: 1,
                    mean_quality_score: Some(f64::NAN),
                    ..Summary::default()
                },
            ),
            ("b".to_string(), s(1, 1, 0, 0, 0, 0)),
        ]);
        let cfg = MabConfig {
            exploration_c: f64::NAN,
            objectives: vec![
                Objective::maximize(Extract::MeanQuality, f64::INFINITY),
                Objective::custom(Direction::Maximize, 1.0, f64::NEG_INFINITY),
            ],
            ..MabConfig::default()
        };

        let decision = select_mab_explain(&arms, &summaries, cfg.clone());
        assert!(arms.contains(&decision.selection.chosen));
        for candidate in &decision.selection.candidates {
            assert!(candidate.ucb.is_finite());
            assert!(candidate.score.is_finite());
            assert!(candidate.objective_values.iter().all(|value| {
                value.value.is_finite()
                    && value.pareto_value.is_finite()
                    && value.scalar_contribution.is_finite()
            }));
        }

        let monitored = MonitoredMabConfig {
            base: cfg,
            ..MonitoredMabConfig::default()
        };
        let decision = select_mab_monitored_explain_with_summaries_and_scores(
            &arms,
            &summaries,
            &BTreeMap::new(),
            &monitored,
        );
        assert!(arms.contains(&decision.selection.chosen));
        for candidate in decision.selection.candidates {
            assert!(candidate.ucb.is_finite());
            assert!(candidate.score.is_finite());
            assert!(candidate.objective_values.iter().all(|value| {
                value.value.is_finite()
                    && value.pareto_value.is_finite()
                    && value.scalar_contribution.is_finite()
            }));
        }
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

            let sel = select_mab(&arms, &m, cfg.clone());
            prop_assert!(sel.chosen == "a" || sel.chosen == "b");
            prop_assert!(sel.frontier.iter().any(|x| x == &sel.chosen));

            // Determinism: same input -> same output.
            let sel2 = select_mab(&arms, &m, cfg.clone());
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
            let sel1 = select_mab(&arms, &m, cfg.clone());

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
        let e1 = sticky.apply_mab(select_mab_explain(&arms, &m1, cfg.clone()));
        assert_eq!(e1.chosen, "a");
        assert_eq!(sticky.dwell(), 1);

        // Now "b" is better, but dwell gate should keep "a" for 2 more decisions.
        let mut m2 = BTreeMap::new();
        m2.insert("a".to_string(), s(10, 5, 0, 0, 0, 0));
        m2.insert("b".to_string(), s(10, 10, 0, 0, 0, 0));

        let e2 = sticky.apply_mab(select_mab_explain(&arms, &m2, cfg.clone()));
        assert_eq!(e2.chosen, "a");
        let e3 = sticky.apply_mab(select_mab_explain(&arms, &m2, cfg.clone()));
        assert_eq!(e3.chosen, "a");

        // Next decision: allowed to switch.
        let e4 = sticky.apply_mab(select_mab_explain(&arms, &m2, cfg));
        assert_eq!(e4.chosen, "b");
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
                        mk_test_candidate_with_calls("a", 10, a_score),
                        mk_test_candidate_with_calls("b", 10, b_score),
                    ],
                    config: cfg.clone(),
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
        assert_eq!(e1.chosen, "a");
        assert_eq!(sticky.previous(), Some("a"));

        // Candidate "b" is only slightly better: margin < 0.5 => keep "a".
        let e2 = sticky.apply_mab(mk("b", 1.0, 1.4));
        assert_eq!(e2.chosen, "a");

        // Candidate "b" is much better: margin >= 0.5 => switch to "b".
        let e3 = sticky.apply_mab(mk("b", 1.0, 1.7));
        assert_eq!(e3.chosen, "b");
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
                candidates: vec![mk_test_candidate("old", 0.0)],
                config: cfg.clone(),
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
            candidates: vec![mk_test_candidate_with_calls("a", 10, 0.0)],
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
        assert_eq!(e.chosen, "a");
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
