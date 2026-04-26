//! Unified decision envelope for policy outputs.
//!
//! Many routing systems want a single, audit-friendly record of a policy decision that can be:
//! - logged (debugging / monitoring)
//! - replayed (offline evaluation)
//! - consumed by wrappers (e.g. stickiness) without heuristics
//!
//! This module provides a small `Decision` struct and a typed `DecisionNote` list that policies
//! can attach to explain "why this choice happened".

use std::collections::BTreeMap;

/// Which policy produced a decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum DecisionPolicy {
    /// Deterministic multi-objective MAB (Pareto + scalarization).
    Mab,
    /// Adversarial bandit (EXP3-IX).
    Exp3Ix,
    /// Thompson sampling (Beta-Bernoulli posterior).
    Thompson,
    /// Linear contextual bandit (LinUCB).
    LinUcb,
    /// Boltzmann (softmax-temperature) policy via Gumbel-max sampling.
    Boltzmann,
}

/// Audit-friendly notes attached to a decision.
///
/// Notes are intentionally small, typed, and stable. Prefer adding new variants
/// over changing existing semantics.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum DecisionNote {
    /// Policy selected an untried arm (stable-order exploration).
    ExploreFirst,

    /// Policy sampled from a probability distribution to choose an arm.
    SampledFromDistribution,

    /// Policy sampled per-arm posteriors / scores and chose the max.
    SampledPosteriorMax,

    /// Policy chose deterministically (argmax / scalarization), with stable tie-breaks.
    DeterministicChoice,

    /// Constraints were applied to filter arms before selection.
    ///
    /// If constraints filtered all arms, `fallback_used=true` and `eligible_arms` is set to the
    /// original arm list (never empty).
    Constraints {
        /// Arms that passed the constraint filter.
        eligible_arms: Vec<String>,
        /// Whether the constraint filtered all arms and selection fell back to the full set.
        fallback_used: bool,
    },

    /// Drift guardrail (change-monitoring) filtered arms before selection.
    ///
    /// If it filtered all arms, `fallback_used=true` and `eligible_arms` is set to the
    /// original arm list (never empty).
    DriftGuard {
        /// Arms that passed the drift guard.
        eligible_arms: Vec<String>,
        /// Whether fallback to the full set was used.
        fallback_used: bool,
        /// Drift metric used for filtering.
        metric: crate::monitor::DriftMetric,
        /// Maximum drift threshold that was applied.
        max_drift: f64,
    },

    /// Categorical KL guard filtered arms before selection.
    ///
    /// This guard uses the statistic `S = n_recent * KL(q_recent || p0_baseline)`.
    CatKlGuard {
        /// Arms that passed the catKL guard.
        eligible_arms: Vec<String>,
        /// Whether fallback to the full set was used.
        fallback_used: bool,
        /// Maximum catKL statistic threshold.
        max_catkl: f64,
        /// Significance level for the test.
        alpha: f64,
        /// Minimum baseline observations required before the guard activates.
        min_baseline: u64,
        /// Minimum recent observations required before the guard activates.
        min_recent: u64,
    },

    /// Categorical CUSUM guard filtered arms before selection.
    ///
    /// This guard uses a CUSUM score over log-likelihood ratios between `p1` and `p0`.
    CusumGuard {
        /// Arms that passed the CUSUM guard.
        eligible_arms: Vec<String>,
        /// Whether fallback to the full set was used.
        fallback_used: bool,
        /// Maximum CUSUM score threshold.
        max_cusum: f64,
        /// Significance level for the test.
        alpha: f64,
        /// Minimum baseline observations required.
        min_baseline: u64,
        /// Minimum recent observations required.
        min_recent: u64,
        /// Alternative hypothesis categorical distribution.
        alt_p: [f64; 4],
    },

    /// Monitoring/uncertainty diagnostics for the chosen arm (when available).
    Diagnostics {
        /// Drift score between baseline and recent windows.
        #[cfg_attr(
            feature = "serde",
            serde(default, skip_serializing_if = "Option::is_none")
        )]
        drift_score: Option<f64>,
        /// Categorical KL divergence statistic.
        #[cfg_attr(
            feature = "serde",
            serde(default, skip_serializing_if = "Option::is_none")
        )]
        catkl_score: Option<f64>,
        /// Maximum CUSUM score across alternative hypotheses.
        #[cfg_attr(
            feature = "serde",
            serde(default, skip_serializing_if = "Option::is_none")
        )]
        cusum_score: Option<f64>,
        /// Wilson half-width for the ok rate estimate.
        #[cfg_attr(
            feature = "serde",
            serde(default, skip_serializing_if = "Option::is_none")
        )]
        ok_half_width: Option<f64>,
        /// Wilson half-width for the junk rate estimate.
        #[cfg_attr(
            feature = "serde",
            serde(default, skip_serializing_if = "Option::is_none")
        )]
        junk_half_width: Option<f64>,
        /// Wilson half-width for the hard-junk rate estimate.
        #[cfg_attr(
            feature = "serde",
            serde(default, skip_serializing_if = "Option::is_none")
        )]
        hard_junk_half_width: Option<f64>,
        /// Mean quality score for the chosen arm (when quality_score has been recorded).
        #[cfg_attr(
            feature = "serde",
            serde(default, skip_serializing_if = "Option::is_none")
        )]
        mean_quality_score: Option<f64>,
    },

    /// Numerical / CDF fallthrough required choosing the last arm as a safe fallback.
    NumericalFallbackToLastArm,

    /// Stickiness kept the previous arm due to a dwell gate.
    StickyKeptPreviousDwell {
        /// Arm that was kept (the incumbent).
        previous: String,
        /// Arm that was considered but rejected.
        candidate: String,
        /// Current dwell count on the previous arm.
        dwell: u64,
        /// Minimum dwell required before switching is allowed.
        min_dwell: u64,
    },

    /// Stickiness kept the previous arm because the candidate advantage was too small.
    StickyKeptPreviousMargin {
        /// Arm that was kept (the incumbent).
        previous: String,
        /// Arm that was considered but rejected.
        candidate: String,
        /// Scalarized score of the previous arm.
        previous_score: f64,
        /// Scalarized score of the candidate arm.
        candidate_score: f64,
        /// Observed margin (candidate_score - previous_score).
        margin: f64,
        /// Minimum margin required to switch.
        min_margin: f64,
    },

    /// Stickiness switched away from the previous arm.
    StickySwitched {
        /// Arm that was abandoned.
        previous: String,
        /// Arm that was selected as the new incumbent.
        candidate: String,
        /// Scalarized score of the previous arm.
        previous_score: f64,
        /// Scalarized score of the candidate arm.
        candidate_score: f64,
        /// Observed margin (candidate_score - previous_score).
        margin: f64,
    },
}

/// A single policy decision in a unified envelope.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Decision {
    /// The policy that produced this decision.
    pub policy: DecisionPolicy,
    /// The selected arm name.
    pub chosen: String,
    /// Optional per-arm probabilities (when the policy has a distribution).
    pub probs: Option<BTreeMap<String, f64>>,
    /// Audit notes describing why this choice happened.
    pub notes: Vec<DecisionNote>,
}
