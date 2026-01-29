//! Unified decision envelope for policy outputs.
//!
//! Many routing systems want a single, audit-friendly record of a policy decision that can be:
//! - logged (debugging / monitoring)
//! - replayed (offline evaluation)
//! - consumed by wrappers (e.g. stickiness) without heuristics
//!
//! This module provides a small `Decision` struct and a typed `DecisionNote` list that policies
//! can attach to explain “why this choice happened”.

use std::collections::BTreeMap;

/// Which policy produced a decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DecisionPolicy {
    Mab,
    Exp3Ix,
    Thompson,
    LinUcb,
}

/// Audit-friendly notes attached to a decision.
///
/// Notes are intentionally small, typed, and stable. Prefer adding new variants
/// over changing existing semantics.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
        eligible_arms: Vec<String>,
        fallback_used: bool,
    },

    /// Numerical / CDF fallthrough required choosing the last arm as a safe fallback.
    NumericalFallbackToLastArm,

    /// Stickiness kept the previous arm due to a dwell gate.
    StickyKeptPreviousDwell {
        previous: String,
        candidate: String,
        dwell: u64,
        min_dwell: u64,
    },

    /// Stickiness kept the previous arm because the candidate advantage was too small.
    StickyKeptPreviousMargin {
        previous: String,
        candidate: String,
        previous_score: f64,
        candidate_score: f64,
        margin: f64,
        min_margin: f64,
    },

    /// Stickiness switched away from the previous arm.
    StickySwitched {
        previous: String,
        candidate: String,
        previous_score: f64,
        candidate_score: f64,
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
