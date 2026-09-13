//! Checked projection of retained interactions into scalar off-policy rows.
//!
//! These helpers do not recover missing contexts or establish target-policy
//! support. The caller resolves the original context using the immutable
//! receipt, then supplies the target probability for its observed action.
//! Exclusion counts are essential: dropping missing rewards can change the
//! estimand and bias an estimate. This module does not correct missingness.

use crate::{
    Channel, DecisionId, DecisionReceipt, InteractionPolicy, LoggedReward, Muxer, Probability,
    ProbabilityAvailability,
};
use std::collections::BTreeMap;
use std::fmt;

/// Semantics of a profile's validated, retained feedback.
///
/// Implement this on canonical feedback, never on a caller-provided receipt
/// or an unvalidated wire payload. Custom implementations are trusted to state
/// their domain's execution and reward semantics honestly.
pub trait EvaluationFeedback {
    /// A scalar reward, if this feedback contains one.
    fn scalar_reward(&self) -> Option<f64>;
    /// Whether this value confirms that the selected action was executed.
    fn confirms_execution(&self) -> bool;
}

impl EvaluationFeedback for bool {
    fn scalar_reward(&self) -> Option<f64> {
        Some(if *self { 1.0 } else { 0.0 })
    }
    fn confirms_execution(&self) -> bool {
        true
    }
}

impl EvaluationFeedback for crate::profiles::BoundedReward {
    fn scalar_reward(&self) -> Option<f64> {
        Some(self.get())
    }
    fn confirms_execution(&self) -> bool {
        true
    }
}

impl EvaluationFeedback for crate::profiles::FiniteReward {
    fn scalar_reward(&self) -> Option<f64> {
        Some(self.get())
    }
    fn confirms_execution(&self) -> bool {
        true
    }
}

impl EvaluationFeedback for crate::profiles::quality::CanonicalQualityFeedback {
    fn scalar_reward(&self) -> Option<f64> {
        self.scalar_reward()
    }
    fn confirms_execution(&self) -> bool {
        self.is_execution()
    }
}

/// Why an interaction could not be projected into a scalar OPE row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[non_exhaustive]
pub enum ProjectionError {
    /// The engine does not retain this decision (or it belongs to another engine).
    NotRetained,
    /// Scalar estimators do not accept ordered batch receipts.
    BatchUnsupported,
    /// No exact, positive probability is available for the executed action.
    ProbabilityUnavailable,
    /// No retained final channel confirms execution.
    ExecutionUnconfirmed,
    /// The requested reward channel has no retained final scalar reward.
    FinalRewardUnavailable,
    /// The custom feedback implementation returned a nonfinite reward.
    InvalidReward,
    /// The caller cannot resolve decision-time context or target-policy evidence.
    TargetEvidenceUnavailable,
}

impl fmt::Display for ProjectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "interaction excluded from scalar evaluation: {self:?}")
    }
}
impl std::error::Error for ProjectionError {}

/// A target-policy lookup could not resolve the original decision evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetEvidenceUnavailable;

impl fmt::Display for TargetEvidenceUnavailable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("original context or target-policy evidence is unavailable")
    }
}
impl std::error::Error for TargetEvidenceUnavailable {}

/// Project one actually retained, finalized single execution.
///
/// The target lookup receives the retained receipt, not a user-reconstructed
/// action/probability pair. It must use the original context and eligible set.
/// A valid probability for one observed action does **not** prove support over
/// all actions; that remains an application-level evaluation prerequisite.
/// Use `reward` for both channels when a scalar reward also confirms execution.
pub fn project_logged_reward<P, F>(
    muxer: &Muxer<P>,
    id: DecisionId,
    execution: &Channel,
    reward: &Channel,
    target: F,
) -> Result<LoggedReward, ProjectionError>
where
    P: InteractionPolicy,
    P::CanonicalFeedback: EvaluationFeedback,
    F: FnOnce(&DecisionReceipt) -> Result<Probability, TargetEvidenceUnavailable>,
{
    let receipt = muxer.receipt(id).ok_or(ProjectionError::NotRetained)?;
    if receipt.is_batch() {
        return Err(ProjectionError::BatchUnsupported);
    }
    let logging_propensity = match receipt.probability() {
        ProbabilityAvailability::Exact(value) if value.get() > 0.0 => value.get(),
        _ => return Err(ProjectionError::ProbabilityUnavailable),
    };
    if !muxer
        .final_feedback(id, 0, execution)
        .is_some_and(EvaluationFeedback::confirms_execution)
    {
        return Err(ProjectionError::ExecutionUnconfirmed);
    }
    let reward = muxer
        .final_feedback(id, 0, reward)
        .and_then(EvaluationFeedback::scalar_reward)
        .ok_or(ProjectionError::FinalRewardUnavailable)?;
    if !reward.is_finite() {
        return Err(ProjectionError::InvalidReward);
    }
    let target_propensity = target(receipt)
        .map_err(|_| ProjectionError::TargetEvidenceUnavailable)?
        .get();
    Ok(LoggedReward {
        reward,
        logging_propensity,
        target_propensity,
    })
}

/// Rows and explicit reason counts for a caller-selected evaluation cohort.
///
/// Call [`Self::record`] exactly once per decision in that cohort. This is an
/// accumulator, not an identity-deduplicating log or an unbiasedness guarantee.
#[derive(Debug, Clone, Default)]
pub struct EvaluationCohort {
    rows: Vec<LoggedReward>,
    excluded: BTreeMap<ProjectionError, usize>,
}

impl EvaluationCohort {
    /// Record either an eligible row or its exclusion reason.
    pub fn record(&mut self, result: Result<LoggedReward, ProjectionError>) {
        match result {
            Ok(row) => self.rows.push(row),
            Err(reason) => *self.excluded.entry(reason).or_default() += 1,
        }
    }
    /// Eligible scalar rows.
    #[must_use]
    pub fn rows(&self) -> &[LoggedReward] {
        &self.rows
    }
    /// Excluded decisions, grouped by the first failed eligibility check.
    #[must_use]
    pub fn exclusions(&self) -> &BTreeMap<ProjectionError, usize> {
        &self.excluded
    }
    /// Number of included plus excluded decisions recorded by the caller.
    #[must_use]
    pub fn total(&self) -> usize {
        self.rows.len() + self.excluded.values().sum::<usize>()
    }
}
