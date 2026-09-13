//! Contextual LinUCB profiles with decision-time feature tickets.

use super::{sample_categorical, BoundedReward};
use crate::{
    Channel, DecisionReason, FeedbackExpectation, InteractionPolicy, LinUcb, LinUcbConfig,
    PolicyDecision, PolicyError, PolicyRequest, Probability, ProbabilityAvailability, TrialRng,
};

/// Allocation mode for [`ContextualProfile`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ContextualMode {
    /// Choose the maximum UCB with point-mass selected propensity.
    Deterministic,
    /// Sample the softmax of UCB scores at this positive finite temperature.
    Softmax {
        /// Softmax temperature.
        temperature: f64,
    },
}

/// LinUCB lifecycle adapter that retains the original feature vector per decision.
#[derive(Debug, Clone)]
pub struct ContextualProfile {
    inner: LinUcb,
    mode: ContextualMode,
    representation_revision: u64,
    dimensions: usize,
}

#[derive(Debug, Clone)]
pub struct ContextualTicket {
    action: String,
    features: Vec<f64>,
    representation_revision: u64,
    reason: DecisionReason,
}

impl ContextualProfile {
    /// Construct a deterministic LinUCB profile.
    #[must_use]
    pub fn new(config: LinUcbConfig) -> Self {
        let dimensions = config.dim.max(1);
        Self {
            inner: LinUcb::new(config),
            mode: ContextualMode::Deterministic,
            representation_revision: 0,
            dimensions,
        }
    }
    /// Select with an explicit allocation mode.
    pub fn with_mode(mut self, mode: ContextualMode) -> Result<Self, PolicyError> {
        validate_mode(mode)?;
        self.mode = mode;
        Ok(self)
    }
    /// Return the configured allocation mode.
    #[must_use]
    pub const fn mode(&self) -> ContextualMode {
        self.mode
    }
    /// Return the current representation revision for diagnostics and replacement boundaries.
    #[must_use]
    pub const fn representation_revision(&self) -> u64 {
        self.representation_revision
    }
    /// Inspect the underlying LinUCB kernel.
    #[must_use]
    pub fn inner(&self) -> &LinUcb {
        &self.inner
    }
}

impl InteractionPolicy for ContextualProfile {
    type Context = [f64];
    type Feedback = BoundedReward;
    type CanonicalFeedback = BoundedReward;
    type Ticket = ContextualTicket;
    type PreparedIssue = LinUcb;
    type PreparedUpdate = LinUcb;

    fn prepare_decision(
        &self,
        request: PolicyRequest<'_, [f64]>,
        rng: &mut TrialRng,
    ) -> Result<PolicyDecision<Self::Ticket, LinUcb>, PolicyError> {
        let features = validate_features(request.context(), self.dimensions)?;
        let mut probe = self.inner.clone();
        let (selection, probability, reason) = match self.mode {
            ContextualMode::Deterministic => {
                let decision = probe
                    .decide(request.eligible(), &features)
                    .ok_or_else(|| PolicyError::new("no eligible action"))?;
                (
                    decision.chosen,
                    ProbabilityAvailability::Exact(Probability::new(1.0).expect("one is valid")),
                    DecisionReason::Deterministic,
                )
            }
            ContextualMode::Softmax { temperature } => {
                validate_temperature(temperature)?;
                let probabilities = probe.probabilities(request.eligible(), &features, temperature);
                let (action, probability) =
                    sample_categorical(request.eligible(), &probabilities, rng)?;
                (
                    action,
                    ProbabilityAvailability::Exact(probability),
                    DecisionReason::CategoricalSample,
                )
            }
        };
        Ok(PolicyDecision {
            selection: selection.clone(),
            probability,
            ticket: Some(ContextualTicket {
                action: selection,
                features,
                representation_revision: self.representation_revision,
                reason,
            }),
            issue: probe,
            expectation: FeedbackExpectation::FinalValue {
                channel: Channel::reward(),
            },
        })
    }
    fn commit_issue(&mut self, issue: LinUcb) {
        self.inner = issue;
    }
    fn normalize(&self, feedback: BoundedReward) -> Result<BoundedReward, PolicyError> {
        Ok(feedback)
    }
    fn decision_reason(&self, ticket: Option<&ContextualTicket>) -> DecisionReason {
        ticket.map_or(DecisionReason::Unspecified, |ticket| ticket.reason)
    }
    fn prepare(
        &self,
        ticket: &ContextualTicket,
        feedback: &BoundedReward,
    ) -> Result<LinUcb, PolicyError> {
        if ticket.representation_revision != self.representation_revision {
            return Err(PolicyError::new(
                "context representation changed while feedback was pending",
            ));
        }
        let mut next = self.inner.clone();
        next.update_reward(&ticket.action, &ticket.features, feedback.get());
        Ok(next)
    }
    fn apply(&mut self, update: LinUcb) {
        self.inner = update;
    }
}

fn validate_mode(mode: ContextualMode) -> Result<(), PolicyError> {
    if let ContextualMode::Softmax { temperature } = mode {
        validate_temperature(temperature)?;
    }
    Ok(())
}

fn validate_temperature(temperature: f64) -> Result<(), PolicyError> {
    if temperature.is_finite() && temperature > 0.0 {
        Ok(())
    } else {
        Err(PolicyError::new(
            "softmax temperature must be finite and positive",
        ))
    }
}

fn validate_features(features: &[f64], dimensions: usize) -> Result<Vec<f64>, PolicyError> {
    if features.len() != dimensions {
        return Err(PolicyError::new(
            "context feature dimension differs from the configured representation",
        ));
    }
    if features.iter().all(|value| value.is_finite()) {
        Ok(features.to_vec())
    } else {
        Err(PolicyError::new("context features must be finite"))
    }
}
