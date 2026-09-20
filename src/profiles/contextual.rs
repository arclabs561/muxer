//! Contextual LinUCB profiles with decision-time feature tickets.

use super::{sample_categorical, BoundedReward};
use crate::{
    Channel, DecisionReason, FeedbackExpectation, InteractionPolicy, LinUcb, LinUcbConfig,
    PolicyDecision, PolicyError, PolicyRequest, Probability, ProbabilityAvailability, TrialRng,
};

#[cfg(feature = "serde")]
use crate::contextual::LinUcbCheckpoint;

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

#[cfg(feature = "serde")]
#[derive(Clone, Copy)]
pub(crate) struct ContextualTicketState<'a> {
    pub(crate) ticket: &'a ContextualTicket,
    pub(crate) action: &'a str,
    pub(crate) reason: DecisionReason,
    pub(crate) values: &'a [(Channel, BoundedReward)],
    pub(crate) missing: &'a [Channel],
}

#[cfg(feature = "serde")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ContextualProfileCheckpoint {
    inner: LinUcbCheckpoint,
    mode: ContextualModeCheckpoint,
    representation_revision: u64,
    dimensions: usize,
}

#[cfg(feature = "serde")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum ContextualModeCheckpoint {
    Deterministic,
    Softmax { temperature: u64 },
}

#[cfg(feature = "serde")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ContextualTicketCheckpoint {
    action: String,
    features: Vec<u64>,
    representation_revision: u64,
    reason: DecisionReason,
}

#[cfg(feature = "serde")]
impl From<&ContextualTicket> for ContextualTicketCheckpoint {
    fn from(ticket: &ContextualTicket) -> Self {
        Self {
            action: ticket.action.clone(),
            features: ticket
                .features
                .iter()
                .map(|value| value.to_bits())
                .collect(),
            representation_revision: ticket.representation_revision,
            reason: ticket.reason,
        }
    }
}

#[cfg(feature = "serde")]
impl ContextualTicketCheckpoint {
    pub(crate) fn into_ticket(self) -> Result<ContextualTicket, PolicyError> {
        if self.action.is_empty() {
            return Err(PolicyError::new(
                "contextual checkpoint ticket action must be non-empty",
            ));
        }
        let features: Vec<f64> = self.features.into_iter().map(f64::from_bits).collect();
        if features.iter().any(|value| !value.is_finite()) {
            return Err(PolicyError::new(
                "contextual checkpoint ticket features must be finite",
            ));
        }
        if !matches!(
            self.reason,
            DecisionReason::Deterministic | DecisionReason::CategoricalSample
        ) {
            return Err(PolicyError::new(
                "contextual checkpoint ticket reason is invalid",
            ));
        }
        Ok(ContextualTicket {
            action: self.action,
            features,
            representation_revision: self.representation_revision,
            reason: self.reason,
        })
    }
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

    pub(crate) fn checkpoint_expectation(&self) -> FeedbackExpectation {
        FeedbackExpectation::FinalValue {
            channel: Channel::reward(),
        }
    }

    #[cfg(feature = "serde")]
    pub(crate) fn checkpoint_state(&self) -> Result<ContextualProfileCheckpoint, PolicyError> {
        let mode = match self.mode {
            ContextualMode::Deterministic => ContextualModeCheckpoint::Deterministic,
            ContextualMode::Softmax { temperature } => {
                validate_temperature(temperature)?;
                ContextualModeCheckpoint::Softmax {
                    temperature: temperature.to_bits(),
                }
            }
        };
        Ok(ContextualProfileCheckpoint {
            inner: self.inner.checkpoint_state()?,
            mode,
            representation_revision: self.representation_revision,
            dimensions: self.dimensions,
        })
    }

    #[cfg(feature = "serde")]
    pub(crate) fn from_checkpoint_state(
        state: ContextualProfileCheckpoint,
    ) -> Result<Self, PolicyError> {
        let inner = LinUcb::from_checkpoint_state(state.inner)?;
        if state.dimensions != inner.checkpoint_dimension() {
            return Err(PolicyError::new(
                "contextual checkpoint dimension does not match LinUcb state",
            ));
        }
        let mode = match state.mode {
            ContextualModeCheckpoint::Deterministic => ContextualMode::Deterministic,
            ContextualModeCheckpoint::Softmax { temperature } => {
                let temperature = f64::from_bits(temperature);
                validate_temperature(temperature)?;
                ContextualMode::Softmax { temperature }
            }
        };
        Ok(Self {
            inner,
            mode,
            representation_revision: state.representation_revision,
            dimensions: state.dimensions,
        })
    }

    #[cfg(feature = "serde")]
    pub(crate) fn validate_checkpoint_tickets(
        &self,
        tickets: &[ContextualTicketState<'_>],
    ) -> Result<(), PolicyError> {
        for state in tickets {
            let ticket = state.ticket;
            if ticket.action != state.action || ticket.reason != state.reason {
                return Err(PolicyError::new(
                    "contextual checkpoint ticket disagrees with its receipt",
                ));
            }
            if ticket.representation_revision != self.representation_revision {
                return Err(PolicyError::new(
                    "contextual checkpoint ticket has a stale representation revision",
                ));
            }
            if ticket.features.len() != self.dimensions
                || ticket.features.iter().any(|value| !value.is_finite())
            {
                return Err(PolicyError::new(
                    "contextual checkpoint ticket features do not match the representation",
                ));
            }
            if !self.inner.contains_state_arm(&ticket.action) {
                return Err(PolicyError::new(
                    "contextual checkpoint ticket action is not retained by LinUcb",
                ));
            }
            let expected_reason = match self.mode {
                ContextualMode::Deterministic => DecisionReason::Deterministic,
                ContextualMode::Softmax { .. } => DecisionReason::CategoricalSample,
            };
            if ticket.reason != expected_reason
                || !state.values.is_empty()
                || !state.missing.is_empty()
            {
                return Err(PolicyError::new(
                    "contextual checkpoint open ticket violates its feedback contract",
                ));
            }
        }
        Ok(())
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
            expectation: self.checkpoint_expectation(),
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

#[cfg(all(test, feature = "serde"))]
mod checkpoint_tests {
    use super::*;

    fn profile() -> ContextualProfile {
        let mut profile = ContextualProfile::new(LinUcbConfig {
            dim: 2,
            ..LinUcbConfig::default()
        });
        let arms = vec!["a".to_owned()];
        let _ = profile.inner.scores(&arms, &[0.0, 0.0]);
        profile
    }

    fn ticket() -> ContextualTicket {
        ContextualTicket {
            action: "a".to_owned(),
            features: vec![0.2, 0.8],
            representation_revision: 0,
            reason: DecisionReason::Deterministic,
        }
    }

    #[test]
    fn profile_checkpoint_rejects_mode_and_dimension_mismatches() {
        let profile = profile();
        let checkpoint = profile.checkpoint_state().unwrap();
        assert!(ContextualProfile::from_checkpoint_state(checkpoint.clone()).is_ok());

        let mut bad_dimension = checkpoint.clone();
        bad_dimension.dimensions = 3;
        assert!(ContextualProfile::from_checkpoint_state(bad_dimension).is_err());

        let mut bad_mode = checkpoint;
        bad_mode.mode = ContextualModeCheckpoint::Softmax {
            temperature: f64::NAN.to_bits(),
        };
        assert!(ContextualProfile::from_checkpoint_state(bad_mode).is_err());
    }

    #[test]
    fn ticket_validation_preserves_original_features_and_revision() {
        let profile = profile();
        let ticket = ticket();
        let state = ContextualTicketState {
            ticket: &ticket,
            action: "a",
            reason: DecisionReason::Deterministic,
            values: &[],
            missing: &[],
        };
        assert!(profile
            .validate_checkpoint_tickets(std::slice::from_ref(&state))
            .is_ok());

        let stale = ContextualTicket {
            representation_revision: 1,
            ..ticket.clone()
        };
        let stale = ContextualTicketState {
            ticket: &stale,
            ..state
        };
        assert!(profile
            .validate_checkpoint_tickets(std::slice::from_ref(&stale))
            .is_err());

        let wrong_features = ContextualTicket {
            features: vec![0.2],
            ..ticket.clone()
        };
        let wrong_features = ContextualTicketState {
            ticket: &wrong_features,
            ..state
        };
        assert!(profile
            .validate_checkpoint_tickets(std::slice::from_ref(&wrong_features))
            .is_err());
    }
}
