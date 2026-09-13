//! The built-in quality-routing profile backed by [`crate::Router`].
//!
//! The profile keeps the Router's control, triage, novelty, coverage, and
//! guardrail selection order.  Its policy implementation is deliberately an
//! adapter: `Router` remains usable directly for historical observations while
//! issued decisions use the runtime's correlated receipt lifecycle.

use crate::{
    BatchInteractionPolicy, BatchSelection, Channel, DecisionReason, FeedbackExpectation,
    InteractionPolicy, Muxer, ObservationId, Outcome, PolicyBatchDecision, PolicyDecision,
    PolicyError, PolicyRequest, ProbabilityAvailability, Router, RouterConfig, RouterDecision,
    RuntimeError, TrialRng,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

/// A finite quality score in the closed unit interval.
///
/// Scores are canonicalized at ingress so duplicate feedback does not depend
/// on distinct floating-point spellings of zero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QualityScore(f64);

impl QualityScore {
    /// Validate a finite score in `[0, 1]`.
    pub fn new(value: f64) -> Result<Self, PolicyError> {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(PolicyError::new(
                "quality score must be finite and in [0, 1]",
            ));
        }
        Ok(Self(if value == 0.0 { 0.0 } else { value }))
    }

    /// Return the validated score.
    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }
}

/// Feedback understood by [`QualityProfile`].
#[derive(Debug, Clone, Copy)]
pub enum QualityFeedback {
    /// The categorical execution result and its cost/latency measurements.
    Execution(Outcome),
    /// A separately produced continuous assessment.
    Score(QualityScore),
}

impl QualityFeedback {
    /// Wrap an execution result.
    #[must_use]
    pub const fn execution(outcome: Outcome) -> Self {
        Self::Execution(outcome)
    }

    /// Validate and wrap a delayed score.
    pub fn score(value: f64) -> Result<Self, PolicyError> {
        Ok(Self::Score(QualityScore::new(value)?))
    }
}

/// Canonical, equality-comparable quality feedback retained for deduplication.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalQualityFeedback(CanonicalQualityFeedbackKind);

#[derive(Debug, Clone, PartialEq, Eq)]
enum CanonicalQualityFeedbackKind {
    Execution {
        ok: bool,
        junk: bool,
        hard_junk: bool,
        cost_units: u64,
        elapsed_ms: u64,
        quality_score: Option<u64>,
    },
    Score(u64),
}

impl CanonicalQualityFeedback {
    /// Whether this feedback carries an execution observation.
    #[must_use]
    pub const fn is_execution(&self) -> bool {
        matches!(&self.0, CanonicalQualityFeedbackKind::Execution { .. })
    }

    /// Return the validated scalar score, when this is score feedback.
    #[must_use]
    pub fn score(&self) -> Option<f64> {
        let CanonicalQualityFeedbackKind::Score(bits) = &self.0 else {
            return None;
        };
        Some(f64::from_bits(*bits))
    }

    /// Return a scalar quality reward from either execution-embedded or
    /// separately reported score feedback.
    #[must_use]
    pub fn scalar_reward(&self) -> Option<f64> {
        match &self.0 {
            CanonicalQualityFeedbackKind::Execution { quality_score, .. } => {
                quality_score.map(f64::from_bits)
            }
            CanonicalQualityFeedbackKind::Score(bits) => Some(f64::from_bits(*bits)),
        }
    }

    fn execution_outcome(&self) -> Option<Outcome> {
        let CanonicalQualityFeedbackKind::Execution {
            ok,
            junk,
            hard_junk,
            cost_units,
            elapsed_ms,
            quality_score,
        } = &self.0
        else {
            return None;
        };
        match quality_score {
            Some(bits) => Some(Outcome::with_quality(
                *ok,
                *junk,
                *hard_junk,
                *cost_units,
                *elapsed_ms,
                f64::from_bits(*bits),
            )),
            None => Some(Outcome::new(
                *ok,
                *junk,
                *hard_junk,
                *cost_units,
                *elapsed_ms,
            )),
        }
    }
}

#[cfg(feature = "serde")]
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum CanonicalQualityFeedbackWire {
    Execution {
        ok: bool,
        junk: bool,
        hard_junk: bool,
        cost_units: u64,
        elapsed_ms: u64,
        quality_score: Option<u64>,
    },
    Score {
        score: u64,
    },
}

#[cfg(feature = "serde")]
impl serde::Serialize for CanonicalQualityFeedback {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let wire = match &self.0 {
            CanonicalQualityFeedbackKind::Execution {
                ok,
                junk,
                hard_junk,
                cost_units,
                elapsed_ms,
                quality_score,
            } => CanonicalQualityFeedbackWire::Execution {
                ok: *ok,
                junk: *junk,
                hard_junk: *hard_junk,
                cost_units: *cost_units,
                elapsed_ms: *elapsed_ms,
                quality_score: *quality_score,
            },
            CanonicalQualityFeedbackKind::Score(score) => {
                CanonicalQualityFeedbackWire::Score { score: *score }
            }
        };
        wire.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for CanonicalQualityFeedback {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = CanonicalQualityFeedbackWire::deserialize(deserializer)?;
        let canonical_score = |bits: u64| -> Result<u64, D::Error> {
            let score =
                QualityScore::new(f64::from_bits(bits)).map_err(serde::de::Error::custom)?;
            if score.get().to_bits() != bits {
                return Err(serde::de::Error::custom(
                    "quality checkpoint feedback score is not canonical",
                ));
            }
            Ok(bits)
        };
        match wire {
            CanonicalQualityFeedbackWire::Execution {
                ok,
                junk,
                hard_junk,
                cost_units,
                elapsed_ms,
                quality_score,
            } => {
                if hard_junk && !junk {
                    return Err(serde::de::Error::custom(
                        "quality checkpoint execution is not canonical",
                    ));
                }
                let quality_score = quality_score.map(canonical_score).transpose()?;
                Ok(Self(CanonicalQualityFeedbackKind::Execution {
                    ok,
                    junk,
                    hard_junk,
                    cost_units,
                    elapsed_ms,
                    quality_score,
                }))
            }
            CanonicalQualityFeedbackWire::Score { score } => Ok(Self(
                CanonicalQualityFeedbackKind::Score(canonical_score(score)?),
            )),
        }
    }
}

/// The immutable evidence retained for one issued quality decision.
#[derive(Debug, Clone)]
pub struct QualityTicket {
    action: Arc<str>,
    observation: ObservationId,
    context: Arc<[f64]>,
    reason: DecisionReason,
}

/// Borrowed open-ticket evidence used to validate a quality runtime checkpoint.
#[cfg(feature = "serde")]
pub(crate) struct QualityTicketState<'a> {
    pub(crate) ticket: &'a QualityTicket,
    pub(crate) action: &'a str,
    pub(crate) reason: DecisionReason,
    pub(crate) values: &'a [(Channel, CanonicalQualityFeedback)],
    pub(crate) missing: &'a [Channel],
}

#[cfg(feature = "serde")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct QualityProfileCheckpoint {
    router: crate::RouterCheckpoint,
    next_observation: u64,
    delayed_score: bool,
    buffered_scores: Vec<(ObservationId, u64)>,
    executed: Vec<ObservationId>,
}

#[cfg(feature = "serde")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct QualityTicketCheckpoint {
    action: String,
    observation: ObservationId,
    context: Vec<f64>,
    reason: DecisionReason,
}

#[cfg(feature = "serde")]
impl From<&QualityTicket> for QualityTicketCheckpoint {
    fn from(ticket: &QualityTicket) -> Self {
        Self {
            action: ticket.action.to_string(),
            observation: ticket.observation,
            context: ticket.context.to_vec(),
            reason: ticket.reason,
        }
    }
}

#[cfg(feature = "serde")]
impl QualityTicketCheckpoint {
    pub(crate) fn into_ticket(self) -> Result<QualityTicket, PolicyError> {
        if self.action.is_empty() {
            return Err(PolicyError::new(
                "quality checkpoint ticket action must be non-empty",
            ));
        }
        if self.context.iter().any(|value| !value.is_finite()) {
            return Err(PolicyError::new(
                "quality checkpoint ticket context must be finite",
            ));
        }
        if !matches!(
            self.reason,
            DecisionReason::Control
                | DecisionReason::Triage
                | DecisionReason::NoveltyOrCoverage
                | DecisionReason::Policy
        ) {
            return Err(PolicyError::new(
                "quality checkpoint ticket reason is not a router reason",
            ));
        }
        Ok(QualityTicket {
            action: Arc::from(self.action),
            observation: self.observation,
            context: self.context.into(),
            reason: self.reason,
        })
    }
}

/// Opaque prepared quality-router update.
#[derive(Debug)]
pub struct QualityUpdate(QualityUpdateInner);

#[derive(Debug)]
enum QualityUpdateInner {
    Execution {
        ticket: QualityTicket,
        outcome: Outcome,
        mark_awaiting_score: bool,
    },
    Score {
        observation: ObservationId,
        score: f64,
    },
    BufferScore {
        observation: ObservationId,
        score: QualityScore,
    },
    OutsideHorizon {
        observation: ObservationId,
    },
}

/// A concrete quality-routing policy for [`crate::Muxer`].
///
/// `QualityProfile` preserves the existing [`Router`] kernels.  It selects
/// only from the runtime's authoritative eligible set and applies feedback to
/// a Router-prepared update only after validation, so a rejected update leaves
/// control, monitoring, and triage state untouched.
#[derive(Debug, Clone)]
pub struct QualityProfile {
    router: Router,
    next_observation: u64,
    delayed_score: bool,
    buffered_scores: BTreeMap<ObservationId, QualityScore>,
    executed: BTreeSet<ObservationId>,
}

/// Builder for the ready-to-run quality profile.
#[derive(Debug, Clone)]
pub struct QualityProfileBuilder {
    actions: Vec<String>,
    config: RouterConfig,
    delayed_score: bool,
}

/// Failure while constructing a quality muxer.
#[derive(Debug)]
pub enum QualityBuildError {
    /// Router configuration or action registration was invalid.
    Router(logp::Error),
    /// Runtime catalogue or retention construction failed.
    Runtime(RuntimeError),
}

impl fmt::Display for QualityBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Router(error) => {
                write!(formatter, "quality router configuration failed: {error}")
            }
            Self::Runtime(error) => {
                write!(formatter, "quality runtime construction failed: {error}")
            }
        }
    }
}

impl std::error::Error for QualityBuildError {}

impl QualityProfileBuilder {
    /// Replace the Router configuration used by the built profile.
    #[must_use]
    pub fn config(mut self, config: RouterConfig) -> Self {
        self.config = config;
        self
    }

    /// Require a separate finalized continuous score for every execution.
    #[must_use]
    pub fn with_delayed_score(mut self) -> Self {
        self.delayed_score = true;
        self
    }

    /// Build the correlated quality router.
    pub fn build(self) -> Result<Muxer<QualityProfile>, QualityBuildError> {
        let mut profile = QualityProfile::new(self.actions.clone(), self.config)
            .map_err(QualityBuildError::Router)?;
        if self.delayed_score {
            profile = profile.with_delayed_score();
        }
        Muxer::new(self.actions, profile).map_err(QualityBuildError::Runtime)
    }
}

impl Muxer<QualityProfile> {
    /// Start building the built-in quality-routing profile.
    #[must_use]
    pub fn quality<I, S>(actions: I) -> QualityProfileBuilder
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        QualityProfileBuilder {
            actions: actions.into_iter().map(Into::into).collect(),
            config: RouterConfig::default(),
            delayed_score: false,
        }
    }
}

impl QualityProfile {
    /// Construct a profile from a configured Router.
    #[must_use]
    pub fn from_router(router: Router) -> Self {
        Self {
            next_observation: router.total_observations(),
            router,
            delayed_score: false,
            buffered_scores: BTreeMap::new(),
            executed: BTreeSet::new(),
        }
    }

    /// Construct a Router-backed quality profile.
    pub fn new(arms: Vec<String>, config: RouterConfig) -> Result<Self, logp::Error> {
        Ok(Self::from_router(Router::new(arms, config)?))
    }

    /// Require a separately finalized score after execution.
    ///
    /// The runtime will retain the decision until both execution and score are
    /// finalized once multi-channel expectations are enabled.
    #[must_use]
    pub fn with_delayed_score(mut self) -> Self {
        self.delayed_score = true;
        self
    }

    /// Whether issued decisions declare a delayed score channel.
    #[must_use]
    pub const fn expects_delayed_score(&self) -> bool {
        self.delayed_score
    }

    /// Inspect the live Router state.
    #[must_use]
    pub fn router(&self) -> &Router {
        &self.router
    }

    /// Number of scores retained while their execution feedback is pending.
    #[must_use]
    pub fn buffered_score_len(&self) -> usize {
        self.buffered_scores.len()
    }

    /// Number of execution rows retained while their declared delayed score is pending.
    #[must_use]
    pub fn awaiting_score_len(&self) -> usize {
        self.executed.len()
    }

    /// Add historical data without manufacturing an issued receipt.
    ///
    /// This has the same semantics as [`Router::observe`] and is intended for
    /// warm starts before issuing decisions.
    pub fn seed(&mut self, action: &str, outcome: Outcome) -> bool {
        self.router.observe(action, outcome)
    }

    /// Channel for the categorical execution observation.
    #[must_use]
    pub fn execution_channel() -> Channel {
        Channel::new("execution").expect("fixed channel is valid")
    }

    /// Channel for the separately finalized continuous quality assessment.
    #[must_use]
    pub fn score_channel() -> Channel {
        Channel::new("quality-score").expect("fixed channel is valid")
    }

    pub(crate) fn checkpoint_expectation(&self) -> FeedbackExpectation {
        if self.delayed_score {
            FeedbackExpectation::FinalValues(vec![Self::execution_channel(), Self::score_channel()])
        } else {
            FeedbackExpectation::FinalValue {
                channel: Self::execution_channel(),
            }
        }
    }

    fn expectation(&self) -> FeedbackExpectation {
        self.checkpoint_expectation()
    }

    /// Capture profile state for the concrete quality runtime checkpoint.
    #[cfg(feature = "serde")]
    pub(crate) fn checkpoint_state(
        &self,
        build_key: &str,
    ) -> Result<QualityProfileCheckpoint, PolicyError> {
        let buffered_scores = self
            .buffered_scores
            .iter()
            .map(|(observation, score)| (*observation, score.get().to_bits()))
            .collect();
        Ok(QualityProfileCheckpoint {
            router: self
                .router
                .checkpoint(build_key)
                .map_err(|error| PolicyError::new(error.to_string()))?,
            next_observation: self.next_observation,
            delayed_score: self.delayed_score,
            buffered_scores,
            executed: self.executed.iter().copied().collect(),
        })
    }

    /// Restore profile state from the concrete quality runtime checkpoint.
    #[cfg(feature = "serde")]
    pub(crate) fn from_checkpoint_state(
        state: QualityProfileCheckpoint,
        build_key: &str,
    ) -> Result<Self, PolicyError> {
        let router = Router::from_checkpoint(state.router, build_key)
            .map_err(|error| PolicyError::new(error.to_string()))?;
        let mut buffered_scores = BTreeMap::new();
        for (observation, bits) in state.buffered_scores {
            if observation.get() > state.next_observation {
                return Err(PolicyError::new(
                    "quality checkpoint buffered observation exceeds profile sequence",
                ));
            }
            let score = QualityScore::new(f64::from_bits(bits))?;
            if score.get().to_bits() != bits {
                return Err(PolicyError::new(
                    "quality checkpoint buffered score is not canonical",
                ));
            }
            if buffered_scores.insert(observation, score).is_some() {
                return Err(PolicyError::new(
                    "quality checkpoint repeats a buffered observation",
                ));
            }
        }
        let mut executed = BTreeSet::new();
        for observation in state.executed {
            if observation.get() > state.next_observation {
                return Err(PolicyError::new(
                    "quality checkpoint awaiting observation exceeds profile sequence",
                ));
            }
            if !executed.insert(observation) {
                return Err(PolicyError::new(
                    "quality checkpoint repeats an awaiting observation",
                ));
            }
        }
        if !state.delayed_score && (!buffered_scores.is_empty() || !executed.is_empty()) {
            return Err(PolicyError::new(
                "quality checkpoint has delayed state without delayed score",
            ));
        }
        if buffered_scores.keys().any(|id| executed.contains(id)) {
            return Err(PolicyError::new(
                "quality checkpoint observation cannot be both buffered and awaiting",
            ));
        }
        Ok(Self {
            router,
            next_observation: state.next_observation,
            delayed_score: state.delayed_score,
            buffered_scores,
            executed,
        })
    }

    /// Validate open runtime tickets against profile-owned delayed-score state.
    #[cfg(feature = "serde")]
    pub(crate) fn validate_checkpoint_tickets(
        &self,
        tickets: &[QualityTicketState<'_>],
    ) -> Result<(), PolicyError> {
        let execution = Self::execution_channel();
        let score = Self::score_channel();
        let mut observations = BTreeSet::new();
        let mut expected_buffered = BTreeMap::new();
        let mut expected_executed = BTreeSet::new();

        for state in tickets {
            let ticket = state.ticket;
            if ticket.action.as_ref() != state.action || ticket.reason != state.reason {
                return Err(PolicyError::new(
                    "quality checkpoint ticket disagrees with its receipt",
                ));
            }
            if !self.router.arms().iter().any(|arm| arm == state.action) {
                return Err(PolicyError::new(
                    "quality checkpoint ticket action is not registered",
                ));
            }
            if ticket.observation.get() > self.next_observation {
                return Err(PolicyError::new(
                    "quality checkpoint ticket observation exceeds profile sequence",
                ));
            }
            if !observations.insert(ticket.observation) {
                return Err(PolicyError::new(
                    "quality checkpoint repeats an open observation",
                ));
            }

            let mut final_execution = false;
            let mut final_score = false;
            let mut final_score_bits = None;
            let mut missing_execution = false;
            let mut missing_score = false;
            let mut channels = BTreeSet::new();
            for (channel, feedback) in state.values {
                if !channels.insert(channel.clone()) {
                    return Err(PolicyError::new(
                        "quality checkpoint repeats a final feedback channel",
                    ));
                }
                if channel == &execution && feedback.is_execution() {
                    final_execution = true;
                } else if channel == &score && feedback.score().is_some() && self.delayed_score {
                    final_score = true;
                    final_score_bits = Some(
                        feedback
                            .score()
                            .expect("score feedback was checked")
                            .to_bits(),
                    );
                } else {
                    return Err(PolicyError::new(
                        "quality checkpoint final feedback violates its channel contract",
                    ));
                }
            }
            for channel in state.missing {
                if !channels.insert(channel.clone()) {
                    return Err(PolicyError::new(
                        "quality checkpoint channel is both final and missing",
                    ));
                }
                if channel == &execution {
                    missing_execution = true;
                } else if channel == &score && self.delayed_score {
                    missing_score = true;
                } else {
                    return Err(PolicyError::new(
                        "quality checkpoint missing channel violates its contract",
                    ));
                }
            }

            if !final_execution
                && !missing_execution
                && self.router.contains_observation_id(ticket.observation)
            {
                return Err(PolicyError::new(
                    "quality checkpoint open ticket collides with a retained observation",
                ));
            }
            if !self.delayed_score {
                if final_execution || missing_execution {
                    return Err(PolicyError::new(
                        "resolved immediate-quality ticket must not remain open",
                    ));
                }
                continue;
            }
            if final_score && !final_execution && !missing_execution {
                expected_buffered.insert(
                    ticket.observation,
                    final_score_bits.expect("final score carries score bits"),
                );
            }
            if final_execution && !final_score && !missing_score {
                expected_executed.insert(ticket.observation);
            }
        }

        let buffered: BTreeMap<_, _> = self
            .buffered_scores
            .iter()
            .map(|(observation, score)| (*observation, score.get().to_bits()))
            .collect();
        if buffered != expected_buffered || self.executed != expected_executed {
            return Err(PolicyError::new(
                "quality checkpoint delayed state does not match open ticket progress",
            ));
        }
        Ok(())
    }

    fn snapshot_context(context: &[f64]) -> Arc<[f64]> {
        context
            .iter()
            .map(|value| if value.is_finite() { *value } else { 0.0 })
            .collect::<Vec<_>>()
            .into()
    }

    fn reason_for(decision: &RouterDecision, action: &str) -> DecisionReason {
        if decision.control_picks.iter().any(|picked| picked == action) {
            DecisionReason::Control
        } else if decision.prechosen.iter().any(|picked| picked == action) {
            DecisionReason::NoveltyOrCoverage
        } else if !decision
            .mab_eligible
            .iter()
            .any(|eligible| eligible == action)
        {
            DecisionReason::Triage
        } else {
            DecisionReason::Policy
        }
    }
}

/// Issuance state that becomes visible only after the runtime validates a choice.
#[derive(Debug, Clone)]
pub struct QualityIssue {
    next_observation: u64,
}

impl InteractionPolicy for QualityProfile {
    type Context = [f64];
    type Feedback = QualityFeedback;
    type CanonicalFeedback = CanonicalQualityFeedback;
    type Ticket = QualityTicket;
    type PreparedIssue = QualityIssue;
    type PreparedUpdate = QualityUpdate;

    fn prepare_decision(
        &self,
        request: PolicyRequest<'_, Self::Context>,
        rng: &mut TrialRng,
    ) -> Result<PolicyDecision<Self::Ticket, Self::PreparedIssue>, PolicyError> {
        let decision = self
            .router
            .select_from(request.eligible(), 1, rng.next_u64())
            .map_err(|error| PolicyError::new(error.to_string()))?;
        let selection = decision
            .primary()
            .ok_or_else(|| PolicyError::new("Router selected no eligible action"))?
            .to_owned();
        let reason = Self::reason_for(&decision, &selection);
        let context = Self::snapshot_context(request.context());
        let mut observation = self
            .next_observation
            .checked_add(1)
            .ok_or_else(|| PolicyError::new("quality observation identity exhausted"))?;
        while self
            .router
            .contains_observation_id(ObservationId::new(observation))
        {
            observation = observation
                .checked_add(1)
                .ok_or_else(|| PolicyError::new("quality observation identity exhausted"))?;
        }

        Ok(PolicyDecision {
            selection: selection.clone(),
            probability: ProbabilityAvailability::Unavailable,
            ticket: Some(QualityTicket {
                action: Arc::from(selection.as_str()),
                observation: ObservationId::new(observation),
                context,
                reason,
            }),
            issue: QualityIssue {
                next_observation: observation,
            },
            expectation: self.expectation(),
        })
    }

    fn commit_issue(&mut self, issue: Self::PreparedIssue) {
        self.next_observation = issue.next_observation;
    }

    fn normalize(&self, feedback: Self::Feedback) -> Result<Self::CanonicalFeedback, PolicyError> {
        Ok(match feedback {
            QualityFeedback::Execution(outcome) => {
                let quality_score = match outcome.quality_score {
                    Some(_score) if self.delayed_score => {
                        return Err(PolicyError::new(
                            "execution quality_score conflicts with the declared quality-score channel",
                        ));
                    }
                    Some(score) => Some(QualityScore::new(score)?.get().to_bits()),
                    None => None,
                };
                CanonicalQualityFeedback(CanonicalQualityFeedbackKind::Execution {
                    ok: outcome.ok,
                    junk: outcome.junk || outcome.hard_junk,
                    hard_junk: outcome.hard_junk,
                    cost_units: outcome.cost_units,
                    elapsed_ms: outcome.elapsed_ms,
                    quality_score,
                })
            }
            QualityFeedback::Score(score) => {
                CanonicalQualityFeedback(CanonicalQualityFeedbackKind::Score(score.get().to_bits()))
            }
        })
    }

    fn feedback_channel(&self, feedback: &Self::CanonicalFeedback) -> Channel {
        match feedback {
            CanonicalQualityFeedback(CanonicalQualityFeedbackKind::Execution { .. }) => {
                Self::execution_channel()
            }
            CanonicalQualityFeedback(CanonicalQualityFeedbackKind::Score(_)) => {
                Self::score_channel()
            }
        }
    }

    fn prepare(
        &self,
        ticket: &Self::Ticket,
        feedback: &Self::CanonicalFeedback,
    ) -> Result<Self::PreparedUpdate, PolicyError> {
        if let Some(outcome) = feedback.execution_outcome() {
            let buffered_score = self.buffered_scores.get(&ticket.observation).copied();
            let outcome = if let Some(score) = buffered_score {
                Outcome::with_quality(
                    outcome.ok,
                    outcome.junk,
                    outcome.hard_junk,
                    outcome.cost_units,
                    outcome.elapsed_ms,
                    score.get(),
                )
            } else {
                outcome
            };
            if !self
                .router
                .prepare_observation(Some(ticket.observation), ticket.action.as_ref())
            {
                return Err(PolicyError::new(
                    "quality execution could not be correlated",
                ));
            }
            return Ok(QualityUpdate(QualityUpdateInner::Execution {
                ticket: ticket.clone(),
                outcome,
                mark_awaiting_score: self.delayed_score && buffered_score.is_none(),
            }));
        }

        let Some(score) = feedback.score() else {
            return Err(PolicyError::new("unknown quality feedback"));
        };
        if !self.delayed_score {
            return Err(PolicyError::new(
                "quality score requires QualityProfile::with_delayed_score",
            ));
        }
        if self.executed.contains(&ticket.observation) {
            if self.router.prepare_quality_score(ticket.observation) {
                Ok(QualityUpdate(QualityUpdateInner::Score {
                    observation: ticket.observation,
                    score,
                }))
            } else {
                Ok(QualityUpdate(QualityUpdateInner::OutsideHorizon {
                    observation: ticket.observation,
                }))
            }
        } else {
            Ok(QualityUpdate(QualityUpdateInner::BufferScore {
                observation: ticket.observation,
                score: QualityScore::new(score).expect("canonical score remains valid"),
            }))
        }
    }

    fn apply(&mut self, update: Self::PreparedUpdate) {
        match update.0 {
            QualityUpdateInner::Execution {
                ticket,
                outcome,
                mark_awaiting_score,
            } => {
                self.router.apply_observation(
                    Some(ticket.observation),
                    ticket.action.as_ref(),
                    outcome,
                    ticket.context.as_ref(),
                );
                self.buffered_scores.remove(&ticket.observation);
                if mark_awaiting_score {
                    self.executed.insert(ticket.observation);
                }
            }
            QualityUpdateInner::Score { observation, score } => {
                self.router.apply_quality_score(observation, score);
                self.executed.remove(&observation);
            }
            QualityUpdateInner::BufferScore { observation, score } => {
                self.buffered_scores.insert(observation, score);
            }
            QualityUpdateInner::OutsideHorizon { observation } => {
                self.executed.remove(&observation);
            }
        }
    }

    fn expire_ticket(&mut self, ticket: &Self::Ticket) {
        self.buffered_scores.remove(&ticket.observation);
        self.executed.remove(&ticket.observation);
    }

    fn missing_channel(&mut self, ticket: &Self::Ticket, channel: &Channel) {
        if channel == &Self::execution_channel() {
            self.buffered_scores.remove(&ticket.observation);
        }
        if channel == &Self::score_channel() {
            self.executed.remove(&ticket.observation);
        }
    }

    fn finish_ticket(&mut self, ticket: &Self::Ticket) {
        self.buffered_scores.remove(&ticket.observation);
        self.executed.remove(&ticket.observation);
    }

    fn decision_reason(&self, ticket: Option<&Self::Ticket>) -> DecisionReason {
        ticket.map_or(DecisionReason::Unspecified, |ticket| ticket.reason)
    }

    fn update_disposition(&self, update: &Self::PreparedUpdate) -> crate::EventDisposition {
        match &update.0 {
            QualityUpdateInner::OutsideHorizon { .. } => crate::EventDisposition::OutsideHorizon,
            _ => crate::EventDisposition::Accepted,
        }
    }
}

impl BatchInteractionPolicy for QualityProfile {
    type PreparedBatchIssue = QualityIssue;

    fn prepare_batch(
        &self,
        request: PolicyRequest<'_, Self::Context>,
        count: usize,
        rng: &mut TrialRng,
    ) -> Result<PolicyBatchDecision<Self::Ticket, Self::PreparedBatchIssue>, PolicyError> {
        if count == 0 {
            return Err(PolicyError::new("quality batch size must be non-zero"));
        }
        let decision = self
            .router
            .select_from(request.eligible(), count, rng.next_u64())
            .map_err(|error| PolicyError::new(error.to_string()))?;
        if decision.chosen.len() != count {
            return Err(PolicyError::new(
                "Router could not select requested batch size",
            ));
        }

        let mut observation = self.next_observation;
        let context = Self::snapshot_context(request.context());
        let mut selections = Vec::with_capacity(count);
        for action in &decision.chosen {
            let reason = Self::reason_for(&decision, action);
            loop {
                observation = observation
                    .checked_add(1)
                    .ok_or_else(|| PolicyError::new("quality observation identity exhausted"))?;
                if !self
                    .router
                    .contains_observation_id(ObservationId::new(observation))
                {
                    break;
                }
            }
            selections.push(BatchSelection {
                selection: action.clone(),
                probability: ProbabilityAvailability::Unavailable,
                ticket: Some(QualityTicket {
                    action: Arc::from(action.as_str()),
                    observation: ObservationId::new(observation),
                    context: Arc::clone(&context),
                    reason,
                }),
                expectation: self.expectation(),
            });
        }

        Ok(PolicyBatchDecision {
            selections,
            issue: QualityIssue {
                next_observation: observation,
            },
        })
    }

    fn commit_batch(&mut self, issue: Self::PreparedBatchIssue) {
        self.commit_issue(issue);
    }
}

#[cfg(all(test, feature = "serde"))]
mod checkpoint_tests {
    use super::*;

    fn ticket(observation: u64) -> QualityTicket {
        QualityTicket {
            action: Arc::from("a"),
            observation: ObservationId::new(observation),
            context: Arc::from([0.25]),
            reason: DecisionReason::Policy,
        }
    }

    #[test]
    fn profile_checkpoint_round_trips_router_and_delayed_state() {
        let mut profile =
            QualityProfile::new(vec!["a".to_owned()], RouterConfig::default()).unwrap();
        assert!(profile.seed("a", Outcome::success(1, 1)));
        profile.next_observation = 7;
        profile.delayed_score = true;
        profile
            .buffered_scores
            .insert(ObservationId::new(6), QualityScore::new(0.5).unwrap());
        let state = profile.checkpoint_state("quality-build").unwrap();
        let restored = QualityProfile::from_checkpoint_state(state, "quality-build").unwrap();
        assert_eq!(restored.router.summary("a").calls, 1);
        assert_eq!(restored.next_observation, 7);
        assert_eq!(restored.buffered_score_len(), 1);
    }

    #[test]
    fn strict_canonical_feedback_and_ticket_decode_reject_repairs() {
        let canonical =
            CanonicalQualityFeedback(CanonicalQualityFeedbackKind::Score(0.5f64.to_bits()));
        let mut wire = serde_json::to_value(canonical).unwrap();
        wire["score"] = serde_json::json!(f64::NAN.to_bits());
        assert!(serde_json::from_value::<CanonicalQualityFeedback>(wire).is_err());

        let checkpoint = QualityTicketCheckpoint {
            action: "a".to_owned(),
            observation: ObservationId::new(1),
            context: vec![f64::NAN],
            reason: DecisionReason::Policy,
        };
        assert!(checkpoint.into_ticket().is_err());
    }

    #[test]
    fn ticket_progress_exactly_matches_delayed_maps() {
        let mut profile = QualityProfile::new(vec!["a".to_owned()], RouterConfig::default())
            .unwrap()
            .with_delayed_score();
        profile.next_observation = 1;
        let ticket = ticket(1);
        let values = vec![(
            QualityProfile::score_channel(),
            CanonicalQualityFeedback(CanonicalQualityFeedbackKind::Score(0.5f64.to_bits())),
        )];
        let state = QualityTicketState {
            ticket: &ticket,
            action: "a",
            reason: DecisionReason::Policy,
            values: &values,
            missing: &[],
        };
        profile
            .buffered_scores
            .insert(ObservationId::new(1), QualityScore::new(0.5).unwrap());
        assert!(profile
            .validate_checkpoint_tickets(std::slice::from_ref(&state))
            .is_ok());
        profile
            .buffered_scores
            .insert(ObservationId::new(1), QualityScore::new(0.4).unwrap());
        assert!(profile
            .validate_checkpoint_tickets(std::slice::from_ref(&state))
            .is_err());
        profile
            .buffered_scores
            .insert(ObservationId::new(1), QualityScore::new(0.5).unwrap());
        assert!(profile
            .router
            .observe_with_id(ObservationId::new(1), "a", Outcome::success(1, 1)));
        assert!(profile
            .validate_checkpoint_tickets(std::slice::from_ref(&state))
            .is_err());
        profile.buffered_scores.clear();
        assert!(profile
            .validate_checkpoint_tickets(std::slice::from_ref(&state))
            .is_err());
    }
}
