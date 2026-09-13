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

/// The immutable evidence retained for one issued quality decision.
#[derive(Debug, Clone)]
pub struct QualityTicket {
    action: String,
    observation: ObservationId,
    context: Vec<f64>,
    reason: DecisionReason,
}

/// A prepared quality-router update.
#[derive(Debug, Clone)]
pub enum QualityUpdate {
    /// A complete, transactionally prepared quality-state replacement.
    Commit {
        /// Router state after the accepted channel update.
        router: Box<Router>,
        /// Scores received before their execution rows.
        buffered_scores: BTreeMap<ObservationId, QualityScore>,
        /// Execution rows still awaiting their delayed scores.
        executed: BTreeSet<ObservationId>,
    },
    /// The execution row has fallen out of every quality window.
    OutsideHorizon {
        /// Scores received before their execution rows.
        buffered_scores: BTreeMap<ObservationId, QualityScore>,
        /// Execution rows still awaiting their delayed scores.
        executed: BTreeSet<ObservationId>,
    },
}

/// A concrete quality-routing policy for [`crate::Muxer`].
///
/// `QualityProfile` preserves the existing [`Router`] kernels.  It selects
/// only from the runtime's authoritative eligible set and applies feedback to
/// a cloned candidate Router before committing it, so a rejected update leaves
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

    fn expectation(&self) -> FeedbackExpectation {
        if self.delayed_score {
            FeedbackExpectation::FinalValues(vec![Self::execution_channel(), Self::score_channel()])
        } else {
            FeedbackExpectation::FinalValue {
                channel: Self::execution_channel(),
            }
        }
    }

    fn snapshot_context(context: &[f64]) -> Vec<f64> {
        context
            .iter()
            .map(|value| if value.is_finite() { *value } else { 0.0 })
            .collect()
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
                action: selection,
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
            let mut router = self.router.clone();
            let mut buffered_scores = self.buffered_scores.clone();
            let mut executed = self.executed.clone();
            let outcome = if let Some(score) = buffered_scores.remove(&ticket.observation) {
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
            if !router.observe_with_id_and_context(
                ticket.observation,
                &ticket.action,
                outcome,
                &ticket.context,
            ) {
                return Err(PolicyError::new(
                    "quality execution could not be correlated",
                ));
            }
            if self.delayed_score && !buffered_scores.contains_key(&ticket.observation) {
                // A score delivered before execution was joined into the row,
                // so it does not need a second awaiting marker.
                if !self.buffered_scores.contains_key(&ticket.observation) {
                    executed.insert(ticket.observation);
                }
            }
            return Ok(QualityUpdate::Commit {
                router: Box::new(router),
                buffered_scores,
                executed,
            });
        }

        let Some(score) = feedback.score() else {
            return Err(PolicyError::new("unknown quality feedback"));
        };
        if !self.delayed_score {
            return Err(PolicyError::new(
                "quality score requires QualityProfile::with_delayed_score",
            ));
        }
        let mut router = self.router.clone();
        let mut buffered_scores = self.buffered_scores.clone();
        let mut executed = self.executed.clone();
        if executed.contains(&ticket.observation) {
            executed.remove(&ticket.observation);
            if router.set_quality_score_for_id(ticket.observation, score) {
                Ok(QualityUpdate::Commit {
                    router: Box::new(router),
                    buffered_scores,
                    executed,
                })
            } else {
                Ok(QualityUpdate::OutsideHorizon {
                    buffered_scores,
                    executed,
                })
            }
        } else {
            buffered_scores.insert(
                ticket.observation,
                QualityScore::new(score).expect("canonical score remains valid"),
            );
            Ok(QualityUpdate::Commit {
                router: Box::new(router),
                buffered_scores,
                executed,
            })
        }
    }

    fn apply(&mut self, update: Self::PreparedUpdate) {
        match update {
            QualityUpdate::Commit {
                router,
                buffered_scores,
                executed,
            } => {
                self.router = *router;
                self.buffered_scores = buffered_scores;
                self.executed = executed;
            }
            QualityUpdate::OutsideHorizon {
                buffered_scores,
                executed,
            } => {
                self.buffered_scores = buffered_scores;
                self.executed = executed;
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
        match update {
            QualityUpdate::OutsideHorizon { .. } => crate::EventDisposition::OutsideHorizon,
            QualityUpdate::Commit { .. } => crate::EventDisposition::Accepted,
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
                    action: action.clone(),
                    observation: ObservationId::new(observation),
                    context: context.clone(),
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
