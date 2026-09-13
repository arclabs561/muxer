//! Scalar-reward profiles backed by the crate's existing bandit kernels.

#[cfg(feature = "boltzmann")]
use super::sample_categorical;
#[cfg(feature = "stochastic")]
use super::BoundedReward;
#[cfg(feature = "boltzmann")]
use super::FiniteReward;
#[cfg(feature = "boltzmann")]
use crate::BanditPolicy;
#[cfg(feature = "stochastic")]
use crate::DecisionNote;
#[cfg(feature = "stochastic")]
use crate::Probability;
use crate::{
    Channel, DecisionReason, FeedbackExpectation, InteractionPolicy, PolicyDecision, PolicyError,
    PolicyRequest, ProbabilityAvailability, TrialRng,
};
#[cfg(feature = "stochastic")]
use crate::{Exp3Ix, Exp3IxConfig, ThompsonConfig, ThompsonSampling};

#[cfg(any(feature = "stochastic", feature = "boltzmann"))]
/// Retained scalar decision identity used to attribute delayed feedback.
#[derive(Debug, Clone)]
pub struct ScalarTicket {
    action: String,
    reason: DecisionReason,
}

#[cfg(feature = "stochastic")]
/// One-use prepared Thompson issuance, including its portable random continuation.
#[derive(Debug)]
pub struct ThompsonIssue {
    kernel: ThompsonSampling,
    rng: TrialRng,
}

#[cfg(feature = "stochastic")]
/// Beta-Bernoulli Thompson sampling with boolean feedback.
#[derive(Debug, Clone)]
pub struct BernoulliThompson {
    inner: ThompsonSampling,
    rng: TrialRng,
}

#[cfg(feature = "stochastic")]
impl BernoulliThompson {
    /// Construct from a Thompson configuration.
    #[must_use]
    pub fn new(config: ThompsonConfig) -> Self {
        Self {
            inner: ThompsonSampling::new(config),
            rng: TrialRng::seeded(0),
        }
    }
    /// Construct with a reproducible profile seed.
    ///
    /// The seed drives the profile's checkpointable issuance stream. This is
    /// distinct from the legacy low-level kernel trace, which remains available
    /// through [`ThompsonSampling::with_seed`].
    #[must_use]
    pub fn with_seed(config: ThompsonConfig, seed: u64) -> Self {
        Self {
            inner: ThompsonSampling::with_seed(config, seed),
            rng: TrialRng::seeded(seed),
        }
    }
    /// Inspect the underlying Thompson kernel.
    #[must_use]
    pub fn inner(&self) -> &ThompsonSampling {
        &self.inner
    }
}

#[cfg(feature = "stochastic")]
impl InteractionPolicy for BernoulliThompson {
    type Context = ();
    type Feedback = bool;
    type CanonicalFeedback = bool;
    type Ticket = ScalarTicket;
    type PreparedIssue = ThompsonIssue;
    type PreparedUpdate = ThompsonSampling;

    fn prepare_decision(
        &self,
        request: PolicyRequest<'_, ()>,
        _: &mut TrialRng,
    ) -> Result<PolicyDecision<Self::Ticket, Self::PreparedIssue>, PolicyError> {
        let mut next = self.inner.clone();
        let mut rng = self.rng.clone();
        let decision = next
            .decide_with_rng(request.eligible(), &mut rng)
            .ok_or_else(|| PolicyError::new("no eligible action"))?;
        let reason = reason_from_notes(&decision.notes);
        Ok(PolicyDecision {
            selection: decision.chosen.clone(),
            probability: ProbabilityAvailability::Unavailable,
            ticket: Some(ScalarTicket {
                action: decision.chosen,
                reason,
            }),
            issue: ThompsonIssue { kernel: next, rng },
            expectation: FeedbackExpectation::FinalValue {
                channel: Channel::reward(),
            },
        })
    }
    fn commit_issue(&mut self, issue: Self::PreparedIssue) {
        self.inner = issue.kernel;
        self.rng = issue.rng;
    }
    fn normalize(&self, feedback: bool) -> Result<bool, PolicyError> {
        Ok(feedback)
    }
    fn decision_reason(&self, ticket: Option<&ScalarTicket>) -> DecisionReason {
        ticket.map_or(DecisionReason::Unspecified, |ticket| ticket.reason)
    }
    fn prepare(
        &self,
        ticket: &Self::Ticket,
        feedback: &bool,
    ) -> Result<Self::PreparedUpdate, PolicyError> {
        let mut next = self.inner.clone();
        next.update_reward(&ticket.action, f64::from(*feedback));
        Ok(next)
    }
    fn apply(&mut self, update: Self::PreparedUpdate) {
        self.inner = update;
    }
}

#[cfg(feature = "stochastic")]
/// Thompson sampling with explicit bounded fractional feedback.
#[derive(Debug, Clone)]
pub struct FractionalThompson {
    inner: ThompsonSampling,
    rng: TrialRng,
}

#[cfg(feature = "stochastic")]
impl FractionalThompson {
    /// Construct from a Thompson configuration.
    #[must_use]
    pub fn new(config: ThompsonConfig) -> Self {
        Self {
            inner: ThompsonSampling::new(config),
            rng: TrialRng::seeded(0),
        }
    }
    /// Construct with a reproducible profile seed.
    ///
    /// The seed drives the profile's checkpointable issuance stream. This is
    /// distinct from the legacy low-level kernel trace, which remains available
    /// through [`ThompsonSampling::with_seed`].
    #[must_use]
    pub fn with_seed(config: ThompsonConfig, seed: u64) -> Self {
        Self {
            inner: ThompsonSampling::with_seed(config, seed),
            rng: TrialRng::seeded(seed),
        }
    }
    /// Inspect the underlying Thompson kernel.
    #[must_use]
    pub fn inner(&self) -> &ThompsonSampling {
        &self.inner
    }
}

#[cfg(feature = "stochastic")]
impl InteractionPolicy for FractionalThompson {
    type Context = ();
    type Feedback = BoundedReward;
    type CanonicalFeedback = BoundedReward;
    type Ticket = ScalarTicket;
    type PreparedIssue = ThompsonIssue;
    type PreparedUpdate = ThompsonSampling;

    fn prepare_decision(
        &self,
        request: PolicyRequest<'_, ()>,
        _: &mut TrialRng,
    ) -> Result<PolicyDecision<Self::Ticket, Self::PreparedIssue>, PolicyError> {
        let mut next = self.inner.clone();
        let mut rng = self.rng.clone();
        let decision = next
            .decide_with_rng(request.eligible(), &mut rng)
            .ok_or_else(|| PolicyError::new("no eligible action"))?;
        let reason = reason_from_notes(&decision.notes);
        Ok(PolicyDecision {
            selection: decision.chosen.clone(),
            probability: ProbabilityAvailability::Unavailable,
            ticket: Some(ScalarTicket {
                action: decision.chosen,
                reason,
            }),
            issue: ThompsonIssue { kernel: next, rng },
            expectation: FeedbackExpectation::FinalValue {
                channel: Channel::reward(),
            },
        })
    }
    fn commit_issue(&mut self, issue: Self::PreparedIssue) {
        self.inner = issue.kernel;
        self.rng = issue.rng;
    }
    fn normalize(&self, feedback: BoundedReward) -> Result<BoundedReward, PolicyError> {
        Ok(feedback)
    }
    fn decision_reason(&self, ticket: Option<&ScalarTicket>) -> DecisionReason {
        ticket.map_or(DecisionReason::Unspecified, |ticket| ticket.reason)
    }
    fn prepare(
        &self,
        ticket: &Self::Ticket,
        feedback: &BoundedReward,
    ) -> Result<Self::PreparedUpdate, PolicyError> {
        let mut next = self.inner.clone();
        next.update_reward(&ticket.action, feedback.get());
        Ok(next)
    }
    fn apply(&mut self, update: Self::PreparedUpdate) {
        self.inner = update;
    }
}

#[cfg(feature = "stochastic")]
/// EXP3-IX with a fixed canonical arm universe and ticketed propensities.
#[derive(Debug, Clone)]
pub struct Exp3Profile {
    inner: Exp3Ix,
    universe: Vec<String>,
}

#[cfg(feature = "stochastic")]
#[derive(Debug, Clone)]
pub struct Exp3Ticket {
    action: String,
    propensity: f64,
    reason: DecisionReason,
}

#[cfg(feature = "stochastic")]
impl Exp3Profile {
    /// Construct an EXP3 profile over one immutable, unique action universe.
    pub fn new(universe: Vec<String>, config: Exp3IxConfig) -> Result<Self, PolicyError> {
        validate_universe(&universe)?;
        Ok(Self {
            inner: Exp3Ix::new(config),
            universe,
        })
    }
    /// Inspect the canonical universe used for all EXP3 loss coordinates.
    #[must_use]
    pub fn universe(&self) -> &[String] {
        &self.universe
    }
    /// Inspect the underlying EXP3-IX kernel.
    #[must_use]
    pub fn inner(&self) -> &Exp3Ix {
        &self.inner
    }
}

#[cfg(feature = "stochastic")]
impl InteractionPolicy for Exp3Profile {
    type Context = ();
    type Feedback = BoundedReward;
    type CanonicalFeedback = BoundedReward;
    type Ticket = Exp3Ticket;
    type PreparedIssue = Exp3Ix;
    type PreparedUpdate = Exp3Ix;

    fn prepare_decision(
        &self,
        request: PolicyRequest<'_, ()>,
        rng: &mut TrialRng,
    ) -> Result<PolicyDecision<Self::Ticket, Self::PreparedIssue>, PolicyError> {
        if request
            .eligible()
            .iter()
            .any(|action| !self.universe.contains(action))
        {
            return Err(PolicyError::new(
                "eligible action is outside the EXP3 canonical universe",
            ));
        }
        let mut next = self.inner.clone();
        let decision = next
            .decide_deterministic_filtered(&self.universe, request.eligible(), rng.next_u64())
            .ok_or_else(|| PolicyError::new("no eligible action"))?;
        let propensity = decision
            .probs
            .as_ref()
            .and_then(|map| map.get(&decision.chosen))
            .copied()
            .ok_or_else(|| PolicyError::new("EXP3 did not provide selected propensity"))?;
        let probability = Probability::new(propensity)
            .map_err(|_| PolicyError::new("EXP3 produced invalid propensity"))?;
        let reason = reason_from_notes(&decision.notes);
        Ok(PolicyDecision {
            selection: decision.chosen.clone(),
            probability: ProbabilityAvailability::Exact(probability),
            ticket: Some(Exp3Ticket {
                action: decision.chosen,
                propensity,
                reason,
            }),
            issue: next,
            expectation: FeedbackExpectation::FinalValue {
                channel: Channel::reward(),
            },
        })
    }
    fn commit_issue(&mut self, issue: Self::PreparedIssue) {
        self.inner = issue;
    }
    fn normalize(&self, feedback: BoundedReward) -> Result<BoundedReward, PolicyError> {
        Ok(feedback)
    }
    fn decision_reason(&self, ticket: Option<&Exp3Ticket>) -> DecisionReason {
        ticket.map_or(DecisionReason::Unspecified, |ticket| ticket.reason)
    }
    fn prepare(
        &self,
        ticket: &Self::Ticket,
        feedback: &BoundedReward,
    ) -> Result<Self::PreparedUpdate, PolicyError> {
        let mut next = self.inner.clone();
        next.update_reward_with_prob(&ticket.action, feedback.get(), ticket.propensity);
        Ok(next)
    }
    fn apply(&mut self, update: Self::PreparedUpdate) {
        self.inner = update;
    }
}

#[cfg(feature = "stochastic")]
fn validate_universe(universe: &[String]) -> Result<(), PolicyError> {
    if universe.is_empty() {
        return Err(PolicyError::new("EXP3 universe must not be empty"));
    }
    let mut seen = std::collections::BTreeSet::new();
    if universe
        .iter()
        .any(|action| action.is_empty() || !seen.insert(action))
    {
        return Err(PolicyError::new(
            "EXP3 universe must contain unique non-empty actions",
        ));
    }
    Ok(())
}

#[cfg(feature = "boltzmann")]
/// Softmax-temperature allocation sampled by the runtime's transactional trial RNG.
#[derive(Debug, Clone)]
pub struct BoltzmannProfile {
    inner: crate::BoltzmannPolicy,
}

#[cfg(feature = "boltzmann")]
impl BoltzmannProfile {
    /// Construct from a Boltzmann configuration.
    #[must_use]
    pub fn new(config: crate::BoltzmannConfig) -> Self {
        Self {
            inner: crate::BoltzmannPolicy::new(config),
        }
    }
    /// Inspect the underlying Boltzmann kernel.
    #[must_use]
    pub fn inner(&self) -> &crate::BoltzmannPolicy {
        &self.inner
    }
}

#[cfg(feature = "boltzmann")]
impl InteractionPolicy for BoltzmannProfile {
    type Context = ();
    type Feedback = FiniteReward;
    type CanonicalFeedback = FiniteReward;
    type Ticket = ScalarTicket;
    type PreparedIssue = ();
    type PreparedUpdate = crate::BoltzmannPolicy;

    fn prepare_decision(
        &self,
        request: PolicyRequest<'_, ()>,
        rng: &mut TrialRng,
    ) -> Result<PolicyDecision<Self::Ticket, Self::PreparedIssue>, PolicyError> {
        let probabilities = self.inner.probs(request.eligible());
        let (selection, probability) = sample_categorical(request.eligible(), &probabilities, rng)?;
        Ok(PolicyDecision {
            selection: selection.clone(),
            probability: ProbabilityAvailability::Exact(probability),
            ticket: Some(ScalarTicket {
                action: selection,
                reason: DecisionReason::CategoricalSample,
            }),
            issue: (),
            expectation: FeedbackExpectation::FinalValue {
                channel: Channel::reward(),
            },
        })
    }
    fn commit_issue(&mut self, _: ()) {}
    fn normalize(&self, feedback: FiniteReward) -> Result<FiniteReward, PolicyError> {
        Ok(feedback)
    }
    fn decision_reason(&self, ticket: Option<&ScalarTicket>) -> DecisionReason {
        ticket.map_or(DecisionReason::Unspecified, |ticket| ticket.reason)
    }
    fn prepare(
        &self,
        ticket: &Self::Ticket,
        feedback: &FiniteReward,
    ) -> Result<Self::PreparedUpdate, PolicyError> {
        let mut next = self.inner.clone();
        next.update_reward(&ticket.action, feedback.get());
        Ok(next)
    }
    fn apply(&mut self, update: Self::PreparedUpdate) {
        self.inner = update;
    }
}

#[cfg(feature = "stochastic")]
fn reason_from_notes(notes: &[DecisionNote]) -> DecisionReason {
    if notes
        .iter()
        .any(|note| matches!(note, DecisionNote::ExploreFirst))
    {
        DecisionReason::ExploreFirst
    } else if notes
        .iter()
        .any(|note| matches!(note, DecisionNote::SampledPosteriorMax))
    {
        DecisionReason::PosteriorSample
    } else {
        DecisionReason::CategoricalSample
    }
}

#[cfg(all(test, feature = "stochastic"))]
mod rng_tests {
    use super::*;

    fn check_transactional_stream<P>(
        mut profile: P,
        reward: P::CanonicalFeedback,
        state: impl Fn(&P) -> crate::TrialRngState,
    ) where
        P: InteractionPolicy<Context = (), Ticket = ScalarTicket, PreparedIssue = ThompsonIssue>,
    {
        let actions = vec!["a".to_owned(), "b".to_owned()];
        let mut runtime_rng = TrialRng::seeded(99);
        for _ in 0..actions.len() {
            let decision = profile
                .prepare_decision(PolicyRequest::new(&actions, &()), &mut runtime_rng)
                .unwrap();
            profile.commit_issue(decision.issue);
            let update = profile.prepare(&decision.ticket.unwrap(), &reward).unwrap();
            profile.apply(update);
        }

        let before = state(&profile);
        let discarded = profile
            .prepare_decision(PolicyRequest::new(&actions, &()), &mut runtime_rng)
            .unwrap();
        assert_eq!(state(&profile), before);
        assert_ne!(discarded.issue.rng.state(), before);
        let retried = profile
            .prepare_decision(PolicyRequest::new(&actions, &()), &mut runtime_rng)
            .unwrap();
        assert_eq!(discarded.selection, retried.selection);
        assert_eq!(discarded.issue.rng.state(), retried.issue.rng.state());
        let delayed_ticket = retried.ticket.unwrap();
        profile.commit_issue(retried.issue);

        let next = profile
            .prepare_decision(PolicyRequest::new(&actions, &()), &mut runtime_rng)
            .unwrap();
        profile.commit_issue(next.issue);
        let before_feedback = state(&profile);
        let update = profile.prepare(&delayed_ticket, &reward).unwrap();
        profile.apply(update);
        assert_eq!(state(&profile), before_feedback);
    }

    #[test]
    fn bernoulli_preparation_and_delayed_feedback_preserve_stream_ownership() {
        check_transactional_stream(
            BernoulliThompson::with_seed(ThompsonConfig::default(), 17),
            true,
            |profile| profile.rng.state(),
        );
    }

    #[test]
    fn fractional_preparation_and_delayed_feedback_preserve_stream_ownership() {
        check_transactional_stream(
            FractionalThompson::with_seed(ThompsonConfig::default(), 17),
            BoundedReward::new(0.6).unwrap(),
            |profile| profile.rng.state(),
        );
    }
}
