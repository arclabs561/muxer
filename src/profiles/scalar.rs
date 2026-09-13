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

#[cfg(all(feature = "serde", feature = "stochastic"))]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BernoulliThompsonCheckpoint {
    schema: u32,
    kind: String,
    crate_version: String,
    build_key: String,
    config: ThompsonConfigCheckpoint,
    posterior: Vec<ThompsonArmCheckpoint>,
    legacy_seed: u64,
    rng: crate::TrialRngState,
}

#[cfg(all(feature = "serde", feature = "stochastic"))]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ThompsonConfigCheckpoint {
    alpha0: u64,
    beta0: u64,
    decay: u64,
    priors: Vec<(String, u64, u64)>,
}

#[cfg(all(feature = "serde", feature = "stochastic"))]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ThompsonArmCheckpoint {
    action: String,
    alpha: u64,
    beta: u64,
    uses: u64,
}

#[cfg(all(feature = "serde", feature = "stochastic"))]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ScalarTicketCheckpoint {
    action: String,
    reason: DecisionReason,
}

#[cfg(all(feature = "serde", feature = "stochastic"))]
pub(crate) struct ScalarTicketState<'a> {
    pub(crate) ticket: &'a ScalarTicket,
    pub(crate) action: &'a str,
    pub(crate) reason: DecisionReason,
    pub(crate) values: &'a [(Channel, bool)],
    pub(crate) missing: &'a [Channel],
}

#[cfg(all(feature = "serde", feature = "stochastic"))]
impl From<&ScalarTicket> for ScalarTicketCheckpoint {
    fn from(ticket: &ScalarTicket) -> Self {
        Self {
            action: ticket.action.clone(),
            reason: ticket.reason,
        }
    }
}

#[cfg(all(feature = "serde", feature = "stochastic"))]
impl ScalarTicketCheckpoint {
    pub(crate) fn into_ticket(self) -> Result<ScalarTicket, PolicyError> {
        if self.action.is_empty() {
            return Err(PolicyError::new(
                "Bernoulli checkpoint ticket action is empty",
            ));
        }
        if !matches!(
            self.reason,
            DecisionReason::ExploreFirst | DecisionReason::PosteriorSample
        ) {
            return Err(PolicyError::new(
                "Bernoulli checkpoint ticket reason is invalid",
            ));
        }
        Ok(ScalarTicket {
            action: self.action,
            reason: self.reason,
        })
    }
}

#[cfg(feature = "stochastic")]
/// Beta-Bernoulli Thompson sampling with boolean feedback.
#[derive(Debug, Clone)]
pub struct BernoulliThompson {
    inner: ThompsonSampling,
    #[cfg(all(feature = "serde", feature = "stochastic"))]
    legacy_seed: u64,
    rng: TrialRng,
}

#[cfg(feature = "stochastic")]
impl BernoulliThompson {
    /// Construct from a Thompson configuration.
    #[must_use]
    pub fn new(config: ThompsonConfig) -> Self {
        Self {
            inner: ThompsonSampling::new(config),
            #[cfg(all(feature = "serde", feature = "stochastic"))]
            legacy_seed: 0,
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
            #[cfg(all(feature = "serde", feature = "stochastic"))]
            legacy_seed: seed,
            rng: TrialRng::seeded(seed),
        }
    }
    /// Inspect the underlying Thompson kernel.
    #[must_use]
    pub fn inner(&self) -> &ThompsonSampling {
        &self.inner
    }

    #[cfg(all(feature = "serde", feature = "stochastic"))]
    pub(crate) fn checkpoint_expectation(&self) -> FeedbackExpectation {
        FeedbackExpectation::FinalValue {
            channel: Channel::reward(),
        }
    }

    #[cfg(all(feature = "serde", feature = "stochastic"))]
    pub(crate) fn checkpoint_state(
        &self,
        build_key: &str,
    ) -> Result<BernoulliThompsonCheckpoint, PolicyError> {
        if build_key.is_empty() {
            return Err(PolicyError::new("Bernoulli checkpoint build key is empty"));
        }
        if !self.inner.has_initial_rng(self.legacy_seed) {
            return Err(PolicyError::new(
                "Bernoulli checkpoint legacy RNG does not match its recorded seed",
            ));
        }
        let config = checkpoint_config(self.inner.config());
        let mut posterior = Vec::with_capacity(self.inner.stats().len());
        for (action, stats) in self.inner.stats() {
            if action.is_empty()
                || !stats.alpha.is_finite()
                || stats.alpha <= 0.0
                || !stats.beta.is_finite()
                || stats.beta <= 0.0
            {
                return Err(PolicyError::new(
                    "Bernoulli checkpoint cannot capture an invalid posterior",
                ));
            }
            posterior.push(ThompsonArmCheckpoint {
                action: action.clone(),
                alpha: stats.alpha.to_bits(),
                beta: stats.beta.to_bits(),
                uses: stats.uses,
            });
        }
        Ok(BernoulliThompsonCheckpoint {
            schema: 1,
            kind: "bernoulli-thompson".to_owned(),
            crate_version: env!("CARGO_PKG_VERSION").to_owned(),
            build_key: build_key.to_owned(),
            config,
            posterior,
            legacy_seed: self.legacy_seed,
            rng: self.rng.state(),
        })
    }

    #[cfg(all(feature = "serde", feature = "stochastic"))]
    pub(crate) fn from_checkpoint_state(
        state: BernoulliThompsonCheckpoint,
        build_key: &str,
    ) -> Result<Self, PolicyError> {
        if state.schema != 1 {
            return Err(PolicyError::new("unsupported Bernoulli checkpoint schema"));
        }
        if state.kind != "bernoulli-thompson" {
            return Err(PolicyError::new("Bernoulli checkpoint kind mismatch"));
        }
        if state.crate_version != env!("CARGO_PKG_VERSION") {
            return Err(PolicyError::new(
                "Bernoulli checkpoint crate version mismatch",
            ));
        }
        if build_key.is_empty() || state.build_key != build_key {
            return Err(PolicyError::new("Bernoulli checkpoint build key mismatch"));
        }
        let config = restore_config(state.config)?;
        let mut arms = std::collections::BTreeMap::new();
        for arm in state.posterior {
            if arm.action.is_empty() || arms.contains_key(&arm.action) {
                return Err(PolicyError::new(
                    "Bernoulli checkpoint repeats or empties a posterior action",
                ));
            }
            let alpha = f64::from_bits(arm.alpha);
            let beta = f64::from_bits(arm.beta);
            if !alpha.is_finite() || alpha <= 0.0 || !beta.is_finite() || beta <= 0.0 {
                return Err(PolicyError::new(
                    "Bernoulli checkpoint posterior is invalid",
                ));
            }
            arms.insert(
                arm.action,
                crate::BetaStats {
                    alpha,
                    beta,
                    uses: arm.uses,
                },
            );
        }
        let rng = TrialRng::from_state(state.rng)?;
        let mut inner = ThompsonSampling::with_seed(config, state.legacy_seed);
        inner.restore(crate::ThompsonState { arms });
        Ok(Self {
            inner,
            legacy_seed: state.legacy_seed,
            rng,
        })
    }

    #[cfg(all(feature = "serde", feature = "stochastic"))]
    pub(crate) fn validate_checkpoint_tickets(
        &self,
        tickets: &[ScalarTicketState<'_>],
    ) -> Result<(), PolicyError> {
        let reward = Channel::reward();
        for state in tickets {
            if state.ticket.action != state.action || state.ticket.reason != state.reason {
                return Err(PolicyError::new(
                    "Bernoulli checkpoint ticket disagrees with receipt",
                ));
            }
            if !matches!(
                state.reason,
                DecisionReason::ExploreFirst | DecisionReason::PosteriorSample
            ) {
                return Err(PolicyError::new(
                    "Bernoulli checkpoint receipt reason is invalid",
                ));
            }
            if !self.inner.stats().contains_key(state.action) {
                return Err(PolicyError::new(
                    "Bernoulli checkpoint ticket action lacks issued kernel state",
                ));
            }
            if state.values.iter().any(|(channel, _)| channel != &reward)
                || state.missing.iter().any(|channel| channel != &reward)
            {
                return Err(PolicyError::new(
                    "Bernoulli checkpoint ticket has an unexpected channel",
                ));
            }
        }
        Ok(())
    }
}

#[cfg(all(feature = "serde", feature = "stochastic"))]
fn checkpoint_config(config: &ThompsonConfig) -> ThompsonConfigCheckpoint {
    ThompsonConfigCheckpoint {
        alpha0: config.alpha0.to_bits(),
        beta0: config.beta0.to_bits(),
        decay: config.decay.to_bits(),
        priors: config
            .priors
            .iter()
            .map(|(action, (alpha, beta))| (action.clone(), alpha.to_bits(), beta.to_bits()))
            .collect(),
    }
}

#[cfg(all(feature = "serde", feature = "stochastic"))]
fn restore_config(state: ThompsonConfigCheckpoint) -> Result<ThompsonConfig, PolicyError> {
    let mut priors = std::collections::BTreeMap::new();
    for (action, alpha, beta) in state.priors {
        if priors
            .insert(action, (f64::from_bits(alpha), f64::from_bits(beta)))
            .is_some()
        {
            return Err(PolicyError::new(
                "Bernoulli checkpoint repeats a prior action",
            ));
        }
    }
    Ok(ThompsonConfig {
        alpha0: f64::from_bits(state.alpha0),
        beta0: f64::from_bits(state.beta0),
        decay: f64::from_bits(state.decay),
        priors,
    })
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

    #[cfg(all(feature = "serde", feature = "stochastic"))]
    #[test]
    fn bernoulli_checkpoint_preserves_profile_and_legacy_seeded_streams() {
        let actions = vec!["a".to_owned(), "b".to_owned()];
        let mut original = BernoulliThompson::with_seed(ThompsonConfig::default(), 73);
        let mut runtime_rng = TrialRng::seeded(5);
        for reward in [true, false] {
            let decision = original
                .prepare_decision(PolicyRequest::new(&actions, &()), &mut runtime_rng)
                .unwrap();
            original.commit_issue(decision.issue);
            original.apply(
                original
                    .prepare(&decision.ticket.unwrap(), &reward)
                    .unwrap(),
            );
        }
        let restored = BernoulliThompson::from_checkpoint_state(
            original.checkpoint_state("same-build").unwrap(),
            "same-build",
        )
        .unwrap();
        let mut original_legacy = original.inner().clone();
        let mut restored_legacy = restored.inner().clone();
        for _ in 0..12 {
            assert_eq!(
                original_legacy.decide(&actions).unwrap().chosen,
                restored_legacy.decide(&actions).unwrap().chosen
            );
        }
        let mut restored = restored;
        for _ in 0..12 {
            let left = original
                .prepare_decision(PolicyRequest::new(&actions, &()), &mut runtime_rng)
                .unwrap();
            let right = restored
                .prepare_decision(PolicyRequest::new(&actions, &()), &mut runtime_rng)
                .unwrap();
            assert_eq!(left.selection, right.selection);
            original.commit_issue(left.issue);
            restored.commit_issue(right.issue);
        }
    }

    #[cfg(feature = "serde")]
    fn trained_bernoulli_checkpoint() -> BernoulliThompsonCheckpoint {
        let actions = vec!["a".to_owned()];
        let mut profile = BernoulliThompson::with_seed(ThompsonConfig::default(), 29);
        let mut runtime_rng = TrialRng::seeded(3);
        let decision = profile
            .prepare_decision(PolicyRequest::new(&actions, &()), &mut runtime_rng)
            .unwrap();
        let ticket = decision.ticket.unwrap();
        profile.commit_issue(decision.issue);
        let update = profile.prepare(&ticket, &true).unwrap();
        profile.apply(update);
        profile.checkpoint_state("same-build").unwrap()
    }

    #[cfg(feature = "serde")]
    #[test]
    fn bernoulli_checkpoint_rejects_malformed_posteriors() {
        for alpha in [0.0, f64::NAN, f64::INFINITY] {
            let mut state = trained_bernoulli_checkpoint();
            state.posterior[0].alpha = alpha.to_bits();
            assert!(BernoulliThompson::from_checkpoint_state(state, "same-build").is_err());
        }

        let mut duplicate = trained_bernoulli_checkpoint();
        duplicate.posterior.push(duplicate.posterior[0].clone());
        assert!(BernoulliThompson::from_checkpoint_state(duplicate, "same-build").is_err());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn bernoulli_checkpoint_preserves_raw_config_bits_and_empty_prior_key() {
        let mut config = ThompsonConfig {
            alpha0: f64::from_bits(0x7ff8_0000_0000_0031),
            beta0: f64::INFINITY,
            decay: -0.0,
            priors: std::collections::BTreeMap::new(),
        };
        config.priors.insert(
            String::new(),
            (f64::NEG_INFINITY, f64::from_bits(0x7ff8_0000_0000_0042)),
        );

        let profile = BernoulliThompson::with_seed(config, 17);
        let saved = profile.checkpoint_state("same-build").unwrap();
        let restored =
            BernoulliThompson::from_checkpoint_state(saved.clone(), "same-build").unwrap();
        let round_tripped = restored.checkpoint_state("same-build").unwrap();

        assert_eq!(saved.config.alpha0, round_tripped.config.alpha0);
        assert_eq!(saved.config.beta0, round_tripped.config.beta0);
        assert_eq!(saved.config.decay, round_tripped.config.decay);
        assert_eq!(saved.config.priors, round_tripped.config.priors);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn bernoulli_checkpoint_rejects_poisoned_posterior_without_mutation() {
        let actions = vec!["poison".to_owned()];
        let config = ThompsonConfig {
            alpha0: f64::NAN,
            ..ThompsonConfig::default()
        };
        let mut profile = BernoulliThompson::with_seed(config, 11);
        let mut runtime_rng = TrialRng::seeded(7);
        let decision = profile
            .prepare_decision(PolicyRequest::new(&actions, &()), &mut runtime_rng)
            .unwrap();
        let ticket = decision.ticket.unwrap();
        profile.commit_issue(decision.issue);
        let update = profile.prepare(&ticket, &true).unwrap();
        profile.apply(update);

        let before_posterior = profile.inner().stats()["poison"].alpha.to_bits();
        let before_rng = profile.rng.state();
        assert!(profile.checkpoint_state("same-build").is_err());
        assert_eq!(
            profile.inner().stats()["poison"].alpha.to_bits(),
            before_posterior
        );
        assert_eq!(profile.rng.state(), before_rng);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn bernoulli_checkpoint_rejects_applied_kernel_with_a_different_seed() {
        let mut profile = BernoulliThompson::with_seed(ThompsonConfig::default(), 11);
        profile.apply(ThompsonSampling::with_seed(ThompsonConfig::default(), 12));

        assert!(profile.checkpoint_state("same-build").is_err());
        assert!(!profile.inner().has_initial_rng(11));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn bernoulli_checkpoint_rejects_issue_committed_from_a_different_seed() {
        let actions = vec!["a".to_owned()];
        let source = BernoulliThompson::with_seed(ThompsonConfig::default(), 12);
        let mut destination = BernoulliThompson::with_seed(ThompsonConfig::default(), 11);
        let mut runtime_rng = TrialRng::seeded(1);
        let decision = source
            .prepare_decision(PolicyRequest::new(&actions, &()), &mut runtime_rng)
            .unwrap();
        destination.commit_issue(decision.issue);

        assert!(destination.checkpoint_state("same-build").is_err());
        assert!(!destination.inner().has_initial_rng(11));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn bernoulli_checkpoint_rejects_applied_kernel_with_advanced_legacy_rng() {
        let actions = vec!["a".to_owned()];
        let mut kernel = ThompsonSampling::with_seed(ThompsonConfig::default(), 11);
        kernel.update_reward("a", 1.0);
        let sampled = kernel.decide(&actions).unwrap();
        assert!(matches!(
            sampled.notes.as_slice(),
            [DecisionNote::SampledPosteriorMax]
        ));

        let mut profile = BernoulliThompson::with_seed(ThompsonConfig::default(), 11);
        profile.apply(kernel);
        let before_alpha = profile.inner().stats()["a"].alpha.to_bits();
        assert!(profile.checkpoint_state("same-build").is_err());
        assert_eq!(profile.inner().stats()["a"].alpha.to_bits(), before_alpha);
        assert!(!profile.inner().has_initial_rng(11));
    }
}
