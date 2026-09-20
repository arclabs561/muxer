//! Feedbackless adapters for caller-owned scores and distributions.

use super::sample_categorical;
use crate::{
    select_candidate_assessments, CandidateAssessment, DecisionReason, FeedbackExpectation,
    InteractionPolicy, MetricObjective, PolicyDecision, PolicyError, PolicyRequest, Probability,
    ProbabilityAvailability, TrialRng,
};
use std::collections::BTreeMap;
use std::convert::Infallible;
use std::marker::PhantomData;

#[cfg(feature = "serde")]
#[path = "external_checkpoint.rs"]
mod external_checkpoint;
#[cfg(feature = "serde")]
pub(crate) use external_checkpoint::{ExternalAssessmentsCheckpoint, ExternalReferenceCheckpoint};

/// A stable, caller-defined identifier for an externally owned model artifact.
///
/// This is an opaque identifier, not a URI, credential, loader, or storage
/// handle. A serialized external-profile checkpoint records it so the caller
/// can resolve the same artifact before resuming the runtime.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModelReference(String);

impl ModelReference {
    /// Construct a non-empty external model artifact identifier.
    pub fn new(value: impl Into<String>) -> Result<Self, PolicyError> {
        let value = value.into();
        if value.is_empty() {
            return Err(PolicyError::new("model reference must not be empty"));
        }
        Ok(Self(value))
    }

    /// Borrow the caller-defined artifact identifier.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Deterministic selection from caller-supplied scores.
///
/// The closure receives canonical eligible actions and must return one finite score
/// for each action. The highest score wins; canonical action order breaks ties.
pub struct ExternalScores<C: ?Sized, F> {
    score: F,
    reference: Option<ModelReference>,
    marker: PhantomData<fn(&C)>,
}

impl<C: ?Sized, F: Clone> Clone for ExternalScores<C, F> {
    fn clone(&self) -> Self {
        Self {
            score: self.score.clone(),
            reference: self.reference.clone(),
            marker: PhantomData,
        }
    }
}

impl<C: ?Sized, F> ExternalScores<C, F> {
    /// Adapt a score closure without requiring a learner or feedback loop.
    #[must_use]
    pub fn new(score: F) -> Self {
        Self {
            score,
            reference: None,
            marker: PhantomData,
        }
    }

    /// Associate this closure with the externally owned artifact needed to
    /// recreate it after serialized continuation.
    #[must_use]
    pub fn with_model_reference(mut self, reference: ModelReference) -> Self {
        self.reference = Some(reference);
        self
    }

    /// Return the artifact identifier supplied for serialized continuation.
    #[must_use]
    pub fn model_reference(&self) -> Option<&ModelReference> {
        self.reference.as_ref()
    }
}

impl<C: ?Sized, F> InteractionPolicy for ExternalScores<C, F>
where
    F: Fn(&C, &[String]) -> BTreeMap<String, f64>,
{
    type Context = C;
    type Feedback = Infallible;
    type CanonicalFeedback = ();
    type Ticket = ();
    type PreparedIssue = ();
    type PreparedUpdate = ();

    fn prepare_decision(
        &self,
        request: PolicyRequest<'_, C>,
        _: &mut TrialRng,
    ) -> Result<PolicyDecision<(), ()>, PolicyError> {
        let scores = (self.score)(request.context(), request.eligible());
        let mut chosen: Option<&String> = None;
        let mut best = f64::NEG_INFINITY;
        for action in request.eligible() {
            let score = scores
                .get(action)
                .copied()
                .ok_or_else(|| PolicyError::new("external scores omitted an eligible action"))?;
            if !score.is_finite() {
                return Err(PolicyError::new("external scores must be finite"));
            }
            if score > best {
                best = score;
                chosen = Some(action);
            }
        }
        let selection = chosen
            .ok_or_else(|| PolicyError::new("no eligible action"))?
            .clone();
        Ok(PolicyDecision {
            selection,
            probability: ProbabilityAvailability::Exact(
                Probability::new(1.0).expect("one is valid"),
            ),
            ticket: None,
            issue: (),
            expectation: FeedbackExpectation::None,
        })
    }
    fn commit_issue(&mut self, _: ()) {}
    fn normalize(&self, feedback: Infallible) -> Result<(), PolicyError> {
        match feedback {}
    }
    fn decision_reason(&self, _: Option<&()>) -> DecisionReason {
        DecisionReason::Deterministic
    }
    fn prepare(&self, _: &(), _: &()) -> Result<(), PolicyError> {
        Ok(())
    }
    fn apply(&mut self, _: ()) {}
}

/// Categorical selection from a caller-supplied probability distribution.
///
/// The closure supplies non-negative finite masses for every eligible action. They
/// are normalized for sampling and the exact normalized selected propensity is
/// recorded on the receipt.
pub struct ExternalDistribution<C: ?Sized, F> {
    distribution: F,
    reference: Option<ModelReference>,
    marker: PhantomData<fn(&C)>,
}

impl<C: ?Sized, F: Clone> Clone for ExternalDistribution<C, F> {
    fn clone(&self) -> Self {
        Self {
            distribution: self.distribution.clone(),
            reference: self.reference.clone(),
            marker: PhantomData,
        }
    }
}

impl<C: ?Sized, F> ExternalDistribution<C, F> {
    /// Adapt a distribution closure without adding a feedback obligation.
    #[must_use]
    pub fn new(distribution: F) -> Self {
        Self {
            distribution,
            reference: None,
            marker: PhantomData,
        }
    }

    /// Associate this closure with the externally owned artifact needed to
    /// recreate it after serialized continuation.
    #[must_use]
    pub fn with_model_reference(mut self, reference: ModelReference) -> Self {
        self.reference = Some(reference);
        self
    }

    /// Return the artifact identifier supplied for serialized continuation.
    #[must_use]
    pub fn model_reference(&self) -> Option<&ModelReference> {
        self.reference.as_ref()
    }
}

impl<C: ?Sized, F> InteractionPolicy for ExternalDistribution<C, F>
where
    F: Fn(&C, &[String]) -> BTreeMap<String, f64>,
{
    type Context = C;
    type Feedback = Infallible;
    type CanonicalFeedback = ();
    type Ticket = ();
    type PreparedIssue = ();
    type PreparedUpdate = ();

    fn prepare_decision(
        &self,
        request: PolicyRequest<'_, C>,
        rng: &mut TrialRng,
    ) -> Result<PolicyDecision<(), ()>, PolicyError> {
        let masses = (self.distribution)(request.context(), request.eligible());
        let (selection, probability) = sample_categorical(request.eligible(), &masses, rng)?;
        Ok(PolicyDecision {
            selection,
            probability: ProbabilityAvailability::Exact(probability),
            ticket: None,
            issue: (),
            expectation: FeedbackExpectation::None,
        })
    }
    fn commit_issue(&mut self, _: ()) {}
    fn normalize(&self, feedback: Infallible) -> Result<(), PolicyError> {
        match feedback {}
    }
    fn decision_reason(&self, _: Option<&()>) -> DecisionReason {
        DecisionReason::CategoricalSample
    }
    fn prepare(&self, _: &(), _: &()) -> Result<(), PolicyError> {
        Ok(())
    }
    fn apply(&mut self, _: ()) {}
}

/// Feedbackless Pareto/scalar selection over caller-produced assessments.
#[derive(Debug, Clone)]
pub struct ExternalAssessments {
    objectives: Vec<MetricObjective>,
}

impl ExternalAssessments {
    /// Construct with explicit metric objectives.
    #[must_use]
    pub fn new(objectives: Vec<MetricObjective>) -> Self {
        Self { objectives }
    }
    /// Borrow the immutable objective definition.
    #[must_use]
    pub fn objectives(&self) -> &[MetricObjective] {
        &self.objectives
    }
}

impl InteractionPolicy for ExternalAssessments {
    type Context = [CandidateAssessment];
    type Feedback = Infallible;
    type CanonicalFeedback = ();
    type Ticket = ();
    type PreparedIssue = ();
    type PreparedUpdate = ();
    fn prepare_decision(
        &self,
        request: PolicyRequest<'_, [CandidateAssessment]>,
        _: &mut TrialRng,
    ) -> Result<PolicyDecision<(), ()>, PolicyError> {
        let assessments: Vec<_> = request
            .context()
            .iter()
            .filter(|item| request.eligible().contains(&item.arm))
            .cloned()
            .collect();
        if assessments.len() != request.eligible().len() {
            return Err(PolicyError::new(
                "external assessments must cover every eligible action exactly once",
            ));
        }
        let selected = select_candidate_assessments(&assessments, &self.objectives)
            .map_err(|error| PolicyError::new(error.to_string()))?;
        let selection = selected
            .chosen
            .ok_or_else(|| PolicyError::new("no eligible assessment"))?;
        Ok(PolicyDecision {
            selection,
            probability: ProbabilityAvailability::Exact(
                Probability::new(1.0).expect("one is valid"),
            ),
            ticket: None,
            issue: (),
            expectation: FeedbackExpectation::None,
        })
    }
    fn commit_issue(&mut self, _: ()) {}
    fn normalize(&self, feedback: Infallible) -> Result<(), PolicyError> {
        match feedback {}
    }
    fn decision_reason(&self, _: Option<&()>) -> DecisionReason {
        DecisionReason::Deterministic
    }
    fn prepare(&self, _: &(), _: &()) -> Result<(), PolicyError> {
        Ok(())
    }
    fn apply(&mut self, _: ()) {}
}
