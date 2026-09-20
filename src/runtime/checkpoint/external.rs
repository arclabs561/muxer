//! Reference-resolved external models share the runtime graph validator.

use super::*;
use crate::profiles::external::ModelReference;
use crate::profiles::external::{ExternalAssessmentsCheckpoint, ExternalReferenceCheckpoint};
use crate::{ExternalAssessments, ExternalDistribution, ExternalScores};

// The two closure adapters differ in selection semantics, not persistence.
macro_rules! referenced_checkpoint {
    ($profile:ident, $wire:ident, $capture:ident, $restore:ident, $label:literal, $reason:pat, $probability:expr) => {
        /// Complete same-build runtime state with caller-resolved model references.
        ///
        /// No model bytes, credentials, or executable closures are serialized.
        #[derive(Clone, serde::Serialize, serde::Deserialize)]
        #[serde(transparent)]
        pub struct $wire(RuntimeCheckpoint<ExternalReferenceCheckpoint, (), ()>);

        impl<C: ?Sized, F> CheckpointProfile for $profile<C, F>
        where
            F: Fn(&C, &[String]) -> BTreeMap<String, f64>,
        {
            const LABEL: &'static str = $label;
            type State = ExternalReferenceCheckpoint;
            type TicketWire = ();

            fn checkpoint_state(&self, key: &str) -> Result<Self::State, PolicyError> {
                self.checkpoint_state(key)
            }
            fn from_checkpoint_state(_: Self::State, _: &str) -> Result<Self, PolicyError> {
                Err(PolicyError::new(
                    "external checkpoint requires a model resolver",
                ))
            }
            fn encode_ticket(_: &()) {}
            fn decode_ticket(_: ()) -> Result<(), PolicyError> {
                Ok(())
            }
            fn checkpoint_expectation(&self) -> FeedbackExpectation {
                FeedbackExpectation::None
            }
            fn validate_receipt_semantics(
                &self,
                receipt: &DecisionReceipt,
            ) -> Result<(), PolicyError> {
                if receipt.batch
                    || receipt.selected.iter().any(|item| {
                        !matches!(item.reason, $reason) || !($probability)(item.probability)
                    })
                {
                    return Err(checkpoint_error::<Self>("receipt selections are invalid"));
                }
                Ok(())
            }
            fn validate_value_channel(&self, _: &Channel, _: &()) -> bool {
                false
            }
            fn validate_open_tickets(
                &self,
                tickets: &[OpenTicket<'_, Self>],
            ) -> Result<(), PolicyError> {
                if tickets.is_empty() {
                    Ok(())
                } else {
                    Err(checkpoint_error::<Self>(
                        "feedbackless model retains tickets",
                    ))
                }
            }
        }

        impl<C: ?Sized, F> Muxer<$profile<C, F>>
        where
            F: Fn(&C, &[String]) -> BTreeMap<String, f64>,
        {
            /// Capture the complete runtime and its explicit model references.
            ///
            /// The application must quiesce capture, protect stored bytes and
            /// fence other writers before restore. Each profile must carry a
            /// model reference identifying immutable caller-owned model state.
            pub fn $capture(&self, build_key: &str) -> Result<$wire, PolicyError> {
                Ok($wire(capture_core(self, build_key)?))
            }
            /// Restore using a fallible caller-owned resolver for every retained model.
            ///
            /// The resolver must return the immutable model identified by the
            /// reference or an error. The caller owns resolver side effects and
            /// external single-writer fencing; local engine reservation occurs
            /// only after every model and the runtime graph validate.
            pub fn $restore<R>(
                checkpoint: $wire,
                build_key: &str,
                mut resolver: R,
            ) -> Result<Self, PolicyError>
            where
                R: FnMut(&ModelReference) -> Result<F, PolicyError>,
            {
                restore_core_with(checkpoint.0, build_key, |state, key| {
                    $profile::from_checkpoint_state_with(state, key, &mut resolver)
                })
            }
        }
    };
}

referenced_checkpoint!(
    ExternalScores,
    ExternalScoresMuxerCheckpoint,
    scores_checkpoint,
    from_scores_checkpoint,
    "external scores",
    DecisionReason::Deterministic,
    |p| matches!(p, ProbabilityAvailability::Exact(value) if value.get() == 1.0)
);
referenced_checkpoint!(
    ExternalDistribution,
    ExternalDistributionMuxerCheckpoint,
    distribution_checkpoint,
    from_distribution_checkpoint,
    "external distribution",
    DecisionReason::CategoricalSample,
    |p| matches!(p, ProbabilityAvailability::Exact(value) if value.get() > 0.0)
);

/// Complete same-build state for a feedbackless assessment runtime.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct AssessmentsMuxerCheckpoint(RuntimeCheckpoint<ExternalAssessmentsCheckpoint, (), ()>);

impl CheckpointProfile for ExternalAssessments {
    const LABEL: &'static str = "external assessments";
    type State = ExternalAssessmentsCheckpoint;
    type TicketWire = ();
    fn checkpoint_state(&self, key: &str) -> Result<Self::State, PolicyError> {
        self.checkpoint_state(key)
    }
    fn from_checkpoint_state(state: Self::State, key: &str) -> Result<Self, PolicyError> {
        Self::from_checkpoint_state(state, key)
    }
    fn encode_ticket(_: &()) {}
    fn decode_ticket(_: ()) -> Result<(), PolicyError> {
        Ok(())
    }
    fn checkpoint_expectation(&self) -> FeedbackExpectation {
        FeedbackExpectation::None
    }
    fn validate_receipt_semantics(&self, receipt: &DecisionReceipt) -> Result<(), PolicyError> {
        if receipt.batch || receipt.selected.iter().any(|item| {
            item.reason != DecisionReason::Deterministic
                || !matches!(item.probability, ProbabilityAvailability::Exact(p) if p.get() == 1.0)
        }) {
            return Err(checkpoint_error::<Self>("receipt selections are invalid"));
        }
        Ok(())
    }
    fn validate_value_channel(&self, _: &Channel, _: &()) -> bool {
        false
    }
    fn validate_open_tickets(&self, tickets: &[OpenTicket<'_, Self>]) -> Result<(), PolicyError> {
        if tickets.is_empty() {
            Ok(())
        } else {
            Err(checkpoint_error::<Self>(
                "feedbackless model retains tickets",
            ))
        }
    }
}

impl Muxer<ExternalAssessments> {
    /// Capture complete runtime state, including self-contained assessment objectives.
    /// The application owns quiescence, stored-byte integrity and writer fencing.
    pub fn assessments_checkpoint(
        &self,
        key: &str,
    ) -> Result<AssessmentsMuxerCheckpoint, PolicyError> {
        Ok(AssessmentsMuxerCheckpoint(capture_core(self, key)?))
    }
    /// Restore a same-build assessment runtime after external single-writer fencing.
    pub fn from_assessments_checkpoint(
        checkpoint: AssessmentsMuxerCheckpoint,
        key: &str,
    ) -> Result<Self, PolicyError> {
        restore_core(checkpoint.0, key)
    }
}
