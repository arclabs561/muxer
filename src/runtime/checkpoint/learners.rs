//! Concrete checkpoint adapters for built-in learner profiles.

use super::*;

#[cfg(feature = "contextual")]
use crate::profiles::contextual::{
    ContextualProfileCheckpoint, ContextualTicketCheckpoint, ContextualTicketState,
};
#[cfg(feature = "boltzmann")]
use crate::profiles::scalar::BoltzmannProfileCheckpoint;
#[cfg(feature = "stochastic")]
use crate::profiles::scalar::Exp3ProfileCheckpoint;
#[cfg(any(feature = "stochastic", feature = "boltzmann"))]
use crate::profiles::scalar::{ScalarProfileTicketCheckpoint, ScalarProfileTicketState};

/// A build-bound, complete serialized snapshot of a fractional Thompson muxer.
#[cfg(feature = "stochastic")]
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct FractionalMuxerCheckpoint(
    pub(super)  RuntimeCheckpoint<
        FractionalThompsonCheckpoint,
        ScalarTicketCheckpoint,
        crate::BoundedReward,
    >,
);

/// A build-bound, complete serialized snapshot of an EXP3 muxer.
#[cfg(feature = "stochastic")]
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct Exp3MuxerCheckpoint(
    pub(super)  RuntimeCheckpoint<
        Exp3ProfileCheckpoint,
        ScalarProfileTicketCheckpoint,
        crate::BoundedReward,
    >,
);

/// A build-bound, complete serialized snapshot of a Boltzmann muxer.
#[cfg(feature = "boltzmann")]
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct BoltzmannMuxerCheckpoint(
    pub(super)  RuntimeCheckpoint<
        BoltzmannProfileCheckpoint,
        ScalarProfileTicketCheckpoint,
        crate::FiniteReward,
    >,
);

/// A build-bound, complete serialized snapshot of a contextual LinUCB muxer.
#[cfg(feature = "contextual")]
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct ContextualMuxerCheckpoint(
    pub(super)  RuntimeCheckpoint<
        ContextualProfileCheckpoint,
        ContextualTicketCheckpoint,
        crate::BoundedReward,
    >,
);

#[cfg(feature = "stochastic")]
impl CheckpointProfile for crate::FractionalThompson {
    const LABEL: &'static str = "fractional";
    learner_checkpoint_basics!(
        FractionalThompsonCheckpoint,
        ScalarTicketCheckpoint,
        crate::FractionalThompson::checkpoint_state,
        crate::FractionalThompson::from_checkpoint_state,
        ScalarTicketCheckpoint::from,
        ScalarTicketCheckpoint::into_ticket,
        crate::FractionalThompson::checkpoint_expectation
    );
    fn validate_receipt_semantics(&self, receipt: &DecisionReceipt) -> Result<(), PolicyError> {
        if receipt.batch
            || receipt.selected.iter().any(|item| {
                !matches!(item.probability, ProbabilityAvailability::Unavailable)
                    || !matches!(
                        item.reason,
                        DecisionReason::ExploreFirst | DecisionReason::PosteriorSample
                    )
            })
        {
            return Err(checkpoint_error::<Self>("receipt selections are invalid"));
        }
        Ok(())
    }
    fn validate_value_channel(&self, channel: &Channel, _value: &crate::BoundedReward) -> bool {
        *channel == Channel::reward()
    }
    fn validate_open_tickets(&self, tickets: &[OpenTicket<'_, Self>]) -> Result<(), PolicyError> {
        let states: Vec<_> = tickets
            .iter()
            .map(|item| crate::profiles::scalar::ScalarTicketState {
                ticket: item.ticket,
                action: item.action,
                reason: item.reason,
                values: item.values,
                missing: item.missing,
            })
            .collect();
        self.validate_checkpoint_tickets(&states)
    }
}

#[cfg(feature = "stochastic")]
impl CheckpointProfile for crate::Exp3Profile {
    const LABEL: &'static str = "EXP3";
    learner_checkpoint_basics!(
        Exp3ProfileCheckpoint,
        ScalarProfileTicketCheckpoint,
        crate::Exp3Profile::checkpoint_state,
        crate::Exp3Profile::from_checkpoint_state,
        ScalarProfileTicketCheckpoint::exp3,
        ScalarProfileTicketCheckpoint::into_exp3,
        crate::Exp3Profile::checkpoint_expectation
    );
    fn validate_receipt_semantics(&self, receipt: &DecisionReceipt) -> Result<(), PolicyError> {
        if receipt.batch
            || receipt.selected.iter().any(|item| {
                !matches!(item.probability, ProbabilityAvailability::Exact(_))
                    || !matches!(
                        item.reason,
                        DecisionReason::ExploreFirst | DecisionReason::CategoricalSample
                    )
            })
        {
            return Err(checkpoint_error::<Self>("receipt selections are invalid"));
        }
        Ok(())
    }
    fn validate_value_channel(&self, channel: &Channel, _value: &crate::BoundedReward) -> bool {
        *channel == Channel::reward()
    }
    fn validate_open_tickets(&self, tickets: &[OpenTicket<'_, Self>]) -> Result<(), PolicyError> {
        let states: Vec<_> = tickets
            .iter()
            .map(|item| ScalarProfileTicketState {
                ticket: item.ticket,
                action: item.action,
                reason: item.reason,
                probability: item.probability,
                values: item.values,
                missing: item.missing,
            })
            .collect();
        self.validate_checkpoint_tickets(&states)
    }
}

#[cfg(feature = "boltzmann")]
impl CheckpointProfile for crate::BoltzmannProfile {
    const LABEL: &'static str = "Boltzmann";
    learner_checkpoint_basics!(
        BoltzmannProfileCheckpoint,
        ScalarProfileTicketCheckpoint,
        crate::BoltzmannProfile::checkpoint_state,
        crate::BoltzmannProfile::from_checkpoint_state,
        ScalarProfileTicketCheckpoint::boltzmann,
        ScalarProfileTicketCheckpoint::into_boltzmann,
        crate::BoltzmannProfile::checkpoint_expectation
    );
    fn validate_receipt_semantics(&self, receipt: &DecisionReceipt) -> Result<(), PolicyError> {
        if receipt.batch
            || receipt.selected.iter().any(|item| {
                !matches!(item.probability, ProbabilityAvailability::Exact(_))
                    || item.reason != DecisionReason::CategoricalSample
            })
        {
            return Err(checkpoint_error::<Self>("receipt selections are invalid"));
        }
        Ok(())
    }
    fn validate_value_channel(&self, channel: &Channel, _value: &crate::FiniteReward) -> bool {
        *channel == Channel::reward()
    }
    fn validate_open_tickets(&self, tickets: &[OpenTicket<'_, Self>]) -> Result<(), PolicyError> {
        let states: Vec<_> = tickets
            .iter()
            .map(|item| ScalarProfileTicketState {
                ticket: item.ticket,
                action: item.action,
                reason: item.reason,
                probability: item.probability,
                values: item.values,
                missing: item.missing,
            })
            .collect();
        self.validate_checkpoint_tickets(&states)
    }
}

#[cfg(feature = "contextual")]
impl CheckpointProfile for crate::ContextualProfile {
    const LABEL: &'static str = "contextual";
    learner_checkpoint_basics!(
        ContextualProfileCheckpoint,
        ContextualTicketCheckpoint,
        |profile: &crate::ContextualProfile, _: &str| profile.checkpoint_state(),
        |state, _: &str| crate::ContextualProfile::from_checkpoint_state(state),
        ContextualTicketCheckpoint::from,
        ContextualTicketCheckpoint::into_ticket,
        crate::ContextualProfile::checkpoint_expectation
    );
    fn validate_receipt_semantics(&self, receipt: &DecisionReceipt) -> Result<(), PolicyError> {
        let valid = !receipt.batch
            && receipt.selected.iter().all(|item| match self.mode() {
                crate::ContextualMode::Deterministic => {
                    matches!(item.probability, ProbabilityAvailability::Exact(probability) if probability.get() == 1.0)
                        && item.reason == DecisionReason::Deterministic
                }
                crate::ContextualMode::Softmax { .. } => {
                    matches!(item.probability, ProbabilityAvailability::Exact(_))
                        && item.reason == DecisionReason::CategoricalSample
                }
            });
        if !valid {
            return Err(checkpoint_error::<Self>(
                "receipt selections are invalid for profile mode",
            ));
        }
        Ok(())
    }
    fn validate_value_channel(&self, channel: &Channel, _value: &crate::BoundedReward) -> bool {
        *channel == Channel::reward()
    }
    fn validate_open_tickets(&self, tickets: &[OpenTicket<'_, Self>]) -> Result<(), PolicyError> {
        let states: Vec<_> = tickets
            .iter()
            .map(|item| ContextualTicketState {
                ticket: item.ticket,
                action: item.action,
                reason: item.reason,
                values: item.values,
                missing: item.missing,
            })
            .collect();
        self.validate_checkpoint_tickets(&states)
    }
}

macro_rules! learner_checkpoint_api {
    ($profile:ty, $wire:ident, $capture:ident, $restore:ident) => {
        impl Muxer<$profile> {
            /// Capture complete, build-bound runtime state without consuming the muxer.
            ///
            /// The non-empty `build_key` identifies the compatible application build;
            /// it is neither authentication nor distributed writer fencing.
            pub fn $capture(&self, build_key: &str) -> Result<$wire, PolicyError> {
                Ok($wire(capture_core(self, build_key)?))
            }

            /// Restore a complete checkpoint after external single-writer fencing.
            pub fn $restore(checkpoint: $wire, build_key: &str) -> Result<Self, PolicyError> {
                restore_core::<$profile>(checkpoint.0, build_key)
            }
        }
    };
}

#[cfg(feature = "stochastic")]
learner_checkpoint_api!(
    crate::FractionalThompson,
    FractionalMuxerCheckpoint,
    fractional_checkpoint,
    from_fractional_checkpoint
);
#[cfg(feature = "stochastic")]
learner_checkpoint_api!(
    crate::Exp3Profile,
    Exp3MuxerCheckpoint,
    exp3_checkpoint,
    from_exp3_checkpoint
);
#[cfg(feature = "boltzmann")]
learner_checkpoint_api!(
    crate::BoltzmannProfile,
    BoltzmannMuxerCheckpoint,
    boltzmann_checkpoint,
    from_boltzmann_checkpoint
);
#[cfg(feature = "contextual")]
learner_checkpoint_api!(
    crate::ContextualProfile,
    ContextualMuxerCheckpoint,
    contextual_checkpoint,
    from_contextual_checkpoint
);
