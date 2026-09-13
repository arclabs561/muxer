use muxer::{
    Channel, FeedbackExpectation, InteractionPolicy, Muxer, PolicyDecision, PolicyError,
    PolicyRequest, ProbabilityAvailability, TrialRng,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

struct EpochPolicy {
    applied: Arc<AtomicU64>,
}

impl InteractionPolicy for EpochPolicy {
    type Context = ();
    type Feedback = bool;
    type CanonicalFeedback = bool;
    type Ticket = ();
    type PreparedIssue = ();
    type PreparedUpdate = bool;
    fn prepare_decision(
        &self,
        request: PolicyRequest<'_, ()>,
        _: &mut TrialRng,
    ) -> Result<PolicyDecision<(), ()>, PolicyError> {
        Ok(PolicyDecision {
            selection: request.eligible()[0].clone(),
            probability: ProbabilityAvailability::Unavailable,
            ticket: Some(()),
            issue: (),
            expectation: FeedbackExpectation::FinalValue {
                channel: Channel::reward(),
            },
        })
    }
    fn commit_issue(&mut self, _: ()) {}
    fn normalize(&self, feedback: bool) -> Result<bool, PolicyError> {
        Ok(feedback)
    }
    fn prepare(&self, _: &(), feedback: &bool) -> Result<bool, PolicyError> {
        Ok(*feedback)
    }
    fn apply(&mut self, update: bool) {
        if update {
            self.applied.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[test]
fn delayed_ticket_updates_its_issuing_epoch_after_replacement() {
    let old_updates = Arc::new(AtomicU64::new(0));
    let new_updates = Arc::new(AtomicU64::new(0));
    let mut muxer = Muxer::new(
        vec!["a".into()],
        EpochPolicy {
            applied: old_updates.clone(),
        },
    )
    .unwrap();
    let receipt = muxer.decide(&()).unwrap();
    muxer
        .replace_policy(EpochPolicy {
            applied: new_updates.clone(),
        })
        .unwrap();
    assert_eq!(receipt.policy_revision(), 0);
    assert_eq!(muxer.decide(&()).unwrap().policy_revision(), 1);
    muxer.tell(receipt.id(), true).unwrap();
    assert_eq!(old_updates.load(Ordering::Relaxed), 1);
    assert_eq!(new_updates.load(Ordering::Relaxed), 0);
}

#[test]
fn terminal_retention_bounds_epoch_lifetime_and_capacity_failure_is_atomic() {
    use muxer::{RuntimeConfig, RuntimeError};
    let old = Arc::new(AtomicU64::new(0));
    let current = Arc::new(AtomicU64::new(0));
    let rejected = Arc::new(AtomicU64::new(0));
    let mut muxer = Muxer::with_config(
        vec!["a".into()],
        EpochPolicy {
            applied: old.clone(),
        },
        RuntimeConfig {
            terminal_capacity: 1,
            retired_epoch_capacity: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let first = muxer.decide(&()).unwrap();
    muxer.tell(first.id(), true).unwrap();
    muxer
        .replace_policy_with_revisions(
            EpochPolicy {
                applied: current.clone(),
            },
            2,
            2,
        )
        .unwrap();
    assert_eq!(muxer.retired_epoch_count(), 1);
    assert_eq!(Arc::strong_count(&old), 2);
    let second = muxer.decide(&()).unwrap();
    assert_eq!(
        (
            second.policy_revision(),
            second.model_revision(),
            second.representation_revision()
        ),
        (1, 2, 2)
    );
    assert_eq!(
        muxer.replace_policy_with_revisions(
            EpochPolicy {
                applied: rejected.clone()
            },
            3,
            3
        ),
        Err(RuntimeError::EpochCapacity)
    );
    muxer.tell(second.id(), true).unwrap();
    assert_eq!(current.load(Ordering::Relaxed), 1);
    assert_eq!(rejected.load(Ordering::Relaxed), 0);
    assert_eq!(muxer.retired_epoch_count(), 0);
    assert_eq!(
        Arc::strong_count(&old),
        1,
        "eviction releases the final epoch owner"
    );
    assert!(muxer.receipt(first.id()).is_none());
    let third = muxer.decide(&()).unwrap();
    assert_eq!(
        (
            third.policy_revision(),
            third.model_revision(),
            third.representation_revision()
        ),
        (1, 2, 2)
    );
}

#[test]
fn explicitly_forgetting_terminal_evidence_releases_old_epoch_only() {
    let applied = Arc::new(AtomicU64::new(0));
    let mut muxer = Muxer::new(
        vec!["a".into()],
        EpochPolicy {
            applied: applied.clone(),
        },
    )
    .unwrap();
    let receipt = muxer.decide(&()).unwrap();
    assert!(muxer.forget_terminal(receipt.id()).is_err());
    muxer.tell(receipt.id(), true).unwrap();
    muxer.replace_policy(EpochPolicy { applied }).unwrap();
    assert_eq!(muxer.retired_epoch_count(), 1);
    muxer.forget_terminal(receipt.id()).unwrap();
    assert_eq!(muxer.retired_epoch_count(), 0);
    assert!(matches!(
        muxer.tell(receipt.id(), true),
        Err(muxer::RuntimeError::UnknownDecision)
    ));
}
