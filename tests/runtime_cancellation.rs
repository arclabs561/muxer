use muxer::{
    DecisionReceipt, Disposition, EventId, EventOutcome, ExecutionKey, FeedbackEvent, ItemStatus,
    Muxer, Outcome, QualityFeedback, QualityProfile, RuntimeError, TerminalStatus,
};

fn runtime() -> Muxer<QualityProfile> {
    Muxer::quality(["a", "b", "c"])
        .with_delayed_score()
        .build()
        .unwrap()
}

fn score_event(
    receipt: &DecisionReceipt,
    position: usize,
    event_id: &str,
    disposition: Disposition<QualityFeedback>,
) -> FeedbackEvent<QualityFeedback> {
    FeedbackEvent::new(
        ExecutionKey::new(receipt.id(), position),
        receipt.selected().nth(position).unwrap(),
        EventId::new(event_id).unwrap(),
        QualityProfile::score_channel(),
        1,
        1,
        disposition,
    )
}

fn complete(mux: &mut Muxer<QualityProfile>, receipt: &DecisionReceipt, position: usize) {
    mux.tell_item(
        receipt.id(),
        position,
        QualityFeedback::execution(Outcome::success(2, 10)),
    )
    .unwrap();
    mux.tell_item(receipt.id(), position, QualityFeedback::score(0.7).unwrap())
        .unwrap();
}

#[test]
fn cancellation_survives_handoff_and_sibling_completion_without_observations() {
    let mut mux = runtime();
    let receipt = mux.decide_batch(2, &[]).unwrap();
    mux.cancel_item(receipt.id(), 1).unwrap();
    mux.cancel_item(receipt.id(), 1).unwrap();
    assert_eq!(mux.pending_len(), 1);
    assert_eq!(mux.item_status(receipt.id(), 0), Some(ItemStatus::Open));
    assert_eq!(
        mux.item_status(receipt.id(), 1),
        Some(ItemStatus::Cancelled)
    );
    let mut mux = Muxer::restore(mux.into_checkpoint());
    assert_eq!(
        mux.item_status(receipt.id(), 1),
        Some(ItemStatus::Cancelled)
    );
    complete(&mut mux, &receipt, 0);
    assert_eq!(mux.pending_len(), 0);
    assert_eq!(
        mux.terminal_status(receipt.id()),
        Some(TerminalStatus::Completed)
    );
    assert_eq!(
        mux.item_status(receipt.id(), 0),
        Some(ItemStatus::Completed)
    );
    assert_eq!(
        mux.item_status(receipt.id(), 1),
        Some(ItemStatus::Cancelled)
    );
    mux.cancel_item(receipt.id(), 1).unwrap();
    assert_eq!(mux.policy().router().total_observations(), 1);
    assert!(mux
        .final_feedback(receipt.id(), 1, &QualityProfile::score_channel())
        .is_none());
    assert_eq!(
        mux.tell_item(receipt.id(), 1, QualityFeedback::score(0.7).unwrap()),
        Err(RuntimeError::Cancelled)
    );
    assert_eq!(
        mux.tell_missing(receipt.id(), 1, QualityProfile::score_channel(), "late"),
        Err(RuntimeError::Cancelled)
    );
}

#[test]
fn cancellation_rejects_provisional_and_final_values_without_losing_evidence() {
    let mut mux = runtime();
    let receipt = mux.decide_batch(2, &[]).unwrap();
    let event = || {
        score_event(
            &receipt,
            0,
            "provisional",
            Disposition::Provisional(QualityFeedback::score(0.4).unwrap()),
        )
    };
    assert_eq!(mux.submit(event()).unwrap(), EventOutcome::Provisional);
    assert_eq!(
        mux.cancel_item(receipt.id(), 0),
        Err(RuntimeError::AlreadyFinalized)
    );
    assert_eq!(mux.submit(event()).unwrap(), EventOutcome::Duplicate);
    assert!(mux
        .provisional_feedback(receipt.id(), 0, &QualityProfile::score_channel())
        .is_some());
    assert_eq!(mux.policy().router().total_observations(), 0);
    assert_eq!(mux.item_status(receipt.id(), 0), Some(ItemStatus::Open));
    mux.tell_item(
        receipt.id(),
        1,
        QualityFeedback::execution(Outcome::success(2, 10)),
    )
    .unwrap();
    assert_eq!(
        mux.cancel_item(receipt.id(), 1),
        Err(RuntimeError::AlreadyFinalized)
    );
    assert_eq!(mux.policy().router().total_observations(), 1);
    mux.tell_item(receipt.id(), 1, QualityFeedback::score(0.7).unwrap())
        .unwrap();
    mux.tell_item(receipt.id(), 0, QualityFeedback::score(0.4).unwrap())
        .unwrap();
    mux.tell_item(
        receipt.id(),
        0,
        QualityFeedback::execution(Outcome::success(2, 10)),
    )
    .unwrap();
    assert_eq!(mux.pending_len(), 0);
    assert_eq!(mux.policy().router().total_observations(), 2);
}

#[test]
fn missing_only_cancellation_preserves_id_deduplication_and_conflict_precedence() {
    let mut mux = runtime();
    let receipt = mux.decide_batch(2, &[]).unwrap();
    let missing = || {
        score_event(
            &receipt,
            1,
            "missing-score",
            Disposition::Missing("unavailable".into()),
        )
    };
    assert_eq!(mux.submit(missing()).unwrap(), EventOutcome::Missing);
    mux.cancel_item(receipt.id(), 1).unwrap();
    assert_eq!(mux.submit(missing()).unwrap(), EventOutcome::Duplicate);
    assert_eq!(
        mux.received_sequence(receipt.id(), &EventId::new("missing-score").unwrap()),
        Some(1)
    );
    assert_eq!(
        mux.submit(score_event(
            &receipt,
            1,
            "missing-score",
            Disposition::Missing("changed".into())
        )),
        Err(RuntimeError::ConflictingEvent)
    );
    assert_eq!(
        mux.submit(score_event(
            &receipt,
            1,
            "fresh",
            Disposition::Final(QualityFeedback::score(0.8).unwrap())
        )),
        Err(RuntimeError::Cancelled)
    );
    complete(&mut mux, &receipt, 0);
    let mut mux = Muxer::restore(mux.into_checkpoint());
    assert_eq!(mux.submit(missing()).unwrap(), EventOutcome::Duplicate);
    assert_eq!(
        mux.missing_reason(receipt.id(), 1, &QualityProfile::score_channel()),
        Some("unavailable")
    );
    assert_eq!(mux.policy().router().total_observations(), 1);
}

#[test]
fn expiry_preserves_completed_and_cancelled_siblings() {
    let mut mux = runtime();
    let receipt = mux.decide_batch(3, &[]).unwrap();
    complete(&mut mux, &receipt, 0);
    mux.cancel_item(receipt.id(), 1).unwrap();
    mux.expire(receipt.id()).unwrap();
    assert_eq!(
        mux.terminal_status(receipt.id()),
        Some(TerminalStatus::Expired)
    );
    assert_eq!(
        mux.item_status(receipt.id(), 0),
        Some(ItemStatus::Completed)
    );
    assert_eq!(
        mux.item_status(receipt.id(), 1),
        Some(ItemStatus::Cancelled)
    );
    assert_eq!(mux.item_status(receipt.id(), 2), Some(ItemStatus::Expired));
    assert_eq!(mux.policy().router().total_observations(), 1);
    assert_eq!(mux.policy().awaiting_score_len(), 0);
    assert_eq!(mux.policy().buffered_score_len(), 0);
}

#[test]
fn invalid_item_or_engine_does_not_close_a_live_batch() {
    let mut mux = runtime();
    let receipt = mux.decide_batch(2, &[]).unwrap();
    let mut other = runtime();
    assert_eq!(
        other.cancel_item(receipt.id(), 0),
        Err(RuntimeError::WrongEngine)
    );
    assert_eq!(
        mux.cancel_item(receipt.id(), 2),
        Err(RuntimeError::WrongPosition)
    );
    assert_eq!(mux.pending_len(), 1);
    complete(&mut mux, &receipt, 0);
    assert_eq!(
        mux.cancel_item(receipt.id(), 0),
        Err(RuntimeError::AlreadyFinalized)
    );
    complete(&mut mux, &receipt, 1);
    assert_eq!(mux.policy().router().total_observations(), 2);
}

#[test]
fn item_and_whole_cancellation_release_only_the_original_epoch_once() {
    use muxer::{
        Channel, FeedbackExpectation, InteractionPolicy, PolicyDecision, PolicyError,
        PolicyRequest, ProbabilityAvailability, TrialRng,
    };
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    struct CleanupPolicy(Arc<AtomicUsize>);
    impl InteractionPolicy for CleanupPolicy {
        type Context = ();
        type Feedback = bool;
        type CanonicalFeedback = bool;
        type Ticket = ();
        type PreparedIssue = ();
        type PreparedUpdate = ();
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
        fn prepare(&self, _: &(), _: &bool) -> Result<(), PolicyError> {
            Ok(())
        }
        fn apply(&mut self, _: ()) {
            panic!("cancellation must not train")
        }
        fn finish_ticket(&mut self, _: &()) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    let old = Arc::new(AtomicUsize::new(0));
    let new = Arc::new(AtomicUsize::new(0));
    let mut mux = Muxer::new(vec!["a".into()], CleanupPolicy(old.clone())).unwrap();
    let item = mux.decide(&()).unwrap();
    let whole = mux.decide(&()).unwrap();
    mux.replace_policy(CleanupPolicy(new.clone())).unwrap();
    mux.cancel_item(item.id(), 0).unwrap();
    mux.cancel_item(item.id(), 0).unwrap();
    assert_eq!(old.load(Ordering::SeqCst), 1);
    mux.cancel(whole.id()).unwrap();
    assert_eq!(old.load(Ordering::SeqCst), 2);
    assert_eq!(new.load(Ordering::SeqCst), 0);
    assert_eq!(mux.pending_len(), 0);
}
