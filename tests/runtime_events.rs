use muxer::{
    Channel, DecisionReceipt, Disposition, EventId, EventOutcome, FeedbackEvent,
    FeedbackExpectation, InteractionPolicy, Muxer, PolicyDecision, PolicyError, PolicyRequest,
    ProbabilityAvailability, RuntimeConfig, RuntimeError, TrialRng,
};

#[derive(Default)]
struct Policy {
    updates: usize,
    missing: usize,
}

impl InteractionPolicy for Policy {
    type Context = ();
    type Feedback = u8;
    type CanonicalFeedback = u8;
    type Ticket = ();
    type PreparedIssue = ();
    type PreparedUpdate = ();

    fn prepare_decision(
        &self,
        request: PolicyRequest<'_, Self::Context>,
        _: &mut TrialRng,
    ) -> Result<PolicyDecision<Self::Ticket, Self::PreparedIssue>, PolicyError> {
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
    fn normalize(&self, feedback: Self::Feedback) -> Result<Self::CanonicalFeedback, PolicyError> {
        Ok(feedback)
    }
    fn prepare(
        &self,
        _: &Self::Ticket,
        _: &Self::CanonicalFeedback,
    ) -> Result<Self::PreparedUpdate, PolicyError> {
        Ok(())
    }
    fn apply(&mut self, _: Self::PreparedUpdate) {
        self.updates += 1;
    }
    fn missing_channel(&mut self, _: &Self::Ticket, _: &Channel) {
        self.missing += 1;
    }
}

fn reward_event(
    receipt: &DecisionReceipt,
    event_id: &str,
    source_revision: u64,
    revision: u64,
    disposition: Disposition<u8>,
) -> FeedbackEvent<u8> {
    FeedbackEvent::new(
        receipt.execution_key(),
        "candidate-a",
        EventId::new(event_id).unwrap(),
        Channel::reward(),
        source_revision,
        revision,
        disposition,
    )
}

fn event(
    receipt: &DecisionReceipt,
    action: &str,
    id: &str,
    disposition: Disposition<u8>,
) -> FeedbackEvent<u8> {
    FeedbackEvent::new(
        receipt.execution_key(),
        action,
        EventId::new(id).unwrap(),
        Channel::new("quality/v1").unwrap(),
        3,
        5,
        disposition,
    )
}

#[test]
fn feedback_event_preserves_correlation_and_stream_identity() {
    let mut muxer = Muxer::new(vec!["candidate-a".into()], Policy::default()).unwrap();
    let receipt = muxer.decide(&()).unwrap();
    let event = event(
        &receipt,
        "candidate-a",
        "assessment-101",
        Disposition::Provisional(9),
    )
    .with_observed_at(1_725_000_000);

    assert_eq!(event.execution().decision().sequence(), 1);
    assert_eq!(event.execution().position(), 0);
    assert_eq!(event.action(), "candidate-a");
    assert_eq!(event.event_id().as_str(), "assessment-101");
    assert_eq!(event.channel().as_str(), "quality/v1");
    assert_eq!(event.source_revision(), 3);
    assert_eq!(event.revision(), 5);
    assert_eq!(event.observed_at(), Some(1_725_000_000));
    assert_eq!(event.disposition(), &Disposition::Provisional(9));
}

#[test]
fn feedback_event_keeps_missing_distinct_from_payload_values() {
    let mut muxer = Muxer::new(vec!["candidate-a".into()], Policy::default()).unwrap();
    let receipt = muxer.decide(&()).unwrap();
    let event = event(
        &receipt,
        "candidate-a",
        "assessment-102",
        Disposition::Missing("label unavailable".into()),
    );

    assert_eq!(
        event.into_disposition(),
        Disposition::Missing("label unavailable".into())
    );
}

#[test]
fn submit_rejects_an_action_that_does_not_match_its_receipt() {
    let mut muxer = Muxer::new(vec!["candidate-a".into()], Policy::default()).unwrap();
    let receipt = muxer.decide(&()).unwrap();
    let event = event(
        &receipt,
        "candidate-b",
        "assessment-103",
        Disposition::Final(1),
    );

    assert!(matches!(
        muxer.submit(event),
        Err(RuntimeError::WrongAction)
    ));
    assert_eq!(muxer.pending_len(), 1);
}

#[test]
fn submit_missing_closes_a_channel_without_manufacturing_feedback() {
    let mut muxer = Muxer::new(vec!["candidate-a".into()], Policy::default()).unwrap();
    let receipt = muxer.decide(&()).unwrap();
    let event = FeedbackEvent::new(
        receipt.execution_key(),
        "candidate-a",
        EventId::new("assessment-104").unwrap(),
        Channel::reward(),
        3,
        5,
        Disposition::Missing("no label".into()),
    );

    assert_eq!(muxer.submit(event.clone()), Ok(EventOutcome::Missing));
    assert_eq!(muxer.submit(event), Ok(EventOutcome::Duplicate));
    assert_eq!(muxer.pending_len(), 0);
    assert_eq!(
        muxer.final_feedback(receipt.id(), 0, &Channel::reward()),
        None
    );
    assert_eq!(muxer.policy().missing, 1);
}

#[test]
fn submit_deduplicates_ids_and_rejects_conflicting_reuse() {
    let mut muxer = Muxer::new(vec!["candidate-a".into()], Policy::default()).unwrap();
    let receipt = muxer.decide(&()).unwrap();

    let provisional = reward_event(&receipt, "event-1", 4, 0, Disposition::Provisional(4));
    assert_eq!(
        muxer.submit(provisional.clone()),
        Ok(EventOutcome::Provisional)
    );
    let received = muxer
        .received_sequence(receipt.id(), &EventId::new("event-1").unwrap())
        .unwrap();
    assert_eq!(muxer.submit(provisional), Ok(EventOutcome::Duplicate));
    assert_eq!(
        muxer.received_sequence(receipt.id(), &EventId::new("event-1").unwrap()),
        Some(received)
    );
    assert!(matches!(
        muxer.submit(reward_event(
            &receipt,
            "event-2",
            4,
            0,
            Disposition::Provisional(7)
        )),
        Err(RuntimeError::ConflictingEvent)
    ));
    assert_eq!(
        muxer.received_sequence(receipt.id(), &EventId::new("event-2").unwrap()),
        None
    );
    assert!(matches!(
        muxer.submit(reward_event(
            &receipt,
            "event-1",
            4,
            0,
            Disposition::Final(4)
        )),
        Err(RuntimeError::ConflictingEvent)
    ));
    assert_eq!(muxer.policy().updates, 0);
}

#[test]
fn submit_orders_provisionals_and_finalizes_once() {
    let mut muxer = Muxer::new(vec!["candidate-a".into()], Policy::default()).unwrap();
    let receipt = muxer.decide(&()).unwrap();

    assert_eq!(
        muxer.submit(reward_event(
            &receipt,
            "event-1",
            4,
            1,
            Disposition::Provisional(3)
        )),
        Ok(EventOutcome::Provisional)
    );
    let first_received = muxer
        .received_sequence(receipt.id(), &EventId::new("event-1").unwrap())
        .unwrap();
    assert_eq!(
        muxer.provisional_feedback(receipt.id(), 0, &Channel::reward()),
        Some(&3)
    );
    assert!(matches!(
        muxer.submit(reward_event(
            &receipt,
            "event-2",
            4,
            0,
            Disposition::Provisional(2)
        )),
        Err(RuntimeError::StaleRevision)
    ));
    assert_eq!(
        muxer.received_sequence(receipt.id(), &EventId::new("event-2").unwrap()),
        None
    );
    assert_eq!(
        muxer.submit(reward_event(
            &receipt,
            "event-3",
            4,
            2,
            Disposition::Provisional(5)
        )),
        Ok(EventOutcome::Provisional)
    );
    assert!(
        muxer
            .received_sequence(receipt.id(), &EventId::new("event-3").unwrap())
            .unwrap()
            > first_received
    );
    assert_eq!(
        muxer.provisional_feedback(receipt.id(), 0, &Channel::reward()),
        Some(&5)
    );
    assert_eq!(
        muxer.submit(reward_event(
            &receipt,
            "event-4",
            4,
            2,
            Disposition::Final(5)
        )),
        Ok(EventOutcome::Accepted)
    );
    assert_eq!(muxer.policy().updates, 1);
    assert_eq!(
        muxer.provisional_feedback(receipt.id(), 0, &Channel::reward()),
        None
    );
    assert_eq!(
        muxer.final_feedback(receipt.id(), 0, &Channel::reward()),
        Some(&5)
    );
    assert_eq!(
        muxer.submit(reward_event(
            &receipt,
            "event-5",
            4,
            2,
            Disposition::Final(5)
        )),
        Ok(EventOutcome::Duplicate)
    );
    assert!(matches!(
        muxer.submit(reward_event(
            &receipt,
            "event-6",
            4,
            3,
            Disposition::Final(8)
        )),
        Err(RuntimeError::AlreadyFinalized)
    ));
    assert_eq!(muxer.policy().updates, 1);
}

#[test]
fn provisional_revisions_consume_bounded_event_retention() {
    let config = RuntimeConfig {
        pending_capacity: 1,
        terminal_capacity: 1,
        event_capacity: 2,
        seed: 0,
        ..RuntimeConfig::default()
    };
    let mut muxer =
        Muxer::with_config(vec!["candidate-a".into()], Policy::default(), config).unwrap();
    let receipt = muxer.decide(&()).unwrap();

    for (id, revision) in [("event-1", 0), ("event-2", 1)] {
        assert_eq!(
            muxer.submit(reward_event(
                &receipt,
                id,
                4,
                revision,
                Disposition::Provisional(1)
            )),
            Ok(EventOutcome::Provisional)
        );
    }
    assert!(matches!(
        muxer.submit(reward_event(
            &receipt,
            "event-3",
            4,
            2,
            Disposition::Provisional(1)
        )),
        Err(RuntimeError::FeedbackCapacity)
    ));
    assert_eq!(muxer.pending_len(), 1);
    assert_eq!(muxer.policy().updates, 0);
}

#[test]
fn source_revision_cannot_silently_replace_a_channel_producer() {
    let mut muxer = Muxer::new(vec!["candidate-a".into()], Policy::default()).unwrap();
    let receipt = muxer.decide(&()).unwrap();

    assert_eq!(
        muxer.submit(reward_event(
            &receipt,
            "event-1",
            4,
            0,
            Disposition::Provisional(1)
        )),
        Ok(EventOutcome::Provisional)
    );
    assert!(matches!(
        muxer.submit(reward_event(
            &receipt,
            "event-2",
            5,
            0,
            Disposition::Provisional(1)
        )),
        Err(RuntimeError::SourceRevisionMismatch)
    ));
}

#[test]
fn event_id_cannot_be_reused_by_a_different_retained_decision() {
    let mut muxer = Muxer::new(vec!["candidate-a".into()], Policy::default()).unwrap();
    let first = muxer.decide(&()).unwrap();
    let second = muxer.decide(&()).unwrap();

    assert_eq!(
        muxer.submit(reward_event(
            &first,
            "global-event",
            1,
            0,
            Disposition::Provisional(1)
        )),
        Ok(EventOutcome::Provisional)
    );
    assert!(matches!(
        muxer.submit(reward_event(
            &second,
            "global-event",
            1,
            0,
            Disposition::Provisional(1)
        )),
        Err(RuntimeError::ConflictingEvent)
    ));
    assert_eq!(muxer.pending_len(), 2);
}

#[test]
fn convenience_final_feedback_closes_an_advanced_provisional_value() {
    let mut muxer = Muxer::new(vec!["candidate-a".into()], Policy::default()).unwrap();
    let receipt = muxer.decide(&()).unwrap();
    assert_eq!(
        muxer.submit(reward_event(
            &receipt,
            "event-1",
            1,
            0,
            Disposition::Provisional(2)
        )),
        Ok(EventOutcome::Provisional)
    );

    assert_eq!(
        muxer.tell(receipt.id(), 7),
        Ok(muxer::EventDisposition::Accepted)
    );
    assert_eq!(
        muxer.provisional_feedback(receipt.id(), 0, &Channel::reward()),
        None
    );
    assert_eq!(
        muxer.final_feedback(receipt.id(), 0, &Channel::reward()),
        Some(&7)
    );
    assert_eq!(muxer.policy().updates, 1);
}

#[test]
fn advanced_final_cannot_reopen_a_channel_closed_missing_by_convenience_api() {
    let mut muxer = Muxer::new(vec!["candidate-a".into()], Policy::default()).unwrap();
    let receipt = muxer.decide(&()).unwrap();
    assert_eq!(
        muxer.tell_missing(receipt.id(), 0, Channel::reward(), "label unavailable"),
        Ok(muxer::EventDisposition::Accepted)
    );

    assert!(matches!(
        muxer.submit(reward_event(
            &receipt,
            "event-1",
            1,
            0,
            Disposition::Final(2)
        )),
        Err(RuntimeError::AlreadyFinalized)
    ));
    assert_eq!(muxer.policy().updates, 0);
    assert_eq!(muxer.policy().missing, 1);
}

#[test]
fn convenience_apis_cannot_bypass_advanced_event_capacity() {
    let config = RuntimeConfig {
        pending_capacity: 2,
        terminal_capacity: 2,
        event_capacity: 1,
        seed: 0,
        ..RuntimeConfig::default()
    };
    let mut muxer =
        Muxer::with_config(vec!["candidate-a".into()], Policy::default(), config).unwrap();
    let first = muxer.decide(&()).unwrap();
    let second = muxer.decide(&()).unwrap();
    for (receipt, event_id) in [(&first, "event-1"), (&second, "event-2")] {
        assert_eq!(
            muxer.submit(reward_event(
                receipt,
                event_id,
                1,
                0,
                Disposition::Provisional(1)
            )),
            Ok(EventOutcome::Provisional)
        );
    }

    assert!(matches!(
        muxer.tell(first.id(), 1),
        Err(RuntimeError::FeedbackCapacity)
    ));
    assert!(matches!(
        muxer.tell_missing(second.id(), 0, Channel::reward(), "unavailable"),
        Err(RuntimeError::FeedbackCapacity)
    ));
    assert_eq!(muxer.policy().updates, 0);
    assert_eq!(muxer.policy().missing, 0);
    assert_eq!(muxer.pending_len(), 2);
}

#[test]
fn cancellation_after_an_advanced_provisional_is_rejected() {
    let mut muxer = Muxer::new(vec!["candidate-a".into()], Policy::default()).unwrap();
    let receipt = muxer.decide(&()).unwrap();
    assert_eq!(
        muxer.submit(reward_event(
            &receipt,
            "event-1",
            1,
            0,
            Disposition::Provisional(1)
        )),
        Ok(EventOutcome::Provisional)
    );

    assert!(matches!(
        muxer.cancel(receipt.id()),
        Err(RuntimeError::AlreadyFinalized)
    ));
    assert_eq!(muxer.pending_len(), 1);
}
