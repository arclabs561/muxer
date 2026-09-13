use muxer::{
    Channel, EventDisposition, FeedbackExpectation, InteractionPolicy, Muxer, PolicyDecision,
    PolicyError, PolicyRequest, ProbabilityAvailability, RuntimeConfig, RuntimeError, TrialRng,
};

#[derive(Clone)]
struct Counter {
    updates: u64,
    fail: bool,
    channels: bool,
}

impl InteractionPolicy for Counter {
    type Context = ();
    type Feedback = (String, u8);
    type CanonicalFeedback = (String, u8);
    type Ticket = String;
    type PreparedIssue = ();
    type PreparedUpdate = u8;
    fn prepare_decision(
        &self,
        request: PolicyRequest<'_, ()>,
        _rng: &mut TrialRng,
    ) -> Result<PolicyDecision<String, ()>, PolicyError> {
        Ok(PolicyDecision {
            selection: request.eligible()[0].clone(),
            probability: ProbabilityAvailability::Unavailable,
            ticket: Some(request.eligible()[0].clone()),
            issue: (),
            expectation: if self.channels {
                FeedbackExpectation::FinalValues(vec![
                    Channel::new("execution").unwrap(),
                    Channel::new("score").unwrap(),
                ])
            } else {
                FeedbackExpectation::FinalValue {
                    channel: Channel::reward(),
                }
            },
        })
    }
    fn commit_issue(&mut self, _: ()) {}
    fn normalize(&self, value: Self::Feedback) -> Result<Self::CanonicalFeedback, PolicyError> {
        if value.1 > 1 {
            Err(PolicyError::new("bad reward"))
        } else {
            Ok(value)
        }
    }
    fn feedback_channel(&self, value: &Self::CanonicalFeedback) -> Channel {
        if self.channels {
            Channel::new(value.0.clone()).unwrap()
        } else {
            Channel::reward()
        }
    }
    fn prepare(&self, _: &String, value: &Self::CanonicalFeedback) -> Result<u8, PolicyError> {
        if self.fail {
            Err(PolicyError::new("rejected"))
        } else {
            Ok(value.1)
        }
    }
    fn apply(&mut self, value: u8) {
        self.updates += u64::from(value);
    }
}

fn mux(policy: Counter) -> Muxer<Counter> {
    Muxer::with_config(
        vec!["a".into(), "b".into()],
        policy,
        RuntimeConfig {
            pending_capacity: 2,
            terminal_capacity: 2,
            event_capacity: 2,
            retired_epoch_capacity: 2,
            seed: 4,
        },
    )
    .unwrap()
}

#[test]
fn canonical_eligibility_identity_and_atomic_update() {
    let mut runtime = mux(Counter {
        updates: 0,
        fail: false,
        channels: false,
    });
    let receipt = runtime.decide_from(&["b".into(), "a".into()], &()).unwrap();
    assert_eq!(receipt.eligible(), &["a", "b"]);
    assert_eq!(
        runtime.tell(receipt.id(), ("ignored".into(), 1)).unwrap(),
        EventDisposition::Accepted
    );
    assert_eq!(runtime.policy().updates, 1);
    assert_eq!(
        runtime.tell(receipt.id(), ("ignored".into(), 1)).unwrap(),
        EventDisposition::Duplicate
    );
    assert_eq!(runtime.policy().updates, 1);
    assert!(matches!(
        runtime.tell(receipt.id(), ("ignored".into(), 0)),
        Err(RuntimeError::AlreadyFinalized)
    ));
    let bad = Muxer::new(
        vec!["a".into()],
        Counter {
            updates: 0,
            fail: true,
            channels: false,
        },
    )
    .unwrap();
    let mut bad = bad;
    let id = bad.decide(&()).unwrap().id();
    assert!(matches!(
        bad.tell(id, ("ignored".into(), 1)),
        Err(RuntimeError::Policy(_))
    ));
    assert_eq!(bad.policy().updates, 0);
    assert_eq!(bad.pending_len(), 1);
}

#[test]
fn multi_channel_retains_ticket_and_wrong_engine_cannot_update() {
    let mut left = mux(Counter {
        updates: 0,
        fail: false,
        channels: true,
    });
    let mut right = mux(Counter {
        updates: 0,
        fail: false,
        channels: true,
    });
    let id = left.decide(&()).unwrap().id();
    assert!(matches!(
        right.tell(id, ("execution".into(), 1)),
        Err(RuntimeError::WrongEngine)
    ));
    left.tell(id, ("execution".into(), 1)).unwrap();
    assert_eq!(left.pending_len(), 1);
    left.tell(id, ("score".into(), 1)).unwrap();
    assert_eq!(left.pending_len(), 0);
    assert_eq!(left.policy().updates, 2);
    assert_eq!(
        left.final_feedback(id, 0, &Channel::new("score").unwrap()),
        Some(&(String::from("score"), 1))
    );
}

#[test]
fn pending_capacity_does_not_consume_a_sequence() {
    let mut runtime = Muxer::with_config(
        vec!["a".into()],
        Counter {
            updates: 0,
            fail: false,
            channels: false,
        },
        RuntimeConfig {
            pending_capacity: 1,
            terminal_capacity: 2,
            event_capacity: 2,
            retired_epoch_capacity: 2,
            seed: 0,
        },
    )
    .unwrap();
    let first = runtime.decide(&()).unwrap();
    assert!(matches!(
        runtime.decide(&()),
        Err(RuntimeError::PendingCapacity)
    ));
    runtime.tell(first.id(), ("x".into(), 1)).unwrap();
    assert_eq!(
        runtime.decide(&()).unwrap().sequence(),
        first.sequence() + 1
    );
}

#[test]
fn batch_capacity_failure_preserves_sequence_rng_and_prepared_issue() {
    use muxer::{QualityProfile, RouterConfig};
    let actions = vec!["a".into(), "b".into(), "c".into(), "d".into()];
    let make = || {
        Muxer::with_config(
            actions.clone(),
            QualityProfile::new(actions.clone(), RouterConfig::default().with_control(2)).unwrap(),
            RuntimeConfig {
                pending_capacity: 1,
                event_capacity: 4,
                seed: 85,
                ..Default::default()
            },
        )
        .unwrap()
    };
    let mut tested = make();
    let mut baseline = make();
    for round in 0..10 {
        let left = tested.decide_batch(2, &[]).unwrap();
        let right = baseline.decide_batch(2, &[]).unwrap();
        assert_eq!(
            left.selected().collect::<Vec<_>>(),
            right.selected().collect::<Vec<_>>()
        );
        assert_eq!(left.sequence(), right.sequence());
        assert_eq!(left.sequence(), round + 1);
        assert!(matches!(
            tested.decide_batch(2, &[]),
            Err(RuntimeError::PendingCapacity)
        ));
        tested.expire(left.id()).unwrap();
        baseline.expire(right.id()).unwrap();
    }
}

#[test]
fn batch_admission_reserves_enough_event_capacity_for_all_items() {
    use muxer::{QualityProfile, RouterConfig};
    let actions = vec!["a".into(), "b".into()];
    let profile = QualityProfile::new(actions.clone(), RouterConfig::default())
        .unwrap()
        .with_delayed_score();
    let mut runtime = Muxer::with_config(
        actions,
        profile,
        RuntimeConfig {
            event_capacity: 3,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(matches!(
        runtime.decide_batch(2, &[]),
        Err(RuntimeError::FeedbackCapacity)
    ));
    assert_eq!(runtime.pending_len(), 0);
    assert_eq!(runtime.decide(&[]).unwrap().sequence(), 1);
}

#[test]
fn missing_reason_is_retained_and_conflicts_are_rejected() {
    let mut runtime = mux(Counter {
        updates: 0,
        fail: false,
        channels: true,
    });
    let receipt = runtime.decide(&()).unwrap();
    let channel = Channel::new("execution").unwrap();
    assert_eq!(
        runtime
            .tell_missing(receipt.id(), 0, channel.clone(), "censored")
            .unwrap(),
        EventDisposition::Accepted
    );
    assert_eq!(
        runtime.missing_reason(receipt.id(), 0, &channel),
        Some("censored")
    );
    assert_eq!(
        runtime
            .tell_missing(receipt.id(), 0, channel, "censored")
            .unwrap(),
        EventDisposition::Duplicate
    );
    assert!(matches!(
        runtime.tell_missing(
            receipt.id(),
            0,
            Channel::new("execution").unwrap(),
            "different"
        ),
        Err(RuntimeError::ConflictingEvent)
    ));
}

#[test]
fn expiry_rejects_fresh_feedback() {
    let mut runtime = mux(Counter {
        updates: 0,
        fail: false,
        channels: false,
    });
    let receipt = runtime.decide(&()).unwrap();
    runtime.expire(receipt.id()).unwrap();
    assert!(matches!(
        runtime.tell(receipt.id(), ("ignored".into(), 1)),
        Err(RuntimeError::Expired)
    ));
}
