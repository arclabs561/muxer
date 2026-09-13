use muxer::{
    Disposition, EventId, EventOutcome, FeedbackEvent, Muxer, Outcome, QualityFeedback,
    QualityProfile, RouterConfig,
};

fn runtime() -> Muxer<QualityProfile> {
    Muxer::quality(["a", "b"])
        .config(RouterConfig::default().with_monitoring(80, 40))
        .with_delayed_score()
        .build()
        .unwrap()
}

#[test]
fn checkpoint_continues_pending_channels_deduplication_rng_and_windows() {
    let mut baseline = runtime();
    let mut restarting = runtime();
    for _ in 0..10 {
        let left = baseline.decide(&[]).unwrap();
        let right = restarting.decide(&[]).unwrap();
        assert_eq!(left.action(), right.action());
        for (mux, receipt) in [(&mut baseline, left), (&mut restarting, right)] {
            mux.tell(
                receipt.id(),
                QualityFeedback::execution(Outcome::success(2, 10)),
            )
            .unwrap();
            mux.tell(receipt.id(), QualityFeedback::score(0.3).unwrap())
                .unwrap();
        }
    }
    let left = baseline.decide(&[]).unwrap();
    let right = restarting.decide(&[]).unwrap();
    let provisional = |receipt: &muxer::DecisionReceipt| {
        FeedbackEvent::new(
            receipt.execution_key(),
            receipt.action(),
            EventId::new("provisional").unwrap(),
            QualityProfile::score_channel(),
            4,
            1,
            Disposition::Provisional(QualityFeedback::score(0.8).unwrap()),
        )
    };
    assert_eq!(
        baseline.submit(provisional(&left)).unwrap(),
        EventOutcome::Provisional
    );
    assert_eq!(
        restarting.submit(provisional(&right)).unwrap(),
        EventOutcome::Provisional
    );
    let engine = restarting.engine_id();
    let checkpoint = restarting.into_checkpoint();
    let mut restored = Muxer::restore(checkpoint);
    assert_eq!(restored.engine_id(), engine);
    assert_eq!(restored.pending_len(), 1);
    assert_eq!(
        restored.submit(provisional(&right)).unwrap(),
        EventOutcome::Duplicate
    );
    assert_eq!(
        restored.received_sequence(right.id(), &EventId::new("provisional").unwrap()),
        Some(1)
    );

    for (mux, receipt) in [(&mut baseline, left), (&mut restored, right)] {
        mux.tell(
            receipt.id(),
            QualityFeedback::execution(Outcome::success(2, 10)),
        )
        .unwrap();
        mux.submit(FeedbackEvent::new(
            receipt.execution_key(),
            receipt.action(),
            EventId::new("final").unwrap(),
            QualityProfile::score_channel(),
            4,
            1,
            Disposition::Final(QualityFeedback::score(0.8).unwrap()),
        ))
        .unwrap();
    }
    assert_eq!(restored.pending_len(), 0);
    for _ in 0..20 {
        let left = baseline.decide(&[]).unwrap();
        let right = restored.decide(&[]).unwrap();
        assert_eq!(left.action(), right.action());
        assert_eq!(left.sequence(), right.sequence());
        for (mux, receipt) in [(&mut baseline, left), (&mut restored, right)] {
            mux.tell(
                receipt.id(),
                QualityFeedback::execution(Outcome::failure(4, 30)),
            )
            .unwrap();
            mux.tell(receipt.id(), QualityFeedback::score(0.1).unwrap())
                .unwrap();
        }
    }
    for action in ["a", "b"] {
        let left = baseline.policy().router().summary(action);
        let right = restored.policy().router().summary(action);
        assert_eq!(
            (left.calls, left.ok, left.mean_quality_score),
            (right.calls, right.ok, right.mean_quality_score)
        );
        assert_eq!(
            baseline
                .policy()
                .router()
                .monitored_window(action)
                .unwrap()
                .recent()
                .summary()
                .calls,
            restored
                .policy()
                .router()
                .monitored_window(action)
                .unwrap()
                .recent()
                .summary()
                .calls
        );
    }
}
