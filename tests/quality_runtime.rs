use muxer::{
    context_bin, DecisionReason, EventDisposition, Muxer, Outcome, QualityFeedback, QualityProfile,
    RouterConfig, RuntimeError, TriageSessionConfig,
};

fn arms() -> Vec<String> {
    vec!["a".to_owned(), "b".to_owned()]
}

fn outcome() -> Outcome {
    Outcome::success(3, 20)
}

#[test]
fn quality_respects_authoritative_candidate_filtering() {
    let profile = QualityProfile::new(arms(), RouterConfig::default()).unwrap();
    let mut muxer = Muxer::new(arms(), profile).unwrap();
    let receipt = muxer.decide_from(&["b".to_owned()], &[]).unwrap();

    assert_eq!(receipt.action(), "b");
    assert_eq!(receipt.eligible(), &["b"]);
    assert_eq!(
        muxer.tell(receipt.id(), QualityFeedback::execution(outcome())),
        Ok(EventDisposition::Accepted)
    );
    assert_eq!(muxer.policy().router().summary("b").calls, 1);
    assert_eq!(muxer.policy().router().summary("a").calls, 0);
}

#[test]
fn quality_receipt_reports_control_stage_reason() {
    let mut muxer = Muxer::quality(["a", "b"])
        .config(RouterConfig::default().with_control(1))
        .build()
        .unwrap();
    let receipt = muxer.decide_batch(2, &[]).unwrap();

    assert_eq!(receipt.reason(), DecisionReason::Control);
}

#[test]
fn immediate_execution_matches_direct_router_observation() {
    let config = RouterConfig::default().with_monitoring(500, 50);
    let mut direct = muxer::Router::new(vec!["a".to_owned()], config.clone()).unwrap();
    let scored = Outcome::with_quality(true, false, false, 3, 20, 0.73);
    assert!(direct.observe("a", scored));

    let profile = QualityProfile::new(vec!["a".to_owned()], config).unwrap();
    let mut muxer = Muxer::new(vec!["a".to_owned()], profile).unwrap();
    let receipt = muxer.decide(&[]).unwrap();
    muxer
        .tell(receipt.id(), QualityFeedback::execution(scored))
        .unwrap();

    let expected = direct.summary("a");
    let actual = muxer.policy().router().summary("a");
    assert_eq!(actual.calls, expected.calls);
    assert_eq!(actual.ok, expected.ok);
    assert_eq!(actual.junk, expected.junk);
    assert_eq!(actual.hard_junk, expected.hard_junk);
    assert_eq!(actual.cost_units, expected.cost_units);
    assert_eq!(actual.elapsed_ms_sum, expected.elapsed_ms_sum);
    assert_eq!(actual.mean_quality_score, expected.mean_quality_score);
    assert_eq!(
        muxer
            .policy()
            .router()
            .monitored_window("a")
            .unwrap()
            .recent()
            .summary()
            .calls,
        direct
            .monitored_window("a")
            .unwrap()
            .recent()
            .summary()
            .calls,
    );
}

#[test]
fn delayed_score_rejects_embedded_execution_score() {
    let profile = QualityProfile::new(arms(), RouterConfig::default())
        .unwrap()
        .with_delayed_score();
    let mut muxer = Muxer::new(arms(), profile).unwrap();
    let receipt = muxer.decide(&[]).unwrap();

    let embedded = Outcome::with_quality(true, false, false, 1, 1, 0.8);
    assert!(matches!(
        muxer.tell(receipt.id(), QualityFeedback::execution(embedded)),
        Err(RuntimeError::Policy(_))
    ));
    assert_eq!(muxer.pending_len(), 1);
    assert_eq!(muxer.policy().router().summary(receipt.action()).calls, 0);
}

#[test]
fn delayed_score_keeps_ticket_and_joins_execution_row() {
    let profile = QualityProfile::new(arms(), RouterConfig::default())
        .unwrap()
        .with_delayed_score();
    let mut muxer = Muxer::new(arms(), profile).unwrap();
    let receipt = muxer.decide_from(&["a".to_owned()], &[]).unwrap();

    assert_eq!(
        muxer.tell(receipt.id(), QualityFeedback::execution(outcome())),
        Ok(EventDisposition::Accepted)
    );
    assert_eq!(muxer.pending_len(), 1);
    assert_eq!(muxer.policy().router().summary("a").calls, 1);
    assert_eq!(
        muxer.tell(receipt.id(), QualityFeedback::score(0.91).unwrap()),
        Ok(EventDisposition::Accepted)
    );
    assert_eq!(muxer.pending_len(), 0);
    assert_eq!(muxer.policy().router().summary("a").calls, 1);
    assert_eq!(muxer.policy().router().mean_quality_score("a"), Some(0.91));
}

#[test]
fn score_before_execution_is_buffered_and_joined_once() {
    let profile = QualityProfile::new(arms(), RouterConfig::default())
        .unwrap()
        .with_delayed_score();
    let mut muxer = Muxer::new(arms(), profile).unwrap();
    let receipt = muxer.decide_from(&["a".to_owned()], &[]).unwrap();

    assert_eq!(
        muxer.tell(receipt.id(), QualityFeedback::score(0.37).unwrap()),
        Ok(EventDisposition::Accepted)
    );
    assert_eq!(muxer.pending_len(), 1);
    assert_eq!(muxer.policy().router().summary("a").calls, 0);
    assert_eq!(
        muxer.tell(receipt.id(), QualityFeedback::execution(outcome())),
        Ok(EventDisposition::Accepted)
    );
    assert_eq!(muxer.pending_len(), 0);
    assert_eq!(muxer.policy().router().summary("a").calls, 1);
    assert_eq!(muxer.policy().router().mean_quality_score("a"), Some(0.37));
}

#[test]
fn ordered_batch_has_distinct_items_and_independent_feedback() {
    let mut muxer = Muxer::quality(arms()).with_delayed_score().build().unwrap();
    let receipt = muxer.decide_batch_from(&arms(), 2, &[]).unwrap();
    let selected: Vec<String> = receipt.selected().map(str::to_owned).collect();
    assert!(receipt.is_batch());
    assert_eq!(selected.len(), 2);
    assert_ne!(selected[0], selected[1]);

    muxer
        .tell_item(receipt.id(), 1, QualityFeedback::score(0.8).unwrap())
        .unwrap();
    muxer
        .tell_item(receipt.id(), 0, QualityFeedback::execution(outcome()))
        .unwrap();
    muxer
        .tell_item(receipt.id(), 0, QualityFeedback::score(0.2).unwrap())
        .unwrap();
    assert_eq!(muxer.pending_len(), 1);
    muxer
        .tell_item(receipt.id(), 1, QualityFeedback::execution(outcome()))
        .unwrap();
    assert_eq!(muxer.pending_len(), 0);
    assert_eq!(muxer.policy().router().summary(&selected[0]).calls, 1);
    assert_eq!(muxer.policy().router().summary(&selected[1]).calls, 1);
    assert_eq!(
        muxer.policy().router().mean_quality_score(&selected[0]),
        Some(0.2)
    );
    assert_eq!(
        muxer.policy().router().mean_quality_score(&selected[1]),
        Some(0.8)
    );
}

#[test]
fn issued_quality_execution_uses_its_decision_time_context() {
    let triage = TriageSessionConfig::default();
    let mut context = vec![0.82, 0.18];
    let expected_bin = context_bin(&context, triage.bin_cfg);
    let profile = QualityProfile::new(
        vec!["a".to_owned()],
        RouterConfig::default().with_triage_cfg(triage),
    )
    .unwrap();
    let mut muxer = Muxer::new(vec!["a".to_owned()], profile).unwrap();
    let receipt = muxer.decide(&context).unwrap();

    context.fill(0.0);
    muxer
        .tell(receipt.id(), QualityFeedback::execution(outcome()))
        .unwrap();

    let tracker = muxer.policy().router().triage_session().unwrap().tracker();
    assert_eq!(tracker.cell_calls("a", expected_bin), 1);
    assert_eq!(
        tracker.cell_calls(
            "a",
            context_bin(&context, TriageSessionConfig::default().bin_cfg)
        ),
        0
    );
}

#[test]
fn score_after_eviction_reports_outside_horizon_without_new_row() {
    let profile = QualityProfile::new(vec!["a".to_owned()], RouterConfig::default().window_cap(1))
        .unwrap()
        .with_delayed_score();
    let mut muxer = Muxer::new(vec!["a".to_owned()], profile).unwrap();
    let late = muxer.decide(&[]).unwrap();
    muxer
        .tell(late.id(), QualityFeedback::execution(outcome()))
        .unwrap();

    for score in [0.2, 0.3] {
        let receipt = muxer.decide(&[]).unwrap();
        muxer
            .tell(receipt.id(), QualityFeedback::execution(outcome()))
            .unwrap();
        muxer
            .tell(receipt.id(), QualityFeedback::score(score).unwrap())
            .unwrap();
    }

    assert_eq!(muxer.policy().router().summary("a").calls, 1);
    assert_eq!(
        muxer.tell(late.id(), QualityFeedback::score(0.8).unwrap()),
        Ok(EventDisposition::OutsideHorizon)
    );
    assert_eq!(muxer.pending_len(), 0);
    assert_eq!(muxer.policy().router().summary("a").calls, 1);
    assert_eq!(muxer.policy().router().mean_quality_score("a"), Some(0.3));
}

#[test]
fn expiring_partial_delayed_feedback_releases_profile_owned_state() {
    let profile = QualityProfile::new(arms(), RouterConfig::default())
        .unwrap()
        .with_delayed_score();
    let mut muxer = Muxer::new(arms(), profile).unwrap();

    for _ in 0..4 {
        let score_first = muxer.decide_from(&["a".to_owned()], &[]).unwrap();
        muxer
            .tell(score_first.id(), QualityFeedback::score(0.4).unwrap())
            .unwrap();
        assert_eq!(muxer.policy().buffered_score_len(), 1);
        muxer.expire(score_first.id()).unwrap();
        assert_eq!(muxer.policy().buffered_score_len(), 0);

        let execution_first = muxer.decide_from(&["a".to_owned()], &[]).unwrap();
        muxer
            .tell(execution_first.id(), QualityFeedback::execution(outcome()))
            .unwrap();
        assert_eq!(muxer.policy().awaiting_score_len(), 1);
        muxer.expire(execution_first.id()).unwrap();
        assert_eq!(muxer.policy().awaiting_score_len(), 0);
    }
    assert_eq!(muxer.pending_len(), 0);
}

#[test]
fn missing_channels_release_partial_delayed_profile_state() {
    let profile = QualityProfile::new(arms(), RouterConfig::default())
        .unwrap()
        .with_delayed_score();
    let mut muxer = Muxer::new(arms(), profile).unwrap();

    let score_first = muxer.decide_from(&["a".to_owned()], &[]).unwrap();
    muxer
        .tell(score_first.id(), QualityFeedback::score(0.4).unwrap())
        .unwrap();
    assert_eq!(muxer.policy().buffered_score_len(), 1);
    muxer
        .tell_missing(
            score_first.id(),
            0,
            QualityProfile::execution_channel(),
            "not run",
        )
        .unwrap();
    assert_eq!(muxer.policy().buffered_score_len(), 0);

    let execution_first = muxer.decide_from(&["a".to_owned()], &[]).unwrap();
    muxer
        .tell(execution_first.id(), QualityFeedback::execution(outcome()))
        .unwrap();
    assert_eq!(muxer.policy().awaiting_score_len(), 1);
    muxer
        .tell_missing(
            execution_first.id(),
            0,
            QualityProfile::score_channel(),
            "unavailable",
        )
        .unwrap();
    assert_eq!(muxer.policy().awaiting_score_len(), 0);
    assert_eq!(muxer.pending_len(), 0);
}

#[test]
fn missing_first_then_final_channel_releases_profile_owned_state() {
    let profile = QualityProfile::new(arms(), RouterConfig::default())
        .unwrap()
        .with_delayed_score();
    let mut muxer = Muxer::new(arms(), profile).unwrap();

    let missing_execution = muxer.decide_from(&["a".to_owned()], &[]).unwrap();
    muxer
        .tell_missing(
            missing_execution.id(),
            0,
            QualityProfile::execution_channel(),
            "not run",
        )
        .unwrap();
    muxer
        .tell(missing_execution.id(), QualityFeedback::score(0.4).unwrap())
        .unwrap();
    assert_eq!(muxer.policy().buffered_score_len(), 0);

    let missing_score = muxer.decide_from(&["a".to_owned()], &[]).unwrap();
    muxer
        .tell_missing(
            missing_score.id(),
            0,
            QualityProfile::score_channel(),
            "unavailable",
        )
        .unwrap();
    muxer
        .tell(missing_score.id(), QualityFeedback::execution(outcome()))
        .unwrap();
    assert_eq!(muxer.policy().awaiting_score_len(), 0);
    assert_eq!(muxer.pending_len(), 0);
}

#[test]
fn final_score_correction_is_rejected_without_changing_router() {
    let profile = QualityProfile::new(arms(), RouterConfig::default())
        .unwrap()
        .with_delayed_score();
    let mut muxer = Muxer::new(arms(), profile).unwrap();
    let receipt = muxer.decide_from(&["a".to_owned()], &[]).unwrap();
    muxer
        .tell(receipt.id(), QualityFeedback::execution(outcome()))
        .unwrap();
    muxer
        .tell(receipt.id(), QualityFeedback::score(0.2).unwrap())
        .unwrap();

    assert!(matches!(
        muxer.tell(receipt.id(), QualityFeedback::score(0.8).unwrap()),
        Err(RuntimeError::AlreadyFinalized)
    ));
    assert_eq!(muxer.policy().router().mean_quality_score("a"), Some(0.2));
}

#[test]
fn checkpoint_keeps_live_quality_triage_state() {
    let triage = TriageSessionConfig {
        min_n: 5,
        threshold: 2.0,
        ..TriageSessionConfig::default()
    };
    let mut profile = QualityProfile::new(
        vec!["a".to_owned()],
        RouterConfig::default().with_triage_cfg(triage),
    )
    .unwrap();
    for _ in 0..10 {
        assert!(profile.seed("a", Outcome::success(1, 1)));
    }
    for _ in 0..20 {
        assert!(profile.seed("a", Outcome::failure(1, 1)));
    }
    assert!(profile.router().mode().is_triage());

    let muxer = Muxer::new(vec!["a".to_owned()], profile).unwrap();
    let restored = Muxer::restore(muxer.into_checkpoint());
    assert!(restored.policy().router().mode().is_triage());
    assert_eq!(
        restored
            .policy()
            .router()
            .triage_session()
            .unwrap()
            .arm_state("a")
            .unwrap()
            .n,
        30
    );
}
