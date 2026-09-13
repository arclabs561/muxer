use muxer::{
    ips_value, project_logged_reward, Channel, EvaluationCohort, FeedbackExpectation,
    InteractionPolicy, Muxer, PolicyDecision, PolicyError, PolicyRequest, Probability,
    ProbabilityAvailability, ProjectionError, TargetEvidenceUnavailable, TrialRng,
};

/// Full-information synthetic environment: action a always succeeds and b
/// always fails. The logger prefers a (0.8); the target is uniform (value 0.5).
#[derive(Clone)]
struct KnownLogger;

impl InteractionPolicy for KnownLogger {
    type Context = ();
    type Feedback = bool;
    type CanonicalFeedback = bool;
    type Ticket = ();
    type PreparedIssue = ();
    type PreparedUpdate = ();

    fn prepare_decision(
        &self,
        request: PolicyRequest<'_, ()>,
        rng: &mut TrialRng,
    ) -> Result<PolicyDecision<(), ()>, PolicyError> {
        let (selection, p) = if rng.unit_f64() < 0.8 {
            ("a", 0.8)
        } else {
            ("b", 0.2)
        };
        assert_eq!(request.eligible(), &["a".to_owned(), "b".to_owned()]);
        Ok(PolicyDecision {
            selection: selection.to_owned(),
            probability: ProbabilityAvailability::Exact(Probability::new(p).unwrap()),
            ticket: Some(()),
            issue: (),
            expectation: FeedbackExpectation::FinalValue {
                channel: Channel::reward(),
            },
        })
    }
    fn commit_issue(&mut self, _: ()) {}
    fn normalize(&self, value: bool) -> Result<bool, PolicyError> {
        Ok(value)
    }
    fn prepare(&self, _: &(), _: &bool) -> Result<(), PolicyError> {
        Ok(())
    }
    fn apply(&mut self, _: ()) {}
}

fn logger() -> Muxer<KnownLogger> {
    Muxer::new(vec!["a".into(), "b".into()], KnownLogger).unwrap()
}

#[test]
fn projection_uses_actual_final_feedback_and_retained_probability() {
    let mut mux = logger();
    let receipt = mux.decide(&()).unwrap();
    let channel = Channel::reward();
    let pending = project_logged_reward(&mux, receipt.id(), &channel, &channel, |_| {
        Ok(Probability::new(0.5).unwrap())
    });
    assert_eq!(pending.unwrap_err(), ProjectionError::ExecutionUnconfirmed);
    mux.tell(receipt.id(), false).unwrap();
    let row = project_logged_reward(&mux, receipt.id(), &channel, &channel, |saved| {
        assert_eq!(saved.id(), receipt.id());
        assert_eq!(saved.action(), receipt.action());
        assert_eq!(saved.eligible(), receipt.eligible());
        Ok(Probability::new(0.5).unwrap())
    })
    .unwrap();
    assert_eq!(row.reward, 0.0);
    assert_eq!(
        row.logging_propensity,
        if receipt.action() == "a" { 0.8 } else { 0.2 }
    );
    assert_eq!(row.target_propensity, 0.5);
    assert_eq!(
        project_logged_reward(&mux, receipt.id(), &channel, &channel, |_| Err(
            TargetEvidenceUnavailable
        ))
        .unwrap_err(),
        ProjectionError::TargetEvidenceUnavailable
    );
}

#[test]
fn missing_and_foreign_decisions_are_counted_not_synthesized_as_zero() {
    let mut mux = logger();
    let mut foreign = logger();
    let missing = mux.decide(&()).unwrap();
    mux.expire(missing.id()).unwrap();
    let foreign = foreign.decide(&()).unwrap();
    let channel = Channel::reward();
    let mut cohort = EvaluationCohort::default();
    for id in [missing.id(), foreign.id()] {
        cohort.record(project_logged_reward(&mux, id, &channel, &channel, |_| {
            Ok(Probability::new(0.5).unwrap())
        }));
    }
    assert_eq!(cohort.total(), 2);
    assert!(cohort.rows().is_empty());
    assert_eq!(
        cohort.exclusions()[&ProjectionError::ExecutionUnconfirmed],
        1
    );
    assert_eq!(cohort.exclusions()[&ProjectionError::NotRetained], 1);
}

#[test]
fn synthetic_full_information_truth_distinguishes_ips_from_naive_average() {
    let mut mux = logger();
    let mut cohort = EvaluationCohort::default();
    let channel = Channel::reward();
    for _ in 0..10_000 {
        let receipt = mux.decide(&()).unwrap();
        mux.tell(receipt.id(), receipt.action() == "a").unwrap();
        cohort.record(project_logged_reward(
            &mux,
            receipt.id(),
            &channel,
            &channel,
            |_| Ok(Probability::new(0.5).unwrap()),
        ));
    }
    assert!(cohort.exclusions().is_empty());
    let ips = ips_value(cohort.rows().iter().copied()).unwrap();
    let naive = cohort.rows().iter().map(|row| row.reward).sum::<f64>() / cohort.total() as f64;
    assert!(
        (ips - 0.5).abs() < 0.02,
        "IPS {ips} should recover known target truth 0.5"
    );
    assert!(
        (naive - 0.5).abs() > 0.2,
        "naive average {naive} should reflect logger bias"
    );
}

#[test]
fn explicitly_ordered_singleton_batch_is_not_a_scalar_decision() {
    let mut mux = Muxer::quality(["a", "b"]).build().unwrap();
    let batch = mux.decide_batch_from(&["a".to_owned()], 1, &[]).unwrap();
    assert_eq!(batch.selected_len(), 1);
    assert!(batch.is_batch());
    let channel = Channel::reward();
    assert_eq!(
        project_logged_reward(&mux, batch.id(), &channel, &channel, |_| Ok(
            Probability::new(1.0).unwrap()
        ))
        .unwrap_err(),
        ProjectionError::BatchUnsupported
    );
}

#[cfg(feature = "stochastic")]
#[test]
fn posterior_max_sampling_does_not_invent_a_propensity() {
    let mut mux = Muxer::bernoulli(["a", "b"]).unwrap();
    let receipt = mux.decide(&()).unwrap();
    mux.tell(receipt.id(), true).unwrap();
    let channel = Channel::reward();
    assert_eq!(
        project_logged_reward(&mux, receipt.id(), &channel, &channel, |_| Ok(
            Probability::new(0.5).unwrap()
        ))
        .unwrap_err(),
        ProjectionError::ProbabilityUnavailable
    );
}
