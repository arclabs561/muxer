#[cfg(any(feature = "stochastic", feature = "contextual"))]
use muxer::BoundedReward;
#[cfg(any(feature = "stochastic", feature = "contextual", feature = "boltzmann"))]
use muxer::EventDisposition;
use muxer::{
    CandidateAssessment, DecisionReason, ExternalAssessments, ExternalDistribution, ExternalScores,
    MetricObjective, Muxer, ProbabilityAvailability,
};
use std::collections::BTreeMap;

fn actions() -> Vec<String> {
    ["a", "b", "c"].map(str::to_owned).to_vec()
}

#[cfg(feature = "stochastic")]
#[test]
fn bernoulli_profile_correlates_feedback_and_deduplicates() {
    use muxer::{BernoulliThompson, ThompsonConfig};
    let mut muxer = Muxer::new(
        actions(),
        BernoulliThompson::with_seed(ThompsonConfig::default(), 3),
    )
    .unwrap();
    let receipt = muxer.decide(&()).unwrap();
    assert!(actions().contains(&receipt.action().to_owned()));
    assert_eq!(receipt.reason(), DecisionReason::ExploreFirst);
    assert_eq!(
        muxer.tell(receipt.id(), true),
        Ok(EventDisposition::Accepted)
    );
    assert_eq!(
        muxer.tell(receipt.id(), true),
        Ok(EventDisposition::Duplicate)
    );
}

#[cfg(feature = "stochastic")]
#[test]
fn exp3_profile_logs_actual_filtered_propensity() {
    use muxer::{Exp3IxConfig, Exp3Profile};
    let mut muxer = Muxer::new(
        actions(),
        Exp3Profile::new(actions(), Exp3IxConfig::default()).unwrap(),
    )
    .unwrap();
    let eligible = vec!["b".to_owned(), "c".to_owned()];
    let receipt = muxer.decide_from(&eligible, &()).unwrap();
    assert!(eligible.contains(&receipt.action().to_owned()));
    assert!(
        matches!(receipt.probability(), ProbabilityAvailability::Exact(value) if value.get() > 0.0)
    );
    assert_eq!(
        muxer.tell(receipt.id(), BoundedReward::new(0.7).unwrap()),
        Ok(EventDisposition::Accepted)
    );
}

#[cfg(feature = "stochastic")]
#[test]
fn exp3_delayed_feedback_matches_kernel_with_original_propensities() {
    use muxer::{Exp3IxConfig, Exp3Profile};
    let config = Exp3IxConfig {
        seed: 9,
        ..Exp3IxConfig::default()
    };
    let mut muxer = Muxer::new(actions(), Exp3Profile::new(actions(), config).unwrap()).unwrap();
    let d1 = muxer
        .decide_from(&["a".to_owned(), "b".to_owned()], &())
        .unwrap();
    let d2 = muxer
        .decide_from(&["b".to_owned(), "c".to_owned()], &())
        .unwrap();
    let d3 = muxer
        .decide_from(&["a".to_owned(), "c".to_owned()], &())
        .unwrap();
    let mut direct = muxer.policy().inner().clone();
    for (receipt, reward) in [(&d3, 0.2), (&d1, 0.9), (&d2, 0.6)] {
        let ProbabilityAvailability::Exact(propensity) = receipt.probability() else {
            panic!("EXP3 must log a propensity");
        };
        direct.update_reward_with_prob(receipt.action(), reward, propensity.get());
        assert_eq!(
            muxer.tell(receipt.id(), BoundedReward::new(reward).unwrap()),
            Ok(EventDisposition::Accepted)
        );
    }
    assert_eq!(
        muxer.policy().inner().snapshot().cum_loss_hat,
        direct.snapshot().cum_loss_hat
    );
}

#[cfg(feature = "stochastic")]
#[test]
fn exp3_rejects_catalogue_actions_outside_its_universe() {
    use muxer::{Exp3IxConfig, Exp3Profile};
    let profile = Exp3Profile::new(
        vec!["a".to_owned(), "b".to_owned()],
        Exp3IxConfig::default(),
    )
    .unwrap();
    let mut muxer = Muxer::new(actions(), profile).unwrap();
    let error = muxer.decide_from(&["c".to_owned()], &()).unwrap_err();
    assert!(error
        .to_string()
        .contains("outside the EXP3 canonical universe"));
}

#[test]
fn external_scores_are_terminal_at_issue() {
    let scorer = ExternalScores::new(|_: &(), _: &[String]| {
        BTreeMap::from([
            ("a".to_owned(), 0.2),
            ("b".to_owned(), 0.9),
            ("c".to_owned(), 0.1),
        ])
    });
    let mut muxer = Muxer::new(actions(), scorer).unwrap();
    let receipt = muxer.decide(&()).unwrap();
    assert_eq!(receipt.action(), "b");
    assert_eq!(receipt.reason(), DecisionReason::Deterministic);
    assert_eq!(muxer.pending_len(), 0);
    assert!(
        matches!(receipt.probability(), ProbabilityAvailability::Exact(value) if value.get() == 1.0)
    );
}

#[test]
fn external_distribution_normalizes_and_samples_logged_mass() {
    let distribution = ExternalDistribution::new(|_: &(), _: &[String]| {
        BTreeMap::from([
            ("a".to_owned(), 2.0),
            ("b".to_owned(), 1.0),
            ("c".to_owned(), 1.0),
        ])
    });
    let mut muxer = Muxer::new(actions(), distribution).unwrap();
    let receipt = muxer.decide(&()).unwrap();
    assert!(
        matches!(receipt.probability(), ProbabilityAvailability::Exact(value) if [0.25, 0.5].contains(&value.get()))
    );
    assert_eq!(receipt.reason(), DecisionReason::CategoricalSample);
}

#[test]
fn external_assessments_filter_to_canonical_eligible_actions() {
    let profile = ExternalAssessments::new(vec![MetricObjective::maximize(0, 1.0)]);
    let mut muxer = Muxer::new(actions(), profile).unwrap();
    let eligible = vec!["b".to_owned(), "a".to_owned()];
    let assessments = vec![
        CandidateAssessment::new("c", 1, vec![100.0]),
        CandidateAssessment::new("a", 1, vec![0.2]),
        CandidateAssessment::new("b", 1, vec![0.9]),
    ];
    let receipt = muxer.decide_from(&eligible, &assessments).unwrap();
    assert_eq!(receipt.eligible(), ["a", "b"]);
    assert_eq!(receipt.action(), "b");
    assert_eq!(muxer.pending_len(), 0);
}

#[cfg(feature = "contextual")]
#[test]
fn contextual_profile_copies_features_and_logs_softmax_propensity() {
    use muxer::{ContextualMode, ContextualProfile, LinUcbConfig};
    let profile = ContextualProfile::new(LinUcbConfig {
        dim: 2,
        ..LinUcbConfig::default()
    })
    .with_mode(ContextualMode::Softmax { temperature: 0.5 })
    .unwrap();
    let mut muxer = Muxer::new(actions(), profile).unwrap();
    let features = vec![0.3, 0.8];
    let receipt = muxer.decide(&features).unwrap();
    assert!(
        matches!(receipt.probability(), ProbabilityAvailability::Exact(value) if value.get() > 0.0)
    );
    assert_eq!(
        muxer.tell(receipt.id(), BoundedReward::new(1.0).unwrap()),
        Ok(EventDisposition::Accepted)
    );
    assert_eq!(
        muxer.policy().inner().snapshot().arms[receipt.action()].uses,
        1
    );
}

#[cfg(feature = "contextual")]
#[test]
fn contextual_delayed_feedback_uses_original_vectors_and_matches_kernel() {
    use muxer::{ContextualProfile, LinUcbConfig};
    let config = LinUcbConfig {
        dim: 2,
        ..LinUcbConfig::default()
    };
    let mut muxer = Muxer::new(actions(), ContextualProfile::new(config)).unwrap();
    let mut first = vec![0.2, 0.9];
    let mut second = vec![0.8, 0.1];
    let d1 = muxer.decide(&first).unwrap();
    let d2 = muxer.decide(&second).unwrap();
    let mut direct = muxer.policy().inner().clone();
    first.fill(-99.0);
    second.fill(77.0);
    for (receipt, original, reward) in [(&d2, &[0.8, 0.1][..], 0.3), (&d1, &[0.2, 0.9][..], 0.85)] {
        direct.update_reward(receipt.action(), original, reward);
        assert_eq!(
            muxer.tell(receipt.id(), BoundedReward::new(reward).unwrap()),
            Ok(EventDisposition::Accepted)
        );
    }
    let actual = muxer.policy().inner().snapshot();
    let expected = direct.snapshot();
    assert_eq!(actual.arms.len(), expected.arms.len());
    for (action, expected_arm) in expected.arms {
        let actual_arm = &actual.arms[&action];
        assert_eq!(actual_arm.uses, expected_arm.uses);
        assert_eq!(actual_arm.a_inv, expected_arm.a_inv);
        assert_eq!(actual_arm.b, expected_arm.b);
    }
}

#[cfg(feature = "boltzmann")]
#[test]
fn boltzmann_profile_records_runtime_sampled_distribution() {
    use muxer::{BoltzmannConfig, BoltzmannProfile, FiniteReward};
    let mut muxer =
        Muxer::new(actions(), BoltzmannProfile::new(BoltzmannConfig::default())).unwrap();
    let receipt = muxer.decide(&()).unwrap();
    assert!(
        matches!(receipt.probability(), ProbabilityAvailability::Exact(value) if value.get() > 0.0)
    );
    assert_eq!(
        muxer.tell(receipt.id(), FiniteReward::new(-0.2).unwrap()),
        Ok(EventDisposition::Accepted)
    );
}
