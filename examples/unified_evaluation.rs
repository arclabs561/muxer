//! Project actual final runtime feedback into the existing scalar IPS helper.

#[cfg(feature = "contextual")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use muxer::{
        ips_value, project_logged_reward, BoundedReward, Channel, ContextualMode,
        ContextualProfile, EvaluationCohort, LinUcbConfig, Muxer, Probability,
    };

    let actions = vec!["a".to_owned(), "b".to_owned()];
    let policy = ContextualProfile::new(LinUcbConfig {
        dim: 1,
        ..Default::default()
    })
    .with_mode(ContextualMode::Softmax { temperature: 1.0 })?;
    let mut muxer = Muxer::new(actions, policy)?;
    let mut cohort = EvaluationCohort::default();
    let reward_channel = Channel::reward();

    // Synthetic full-information fixture: a succeeds and b fails, so the
    // uniform target's true value is 0.5. Features are fixed and resolvable.
    for _ in 0..2_000 {
        let decision = muxer.decide(&[1.0])?;
        let reward = if decision.action() == "a" { 1.0 } else { 0.0 };
        muxer.tell(decision.id(), BoundedReward::new(reward)?)?;
        cohort.record(project_logged_reward(
            &muxer,
            decision.id(),
            &reward_channel,
            &reward_channel,
            |receipt| {
                // In a real application resolve the original features by ID
                // and evaluate the target on receipt.eligible(). Do not use
                // today's availability or a newer embedding representation.
                assert_eq!(receipt.eligible().len(), 2);
                Ok(Probability::new(0.5).expect("uniform target"))
            },
        ));
    }
    assert!(cohort.exclusions().is_empty());
    println!(
        "{} rows; IPS {:.3}; synthetic target truth 0.500",
        cohort.total(),
        ips_value(cohort.rows().iter().copied())?
    );
    Ok(())
}

#[cfg(not(feature = "contextual"))]
fn main() {
    eprintln!("run with --features contextual");
}
