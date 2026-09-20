#![cfg(feature = "serde")]

#[cfg(any(feature = "stochastic", feature = "boltzmann"))]
mod support;

#[cfg(feature = "stochastic")]
mod stochastic_profiles {
    use muxer::{
        BoundedReward, Exp3IxConfig, Exp3MuxerCheckpoint, Exp3Profile, FractionalMuxerCheckpoint,
        FractionalThompson, Muxer, RuntimeConfig, ThompsonConfig,
    };
    use serde::{Deserialize, Serialize};
    use serde_json::Value;
    use std::io::Read;

    use super::support;
    const KEY: &str = "builtin-fractional-checkpoint-v1";

    #[derive(Serialize, Deserialize)]
    struct FractionalHandoff {
        checkpoint: FractionalMuxerCheckpoint,
        old_id: muxer::DecisionId,
    }

    fn arms() -> Vec<String> {
        vec!["a".into(), "b".into()]
    }
    fn policy(seed: u64) -> FractionalThompson {
        FractionalThompson::with_seed(ThompsonConfig::default(), seed)
    }
    fn reward(value: f64) -> BoundedReward {
        BoundedReward::new(value).unwrap()
    }

    fn fractional_handoff() -> (Muxer<FractionalThompson>, FractionalHandoff) {
        let mut muxer = Muxer::with_config(
            arms(),
            policy(11),
            RuntimeConfig {
                terminal_capacity: 3,
                ..RuntimeConfig::default()
            },
        )
        .unwrap();
        let old = muxer.decide(&()).unwrap();
        muxer.replace_policy(policy(19)).unwrap();
        let checkpoint = muxer.fractional_checkpoint(KEY).unwrap();
        (
            muxer,
            FractionalHandoff {
                checkpoint,
                old_id: old.id(),
            },
        )
    }

    fn continue_fractional(
        mut muxer: Muxer<FractionalThompson>,
        handoff: FractionalHandoff,
    ) -> Value {
        muxer.tell(handoff.old_id, reward(0.25)).unwrap();
        let mut choices = Vec::new();
        for index in 0..32 {
            let receipt = muxer.decide(&()).unwrap();
            choices.push((receipt.id().sequence(), receipt.action().to_owned()));
            muxer
                .tell(receipt.id(), reward((index % 11) as f64 / 10.0))
                .unwrap();
        }
        assert_eq!(muxer.retired_epoch_count(), 0);
        serde_json::json!({"choices": choices, "checkpoint": muxer.fractional_checkpoint(KEY).unwrap()})
    }

    #[test]
    fn fractional_checkpoint_matches_uninterrupted_process_continuation() {
        let (muxer, handoff) = fractional_handoff();
        let encoded = serde_json::to_vec(&handoff).unwrap();
        let expected = continue_fractional(muxer, handoff);
        let output = support::run_checkpoint_child(
            "stochastic_profiles::fractional_checkpoint_child",
            &encoded,
        );
        support::assert_child_success(&output);
        let stdout = String::from_utf8(output.stdout).unwrap();
        let actual: Value = serde_json::from_str(
            stdout
                .lines()
                .find_map(|line| line.strip_prefix("FRACTIONAL_RESULT="))
                .unwrap(),
        )
        .unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    #[ignore = "subprocess helper"]
    fn fractional_checkpoint_child() {
        let mut encoded = Vec::new();
        std::io::stdin().read_to_end(&mut encoded).unwrap();
        let restore: FractionalHandoff = serde_json::from_slice(&encoded).unwrap();
        let muxer = Muxer::from_fractional_checkpoint(restore.checkpoint, KEY).unwrap();
        let handoff: FractionalHandoff = serde_json::from_slice(&encoded).unwrap();
        println!("FRACTIONAL_RESULT={}", continue_fractional(muxer, handoff));
    }

    #[derive(Serialize, Deserialize)]
    struct Exp3Handoff {
        checkpoint: Exp3MuxerCheckpoint,
        old_id: muxer::DecisionId,
    }
    fn exp3_policy() -> Exp3Profile {
        Exp3Profile::new(arms(), Exp3IxConfig::default()).unwrap()
    }
    fn exp3_handoff() -> (Muxer<Exp3Profile>, Exp3Handoff) {
        let mut muxer = Muxer::with_config(
            arms(),
            exp3_policy(),
            RuntimeConfig {
                terminal_capacity: 3,
                ..RuntimeConfig::default()
            },
        )
        .unwrap();
        let subset = vec!["a".to_owned()];
        let old = muxer.decide_from(&subset, &()).unwrap();
        muxer.replace_policy(exp3_policy()).unwrap();
        let current = muxer.decide(&()).unwrap();
        muxer.tell(current.id(), reward(0.5)).unwrap();
        let checkpoint = muxer.exp3_checkpoint(KEY).unwrap();
        (
            muxer,
            Exp3Handoff {
                checkpoint,
                old_id: old.id(),
            },
        )
    }
    fn continue_exp3(mut muxer: Muxer<Exp3Profile>, handoff: Exp3Handoff) -> Value {
        muxer.tell(handoff.old_id, reward(0.4)).unwrap();
        let mut choices = Vec::new();
        for index in 0..32 {
            let subset = if index % 2 == 0 {
                vec!["a".to_owned()]
            } else {
                arms()
            };
            let receipt = muxer.decide_from(&subset, &()).unwrap();
            choices.push((receipt.id().sequence(), receipt.action().to_owned()));
            muxer
                .tell(receipt.id(), reward((index % 10) as f64 / 10.0))
                .unwrap();
        }
        assert_eq!(muxer.retired_epoch_count(), 0);
        serde_json::json!({"choices": choices, "checkpoint": muxer.exp3_checkpoint(KEY).unwrap()})
    }
    #[test]
    fn exp3_checkpoint_matches_uninterrupted_process_continuation() {
        let (muxer, handoff) = exp3_handoff();
        let encoded = serde_json::to_vec(&handoff).unwrap();
        let expected = continue_exp3(muxer, handoff);
        let output =
            support::run_checkpoint_child("stochastic_profiles::exp3_checkpoint_child", &encoded);
        support::assert_child_success(&output);
        let stdout = String::from_utf8(output.stdout).unwrap();
        let actual: Value = serde_json::from_str(
            stdout
                .lines()
                .find_map(|line| line.strip_prefix("EXP3_RESULT="))
                .unwrap(),
        )
        .unwrap();
        assert_eq!(actual, expected);
    }
    #[test]
    #[ignore = "subprocess helper"]
    fn exp3_checkpoint_child() {
        let mut encoded = Vec::new();
        std::io::stdin().read_to_end(&mut encoded).unwrap();
        let restore: Exp3Handoff = serde_json::from_slice(&encoded).unwrap();
        let muxer = Muxer::from_exp3_checkpoint(restore.checkpoint, KEY).unwrap();
        let handoff: Exp3Handoff = serde_json::from_slice(&encoded).unwrap();
        println!("EXP3_RESULT={}", continue_exp3(muxer, handoff));
    }
    #[test]
    fn exp3_checkpoint_rejects_ticket_propensity_mismatch_before_valid_restore() {
        let (_, handoff) = exp3_handoff();
        let encoded = serde_json::to_vec(&handoff).unwrap();
        let output = support::run_checkpoint_child(
            "stochastic_profiles::exp3_checkpoint_corrupt_child",
            &encoded,
        );
        support::assert_child_success(&output);
    }
    #[test]
    #[ignore = "subprocess helper"]
    fn exp3_checkpoint_corrupt_child() {
        let mut encoded = Vec::new();
        std::io::stdin().read_to_end(&mut encoded).unwrap();
        let mut wire: Value = serde_json::from_slice(&encoded).unwrap();
        wire["checkpoint"]["pending"][0]["items"][0]["ticket"]["propensity"] =
            serde_json::json!(0.5f64.to_bits());
        let bad: Exp3Handoff = serde_json::from_value(wire).unwrap();
        let error = match Muxer::from_exp3_checkpoint(bad.checkpoint, KEY) {
            Ok(_) => panic!("corrupt checkpoint restored"),
            Err(error) => error,
        };
        assert!(error
            .to_string()
            .contains("EXP3 checkpoint ticket is invalid"));
        let valid: Exp3Handoff = serde_json::from_slice(&encoded).unwrap();
        assert!(Muxer::from_exp3_checkpoint(valid.checkpoint, KEY).is_ok());
    }
}

#[cfg(feature = "boltzmann")]
mod boltzmann_profile {
    use muxer::{
        BoltzmannConfig, BoltzmannMuxerCheckpoint, BoltzmannProfile, FiniteReward, Muxer,
        RuntimeConfig,
    };
    use serde::{Deserialize, Serialize};
    use serde_json::Value;
    use std::io::Read;

    use super::support;

    const KEY: &str = "builtin-boltzmann-checkpoint-v1";

    #[derive(Serialize, Deserialize)]
    struct Handoff {
        checkpoint: BoltzmannMuxerCheckpoint,
        old: muxer::DecisionId,
    }

    fn arms() -> Vec<String> {
        vec!["a".into(), "b".into()]
    }

    fn policy() -> BoltzmannProfile {
        BoltzmannProfile::new(BoltzmannConfig::default())
    }

    fn reward(value: f64) -> FiniteReward {
        FiniteReward::new(value).unwrap()
    }

    fn handoff() -> (Muxer<BoltzmannProfile>, Handoff) {
        let mut muxer = Muxer::with_config(
            arms(),
            policy(),
            RuntimeConfig {
                terminal_capacity: 3,
                ..Default::default()
            },
        )
        .unwrap();
        let old = muxer.decide(&()).unwrap();
        muxer.replace_policy(policy()).unwrap();
        let checkpoint = muxer.boltzmann_checkpoint(KEY).unwrap();
        (
            muxer,
            Handoff {
                checkpoint,
                old: old.id(),
            },
        )
    }

    fn continue_run(mut muxer: Muxer<BoltzmannProfile>, handoff: Handoff) -> Value {
        muxer.tell(handoff.old, reward(-0.5)).unwrap();
        let mut choices = Vec::new();
        for index in 0..32 {
            let receipt = muxer.decide(&()).unwrap();
            assert!(matches!(
                receipt.probability(),
                muxer::ProbabilityAvailability::Exact(_)
            ));
            choices.push((receipt.id().sequence(), receipt.action().to_owned()));
            muxer
                .tell(receipt.id(), reward(index as f64 / 7.0 - 2.0))
                .unwrap();
        }
        assert_eq!(muxer.retired_epoch_count(), 0);
        serde_json::json!({"choices":choices,"checkpoint":muxer.boltzmann_checkpoint(KEY).unwrap()})
    }

    #[test]
    fn boltzmann_checkpoint_matches_process() {
        let (muxer, handoff) = handoff();
        let bytes = serde_json::to_vec(&handoff).unwrap();
        let expected = continue_run(muxer, handoff);
        let output =
            support::run_checkpoint_child("boltzmann_profile::boltzmann_checkpoint_child", &bytes);
        support::assert_child_success(&output);
        let stdout = String::from_utf8(output.stdout).unwrap();
        let actual: Value = serde_json::from_str(
            stdout
                .lines()
                .find_map(|line| line.strip_prefix("BOLTZ_RESULT="))
                .unwrap(),
        )
        .unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    #[ignore = "subprocess helper"]
    fn boltzmann_checkpoint_child() {
        let mut bytes = Vec::new();
        std::io::stdin().read_to_end(&mut bytes).unwrap();
        let restore: Handoff = serde_json::from_slice(&bytes).unwrap();
        let muxer = Muxer::from_boltzmann_checkpoint(restore.checkpoint, KEY).unwrap();
        let handoff: Handoff = serde_json::from_slice(&bytes).unwrap();
        println!("BOLTZ_RESULT={}", continue_run(muxer, handoff));
    }
}
