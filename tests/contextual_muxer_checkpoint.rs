#![cfg(all(feature = "serde", feature = "contextual"))]

use muxer::{
    BoundedReward, ContextualMode, ContextualMuxerCheckpoint, ContextualProfile, DecisionId,
    LinUcbConfig, Muxer, RuntimeConfig,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::Read;

mod support;

const BUILD_KEY: &str = "contextual-muxer-checkpoint-v1";

#[derive(Serialize, Deserialize)]
struct Handoff {
    checkpoint: ContextualMuxerCheckpoint,
    old_first: DecisionId,
    old_second: DecisionId,
    current: DecisionId,
}

fn actions() -> Vec<String> {
    vec!["a".to_owned(), "b".to_owned(), "c".to_owned()]
}

fn profile(mode: ContextualMode) -> ContextualProfile {
    ContextualProfile::new(LinUcbConfig {
        dim: 3,
        alpha: 0.7,
        lambda: 1.3,
        seed: 17,
        decay: 0.95,
    })
    .with_mode(mode)
    .unwrap()
}

fn handoff(mode: ContextualMode) -> (Muxer<ContextualProfile>, Handoff) {
    let mut muxer = Muxer::with_config(actions(), profile(mode), RuntimeConfig::default()).unwrap();
    for (features, reward) in [([1.0, 0.0, 0.0], 0.9), ([0.0, 1.0, 0.0], 0.2)] {
        let receipt = muxer.decide(&features).unwrap();
        muxer
            .tell(receipt.id(), BoundedReward::new(reward).unwrap())
            .unwrap();
    }

    let old_first = muxer.decide(&[0.9, 0.1, 0.0]).unwrap();
    let old_second = muxer.decide(&[0.0, 0.3, 0.7]).unwrap();
    let next_mode = match mode {
        ContextualMode::Deterministic => ContextualMode::Softmax { temperature: 0.6 },
        ContextualMode::Softmax { .. } => ContextualMode::Deterministic,
    };
    muxer
        .replace_policy_with_revisions(profile(next_mode), 7, 11)
        .unwrap();
    let current = muxer.decide(&[0.2, 0.5, 0.3]).unwrap();
    let checkpoint = muxer.contextual_checkpoint(BUILD_KEY).unwrap();
    (
        muxer,
        Handoff {
            checkpoint,
            old_first: old_first.id(),
            old_second: old_second.id(),
            current: current.id(),
        },
    )
}

fn continue_run(mut muxer: Muxer<ContextualProfile>, ids: Handoff) -> Value {
    // These two observations must update the retired original feature kernel.
    muxer
        .tell(ids.old_first, BoundedReward::new(0.8).unwrap())
        .unwrap();
    muxer
        .tell(ids.old_second, BoundedReward::new(0.1).unwrap())
        .unwrap();
    muxer
        .tell(ids.current, BoundedReward::new(0.6).unwrap())
        .unwrap();
    assert_eq!(muxer.retired_epoch_count(), 1);

    let mut choices = Vec::new();
    for index in 0..32 {
        let features = [
            (index % 3) as f64 / 2.0,
            ((index + 1) % 3) as f64 / 2.0,
            ((index + 2) % 3) as f64 / 2.0,
        ];
        let receipt = muxer.decide(&features).unwrap();
        choices.push((receipt.id().sequence(), receipt.action().to_owned()));
        muxer
            .tell(
                receipt.id(),
                BoundedReward::new((index % 5) as f64 / 4.0).unwrap(),
            )
            .unwrap();
    }
    json!({"choices": choices, "checkpoint": muxer.contextual_checkpoint(BUILD_KEY).unwrap()})
}

fn assert_fresh_process(mode: ContextualMode, child: &str) {
    let (muxer, state) = handoff(mode);
    let bytes = serde_json::to_vec(&state).unwrap();
    let expected = continue_run(muxer, state);
    let output = support::run_checkpoint_child(child, &bytes);
    support::assert_child_success(&output);
    let stdout = String::from_utf8(output.stdout).unwrap();
    let actual: Value = serde_json::from_str(
        stdout
            .lines()
            .find_map(|line| line.strip_prefix("CONTEXTUAL_RESULT="))
            .unwrap(),
    )
    .unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn contextual_checkpoint_deterministic_matches_fresh_process() {
    assert_fresh_process(
        ContextualMode::Deterministic,
        "contextual_checkpoint_deterministic_child",
    );
}

#[test]
fn contextual_checkpoint_softmax_matches_fresh_process() {
    assert_fresh_process(
        ContextualMode::Softmax { temperature: 0.4 },
        "contextual_checkpoint_softmax_child",
    );
}

#[test]
#[ignore = "subprocess helper"]
fn contextual_checkpoint_deterministic_child() {
    child();
}

#[test]
#[ignore = "subprocess helper"]
fn contextual_checkpoint_softmax_child() {
    child();
}

fn child() {
    let mut bytes = Vec::new();
    std::io::stdin().read_to_end(&mut bytes).unwrap();
    let state: Handoff = serde_json::from_slice(&bytes).unwrap();
    let muxer = Muxer::from_contextual_checkpoint(state.checkpoint, BUILD_KEY).unwrap();
    let ids: Handoff = serde_json::from_slice(&bytes).unwrap();
    println!("CONTEXTUAL_RESULT={}", continue_run(muxer, ids));
}

#[test]
fn contextual_checkpoint_rejects_tampering_before_a_valid_restore() {
    let (_, state) = handoff(ContextualMode::Deterministic);
    let bytes = serde_json::to_vec(&state).unwrap();
    let output = support::run_checkpoint_child("contextual_checkpoint_corrupt_child", &bytes);
    support::assert_child_success(&output);
}

#[test]
#[ignore = "subprocess helper"]
fn contextual_checkpoint_corrupt_child() {
    let mut bytes = Vec::new();
    std::io::stdin().read_to_end(&mut bytes).unwrap();
    let assert_bad = |wire: Value, expected: &str| {
        let state: Handoff = serde_json::from_value(wire).unwrap();
        let error = match Muxer::from_contextual_checkpoint(state.checkpoint, BUILD_KEY) {
            Ok(_) => panic!("corrupt contextual checkpoint restored"),
            Err(error) => error,
        };
        assert!(error.to_string().contains(expected), "{error}");
    };
    let base: Value = serde_json::from_slice(&bytes).unwrap();

    let mut matrix = base.clone();
    matrix["checkpoint"]["policy"]["inner"]["stats"][0]["a_inv"][0] = json!(f64::NAN.to_bits());
    assert_bad(matrix, "LinUcb checkpoint arm state must be finite");
    let mut dimension = base.clone();
    dimension["checkpoint"]["policy"]["dimensions"] = json!(99);
    assert_bad(dimension, "contextual checkpoint dimension does not match");
    let mut features = base.clone();
    features["checkpoint"]["pending"][0]["items"][0]["ticket"]["features"] = json!([1]);
    assert_bad(features, "ticket features do not match");
    let mut revision = base.clone();
    revision["checkpoint"]["pending"][0]["items"][0]["ticket"]["representation_revision"] =
        json!(99);
    assert_bad(revision, "stale representation revision");
    let mut mode = base.clone();
    mode["checkpoint"]["policy"]["mode"] =
        json!({"kind": "softmax", "temperature": f64::NAN.to_bits()});
    assert_bad(mode, "softmax temperature must be finite and positive");
    let mut receipt = base;
    receipt["checkpoint"]["pending"][0]["receipt"]["selected"][0]["reason"] = json!("Control");
    assert_bad(receipt, "receipt selections are invalid for profile mode");

    let valid: Handoff = serde_json::from_slice(&bytes).unwrap();
    assert!(Muxer::from_contextual_checkpoint(valid.checkpoint, BUILD_KEY).is_ok());
}
