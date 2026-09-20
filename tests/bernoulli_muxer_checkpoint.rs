#![cfg(all(feature = "serde", feature = "stochastic"))]

use muxer::{
    BernoulliMuxerCheckpoint, BernoulliThompson, Channel, DecisionId, DecisionReceipt, Disposition,
    EventId, EventOutcome, FeedbackEvent, Muxer, RuntimeConfig, ThompsonConfig,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::Read;

mod support;

const BUILD_KEY: &str = "bernoulli-muxer-checkpoint-v1";

#[derive(Serialize, Deserialize)]
struct Handoff {
    checkpoint: BernoulliMuxerCheckpoint,
    old: DecisionId,
    provisional: DecisionId,
}

fn actions() -> Vec<String> {
    vec!["a".into(), "b".into()]
}
fn policy(seed: u64) -> BernoulliThompson {
    BernoulliThompson::with_seed(ThompsonConfig::default(), seed)
}
fn reward_event(
    receipt: &DecisionReceipt,
    name: &str,
    revision: u64,
    value: Disposition<bool>,
) -> FeedbackEvent<bool> {
    FeedbackEvent::new(
        receipt.execution_key(),
        receipt.action(),
        EventId::new(name).unwrap(),
        Channel::reward(),
        1,
        revision,
        value,
    )
}
fn handoff() -> (Muxer<BernoulliThompson>, Handoff) {
    let mut muxer = Muxer::with_config(
        actions(),
        policy(17),
        RuntimeConfig {
            terminal_capacity: 3,
            ..RuntimeConfig::default()
        },
    )
    .unwrap();
    let old = muxer.decide(&()).unwrap();
    muxer.replace_policy(policy(23)).unwrap();
    let provisional = muxer.decide(&()).unwrap();
    assert_eq!(
        muxer
            .submit(reward_event(
                &provisional,
                "provisional",
                1,
                Disposition::Provisional(true)
            ))
            .unwrap(),
        EventOutcome::Provisional
    );
    let missing = muxer.decide(&()).unwrap();
    assert_eq!(
        muxer
            .submit(reward_event(
                &missing,
                "missing",
                1,
                Disposition::Missing("unavailable".into())
            ))
            .unwrap(),
        EventOutcome::Missing
    );
    let cancelled = muxer.decide(&()).unwrap();
    muxer.cancel(cancelled.id()).unwrap();
    let expired = muxer.decide(&()).unwrap();
    muxer.expire(expired.id()).unwrap();
    let checkpoint = muxer.bernoulli_checkpoint(BUILD_KEY).unwrap();
    (
        muxer,
        Handoff {
            checkpoint,
            old: old.id(),
            provisional: provisional.id(),
        },
    )
}
fn continue_run(mut muxer: Muxer<BernoulliThompson>, ids: Handoff) -> Value {
    let receipt = muxer.receipt(ids.provisional).unwrap().clone();
    assert_eq!(
        muxer
            .submit(reward_event(
                &receipt,
                "provisional",
                1,
                Disposition::Provisional(true)
            ))
            .unwrap(),
        EventOutcome::Duplicate
    );
    muxer
        .submit(reward_event(
            &receipt,
            "provisional-final",
            2,
            Disposition::Final(true),
        ))
        .unwrap();
    muxer.tell(ids.old, false).unwrap();
    let mut choices = Vec::new();
    for index in 0..32 {
        let receipt = muxer.decide(&()).unwrap();
        choices.push((receipt.id().sequence(), receipt.action().to_owned()));
        muxer.tell(receipt.id(), index % 3 != 0).unwrap();
    }
    assert_eq!(muxer.retired_epoch_count(), 0);
    json!({"choices": choices, "checkpoint": muxer.bernoulli_checkpoint(BUILD_KEY).unwrap()})
}

#[test]
fn serialized_bernoulli_checkpoint_matches_uninterrupted_continuation() {
    let (muxer, state) = handoff();
    let bytes = serde_json::to_vec(&state).unwrap();
    let expected = continue_run(muxer, state);
    let output = support::run_checkpoint_child("bernoulli_checkpoint_child", &bytes);
    support::assert_child_success(&output);
    let stdout = String::from_utf8(output.stdout).unwrap();
    let actual: Value = serde_json::from_str(
        stdout
            .lines()
            .find_map(|line| line.strip_prefix("BERNOULLI_RESULT="))
            .unwrap(),
    )
    .unwrap();
    assert_eq!(actual, expected);
}

#[test]
#[ignore = "subprocess helper"]
fn bernoulli_checkpoint_child() {
    let mut bytes = Vec::new();
    std::io::stdin().read_to_end(&mut bytes).unwrap();
    let restore: Handoff = serde_json::from_slice(&bytes).unwrap();
    let muxer = Muxer::from_bernoulli_checkpoint(restore.checkpoint, BUILD_KEY).unwrap();
    let ids: Handoff = serde_json::from_slice(&bytes).unwrap();
    println!("BERNOULLI_RESULT={}", continue_run(muxer, ids));
}

#[test]
fn bernoulli_checkpoint_rejects_corruption_before_reservation() {
    let (_, state) = handoff();
    let bytes = serde_json::to_vec(&state).unwrap();
    let output = support::run_checkpoint_child("bernoulli_corrupt_child", &bytes);
    support::assert_child_success(&output);
}
#[test]
#[ignore = "subprocess helper"]
fn bernoulli_corrupt_child() {
    let mut bytes = Vec::new();
    std::io::stdin().read_to_end(&mut bytes).unwrap();
    let assert_bad = |wire: Value, expected: &str| {
        let state: Handoff = serde_json::from_value(wire).unwrap();
        let err = match Muxer::from_bernoulli_checkpoint(state.checkpoint, BUILD_KEY) {
            Ok(_) => panic!("corrupt checkpoint restored"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains(expected),
            "expected {expected}, got {err}"
        );
    };
    let mut wire: Value = serde_json::from_slice(&bytes).unwrap();
    wire["checkpoint"]["policy"]["schema"] = json!(0);
    assert_bad(wire, "unsupported Bernoulli checkpoint schema");
    let mut wire: Value = serde_json::from_slice(&bytes).unwrap();
    wire["checkpoint"]["policy"]["kind"] = json!("wrong");
    assert_bad(wire, "Bernoulli checkpoint kind mismatch");
    let mut wire: Value = serde_json::from_slice(&bytes).unwrap();
    wire["checkpoint"]["policy"]["crate_version"] = json!("wrong");
    assert_bad(wire, "Bernoulli checkpoint crate version mismatch");
    let mut wire: Value = serde_json::from_slice(&bytes).unwrap();
    wire["checkpoint"]["policy"]["build_key"] = json!("wrong");
    assert_bad(wire, "Bernoulli checkpoint build key mismatch");
    let mut wire: Value = serde_json::from_slice(&bytes).unwrap();
    wire["checkpoint"]["policy"]["rng"]["version"] = json!(0);
    assert_bad(wire, "unsupported trial RNG state version");
    let mut wire: Value = serde_json::from_slice(&bytes).unwrap();
    let duplicate = wire["checkpoint"]["policy"]["posterior"][0].clone();
    wire["checkpoint"]["policy"]["posterior"]
        .as_array_mut()
        .unwrap()
        .push(duplicate);
    assert_bad(
        wire,
        "Bernoulli checkpoint repeats or empties a posterior action",
    );
    let mut wire: Value = serde_json::from_slice(&bytes).unwrap();
    wire["checkpoint"]["terminal"][0]["receipt"]["batch"] = json!(true);
    assert_bad(wire, "bernoulli checkpoint receipt selections are invalid");
    let mut wire: Value = serde_json::from_slice(&bytes).unwrap();
    wire["checkpoint"]["terminal"][0]["receipt"]["selected"][0]["reason"] = json!("Control");
    assert_bad(wire, "bernoulli checkpoint receipt selections are invalid");
    let mut wire: Value = serde_json::from_slice(&bytes).unwrap();
    wire["checkpoint"]["terminal"][0]["receipt"]["selected"][0]["probability"] =
        json!({"Exact": 0.5});
    assert_bad(wire, "bernoulli checkpoint receipt selections are invalid");
    let state: Handoff = serde_json::from_slice(&bytes).unwrap();
    assert!(Muxer::from_bernoulli_checkpoint(state.checkpoint, BUILD_KEY).is_ok());
}
