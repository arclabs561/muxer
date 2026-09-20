#![cfg(feature = "serde")]

use muxer::{
    CandidateAssessment, ExternalAssessments, ExternalDistribution, ExternalScores,
    InteractionPolicy, MetricObjective, ModelReference, Muxer, PolicyError, RuntimeConfig,
};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::Read;

mod support;
const KEY: &str = "external-restart-test";
type Model = fn(&u64, &[String]) -> BTreeMap<String, f64>;

fn model_a(context: &u64, actions: &[String]) -> BTreeMap<String, f64> {
    actions
        .iter()
        .enumerate()
        .map(|(i, a)| (a.clone(), 1.0 + ((context + i as u64) % 3) as f64))
        .collect()
}
fn model_b(context: &u64, actions: &[String]) -> BTreeMap<String, f64> {
    actions
        .iter()
        .enumerate()
        .map(|(i, a)| (a.clone(), 1.0 + ((context + 2 * i as u64) % 5) as f64))
        .collect()
}
fn resolve(reference: &ModelReference) -> Result<Model, PolicyError> {
    match reference.as_str() {
        "model-a" => Ok(model_a),
        "model-b" => Ok(model_b),
        _ => Err(PolicyError::new("model artifact is unavailable")),
    }
}
fn reference(name: &str) -> ModelReference {
    ModelReference::new(name).unwrap()
}
fn actions() -> Vec<String> {
    vec!["a".into(), "b".into()]
}
fn config() -> RuntimeConfig {
    RuntimeConfig {
        terminal_capacity: 4,
        ..RuntimeConfig::default()
    }
}

fn continue_model<P: InteractionPolicy<Context = u64>>(
    mut muxer: Muxer<P>,
    capture: impl Fn(&Muxer<P>) -> Value,
) -> Value {
    let mut choices = Vec::new();
    for context in 0..32 {
        let receipt = muxer.decide(&context).unwrap();
        choices.push(json!([
            receipt.id(),
            receipt.action(),
            receipt.probability()
        ]));
        assert_eq!(
            muxer.pending_len(),
            0,
            "feedbackless decisions must not acquire tickets"
        );
    }
    assert_eq!(muxer.retired_epoch_count(), 0);
    json!({"choices": choices, "checkpoint": capture(&muxer)})
}
fn continue_assessments(mut muxer: Muxer<ExternalAssessments>) -> Value {
    let mut choices = Vec::new();
    for i in 0..32 {
        let input = vec![
            CandidateAssessment::new("a", 1, vec![i as f64]),
            CandidateAssessment::new("b", 1, vec![16.0]),
        ];
        let receipt = muxer.decide(&input).unwrap();
        assert_eq!(receipt.action(), if i >= 16 { "a" } else { "b" });
        choices.push(json!([
            receipt.id(),
            receipt.action(),
            receipt.probability()
        ]));
        assert_eq!(muxer.pending_len(), 0);
    }
    json!({"choices": choices, "checkpoint": muxer.assessments_checkpoint(KEY).unwrap()})
}
fn check_child(input: Value, expected: Value) {
    let output = support::run_checkpoint_child(
        "external_restart_child",
        &serde_json::to_vec(&input).unwrap(),
    );
    support::assert_child_success(&output);
    let stdout = String::from_utf8(output.stdout).unwrap();
    let actual: Value = serde_json::from_str(
        stdout
            .lines()
            .find_map(|l| l.strip_prefix("RESULT="))
            .unwrap(),
    )
    .unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn references_restore_scores_distribution_and_assessments_in_fresh_processes() {
    let mut scores = Muxer::with_config(
        actions(),
        ExternalScores::<u64, Model>::new(model_a).with_model_reference(reference("model-a")),
        config(),
    )
    .unwrap();
    scores.decide(&0).unwrap();
    scores
        .replace_policy(
            ExternalScores::new(model_b as Model).with_model_reference(reference("model-b")),
        )
        .unwrap();
    scores.decide(&1).unwrap();
    assert_eq!(scores.retired_epoch_count(), 1);
    let saved = json!({"kind": "scores", "checkpoint": scores.scores_checkpoint(KEY).unwrap()});
    check_child(
        saved,
        continue_model(scores, |m| json!(m.scores_checkpoint(KEY).unwrap())),
    );

    let mut distribution = Muxer::with_config(
        actions(),
        ExternalDistribution::<u64, Model>::new(model_a).with_model_reference(reference("model-a")),
        config(),
    )
    .unwrap();
    distribution.decide(&0).unwrap();
    distribution
        .replace_policy(
            ExternalDistribution::new(model_b as Model).with_model_reference(reference("model-b")),
        )
        .unwrap();
    distribution.decide(&1).unwrap();
    let saved = json!({"kind": "distribution", "checkpoint": distribution.distribution_checkpoint(KEY).unwrap()});
    check_child(
        saved,
        continue_model(distribution, |m| {
            json!(m.distribution_checkpoint(KEY).unwrap())
        }),
    );

    let assessments = Muxer::with_config(
        actions(),
        ExternalAssessments::new(vec![MetricObjective::maximize(0, 1.0)]),
        config(),
    )
    .unwrap();
    let saved = json!({"kind": "assessments", "checkpoint": assessments.assessments_checkpoint(KEY).unwrap()});
    check_child(saved, continue_assessments(assessments));
}

#[test]
#[ignore = "fresh-process helper invoked by acceptance test"]
fn external_restart_child() {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).unwrap();
    let mut input: Value = serde_json::from_str(&input).unwrap();
    let checkpoint = input["checkpoint"].take();
    let result = match input["kind"].as_str().unwrap() {
        "scores" => {
            // Bad metadata must not invoke the caller's resolver.
            let mut corrupt = checkpoint.clone();
            corrupt["policy"]["envelope"]["build_key"] = json!("wrong");
            let mut calls = 0;
            let error = Muxer::<ExternalScores<u64, Model>>::from_scores_checkpoint(
                serde_json::from_value(corrupt).unwrap(),
                KEY,
                |r| {
                    calls += 1;
                    resolve(r)
                },
            )
            .err()
            .unwrap();
            assert!(error.to_string().contains("build key mismatch"));
            assert_eq!(calls, 0);
            let mut corrupt = checkpoint.clone();
            corrupt["policy"]["reference"] = json!("unavailable");
            let error = Muxer::<ExternalScores<u64, Model>>::from_scores_checkpoint(
                serde_json::from_value(corrupt).unwrap(),
                KEY,
                resolve,
            )
            .err()
            .unwrap();
            assert_eq!(error.to_string(), "model artifact is unavailable");
            // Both failures happened before reserving this engine namespace.
            let muxer = Muxer::from_scores_checkpoint(
                serde_json::from_value(checkpoint).unwrap(),
                KEY,
                resolve,
            )
            .unwrap();
            continue_model(muxer, |m| json!(m.scores_checkpoint(KEY).unwrap()))
        }
        "distribution" => {
            let muxer = Muxer::from_distribution_checkpoint(
                serde_json::from_value(checkpoint).unwrap(),
                KEY,
                resolve,
            )
            .unwrap();
            continue_model(muxer, |m| json!(m.distribution_checkpoint(KEY).unwrap()))
        }
        "assessments" => continue_assessments(
            Muxer::from_assessments_checkpoint(serde_json::from_value(checkpoint).unwrap(), KEY)
                .unwrap(),
        ),
        other => panic!("unexpected fixture kind {other}"),
    };
    println!("RESULT={result}");
}

#[test]
fn anonymous_models_remain_usable_but_cannot_claim_portable_identity() {
    assert!(ModelReference::new("").is_err());
    let mut scores = Muxer::new(actions(), ExternalScores::<u64, Model>::new(model_a)).unwrap();
    let error = scores.scores_checkpoint(KEY).err().unwrap();
    assert_eq!(
        error.to_string(),
        "external checkpoint requires a model reference"
    );
    assert_eq!(scores.decide(&0).unwrap().action(), "b");
    assert_eq!(scores.pending_len(), 0);
}

#[test]
fn reward_wire_decoding_cannot_bypass_range_and_finiteness() {
    for invalid in [f64::NAN, f64::INFINITY, -0.5, 1.5] {
        assert!(serde_json::from_value::<muxer::BoundedReward>(json!(invalid.to_bits())).is_err());
    }
    assert!(
        serde_json::from_value::<muxer::FiniteReward>(json!(f64::NEG_INFINITY.to_bits())).is_err()
    );
    for value in [0.0, 0.25, 1.0] {
        let reward = muxer::BoundedReward::new(value).unwrap();
        let decoded: muxer::BoundedReward =
            serde_json::from_value(serde_json::to_value(reward).unwrap()).unwrap();
        assert_eq!(decoded.get(), value);
    }
}
