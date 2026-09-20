#![cfg(feature = "serde")]

use muxer::{
    Channel, DecisionId, DecisionReceipt, Disposition, EventId, EventOutcome, ExecutionKey,
    FeedbackEvent, ItemStatus, Muxer, Outcome, QualityFeedback, QualityMuxerCheckpoint,
    QualityProfile, RouterConfig, RuntimeConfig, TerminalStatus, TriageSessionConfig,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::Read;

mod support;

const BUILD_KEY: &str = "quality-muxer-checkpoint-test-v1";

#[derive(Serialize, Deserialize)]
struct Handoff {
    checkpoint: QualityMuxerCheckpoint,
    score_before_execution: DecisionId,
    execution_before_score: DecisionId,
    old_epoch: DecisionId,
    provisional: DecisionId,
    mixed_pending_batch: DecisionId,
}

fn actions() -> Vec<String> {
    vec!["a".to_owned(), "b".to_owned()]
}

fn execution() -> QualityFeedback {
    QualityFeedback::execution(Outcome::success(2, 10))
}

fn profile() -> QualityProfile {
    let triage = TriageSessionConfig {
        min_n: 5,
        threshold: 2.0,
        ..TriageSessionConfig::default()
    };
    let mut profile =
        QualityProfile::new(actions(), RouterConfig::default().with_triage_cfg(triage))
            .unwrap()
            .with_delayed_score();
    for _ in 0..10 {
        assert!(profile.seed("a", Outcome::success(1, 1)));
    }
    for _ in 0..20 {
        assert!(profile.seed("a", Outcome::failure(1, 1)));
    }
    assert!(profile.router().mode().is_triage());
    profile
}

fn event(
    receipt: &DecisionReceipt,
    position: usize,
    event_id: &str,
    channel: Channel,
    revision: u64,
    disposition: Disposition<QualityFeedback>,
) -> FeedbackEvent<QualityFeedback> {
    FeedbackEvent::new(
        ExecutionKey::new(receipt.id(), position),
        receipt.selected().nth(position).unwrap(),
        EventId::new(event_id).unwrap(),
        channel,
        1,
        revision,
        disposition,
    )
}

fn initial_handoff() -> (Muxer<QualityProfile>, Handoff) {
    let a = vec!["a".to_owned()];
    let mut muxer = Muxer::with_config(
        actions(),
        profile(),
        RuntimeConfig {
            terminal_capacity: 3,
            ..RuntimeConfig::default()
        },
    )
    .unwrap();

    // This ticket must continue to normalize against the first policy epoch.
    let old_epoch = muxer.decide_from(&a, &[]).unwrap();
    muxer.replace_policy(profile()).unwrap();
    assert_eq!(muxer.retired_epoch_count(), 1);
    assert!(muxer.policy().router().mode().is_triage());

    // Delayed score arrives first and must survive in the profile's buffer.
    let score_before_execution = muxer.decide_from(&a, &[]).unwrap();
    assert_eq!(
        muxer
            .submit(event(
                &score_before_execution,
                0,
                "score-before-execution",
                QualityProfile::score_channel(),
                1,
                Disposition::Final(QualityFeedback::score(0.31).unwrap()),
            ))
            .unwrap(),
        EventOutcome::Accepted
    );

    // Execution arrives first and must survive in the profile's awaiting set.
    let execution_before_score = muxer.decide_from(&a, &[]).unwrap();
    muxer
        .tell(execution_before_score.id(), execution())
        .unwrap();

    // The retained provisional envelope must keep its idempotency identity.
    let provisional = muxer.decide_from(&a, &[]).unwrap();
    assert_eq!(
        muxer
            .submit(event(
                &provisional,
                0,
                "provisional-score",
                QualityProfile::score_channel(),
                1,
                Disposition::Provisional(QualityFeedback::score(0.42).unwrap()),
            ))
            .unwrap(),
        EventOutcome::Provisional
    );

    // A terminal batch carries both finalized and missing event evidence, with
    // one item cancelled after its missing channel was recorded.
    let batch = muxer.decide_batch(2, &[]).unwrap();
    muxer.tell_item(batch.id(), 0, execution()).unwrap();
    muxer
        .tell_item(batch.id(), 0, QualityFeedback::score(0.73).unwrap())
        .unwrap();
    assert_eq!(
        muxer
            .submit(event(
                &batch,
                1,
                "batch-score-missing",
                QualityProfile::score_channel(),
                1,
                Disposition::Missing("not assessed".to_owned()),
            ))
            .unwrap(),
        EventOutcome::Missing
    );
    muxer.cancel_item(batch.id(), 1).unwrap();
    assert_eq!(
        muxer.item_status(batch.id(), 0),
        Some(ItemStatus::Completed)
    );
    assert_eq!(
        muxer.item_status(batch.id(), 1),
        Some(ItemStatus::Cancelled)
    );
    assert_eq!(
        muxer.terminal_status(batch.id()),
        Some(TerminalStatus::Completed)
    );

    // Capture one completed sibling alongside an unresolved sibling. This is
    // distinct from the terminal batch above: restore must retain the open
    // ticket without discarding the completed item's final evidence.
    let mixed_pending_batch = muxer.decide_batch(2, &[]).unwrap();
    muxer
        .tell_item(mixed_pending_batch.id(), 0, execution())
        .unwrap();
    muxer
        .tell_item(
            mixed_pending_batch.id(),
            0,
            QualityFeedback::score(0.61).unwrap(),
        )
        .unwrap();
    assert_eq!(
        muxer.item_status(mixed_pending_batch.id(), 0),
        Some(ItemStatus::Completed)
    );
    assert_eq!(
        muxer.item_status(mixed_pending_batch.id(), 1),
        Some(ItemStatus::Open)
    );

    let handoff = Handoff {
        checkpoint: muxer.quality_checkpoint(BUILD_KEY).unwrap(),
        score_before_execution: score_before_execution.id(),
        execution_before_score: execution_before_score.id(),
        old_epoch: old_epoch.id(),
        provisional: provisional.id(),
        mixed_pending_batch: mixed_pending_batch.id(),
    };
    (muxer, handoff)
}

fn continue_after_restore(mut muxer: Muxer<QualityProfile>, handoff: Handoff) -> Value {
    assert!(muxer.policy().router().mode().is_triage());

    muxer.expire(handoff.mixed_pending_batch).unwrap();
    assert_eq!(
        muxer.terminal_status(handoff.mixed_pending_batch),
        Some(TerminalStatus::Expired)
    );
    assert_eq!(
        muxer.item_status(handoff.mixed_pending_batch, 0),
        Some(ItemStatus::Completed)
    );
    assert_eq!(
        muxer.item_status(handoff.mixed_pending_batch, 1),
        Some(ItemStatus::Expired)
    );

    // A byte-for-byte repeated provisional event is still a duplicate after
    // serialization. A newer final envelope then resolves the channel once.
    let provisional_receipt = muxer.receipt(handoff.provisional).unwrap().clone();
    assert_eq!(
        muxer
            .submit(event(
                &provisional_receipt,
                0,
                "provisional-score",
                QualityProfile::score_channel(),
                1,
                Disposition::Provisional(QualityFeedback::score(0.42).unwrap()),
            ))
            .unwrap(),
        EventOutcome::Duplicate
    );
    assert_eq!(
        muxer
            .submit(event(
                &provisional_receipt,
                0,
                "provisional-score-final",
                QualityProfile::score_channel(),
                2,
                Disposition::Final(QualityFeedback::score(0.44).unwrap()),
            ))
            .unwrap(),
        EventOutcome::Accepted
    );
    muxer.tell(handoff.provisional, execution()).unwrap();

    muxer
        .tell(handoff.score_before_execution, execution())
        .unwrap();
    muxer
        .tell(
            handoff.execution_before_score,
            QualityFeedback::score(0.67).unwrap(),
        )
        .unwrap();
    muxer.tell(handoff.old_epoch, execution()).unwrap();
    muxer
        .tell(handoff.old_epoch, QualityFeedback::score(0.79).unwrap())
        .unwrap();
    assert_eq!(muxer.pending_len(), 0);

    // Identical post-restart decision sequences demonstrate persisted RNG,
    // profile state, and receipt sequencing rather than only a JSON roundtrip.
    let mut choices = Vec::new();
    for score in [0.11, 0.24, 0.58, 0.89] {
        let receipt = muxer.decide(&[]).unwrap();
        choices.push(receipt.action().to_owned());
        muxer.tell(receipt.id(), execution()).unwrap();
        muxer
            .tell(receipt.id(), QualityFeedback::score(score).unwrap())
            .unwrap();
    }
    assert_eq!(muxer.retired_epoch_count(), 0);

    json!({
        "choices": choices,
        "checkpoint": muxer.quality_checkpoint(BUILD_KEY).unwrap(),
    })
}

#[test]
fn serialized_quality_muxer_checkpoint_matches_uninterrupted_execution() {
    let (muxer, handoff) = initial_handoff();
    let encoded = serde_json::to_vec(&handoff).unwrap();
    let expected = continue_after_restore(muxer, handoff);

    let output = support::run_checkpoint_child("quality_muxer_checkpoint_child", &encoded);
    support::assert_child_success(&output);
    let stdout = String::from_utf8(output.stdout).unwrap();
    let result = stdout
        .lines()
        .find_map(|line| line.strip_prefix("QUALITY_MUXER_CHECKPOINT_RESULT="))
        .unwrap();
    let actual: Value = serde_json::from_str(result).unwrap();
    assert_eq!(actual, expected);
}

#[test]
#[ignore = "invoked by serialized_quality_muxer_checkpoint_matches_uninterrupted_execution"]
fn quality_muxer_checkpoint_child() {
    let mut encoded = Vec::new();
    std::io::stdin().read_to_end(&mut encoded).unwrap();
    let checkpoint_handoff: Handoff = serde_json::from_slice(&encoded).unwrap();
    let muxer = Muxer::from_quality_checkpoint(checkpoint_handoff.checkpoint, BUILD_KEY).unwrap();
    let handoff: Handoff = serde_json::from_slice(&encoded).unwrap();
    println!(
        "QUALITY_MUXER_CHECKPOINT_RESULT={}",
        continue_after_restore(muxer, handoff)
    );
}

#[test]
fn quality_muxer_checkpoint_rejects_tampering_without_reserving_namespace() {
    let (_, handoff) = initial_handoff();
    let encoded = serde_json::to_vec(&handoff).unwrap();

    let mut malformed: Value = serde_json::from_slice(&encoded).unwrap();
    malformed["checkpoint"]
        .as_object_mut()
        .unwrap()
        .insert("unexpected".to_owned(), Value::Null);
    assert!(serde_json::from_value::<Handoff>(malformed).is_err());

    let output = support::run_checkpoint_child(
        "quality_muxer_checkpoint_corrupt_then_valid_child",
        &encoded,
    );
    support::assert_child_success(&output);
}

#[test]
fn quality_checkpoint_rejects_cloned_orphaned_delayed_profile_state() {
    let (muxer, _) = initial_handoff();
    assert!(muxer.quality_checkpoint(BUILD_KEY).is_ok());

    // Cloning a profile carrying score-first/execution-first state does not
    // clone the corresponding receipts. The capture must reject that orphaned
    // state rather than serializing a checkpoint that cannot be restored.
    let orphaned = Muxer::new(actions(), muxer.policy().clone()).unwrap();
    let error = match orphaned.quality_checkpoint(BUILD_KEY) {
        Ok(_) => panic!("orphaned delayed profile state checkpointed"),
        Err(error) => error,
    };
    assert!(error
        .to_string()
        .contains("quality checkpoint delayed state does not match open ticket progress"));
}

#[test]
fn quality_checkpoint_v1_fixture_restores_in_a_fresh_process() {
    // Generated with the pre-extraction implementation at commit 93395d5.
    // Keep these historical bytes fixed; regenerating with current code would
    // no longer check backward compatibility of the quality-v1 wire shape.
    let output = support::run_checkpoint_child(
        "quality_checkpoint_v1_fixture_child",
        include_bytes!("fixtures/quality_checkpoint_v1.json"),
    );
    support::assert_child_success(&output);
}

#[test]
#[ignore = "invoked by quality_checkpoint_v1_fixture_restores_in_a_fresh_process"]
fn quality_checkpoint_v1_fixture_child() {
    let mut encoded = Vec::new();
    std::io::stdin().read_to_end(&mut encoded).unwrap();
    let checkpoint: QualityMuxerCheckpoint = serde_json::from_slice(&encoded).unwrap();
    let mut muxer = Muxer::from_quality_checkpoint(checkpoint, "quality-v1-fixture").unwrap();
    let pending: DecisionId = serde_json::from_value(json!({"engine": 1, "sequence": 1})).unwrap();
    muxer.tell(pending, execution()).unwrap();
    assert_eq!(muxer.pending_len(), 0);
    assert_eq!(
        muxer.terminal_status(pending),
        Some(TerminalStatus::Completed)
    );

    let next = muxer.decide(&[]).unwrap();
    assert_eq!(next.action(), "a");
    muxer.tell(next.id(), execution()).unwrap();
    muxer
        .tell(next.id(), QualityFeedback::score(0.63).unwrap())
        .unwrap();
    assert_eq!(muxer.pending_len(), 0);
    assert!(muxer.quality_checkpoint("quality-v1-fixture").is_ok());
}

#[test]
#[ignore = "invoked by quality_muxer_checkpoint_rejects_tampering_without_reserving_namespace"]
fn quality_muxer_checkpoint_corrupt_then_valid_child() {
    let mut encoded = Vec::new();
    std::io::stdin().read_to_end(&mut encoded).unwrap();

    let mut corrupt: Value = serde_json::from_slice(&encoded).unwrap();
    corrupt["checkpoint"]["version"] = json!(0);
    assert_corrupt_restore_error(corrupt, "unsupported quality muxer checkpoint identity");

    // A structurally valid receipt with the wrong item count must reject
    // before dereferencing parallel selected-item arrays.
    let mut corrupt: Value = serde_json::from_slice(&encoded).unwrap();
    corrupt["checkpoint"]["pending"][0]["items"]
        .as_array_mut()
        .unwrap()
        .pop();
    assert_corrupt_restore_error(corrupt, "quality checkpoint pending item count mismatch");

    // Advanced EventIds and arrival sequences are runtime-global, not merely
    // ledger-local. Reusing either in a different retained receipt is valid
    // JSON but cannot arise from accepted runtime events.
    let mut corrupt: Value = serde_json::from_slice(&encoded).unwrap();
    let event_id = corrupt["checkpoint"]["terminal"][0]["ledger"]["events"][0]["event_id"].clone();
    first_pending_ledger_event(&mut corrupt)["event_id"] = event_id;
    assert_corrupt_restore_error(corrupt, "quality checkpoint repeats retained event ID");

    let mut corrupt: Value = serde_json::from_slice(&encoded).unwrap();
    let received_sequence =
        corrupt["checkpoint"]["terminal"][0]["ledger"]["events"][0]["received_sequence"].clone();
    first_pending_ledger_event(&mut corrupt)["received_sequence"] = received_sequence;
    assert_corrupt_restore_error(
        corrupt,
        "quality checkpoint repeats retained event arrival sequence",
    );

    // A pending receipt with every item resolved belongs in terminal retention,
    // even if its final evidence otherwise has valid channel shape.
    let mut corrupt: Value = serde_json::from_slice(&encoded).unwrap();
    let mixed = mixed_pending_record(&mut corrupt);
    let completed_values = mixed["items"][0]["values"].clone();
    mixed["items"][1]["ticket"] = Value::Null;
    mixed["items"][1]["status"] = json!("Completed");
    mixed["items"][1]["values"] = completed_values;
    mixed["items"][1]["missing"] = json!([]);
    mixed["items"][1]["missing_reasons"] = json!({});
    assert_corrupt_restore_error(
        corrupt,
        "quality checkpoint retains an all-closed pending record",
    );

    // A cancelled terminal item is only valid while it has unresolved channels.
    let mut corrupt: Value = serde_json::from_slice(&encoded).unwrap();
    terminal_batch_record(&mut corrupt)["items"][1]["missing_reasons"]
        .as_object_mut()
        .unwrap()
        .insert("execution".to_owned(), json!("not run"));
    assert_corrupt_restore_error(
        corrupt,
        "quality checkpoint closed item is already fully resolved",
    );

    // Whole-decision cancellation cannot retain a completed final-valued item.
    let mut corrupt: Value = serde_json::from_slice(&encoded).unwrap();
    terminal_batch_record(&mut corrupt)["status"] = json!("Cancelled");
    assert_corrupt_restore_error(
        corrupt,
        "quality checkpoint terminal close state is invalid",
    );

    // Quality Router receipts never expose an exact propensity or reasons from
    // the generic exploration vocabulary; either would manufacture OPE input.
    let mut corrupt: Value = serde_json::from_slice(&encoded).unwrap();
    terminal_batch_record(&mut corrupt)["receipt"]["selected"][0]["probability"] =
        json!({"Exact": 0.5});
    assert_corrupt_restore_error(corrupt, "quality checkpoint receipt selections are invalid");

    let mut corrupt: Value = serde_json::from_slice(&encoded).unwrap();
    terminal_batch_record(&mut corrupt)["receipt"]["selected"][0]["reason"] = json!("ExploreFirst");
    assert_corrupt_restore_error(corrupt, "quality checkpoint receipt selections are invalid");

    // Configuration revision follows the Quality policy epoch and its catalogue
    // is immutable. Model and representation revisions remain caller labels.
    let mut corrupt: Value = serde_json::from_slice(&encoded).unwrap();
    corrupt["checkpoint"]["config_revision"] = json!(99);
    assert_corrupt_restore_error(corrupt, "quality checkpoint runtime revisions are invalid");

    let mut corrupt: Value = serde_json::from_slice(&encoded).unwrap();
    corrupt["checkpoint"]["catalogue_revision"] = json!(1);
    assert_corrupt_restore_error(corrupt, "quality checkpoint runtime revisions are invalid");

    let mut corrupt: Value = serde_json::from_slice(&encoded).unwrap();
    terminal_batch_record(&mut corrupt)["receipt"]["config_revision"] = json!(99);
    assert_corrupt_restore_error(corrupt, "quality checkpoint receipt revisions are invalid");

    let mut corrupt: Value = serde_json::from_slice(&encoded).unwrap();
    terminal_batch_record(&mut corrupt)["receipt"]["catalogue_revision"] = json!(1);
    assert_corrupt_restore_error(corrupt, "quality checkpoint receipt revisions are invalid");

    let valid: Handoff = serde_json::from_slice(&encoded).unwrap();
    let restored = Muxer::from_quality_checkpoint(valid.checkpoint, BUILD_KEY).unwrap();
    let fresh = Muxer::new(actions(), profile()).unwrap();
    assert!(fresh.engine_id().get() > restored.engine_id().get());

    let duplicate: Handoff = serde_json::from_slice(&encoded).unwrap();
    assert!(Muxer::from_quality_checkpoint(duplicate.checkpoint, BUILD_KEY).is_err());
}

fn assert_corrupt_restore_error(corrupt: Value, expected: &str) {
    let corrupt: Handoff = serde_json::from_value(corrupt).unwrap();
    let error = match Muxer::from_quality_checkpoint(corrupt.checkpoint, BUILD_KEY) {
        Ok(_) => panic!("corrupt checkpoint restored"),
        Err(error) => error,
    };
    assert!(error.to_string().contains(expected), "{error}");
}

fn first_pending_ledger_event(checkpoint: &mut Value) -> &mut Value {
    checkpoint["checkpoint"]["pending"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find_map(|pending| pending["ledger"]["events"].as_array_mut()?.first_mut())
        .unwrap()
}

fn mixed_pending_record(checkpoint: &mut Value) -> &mut Value {
    checkpoint["checkpoint"]["pending"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|pending| {
            let Some(items) = pending["items"].as_array() else {
                return false;
            };
            items.len() == 2 && items[0]["status"] == "Completed" && items[1]["status"] == "Open"
        })
        .unwrap()
}

fn terminal_batch_record(checkpoint: &mut Value) -> &mut Value {
    checkpoint["checkpoint"]["terminal"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|terminal| {
            let Some(items) = terminal["items"].as_array() else {
                return false;
            };
            terminal["status"] == "Completed"
                && items.len() == 2
                && items[0]["status"] == "Completed"
                && items[1]["status"] == "Cancelled"
        })
        .unwrap()
}
