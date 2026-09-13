#![cfg(feature = "serde")]

use muxer::{ObservationId, Outcome, Router, RouterCheckpoint, RouterConfig};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::process::{Command, Stdio};

const BUILD_KEY: &str = "router-checkpoint-test-v1";

fn live_router() -> Router {
    let mut router = Router::new(
        vec!["a".to_owned(), "b".to_owned()],
        RouterConfig::default()
            .with_monitoring(80, 40)
            .with_triage(),
    )
    .unwrap();
    for i in 0..96 {
        let (arm, outcome) = if i % 2 == 0 {
            ("a", Outcome::success(2, 10))
        } else {
            ("b", Outcome::failure(4, 30))
        };
        assert!(router.observe_with_id_and_context(
            ObservationId::new(i),
            arm,
            outcome,
            &[0.2, (i % 3) as f64],
        ));
    }
    assert!(
        router
            .triage_session()
            .unwrap()
            .arm_state("b")
            .unwrap()
            .alarmed
    );
    // Acknowledgement resets detector/recent-window history, not the tracker
    // or baseline; a valid full checkpoint must preserve that asymmetry.
    router.acknowledge_change("a");
    router
}

fn continue_router(mut router: Router) -> Value {
    assert!(router.set_quality_score_for_id(ObservationId::new(95), 0.63));
    let mut trace = Vec::new();
    for i in 100..140 {
        let decision = router.select(2, i);
        let primary = decision.primary().unwrap().to_owned();
        trace.push(serde_json::to_value(&decision).unwrap());
        assert!(router.observe_with_id_and_context(
            ObservationId::new(1000 + i),
            &primary,
            Outcome::with_quality(true, i % 3 == 0, false, 2, 10, 0.75),
            &[0.4, (i % 3) as f64],
        ));
    }
    json!({"trace": trace, "state": router.checkpoint(BUILD_KEY).unwrap()})
}

#[test]
fn complete_checkpoint_crosses_process_boundary_without_resetting_history() {
    let router = live_router();
    let encoded = serde_json::to_vec(&router.checkpoint(BUILD_KEY).unwrap()).unwrap();
    let expected = continue_router(router);

    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "router_checkpoint_child",
            "--ignored",
            "--nocapture",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(&encoded).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let result = stdout
        .lines()
        .find_map(|line| line.strip_prefix("CHECKPOINT_RESULT="))
        .unwrap();
    let actual: Value = serde_json::from_str(result).unwrap();
    assert_eq!(actual, expected);
}

#[test]
#[ignore = "invoked by complete_checkpoint_crosses_process_boundary_without_resetting_history"]
fn router_checkpoint_child() {
    let mut encoded = Vec::new();
    std::io::stdin().read_to_end(&mut encoded).unwrap();
    let checkpoint: RouterCheckpoint = serde_json::from_slice(&encoded).unwrap();
    let router = Router::from_checkpoint(checkpoint, BUILD_KEY).unwrap();
    println!("CHECKPOINT_RESULT={}", continue_router(router));
}

#[test]
fn strict_window_decode_rejects_repairs_allowed_by_legacy_snapshots() {
    let value = serde_json::to_value(live_router().checkpoint(BUILD_KEY).unwrap()).unwrap();
    for field in ["ids", "buf"] {
        let mut malformed = value.clone();
        malformed["windows"]["a"][field]
            .as_array_mut()
            .unwrap()
            .pop();
        assert!(serde_json::from_value::<RouterCheckpoint>(malformed).is_err());
    }
    for score in [json!(-0.1), json!(1.1)] {
        let mut malformed = value.clone();
        malformed["windows"]["a"]["buf"][0]["quality_score"] = score;
        assert!(serde_json::from_value::<RouterCheckpoint>(malformed).is_err());
    }
    let mut malformed = value.clone();
    malformed["windows"]["a"]["buf"][0]["hard_junk"] = json!(true);
    malformed["windows"]["a"]["buf"][0]["junk"] = json!(false);
    assert!(serde_json::from_value::<RouterCheckpoint>(malformed).is_err());

    let mut missing_ids = value;
    missing_ids["windows"]["a"]
        .as_object_mut()
        .unwrap()
        .remove("ids");
    assert!(serde_json::from_value::<RouterCheckpoint>(missing_ids).is_err());
}

#[test]
fn warm_start_still_resets_triage_but_complete_checkpoint_does_not() {
    let router = live_router();
    let restored =
        Router::from_checkpoint(router.checkpoint(BUILD_KEY).unwrap(), BUILD_KEY).unwrap();
    let warm = Router::from_snapshot(router.snapshot()).unwrap();
    assert!(
        restored
            .triage_session()
            .unwrap()
            .arm_state("b")
            .unwrap()
            .alarmed
    );
    assert!(
        !warm
            .triage_session()
            .unwrap()
            .arm_state("b")
            .unwrap()
            .alarmed
    );
    assert!(
        Router::from_checkpoint(router.checkpoint(BUILD_KEY).unwrap(), "different-build").is_err()
    );
}

#[test]
fn removed_arm_discards_only_its_history_and_remains_checkpointable() {
    let mut router = live_router();
    let before_b = router.triage_session().unwrap().arm_state("b").unwrap();
    router.remove_arm("a").unwrap();
    let checkpoint = router.checkpoint(BUILD_KEY).unwrap();
    let mut restored = Router::from_checkpoint(checkpoint, BUILD_KEY).unwrap();
    let after_b = restored.triage_session().unwrap().arm_state("b").unwrap();
    assert_eq!(before_b.n, after_b.n);
    assert_eq!(before_b.score_max, after_b.score_max);
    assert_eq!(before_b.alarmed, after_b.alarmed);
    restored.add_arm("a".to_owned()).unwrap();
    let triage = restored.triage_session().unwrap();
    for bin in triage.tracker().active_bins() {
        assert_eq!(triage.tracker().cell_calls("a", bin), 0);
    }
    assert!(triage.tracker().total_calls() > 0);
    assert!(restored.checkpoint(BUILD_KEY).is_ok());
}

#[test]
fn independently_seeded_warm_histories_are_preserved_without_suffix_repair() {
    let router = live_router();
    let mut snapshot = router.snapshot();
    let mut primary = muxer::Window::new(snapshot.cfg.window_cap);
    primary.push(Outcome::success(2, 10));
    snapshot.windows.insert("b".to_owned(), primary);
    let seeded = Router::from_snapshot(snapshot).unwrap();
    let before = serde_json::to_value(seeded.checkpoint(BUILD_KEY).unwrap()).unwrap();
    let restored =
        Router::from_checkpoint(serde_json::from_value(before.clone()).unwrap(), BUILD_KEY)
            .unwrap();
    assert_eq!(
        serde_json::to_value(restored.checkpoint(BUILD_KEY).unwrap()).unwrap(),
        before
    );
}
