//! Property and integration tests for Router.

use muxer::{Outcome, Router, RouterConfig, RouterMode, TriageSessionConfig};
use proptest::prelude::*;

fn clean() -> Outcome {
    Outcome {
        ok: true,
        junk: false,
        hard_junk: false,
        cost_units: 1,
        elapsed_ms: 50,
        quality_score: None,
    }
}

fn bad() -> Outcome {
    Outcome {
        ok: false,
        junk: true,
        hard_junk: true,
        cost_units: 1,
        elapsed_ms: 200,
        quality_score: None,
    }
}

fn arms(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("arm{i}")).collect()
}

// ---------------------------------------------------------------------------
// Property tests
// ---------------------------------------------------------------------------

proptest! {
    /// Router.select always returns arms from the arm set, no duplicates, len <= k.
    #[test]
    fn router_select_invariants(
        n_arms in 1usize..8,
        k in 1usize..6,
        n_obs in 0usize..50,
        seed in any::<u64>(),
    ) {
        let a = arms(n_arms);
        let mut r = Router::new(a.clone(), RouterConfig::default()).unwrap();

        // Populate some observations.
        for i in 0..n_obs {
            let arm = &a[i % n_arms];
            r.observe(arm, clean());
        }

        let d = r.select(k, seed);

        // Chosen arms are a subset.
        for c in &d.chosen {
            prop_assert!(a.contains(c), "chosen {c} not in arms");
        }
        // Uniqueness.
        let mut s = d.chosen.clone();
        s.sort();
        s.dedup();
        prop_assert_eq!(s.len(), d.chosen.len(), "chosen must be unique");
        // Bounded by k and arm count.
        prop_assert!(d.chosen.len() <= k.min(n_arms));
        // Prechosen is a subset of chosen.
        for p in &d.prechosen {
            prop_assert!(d.chosen.contains(p), "prechosen {p} not in chosen");
        }
    }

    /// Observe never panics for any outcome/arm combination.
    #[test]
    fn router_observe_never_panics(
        n_arms in 1usize..6,
        ok in any::<bool>(),
        junk in any::<bool>(),
        hard_junk in any::<bool>(),
        cost_units in 0u64..1000,
        elapsed_ms in 0u64..100_000,
    ) {
        let a = arms(n_arms);
        let mut r = Router::new(a.clone(), RouterConfig::default()).unwrap();
        let o = Outcome {
            ok,
            junk: junk || hard_junk,
            hard_junk,
            cost_units,
            elapsed_ms,
            quality_score: None,
        };
        // Observe on a known arm.
        r.observe(&a[0], o);
        // Observe on an unknown arm (should be a no-op, not panic).
        r.observe("nonexistent", o);
    }

    /// Determinism: same seed + same observations → same picks.
    #[test]
    fn router_select_is_deterministic(
        n_arms in 2usize..6,
        n_obs in 0usize..40,
        k in 1usize..4,
        seed in any::<u64>(),
    ) {
        let a = arms(n_arms);
        let mut r = Router::new(a.clone(), RouterConfig::default()).unwrap();
        for i in 0..n_obs {
            r.observe(&a[i % n_arms], clean());
        }
        let d1 = r.select(k, seed);
        let d2 = r.select(k, seed);
        prop_assert_eq!(d1.chosen, d2.chosen, "same seed must give same result");
    }

    /// Mode is Normal when no triage is configured.
    #[test]
    fn router_without_triage_is_always_normal(
        n_arms in 1usize..5,
        n_obs in 0usize..100,
    ) {
        let a = arms(n_arms);
        let mut r = Router::new(a.clone(), RouterConfig::default()).unwrap();
        for i in 0..n_obs {
            r.observe(&a[i % n_arms], bad());
        }
        prop_assert_eq!(r.mode(), RouterMode::Normal, "no triage cfg → always Normal");
    }
}

// ---------------------------------------------------------------------------
// Integration tests
// ---------------------------------------------------------------------------

#[test]
fn router_full_triage_lifecycle() {
    let tcfg = TriageSessionConfig {
        min_n: 10,
        threshold: 3.0,
        ..TriageSessionConfig::default()
    };
    let cfg = RouterConfig::default().with_triage_cfg(tcfg);
    let mut r = Router::new(vec!["good".to_string(), "bad".to_string()], cfg).unwrap();

    // Phase 1: clean baseline.
    for _ in 0..20 {
        r.observe("good", clean());
        r.observe("bad", clean());
    }
    assert_eq!(r.mode(), RouterMode::Normal);

    // Phase 2: inject failures.
    for _ in 0..30 {
        r.observe("bad", bad());
    }
    assert!(r.mode().is_triage());
    assert!(r.mode().alarmed_arms().contains(&"bad".to_string()));

    // Phase 3: investigate — Router routes to alarmed arm.
    let d = r.select(2, 0);
    assert!(
        d.chosen.contains(&"bad".to_string()),
        "triage should prioritize alarmed arm"
    );

    // Phase 4: acknowledge.
    r.acknowledge_change("bad");
    assert_eq!(r.mode(), RouterMode::Normal);

    // Phase 5: back to normal selection.
    let d = r.select(1, 42);
    assert!(["good", "bad"].iter().any(|a| d.primary() == Some(a)));
}

#[test]
fn router_large_k_coverage_with_monitoring() {
    let n = 25;
    let cfg = RouterConfig::default()
        .with_monitoring(200, 50)
        .with_coverage(0.01, 1)
        .window_cap(50);
    let mut r = Router::new(arms(n), cfg).unwrap();

    let mut seen = std::collections::HashSet::new();
    for round in 0..12 {
        let d = r.select(3, round as u64);
        for arm in &d.chosen {
            seen.insert(arm.clone());
            r.observe(arm, clean());
        }
    }
    assert_eq!(
        seen.len(),
        n,
        "K={n} arms should all be explored within 12 rounds (k=3)"
    );
}

#[test]
fn router_delayed_junk_labeling_updates_windows() {
    let mut r = Router::new(arms(2), RouterConfig::default()).unwrap();
    r.observe(
        "arm0",
        Outcome {
            ok: true,
            junk: false,
            hard_junk: false,
            cost_units: 1,
            elapsed_ms: 50,
            quality_score: None,
        },
    );
    // Quality discovered after the call.
    r.set_last_junk_level("arm0", true, false);
    let s = r.summary("arm0");
    assert_eq!(s.junk, 1, "junk should be updated via delayed labeling");
    assert_eq!(s.hard_junk, 0);
}

#[test]
fn router_control_picks_subset_of_chosen() {
    let cfg = RouterConfig::default().with_control(1);
    let r = Router::new(arms(5), cfg).unwrap();
    for _ in 0..50 {
        let d = r.select(3, 42);
        for p in &d.control_picks {
            assert!(d.chosen.contains(p), "control pick must be in chosen");
        }
    }
}

#[test]
fn router_k_larger_than_arm_count_saturates() {
    let r = Router::new(arms(3), RouterConfig::default()).unwrap();
    let d = r.select(10, 0); // k=10 > K=3
    assert!(d.chosen.len() <= 3, "can't pick more arms than exist");
}

#[test]
fn router_summary_tracks_outcomes() {
    let mut r = Router::new(arms(2), RouterConfig::default()).unwrap();
    for _ in 0..10 {
        r.observe("arm0", clean());
    }
    for _ in 0..5 {
        r.observe("arm0", bad());
    }
    let s = r.summary("arm0");
    assert_eq!(s.calls, 15);
    assert_eq!(s.ok, 10);
    assert_eq!(s.hard_junk, 5);
}

// ---------------------------------------------------------------------------
// Snapshot / restore
// ---------------------------------------------------------------------------

#[test]
fn router_snapshot_restores_window_state() {
    let mut r = Router::new(arms(3), RouterConfig::default()).unwrap();
    for _ in 0..15 {
        r.observe("arm0", clean());
        r.observe("arm1", bad());
    }
    let snap = r.snapshot();
    assert_eq!(snap.total_observations, 30);

    let r2 = Router::from_snapshot(snap).unwrap();
    assert_eq!(r2.summary("arm0").calls, 15);
    assert_eq!(r2.summary("arm1").calls, 15);
    assert_eq!(r2.summary("arm1").hard_junk, 15);
    assert_eq!(r2.total_observations(), 30);
}

#[test]
fn router_snapshot_arms_match() {
    let a = arms(4);
    let r = Router::new(a.clone(), RouterConfig::default()).unwrap();
    let snap = r.snapshot();
    assert_eq!(snap.arms, a);
}

#[test]
fn router_snapshot_restores_monitored_windows() {
    let cfg = RouterConfig::default().with_monitoring(200, 50);
    let mut r = Router::new(arms(2), cfg).unwrap();
    for _ in 0..20 {
        r.observe("arm0", clean());
    }
    let snap = r.snapshot();
    let r2 = Router::from_snapshot(snap).unwrap();
    assert!(r2.monitored_window("arm0").is_some());
    assert_eq!(r2.monitored_window("arm0").unwrap().recent_len(), 20);
}

#[test]
fn router_from_snapshot_resets_cusum() {
    let tcfg = TriageSessionConfig {
        min_n: 5,
        threshold: 2.0,
        ..TriageSessionConfig::default()
    };
    let cfg = RouterConfig::default().with_triage_cfg(tcfg);
    let mut r = Router::new(arms(2), cfg).unwrap();
    for _ in 0..10 {
        r.observe("arm0", clean());
    }
    for _ in 0..20 {
        r.observe("arm0", bad());
    }
    assert!(r.mode().is_triage());

    // Snapshot + restore: CUSUM is reset.
    let snap = r.snapshot();
    let r2 = Router::from_snapshot(snap).unwrap();
    assert_eq!(
        r2.mode(),
        RouterMode::Normal,
        "CUSUM should be reset on restore"
    );
    // But window data is preserved.
    assert_eq!(r2.summary("arm0").calls, 30);
}

#[test]
fn router_acknowledge_all_clears_all_alarmed() {
    let tcfg = TriageSessionConfig {
        min_n: 5,
        threshold: 2.0,
        ..TriageSessionConfig::default()
    };
    let cfg = RouterConfig::default().with_triage_cfg(tcfg);
    let mut r = Router::new(arms(3), cfg).unwrap();
    for _ in 0..10 {
        r.observe("arm0", clean());
        r.observe("arm1", clean());
        r.observe("arm2", clean());
    }
    for _ in 0..30 {
        r.observe("arm0", bad());
        r.observe("arm1", bad());
    }
    assert!(r.mode().is_triage());
    r.acknowledge_all_changes();
    assert_eq!(r.mode(), RouterMode::Normal, "all alarms cleared");
}
