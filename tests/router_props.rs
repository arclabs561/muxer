//! Property and integration tests for Router.

use muxer::{
    ObservationId, Outcome, PipelineOrder, Router, RouterConfig, RouterMode, TriageSessionConfig,
};
use proptest::prelude::*;

fn clean() -> Outcome {
    Outcome::success(1, 50)
}

fn bad() -> Outcome {
    Outcome::failure(1, 200)
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
            let _ = r.observe(arm, clean());
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
        let o = Outcome::new(ok, junk, hard_junk, cost_units, elapsed_ms);
        // Observe on a known arm.
        let _ = r.observe(&a[0], o);
        // Observe on an unknown arm (should be a no-op, not panic).
        let _ = r.observe("nonexistent", o);
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
            let _ = r.observe(&a[i % n_arms], clean());
        }
        let d1 = r.select(k, seed);
        let d2 = r.select(k, seed);
        prop_assert_eq!(d1.chosen, d2.chosen, "same seed must give same result");
    }

    /// Every Router decision field remains inside the authoritative request-local set.
    #[test]
    fn router_select_from_invariants(
        n_arms in 1usize..8,
        mask in any::<u8>(),
        k in 0usize..8,
        n_obs in 0usize..40,
        seed in any::<u64>(),
    ) {
        let a = arms(n_arms);
        let mut r = Router::new(a.clone(), RouterConfig::default()).unwrap();
        for i in 0..n_obs {
            let _ = r.observe(&a[i % n_arms], clean());
        }

        let mut eligible: Vec<String> = a
            .iter()
            .enumerate()
            .filter(|(index, _)| mask & (1 << index) != 0)
            .map(|(_, arm)| arm.clone())
            .collect();
        eligible.reverse();

        let d = r.select_from(&eligible, k, seed).unwrap();
        for arm in d
            .chosen
            .iter()
            .chain(&d.prechosen)
            .chain(&d.control_picks)
            .chain(&d.mab_eligible)
        {
            prop_assert!(eligible.contains(arm), "decision leaked ineligible arm {arm}");
        }
        for cell in &d.triage_cells {
            prop_assert!(eligible.contains(&cell.arm), "triage leaked ineligible arm {}", cell.arm);
        }

        let mut unique = d.chosen.clone();
        unique.sort();
        unique.dedup();
        prop_assert_eq!(unique.len(), d.chosen.len(), "chosen must be unique");
        prop_assert!(d.chosen.len() <= k.min(eligible.len()));
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
            let _ = r.observe(&a[i % n_arms], bad());
        }
        prop_assert_eq!(r.mode(), RouterMode::Normal, "no triage cfg → always Normal");
    }
}

// ---------------------------------------------------------------------------
// Integration tests
// ---------------------------------------------------------------------------

#[test]
fn router_select_from_rejects_unknown_and_duplicate_arms() {
    let r = Router::new(arms(3), RouterConfig::default()).unwrap();

    let unknown = r.select_from(&["missing".to_string()], 0, 0).unwrap_err();
    assert!(unknown.to_string().contains("registered"));

    let duplicate = r
        .select_from(&["arm1".to_string(), "arm1".to_string()], 1, 0)
        .unwrap_err();
    assert!(duplicate.to_string().contains("unique"));
}

#[test]
fn router_select_from_empty_set_returns_empty_decision() {
    let r = Router::new(arms(3), RouterConfig::default()).unwrap();
    let d = r.select_from(&[], 3, 0).unwrap();
    assert!(d.chosen.is_empty());
    assert!(d.prechosen.is_empty());
    assert!(d.control_picks.is_empty());
    assert!(d.mab_eligible.is_empty());
    assert!(d.triage_cells.is_empty());
}

#[test]
fn router_identified_labels_match_out_of_order_and_reject_duplicate_ids() {
    let first = ObservationId::new(10);
    let second = ObservationId::new(11);
    let mut router = Router::new(arms(1), RouterConfig::default()).unwrap();

    assert!(router.observe_with_id(first, "arm0", Outcome::success(1, 10)));
    assert!(router.observe_with_id(second, "arm0", Outcome::success(1, 10)));
    assert!(!router.observe_with_id(first, "arm0", Outcome::success(1, 10)));

    assert!(router.set_quality_score_for_id(first, 0.9));
    assert!(router.set_quality_score_for_id(second, 0.1));
    assert!((router.summary("arm0").mean_quality_score.unwrap() - 0.5).abs() < 1e-12);

    assert!(router.set_junk_level_for_id(first, true, false));
    assert!(router.set_junk_level_for_id(second, false, true));
    let summary = router.summary("arm0");
    assert_eq!(summary.junk, 1);
    assert_eq!(summary.hard_junk, 0);
    assert!(!router.set_quality_score_for_id(ObservationId::new(99), 0.5));
}

#[test]
fn router_rejects_duplicate_id_retained_only_by_monitoring() {
    let first = ObservationId::new(20);
    let second = ObservationId::new(21);
    let mut cfg = RouterConfig::default()
        .window_cap(1)
        .with_monitoring(20, 10);
    cfg.mab.catkl_min_baseline = 20;
    cfg.mab.catkl_min_recent = 10;
    cfg.mab.cusum_min_baseline = 20;
    cfg.mab.cusum_min_recent = 10;
    let mut router = Router::new(arms(1), cfg).unwrap();

    assert!(router.observe_with_id(first, "arm0", clean()));
    assert!(router.observe_with_id(second, "arm0", clean()));

    // The primary window has evicted `first`, but monitoring still retains it.
    assert!(!router.observe_with_id(first, "arm0", clean()));
    assert!(router.set_quality_score_for_id(first, 0.25));
}

#[test]
fn router_select_from_uses_registration_order_and_matches_select_for_all_arms() {
    let registered = arms(4);
    let r = Router::new(registered.clone(), RouterConfig::default()).unwrap();

    let mut reversed = registered.clone();
    reversed.reverse();
    let from = r.select_from(&reversed, 3, 42).unwrap();
    let all = r.select(3, 42);

    assert_eq!(from.chosen, all.chosen);
    assert_eq!(from.prechosen, all.prechosen);
    assert_eq!(from.control_picks, all.control_picks);
    assert_eq!(from.mab_eligible, all.mab_eligible);
    assert_eq!(from.triage_cells, all.triage_cells);
    assert_eq!(from.mode, all.mode);
}

#[test]
fn router_select_from_constrains_control_and_coverage() {
    let mut cfg = RouterConfig::default()
        .with_control(1)
        .with_coverage(0.5, 2);
    cfg.novelty_enabled = false;
    let mut r = Router::new(arms(4), cfg).unwrap();
    for arm in r.arms().to_vec() {
        let _ = r.observe(&arm, Outcome::success(1, 200));
    }

    let eligible = vec!["arm1".to_string(), "arm3".to_string()];
    let d = r.select_from(&eligible, 2, 9).unwrap();

    assert!(!d.control_picks.is_empty());
    assert!(!d.prechosen.is_empty());
    assert_eq!(d.chosen.len(), 2);
    for arm in d
        .chosen
        .iter()
        .chain(&d.prechosen)
        .chain(&d.control_picks)
        .chain(&d.mab_eligible)
    {
        assert!(
            eligible.contains(arm),
            "decision leaked ineligible arm {arm}"
        );
    }
}

#[test]
fn router_select_from_keeps_guardrail_fallback_inside_candidate_set() {
    let eligible = vec!["arm1".to_string()];

    let mut fallback_cfg = RouterConfig::default().with_guardrail(50.0);
    fallback_cfg.novelty_enabled = false;
    let mut fallback = Router::new(arms(2), fallback_cfg).unwrap();
    let _ = fallback.observe("arm1", Outcome::success(1, 200));

    let d = fallback.select_from(&eligible, 1, 0).unwrap();
    assert_eq!(d.chosen, eligible);
    assert_eq!(d.mab_eligible, eligible);

    let mut strict_cfg = RouterConfig::default().with_guardrail(50.0);
    strict_cfg.novelty_enabled = false;
    strict_cfg.pipeline_order = PipelineOrder::GuardrailFirst;
    let mut strict = Router::new(arms(2), strict_cfg).unwrap();
    let _ = strict.observe("arm1", Outcome::success(1, 200));

    let d = strict.select_from(&eligible, 1, 0).unwrap();
    assert!(d.chosen.is_empty());
    assert!(d.mab_eligible.is_empty());
}

#[test]
fn router_select_from_keeps_mab_constraint_fallback_inside_candidate_set() {
    let mut cfg = RouterConfig {
        novelty_enabled: false,
        ..RouterConfig::default()
    };
    cfg.mab.base.max_junk_rate = Some(0.0);
    let mut r = Router::new(arms(2), cfg).unwrap();
    let _ = r.observe("arm0", clean());
    let _ = r.observe("arm1", bad());

    let eligible = vec!["arm1".to_string()];
    let d = r.select_from(&eligible, 1, 0).unwrap();

    assert_eq!(d.chosen, eligible);
    assert_eq!(d.mab_eligible, eligible);
}

#[test]
fn router_select_from_constrains_triage_picks_and_cells() {
    let tcfg = TriageSessionConfig {
        min_n: 5,
        threshold: 2.0,
        ..TriageSessionConfig::default()
    };
    let cfg = RouterConfig::default().with_triage_cfg(tcfg);
    let mut r = Router::new(vec!["excluded".to_string(), "eligible".to_string()], cfg).unwrap();

    for _ in 0..30 {
        let _ = r.observe_with_context("excluded", bad(), &[0.1]);
        let _ = r.observe_with_context("eligible", bad(), &[0.9]);
    }
    assert!(r.mode().is_triage());

    let eligible = vec!["eligible".to_string()];
    let d = r.select_from(&eligible, 1, 0).unwrap();
    assert!(d.mode.alarmed_arms().contains(&"excluded".to_string()));
    assert_eq!(d.chosen, eligible);
    assert!(d.control_picks.iter().all(|arm| arm == "eligible"));
    assert!(d.prechosen.iter().all(|arm| arm == "eligible"));
    assert!(d.mab_eligible.iter().all(|arm| arm == "eligible"));
    assert!(!d.triage_cells.is_empty());
    assert!(d.triage_cells.iter().all(|cell| cell.arm == "eligible"));
}

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
        let _ = r.observe("good", clean());
        let _ = r.observe("bad", clean());
    }
    assert_eq!(r.mode(), RouterMode::Normal);

    // Phase 2: inject failures.
    for _ in 0..30 {
        let _ = r.observe("bad", bad());
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
            let _ = r.observe(arm, clean());
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
    let _ = r.observe("arm0", Outcome::success(1, 50));
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
        let _ = r.observe("arm0", clean());
    }
    for _ in 0..5 {
        let _ = r.observe("arm0", bad());
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
        let _ = r.observe("arm0", clean());
        let _ = r.observe("arm1", bad());
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
        let _ = r.observe("arm0", clean());
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
        let _ = r.observe("arm0", clean());
    }
    for _ in 0..20 {
        let _ = r.observe("arm0", bad());
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
        let _ = r.observe("arm0", clean());
        let _ = r.observe("arm1", clean());
        let _ = r.observe("arm2", clean());
    }
    for _ in 0..30 {
        let _ = r.observe("arm0", bad());
        let _ = r.observe("arm1", bad());
    }
    assert!(r.mode().is_triage());
    r.acknowledge_all_changes();
    assert_eq!(r.mode(), RouterMode::Normal, "all alarms cleared");
}

// ---------------------------------------------------------------------------
// BaRP property: one policy, many trade-offs (arXiv:2510.07429)
// ---------------------------------------------------------------------------

proptest! {
    /// For any two arms where one has the higher ok-rate and the other the
    /// lower cost, the preference (objective weight vector) alone decides the
    /// route: a quality-only weight routes to the higher ok-rate arm, a
    /// cost-only weight routes to the cheaper arm. The per-arm `Summary`
    /// (the learned state) is identical across both calls -- no retraining,
    /// only the preference changes. This is BaRP's preference-tunable
    /// inference, realized by muxer's multi-objective scalarization.
    #[test]
    fn preference_vector_alone_decides_route(
        ok_lo in 5u64..25,
        ok_hi_extra in 1u64..25,
        cost_lo in 1u64..50,
        cost_hi_extra in 1u64..150,
    ) {
        use muxer::{select_mab, Extract, MabConfig, Objective, Summary};
        use std::collections::BTreeMap;

        let calls = 50u64;
        let ok_hi = (ok_lo + ok_hi_extra).min(calls); // strictly higher ok-rate
        let cost_hi = cost_lo + cost_hi_extra; // strictly higher cost
        prop_assume!(ok_hi > ok_lo);

        let arms = vec!["premium".to_string(), "budget".to_string()];
        let mut m = BTreeMap::new();
        let mk = |ok: u64, cost_units: u64| Summary {
            calls,
            ok,
            junk: 0,
            hard_junk: 0,
            cost_units,
            elapsed_ms_sum: calls * 40,
            mean_quality_score: None,
        };
        m.insert("premium".to_string(), mk(ok_hi, cost_hi));
        m.insert("budget".to_string(), mk(ok_lo, cost_lo));

        // Pure-quality preference: only ok-rate is weighted in scalarization.
        let quality_pref = MabConfig {
            exploration_c: 0.0,
            objectives: vec![
                Objective::maximize(Extract::OkRateUcb, 1.0),
                Objective::minimize(Extract::MeanCost, 0.0),
            ],
            ..MabConfig::default()
        };
        // Pure-cost preference: only cost is weighted in scalarization.
        let cost_pref = MabConfig {
            exploration_c: 0.0,
            objectives: vec![
                Objective::maximize(Extract::OkRateUcb, 0.0),
                Objective::minimize(Extract::MeanCost, 1.0),
            ],
            ..MabConfig::default()
        };

        prop_assert_eq!(select_mab(&arms, &m, quality_pref).chosen, "premium");
        prop_assert_eq!(select_mab(&arms, &m, cost_pref).chosen, "budget");
    }
}
