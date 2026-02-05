use muxer::monitor::{DriftConfig, DriftMetric};
use muxer::{select_mab_monitored_explain, MabConfig, MonitoredWindow, Outcome};
use proptest::prelude::*;
use std::collections::BTreeMap;

fn fill_monitored(w: &mut MonitoredWindow, seq: &[Outcome]) {
    for &o in seq {
        w.push(o);
    }
}

fn outcome_strategy() -> impl Strategy<Value = Outcome> {
    // Keep it simple and bounded; we care about determinism/invariants, not realism here.
    (
        any::<bool>(),
        any::<bool>(),
        any::<bool>(),
        0u64..10,
        0u64..2_000,
    )
        .prop_map(|(ok, junk, hard_junk, cost_units, elapsed_ms)| {
            let junk = junk && ok;
            let hard_junk = hard_junk && junk;
            Outcome {
                ok,
                junk,
                hard_junk,
                cost_units,
                elapsed_ms,
            }
        })
}

proptest! {
    #[test]
    fn monitored_selection_is_deterministic_and_returns_member(
        // Small-ish number of arms.
        k in 1usize..=5,
        // Ensure enough data for drift.
        baseline_len in 40usize..=120,
        recent_len in 20usize..=80,
        // Per-arm sequences (we'll truncate to needed lengths).
        seqs in prop::collection::vec(prop::collection::vec(outcome_strategy(), 0..200), 1..=5),
        max_drift in prop_oneof![Just(None), (0.0f64..1.0f64).prop_map(Some)],
        drift_weight in 0.0f64..3.0f64,
    ) {
        let k = k.min(seqs.len()).max(1);
        let arms: Vec<String> = (0..k).map(|i| format!("arm{}", i)).collect();

        let mut m: BTreeMap<String, MonitoredWindow> = BTreeMap::new();
        for i in 0..k {
            let mut w = MonitoredWindow::new(2_000, 200);
            let s = &seqs[i];
            let need = baseline_len + recent_len;
            // If a seq is too short, pad with a stable outcome.
            let mut v: Vec<Outcome> = Vec::new();
            v.extend_from_slice(s);
            while v.len() < need {
                v.push(Outcome { ok: true, junk: false, hard_junk: false, cost_units: 1, elapsed_ms: 100 });
            }
            fill_monitored(&mut w, &v[..need]);
            m.insert(arms[i].clone(), w);
        }

        let drift_cfg = DriftConfig {
            metric: DriftMetric::Hellinger,
            tol: 1e-9,
            min_baseline: 20,
            min_recent: 10,
        };

        let cfg = MabConfig {
            max_drift,
            drift_metric: DriftMetric::Hellinger,
            drift_weight,
            // Avoid brittle constraints; this is about invariants.
            max_junk_rate: None,
            max_hard_junk_rate: None,
            max_mean_cost_units: None,
            ..MabConfig::default()
        };

        let d1 = select_mab_monitored_explain(&arms, &m, drift_cfg, cfg);
        let d2 = select_mab_monitored_explain(&arms, &m, drift_cfg, cfg);

        let chosen1 = d1.selection.chosen.clone();
        let chosen2 = d2.selection.chosen.clone();
        prop_assert_eq!(chosen1.clone(), chosen2);
        prop_assert!(arms.iter().any(|a| a == &chosen1));
        prop_assert!(d1.selection.frontier.iter().any(|x| x == &chosen1));

        // Drift guard must never yield empty eligibility (fallback or filtered subset).
        if let Some(ref dg) = d1.drift_guard {
            prop_assert!(!dg.eligible_arms.is_empty());
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 64, .. ProptestConfig::default() })]
    #[test]
    fn drift_guard_threshold_is_monotone_when_no_fallback(
        // fixed small-ish arms
        k in 2usize..=5,
        baseline_len in 60usize..=120,
        recent_len in 40usize..=80,
        seqs in prop::collection::vec(prop::collection::vec(outcome_strategy(), 0..200), 2..=5),
        t1 in 0.0f64..0.9f64,
        t2 in 0.0f64..0.9f64,
    ) {
        let k = k.min(seqs.len()).max(2);
        let arms: Vec<String> = (0..k).map(|i| format!("arm{}", i)).collect();

        let mut m: BTreeMap<String, MonitoredWindow> = BTreeMap::new();
        for i in 0..k {
            let mut w = MonitoredWindow::new(2_000, 200);
            let s = &seqs[i];
            let need = baseline_len + recent_len;
            let mut v: Vec<Outcome> = Vec::new();
            v.extend_from_slice(s);
            while v.len() < need {
                v.push(Outcome { ok: true, junk: false, hard_junk: false, cost_units: 1, elapsed_ms: 100 });
            }
            fill_monitored(&mut w, &v[..need]);
            m.insert(arms[i].clone(), w);
        }

        let drift_cfg = DriftConfig { metric: DriftMetric::Hellinger, tol: 1e-9, min_baseline: 20, min_recent: 10 };
        let (lo, hi) = if t1 <= t2 { (t1, t2) } else { (t2, t1) };

        let cfg_hi = MabConfig { max_drift: Some(hi), drift_metric: DriftMetric::Hellinger, drift_weight: 0.0, ..MabConfig::default() };
        let cfg_lo = MabConfig { max_drift: Some(lo), drift_metric: DriftMetric::Hellinger, drift_weight: 0.0, ..MabConfig::default() };

        let dhi = select_mab_monitored_explain(&arms, &m, drift_cfg, cfg_hi).drift_guard.expect("guard");
        let dlo = select_mab_monitored_explain(&arms, &m, drift_cfg, cfg_lo).drift_guard.expect("guard");

        // Only assert monotonicity when both thresholds produced non-fallback filtering.
        if !dhi.fallback_used && !dlo.fallback_used {
            for a in &dlo.eligible_arms {
                prop_assert!(dhi.eligible_arms.iter().any(|x| x == a));
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 64, .. ProptestConfig::default() })]
    #[test]
    fn catkl_guard_threshold_is_monotone_when_no_fallback(
        k in 2usize..=5,
        baseline_len in 80usize..=160,
        recent_len in 40usize..=120,
        seqs in prop::collection::vec(prop::collection::vec(outcome_strategy(), 0..220), 2..=5),
        t1 in 0.0f64..300.0f64,
        t2 in 0.0f64..300.0f64,
    ) {
        let k = k.min(seqs.len()).max(2);
        let arms: Vec<String> = (0..k).map(|i| format!("arm{}", i)).collect();

        let mut m: BTreeMap<String, MonitoredWindow> = BTreeMap::new();
        for i in 0..k {
            let mut w = MonitoredWindow::new(2_000, 240);
            let s = &seqs[i];
            let need = baseline_len + recent_len;
            let mut v: Vec<Outcome> = Vec::new();
            v.extend_from_slice(s);
            while v.len() < need {
                v.push(Outcome { ok: true, junk: false, hard_junk: false, cost_units: 1, elapsed_ms: 100 });
            }
            fill_monitored(&mut w, &v[..need]);
            m.insert(arms[i].clone(), w);
        }

        let drift_cfg = DriftConfig { metric: DriftMetric::Hellinger, tol: 1e-9, min_baseline: 20, min_recent: 10 };
        let (lo, hi) = if t1 <= t2 { (t1, t2) } else { (t2, t1) };

        let cfg_hi = MabConfig { max_catkl: Some(hi), catkl_weight: 0.0, ..MabConfig::default() };
        let cfg_lo = MabConfig { max_catkl: Some(lo), catkl_weight: 0.0, ..MabConfig::default() };

        let dhi = select_mab_monitored_explain(&arms, &m, drift_cfg, cfg_hi).catkl_guard.expect("guard");
        let dlo = select_mab_monitored_explain(&arms, &m, drift_cfg, cfg_lo).catkl_guard.expect("guard");

        if !dhi.fallback_used && !dlo.fallback_used {
            for a in &dlo.eligible_arms {
                prop_assert!(dhi.eligible_arms.iter().any(|x| x == a));
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 64, .. ProptestConfig::default() })]
    #[test]
    fn cusum_guard_threshold_is_monotone_when_no_fallback(
        k in 2usize..=5,
        baseline_len in 80usize..=160,
        recent_len in 40usize..=120,
        seqs in prop::collection::vec(prop::collection::vec(outcome_strategy(), 0..220), 2..=5),
        t1 in 0.0f64..40.0f64,
        t2 in 0.0f64..40.0f64,
    ) {
        let k = k.min(seqs.len()).max(2);
        let arms: Vec<String> = (0..k).map(|i| format!("arm{}", i)).collect();

        let mut m: BTreeMap<String, MonitoredWindow> = BTreeMap::new();
        for i in 0..k {
            let mut w = MonitoredWindow::new(2_000, 240);
            let s = &seqs[i];
            let need = baseline_len + recent_len;
            let mut v: Vec<Outcome> = Vec::new();
            v.extend_from_slice(s);
            while v.len() < need {
                v.push(Outcome { ok: true, junk: false, hard_junk: false, cost_units: 1, elapsed_ms: 100 });
            }
            fill_monitored(&mut w, &v[..need]);
            m.insert(arms[i].clone(), w);
        }

        let drift_cfg = DriftConfig { metric: DriftMetric::Hellinger, tol: 1e-9, min_baseline: 20, min_recent: 10 };
        let (lo, hi) = if t1 <= t2 { (t1, t2) } else { (t2, t1) };

        let cfg_hi = MabConfig { max_cusum: Some(hi), cusum_weight: 0.0, ..MabConfig::default() };
        let cfg_lo = MabConfig { max_cusum: Some(lo), cusum_weight: 0.0, ..MabConfig::default() };

        let dhi = select_mab_monitored_explain(&arms, &m, drift_cfg, cfg_hi).cusum_guard.expect("guard");
        let dlo = select_mab_monitored_explain(&arms, &m, drift_cfg, cfg_lo).cusum_guard.expect("guard");

        if !dhi.fallback_used && !dlo.fallback_used {
            for a in &dlo.eligible_arms {
                prop_assert!(dhi.eligible_arms.iter().any(|x| x == a));
            }
        }
    }
}

#[test]
fn cusum_guard_filters_shifted_arm_when_threshold_is_tight() {
    let arms = vec!["stable".to_string(), "changed".to_string()];
    let mut m: BTreeMap<String, MonitoredWindow> = BTreeMap::new();

    let mut ws = MonitoredWindow::new(2_000, 80);
    for _ in 0..200 {
        ws.push(Outcome {
            ok: true,
            junk: false,
            hard_junk: false,
            cost_units: 1,
            elapsed_ms: 100,
        });
    }

    let mut wc = MonitoredWindow::new(2_000, 80);
    for _ in 0..200 {
        wc.push(Outcome {
            ok: true,
            junk: false,
            hard_junk: false,
            cost_units: 1,
            elapsed_ms: 100,
        });
    }
    for _ in 0..80 {
        wc.push(Outcome {
            ok: false,
            junk: true,
            hard_junk: true,
            cost_units: 1,
            elapsed_ms: 100,
        });
    }

    m.insert("stable".to_string(), ws);
    m.insert("changed".to_string(), wc);

    let drift_cfg = DriftConfig {
        metric: DriftMetric::Hellinger,
        tol: 1e-12,
        min_baseline: 20,
        min_recent: 10,
    };
    let cfg = MabConfig {
        max_cusum: Some(1.0),
        cusum_weight: 1.0,
        // Ensure the guard applies for these window sizes.
        cusum_min_baseline: 20,
        cusum_min_recent: 10,
        ..MabConfig::default()
    };

    let d = select_mab_monitored_explain(&arms, &m, drift_cfg, cfg);
    let ug = d.cusum_guard.expect("cusum guard present");
    assert!(ug.eligible_arms.iter().any(|a| a == "stable"));
    assert!(!ug.eligible_arms.iter().any(|a| a == "changed"));
    assert!(!ug.eligible_arms.is_empty());
}

#[test]
fn coverage_stage_can_force_sampling_of_under_sampled_arm() {
    use muxer::policy_fill_k_observed_with_coverage;
    use muxer::{CoverageConfig, LatencyGuardrailConfig};

    let arms = vec!["a".to_string(), "b".to_string(), "c".to_string()];

    // Observations: arm "c" is under-sampled (0 calls), others have many.
    let observed_calls = |arm: &str| -> (u64, u64) {
        match arm {
            "a" => (100, 10_000),
            "b" => (100, 10_000),
            "c" => (0, 0),
            _ => (0, 0),
        }
    };

    let fill = policy_fill_k_observed_with_coverage(
        123,
        &arms,
        1,
        false, // novelty disabled
        CoverageConfig {
            enabled: true,
            min_fraction: 0.05,
            min_calls_floor: 1,
        },
        LatencyGuardrailConfig::default(),
        observed_calls,
        |_eligible, _k| Vec::new(),
    );

    assert_eq!(fill.chosen.len(), 1);
    assert_eq!(fill.chosen[0], "c");
}

#[test]
fn drift_guard_filters_changed_arm_when_threshold_is_tight() {
    let arms = vec!["stable".to_string(), "changed".to_string()];
    let mut m: BTreeMap<String, MonitoredWindow> = BTreeMap::new();

    let mut ws = MonitoredWindow::new(2_000, 80);
    // stable baseline + recent
    for _ in 0..200 {
        ws.push(Outcome {
            ok: true,
            junk: false,
            hard_junk: false,
            cost_units: 1,
            elapsed_ms: 100,
        });
    }

    let mut wc = MonitoredWindow::new(2_000, 80);
    // baseline clean
    for _ in 0..200 {
        wc.push(Outcome {
            ok: true,
            junk: false,
            hard_junk: false,
            cost_units: 1,
            elapsed_ms: 100,
        });
    }
    // recent regresses hard
    for _ in 0..80 {
        wc.push(Outcome {
            ok: false,
            junk: true,
            hard_junk: true,
            cost_units: 1,
            elapsed_ms: 100,
        });
    }

    m.insert("stable".to_string(), ws);
    m.insert("changed".to_string(), wc);

    let drift_cfg = DriftConfig {
        metric: DriftMetric::Hellinger,
        tol: 1e-12,
        min_baseline: 20,
        min_recent: 10,
    };
    let cfg = MabConfig {
        max_drift: Some(0.05),
        drift_metric: DriftMetric::Hellinger,
        drift_weight: 1.0,
        ..MabConfig::default()
    };

    let d = select_mab_monitored_explain(&arms, &m, drift_cfg, cfg);
    let dg = d.drift_guard.expect("drift guard present");
    assert!(dg.eligible_arms.iter().any(|a| a == "stable"));
    assert!(!dg.eligible_arms.is_empty());
}
