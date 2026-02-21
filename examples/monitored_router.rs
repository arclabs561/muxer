#[cfg(not(feature = "stochastic"))]
fn main() {
    eprintln!("This example requires: cargo run --example monitored_router --features stochastic");
}

#[cfg(feature = "stochastic")]
fn main() {
    use muxer::{
        monitor::{DriftConfig, DriftMetric},
        select_mab_monitored_decide, MabConfig, MonitoredWindow, Outcome,
    };
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;
    use std::collections::BTreeMap;

    #[derive(Clone, Copy)]
    struct ArmTruth {
        ok_p: f64,
        junk_p: f64,
        hard_junk_p: f64,
        mean_cost_units: u64,
        mean_latency_ms: u64,
    }

    fn sample_outcome(rng: &mut StdRng, t: ArmTruth) -> Outcome {
        let ok = rng.random::<f64>() < t.ok_p;
        let junk = ok && (rng.random::<f64>() < t.junk_p);
        let hard_junk = junk && (rng.random::<f64>() < t.hard_junk_p);
        Outcome {
            ok,
            junk,
            hard_junk,
            cost_units: t.mean_cost_units.saturating_add(rng.random_range(0..=2)),
            elapsed_ms: t.mean_latency_ms.saturating_add(rng.random_range(0..=200)),
            quality_score: None,
        }
    }

    // Stable order is part of determinism.
    let arms = vec![
        "fast".to_string(),
        "cheap".to_string(),
        "reliable".to_string(),
    ];

    let truth_before: BTreeMap<&'static str, ArmTruth> = BTreeMap::from([
        (
            "fast",
            ArmTruth {
                ok_p: 0.92,
                junk_p: 0.06,
                hard_junk_p: 0.10,
                mean_cost_units: 2,
                mean_latency_ms: 250,
            },
        ),
        (
            "cheap",
            ArmTruth {
                ok_p: 0.87,
                junk_p: 0.04,
                hard_junk_p: 0.05,
                mean_cost_units: 1,
                mean_latency_ms: 800,
            },
        ),
        (
            "reliable",
            ArmTruth {
                ok_p: 0.96,
                junk_p: 0.02,
                hard_junk_p: 0.05,
                mean_cost_units: 4,
                mean_latency_ms: 650,
            },
        ),
    ]);

    // Change point: "fast" regresses.
    let truth_after_fast: ArmTruth = ArmTruth {
        ok_p: 0.45,
        junk_p: 0.70,
        hard_junk_p: 0.90,
        mean_cost_units: 2,
        mean_latency_ms: 260,
    };

    // Each arm has a baseline and recent window.
    let mut windows: BTreeMap<String, MonitoredWindow> = arms
        .iter()
        .map(|a| (a.clone(), MonitoredWindow::new(2_000, 80)))
        .collect();

    // Drift computation config.
    let drift_cfg = DriftConfig {
        metric: DriftMetric::Hellinger,
        tol: 1e-12,
        min_baseline: 40,
        min_recent: 20,
    };

    // Selection config:
    // - same multi-objective tuning as usual
    // - plus a drift guard and drift penalty
    let cfg = MabConfig {
        // Prefer low latency (so "fast" wins pre-change).
        cost_weight: 0.10,
        latency_weight: 0.0040,
        junk_weight: 1.2,
        hard_junk_weight: 2.0,
        // monitored bits
        max_drift: Some(0.12),
        drift_metric: DriftMetric::Hellinger,
        drift_weight: 2.0,
        ..MabConfig::default()
    };

    let mut rng = StdRng::seed_from_u64(123);
    let mut chosen_counts: BTreeMap<String, u64> = BTreeMap::new();

    for t in 0..2_000u64 {
        let policy = select_mab_monitored_decide(&arms, &windows, drift_cfg, cfg);

        // Maintenance sampling: without *some* coverage, you can't detect changes in unpulled arms.
        // This is intentionally simple/deterministic for the demo.
        let do_maintenance = t % 25 == 0;
        let chosen_actual = if do_maintenance {
            arms[(t as usize / 25) % arms.len()].clone()
        } else {
            policy.chosen.clone()
        };
        *chosen_counts.entry(chosen_actual.clone()).or_insert(0) += 1;

        let tr = if chosen_actual == "fast" && t >= 1_000 {
            truth_after_fast
        } else {
            *truth_before
                .get(chosen_actual.as_str())
                .expect("missing truth for arm")
        };

        let o = sample_outcome(&mut rng, tr);
        windows.get_mut(&chosen_actual).unwrap().push(o);

        if t % 200 == 0 {
            eprintln!(
                "t={:4} chosen_actual={} chosen_policy={} maintenance={} notes={:?} counts={:?}",
                t, chosen_actual, policy.chosen, do_maintenance, policy.notes, chosen_counts
            );
        }
    }
}
