#[cfg(not(feature = "stochastic"))]
fn main() {
    eprintln!("This example requires: cargo run --example end_to_end_router --features stochastic");
}

#[cfg(feature = "stochastic")]
fn main() {
    use muxer::{select_mab_explain, MabConfig, Outcome, StickyConfig, StickyMab, Window};
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;
    use std::collections::BTreeMap;

    #[derive(Clone, Copy)]
    struct ArmTruth {
        ok_p: f64,
        // “true junkiness” might only be known after downstream work.
        junk_p: f64,
        hard_junk_p: f64,
        mean_cost_units: u64,
        mean_latency_ms: u64,
    }

    fn sample_outcome(rng: &mut StdRng, t: ArmTruth) -> (Outcome, bool, bool) {
        let ok = rng.random::<f64>() < t.ok_p;

        // This is discovered later.
        let is_junk = ok && (rng.random::<f64>() < t.junk_p);
        let is_hard = is_junk && (rng.random::<f64>() < t.hard_junk_p);

        let cost_units = t.mean_cost_units.saturating_add(rng.random_range(0..=2));
        let elapsed_ms = t.mean_latency_ms.saturating_add(rng.random_range(0..=250));

        (
            Outcome {
                ok,
                // initial label is unknown (set later)
                junk: false,
                hard_junk: false,
                cost_units,
                elapsed_ms,
            },
            is_junk,
            is_hard,
        )
    }

    // Stable order is part of determinism contract.
    let arms = vec![
        "cheap".to_string(),
        "fast".to_string(),
        "reliable".to_string(),
    ];

    let truth: BTreeMap<&'static str, ArmTruth> = BTreeMap::from([
        (
            "cheap",
            ArmTruth {
                ok_p: 0.88,
                junk_p: 0.03,
                hard_junk_p: 0.05,
                mean_cost_units: 1,
                mean_latency_ms: 520,
            },
        ),
        (
            "fast",
            ArmTruth {
                ok_p: 0.90,
                junk_p: 0.10,
                hard_junk_p: 0.20,
                mean_cost_units: 2,
                mean_latency_ms: 260,
            },
        ),
        (
            "reliable",
            ArmTruth {
                ok_p: 0.96,
                junk_p: 0.02,
                hard_junk_p: 0.05,
                mean_cost_units: 4,
                mean_latency_ms: 680,
            },
        ),
    ]);

    let mut windows: BTreeMap<String, Window> =
        arms.iter().map(|a| (a.clone(), Window::new(80))).collect();

    // Constraints first, then trade-offs.
    let cfg = MabConfig {
        // constrain bad operational behavior
        max_hard_junk_rate: Some(0.10),
        // trade-off tuning
        cost_weight: 0.25,
        latency_weight: 0.0012,
        junk_weight: 1.2,
        hard_junk_weight: 2.0,
        ..MabConfig::default()
    };

    // Add stickiness to reduce flapping.
    let mut sticky = StickyMab::new(StickyConfig {
        min_dwell: 10,
        min_switch_margin: 0.02,
    });

    let mut rng = StdRng::seed_from_u64(123);
    let mut counts: BTreeMap<String, u64> = BTreeMap::new();

    for t in 0..1_200u64 {
        let summaries: BTreeMap<String, _> = windows
            .iter()
            .map(|(k, w)| (k.clone(), w.summary()))
            .collect();

        let base = select_mab_explain(&arms, &summaries, cfg);
        let d = sticky.apply_mab_decide(base);
        let chosen = d.chosen.clone();
        *counts.entry(chosen.clone()).or_insert(0) += 1;

        // Start request: push immediate outcome.
        let tr = *truth.get(chosen.as_str()).expect("missing truth");
        let (o, late_junk, late_hard) = sample_outcome(&mut rng, tr);
        let w = windows.get_mut(&chosen).unwrap();
        w.push(o);

        // Later: downstream parsing/classification updates last event.
        w.set_last_junk_level(late_junk, late_hard);

        if t % 200 == 0 {
            let s = w.summary();
            // Intentionally log a single line per tick (easy to ingest into log pipelines).
            eprintln!(
                "t={:4} decision={:?} dwell={} counts={:?} ok_rate={:.3} junk_rate={:.3} hard_junk_rate={:.3}",
                t,
                d,
                sticky.dwell(),
                counts,
                s.ok_rate(),
                s.junk_rate(),
                s.hard_junk_rate()
            );
        }
    }
}
