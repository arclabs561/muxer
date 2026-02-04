#[cfg(not(feature = "stochastic"))]
fn main() {
    eprintln!(
        "This example requires: cargo run --example deterministic_router --features stochastic"
    );
}

#[cfg(feature = "stochastic")]
fn main() {
    use muxer::{select_mab_explain, MabConfig, Outcome, Window};
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

        // Simple-ish jitter around means.
        let cost_units = t.mean_cost_units.saturating_add(rng.random_range(0..=2));
        let elapsed_ms = t.mean_latency_ms.saturating_add(rng.random_range(0..=200));

        Outcome {
            ok,
            junk,
            hard_junk,
            cost_units,
            elapsed_ms,
        }
    }

    // Stable arm order matters for deterministic tie-breaks.
    let arms = vec![
        "fast".to_string(),
        "cheap".to_string(),
        "reliable".to_string(),
    ];

    // Underlying "truth" for the simulation.
    let truth: BTreeMap<&'static str, ArmTruth> = BTreeMap::from([
        (
            "fast",
            ArmTruth {
                ok_p: 0.92,
                junk_p: 0.08,
                hard_junk_p: 0.15,
                mean_cost_units: 2,
                mean_latency_ms: 250,
            },
        ),
        (
            "cheap",
            ArmTruth {
                ok_p: 0.88,
                junk_p: 0.06,
                hard_junk_p: 0.10,
                mean_cost_units: 1,
                mean_latency_ms: 450,
            },
        ),
        (
            "reliable",
            ArmTruth {
                ok_p: 0.95,
                junk_p: 0.03,
                hard_junk_p: 0.05,
                mean_cost_units: 3,
                mean_latency_ms: 550,
            },
        ),
    ]);

    // One sliding window per arm.
    let mut windows: BTreeMap<String, Window> =
        arms.iter().map(|a| (a.clone(), Window::new(50))).collect();

    // Heuristic selection knobs. (Tune weights for your system.)
    let cfg = MabConfig {
        exploration_c: 0.7,
        cost_weight: 0.20,
        latency_weight: 0.0015,
        junk_weight: 1.5,
        hard_junk_weight: 2.0,
        ..MabConfig::default()
    };

    let mut rng = StdRng::seed_from_u64(123);
    let mut chosen_counts: BTreeMap<String, u64> = BTreeMap::new();

    for i in 0..2_000u64 {
        let summaries: BTreeMap<String, _> = windows
            .iter()
            .map(|(k, w)| (k.clone(), w.summary()))
            .collect();

        let decision = select_mab_explain(&arms, &summaries, cfg);
        let sel = decision.selection;
        *chosen_counts.entry(sel.chosen.clone()).or_insert(0) += 1;

        let t = *truth
            .get(sel.chosen.as_str())
            .expect("truth missing for arm");
        let o = sample_outcome(&mut rng, t);
        windows
            .get_mut(&sel.chosen)
            .expect("window missing for arm")
            .push(o);

        if i % 250 == 0 {
            eprintln!(
                "t={:4} chosen={} explore_first={} fallback={} frontier={:?} counts={:?}",
                i,
                sel.chosen,
                decision.explore_first,
                decision.constraints_fallback_used,
                sel.frontier,
                chosen_counts
            );
        }
    }
}
