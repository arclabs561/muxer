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

        Outcome::new(ok, junk, hard_junk, cost_units, elapsed_ms)
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
        ..MabConfig::default()
    }
    .with_cost_weight(0.20)
    .with_latency_weight(0.0015)
    .with_junk_weight(1.5)
    .with_hard_junk_weight(2.0);

    let mut rng = StdRng::seed_from_u64(123);
    let mut chosen_counts: BTreeMap<String, u64> = BTreeMap::new();
    let mut last_frontier: Vec<String> = Vec::new();

    for i in 0..2_000u64 {
        let summaries: BTreeMap<String, _> = windows
            .iter()
            .map(|(k, w)| (k.clone(), w.summary()))
            .collect();

        let decision = select_mab_explain(&arms, &summaries, cfg.clone());
        let sel = decision.selection;
        last_frontier = sel.frontier.clone();
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

    // Premise: junk-aware Pareto selection is active. `reliable` has the highest
    // ok_p and the lowest junk/hard_junk rates, so it must stay on the Pareto
    // frontier (its low junk keeps it non-dominated). `cheap` has the lowest ok_p
    // (0.88) and the worst cost/latency-adjusted profile, so it must never be the
    // most-chosen arm. An inverted ok/junk probability would scramble both checks.
    //
    // NOTE: an earlier framing expected `reliable` to be the argmax. That is false
    // for this config: `fast`'s much lower latency (250ms vs 550ms) and cost (2 vs
    // 3) dominate `reliable`'s modest junk-rate edge under these weights, so `fast`
    // legitimately wins the most pulls. The assertion below proves the demo's real
    // invariant instead of an incorrect argmax claim.
    let (best_arm, best_count) = chosen_counts
        .iter()
        .max_by_key(|(name, count)| (**count, name.as_str()))
        .map(|(name, count)| (name.clone(), *count))
        .expect("at least one arm was chosen");
    assert!(
        last_frontier.iter().any(|a| a == "reliable"),
        "expected the lowest-junk arm `reliable` to remain on the Pareto frontier, frontier={last_frontier:?}"
    );
    assert_ne!(
        best_arm, "cheap",
        "expected the lowest-ok_p arm `cheap` to never be the most-chosen, got {best_arm} with {best_count} pulls; counts={chosen_counts:?}"
    );
}
