use muxer::{select_mab_decide, DecisionNote, MabConfig, Outcome, Window};
use std::collections::BTreeMap;

fn main() {
    // Motivated example: constraints first, then tune trade-offs.
    // Imagine three providers where one is cheaper but sometimes fails,
    // another is fast but sometimes junky, and one is expensive but reliable.
    let arms = vec![
        "cheap".to_string(),
        "fast".to_string(),
        "reliable".to_string(),
    ];

    let mut windows: BTreeMap<String, Window> =
        arms.iter().map(|a| (a.clone(), Window::new(50))).collect();

    // Preload windows with "recent history".
    for _ in 0..45 {
        windows
            .get_mut("cheap")
            .unwrap()
            .push(Outcome::success(1, 450));
    }
    for _ in 0..5 {
        // Treat operational failures as "hard junk" for routing purposes.
        windows
            .get_mut("cheap")
            .unwrap()
            .push(Outcome::failure(1, 200));
    }

    for _ in 0..45 {
        windows
            .get_mut("fast")
            .unwrap()
            .push(Outcome::degraded(2, 250));
    }
    for _ in 0..5 {
        windows
            .get_mut("fast")
            .unwrap()
            .push(Outcome::success(2, 260));
    }

    for _ in 0..50 {
        windows
            .get_mut("reliable")
            .unwrap()
            .push(Outcome::success(4, 650));
    }

    let summaries: BTreeMap<String, _> = windows
        .iter()
        .map(|(k, w)| (k.clone(), w.summary()))
        .collect();

    // First: empirical filters to avoid clearly bad arms when alternatives remain.
    let cfg_constraints = MabConfig {
        max_hard_junk_rate: Some(0.05),
        max_junk_rate: Some(0.10),
        ..MabConfig::default()
    };
    let d1 = select_mab_decide(&arms, &summaries, cfg_constraints);
    eprintln!("constraints-only decision={:?}", d1);

    // Premise: `fast` is junky (45/50 degraded => ~90% junk rate), so the hard
    // constraints (max_junk_rate=0.10, max_hard_junk_rate=0.05) must exclude it
    // from the eligible frontier and it must not be the chosen arm.
    let eligible_excludes_fast = d1.notes.iter().any(|n| {
        matches!(
            n,
            DecisionNote::Constraints { eligible_arms, fallback_used: false }
                if !eligible_arms.iter().any(|a| a == "fast")
        )
    });
    assert!(
        eligible_excludes_fast,
        "expected the constraints to exclude `fast` from the eligible frontier, notes={:?}",
        d1.notes
    );
    assert_ne!(
        d1.chosen, "fast",
        "expected the constrained decision to not choose the junk arm `fast`, got {}",
        d1.chosen
    );

    // Then: tune trade-offs once all candidates are "acceptable".
    let cfg_tradeoffs = MabConfig {
        max_hard_junk_rate: Some(0.20),
        max_junk_rate: Some(0.25),
        ..MabConfig::default()
    }
    .with_cost_weight(0.30)
    .with_latency_weight(0.001)
    .with_junk_weight(1.5)
    .with_hard_junk_weight(2.0);
    let d2 = select_mab_decide(&arms, &summaries, cfg_tradeoffs);
    eprintln!("tradeoffs decision={:?}", d2);
}
