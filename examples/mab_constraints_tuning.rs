use muxer::{select_mab, MabConfig, Outcome, Window};
use std::collections::BTreeMap;

fn main() {
    // Motivated example: constraints first, then tune trade-offs.
    // Imagine three providers where one is cheaper but sometimes rate-limits,
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
        windows.get_mut("cheap").unwrap().push(Outcome {
            ok: true,
            http_429: false,
            junk: false,
            hard_junk: false,
            cost_units: 1,
            elapsed_ms: 450,
        });
    }
    for _ in 0..5 {
        windows.get_mut("cheap").unwrap().push(Outcome {
            ok: false,
            http_429: true,
            junk: false,
            hard_junk: false,
            cost_units: 1,
            elapsed_ms: 200,
        });
    }

    for _ in 0..45 {
        windows.get_mut("fast").unwrap().push(Outcome {
            ok: true,
            http_429: false,
            junk: true,
            hard_junk: false,
            cost_units: 2,
            elapsed_ms: 250,
        });
    }
    for _ in 0..5 {
        windows.get_mut("fast").unwrap().push(Outcome {
            ok: true,
            http_429: false,
            junk: false,
            hard_junk: false,
            cost_units: 2,
            elapsed_ms: 260,
        });
    }

    for _ in 0..50 {
        windows.get_mut("reliable").unwrap().push(Outcome {
            ok: true,
            http_429: false,
            junk: false,
            hard_junk: false,
            cost_units: 4,
            elapsed_ms: 650,
        });
    }

    let summaries: BTreeMap<String, _> = windows
        .iter()
        .map(|(k, w)| (k.clone(), w.summary()))
        .collect();

    // First: hard constraints to avoid clearly-bad arms.
    let cfg_constraints = MabConfig {
        max_http_429_rate: Some(0.05),
        max_junk_rate: Some(0.10),
        ..MabConfig::default()
    };
    let sel1 = select_mab(&arms, &summaries, cfg_constraints);
    eprintln!(
        "constraints-only chosen={} frontier={:?}",
        sel1.chosen, sel1.frontier
    );

    // Then: tune trade-offs once all candidates are "acceptable".
    let cfg_tradeoffs = MabConfig {
        max_http_429_rate: Some(0.20),
        max_junk_rate: Some(0.25),
        cost_weight: 0.30,
        latency_weight: 0.001,
        junk_weight: 1.5,
        hard_junk_weight: 2.0,
        ..MabConfig::default()
    };
    let sel2 = select_mab(&arms, &summaries, cfg_tradeoffs);
    eprintln!(
        "tradeoffs chosen={} frontier={:?}",
        sel2.chosen, sel2.frontier
    );
}
