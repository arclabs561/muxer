use muxer::{select_mab_explain, MabConfig, StickyConfig, StickyMab, Summary};
use std::collections::BTreeMap;

fn main() {
    let arms = vec!["a".to_string(), "b".to_string()];

    // Base config prefers higher success; you can also add weights/constraints.
    let mab_cfg = MabConfig::default();

    // Stickiness: stay at least 5 decisions before switching, and require a small margin.
    let mut sticky = StickyMab::new(StickyConfig {
        min_dwell: 5,
        min_switch_margin: 0.01,
    });

    // Two regimes: initially "a" is better, then "b" becomes better.
    for t in 0..30u64 {
        let mut summaries = BTreeMap::new();
        if t < 12 {
            summaries.insert(
                "a".to_string(),
                Summary {
                    calls: 50,
                    ok: 49,
                    http_429: 0,
                    junk: 0,
                    hard_junk: 0,
                    cost_units: 50,
                    elapsed_ms_sum: 5000,
                },
            );
            summaries.insert(
                "b".to_string(),
                Summary {
                    calls: 50,
                    ok: 46,
                    http_429: 0,
                    junk: 0,
                    hard_junk: 0,
                    cost_units: 50,
                    elapsed_ms_sum: 5000,
                },
            );
        } else {
            summaries.insert(
                "a".to_string(),
                Summary {
                    calls: 50,
                    ok: 46,
                    http_429: 0,
                    junk: 0,
                    hard_junk: 0,
                    cost_units: 50,
                    elapsed_ms_sum: 5000,
                },
            );
            summaries.insert(
                "b".to_string(),
                Summary {
                    calls: 50,
                    ok: 49,
                    http_429: 0,
                    junk: 0,
                    hard_junk: 0,
                    cost_units: 50,
                    elapsed_ms_sum: 5000,
                },
            );
        }

        let base = select_mab_explain(&arms, &summaries, mab_cfg);
        let d = sticky.apply_mab_decide(base);

        eprintln!("t={:2} decision={:?} dwell={}", t, d, sticky.dwell());
    }
}
