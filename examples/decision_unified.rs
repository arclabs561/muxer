use muxer::{select_mab_decide, select_mab_explain, MabConfig, StickyConfig, StickyMab, Summary};
use std::collections::BTreeMap;

fn main() {
    // This example is deterministic-only: it compiles under `--no-default-features`.
    // It demonstrates a single `Decision` record shape (policy output + stickiness).

    let arms = vec!["a".to_string(), "b".to_string()];

    let cfg = MabConfig {
        // Make a visible trade-off so the choice is not a tie.
        cost_weight: 0.2,
        ..MabConfig::default()
    };

    // Synthetic summaries: arm "a" is slightly more successful, arm "b" is cheaper.
    let summaries: BTreeMap<String, Summary> = BTreeMap::from([
        (
            "a".to_string(),
            Summary {
                calls: 100,
                ok: 95,
                junk: 0,
                hard_junk: 0,
                cost_units: 200,      // mean 2.0
                elapsed_ms_sum: 5000, // mean 50ms
                mean_quality_score: None,
            },
        ),
        (
            "b".to_string(),
            Summary {
                calls: 100,
                ok: 93,
                junk: 0,
                hard_junk: 0,
                cost_units: 100,      // mean 1.0
                elapsed_ms_sum: 5000, // mean 50ms
                mean_quality_score: None,
            },
        ),
    ]);

    // Base decision (no stickiness).
    let base = select_mab_decide(&arms, &summaries, cfg);
    eprintln!("base={:?}", base);

    // Stickiness-wrapped decision (still unified).
    let mut sticky = StickyMab::new(StickyConfig {
        min_dwell: 2,
        min_switch_margin: 0.0,
    });
    let d1 = sticky.apply_mab_decide(select_mab_explain(&arms, &summaries, cfg));
    let d2 = sticky.apply_mab_decide(select_mab_explain(&arms, &summaries, cfg));
    let d3 = sticky.apply_mab_decide(select_mab_explain(&arms, &summaries, cfg));

    eprintln!("sticky t=1 {:?} (dwell={})", d1, sticky.dwell());
    eprintln!("sticky t=2 {:?} (dwell={})", d2, sticky.dwell());
    eprintln!("sticky t=3 {:?} (dwell={})", d3, sticky.dwell());
}
