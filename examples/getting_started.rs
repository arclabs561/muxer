//! Getting Started — 3 backends in under 60 lines.
//!
//! No CUSUM thresholds.  No KL divergence.  No Pareto frontiers.
//! Just: create a Router → pick an arm → make a call → score it → observe.
//!
//! This covers the 80% case: a small number of arms (here 3), deterministic
//! routing from caller-defined outcomes, plus automatic explore-first for new arms.
//!
//! Run with:
//!   cargo run --example getting_started

use muxer::{Outcome, Router, RouterConfig};

fn main() {
    // -----------------------------------------------------------------
    // 1. Create a Router for 3 backends.
    // -----------------------------------------------------------------
    let arms = vec![
        "gpt-4o".to_string(),
        "claude-sonnet".to_string(),
        "gemini-pro".to_string(),
    ];

    // RouterConfig::default() gives you:
    //   - 100-outcome sliding window per arm
    //   - explore-first (each arm tried before exploitation starts)
    //   - deterministic UCB selection (same config → same picks)
    //
    // For throughput-aware window sizing:
    //   let cap = muxer::suggested_window_cap(500, 0.14); // ~500 calls/day, weekly change
    //   RouterConfig::default().window_cap(cap)
    let mut router = Router::new(arms, RouterConfig::default()).unwrap();

    // Simulated profiles: quality, junk, cost, and latency.
    let profile_for = |arm: &str| -> (f64, bool, u64, u64) {
        match arm {
            "gpt-4o" => (0.92, false, 8, 300), // higher quality, higher cost
            "claude-sonnet" => (0.78, false, 5, 200), // lower quality, lower cost
            "gemini-pro" => (0.55, true, 3, 150), // lower quality, frequent junk
            _ => (1.0, false, 0, 0),
        }
    };

    // -----------------------------------------------------------------
    // 2. Routing loop.
    // -----------------------------------------------------------------
    println!("=== First 6 rounds (explore-first) ===");
    for round in 0..30_u64 {
        // Pick 1 arm. Use k=2 or k=3 for faster initial coverage of many arms.
        let decision = router.select(1, round);
        let arm = decision.primary().unwrap().to_string();

        // Score the completed call, then record one finalized outcome.
        let (quality, is_junk, cost, elapsed_ms) = profile_for(&arm);
        let outcome = Outcome::with_quality(true, is_junk, false, cost, elapsed_ms, quality);
        assert!(router.observe(&arm, outcome));

        if round < 6 {
            println!("  round {round:2}: chose {:15}  quality={quality:.2}  junk={is_junk}  prechosen={:?}",
                     arm, decision.prechosen);
        }
    }

    // -----------------------------------------------------------------
    // 3. Inspect results.
    // -----------------------------------------------------------------
    println!("\n=== After 30 rounds ===");
    for arm in router.arms() {
        let s = router.summary(arm);
        let q = s
            .mean_quality_score
            .map(|v| format!("{v:.2}"))
            .unwrap_or("—".into());
        println!(
            "  {:15}  calls={:2}  ok={:.2}  junk={:.2}  quality={}",
            arm,
            s.calls,
            s.ok_rate(),
            s.junk_rate(),
            q
        );
    }

    // Select from the current window statistics and configured objectives.
    let best = router.select(1, 99);
    println!(
        "\nBest arm now: {:?}  (mode: {:?})",
        best.primary().unwrap(),
        router.mode()
    );
    // gpt-4o and claude-sonnet are both Pareto-efficient: one has higher
    // quality, the other lower cost and latency. Their default scalar scores
    // tie, so the selector uses its stable name tiebreak.

    // -----------------------------------------------------------------
    // 4. Give quality non-zero scalar weight to resolve the tradeoff.
    // -----------------------------------------------------------------
    use muxer::{select_mab, MabConfig};
    use std::collections::BTreeMap;

    let cfg = MabConfig {
        exploration_c: 0.0,
        ..MabConfig::default()
    }
    .with_quality_weight(1.0);
    let summaries: BTreeMap<String, _> = router
        .arms()
        .iter()
        .map(|a| (a.clone(), router.summary(a)))
        .collect();
    let sel = select_mab(router.arms(), &summaries, cfg);
    println!("\nWith quality_weight=1.0: best arm is {:?}", sel.chosen);
    // "gpt-4o" wins because mean_quality_score=0.92 > claude-sonnet's 0.78.

    // -----------------------------------------------------------------
    // 5. Where to go next.
    // -----------------------------------------------------------------
    // - Add monitoring + triage: RouterConfig::default().with_monitoring(400, 80).with_triage()
    // - Use select(k=3) for faster initial coverage of 20+ arms.
    // - Combined configuration demo: cargo run --example router_production --features stochastic
}
