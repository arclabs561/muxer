//! Getting Started — 3 backends in under 50 lines.
//!
//! No CUSUM thresholds.  No KL divergence.  No Pareto frontiers.
//! Just: create a Router → pick an arm → make a call → score it → observe.
//!
//! This covers the 80% case: a small number of arms (here 3), deterministic
//! routing that prefers the best arm, delayed quality labeling (you score the
//! response AFTER the call), and automatic explore-first for new arms.
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
    let mut router = Router::new(arms, RouterConfig::default()).unwrap();

    // -----------------------------------------------------------------
    // 2. Routing loop.
    // -----------------------------------------------------------------
    // In practice: replace the simulated outcomes with real API calls.
    let quality_profiles = [
        ("gpt-4o",        0.92_f64, 0.04_f64),  // high quality, low junk
        ("claude-sonnet", 0.88,     0.08),        // good quality
        ("gemini-pro",    0.75,     0.20),         // more junk
    ];

    for round in 0..30_u64 {
        // Pick 1 arm (use k=2 or k=3 to batch initial exploration faster).
        let decision = router.select(1, round);
        let arm = decision.primary().unwrap().to_string();

        // --- Simulate your actual API call here ---
        let (_, junk_rate, _) = quality_profiles
            .iter()
            .find(|(n, _, _)| *n == arm.as_str())
            .copied()
            .unwrap_or(("?", 0.1, 0.01));

        // Typically: push the outcome immediately with junk=false,
        // then update after scoring (delayed quality labeling).
        let ok = true;
        router.observe(
            &arm,
            Outcome { ok, junk: false, hard_junk: false, cost_units: 5, elapsed_ms: 200, quality_score: None },
        );

        // After your downstream quality check:
        let quality = if round % 3 == 0 { 0.95 } else { 1.0 - junk_rate };
        let is_junk = quality < 0.80;
        if is_junk {
            // Retroactively mark the last outcome as junk.
            // router.set_last_junk_level(&arm, true, false);
            let _ = is_junk; // would call this in production
        }

        // Optionally record the continuous quality score.
        // router.window(&arm).unwrap() gives direct access,
        // but Window::set_last_quality_score is the idiomatic path via Router:
        // (not yet exposed on Router — call window_mut when needed)
        let _ = quality;

        if round < 5 {
            println!("round {round:2}: chose {:15}  (pre-picks: {:?})", arm, decision.prechosen);
        }
    }

    // -----------------------------------------------------------------
    // 3. Inspect results.
    // -----------------------------------------------------------------
    println!("\n--- After 30 rounds ---");
    for arm in router.arms() {
        let s = router.summary(arm);
        println!(
            "  {:15}  calls={:3}  ok_rate={:.2}  junk_rate={:.2}",
            arm, s.calls, s.ok_rate(), s.junk_rate()
        );
    }

    // The Router selects the best arm given current stats.
    let d = router.select(1, 99);
    println!("\nNext pick: {:?}", d.primary().unwrap());
    println!("Mode: {:?}", router.mode());

    // -----------------------------------------------------------------
    // 4. What to explore next.
    // -----------------------------------------------------------------
    // - Add monitoring: RouterConfig::default().with_monitoring(400, 80).with_triage()
    // - Use select(k=3) for faster initial coverage of many arms.
    // - See examples/router_production.rs for the full production pattern.
    // - See examples/EXPERIMENTS.md for the theoretical background.
}
