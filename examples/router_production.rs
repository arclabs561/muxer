//! Production routing pattern — monitoring + triage + coverage + control + calibration.
//!
//! This example demonstrates the full production configuration:
//!
//! 1. Calibrate a CUSUM threshold before deployment.
//! 2. Build a `Router` with monitoring, triage, coverage, and control arms.
//! 3. Run a simulated routing loop with gradual quality degradation.
//! 4. Observe triage mode firing, investigate, and acknowledge.
//! 5. Snapshot the router state (for persistence across restarts).
//!
//! Run with:
//!   cargo run --example router_production --features stochastic

#[cfg(not(feature = "stochastic"))]
fn main() {
    eprintln!("This example requires: cargo run --example router_production --features stochastic");
}

#[cfg(feature = "stochastic")]
fn main() {
    use muxer::{
        calibrate_cusum_threshold, Outcome, Router, RouterConfig, RouterSnapshot,
        TriageSessionConfig, suggested_window_cap,
    };
    use rand::{Rng, SeedableRng};

    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

    // -----------------------------------------------------------------------
    // Step 1: Calibrate CUSUM threshold
    // -----------------------------------------------------------------------
    // We want P[alarm within 500 rounds under null] ≤ 0.05.
    // Use a small n_trials for demo speed; production should use >= 2000.
    println!("Calibrating CUSUM threshold (n_trials=500)...");
    let p0 = vec![0.85, 0.05, 0.05, 0.05]; // expected null: 85% ok, 5% each degraded category
    let alts = vec![
        vec![0.40, 0.10, 0.40, 0.10], // hard-junk spike
        vec![0.40, 0.10, 0.10, 0.40], // fail spike
    ];
    let cal = calibrate_cusum_threshold(&p0, &alts, 0.05, 500, 500, 1e-3, 20, 42, false)
        .expect("calibration");
    println!(
        "  threshold = {:.2},  fa_hat = {:.3},  wilson_hi = {:.3}",
        cal.threshold, cal.fa_hat, cal.fa_wilson_hi
    );
    println!("  grid_satisfied: {}", cal.grid_satisfied);

    // -----------------------------------------------------------------------
    // Step 2: Build Router with full production config
    // -----------------------------------------------------------------------
    let arms: Vec<String> = vec!["svc-alpha", "svc-beta", "svc-gamma"]
        .into_iter()
        .map(|s| s.to_string())
        .collect();

    // SW-UCB-derived window: ~500 calls/day, expect change ~once a week → cap ~60.
    let window = suggested_window_cap(500, 1.0 / 7.0);
    println!("\nSuggested window cap: {window}");

    let tcfg = TriageSessionConfig {
        threshold: cal.threshold,
        min_n: 20,
        ..TriageSessionConfig::default()
    };

    let cfg = RouterConfig::default()
        .with_monitoring(400, 80)          // baseline/recent windows
        .with_triage_cfg(tcfg)             // calibrated CUSUM threshold
        .with_coverage(0.05, 3)            // ≥5% of traffic to each arm
        .with_guardrail(2_000.0)           // filter arms with mean latency > 2s
        .with_control(1)                   // 1 random control pick per k=3 batch
        .window_cap(window);

    let mut router = Router::new(arms.clone(), cfg).unwrap();

    // -----------------------------------------------------------------------
    // Step 3: Routing loop — phases
    // -----------------------------------------------------------------------

    // Phase A: healthy baseline (200 rounds).
    println!("\nPhase A: healthy baseline (200 rounds)...");
    for i in 0..200_u64 {
        let d = router.select(3, i);
        for arm in &d.chosen {
            let ok_prob = 0.92;
            let junk_prob = 0.04;
            let ok = rng.random::<f64>() < ok_prob;
            let junk = ok && rng.random::<f64>() < junk_prob;
            router.observe(arm, Outcome { ok, junk, hard_junk: !ok, cost_units: 3, elapsed_ms: rng.random_range(50..150), quality_score: None });
        }
    }
    println!("  total observations: {}", router.total_observations());
    for arm in &arms {
        let s = router.summary(arm);
        println!(
            "  {arm}: calls={} ok_rate={:.3} junk_rate={:.3}",
            s.calls, s.ok_rate(), s.junk_rate()
        );
    }
    println!("  mode: {:?}", router.mode());

    // Phase B: inject regression on svc-beta (50 rounds of hard failures).
    println!("\nPhase B: injecting regression on svc-beta...");
    for i in 200..250_u64 {
        let d = router.select(3, i);
        for arm in &d.chosen {
            let outcome = if arm == "svc-beta" {
                // Inject hard failures on beta.
                Outcome { ok: false, junk: true, hard_junk: true, cost_units: 3, elapsed_ms: 500, quality_score: None }
            } else {
                Outcome { ok: true, junk: false, hard_junk: false, cost_units: 3, elapsed_ms: rng.random_range(50..150), quality_score: None }
            };
            router.observe_with_context(arm, outcome, &[0.5_f64]);
        }
    }

    let mode = router.mode();
    println!("  mode after regression: {mode:?}");
    if mode.is_triage() {
        println!("  alarmed: {:?}", mode.alarmed_arms());
    } else {
        println!("  (CUSUM not yet alarmed — may need more observations or lower threshold)");
    }

    // Phase C: acknowledge and return to normal.
    if router.mode().is_triage() {
        println!("\nPhase C: acknowledging regression...");
        router.acknowledge_all_changes();
        println!("  mode after acknowledge: {:?}", router.mode());
    }

    // -----------------------------------------------------------------------
    // Step 4: Snapshot
    // -----------------------------------------------------------------------
    println!("\nSnapshotting router state...");
    let snap: RouterSnapshot = router.snapshot();
    println!("  arms: {:?}", snap.arms);
    println!("  total_observations: {}", snap.total_observations);

    // Restore from snapshot (simulates process restart).
    let router2 = Router::from_snapshot(snap).unwrap();
    println!("  restored: total_observations = {}", router2.total_observations());
    assert_eq!(router.total_observations(), router2.total_observations());

    for arm in &arms {
        let s1 = router.summary(arm);
        let s2 = router2.summary(arm);
        assert_eq!(s1.calls, s2.calls, "window data should be identical after restore");
    }
    println!("  window data identical after restore ✓");
    println!("  CUSUM state reset after restore (no stale alarms) ✓");
    assert!(!router2.mode().is_triage());

    println!("\nDone.");
}
