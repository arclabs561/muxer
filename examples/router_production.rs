//! Combined routing configuration demo.
//!
//! This example combines several independent primitives:
//!
//! 1. Select a CUSUM threshold from simulated null scores.
//! 2. Build a `Router` with monitoring, triage, coverage, and control arms.
//! 3. Run a simulated routing loop with gradual quality degradation.
//! 4. Observe triage mode firing and reset its current evidence.
//! 5. Snapshot the router state (for persistence across restarts).
//!
//! The threshold helper covers one detector bank in sample time. It is not a
//! system-wide calibration for the Router composition shown below.
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
        calibrate_cusum_threshold, suggested_window_cap, Outcome, Router, RouterConfig,
        RouterSnapshot, TriageSessionConfig,
    };
    use rand::{Rng, SeedableRng};

    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

    // -----------------------------------------------------------------------
    // Step 1: Select an illustrative CUSUM threshold
    // -----------------------------------------------------------------------
    // Target an in-sample null alarm estimate of at most 0.05 over 500 samples.
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
    // Step 2: Build a Router combining the available primitives
    // -----------------------------------------------------------------------
    let arms: Vec<String> = vec!["svc-alpha", "svc-beta", "svc-gamma"]
        .into_iter()
        .map(|s| s.to_string())
        .collect();

    // Rule of thumb: ~500 calls/day, expect change ~once a week → cap ~60.
    let window = suggested_window_cap(500, 1.0 / 7.0);
    println!("\nSuggested window cap: {window}");

    let tcfg = TriageSessionConfig {
        threshold: cal.threshold,
        min_n: 20,
        ..TriageSessionConfig::default()
    };

    let cfg = RouterConfig::default()
        .with_monitoring(400, 80) // baseline/recent windows
        .with_triage_cfg(tcfg) // threshold selected by the single-bank simulation above
        .with_coverage(0.05, 3) // target a 5% empirical share and at least 3 calls
        .with_guardrail(2_000.0) // base-policy mean-latency filter at 2s
        .with_control(1) // 1 seeded comparison pick per k=3 batch
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
            assert!(router.observe(
                arm,
                Outcome::new(ok, junk, !ok, 3, rng.random_range(50..150)),
            ));
        }
    }
    println!("  total observations: {}", router.total_observations());
    for arm in &arms {
        let s = router.summary(arm);
        println!(
            "  {arm}: calls={} ok_rate={:.3} junk_rate={:.3}",
            s.calls,
            s.ok_rate(),
            s.junk_rate()
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
                Outcome::failure(3, 500)
            } else {
                Outcome::success(3, rng.random_range(50..150))
            };
            assert!(router.observe_with_context(arm, outcome, &[0.5_f64]));
        }
    }

    let mode = router.mode();
    println!("  mode after regression: {mode:?}");
    if mode.is_triage() {
        println!("  alarmed: {:?}", mode.alarmed_arms());
    } else {
        println!("  (CUSUM not yet alarmed — may need more observations or lower threshold)");
    }

    // Phase C: clear current alert evidence and return to normal mode.
    if router.mode().is_triage() {
        println!("\nPhase C: clearing current alert evidence...");
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
    println!(
        "  restored: total_observations = {}",
        router2.total_observations()
    );
    assert_eq!(router.total_observations(), router2.total_observations());

    for arm in &arms {
        let s1 = router.summary(arm);
        let s2 = router2.summary(arm);
        assert_eq!(
            s1.calls, s2.calls,
            "window data should be identical after restore"
        );
    }
    println!("  window data identical after restore");
    println!("  CUSUM state starts fresh after restore");
    assert!(!router2.mode().is_triage());

    println!("\nDone.");
}
