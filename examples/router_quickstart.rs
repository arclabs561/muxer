//! Router quickstart — the complete routing lifecycle in one example.
//!
//! Shows:
//! 1. Basic select → observe loop (both arms, normal mode).
//! 2. Triage detection: sustained hard failures trigger alarm, switch to triage mode.
//! 3. Acknowledgment: after investigation, acknowledge_change resets the detector.
//! 4. Large-K batch exploration: K=20 arms covered quickly with k=3 per round.
//!
//! Run with:
//!   cargo run --example router_quickstart

use muxer::{Outcome, Router, RouterConfig, TriageSessionConfig};

fn clean() -> Outcome {
    Outcome {
        ok: true,
        junk: false,
        hard_junk: false,
        cost_units: 2,
        elapsed_ms: 80,
        quality_score: None,
    }
}

fn degraded() -> Outcome {
    Outcome {
        ok: false,
        junk: true,
        hard_junk: true,
        cost_units: 2,
        elapsed_ms: 300,
        quality_score: None,
    }
}

fn main() {
    // -----------------------------------------------------------------------
    // 1. Basic two-arm routing
    // -----------------------------------------------------------------------
    println!("=== 1. Basic two-arm routing ===");

    let arms = vec!["arm-a".to_string(), "arm-b".to_string()];
    let mut router = Router::new(arms, RouterConfig::default()).unwrap();

    // First two calls explore each arm.
    for i in 0..10 {
        let d = router.select(1, i as u64);
        let arm = d.primary().unwrap().to_string();
        println!("  round {i}: chose {arm:?}  prechosen={:?}", d.prechosen);
        router.observe(&arm, clean());
    }

    // -----------------------------------------------------------------------
    // 2. Quality divergence — arm-b accumulates junk
    // -----------------------------------------------------------------------
    println!("\n=== 2. Quality divergence ===");

    for _ in 0..30 {
        router.observe(
            "arm-b",
            Outcome {
                ok: true,
                junk: true,
                hard_junk: false,
                ..clean()
            },
        );
        router.observe("arm-a", clean());
    }

    let d = router.select(1, 99);
    println!(
        "  After arm-b accumulates junk: chose {:?}",
        d.primary().unwrap()
    );
    assert_eq!(
        d.primary().unwrap(),
        "arm-a",
        "should prefer arm-a after arm-b degrades"
    );

    // -----------------------------------------------------------------------
    // 3. Triage mode: hard failures on arm-b trigger CUSUM alarm
    // -----------------------------------------------------------------------
    println!("\n=== 3. Triage mode ===");

    let tcfg = TriageSessionConfig {
        min_n: 10,
        threshold: 3.0,
        ..TriageSessionConfig::default()
    };
    let cfg = RouterConfig::default().with_triage_cfg(tcfg);

    let arms2 = vec!["stable".to_string(), "degraded".to_string()];
    let mut r2 = Router::new(arms2, cfg).unwrap();

    // Seed with clean baseline.
    for _ in 0..20 {
        r2.observe("stable", clean());
        r2.observe("degraded", clean());
    }

    // Inject hard failures.
    for _ in 0..30 {
        r2.observe("degraded", degraded());
    }

    println!("  mode after failures: {:?}", r2.mode());
    assert!(
        r2.mode().is_triage(),
        "should be in triage after hard failures"
    );
    println!("  alarmed arms: {:?}", r2.mode().alarmed_arms());

    // In triage mode, select routes investigation traffic to alarmed arm.
    let d = r2.select(2, 0);
    println!("  triage picks: {:?}", d.chosen);
    println!("  triage cells: {} cells", d.triage_cells.len());

    // Acknowledge: reset CUSUM, promote recent → baseline.
    r2.acknowledge_change("degraded");
    println!("  mode after acknowledge: {:?}", r2.mode());
    assert!(
        !r2.mode().is_triage(),
        "should return to Normal after acknowledge"
    );

    // -----------------------------------------------------------------------
    // 4. Large-K batch exploration
    // -----------------------------------------------------------------------
    println!("\n=== 4. Large-K batch exploration (K=20, k=3) ===");

    let n = 20;
    let arms_large: Vec<String> = (0..n).map(|i| format!("svc-{i:02}")).collect();
    let cfg_large = RouterConfig::default().with_coverage(0.02, 1);
    let mut rl = Router::new(arms_large, cfg_large).unwrap();

    let mut seen = std::collections::HashSet::new();
    let mut rounds = 0;
    while seen.len() < n && rounds < 20 {
        let d = rl.select(3, rounds as u64);
        for arm in &d.chosen {
            seen.insert(arm.clone());
            rl.observe(arm, clean());
        }
        rounds += 1;
    }
    println!("  Covered all {n} arms in {rounds} rounds with k=3");
    assert_eq!(seen.len(), n, "all arms should be covered");

    // -----------------------------------------------------------------------
    // 5. Dynamic arm management
    // -----------------------------------------------------------------------
    println!("\n=== 5. Dynamic arm management ===");

    let mut rd = Router::new(
        vec!["old-a".to_string(), "old-b".to_string()],
        RouterConfig::default(),
    )
    .unwrap();
    for _ in 0..20 {
        rd.observe("old-a", clean());
        rd.observe("old-b", clean());
    }

    rd.add_arm("new-c".to_string()).unwrap();
    let d = rd.select(1, 0);
    println!("  After add_arm('new-c'): chose {:?}", d.primary().unwrap());
    assert_eq!(
        d.primary().unwrap(),
        "new-c",
        "newly added arm should be explored first"
    );

    rd.remove_arm("old-b");
    for _ in 0..100 {
        let d = rd.select(1, 0);
        assert_ne!(
            d.primary().unwrap(),
            "old-b",
            "removed arm must not be selected"
        );
    }
    println!("  remove_arm('old-b'): never selected again ✓");

    println!("\nAll assertions passed.");
}
