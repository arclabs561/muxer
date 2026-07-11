//! Router quickstart: routing modes and state management.
//!
//! Shows:
//! 1. Basic select → observe loop (both arms, normal mode).
//! 2. Request-local eligibility: every selection stage stays inside the supplied set.
//! 3. Triage detection: sustained hard failures trigger alarm, switch to triage mode.
//! 4. Evidence reset: acknowledge_change clears current detector evidence.
//! 5. Large-K batch exploration: K=20 arms covered quickly with k=3 per round.
//!
//! Run with:
//!   cargo run --example router_quickstart

use muxer::{Outcome, Router, RouterConfig, TriageSessionConfig};

fn clean() -> Outcome {
    Outcome::success(2, 80)
}

fn degraded() -> Outcome {
    Outcome::failure(2, 300)
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
        assert!(router.observe(&arm, clean()));
    }

    // -----------------------------------------------------------------------
    // 2. Request-local eligibility
    // -----------------------------------------------------------------------
    println!("\n=== 2. Request-local eligibility ===");

    let eligible = vec!["arm-b".to_string()];
    let d = router.select_from(&eligible, 1, 0).unwrap();
    assert_eq!(d.primary(), Some("arm-b"));
    println!("  eligible={eligible:?}, chose={:?}", d.primary().unwrap());

    // -----------------------------------------------------------------------
    // 3. Quality divergence: arm-b accumulates junk
    // -----------------------------------------------------------------------
    println!("\n=== 3. Quality divergence ===");

    for _ in 0..30 {
        assert!(router.observe("arm-b", Outcome::degraded(2, 80)));
        assert!(router.observe("arm-a", clean()));
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
    // 4. Triage mode: hard failures on arm-b trigger CUSUM alarm
    // -----------------------------------------------------------------------
    println!("\n=== 4. Triage mode ===");

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
        assert!(r2.observe("stable", clean()));
        assert!(r2.observe("degraded", clean()));
    }

    // Inject hard failures.
    for _ in 0..30 {
        assert!(r2.observe("degraded", degraded()));
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

    // Clear current CUSUM evidence. This is not rebaselining or incident resolution.
    r2.acknowledge_change("degraded");
    println!("  mode after acknowledge: {:?}", r2.mode());
    assert!(
        !r2.mode().is_triage(),
        "should return to Normal after acknowledge"
    );

    // -----------------------------------------------------------------------
    // 5. Large-K batch exploration
    // -----------------------------------------------------------------------
    println!("\n=== 5. Large-K batch exploration (K=20, k=3) ===");

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
            assert!(rl.observe(arm, clean()));
        }
        rounds += 1;
    }
    println!("  Covered all {n} arms in {rounds} rounds with k=3");
    assert_eq!(seen.len(), n, "all arms should be covered");

    // -----------------------------------------------------------------------
    // 6. Dynamic arm management
    // -----------------------------------------------------------------------
    println!("\n=== 6. Dynamic arm management ===");

    let mut rd = Router::new(
        vec!["old-a".to_string(), "old-b".to_string()],
        RouterConfig::default(),
    )
    .unwrap();
    for _ in 0..20 {
        assert!(rd.observe("old-a", clean()));
        assert!(rd.observe("old-b", clean()));
    }

    rd.add_arm("new-c".to_string()).unwrap();
    let d = rd.select(1, 0);
    println!("  After add_arm('new-c'): chose {:?}", d.primary().unwrap());
    assert_eq!(
        d.primary().unwrap(),
        "new-c",
        "newly added arm should be explored first"
    );

    let _ = rd.remove_arm("old-b");
    for _ in 0..100 {
        let d = rd.select(1, 0);
        assert_ne!(
            d.primary().unwrap(),
            "old-b",
            "removed arm must not be selected"
        );
    }
    println!("  remove_arm('old-b'): never selected again");

    println!("\nAll assertions passed.");
}
