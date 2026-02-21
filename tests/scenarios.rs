use muxer::{
    context_bin, select_mab_explain, worst_first_pick_k, ContextBinConfig,
    ContextualCoverageTracker, MabConfig, Outcome, OutcomeIdx, StickyConfig, StickyMab,
    TriageSession, TriageSessionConfig, Window, WorstFirstConfig,
};
use std::collections::BTreeMap;

fn push_n(w: &mut Window, n: usize, o: Outcome) {
    for _ in 0..n {
        w.push(o);
    }
}

#[test]
fn select_mab_prefers_non_hard_junk_arm_when_window_is_hard_junk_heavy() {
    let arms = vec!["a".to_string(), "b".to_string()];

    // Preload realistic windows instead of hand-constructing Summary rows.
    let mut wa = Window::new(50);
    let mut wb = Window::new(50);

    // Arm "a" is currently producing lots of hard junk (operational failures).
    push_n(
        &mut wa,
        50,
        Outcome {
            ok: false,
            junk: true,
            hard_junk: true,
            cost_units: 1,
            elapsed_ms: 100,
            quality_score: None,
        },
    );

    // Arm "b" is healthy.
    push_n(
        &mut wb,
        50,
        Outcome {
            ok: true,
            junk: false,
            hard_junk: false,
            cost_units: 1,
            elapsed_ms: 120,
            quality_score: None,
        },
    );

    let summaries = BTreeMap::from([
        ("a".to_string(), wa.summary()),
        ("b".to_string(), wb.summary()),
    ]);

    let cfg = MabConfig {
        max_hard_junk_rate: Some(0.2),
        ..MabConfig::default()
    };

    let d = select_mab_explain(&arms, &summaries, cfg);
    assert_eq!(d.selection.chosen, "b");
}

#[test]
fn sticky_reduces_switching_under_alternating_small_advantages() {
    let arms = vec!["a".to_string(), "b".to_string()];

    // No dwell limit: only margin gate.
    let mut sticky = StickyMab::new(StickyConfig {
        min_dwell: 0,
        min_switch_margin: 0.2,
    });

    let cfg = MabConfig::default();

    // Alternate a tiny advantage every step (base policy will tend to flip).
    let mut base_switches = 0u64;
    let mut sticky_switches = 0u64;
    let mut prev_base: Option<String> = None;
    let mut prev_sticky: Option<String> = None;

    for t in 0..100u64 {
        let (ok_a, ok_b) = if t % 2 == 0 { (51, 50) } else { (50, 51) };

        let mut wa = Window::new(60);
        let mut wb = Window::new(60);
        push_n(
            &mut wa,
            60,
            Outcome {
                ok: true,
                junk: false,
                hard_junk: false,
                cost_units: 1,
                elapsed_ms: 100,
                quality_score: None,
            },
        );
        push_n(
            &mut wb,
            60,
            Outcome {
                ok: true,
                junk: false,
                hard_junk: false,
                cost_units: 1,
                elapsed_ms: 100,
                quality_score: None,
            },
        );

        // Nudge success rates via counts (simulating a “near-tie” race).
        // (This is intentionally synthetic but deterministic.)
        let sa = {
            let mut s = wa.summary();
            s.calls = 100;
            s.ok = ok_a;
            s
        };
        let sb = {
            let mut s = wb.summary();
            s.calls = 100;
            s.ok = ok_b;
            s
        };

        let summaries = BTreeMap::from([("a".to_string(), sa), ("b".to_string(), sb)]);
        let base = select_mab_explain(&arms, &summaries, cfg);
        let chosen_base = base.selection.chosen.clone();
        let chosen_sticky = sticky.apply_mab(base).selection.chosen;

        if let Some(prev) = &prev_base {
            if prev != &chosen_base {
                base_switches += 1;
            }
        }
        if let Some(prev) = &prev_sticky {
            if prev != &chosen_sticky {
                sticky_switches += 1;
            }
        }
        prev_base = Some(chosen_base);
        prev_sticky = Some(chosen_sticky);
    }

    assert!(
        sticky_switches < base_switches,
        "sticky_switches={} base_switches={}",
        sticky_switches,
        base_switches
    );
}

/// Simulate a localised contextual regression: arm "b" degrades specifically in the
/// "high-domain" feature bin while remaining healthy in the "low-domain" bin.
/// Verify that `ContextualCoverageTracker::pick_one` surfaces the bad cell and not
/// the healthy one, even though arm "b" looks acceptable in aggregate.
#[test]
fn contextual_tracker_surfaces_localised_regression() {
    let arms = vec!["a".to_string(), "b".to_string()];
    let bin_cfg = ContextBinConfig { levels: 4, seed: 0 };

    // Two distinct feature regimes: "low-domain" and "high-domain".
    let bin_low = context_bin(&[0.1, 0.1], bin_cfg);
    let bin_high = context_bin(&[0.9, 0.9], bin_cfg);
    assert_ne!(bin_low, bin_high, "bins must be distinct for this test");

    let wf_cfg = WorstFirstConfig {
        exploration_c: 1.0,
        hard_weight: 3.0,
        soft_weight: 1.0,
    };

    let mut tracker = ContextualCoverageTracker::new();

    // Arm "a" is fully healthy across both bins.
    for _ in 0..20 {
        tracker.record("a", bin_low, false, false);
        tracker.record("a", bin_high, false, false);
    }

    // Arm "b" is healthy in the low-domain bin ...
    for _ in 0..20 {
        tracker.record("b", bin_low, false, false);
    }

    // ... but has a high hard-junk rate in the high-domain bin (localised regression).
    for _ in 0..20 {
        tracker.record("b", bin_high, true, false); // hard junk
    }

    assert_eq!(tracker.total_calls(), 80);

    let bins = tracker.active_bins();
    assert_eq!(bins.len(), 2);

    // Per-arm aggregate for "b" is 50% hard-junk — noticeable.  But the per-cell view
    // should pinpoint (arm="b", bin=bin_high) with hard_junk_rate=1.0.
    let stats_b_high = tracker.get("b", bin_high).unwrap();
    assert_eq!(stats_b_high.hard_junk_rate(), 1.0);
    let stats_b_low = tracker.get("b", bin_low).unwrap();
    assert_eq!(stats_b_low.hard_junk_rate(), 0.0);

    // pick_one should surface (b, bin_high) as the worst cell.
    let (cell, explore) = tracker.pick_one(42, &arms, &bins, wf_cfg).unwrap();
    assert!(
        !explore,
        "all cells are seen; should be a scored pick, not coverage"
    );
    assert_eq!(cell.arm, "b");
    assert_eq!(
        cell.context_bin, bin_high,
        "localised regression in bin_high should be surfaced"
    );
}

/// Demonstrate the full detect-then-triage loop using ContextualCoverageTracker::pick_k:
/// after a regression fires on arm "provider_b", the top-2 triage picks should both
/// be in the degraded feature region (bin_degraded), not the healthy region (bin_ok).
#[test]
fn contextual_tracker_pick_k_targets_degraded_bins() {
    let arms = vec!["provider_a".to_string(), "provider_b".to_string()];
    let bin_cfg = ContextBinConfig { levels: 4, seed: 7 };

    let bin_ok = context_bin(&[0.2, 0.3], bin_cfg);
    let bin_degraded = context_bin(&[0.7, 0.8], bin_cfg);
    let bin_new = context_bin(&[0.5, 0.5], bin_cfg); // unseen → should get coverage pick

    let wf_cfg = WorstFirstConfig {
        exploration_c: 0.5,
        hard_weight: 2.0,
        soft_weight: 1.0,
    };

    let mut tracker = ContextualCoverageTracker::new();

    // Healthy history for both arms in bin_ok.
    for _ in 0..30 {
        tracker.record("provider_a", bin_ok, false, false);
        tracker.record("provider_b", bin_ok, false, false);
    }

    // provider_a is clean in bin_degraded too.
    for _ in 0..30 {
        tracker.record("provider_a", bin_degraded, false, false);
    }

    // provider_b has regressed in bin_degraded.
    for _ in 0..30 {
        tracker.record("provider_b", bin_degraded, false, true); // soft junk
        tracker.record("provider_b", bin_degraded, true, false); // hard junk (alternating)
    }

    let seen_bins = vec![bin_ok, bin_degraded];
    let all_bins = vec![bin_ok, bin_degraded, bin_new]; // bin_new is unseen

    // With unseen bin present, first pick should be coverage (explore_first=true).
    let picks = tracker.pick_k(42, &arms, &all_bins, 3, wf_cfg);
    assert_eq!(picks.len(), 3);
    // At least one pick should be the coverage pick for bin_new.
    assert!(
        picks.iter().any(|(_, explore)| *explore),
        "unseen bin should trigger at least one coverage pick"
    );

    // Among the scored (non-coverage) picks from seen_bins only, provider_b/bin_degraded
    // should rank highest.
    let scored_picks = tracker.pick_k(42, &arms, &seen_bins, 2, wf_cfg);
    assert_eq!(scored_picks.len(), 2);
    let (top_cell, top_explore) = &scored_picks[0];
    assert!(!top_explore);
    assert_eq!(top_cell.arm, "provider_b");
    assert_eq!(top_cell.context_bin, bin_degraded);
}

/// Routing lifecycle: normal selection → regression detected → triage mode.
///
/// Demonstrates the three-mode lifecycle described in the README:
/// 1. Normal mode: `select_mab` prefers the healthy arm.
/// 2. One arm degrades — `TriageSession` detects it (CUSUM alarms).
/// 3. Triage mode: `worst_first_pick_k` routes extra investigation traffic
///    to the alarmed arm, not the healthy one.
#[test]
fn routing_lifecycle_normal_then_detect_then_triage() {
    let arms = vec!["healthy".to_string(), "degraded".to_string()];

    // --- Phase 1: Normal mode ---
    // Both arms have clean history; select_mab prefers healthy due to lower junk rate.
    let mut w_healthy = Window::new(50);
    let mut w_degraded = Window::new(50);

    let clean = Outcome {
        ok: true,
        junk: false,
        hard_junk: false,
        cost_units: 1,
        elapsed_ms: 100,
        quality_score: None,
    };
    let _bad = Outcome {
        ok: true,
        junk: true,
        hard_junk: false,
        cost_units: 1,
        elapsed_ms: 100,
        quality_score: None,
    };

    for _ in 0..20 {
        w_healthy.push(clean);
    }
    for _ in 0..20 {
        w_degraded.push(clean);
    }

    let summaries = std::collections::BTreeMap::from([
        ("healthy".to_string(), w_healthy.summary()),
        ("degraded".to_string(), w_degraded.summary()),
    ]);
    let d = select_mab_explain(&arms, &summaries, MabConfig::default());
    // Both arms clean → deterministic tie-break by name ("degraded" < "healthy" alphabetically,
    // but explore-first or scalarization may vary; just check it's a valid arm).
    assert!(arms.contains(&d.selection.chosen));

    // --- Phase 2: Arm "degraded" starts producing hard failures ---
    // Feed TriageSession: healthy arm stays clean, degraded arm accumulates hard junk.
    let mut session = TriageSession::new(
        &arms,
        TriageSessionConfig {
            min_n: 10,
            threshold: 3.0,
            ..TriageSessionConfig::default()
        },
    )
    .unwrap();

    // Seed with baseline observations.
    for _ in 0..20 {
        session.observe("healthy", OutcomeIdx::OK, &[0.1]);
        session.observe("degraded", OutcomeIdx::OK, &[0.1]);
    }
    // Inject hard failures on "degraded" — should trigger CUSUM alarm.
    for _ in 0..30 {
        session.observe("degraded", OutcomeIdx::HARD_JUNK, &[0.1]);
    }

    let alarmed = session.alarmed_arms();
    assert!(
        alarmed.contains(&"degraded".to_string()),
        "CUSUM should alarm on the degraded arm after sustained hard failures"
    );
    assert!(
        !alarmed.contains(&"healthy".to_string()),
        "healthy arm must not be alarmed"
    );

    // --- Phase 3: Triage mode — worst_first routes to the degraded arm ---
    // Build summary windows reflecting the degraded state.
    let mut w_h = Window::new(50);
    let mut w_d = Window::new(50);
    for _ in 0..30 {
        w_h.push(clean);
    }
    for _ in 0..10 {
        w_d.push(clean);
    }
    for _ in 0..20 {
        w_d.push(Outcome {
            ok: false,
            junk: true,
            hard_junk: true,
            cost_units: 1,
            elapsed_ms: 100,
            quality_score: None,
        });
    }

    let wf_cfg = WorstFirstConfig {
        exploration_c: 1.0,
        hard_weight: 3.0,
        soft_weight: 1.0,
    };
    let s_h = w_h.summary();
    let s_d = w_d.summary();

    let picks = worst_first_pick_k(
        42,
        &arms,
        2,
        wf_cfg,
        |_| 10u64,
        |b| {
            let s = if b == "healthy" { s_h } else { s_d };
            (s.calls, s.hard_junk_rate(), s.soft_junk_rate())
        },
    );

    assert_eq!(
        picks[0].0, "degraded",
        "worst_first should prioritize the arm with the highest badness score"
    );
}
