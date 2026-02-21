//! Property tests and contracts for `muxer`'s structural invariants.
//!
//! These tests enforce the promises made in the module-level documentation:
//!
//! 1. **`hard_junk ⊆ junk`**: for any sequence of outcomes pushed into a `Window`,
//!    `summary.hard_junk <= summary.junk` always holds.
//!
//! 2. **`set_last_junk_level` contract**: setting `hard_junk=true` with `junk=false`
//!    must clear `hard_junk` (the struct enforces `hard_junk && junk`).
//!
//! 3. **Wilson half-width monotonicity**: more observations → tighter confidence
//!    interval.  This is the operational basis for the two-clocks claim: more
//!    sampling → lower estimation uncertainty.
//!
//! 4. **`select_mab` with zero exploration**: when `exploration_c=0`, the selector
//!    reduces to pure ok-rate maximization (no UCB bonus).
//!
//! 5. **Coverage quota contract**: `coverage_pick_under_sampled` only returns arms
//!    that are genuinely below the computed quota.
//!
//! 6. **Summary rate relationships**: `soft_junk_rate = junk_rate - hard_junk_rate`
//!    and `ok_rate + soft_junk_rate + hard_junk_rate` do not need to sum to 1
//!    (they are independent counts), but each rate is in `[0, 1]`.

use muxer::monitor::{apply_rate_bound, RateBoundMode};
use muxer::{coverage_pick_under_sampled, CoverageConfig, MabConfig, Outcome, Summary, Window};
use proptest::prelude::*;
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// 1. hard_junk ⊆ junk (invariant from the Outcome contract)
// ---------------------------------------------------------------------------

fn arb_outcome() -> impl Strategy<Value = Outcome> {
    (
        any::<bool>(),
        any::<bool>(),
        any::<bool>(),
        0u64..100,
        0u64..5_000,
    )
        .prop_map(|(ok, junk, hard_junk, cost_units, elapsed_ms)| Outcome {
            ok,
            // hard_junk implies junk: callers are expected to follow this contract.
            junk: junk || hard_junk,
            hard_junk,
            cost_units,
            elapsed_ms,
            quality_score: None,
        })
}

proptest! {
    /// For any sequence of outcomes pushed into a Window,
    /// `summary.hard_junk <= summary.junk` must hold.
    ///
    /// This is the `hard_junk ⊆ junk` structural invariant from the docs.
    #[test]
    fn hard_junk_never_exceeds_junk_in_summary(
        outcomes in prop::collection::vec(arb_outcome(), 0..200),
        cap in 1usize..64,
    ) {
        let mut w = Window::new(cap);
        for o in &outcomes {
            w.push(*o);
        }
        let s = w.summary();
        prop_assert!(
            s.hard_junk <= s.junk,
            "hard_junk={} > junk={}", s.hard_junk, s.junk
        );
        prop_assert!(s.hard_junk_rate() <= s.junk_rate() + 1e-12,
            "hard_junk_rate={} > junk_rate={}", s.hard_junk_rate(), s.junk_rate());
    }

    /// `soft_junk_rate = junk_rate - hard_junk_rate` — always in [0, 1].
    #[test]
    fn soft_junk_rate_equals_junk_minus_hard(
        outcomes in prop::collection::vec(arb_outcome(), 0..200),
        cap in 1usize..64,
    ) {
        let mut w = Window::new(cap);
        for o in &outcomes {
            w.push(*o);
        }
        let s = w.summary();
        let soft = s.soft_junk_rate();
        let expected = (s.junk_rate() - s.hard_junk_rate()).max(0.0);
        prop_assert!(
            (soft - expected).abs() < 1e-12,
            "soft={soft} expected={expected}"
        );
        prop_assert!((0.0..=1.0 + 1e-12).contains(&soft), "soft_junk_rate out of range: {soft}");
    }

    /// All rates are in [0, 1].
    #[test]
    fn all_rates_in_unit_interval(
        outcomes in prop::collection::vec(arb_outcome(), 0..200),
        cap in 1usize..64,
    ) {
        let mut w = Window::new(cap);
        for o in &outcomes {
            w.push(*o);
        }
        let s = w.summary();
        let eps = 1e-12;
        prop_assert!(s.ok_rate() >= 0.0 && s.ok_rate() <= 1.0 + eps, "ok_rate={}", s.ok_rate());
        prop_assert!(s.junk_rate() >= 0.0 && s.junk_rate() <= 1.0 + eps, "junk_rate={}", s.junk_rate());
        prop_assert!(s.hard_junk_rate() >= 0.0 && s.hard_junk_rate() <= 1.0 + eps, "hard_junk_rate={}", s.hard_junk_rate());
        prop_assert!(s.soft_junk_rate() >= 0.0 && s.soft_junk_rate() <= 1.0 + eps, "soft_junk_rate={}", s.soft_junk_rate());
    }
}

// ---------------------------------------------------------------------------
// 2. set_last_junk_level contract
// ---------------------------------------------------------------------------

#[test]
fn set_last_junk_level_hard_junk_cleared_when_junk_false() {
    // Contract: `set_last_junk_level(false, true)` must NOT set hard_junk,
    // because hard_junk is only meaningful when junk=true.
    let mut w = Window::new(10);
    w.push(Outcome {
        ok: true,
        junk: false,
        hard_junk: false,
        cost_units: 0,
        elapsed_ms: 0,
        quality_score: None,
    });
    w.set_last_junk_level(false, true); // junk=false, hard_junk=true (invalid combo)
    let s = w.summary();
    assert_eq!(s.hard_junk, 0, "hard_junk must be cleared when junk=false");
    assert_eq!(s.junk, 0, "junk must be 0");
}

#[test]
fn set_last_junk_level_both_set_when_both_true() {
    let mut w = Window::new(10);
    w.push(Outcome {
        ok: false,
        junk: false,
        hard_junk: false,
        cost_units: 0,
        elapsed_ms: 0,
        quality_score: None,
    });
    w.set_last_junk_level(true, true);
    let s = w.summary();
    assert_eq!(s.junk, 1, "junk must be 1");
    assert_eq!(s.hard_junk, 1, "hard_junk must be 1");
}

#[test]
fn set_last_junk_level_soft_junk_only() {
    // junk=true but hard_junk=false → soft junk only.
    let mut w = Window::new(10);
    w.push(Outcome {
        ok: true,
        junk: false,
        hard_junk: false,
        cost_units: 0,
        elapsed_ms: 0,
        quality_score: None,
    });
    w.set_last_junk_level(true, false);
    let s = w.summary();
    assert_eq!(s.junk, 1);
    assert_eq!(s.hard_junk, 0, "hard_junk must remain 0 for soft junk");
}

// ---------------------------------------------------------------------------
// 3. Wilson half-width monotonicity
// ---------------------------------------------------------------------------
//
// More observations → tighter confidence interval.
// This is the "two clocks" operational basis: sampling rate controls both
// estimation uncertainty and average detection delay proportionally.

proptest! {
    /// Wilson half-width is (weakly) decreasing when n grows with the success
    /// FRACTION held exactly constant (exact proportional scaling).
    ///
    /// The property "more sampling → tighter interval" holds only when the
    /// observed rate stays fixed — that is, `s_large = s_small * m` and
    /// `n_large = n_small * m` for integer multiplier `m`.  Holding just the
    /// count fixed (different fraction) can widen the interval toward p=0.5.
    ///
    /// This tests the "more sampling → lower estimation uncertainty" claim.
    #[test]
    fn wilson_half_width_decreases_with_proportional_scaling(
        successes in 0u64..50,
        n_small in 5u64..50,
        multiplier in 2u64..8,
        z in prop_oneof![Just(1.0f64), Just(1.645), Just(1.96)],
    ) {
        let s_small = successes.min(n_small);
        let n_large = n_small * multiplier;
        let s_large = s_small * multiplier; // exact same fraction

        for mode in [RateBoundMode::Upper, RateBoundMode::Lower, RateBoundMode::None] {
            let (_, hw_small) = apply_rate_bound(s_small, n_small, z, mode);
            let (_, hw_large) = apply_rate_bound(s_large, n_large, z, mode);
            prop_assert!(
                hw_small >= hw_large - 1e-10,
                "half_width should decrease with proportional scaling: \
                 {s_small}/{n_small} hw={hw_small:.6} vs {s_large}/{n_large} hw={hw_large:.6} (mode={mode:?})"
            );
        }
    }

    /// Wilson half-width is non-negative and finite.
    #[test]
    fn wilson_half_width_is_non_negative_finite(
        successes in 0u64..1000,
        trials in 1u64..1000,
        z in 0.01f64..4.0,
    ) {
        let trials = trials.max(successes);
        for mode in [RateBoundMode::Upper, RateBoundMode::Lower, RateBoundMode::None] {
            let (rate, hw) = apply_rate_bound(successes, trials, z, mode);
            prop_assert!(rate.is_finite() && (0.0..=1.0 + 1e-12).contains(&rate),
                "rate={rate} out of range (successes={successes}, trials={trials})");
            prop_assert!(hw.is_finite() && hw >= 0.0,
                "hw={hw} invalid (successes={successes}, trials={trials})");
        }
    }
}

// ---------------------------------------------------------------------------
// 4. select_mab with exploration_c=0 selects by pure ok_rate
// ---------------------------------------------------------------------------
//
// When exploration_c=0 there is no UCB bonus, so the selector reduces to
// argmax ok_rate (with Pareto filtering on cost/latency/junk as tiebreaks).
// This tests the scalarization contract.

#[test]
fn select_mab_zero_exploration_selects_highest_ok_rate() {
    use muxer::select_mab;
    let arms = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let mut summaries = BTreeMap::new();
    // All arms identical except ok count. Enough calls so exploration doesn't dominate.
    summaries.insert(
        "a".to_string(),
        Summary {
            calls: 100,
            ok: 95,
            junk: 0,
            hard_junk: 0,
            cost_units: 10,
            elapsed_ms_sum: 1000,
            mean_quality_score: None,
        },
    );
    summaries.insert(
        "b".to_string(),
        Summary {
            calls: 100,
            ok: 80,
            junk: 0,
            hard_junk: 0,
            cost_units: 10,
            elapsed_ms_sum: 1000,
            mean_quality_score: None,
        },
    );
    summaries.insert(
        "c".to_string(),
        Summary {
            calls: 100,
            ok: 60,
            junk: 0,
            hard_junk: 0,
            cost_units: 10,
            elapsed_ms_sum: 1000,
            mean_quality_score: None,
        },
    );

    let cfg = MabConfig {
        exploration_c: 0.0,
        ..MabConfig::default()
    };
    let sel = select_mab(&arms, &summaries, cfg);
    assert_eq!(
        sel.chosen, "a",
        "zero exploration: should pick arm with highest ok_rate"
    );
}

#[test]
fn select_mab_zero_exploration_prefers_lower_junk_on_tiebreak() {
    use muxer::select_mab;
    let arms = vec!["a".to_string(), "b".to_string()];
    let mut summaries = BTreeMap::new();
    // Same ok rate, different junk rate.
    summaries.insert(
        "a".to_string(),
        Summary {
            calls: 100,
            ok: 90,
            junk: 0,
            hard_junk: 0,
            cost_units: 10,
            elapsed_ms_sum: 1000,
            mean_quality_score: None,
        },
    );
    summaries.insert(
        "b".to_string(),
        Summary {
            calls: 100,
            ok: 90,
            junk: 5,
            hard_junk: 0,
            cost_units: 10,
            elapsed_ms_sum: 1000,
            mean_quality_score: None,
        },
    );

    let cfg = MabConfig {
        exploration_c: 0.0,
        junk_weight: 1.0,
        ..MabConfig::default()
    };
    let sel = select_mab(&arms, &summaries, cfg);
    assert_eq!(
        sel.chosen, "a",
        "zero exploration + junk_weight: should prefer lower junk"
    );
}

// ---------------------------------------------------------------------------
// 5. Coverage quota contract
// ---------------------------------------------------------------------------

proptest! {
    /// `coverage_pick_under_sampled` only returns arms that are below the quota.
    ///
    /// Quota = max(min_calls_floor, ceil(min_fraction * total_calls)).
    /// If an arm has calls >= quota, it must not be returned as a coverage pick.
    #[test]
    fn coverage_only_returns_arms_below_quota(
        n_arms in 2usize..8,
        total_calls_per_arm in 0u64..200,
        min_fraction in 0.01f64..0.5,
        min_calls_floor in 1u64..20,
        k in 1usize..5,
    ) {
        let arms: Vec<String> = (0..n_arms).map(|i| format!("arm{i}")).collect();
        let cfg = CoverageConfig {
            enabled: true,
            min_fraction,
            min_calls_floor,
        };

        let total: u64 = total_calls_per_arm * n_arms as u64;
        let quota = (min_calls_floor)
            .max((min_fraction * total as f64).ceil() as u64);

        let picks = coverage_pick_under_sampled(
            42,
            &arms,
            k,
            cfg,
            |_arm| total_calls_per_arm, // all arms same call count
        );

        // If all arms are at or above quota, coverage should return nothing.
        if total_calls_per_arm >= quota {
            prop_assert!(
                picks.is_empty(),
                "all arms above quota={quota} but picks={picks:?}"
            );
        }
    }

    /// Coverage picks are always a subset of the input arms.
    #[test]
    fn coverage_picks_are_subset_of_arms(
        n_arms in 1usize..8,
        k in 1usize..5,
        calls in prop::collection::vec(0u64..100, 1..8),
    ) {
        let n_arms = n_arms.min(calls.len());
        let arms: Vec<String> = (0..n_arms).map(|i| format!("arm{i}")).collect();
        let calls: Vec<u64> = calls.into_iter().take(n_arms).collect();

        let cfg = CoverageConfig {
            enabled: true,
            min_fraction: 0.05,
            min_calls_floor: 1,
        };

        let picks = coverage_pick_under_sampled(42, &arms, k, cfg, |arm| {
            let idx: usize = arm[3..].parse().unwrap_or(0);
            *calls.get(idx).unwrap_or(&0)
        });

        for p in &picks {
            prop_assert!(arms.contains(p), "pick {p} not in arms");
        }
        prop_assert!(picks.len() <= k, "picks.len()={} > k={k}", picks.len());
    }
}

// ---------------------------------------------------------------------------
// 6. Quality score signal
// ---------------------------------------------------------------------------

#[test]
fn mean_quality_score_returns_none_with_no_scores() {
    let mut w = Window::new(10);
    for _ in 0..5 {
        w.push(Outcome {
            ok: true,
            junk: false,
            hard_junk: false,
            cost_units: 1,
            elapsed_ms: 50,
            quality_score: None,
        });
    }
    assert!(w.mean_quality_score().is_none());
    let s = w.summary();
    assert!(s.mean_quality_score.is_none());
}

#[test]
fn mean_quality_score_averages_set_scores() {
    let mut w = Window::new(10);
    w.push(Outcome {
        ok: true,
        junk: false,
        hard_junk: false,
        cost_units: 1,
        elapsed_ms: 50,
        quality_score: Some(0.8),
    });
    w.push(Outcome {
        ok: true,
        junk: false,
        hard_junk: false,
        cost_units: 1,
        elapsed_ms: 50,
        quality_score: Some(0.6),
    });
    w.push(Outcome {
        ok: true,
        junk: false,
        hard_junk: false,
        cost_units: 1,
        elapsed_ms: 50,
        quality_score: None,
    });
    let q = w.mean_quality_score().unwrap();
    assert!((q - 0.7).abs() < 1e-10, "mean of 0.8+0.6 = 0.7, got {q}");
    let s = w.summary();
    assert!((s.mean_quality_score.unwrap() - 0.7).abs() < 1e-10);
}

#[test]
fn set_last_quality_score_clamps_and_updates() {
    let mut w = Window::new(5);
    w.push(Outcome {
        ok: true,
        junk: false,
        hard_junk: false,
        cost_units: 0,
        elapsed_ms: 0,
        quality_score: None,
    });
    w.set_last_quality_score(1.5); // clamped to 1.0
    assert_eq!(w.iter().last().unwrap().quality_score, Some(1.0));
    w.set_last_quality_score(-0.5); // clamped to 0.0
    assert_eq!(w.iter().last().unwrap().quality_score, Some(0.0));
}

#[test]
fn quality_weight_influences_arm_selection() {
    use muxer::{select_mab, MabConfig, Summary};
    use std::collections::BTreeMap;

    let arms = vec!["a".to_string(), "b".to_string()];
    let mut summaries = BTreeMap::new();
    // arm "a": same ok_rate, but higher quality score
    summaries.insert(
        "a".to_string(),
        Summary {
            calls: 50,
            ok: 45,
            junk: 0,
            hard_junk: 0,
            cost_units: 5,
            elapsed_ms_sum: 2500,
            mean_quality_score: Some(0.90),
        },
    );
    // arm "b": same ok_rate, lower quality score
    summaries.insert(
        "b".to_string(),
        Summary {
            calls: 50,
            ok: 45,
            junk: 0,
            hard_junk: 0,
            cost_units: 5,
            elapsed_ms_sum: 2500,
            mean_quality_score: Some(0.50),
        },
    );

    let cfg_no_quality = MabConfig {
        exploration_c: 0.0,
        ..MabConfig::default()
    };
    let cfg_with_quality = MabConfig {
        exploration_c: 0.0,
        quality_weight: 1.0,
        ..MabConfig::default()
    };

    // Without quality weight: tie-break by name ("a" < "b" → picks "a" anyway)
    let _ = select_mab(&arms, &summaries, cfg_no_quality);

    // With quality weight: "a" should score higher due to quality bonus.
    let sel = select_mab(&arms, &summaries, cfg_with_quality);
    assert_eq!(
        sel.chosen, "a",
        "arm with higher quality score should be preferred"
    );
}

proptest! {
    /// Higher mean_quality_score on arm A vs arm B, all else equal and with
    /// quality_weight > 0 and exploration_c=0: A should always be chosen.
    #[test]
    fn quality_weight_is_monotone(
        q_a in 0.6f64..1.0,
        q_b in 0.0f64..0.5,
        quality_weight in 0.1f64..2.0,
    ) {
        use muxer::{select_mab, MabConfig, Summary};
        use std::collections::BTreeMap;

        let arms = vec!["a".to_string(), "b".to_string()];
        let mut summaries = BTreeMap::new();
        summaries.insert("a".to_string(), Summary {
            calls: 100, ok: 90, junk: 0, hard_junk: 0, cost_units: 5, elapsed_ms_sum: 5000,
            mean_quality_score: Some(q_a),
        });
        summaries.insert("b".to_string(), Summary {
            calls: 100, ok: 90, junk: 0, hard_junk: 0, cost_units: 5, elapsed_ms_sum: 5000,
            mean_quality_score: Some(q_b),
        });

        let cfg = MabConfig { exploration_c: 0.0, quality_weight, ..MabConfig::default() };
        let sel = select_mab(&arms, &summaries, cfg);
        prop_assert_eq!(&sel.chosen, "a",
            "arm with quality={} should beat arm with quality={} (weight={})",
            q_a, q_b, quality_weight);
    }
}

// ---------------------------------------------------------------------------
// 7. select_mab chosen is always a member of arms (regression guard)
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn select_mab_chosen_is_always_member_of_arms(
        n_arms in 1usize..6,
        calls in prop::collection::vec(0u64..200, 1..6),
        ok_frac in 0.0f64..1.0,
        junk_frac in 0.0f64..0.5,
    ) {
        use muxer::select_mab;

        let n = n_arms.min(calls.len());
        let arms: Vec<String> = (0..n).map(|i| format!("arm{i}")).collect();

        let mut summaries = BTreeMap::new();
        for (i, &c) in calls.iter().take(n).enumerate() {
            let ok = (ok_frac * c as f64).round() as u64;
            let junk = (junk_frac * c as f64).round() as u64;
            summaries.insert(format!("arm{i}"), Summary {
                calls: c,
                ok: ok.min(c),
                junk: junk.min(c),
                hard_junk: 0,
                cost_units: 1,
                elapsed_ms_sum: c * 100,
                mean_quality_score: None,
            });
        }

        let sel = select_mab(&arms, &summaries, MabConfig::default());
        prop_assert!(
            arms.contains(&sel.chosen),
            "chosen={} not in arms={arms:?}", sel.chosen
        );
        prop_assert!(
            sel.frontier.iter().any(|x| x == &sel.chosen),
            "chosen not on frontier"
        );
    }
}
