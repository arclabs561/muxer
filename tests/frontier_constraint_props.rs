//! Property tests for Pareto frontier dominance, constraint fallback, and
//! `Outcome` constructor invariants (hughes lens: high-value property tests).

use muxer::{select_mab, MabConfig, Outcome, Summary};
use proptest::prelude::*;
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// 1. Pareto frontier dominance: every frontier arm weakly dominates at least
//    one non-frontier arm on at least one objective — equivalently, no
//    non-frontier arm strictly dominates a frontier arm.
// ---------------------------------------------------------------------------

proptest! {
    /// No non-frontier arm strictly dominates any frontier arm across all Pareto
    /// objectives (ok_rate, -cost, -latency, -hard_junk_rate, -soft_junk_rate).
    ///
    /// This validates the structural promise of Pareto filtering: the frontier
    /// is a non-dominated set.
    #[test]
    fn frontier_is_non_dominated(
        n_arms in 2usize..7,
        calls in prop::collection::vec(10u64..200, 2..7),
        ok_frac in prop::collection::vec(0.0f64..1.0, 2..7),
        junk_frac in prop::collection::vec(0.0f64..0.5, 2..7),
        hard_junk_frac in prop::collection::vec(0.0f64..0.3, 2..7),
        cost_per_call in prop::collection::vec(1u64..50, 2..7),
        lat_per_call in prop::collection::vec(10u64..500, 2..7),
    ) {
        let n = n_arms.min(calls.len()).min(ok_frac.len()).min(junk_frac.len())
            .min(hard_junk_frac.len()).min(cost_per_call.len()).min(lat_per_call.len());
        let arms: Vec<String> = (0..n).map(|i| format!("arm{i}")).collect();

        let mut summaries = BTreeMap::new();
        for i in 0..n {
            let c = calls[i];
            let ok = ((ok_frac[i]) * c as f64).round() as u64;
            let total_junk = ((junk_frac[i]) * c as f64).round() as u64;
            let hard = ((hard_junk_frac[i]) * c as f64).round().min(total_junk as f64) as u64;
            summaries.insert(format!("arm{i}"), Summary {
                calls: c,
                ok: ok.min(c),
                junk: total_junk.min(c),
                hard_junk: hard.min(total_junk.min(c)),
                cost_units: cost_per_call[i] * c,
                elapsed_ms_sum: lat_per_call[i] * c,
                mean_quality_score: None,
            });
        }

        let cfg = MabConfig { exploration_c: 0.0, ..MabConfig::default() };
        let sel = select_mab(&arms, &summaries, cfg);

        // Compute objective vectors (maximize space).
        let obj = |name: &str| -> Vec<f64> {
            let s = summaries[name];
            let ok_rate = s.ok_rate();
            let mean_cost = s.mean_cost_units();
            let mean_lat = s.mean_elapsed_ms();
            let hard_junk_rate = s.hard_junk_rate();
            let soft_junk_rate = s.soft_junk_rate();
            vec![ok_rate, -mean_cost, -mean_lat, -hard_junk_rate, -soft_junk_rate]
        };

        // For every non-frontier arm, verify it does NOT strictly dominate any
        // frontier arm.
        let non_frontier: Vec<&String> = arms.iter()
            .filter(|a| !sel.frontier.contains(a))
            .collect();

        for nf in &non_frontier {
            let nf_obj = obj(nf);
            for f_arm in &sel.frontier {
                let f_obj = obj(f_arm);
                // "nf strictly dominates f" means nf >= f on all dims AND nf > f on at least one.
                let all_geq = nf_obj.iter().zip(&f_obj).all(|(a, b)| *a >= *b - 1e-12);
                let any_gt = nf_obj.iter().zip(&f_obj).any(|(a, b)| *a > *b + 1e-12);
                prop_assert!(
                    !(all_geq && any_gt),
                    "non-frontier arm {} strictly dominates frontier arm {}: nf={:?} f={:?}",
                    nf, f_arm, nf_obj, f_obj,
                );
            }
        }
    }

    /// The chosen arm is always on the Pareto frontier.
    #[test]
    fn chosen_is_on_frontier(
        n_arms in 1usize..6,
        calls in prop::collection::vec(1u64..200, 1..6),
        ok_frac in prop::collection::vec(0.0f64..1.0, 1..6),
    ) {
        let n = n_arms.min(calls.len()).min(ok_frac.len());
        let arms: Vec<String> = (0..n).map(|i| format!("arm{i}")).collect();

        let mut summaries = BTreeMap::new();
        for i in 0..n {
            let c = calls[i];
            let ok = ((ok_frac[i]) * c as f64).round() as u64;
            summaries.insert(format!("arm{i}"), Summary {
                calls: c,
                ok: ok.min(c),
                junk: 0,
                hard_junk: 0,
                cost_units: c,
                elapsed_ms_sum: c * 100,
                mean_quality_score: None,
            });
        }

        let sel = select_mab(&arms, &summaries, MabConfig::default());
        prop_assert!(
            sel.frontier.contains(&sel.chosen),
            "chosen {} not on frontier {:?}", sel.chosen, sel.frontier
        );
    }
}

// ---------------------------------------------------------------------------
// 2. Constraint fallback: when max_junk_rate is set, either the chosen arm
//    satisfies the constraint OR all arms violate it (fallback behavior).
// ---------------------------------------------------------------------------

proptest! {
    /// When `max_junk_rate` is configured, the chosen arm respects the constraint
    /// unless every arm exceeds it (constraint fallback).
    #[test]
    fn constrained_selection_respects_or_falls_back(
        n_arms in 2usize..6,
        calls in prop::collection::vec(20u64..200, 2..6),
        junk_frac in prop::collection::vec(0.0f64..1.0, 2..6),
        max_junk_rate in 0.05f64..0.95,
    ) {
        let n = n_arms.min(calls.len()).min(junk_frac.len());
        let arms: Vec<String> = (0..n).map(|i| format!("arm{i}")).collect();

        let mut summaries = BTreeMap::new();
        for i in 0..n {
            let c = calls[i];
            let junk = ((junk_frac[i]) * c as f64).round() as u64;
            summaries.insert(format!("arm{i}"), Summary {
                calls: c,
                ok: c.saturating_sub(junk),
                junk: junk.min(c),
                hard_junk: 0,
                cost_units: c,
                elapsed_ms_sum: c * 100,
                mean_quality_score: None,
            });
        }

        let cfg = MabConfig {
            exploration_c: 0.0,
            max_junk_rate: Some(max_junk_rate),
            ..MabConfig::default()
        };
        let sel = select_mab(&arms, &summaries, cfg);

        let chosen_junk_rate = summaries[&sel.chosen].junk_rate();
        let any_eligible = arms.iter().any(|a| summaries[a].junk_rate() <= max_junk_rate);

        if any_eligible {
            prop_assert!(
                chosen_junk_rate <= max_junk_rate + 1e-12,
                "constraint violated: chosen {} has junk_rate={:.4} > max={:.4}",
                sel.chosen, chosen_junk_rate, max_junk_rate
            );
        }
        // If no arms are eligible, fallback selects from the full set -- any arm is valid.
    }
}

// ---------------------------------------------------------------------------
// 3. Outcome::new and Outcome::with_quality enforce hard_junk => junk.
// ---------------------------------------------------------------------------

proptest! {
    /// `Outcome::new` always produces outcomes where `hard_junk => junk`.
    #[test]
    fn outcome_new_enforces_hard_junk_implies_junk(
        ok in any::<bool>(),
        junk in any::<bool>(),
        hard_junk in any::<bool>(),
        cost_units in 0u64..1000,
        elapsed_ms in 0u64..10_000,
    ) {
        let o = Outcome::new(ok, junk, hard_junk, cost_units, elapsed_ms);
        if o.hard_junk {
            prop_assert!(o.junk, "hard_junk=true but junk=false");
        }
    }

    /// `Outcome::with_quality` enforces the same invariant and clamps quality to [0, 1].
    #[test]
    fn outcome_with_quality_enforces_invariants(
        ok in any::<bool>(),
        junk in any::<bool>(),
        hard_junk in any::<bool>(),
        cost_units in 0u64..1000,
        elapsed_ms in 0u64..10_000,
        quality in -1.0f64..2.0,
    ) {
        let o = Outcome::with_quality(ok, junk, hard_junk, cost_units, elapsed_ms, quality);
        if o.hard_junk {
            prop_assert!(o.junk, "hard_junk=true but junk=false");
        }
        let q = o.quality_score.unwrap();
        prop_assert!(
            (0.0..=1.0).contains(&q),
            "quality_score {} not in [0, 1]", q
        );
    }
}
