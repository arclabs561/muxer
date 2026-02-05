//! Coverage / maintenance sampling helpers.
//!
//! These primitives exist to make “keep sampling every arm” an explicit, testable contract.
//! They are intentionally domain-agnostic: callers supply `observed_calls` however they want
//! (lifetime calls, sliding-window calls, per-slice calls, etc.).

use crate::{stable_hash64, stable_hash64_u64};
use std::collections::BTreeSet;

/// Coverage / maintenance sampling configuration.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CoverageConfig {
    /// Enable coverage picks.
    pub enabled: bool,
    /// Minimum fraction of total observed calls that each arm should receive.
    ///
    /// Example: `0.02` targets ~2% of calls to each arm (when feasible).
    pub min_fraction: f64,
    /// Absolute minimum calls to target per arm (useful early, or for short windows).
    pub min_calls_floor: u64,
}

impl Default for CoverageConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_fraction: 0.0,
            min_calls_floor: 0,
        }
    }
}

#[derive(Debug, Clone)]
struct DeficitRow {
    arm: String,
    calls: u64,
    deficit: u64,
    tie: u64,
}

fn sanitize_fraction(x: f64) -> f64 {
    if x.is_finite() && x > 0.0 {
        x.min(1.0)
    } else {
        0.0
    }
}

/// Deterministically pick up to `k` arms that are under-sampled relative to a target quota.
///
/// The quota is computed as:
/// `target = max(min_calls_floor, ceil(min_fraction * total_calls))`.
///
/// Then each arm has deficit `max(0, target - calls(arm))`, and we return arms in descending
/// deficit order with a deterministic hash tie-break.
///
/// If your arms are naturally indexed (`0..n_arms`) and you need this in a tight loop,
/// prefer `coverage_pick_under_sampled_idx` to avoid `String` cloning and set/sort overhead.
#[must_use]
pub fn coverage_pick_under_sampled<F>(
    seed: u64,
    arms: &[String],
    k: usize,
    cfg: CoverageConfig,
    mut observed_calls: F,
) -> Vec<String>
where
    F: FnMut(&str) -> u64,
{
    if !cfg.enabled || k == 0 || arms.is_empty() {
        return Vec::new();
    }

    // Defensive: treat the input as a set, preserving first-occurrence order.
    let mut seen = BTreeSet::<String>::new();
    let mut arms_unique: Vec<String> = Vec::new();
    for a in arms {
        if seen.insert(a.clone()) {
            arms_unique.push(a.clone());
        }
    }
    if arms_unique.is_empty() {
        return Vec::new();
    }

    let frac = sanitize_fraction(cfg.min_fraction);
    let floor = cfg.min_calls_floor;

    let mut rows: Vec<DeficitRow> = Vec::with_capacity(arms_unique.len());
    let mut total_calls: u64 = 0;
    for a in &arms_unique {
        let c = observed_calls(a.as_str());
        total_calls = total_calls.saturating_add(c);
        rows.push(DeficitRow {
            arm: a.clone(),
            calls: c,
            deficit: 0,
            tie: stable_hash64(seed ^ 0x434F_5645, a), // "COVE"
        });
    }

    let target_from_frac = if frac > 0.0 && total_calls > 0 {
        (frac * (total_calls as f64)).ceil() as u64
    } else {
        0
    };
    let target = floor.max(target_from_frac);
    if target == 0 {
        return Vec::new();
    }

    for r in &mut rows {
        r.deficit = target.saturating_sub(r.calls);
    }

    // Keep only those behind target.
    rows.retain(|r| r.deficit > 0);
    if rows.is_empty() {
        return Vec::new();
    }

    rows.sort_by(|a, b| {
        b.deficit
            .cmp(&a.deficit)
            .then_with(|| a.tie.cmp(&b.tie))
            .then_with(|| a.arm.cmp(&b.arm))
    });

    rows.into_iter()
        .take(k.min(arms_unique.len()))
        .map(|r| r.arm)
        .collect()
}

#[derive(Debug, Clone, Copy)]
struct DeficitRowIdx {
    arm: usize,
    calls: u64,
    deficit: u64,
    tie: u64,
}

/// Deterministically pick up to `k` arms (by index) that are under-sampled relative to a target quota.
///
/// This is an allocation- and clone-free variant of `coverage_pick_under_sampled` for tight loops,
/// where arms are naturally represented as indices `0..n_arms`.
#[must_use]
pub fn coverage_pick_under_sampled_idx<F>(
    seed: u64,
    n_arms: usize,
    k: usize,
    cfg: CoverageConfig,
    mut observed_calls: F,
) -> Vec<usize>
where
    F: FnMut(usize) -> u64,
{
    if !cfg.enabled || k == 0 || n_arms == 0 {
        return Vec::new();
    }

    let frac = sanitize_fraction(cfg.min_fraction);
    let floor = cfg.min_calls_floor;

    let mut rows: Vec<DeficitRowIdx> = Vec::with_capacity(n_arms);
    let mut total_calls: u64 = 0;
    for arm in 0..n_arms {
        let c = observed_calls(arm);
        total_calls = total_calls.saturating_add(c);
        rows.push(DeficitRowIdx {
            arm,
            calls: c,
            deficit: 0,
            tie: stable_hash64_u64(seed ^ 0x434F_5645, arm as u64), // "COVE"
        });
    }

    let target_from_frac = if frac > 0.0 && total_calls > 0 {
        (frac * (total_calls as f64)).ceil() as u64
    } else {
        0
    };
    let target = floor.max(target_from_frac);
    if target == 0 {
        return Vec::new();
    }

    for r in &mut rows {
        r.deficit = target.saturating_sub(r.calls);
    }

    rows.retain(|r| r.deficit > 0);
    if rows.is_empty() {
        return Vec::new();
    }

    rows.sort_by(|a, b| {
        b.deficit
            .cmp(&a.deficit)
            .then_with(|| a.tie.cmp(&b.tie))
            .then_with(|| a.arm.cmp(&b.arm))
    });

    rows.into_iter()
        .take(k.min(n_arms))
        .map(|r| r.arm)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn coverage_pick_is_deterministic(
            seed in any::<u64>(),
            k in 0usize..10,
            min_fraction in prop_oneof![Just(0.0), 1e-6f64..0.5f64],
            min_floor in 0u64..20,
            // small arm sets
            arms in prop::collection::vec("[a-z]{1,8}", 0..8),
            // observed calls aligned to arms length (truncate/pad)
            calls in prop::collection::vec(0u64..50, 0..8),
        ) {
            let arms_raw: Vec<String> = arms.into_iter().collect();
            // Mirror the function's "first-occurrence" semantics.
            let mut seen = BTreeSet::<String>::new();
            let mut arms: Vec<String> = Vec::new();
            for a in arms_raw {
                if seen.insert(a.clone()) {
                    arms.push(a);
                }
            }
            let cfg = CoverageConfig { enabled: true, min_fraction, min_calls_floor: min_floor };

            // Build a stable calls map aligned to the deduped arm list.
            let mut calls_map = std::collections::BTreeMap::<String, u64>::new();
            for (i, a) in arms.iter().enumerate() {
                calls_map.insert(a.clone(), calls.get(i).copied().unwrap_or(0));
            }

            let pick1 = coverage_pick_under_sampled(seed, &arms, k, cfg, |a| {
                calls_map.get(a).copied().unwrap_or(0)
            });
            let pick2 = coverage_pick_under_sampled(seed, &arms, k, cfg, |a| {
                calls_map.get(a).copied().unwrap_or(0)
            });
            prop_assert_eq!(pick1.clone(), pick2);

            // Picks are members and unique.
            for p in &pick1 {
                prop_assert!(arms.iter().any(|a| a == p));
            }
            let mut uniq = pick1.clone();
            uniq.sort();
            uniq.dedup();
            prop_assert_eq!(uniq.len(), pick1.len());
        }

        #[test]
        fn coverage_pick_idx_is_deterministic_and_valid(
            seed in any::<u64>(),
            n_arms in 0usize..32,
            k in 0usize..10,
            min_fraction in prop_oneof![Just(0.0), 1e-6f64..0.5f64],
            min_floor in 0u64..20,
            calls in prop::collection::vec(0u64..50, 0..32),
        ) {
            let cfg = CoverageConfig { enabled: true, min_fraction, min_calls_floor: min_floor };

            let pick1 = coverage_pick_under_sampled_idx(seed, n_arms, k, cfg, |idx| {
                calls.get(idx).copied().unwrap_or(0)
            });
            let pick2 = coverage_pick_under_sampled_idx(seed, n_arms, k, cfg, |idx| {
                calls.get(idx).copied().unwrap_or(0)
            });
            prop_assert_eq!(&pick1, &pick2);

            // Picks are unique and in-range.
            let mut uniq = pick1.clone();
            uniq.sort();
            uniq.dedup();
            prop_assert_eq!(uniq.len(), pick1.len());
            for &p in &pick1 {
                prop_assert!(p < n_arms);
            }

            // Validate "deficit>0" constraint against the spec.
            let frac = sanitize_fraction(cfg.min_fraction);
            let floor = cfg.min_calls_floor;
            let mut total_calls: u64 = 0;
            for arm in 0..n_arms {
                total_calls = total_calls.saturating_add(calls.get(arm).copied().unwrap_or(0));
            }
            let target_from_frac = if frac > 0.0 && total_calls > 0 {
                (frac * (total_calls as f64)).ceil() as u64
            } else {
                0
            };
            let target = floor.max(target_from_frac);
            if target == 0 || n_arms == 0 || k == 0 {
                prop_assert!(pick1.is_empty());
                return Ok(());
            }
            let mut deficits = Vec::with_capacity(n_arms);
            for arm in 0..n_arms {
                let c = calls.get(arm).copied().unwrap_or(0);
                deficits.push(target.saturating_sub(c));
            }

            let deficit_arms: usize = deficits.iter().filter(|&&d| d > 0).count();
            prop_assert!(pick1.len() <= deficit_arms.min(k).min(n_arms));
            for &p in &pick1 {
                prop_assert!(deficits[p] > 0);
            }
        }
    }

    #[test]
    fn coverage_pick_idx_matches_string_when_deficits_are_distinct() {
        let cfg = CoverageConfig {
            enabled: true,
            min_fraction: 0.0,
            min_calls_floor: 10,
        };
        let calls = [0u64, 1, 2, 10, 11];
        let n_arms = calls.len();
        let k = 3;

        let arms: Vec<String> = (0..n_arms).map(|i| format!("arm{i}")).collect();
        let s = coverage_pick_under_sampled(123, &arms, k, cfg, |a| {
            let idx = a
                .strip_prefix("arm")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0);
            calls.get(idx).copied().unwrap_or(0)
        });
        assert_eq!(
            s,
            vec!["arm0".to_string(), "arm1".to_string(), "arm2".to_string()]
        );

        let idx = coverage_pick_under_sampled_idx(123, n_arms, k, cfg, |i| calls[i]);
        assert_eq!(idx, vec![0, 1, 2]);
    }
}
