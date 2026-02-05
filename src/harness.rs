//! Harness-oriented glue helpers.
//!
//! These helpers are intentionally lightweight and dependency-free; they exist so that
//! external harnesses and CLIs can share *exact* deterministic selection semantics without
//! re-implementing “glue” logic (novelty → guardrail → pick-rest).

use crate::{coverage_pick_under_sampled, CoverageConfig};
use crate::{novelty_pick_unseen, stable_hash64, LatencyGuardrailConfig};

/// Re-export `LatencyGuardrailConfig` under the harness-friendly name used by some routers.
pub type LatencyGuardrail = LatencyGuardrailConfig;

/// Planned policy result for a single selection step.
///
/// This is intentionally “muxer-side”: it only depends on observed stats, novelty, and guardrails,
/// not on application semantics.
#[derive(Debug, Clone)]
pub struct PolicyPlan {
    /// Arms to pre-pick for novelty/coverage (slice-unseen).
    pub prechosen: Vec<String>,
    /// Arms eligible for the algorithmic selector after applying observed guardrails.
    pub eligible: Vec<String>,
    /// If true, the observed guardrail filtered all remaining arms and requested stop-early.
    pub stop_early: bool,
}

/// Result of filling up to `k` picks using the shared policy pipeline.
#[derive(Debug, Clone)]
pub struct PolicyFill {
    /// Final chosen arms (order-preserving).
    pub chosen: Vec<String>,
    /// The policy plan computed for this step.
    pub plan: PolicyPlan,
    /// The eligible set that the algorithm selector actually saw.
    pub eligible_used: Vec<String>,
    /// True if the guardrail filtered all arms and we fell back to the unguarded remaining set.
    pub fallback_used: bool,
    /// True if the guardrail requested stop-early and we honored it.
    pub stopped_early: bool,
}

/// Apply the latency guardrail using *observed* (non-smoothed) stats.
///
/// This helper is useful when summaries are smoothed via priors: it keeps guardrails meaningful by
/// ensuring `require_measured` and `max_mean_ms` are based on real observations.
///
/// Returns `(eligible_arms, stop_early)`.
pub fn guardrail_filter_observed<F>(
    seed: u64,
    arms: &[String],
    guard: LatencyGuardrail,
    mut observed: F,
) -> (Vec<String>, bool)
where
    F: FnMut(&str) -> (u64, f64),
{
    let Some(max_ms) = guard.max_mean_ms else {
        return (arms.to_vec(), false);
    };
    let mut eligible: Vec<String> = arms
        .iter()
        .filter(|b| {
            let (calls, mean_ms) = observed(b.as_str());
            if guard.require_measured && calls == 0 {
                return false;
            }
            mean_ms <= max_ms
        })
        .cloned()
        .collect();
    eligible.sort_by_key(|b| stable_hash64(seed ^ 0x4755_4152, b)); // "GUAR"

    if eligible.is_empty() {
        if guard.allow_fewer {
            return (Vec::new(), true);
        }
        return (arms.to_vec(), false);
    }

    (eligible, false)
}

/// Like `guardrail_filter_observed`, but the observation function returns the raw elapsed-ms sum.
///
/// This centralizes mean-latency calculation so callers only provide raw counters.
pub fn guardrail_filter_observed_elapsed<F>(
    seed: u64,
    arms: &[String],
    guard: LatencyGuardrail,
    mut observed: F,
) -> (Vec<String>, bool)
where
    F: FnMut(&str) -> (u64, u64),
{
    guardrail_filter_observed(seed, arms, guard, |b| {
        let (calls, elapsed_ms_sum) = observed(b);
        let mean_ms = if calls == 0 {
            0.0
        } else {
            (elapsed_ms_sum as f64) / (calls as f64)
        };
        (calls, mean_ms)
    })
}

/// Compute the policy plan: novelty pre-picks + observed latency guardrail.
pub fn policy_plan_observed<F>(
    seed: u64,
    arms: &[String],
    k: usize,
    novelty_enabled: bool,
    guard: LatencyGuardrail,
    mut observed: F,
) -> PolicyPlan
where
    F: FnMut(&str) -> (u64, u64),
{
    let prechosen = novelty_pick_unseen(seed, arms, k, novelty_enabled, |b| observed(b).0);

    let remaining: Vec<String> = arms
        .iter()
        .filter(|b| !prechosen.contains(*b))
        .cloned()
        .collect();

    let (eligible, stop_early) =
        guardrail_filter_observed_elapsed(seed ^ 0x504C_414E, &remaining, guard, observed); // "PLAN"

    PolicyPlan {
        prechosen,
        eligible,
        stop_early,
    }
}

/// Like `policy_plan_observed`, but also adds a deterministic “coverage” pre-pick stage
/// to enforce minimum sampling quotas (useful for monitoring/change detection).
///
/// Order: novelty (unseen) first, then coverage (under-sampled), then guardrail.
pub fn policy_plan_observed_with_coverage<F>(
    seed: u64,
    arms: &[String],
    k: usize,
    novelty_enabled: bool,
    coverage: CoverageConfig,
    guard: LatencyGuardrail,
    mut observed: F,
) -> PolicyPlan
where
    F: FnMut(&str) -> (u64, u64),
{
    let pre_novel = novelty_pick_unseen(seed, arms, k, novelty_enabled, |b| observed(b).0);
    let remaining_after_novel: Vec<String> = arms
        .iter()
        .filter(|b| !pre_novel.contains(*b))
        .cloned()
        .collect();

    let need_cov = k.saturating_sub(pre_novel.len());
    let mut pre_cov = coverage_pick_under_sampled(
        seed ^ 0x434F_5645, // "COVE"
        &remaining_after_novel,
        need_cov,
        coverage,
        |b| observed(b).0,
    );

    // Preserve deterministic order: novelty picks first, then coverage picks.
    let mut prechosen = pre_novel;
    for b in pre_cov.drain(..) {
        if prechosen.len() >= k {
            break;
        }
        if prechosen.contains(&b) {
            continue;
        }
        prechosen.push(b);
    }

    let remaining: Vec<String> = arms
        .iter()
        .filter(|b| !prechosen.contains(*b))
        .cloned()
        .collect();

    let (eligible, stop_early) =
        guardrail_filter_observed_elapsed(seed ^ 0x504C_414E, &remaining, guard, observed); // "PLAN"

    PolicyPlan {
        prechosen,
        eligible,
        stop_early,
    }
}

/// Shared muxer “policy pipeline”: novelty → observed guardrail → fill the remaining \(k\).
///
/// Semantics note:
/// - If the guardrail requests stop-early but we have **zero** picks so far, we **fall back**
///   to the unguarded remaining set so callers don't accidentally pick nothing.
pub fn policy_fill_k_observed_with<F, P>(
    seed: u64,
    arms: &[String],
    k: usize,
    novelty_enabled: bool,
    guard: LatencyGuardrail,
    mut observed: F,
    mut pick_rest: P,
) -> PolicyFill
where
    F: FnMut(&str) -> (u64, u64),
    P: FnMut(&[String], usize) -> Vec<String>,
{
    let plan = policy_plan_observed(seed, arms, k, novelty_enabled, guard, &mut observed);
    let mut chosen = plan.prechosen.clone();

    if chosen.len() >= k {
        return PolicyFill {
            chosen,
            plan,
            eligible_used: Vec::new(),
            fallback_used: false,
            stopped_early: false,
        };
    }

    // If guardrail says “stop early”, only honor it if we already picked something.
    if plan.stop_early && !chosen.is_empty() {
        return PolicyFill {
            chosen,
            eligible_used: Vec::new(),
            plan,
            fallback_used: false,
            stopped_early: true,
        };
    }

    // If the guardrail filtered everything:
    // - If we have picks already and allow_fewer, stop.
    // - Otherwise, fall back to the unguarded remaining set so we can still pick something.
    let mut eligible_used = plan.eligible.clone();
    let mut fallback_used = false;
    let mut stopped_early = false;
    if eligible_used.is_empty() {
        if guard.require_measured {
            stopped_early = true;
            return PolicyFill {
                chosen,
                eligible_used,
                plan,
                fallback_used,
                stopped_early,
            };
        }
        if guard.allow_fewer && !chosen.is_empty() {
            stopped_early = true;
            return PolicyFill {
                chosen,
                eligible_used,
                plan,
                fallback_used,
                stopped_early,
            };
        }
        eligible_used = arms
            .iter()
            .filter(|b| !chosen.contains(*b))
            .cloned()
            .collect();
        fallback_used = true;
    }

    let remaining_k = k.saturating_sub(chosen.len());
    if remaining_k > 0 && !eligible_used.is_empty() {
        // Defensive: the algorithm picker is expected to return <= remaining_k picks from
        // `eligible_used`, without duplicates. Enforce those invariants here.
        let rest = pick_rest(&eligible_used, remaining_k);
        for b in rest {
            if chosen.len() >= k {
                break;
            }
            if !eligible_used.contains(&b) {
                continue;
            }
            if chosen.contains(&b) {
                continue;
            }
            chosen.push(b);
        }
    }

    PolicyFill {
        chosen,
        plan,
        eligible_used,
        fallback_used,
        stopped_early,
    }
}

/// Like `policy_fill_k_observed_with`, but includes an optional “coverage” pre-pick stage.
#[allow(clippy::too_many_arguments)]
pub fn policy_fill_k_observed_with_coverage<F, P>(
    seed: u64,
    arms: &[String],
    k: usize,
    novelty_enabled: bool,
    coverage: CoverageConfig,
    guard: LatencyGuardrail,
    mut observed: F,
    mut pick_rest: P,
) -> PolicyFill
where
    F: FnMut(&str) -> (u64, u64),
    P: FnMut(&[String], usize) -> Vec<String>,
{
    let plan = policy_plan_observed_with_coverage(
        seed,
        arms,
        k,
        novelty_enabled,
        coverage,
        guard,
        &mut observed,
    );
    let mut chosen = plan.prechosen.clone();

    if chosen.len() >= k {
        return PolicyFill {
            chosen,
            plan,
            eligible_used: Vec::new(),
            fallback_used: false,
            stopped_early: false,
        };
    }

    if plan.stop_early && !chosen.is_empty() {
        return PolicyFill {
            chosen,
            eligible_used: Vec::new(),
            plan,
            fallback_used: false,
            stopped_early: true,
        };
    }

    let mut eligible_used = plan.eligible.clone();
    let mut fallback_used = false;
    let mut stopped_early = false;
    if eligible_used.is_empty() {
        if guard.require_measured {
            stopped_early = true;
            return PolicyFill {
                chosen,
                eligible_used,
                plan,
                fallback_used,
                stopped_early,
            };
        }
        if guard.allow_fewer && !chosen.is_empty() {
            stopped_early = true;
            return PolicyFill {
                chosen,
                eligible_used,
                plan,
                fallback_used,
                stopped_early,
            };
        }
        eligible_used = arms
            .iter()
            .filter(|b| !chosen.contains(*b))
            .cloned()
            .collect();
        fallback_used = true;
    }

    let remaining_k = k.saturating_sub(chosen.len());
    if remaining_k > 0 && !eligible_used.is_empty() {
        let rest = pick_rest(&eligible_used, remaining_k);
        for b in rest {
            if chosen.len() >= k {
                break;
            }
            if !eligible_used.contains(&b) {
                continue;
            }
            if chosen.contains(&b) {
                continue;
            }
            chosen.push(b);
        }
    }

    PolicyFill {
        chosen,
        plan,
        eligible_used,
        fallback_used,
        stopped_early,
    }
}

/// Shared “select K without replacement” driver.
///
/// Callers provide a picker that can return up to `k_remaining` candidates for the current
/// remaining set; the driver enforces de-duplication and ordering.
pub fn select_k_without_replacement_by_with_meta<F, M>(
    seed: u64,
    items: &[String],
    k: usize,
    mut pick: F,
) -> Vec<(String, M)>
where
    F: FnMut(u64, &[String], usize) -> Vec<(String, M)>,
{
    if k == 0 || items.is_empty() {
        return Vec::new();
    }
    let mut remaining: Vec<String> = items.to_vec();
    let mut out: Vec<(String, M)> = Vec::new();

    while !remaining.is_empty() && out.len() < k {
        let need = k - out.len();
        let batch = pick(seed ^ (out.len() as u64), &remaining, need);
        if batch.is_empty() {
            break;
        }
        let mut made_progress = false;
        for (b, meta) in batch {
            if out.len() >= k {
                break;
            }
            if !remaining.contains(&b) {
                continue;
            }
            if out.iter().any(|(x, _)| x == &b) {
                continue;
            }
            remaining.retain(|x| x != &b);
            out.push((b, meta));
            made_progress = true;
        }
        if !made_progress {
            break;
        }
    }
    out
}

/// Convenience wrapper over `select_k_without_replacement_by_with_meta` when no meta is needed.
#[must_use]
pub fn select_k_without_replacement_by<F>(
    seed: u64,
    items: &[String],
    k: usize,
    mut pick: F,
) -> Vec<String>
where
    F: FnMut(u64, &[String], usize) -> Vec<String>,
{
    select_k_without_replacement_by_with_meta(seed, items, k, |s, rem, need| {
        pick(s, rem, need).into_iter().map(|b| (b, ())).collect()
    })
    .into_iter()
    .map(|(b, _)| b)
    .collect()
}
