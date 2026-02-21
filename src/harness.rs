//! Harness-oriented glue helpers.
//!
//! These helpers are intentionally lightweight and dependency-free; they exist so that
//! external harnesses and CLIs can share *exact* deterministic selection semantics without
//! re-implementing “glue” logic (novelty → guardrail → pick-rest).
//!
//! ## Pipeline ordering
//!
//! The policy pipeline has two valid orderings, controlled by [`PipelineOrder`]:
//!
//! - **`NoveltyFirst`** (default): novelty/coverage pre-picks run *before* the guardrail,
//!   so unseen arms can bypass `require_measured`.  The guardrail applies only to the
//!   remaining arms after pre-picks.  Fallback-on-empty uses the unguarded set.
//!
//! - **`GuardrailFirst`**: the guardrail applies as a hard constraint *before*
//!   novelty/coverage, so `require_measured` blocks all unmeasured arms including unseen
//!   ones.  No fallback: if the guardrail empties the set, `stop_early=true`.

use crate::{coverage_pick_under_sampled, CoverageConfig};
use crate::{novelty_pick_unseen, stable_hash64, LatencyGuardrailConfig};

/// Pipeline ordering: whether novelty/coverage pre-picks run before or after the guardrail.
///
/// - **`NoveltyFirst`** (default): novelty/coverage pre-picks run *before* the guardrail.
/// - **`GuardrailFirst`**: the guardrail applies as a hard constraint *before* novelty/coverage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PipelineOrder {
    /// Novelty/coverage pre-picks run first; guardrail applies to the remainder.
    /// Unseen arms can bypass `require_measured` if novelty fills all k.
    #[default]
    NoveltyFirst,
    /// Guardrail applies first as a hard constraint (no fallback); novelty/coverage
    /// run only within the guarded set.
    GuardrailFirst,
}

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

/// Like [`guardrail_filter_observed`], but **strict**: never falls back to the unguarded set.
///
/// Semantics:
/// - If `guard.max_mean_ms` is `None`, returns `arms` unchanged.
/// - Otherwise filters by `mean_ms <= max_mean_ms` (and, if `require_measured`, `calls > 0`).
/// - If filtering yields an empty set, returns `(vec![], true)` regardless of `allow_fewer`.
///
/// This is useful when you want guardrails to be **hard constraints**, including for novelty/coverage
/// pre-picks (see `*_guardrail_first_*` helpers below).
pub fn guardrail_filter_observed_strict<F>(
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
        return (Vec::new(), true);
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

/// Like [`guardrail_filter_observed_strict`], but the observation function returns raw elapsed-ms sum.
pub fn guardrail_filter_observed_strict_elapsed<F>(
    seed: u64,
    arms: &[String],
    guard: LatencyGuardrail,
    mut observed: F,
) -> (Vec<String>, bool)
where
    F: FnMut(&str) -> (u64, u64),
{
    guardrail_filter_observed_strict(seed, arms, guard, |b| {
        let (calls, elapsed_ms_sum) = observed(b);
        let mean_ms = if calls == 0 {
            0.0
        } else {
            (elapsed_ms_sum as f64) / (calls as f64)
        };
        (calls, mean_ms)
    })
}

/// Generic policy plan: compute pre-picks (novelty + optional coverage) and guardrail
/// filtering, with ordering controlled by `order`.
///
/// This is the canonical implementation that all `policy_plan_observed*` functions delegate to.
#[allow(clippy::too_many_arguments)]
pub fn policy_plan_generic<F>(
    seed: u64,
    arms: &[String],
    k: usize,
    novelty_enabled: bool,
    coverage: CoverageConfig,
    guard: LatencyGuardrail,
    order: PipelineOrder,
    mut observed: F,
) -> PolicyPlan
where
    F: FnMut(&str) -> (u64, u64),
{
    match order {
        PipelineOrder::GuardrailFirst => {
            let (guarded, stop_early) = guardrail_filter_observed_strict_elapsed(
                seed ^ 0x504C_414E,
                arms,
                guard,
                &mut observed,
            );
            if stop_early || guarded.is_empty() {
                return PolicyPlan {
                    prechosen: Vec::new(),
                    eligible: Vec::new(),
                    stop_early: true,
                };
            }
            let prechosen = prepick_novelty_coverage(
                seed,
                &guarded,
                k,
                novelty_enabled,
                coverage,
                &mut observed,
            );
            let eligible: Vec<String> = guarded
                .iter()
                .filter(|b| !prechosen.contains(*b))
                .cloned()
                .collect();
            PolicyPlan {
                prechosen,
                eligible,
                stop_early: false,
            }
        }
        PipelineOrder::NoveltyFirst => {
            let prechosen =
                prepick_novelty_coverage(seed, arms, k, novelty_enabled, coverage, &mut observed);
            let remaining: Vec<String> = arms
                .iter()
                .filter(|b| !prechosen.contains(*b))
                .cloned()
                .collect();
            let (eligible, stop_early) =
                guardrail_filter_observed_elapsed(seed ^ 0x504C_414E, &remaining, guard, observed);
            PolicyPlan {
                prechosen,
                eligible,
                stop_early,
            }
        }
    }
}

/// Shared novelty + coverage pre-pick logic.
fn prepick_novelty_coverage<F>(
    seed: u64,
    arms: &[String],
    k: usize,
    novelty_enabled: bool,
    coverage: CoverageConfig,
    observed: &mut F,
) -> Vec<String>
where
    F: FnMut(&str) -> (u64, u64),
{
    let pre_novel = novelty_pick_unseen(seed, arms, k, novelty_enabled, |b| observed(b).0);

    if !coverage.enabled {
        return pre_novel;
    }

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
    prechosen
}

/// Compute the policy plan: novelty pre-picks + observed latency guardrail.
///
/// Important semantics:
/// - Novelty picks are computed **before** the guardrail filter (so they can include unmeasured arms
///   even when `require_measured=true`, if novelty fills all \(k\)).
/// - If you want the guardrail to apply as a **hard constraint** to novelty/coverage picks, use the
///   `*_guardrail_first_*` helpers below.
pub fn policy_plan_observed<F>(
    seed: u64,
    arms: &[String],
    k: usize,
    novelty_enabled: bool,
    guard: LatencyGuardrail,
    observed: F,
) -> PolicyPlan
where
    F: FnMut(&str) -> (u64, u64),
{
    policy_plan_generic(
        seed,
        arms,
        k,
        novelty_enabled,
        CoverageConfig::default(),
        guard,
        PipelineOrder::NoveltyFirst,
        observed,
    )
}

/// Like `policy_plan_observed`, but also adds a deterministic "coverage" pre-pick stage
/// to enforce minimum sampling quotas (useful for monitoring/change detection).
///
/// Delegates to [`policy_plan_generic`] with `PipelineOrder::NoveltyFirst`.
pub fn policy_plan_observed_with_coverage<F>(
    seed: u64,
    arms: &[String],
    k: usize,
    novelty_enabled: bool,
    coverage: CoverageConfig,
    guard: LatencyGuardrail,
    observed: F,
) -> PolicyPlan
where
    F: FnMut(&str) -> (u64, u64),
{
    policy_plan_generic(
        seed,
        arms,
        k,
        novelty_enabled,
        coverage,
        guard,
        PipelineOrder::NoveltyFirst,
        observed,
    )
}

/// Guardrail-first policy plan: apply observed guardrail first, then novelty pre-picks.
///
/// Delegates to [`policy_plan_generic`] with `PipelineOrder::GuardrailFirst`.
pub fn policy_plan_observed_guardrail_first<F>(
    seed: u64,
    arms: &[String],
    k: usize,
    novelty_enabled: bool,
    guard: LatencyGuardrail,
    observed: F,
) -> PolicyPlan
where
    F: FnMut(&str) -> (u64, u64),
{
    policy_plan_generic(
        seed,
        arms,
        k,
        novelty_enabled,
        CoverageConfig::default(),
        guard,
        PipelineOrder::GuardrailFirst,
        observed,
    )
}

/// Guardrail-first policy plan with coverage: apply guardrail first, then novelty+coverage.
///
/// Delegates to [`policy_plan_generic`] with `PipelineOrder::GuardrailFirst`.
pub fn policy_plan_observed_guardrail_first_with_coverage<F>(
    seed: u64,
    arms: &[String],
    k: usize,
    novelty_enabled: bool,
    coverage: CoverageConfig,
    guard: LatencyGuardrail,
    observed: F,
) -> PolicyPlan
where
    F: FnMut(&str) -> (u64, u64),
{
    policy_plan_generic(
        seed,
        arms,
        k,
        novelty_enabled,
        coverage,
        guard,
        PipelineOrder::GuardrailFirst,
        observed,
    )
}

/// Generic policy fill: compute a plan via [`policy_plan_generic`], then fill remaining
/// picks using `pick_rest`.
///
/// This is the canonical implementation that all `policy_fill_k_observed*` functions delegate to.
///
/// Fallback semantics (NoveltyFirst only): if the guardrail empties the eligible set and
/// we have zero picks, fall back to the unguarded remaining set (unless `require_measured`
/// is set, in which case stop early). GuardrailFirst never falls back.
#[allow(clippy::too_many_arguments)]
pub fn policy_fill_generic<F, P>(
    seed: u64,
    arms: &[String],
    k: usize,
    novelty_enabled: bool,
    coverage: CoverageConfig,
    guard: LatencyGuardrail,
    order: PipelineOrder,
    mut observed: F,
    mut pick_rest: P,
) -> PolicyFill
where
    F: FnMut(&str) -> (u64, u64),
    P: FnMut(&[String], usize) -> Vec<String>,
{
    let plan = policy_plan_generic(
        seed,
        arms,
        k,
        novelty_enabled,
        coverage,
        guard,
        order,
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

    // If guardrail says "stop early", only honor it if we already picked something.
    if plan.stop_early && !chosen.is_empty() {
        return PolicyFill {
            chosen,
            eligible_used: Vec::new(),
            plan,
            fallback_used: false,
            stopped_early: true,
        };
    }

    // GuardrailFirst: never fall back; if eligible is empty, stop.
    if order == PipelineOrder::GuardrailFirst {
        if plan.stop_early {
            return PolicyFill {
                chosen,
                plan,
                eligible_used: Vec::new(),
                fallback_used: false,
                stopped_early: true,
            };
        }
        let eligible_used = plan.eligible.clone();
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
        return PolicyFill {
            chosen,
            plan,
            eligible_used,
            fallback_used: false,
            stopped_early: false,
        };
    }

    // NoveltyFirst: may fall back when eligible is empty.
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

/// Delegates to [`policy_fill_generic`] with `PipelineOrder::NoveltyFirst`.
pub fn policy_fill_k_observed_with<F, P>(
    seed: u64,
    arms: &[String],
    k: usize,
    novelty_enabled: bool,
    guard: LatencyGuardrail,
    observed: F,
    pick_rest: P,
) -> PolicyFill
where
    F: FnMut(&str) -> (u64, u64),
    P: FnMut(&[String], usize) -> Vec<String>,
{
    policy_fill_generic(
        seed,
        arms,
        k,
        novelty_enabled,
        CoverageConfig::default(),
        guard,
        PipelineOrder::NoveltyFirst,
        observed,
        pick_rest,
    )
}

/// Delegates to [`policy_fill_generic`] with `PipelineOrder::GuardrailFirst`.
pub fn policy_fill_k_observed_guardrail_first_with<F, P>(
    seed: u64,
    arms: &[String],
    k: usize,
    novelty_enabled: bool,
    guard: LatencyGuardrail,
    observed: F,
    pick_rest: P,
) -> PolicyFill
where
    F: FnMut(&str) -> (u64, u64),
    P: FnMut(&[String], usize) -> Vec<String>,
{
    policy_fill_generic(
        seed,
        arms,
        k,
        novelty_enabled,
        CoverageConfig::default(),
        guard,
        PipelineOrder::GuardrailFirst,
        observed,
        pick_rest,
    )
}

/// Delegates to [`policy_fill_generic`] with `PipelineOrder::NoveltyFirst` and caller-provided coverage.
#[allow(clippy::too_many_arguments)]
pub fn policy_fill_k_observed_with_coverage<F, P>(
    seed: u64,
    arms: &[String],
    k: usize,
    novelty_enabled: bool,
    coverage: CoverageConfig,
    guard: LatencyGuardrail,
    observed: F,
    pick_rest: P,
) -> PolicyFill
where
    F: FnMut(&str) -> (u64, u64),
    P: FnMut(&[String], usize) -> Vec<String>,
{
    policy_fill_generic(
        seed,
        arms,
        k,
        novelty_enabled,
        coverage,
        guard,
        PipelineOrder::NoveltyFirst,
        observed,
        pick_rest,
    )
}

/// Delegates to [`policy_fill_generic`] with `PipelineOrder::GuardrailFirst` and caller-provided coverage.
#[allow(clippy::too_many_arguments)]
pub fn policy_fill_k_observed_guardrail_first_with_coverage<F, P>(
    seed: u64,
    arms: &[String],
    k: usize,
    novelty_enabled: bool,
    coverage: CoverageConfig,
    guard: LatencyGuardrail,
    observed: F,
    pick_rest: P,
) -> PolicyFill
where
    F: FnMut(&str) -> (u64, u64),
    P: FnMut(&[String], usize) -> Vec<String>,
{
    policy_fill_generic(
        seed,
        arms,
        k,
        novelty_enabled,
        coverage,
        guard,
        PipelineOrder::GuardrailFirst,
        observed,
        pick_rest,
    )
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

// =============================================================================
// Contextual decision pipeline
// =============================================================================

/// Result of a contextual policy fill (LinUCB-based selection with context features).
///
/// This extends the non-contextual `PolicyFill` with context metadata for logging
/// and offline evaluation (propensity scores, per-arm UCB scores).
#[cfg(feature = "contextual")]
#[derive(Debug, Clone)]
pub struct ContextualPolicyFill {
    /// Base fill result (chosen arms, plan, guardrail metadata).
    pub fill: PolicyFill,
    /// Context vector used for this decision.
    pub context: Vec<f64>,
    /// Per-arm (ucb, mean, bonus) scores at decision time.
    pub scores: std::collections::BTreeMap<String, (f64, f64, f64)>,
}

/// Contextual variant of `policy_fill_k_observed_with`: uses LinUCB for the
/// algorithmic selection step instead of a flat MAB/EXP3-IX policy.
///
/// This enables the **contextual regime** where routing objectives diverge:
/// LinUCB learns to route based on per-request features (language, domain, etc.)
/// so that "biomedical text -> backend X" is learned from all slices simultaneously,
/// without maintaining separate per-facet histories.
///
/// The caller supplies:
/// - `linucb`: a mutable reference to a `LinUcb` instance (caller manages persistence)
/// - `context`: the feature vector for this decision (e.g. dataset metadata)
/// - `observed`: per-arm (calls, elapsed_ms_sum) for guardrail filtering
///
/// The harness pipeline is: novelty -> observed guardrail -> LinUCB argmax fill.
#[cfg(feature = "contextual")]
#[allow(clippy::too_many_arguments)]
pub fn policy_fill_k_contextual<F>(
    seed: u64,
    arms: &[String],
    k: usize,
    novelty_enabled: bool,
    guard: LatencyGuardrail,
    linucb: &mut crate::LinUcb,
    context: &[f64],
    observed: F,
) -> ContextualPolicyFill
where
    F: FnMut(&str) -> (u64, u64),
{
    let scores = linucb.scores(arms, context);

    let fill = policy_fill_k_observed_with(
        seed,
        arms,
        k,
        novelty_enabled,
        guard,
        observed,
        |eligible, remaining_k| {
            // LinUCB argmax selection: pick the top-scoring eligible arms.
            let mut scored: Vec<(f64, String)> = eligible
                .iter()
                .map(|a| {
                    let ucb = scores.get(a).map(|t| t.0).unwrap_or(0.0);
                    (ucb, a.clone())
                })
                .collect();
            // Sort by UCB descending, then lexicographic for stable tie-break.
            scored.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            scored
                .into_iter()
                .take(remaining_k)
                .map(|(_, arm)| arm)
                .collect()
        },
    );

    ContextualPolicyFill {
        fill,
        context: context.to_vec(),
        scores,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arms2() -> Vec<String> {
        vec!["a".to_string(), "b".to_string()]
    }

    fn arms3() -> Vec<String> {
        vec!["a".to_string(), "b".to_string(), "c".to_string()]
    }

    fn guard_strict(max_ms: f64) -> LatencyGuardrail {
        LatencyGuardrail {
            max_mean_ms: Some(max_ms),
            require_measured: true,
            allow_fewer: false,
        }
    }

    fn guard_soft(max_ms: f64) -> LatencyGuardrail {
        LatencyGuardrail {
            max_mean_ms: Some(max_ms),
            require_measured: false,
            allow_fewer: true,
        }
    }

    // NoveltyFirst: unseen arm bypasses require_measured guardrail —
    // the key semantic difference vs GuardrailFirst.
    // Uses three arms so the guardrail has a passing arm ("fast") and doesn't fall back.
    #[test]
    fn novelty_first_unseen_bypasses_require_measured() {
        let arms = vec!["unseen".to_string(), "slow".to_string(), "fast".to_string()];
        let plan = policy_plan_generic(
            42,
            &arms,
            3,
            true,
            CoverageConfig::default(),
            guard_strict(50.0),
            PipelineOrder::NoveltyFirst,
            |b| match b {
                "unseen" => (0, 0),   // 0 calls → novelty pick
                "slow" => (10, 1000), // mean 100ms > 50ms
                "fast" => (10, 200),  // mean 20ms < 50ms
                _ => (0, 0),
            },
        );
        assert!(
            plan.prechosen.contains(&"unseen".to_string()),
            "NoveltyFirst: unseen arm should be prechosen despite require_measured"
        );
        assert!(
            plan.eligible.contains(&"fast".to_string()),
            "fast arm should be eligible after guardrail"
        );
        assert!(
            !plan.eligible.contains(&"slow".to_string()),
            "slow arm (100ms > 50ms) must not be eligible"
        );
    }

    // GuardrailFirst: unseen arm is blocked when require_measured=true.
    #[test]
    fn guardrail_first_blocks_unseen_with_require_measured() {
        let arms = vec!["unseen".to_string(), "fast".to_string()];
        let plan = policy_plan_generic(
            42,
            &arms,
            2,
            true,
            CoverageConfig::default(),
            guard_strict(50.0),
            PipelineOrder::GuardrailFirst,
            |b| {
                if b == "unseen" {
                    (0, 0)
                } else {
                    (10, 200) // mean 20ms — fast
                }
            },
        );
        assert!(
            !plan.prechosen.contains(&"unseen".to_string()),
            "GuardrailFirst: unseen arm must not bypass require_measured"
        );
        assert!(
            plan.eligible.contains(&"fast".to_string()),
            "measured fast arm should be eligible"
        );
    }

    // GuardrailFirst: all arms unmeasured → stop_early immediately.
    #[test]
    fn guardrail_first_all_unmeasured_stops_early() {
        let plan = policy_plan_generic(
            42,
            &arms2(),
            2,
            true,
            CoverageConfig::default(),
            LatencyGuardrail {
                max_mean_ms: Some(50.0),
                require_measured: true,
                allow_fewer: true,
            },
            PipelineOrder::GuardrailFirst,
            |_| (0, 0),
        );
        assert!(plan.stop_early);
        assert!(plan.prechosen.is_empty());
        assert!(plan.eligible.is_empty());
    }

    // NoveltyFirst: unseen arm picked for novelty; remaining arms go through guardrail.
    #[test]
    fn novelty_first_guardrail_applies_to_remainder() {
        // "a" is unseen; "b" is fast; "c" is slow.
        let arms = arms3();
        let plan = policy_plan_generic(
            42,
            &arms,
            3,
            true,
            CoverageConfig::default(),
            guard_soft(50.0),
            PipelineOrder::NoveltyFirst,
            |b| match b {
                "a" => (0, 0),   // unseen → novelty pick
                "b" => (5, 100), // mean 20ms — fast
                "c" => (5, 750), // mean 150ms — slow
                _ => (0, 0),
            },
        );
        assert!(
            plan.prechosen.contains(&"a".to_string()),
            "unseen arm prechosen"
        );
        assert!(
            plan.eligible.contains(&"b".to_string()),
            "fast arm eligible"
        );
        assert!(
            !plan.eligible.contains(&"c".to_string()),
            "slow arm filtered"
        );
    }

    // policy_fill_generic: prechosen fills k without calling pick_rest.
    #[test]
    fn policy_fill_generic_prechosen_fills_without_algorithm() {
        let arms = arms2();
        let fill = policy_fill_generic(
            42,
            &arms,
            1,
            true,
            CoverageConfig::default(),
            LatencyGuardrail::default(),
            PipelineOrder::NoveltyFirst,
            |b| if b == "a" { (0, 0) } else { (5, 100) },
            |_eligible, _k| panic!("pick_rest must not be called when prechosen fills k"),
        );
        assert_eq!(fill.chosen, vec!["a".to_string()]);
        assert!(!fill.stopped_early);
        assert!(!fill.fallback_used);
    }

    // policy_fill_generic: prechosen + algorithm together when prechosen < k.
    #[test]
    fn policy_fill_generic_combines_prechosen_and_algorithm() {
        let arms = arms3();
        let fill = policy_fill_generic(
            42,
            &arms,
            2,
            true,
            CoverageConfig::default(),
            LatencyGuardrail::default(),
            PipelineOrder::NoveltyFirst,
            |b| if b == "a" { (0, 0) } else { (5, 100) },
            |eligible, _k| eligible.to_vec(),
        );
        assert_eq!(fill.chosen.len(), 2);
        assert!(
            fill.chosen.contains(&"a".to_string()),
            "prechosen novelty arm included"
        );
        // uniqueness
        let mut s = fill.chosen.clone();
        s.sort();
        s.dedup();
        assert_eq!(s.len(), fill.chosen.len(), "chosen must be unique");
    }

    // GuardrailFirst with stop_early stops even if prechosen is empty.
    #[test]
    fn guardrail_first_stop_early_halts_fill() {
        let arms = arms2();
        let fill = policy_fill_generic(
            42,
            &arms,
            2,
            false,
            CoverageConfig::default(),
            LatencyGuardrail {
                max_mean_ms: Some(10.0),
                require_measured: true,
                allow_fewer: true,
            },
            PipelineOrder::GuardrailFirst,
            |_| (0, 0), // all unmeasured → stop_early
            |_eligible, _k| unreachable!("must not reach algorithm"),
        );
        assert!(fill.stopped_early);
        assert!(fill.chosen.is_empty());
    }

    // guardrail_filter_observed: filters by mean latency, falls back when all filtered.
    #[test]
    fn guardrail_filter_observed_filters_and_falls_back() {
        let arms = arms2();
        // Both arms above threshold → fallback to full set.
        let (eligible, stop_early) = guardrail_filter_observed(
            42,
            &arms,
            LatencyGuardrail {
                max_mean_ms: Some(10.0),
                require_measured: false,
                allow_fewer: false,
            },
            |_| (1, 100.0), // mean 100ms
        );
        assert!(!stop_early);
        assert_eq!(eligible, arms, "fallback returns original arms");
    }

    // guardrail_filter_observed_strict: never falls back, returns empty + stop_early.
    #[test]
    fn guardrail_filter_observed_strict_returns_empty_no_fallback() {
        let arms = arms2();
        let (eligible, stop_early) = guardrail_filter_observed_strict(
            42,
            &arms,
            LatencyGuardrail {
                max_mean_ms: Some(10.0),
                require_measured: false,
                allow_fewer: true,
            },
            |_| (1, 100.0),
        );
        assert!(stop_early);
        assert!(eligible.is_empty());
    }
}
