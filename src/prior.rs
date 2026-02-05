//! Prior / smoothing helpers for windowed summaries.

use crate::Summary;

/// Apply a pseudo-count prior to a `Summary` until it reaches `target_calls`.
///
/// This converts the prior's rates/means into approximate counts and adds them to `out`.
pub fn apply_prior_counts_to_summary(out: &mut Summary, prior: Summary, target_calls: u64) {
    if target_calls == 0 || out.calls >= target_calls {
        return;
    }
    if prior.calls == 0 {
        return;
    }
    let need = target_calls.saturating_sub(out.calls);
    if need == 0 {
        return;
    }

    let need_f = need as f64;
    let ok = (need_f * prior.ok_rate()).round() as u64;
    let junk = (need_f * prior.junk_rate()).round() as u64;
    let hard_junk = (need_f * prior.hard_junk_rate()).round() as u64;
    let cost_units = (need_f * prior.mean_cost_units()).round() as u64;
    let elapsed_ms_sum = (need_f * prior.mean_elapsed_ms()).round() as u64;

    out.calls = out.calls.saturating_add(need);
    out.ok = out.ok.saturating_add(ok.min(need));
    out.junk = out.junk.saturating_add(junk.min(need));
    out.hard_junk = out.hard_junk.saturating_add(hard_junk.min(need));
    out.cost_units = out.cost_units.saturating_add(cost_units);
    out.elapsed_ms_sum = out.elapsed_ms_sum.saturating_add(elapsed_ms_sum);

    // Defensive invariant: counts must not exceed calls.
    out.ok = out.ok.min(out.calls);
    out.junk = out.junk.min(out.calls);
    out.hard_junk = out.hard_junk.min(out.calls);
}
