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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_summary(calls: u64, ok: u64, junk: u64, hard_junk: u64) -> Summary {
        Summary {
            calls,
            ok,
            junk,
            hard_junk,
            cost_units: 0,
            elapsed_ms_sum: 0,
        }
    }

    #[test]
    fn no_op_when_already_at_target() {
        let mut out = make_summary(10, 8, 1, 0);
        let prior = make_summary(100, 80, 10, 5);
        apply_prior_counts_to_summary(&mut out, prior, 10);
        assert_eq!(out.calls, 10, "should not change when calls >= target");
    }

    #[test]
    fn no_op_when_target_zero() {
        let mut out = make_summary(0, 0, 0, 0);
        let prior = make_summary(100, 80, 10, 5);
        apply_prior_counts_to_summary(&mut out, prior, 0);
        assert_eq!(out.calls, 0);
    }

    #[test]
    fn no_op_when_prior_empty() {
        let mut out = make_summary(0, 0, 0, 0);
        let prior = make_summary(0, 0, 0, 0);
        apply_prior_counts_to_summary(&mut out, prior, 10);
        assert_eq!(out.calls, 0, "empty prior should not add counts");
    }

    #[test]
    fn adds_pseudo_counts() {
        let mut out = make_summary(0, 0, 0, 0);
        let prior = make_summary(100, 50, 20, 10);
        apply_prior_counts_to_summary(&mut out, prior, 10);
        assert_eq!(out.calls, 10);
        assert_eq!(out.ok, 5); // 10 * 0.5
        assert_eq!(out.junk, 2); // 10 * 0.2
        assert_eq!(out.hard_junk, 1); // 10 * 0.1
    }

    #[test]
    fn invariant_counts_do_not_exceed_calls() {
        let mut out = make_summary(5, 5, 0, 0);
        // Prior with 100% ok + 100% junk (impossible rates, but defensive).
        let prior = make_summary(10, 10, 10, 10);
        apply_prior_counts_to_summary(&mut out, prior, 20);
        assert!(out.ok <= out.calls);
        assert!(out.junk <= out.calls);
        assert!(out.hard_junk <= out.calls);
    }
}
