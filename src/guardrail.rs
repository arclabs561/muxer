use std::collections::BTreeMap;

use crate::Summary;

/// Latency guardrail configuration.
///
/// This is an external hard filter typically applied *before* selection:
/// exclude arms whose mean elapsed time is above `max_mean_ms`.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LatencyGuardrailConfig {
    /// If set, exclude arms whose `Summary::mean_elapsed_ms()` exceeds this value.
    pub max_mean_ms: Option<f64>,
    /// If true, arms with `calls == 0` are excluded when `max_mean_ms` is set.
    pub require_measured: bool,
    /// If true, and the filter would eliminate all remaining arms, callers may stop early
    /// *after* at least one arm has already been chosen in this multi-pick selection.
    pub allow_fewer: bool,
}

/// Output of applying a latency guardrail to a candidate set.
#[derive(Debug, Clone)]
pub struct LatencyGuardrailDecision {
    /// Arms to consider after guardrail application.
    pub eligible: Vec<String>,
    /// Whether we fell back to the full input set because the guardrail would have eliminated all arms.
    pub fallback_used: bool,
    /// Whether the caller should stop early (return fewer than requested) because the guardrail eliminated all
    /// remaining arms and `allow_fewer` is enabled.
    pub stop_early: bool,
}

/// Apply a latency guardrail to an ordered arm set.
///
/// Semantics:
/// - If `cfg.max_mean_ms` is `None`, returns the input arms unchanged.
/// - Otherwise, filters by `mean_elapsed_ms <= max_mean_ms` (and, if `require_measured`, `calls > 0`).
/// - If filtering yields an empty set:
///   - if `cfg.allow_fewer && already_chosen > 0`, return `eligible=[]` with `stop_early=true`
///   - else fall back to the full input set (`fallback_used=true`)
///
/// Returned `eligible` is sorted lexicographically for determinism.
pub fn apply_latency_guardrail(
    arms_in_order: &[String],
    summaries: &BTreeMap<String, Summary>,
    cfg: LatencyGuardrailConfig,
    already_chosen: usize,
) -> LatencyGuardrailDecision {
    let Some(max_ms) = cfg.max_mean_ms else {
        return LatencyGuardrailDecision {
            eligible: arms_in_order.to_vec(),
            fallback_used: false,
            stop_early: false,
        };
    };
    let max_ms = if max_ms.is_finite() && max_ms >= 0.0 {
        max_ms
    } else {
        // Non-finite thresholds are treated as "off".
        return LatencyGuardrailDecision {
            eligible: arms_in_order.to_vec(),
            fallback_used: false,
            stop_early: false,
        };
    };

    let mut eligible: Vec<String> = arms_in_order
        .iter()
        .filter(|a| {
            let s = summaries.get(*a).copied().unwrap_or_default();
            if cfg.require_measured && s.calls == 0 {
                return false;
            }
            s.mean_elapsed_ms() <= max_ms
        })
        .cloned()
        .collect();

    if eligible.is_empty() {
        if cfg.allow_fewer && already_chosen > 0 {
            return LatencyGuardrailDecision {
                eligible,
                fallback_used: false,
                stop_early: true,
            };
        }
        return LatencyGuardrailDecision {
            eligible: arms_in_order.to_vec(),
            fallback_used: true,
            stop_early: false,
        };
    }

    eligible.sort();
    eligible.dedup();

    LatencyGuardrailDecision {
        eligible,
        fallback_used: false,
        stop_early: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(calls: u64, elapsed_ms_sum: u64) -> Summary {
        Summary {
            calls,
            ok: calls,
            junk: 0,
            hard_junk: 0,
            cost_units: 0,
            elapsed_ms_sum,
            mean_quality_score: None,
        }
    }

    #[test]
    fn guardrail_off_returns_all_arms() {
        let arms = vec!["a".to_string(), "b".to_string()];
        let mut m = BTreeMap::new();
        m.insert("a".to_string(), s(1, 10));
        m.insert("b".to_string(), s(1, 10));
        let d = apply_latency_guardrail(&arms, &m, LatencyGuardrailConfig::default(), 0);
        assert_eq!(d.eligible, arms);
        assert!(!d.fallback_used);
        assert!(!d.stop_early);
    }

    #[test]
    fn guardrail_filters_by_mean_latency_and_sorts() {
        let arms = vec!["b".to_string(), "a".to_string(), "c".to_string()];
        let mut m = BTreeMap::new();
        m.insert("a".to_string(), s(2, 100)); // 50ms
        m.insert("b".to_string(), s(2, 10)); // 5ms
        m.insert("c".to_string(), s(2, 1_000)); // 500ms
        let cfg = LatencyGuardrailConfig {
            max_mean_ms: Some(60.0),
            require_measured: false,
            allow_fewer: false,
        };
        let d = apply_latency_guardrail(&arms, &m, cfg, 0);
        assert_eq!(d.eligible, vec!["a".to_string(), "b".to_string()]);
        assert!(!d.fallback_used);
        assert!(!d.stop_early);
    }

    #[test]
    fn guardrail_require_measured_excludes_untried() {
        let arms = vec!["a".to_string(), "b".to_string()];
        let mut m = BTreeMap::new();
        m.insert("a".to_string(), s(0, 0));
        m.insert("b".to_string(), s(1, 10));
        let cfg = LatencyGuardrailConfig {
            max_mean_ms: Some(20.0),
            require_measured: true,
            allow_fewer: false,
        };
        let d = apply_latency_guardrail(&arms, &m, cfg, 0);
        assert_eq!(d.eligible, vec!["b".to_string()]);
    }

    #[test]
    fn guardrail_empty_falls_back_or_stops_early() {
        let arms = vec!["a".to_string(), "b".to_string()];
        let mut m = BTreeMap::new();
        m.insert("a".to_string(), s(1, 1_000)); // 1000ms
        m.insert("b".to_string(), s(1, 1_000)); // 1000ms
        let cfg = LatencyGuardrailConfig {
            max_mean_ms: Some(10.0),
            require_measured: false,
            allow_fewer: true,
        };

        // No previous picks => fallback.
        let d0 = apply_latency_guardrail(&arms, &m, cfg, 0);
        assert_eq!(d0.eligible, arms);
        assert!(d0.fallback_used);
        assert!(!d0.stop_early);

        // After at least one pick => stop early.
        let d1 = apply_latency_guardrail(&arms, &m, cfg, 1);
        assert!(d1.eligible.is_empty());
        assert!(!d1.fallback_used);
        assert!(d1.stop_early);
    }
}
