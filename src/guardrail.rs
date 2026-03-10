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
