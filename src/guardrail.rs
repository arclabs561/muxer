/// Latency guardrail configuration.
///
/// This empirical filter excludes arms whose observed mean elapsed time is
/// above `max_mean_ms`. [`crate::PipelineOrder::GuardrailFirst`] makes the
/// result strict within the novelty/coverage/MAB policy stage; Router control
/// and triage picks still run before it. This is not a safety, capability, or
/// readiness constraint. Pass authoritative eligibility to
/// [`crate::Router::select_from`] instead.
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
