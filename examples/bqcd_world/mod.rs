use crate::common::normalize;

/// Shared “toy world” for the BQCD sampling experiments.
///
/// Returns `(p0, p1)` where:
/// - `p0[k]` is the per-arm baseline categorical distribution
/// - `p1[k]` is the per-arm post-change categorical distribution (arm-specific severities)
pub fn make_world() -> ([[f64; 4]; 6], [[f64; 4]; 6]) {
    // Baselines: subtly different arms.
    let p0 = [
        normalize([0.90, 0.03, 0.02, 0.05]),
        normalize([0.88, 0.04, 0.02, 0.06]),
        normalize([0.92, 0.02, 0.01, 0.05]),
        normalize([0.86, 0.05, 0.03, 0.06]),
        normalize([0.91, 0.03, 0.01, 0.05]),
        normalize([0.89, 0.03, 0.03, 0.05]),
    ];

    // Post-change distributions (arm-specific severities).
    let p1 = [
        normalize([0.20, 0.10, 0.20, 0.50]),
        normalize([0.40, 0.08, 0.12, 0.40]),
        normalize([0.75, 0.06, 0.04, 0.15]),
        normalize([0.78, 0.06, 0.05, 0.11]),
        normalize([0.80, 0.05, 0.04, 0.11]),
        normalize([0.82, 0.05, 0.03, 0.10]),
    ];

    (p0, p1)
}
