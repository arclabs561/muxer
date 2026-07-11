//! Utility helpers: window sizing, scaling guidance.

/// Suggest a `Window` capacity based on expected throughput and changepoint rate.
///
/// This rule of thumb takes the square root of the expected calls between changes:
/// `sqrt(throughput / change_rate)`. It is not the window schedule analyzed for
/// SW-UCB and carries no regret guarantee.
///
/// # Arguments
///
/// - `throughput`: expected number of calls per arm per period (e.g. per day).
/// - `change_rate`: expected fraction of periods in which the arm changes
///   (e.g. `0.01` = one change per 100 periods on average).
///   Must be in `(0, 1]`.
///
/// # Returns
///
/// A suggested window capacity, clamped to `[10, 10_000]`.
///
/// # Example
///
/// ```rust
/// use muxer::suggested_window_cap;
///
/// // 500 calls/day, expect a quality change roughly once a week (1/7 ≈ 0.14).
/// let cap = suggested_window_cap(500, 0.14);
/// assert!(cap >= 10);
/// ```
pub fn suggested_window_cap(throughput: u64, change_rate: f64) -> usize {
    let change_rate = if change_rate.is_finite() && change_rate > 0.0 {
        change_rate.min(1.0)
    } else {
        0.01 // conservative fallback
    };
    let t = (throughput as f64).max(1.0);
    // Square root of the expected calls between changes.
    let cap = (t / change_rate).sqrt().round() as usize;
    cap.clamp(10, 10_000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suggested_window_cap_is_clamped() {
        // Zero throughput → min cap.
        assert_eq!(suggested_window_cap(0, 0.1), 10);
        // Very high throughput + rare changes → max cap.
        assert_eq!(suggested_window_cap(1_000_000, 0.000_01), 10_000);
    }

    #[test]
    fn suggested_window_cap_increases_with_throughput() {
        let low = suggested_window_cap(100, 0.05);
        let high = suggested_window_cap(10_000, 0.05);
        assert!(high >= low, "higher throughput should yield larger window");
    }

    #[test]
    fn suggested_window_cap_decreases_with_higher_change_rate() {
        let slow = suggested_window_cap(1000, 0.01);
        let fast = suggested_window_cap(1000, 0.5);
        assert!(
            slow >= fast,
            "more frequent changes should yield a smaller suggested window"
        );
    }
}
