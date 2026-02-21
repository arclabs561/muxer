//! Utility helpers: window sizing, scaling guidance.

/// Suggest a `Window` capacity based on expected throughput and changepoint rate.
///
/// Uses the SW-UCB scaling `O(sqrt(T / Υ_T))` from Garivier & Moulines 2008
/// (arXiv:0805.3415), where `T` = observations per arm per period and
/// `Υ_T / T = change_rate` = expected fraction of rounds where a change occurs.
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
    // sqrt(T / Υ_T) where Υ_T/T = change_rate, so sqrt(1 / change_rate).
    let cap = (t / change_rate).sqrt().round() as usize;
    cap.clamp(10, 10_000)
}

/// Suggest a window capacity suitable for K arms given a total budget.
///
/// With K arms and a total call budget T, each arm gets roughly T/K calls.
/// This adjusts the per-arm window suggestion for larger arm counts.
///
/// # Example
///
/// ```rust
/// use muxer::suggested_window_cap_for_k;
///
/// // 30 arms, 3000 total calls, expect ~1% change rate.
/// let cap = suggested_window_cap_for_k(30, 3000, 0.01);
/// assert!(cap >= 10);
/// ```
pub fn suggested_window_cap_for_k(k: usize, total_throughput: u64, change_rate: f64) -> usize {
    let k = k.max(1) as u64;
    let per_arm = total_throughput / k;
    suggested_window_cap(per_arm.max(1), change_rate)
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
        assert!(slow >= fast, "more frequent changes → smaller optimal window");
    }

    #[test]
    fn suggested_window_cap_for_k_scales_down_with_more_arms() {
        let few = suggested_window_cap_for_k(2, 1000, 0.05);
        let many = suggested_window_cap_for_k(50, 1000, 0.05);
        assert!(few >= many, "more arms share the same budget → smaller per-arm window");
    }
}
