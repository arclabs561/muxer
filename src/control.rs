//! Seeded comparison-arm selection.
//!
//! A control budget reserves part of a multi-pick decision for a subset ranked by
//! stable hash of the caller's seed. The result is reproducible, not intrinsically
//! random. It has no statistical inclusion-probability or unbiasedness guarantee
//! unless the caller supplies seeds from a defined randomization mechanism.
//!
//! Reusing one seed can select the same subset indefinitely. Use
//! [`crate::CoverageConfig`] when the requirement is an empirical sampling floor.

use crate::pick_random_subset;

/// Configuration for control arm reservation.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Default)]
pub struct ControlConfig {
    /// Enable control arm selection.
    pub enabled: bool,
    /// Number of control picks per `pick_control_arms` call.
    ///
    /// Typically 0–2 for small K; up to `k/10` for large K.
    pub control_k: usize,
}

impl ControlConfig {
    /// Create a config with `control_k` enabled.
    #[must_use]
    pub fn with_k(control_k: usize) -> Self {
        Self {
            enabled: control_k > 0,
            control_k,
        }
    }

    /// Fraction of a k-pick budget to reserve as control, rounded up.
    ///
    /// Example: `ControlConfig::fraction(0.1)` with k=10 → control_k=1.
    #[must_use]
    pub fn fraction(frac: f64, total_k: usize) -> Self {
        let k = ((frac * total_k as f64).ceil() as usize).min(total_k);
        Self {
            enabled: k > 0,
            control_k: k,
        }
    }
}

/// Pick up to `cfg.control_k` arms by seeded stable-hash order.
///
/// Returns a subset of `arms` chosen without replacement using `stable_hash64`.
/// The result is deterministic: same seed + same arms → same control picks.
///
/// Control picks should be excluded from the MAB selection budget:
///
/// ```rust
/// use muxer::{ControlConfig, pick_control_arms};
///
/// let arms = vec!["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string()];
/// let cfg = ControlConfig::with_k(1);
/// let controls = pick_control_arms(42, &arms, cfg);
/// assert_eq!(controls.len(), 1);
/// assert!(arms.contains(&controls[0]));
/// ```
#[must_use]
pub fn pick_control_arms(seed: u64, arms: &[String], cfg: ControlConfig) -> Vec<String> {
    if !cfg.enabled || cfg.control_k == 0 || arms.is_empty() {
        return Vec::new();
    }
    pick_random_subset(seed ^ 0xC0E1_1A11, arms, cfg.control_k)
}

/// Split a k-pick budget into control and MAB allocations.
///
/// Returns `(control_picks, mab_k)` where `mab_k = k - control_picks.len()`.
/// `mab_k` is guaranteed to be at least 1 if `k >= 1` and there are arms remaining
/// after removing controls.
#[must_use]
pub fn split_control_budget(
    seed: u64,
    arms: &[String],
    k: usize,
    cfg: ControlConfig,
) -> (Vec<String>, usize) {
    if !cfg.enabled || k == 0 || arms.is_empty() {
        return (Vec::new(), k);
    }
    let max_control = cfg.control_k.min(k.saturating_sub(1)); // always leave room for ≥1 MAB pick
    let control_cfg = ControlConfig {
        control_k: max_control,
        ..cfg
    };
    let controls = pick_control_arms(seed, arms, control_cfg);
    let mab_k = k.saturating_sub(controls.len());
    (controls, mab_k)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arms(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("arm{i}")).collect()
    }

    #[test]
    fn pick_control_arms_disabled() {
        assert!(pick_control_arms(42, &arms(5), ControlConfig::default()).is_empty());
    }

    #[test]
    fn pick_control_arms_respects_k() {
        let picks = pick_control_arms(42, &arms(10), ControlConfig::with_k(3));
        assert_eq!(picks.len(), 3);
    }

    #[test]
    fn pick_control_arms_subset_of_arms() {
        let a = arms(6);
        let picks = pick_control_arms(99, &a, ControlConfig::with_k(2));
        for p in &picks {
            assert!(a.contains(p));
        }
    }

    #[test]
    fn pick_control_arms_deterministic() {
        let a = arms(8);
        let cfg = ControlConfig::with_k(3);
        assert_eq!(pick_control_arms(7, &a, cfg), pick_control_arms(7, &a, cfg));
    }

    #[test]
    fn pick_control_arms_unique() {
        let a = arms(10);
        let picks = pick_control_arms(1, &a, ControlConfig::with_k(5));
        let mut s = picks.clone();
        s.sort();
        s.dedup();
        assert_eq!(s.len(), picks.len(), "picks must be unique");
    }

    #[test]
    fn split_control_always_leaves_mab_budget() {
        let a = arms(5);
        let (ctrl, mab_k) = split_control_budget(0, &a, 3, ControlConfig::with_k(10));
        // control_k=10 would consume all 3 picks, but we cap at k-1=2 to leave ≥1 for MAB.
        assert!(mab_k >= 1, "mab_k={mab_k}");
        assert!(ctrl.len() < 3, "controls={ctrl:?}");
    }

    #[test]
    fn control_config_fraction() {
        let cfg = ControlConfig::fraction(0.1, 20);
        assert_eq!(cfg.control_k, 2);
        assert!(cfg.enabled);
    }
}
