//! Unified `BanditPolicy` trait for stateful stochastic policies.
//!
//! [`ThompsonSampling`] and [`Exp3Ix`] both share the same two-method interface:
//! `decide(arms) -> Option<Decision>` and `update_reward(arm, reward)`.
//! This trait makes that explicit and enables generic code over both.
//!
//! `select_mab` is intentionally **not** included: it is a stateless function
//! that operates on caller-maintained `Summary` snapshots and has different
//! update semantics (you push `Outcome`s to a `Window`, not scalar rewards).
//!
//! `LinUcb` is also not included: it requires a per-request feature context
//! vector that doesn't fit the arm-only interface.  Use the `contextual`
//! feature's `LinUcb::decide(arms, context)` directly.

use crate::Decision;

/// Common interface for stateful stochastic bandit policies.
///
/// Both [`ThompsonSampling`][crate::ThompsonSampling] and
/// [`Exp3Ix`][crate::Exp3Ix] implement this trait, enabling generic routing
/// harnesses that can swap between policies without code changes.
///
/// # Example
///
/// ```rust
/// use muxer::{BanditPolicy, Exp3Ix, Exp3IxConfig, ThompsonSampling, ThompsonConfig};
///
/// fn run_policy<P: BanditPolicy>(policy: &mut P, arms: &[String]) {
///     if let Some(d) = policy.decide(arms) {
///         // ... make the call ...
///         policy.update_reward(&d.chosen, 0.8);
///     }
/// }
///
/// let arms = vec!["a".to_string(), "b".to_string()];
/// let mut ts = ThompsonSampling::with_seed(ThompsonConfig::default(), 0);
/// let mut ex = Exp3Ix::new(Exp3IxConfig::default());
///
/// run_policy(&mut ts, &arms);
/// run_policy(&mut ex, &arms);
/// ```
#[cfg(feature = "stochastic")]
pub trait BanditPolicy {
    /// Select an arm, returning a [`Decision`] with the chosen arm and
    /// optional probability distribution.
    ///
    /// Returns `None` only if `arms` is empty.
    fn decide(&mut self, arms: &[String]) -> Option<Decision>;

    /// Update the policy with a scalar reward for `arm`.
    ///
    /// The reward should be in `[0.0, 1.0]`.  Values outside this range are
    /// accepted but may degrade numerical stability for some policies.
    fn update_reward(&mut self, arm: &str, reward: f64);
}

#[cfg(feature = "stochastic")]
impl BanditPolicy for crate::ThompsonSampling {
    fn decide(&mut self, arms: &[String]) -> Option<Decision> {
        self.decide(arms)
    }
    fn update_reward(&mut self, arm: &str, reward: f64) {
        self.update_reward(arm, reward);
    }
}

#[cfg(feature = "stochastic")]
impl BanditPolicy for crate::Exp3Ix {
    fn decide(&mut self, arms: &[String]) -> Option<Decision> {
        self.decide(arms)
    }
    fn update_reward(&mut self, arm: &str, reward: f64) {
        self.update_reward(arm, reward);
    }
}

#[cfg(all(test, feature = "stochastic"))]
mod tests {
    use super::*;
    use crate::{Exp3Ix, Exp3IxConfig, ThompsonConfig, ThompsonSampling};

    fn arms() -> Vec<String> {
        vec!["a".to_string(), "b".to_string(), "c".to_string()]
    }

    fn run_generic<P: BanditPolicy>(p: &mut P) {
        let a = arms();
        for _ in 0..10 {
            if let Some(d) = p.decide(&a) {
                p.update_reward(&d.chosen, 0.7);
            }
        }
    }

    #[test]
    fn thompson_implements_bandit_policy() {
        let mut ts = ThompsonSampling::with_seed(ThompsonConfig::default(), 0);
        run_generic(&mut ts);
    }

    #[test]
    fn exp3ix_implements_bandit_policy() {
        let mut ex = Exp3Ix::new(Exp3IxConfig::default());
        run_generic(&mut ex);
    }

    #[test]
    fn bandit_policy_decide_returns_member_of_arms() {
        let a = arms();
        let mut ts = ThompsonSampling::with_seed(ThompsonConfig::default(), 42);
        for _ in 0..20 {
            let d = ts.decide(&a).unwrap();
            assert!(a.contains(&d.chosen));
            ts.update_reward(&d.chosen, 0.5);
        }
    }

    #[test]
    fn bandit_policy_returns_none_on_empty_arms() {
        let mut ts = ThompsonSampling::with_seed(ThompsonConfig::default(), 0);
        assert!(ts.decide(&[]).is_none());
    }
}
