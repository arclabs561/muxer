//! Ready-to-use policies for the correlated [`crate::Muxer`] lifecycle.
//!
//! The profiles keep the existing statistical kernels available as standalone
//! primitives while giving common applications a complete issue/feedback loop.

/// A finite scalar reward in the closed interval `[0, 1]`.
///
/// It is a separate type from [`bool`], so fractional pseudo-count updates are
/// never mistaken for Bernoulli observations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundedReward(u64);

impl BoundedReward {
    /// Validate a finite reward in the closed unit interval.
    pub fn new(value: f64) -> Result<Self, crate::PolicyError> {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(crate::PolicyError::new(
                "reward must be finite and in [0, 1]",
            ));
        }
        Ok(Self(if value == 0.0 {
            0.0f64.to_bits()
        } else {
            value.to_bits()
        }))
    }

    /// Return the validated numeric reward.
    #[must_use]
    pub fn get(self) -> f64 {
        f64::from_bits(self.0)
    }
}

impl TryFrom<f64> for BoundedReward {
    type Error = crate::PolicyError;
    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

/// A finite scalar reward with no fixed range.
///
/// The optional `BoltzmannProfile` preserves the
/// legacy Boltzmann kernel's optional clipping behavior with this type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FiniteReward(u64);

impl FiniteReward {
    /// Validate a finite scalar reward.
    pub fn new(value: f64) -> Result<Self, crate::PolicyError> {
        if !value.is_finite() {
            return Err(crate::PolicyError::new("reward must be finite"));
        }
        Ok(Self(if value == 0.0 {
            0.0f64.to_bits()
        } else {
            value.to_bits()
        }))
    }
    /// Return the validated numeric reward.
    #[must_use]
    pub fn get(self) -> f64 {
        f64::from_bits(self.0)
    }
}

impl TryFrom<f64> for FiniteReward {
    type Error = crate::PolicyError;
    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

// Checkpoint feedback preserves numeric bits without allowing serde to bypass
// the reward constructors' finite/range invariants.
#[cfg(feature = "serde")]
macro_rules! reward_serde {
    ($reward:ty) => {
        impl serde::Serialize for $reward {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serializer.serialize_u64(self.0)
            }
        }
        impl<'de> serde::Deserialize<'de> for $reward {
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let bits = <u64 as serde::Deserialize>::deserialize(deserializer)?;
                Self::new(f64::from_bits(bits)).map_err(serde::de::Error::custom)
            }
        }
    };
}
#[cfg(feature = "serde")]
reward_serde!(BoundedReward);
#[cfg(feature = "serde")]
reward_serde!(FiniteReward);

#[cfg(feature = "contextual")]
pub(crate) mod contextual;
pub(crate) mod external;
pub mod quality;
#[cfg(any(feature = "stochastic", feature = "boltzmann"))]
pub(crate) mod scalar;

#[cfg(feature = "contextual")]
pub use contextual::{ContextualMode, ContextualProfile};
pub use external::{ExternalAssessments, ExternalDistribution, ExternalScores, ModelReference};
pub use quality::{
    QualityBuildError, QualityFeedback, QualityProfile, QualityProfileBuilder, QualityScore,
};
#[cfg(feature = "boltzmann")]
pub use scalar::BoltzmannProfile;
#[cfg(feature = "stochastic")]
pub use scalar::{BernoulliThompson, Exp3Profile, FractionalThompson};

#[cfg(feature = "stochastic")]
impl crate::Muxer<BernoulliThompson> {
    /// Construct the shortest complete boolean-reward muxer profile.
    pub fn bernoulli(
        actions: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<Self, crate::RuntimeError> {
        Self::new(
            actions.into_iter().map(Into::into).collect(),
            BernoulliThompson::new(crate::ThompsonConfig::default()),
        )
    }
}

pub(crate) fn sample_categorical(
    actions: &[String],
    masses: &std::collections::BTreeMap<String, f64>,
    rng: &mut crate::TrialRng,
) -> Result<(String, crate::Probability), crate::PolicyError> {
    let mut total = 0.0;
    for action in actions {
        let mass = masses
            .get(action)
            .copied()
            .ok_or_else(|| crate::PolicyError::new("distribution omitted an eligible action"))?;
        if !mass.is_finite() || mass < 0.0 {
            return Err(crate::PolicyError::new(
                "distribution has invalid probability",
            ));
        }
        total += mass;
    }
    if !total.is_finite() || total <= 0.0 {
        return Err(crate::PolicyError::new(
            "distribution has zero total probability",
        ));
    }
    let draw = rng.unit_f64();
    let mut cumulative = 0.0;
    let mut last_positive = None;
    for action in actions {
        let probability = masses[action] / total;
        if probability > 0.0 {
            last_positive = Some((action, probability));
        }
        cumulative += probability;
        if draw < cumulative {
            return Ok((
                action.clone(),
                crate::Probability::new(probability)
                    .map_err(|_| crate::PolicyError::new("invalid selected probability"))?,
            ));
        }
    }
    let (action, probability) = last_positive
        .ok_or_else(|| crate::PolicyError::new("distribution has zero total probability"))?;
    Ok((
        action.clone(),
        crate::Probability::new(probability)
            .map_err(|_| crate::PolicyError::new("invalid selected probability"))?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn categorical_never_reports_zero_mass_for_a_zero_mass_last_action() {
        let actions = vec!["positive".to_owned(), "zero".to_owned()];
        let masses = std::collections::BTreeMap::from([
            ("positive".to_owned(), f64::from_bits(1)),
            ("zero".to_owned(), 0.0),
        ]);
        for seed in 0..64 {
            let (_, probability) =
                sample_categorical(&actions, &masses, &mut crate::TrialRng::seeded(seed)).unwrap();
            assert!(probability.get() > 0.0);
        }
    }

    #[test]
    fn categorical_subnormal_equal_masses_follow_the_logged_half_distribution() {
        let actions = vec!["left".to_owned(), "right".to_owned()];
        let masses = std::collections::BTreeMap::from([
            ("left".to_owned(), f64::from_bits(1)),
            ("right".to_owned(), f64::from_bits(1)),
        ]);
        let mut left = 0usize;
        for seed in 0..10_000 {
            let (action, probability) =
                sample_categorical(&actions, &masses, &mut crate::TrialRng::seeded(seed)).unwrap();
            assert_eq!(probability.get(), 0.5);
            left += usize::from(action == "left");
        }
        let frequency = left as f64 / 10_000.0;
        assert!(
            (frequency - 0.5).abs() <= 0.03,
            "fixed-seed frequency was {frequency}"
        );
    }
}
