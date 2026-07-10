//! Scalar off-policy evaluation estimators.
//!
//! These helpers operate on already-logged rewards and propensities. They do
//! not define a storage format, infer a scalar utility from [`crate::Outcome`],
//! or provide confidence intervals.

use std::error::Error;
use std::fmt;

/// One logged reward with enough propensity information for scalar OPE.
///
/// `logging_propensity` is the probability that the logging policy assigned to
/// the observed action. `target_propensity` is the probability that the policy
/// being evaluated would assign to that same observed action.
///
/// Rewards are caller-defined and only required to be finite. Many contextual
/// bandit inference results assume bounded rewards, often in `[0, 1]`; enforce
/// that at the call site if you need those guarantees.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LoggedReward {
    /// Observed scalar reward for the logged action.
    pub reward: f64,
    /// Probability assigned to the logged action by the logging policy.
    ///
    /// Must be finite and in `(0, 1]`.
    pub logging_propensity: f64,
    /// Probability assigned to the logged action by the target policy.
    ///
    /// Must be finite and in `[0, 1]`. Zero is accepted and contributes zero
    /// weight for that row.
    pub target_propensity: f64,
}

/// Error returned by off-policy evaluation helpers.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum OpeError {
    /// The input contained no rows.
    Empty,
    /// A reward was NaN or infinite.
    InvalidReward {
        /// Zero-based row index.
        index: usize,
        /// Invalid reward value.
        reward: f64,
    },
    /// A logging propensity was not finite or not in `(0, 1]`.
    InvalidLoggingPropensity {
        /// Zero-based row index.
        index: usize,
        /// Invalid logging propensity value.
        propensity: f64,
    },
    /// A target propensity was not finite or not in `[0, 1]`.
    InvalidTargetPropensity {
        /// Zero-based row index.
        index: usize,
        /// Invalid target propensity value.
        propensity: f64,
    },
    /// Dividing target propensity by logging propensity did not produce a finite weight.
    InvalidImportanceWeight {
        /// Zero-based row index.
        index: usize,
        /// Logging propensity used in the weight denominator.
        logging_propensity: f64,
        /// Target propensity used in the weight numerator.
        target_propensity: f64,
    },
    /// Multiplying an importance weight by reward did not produce a finite contribution.
    InvalidWeightedReward {
        /// Zero-based row index.
        index: usize,
        /// Logged reward.
        reward: f64,
        /// Importance weight for this row.
        weight: f64,
    },
    /// Self-normalized IPS has no positive target-policy weight.
    ZeroTotalTargetWeight,
}

impl fmt::Display for OpeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Empty => write!(f, "OPE rows must be non-empty"),
            Self::InvalidReward { index, reward } => {
                write!(f, "OPE row {index} has invalid reward {reward}")
            }
            Self::InvalidLoggingPropensity { index, propensity } => write!(
                f,
                "OPE row {index} has invalid logging propensity {propensity}"
            ),
            Self::InvalidTargetPropensity { index, propensity } => write!(
                f,
                "OPE row {index} has invalid target propensity {propensity}"
            ),
            Self::InvalidImportanceWeight {
                index,
                logging_propensity,
                target_propensity,
            } => write!(
                f,
                "OPE row {index} has non-finite importance weight {target_propensity}/{logging_propensity}"
            ),
            Self::InvalidWeightedReward {
                index,
                reward,
                weight,
            } => write!(
                f,
                "OPE row {index} has non-finite weighted reward {reward} * {weight}"
            ),
            Self::ZeroTotalTargetWeight => {
                write!(f, "self-normalized IPS requires positive target-policy weight")
            }
        }
    }
}

impl Error for OpeError {}

/// Estimate a target policy value with inverse propensity scoring.
///
/// Returns `(1 / n) * sum_i ((target_i / logging_i) * reward_i)`.
///
/// ```
/// use muxer::{ips_value, LoggedReward};
///
/// let rows = [
///     LoggedReward { reward: 1.0, logging_propensity: 0.8, target_propensity: 0.5 },
///     LoggedReward { reward: 0.0, logging_propensity: 0.2, target_propensity: 0.5 },
/// ];
///
/// let value = ips_value(rows).unwrap();
/// assert!(value.is_finite());
/// ```
pub fn ips_value(rows: impl IntoIterator<Item = LoggedReward>) -> Result<f64, OpeError> {
    let mut weighted_reward_sum = 0.0;
    let mut count = 0usize;

    for (index, row) in rows.into_iter().enumerate() {
        let weight = validate_row(index, row)?;
        let weighted_reward = weighted_reward(index, row.reward, weight)?;
        weighted_reward_sum += weighted_reward;
        if !weighted_reward_sum.is_finite() {
            return Err(OpeError::InvalidWeightedReward {
                index,
                reward: row.reward,
                weight,
            });
        }
        count += 1;
    }

    if count == 0 {
        return Err(OpeError::Empty);
    }

    Ok(weighted_reward_sum / count as f64)
}

/// Estimate a target policy value with self-normalized IPS.
///
/// Returns `sum_i (w_i * reward_i) / sum_i w_i`, where
/// `w_i = target_i / logging_i`. This can be more numerically stable than IPS,
/// but it is not the same estimator and can be biased in finite samples.
///
/// ```
/// use muxer::{self_normalized_ips_value, LoggedReward};
///
/// let rows = [
///     LoggedReward { reward: 1.0, logging_propensity: 0.8, target_propensity: 0.5 },
///     LoggedReward { reward: 0.0, logging_propensity: 0.2, target_propensity: 0.5 },
/// ];
///
/// let value = self_normalized_ips_value(rows).unwrap();
/// assert!((0.0..=1.0).contains(&value));
/// ```
pub fn self_normalized_ips_value(
    rows: impl IntoIterator<Item = LoggedReward>,
) -> Result<f64, OpeError> {
    let mut weighted_reward_sum = 0.0;
    let mut weight_sum = 0.0;
    let mut count = 0usize;

    for (index, row) in rows.into_iter().enumerate() {
        let weight = validate_row(index, row)?;
        let weighted_reward = weighted_reward(index, row.reward, weight)?;
        weighted_reward_sum += weighted_reward;
        if !weighted_reward_sum.is_finite() {
            return Err(OpeError::InvalidWeightedReward {
                index,
                reward: row.reward,
                weight,
            });
        }
        weight_sum += weight;
        if !weight_sum.is_finite() {
            return Err(OpeError::InvalidImportanceWeight {
                index,
                logging_propensity: row.logging_propensity,
                target_propensity: row.target_propensity,
            });
        }
        count += 1;
    }

    if count == 0 {
        return Err(OpeError::Empty);
    }
    if weight_sum <= 0.0 {
        return Err(OpeError::ZeroTotalTargetWeight);
    }

    Ok(weighted_reward_sum / weight_sum)
}

fn validate_row(index: usize, row: LoggedReward) -> Result<f64, OpeError> {
    if !row.reward.is_finite() {
        return Err(OpeError::InvalidReward {
            index,
            reward: row.reward,
        });
    }
    if !row.logging_propensity.is_finite()
        || row.logging_propensity <= 0.0
        || row.logging_propensity > 1.0
    {
        return Err(OpeError::InvalidLoggingPropensity {
            index,
            propensity: row.logging_propensity,
        });
    }
    if !row.target_propensity.is_finite()
        || row.target_propensity < 0.0
        || row.target_propensity > 1.0
    {
        return Err(OpeError::InvalidTargetPropensity {
            index,
            propensity: row.target_propensity,
        });
    }

    let weight = row.target_propensity / row.logging_propensity;
    if !weight.is_finite() {
        return Err(OpeError::InvalidImportanceWeight {
            index,
            logging_propensity: row.logging_propensity,
            target_propensity: row.target_propensity,
        });
    }

    Ok(weight)
}

fn weighted_reward(index: usize, reward: f64, weight: f64) -> Result<f64, OpeError> {
    let weighted_reward = reward * weight;
    if !weighted_reward.is_finite() {
        return Err(OpeError::InvalidWeightedReward {
            index,
            reward,
            weight,
        });
    }
    Ok(weighted_reward)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn biased_logged_sample() -> Vec<LoggedReward> {
        let mut rows = Vec::new();
        for _ in 0..8 {
            rows.push(LoggedReward {
                reward: 1.0,
                logging_propensity: 0.8,
                target_propensity: 0.5,
            });
        }
        for _ in 0..2 {
            rows.push(LoggedReward {
                reward: 0.0,
                logging_propensity: 0.2,
                target_propensity: 0.5,
            });
        }
        rows
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "actual={actual} expected={expected}"
        );
    }

    #[test]
    fn ips_corrects_a_biased_logged_sample() {
        let rows = biased_logged_sample();
        let naive = rows.iter().map(|row| row.reward).sum::<f64>() / rows.len() as f64;

        assert_close(naive, 0.8);
        assert_close(ips_value(rows).unwrap(), 0.5);
    }

    #[test]
    fn self_normalized_ips_corrects_a_biased_logged_sample() {
        let rows = biased_logged_sample();
        assert_close(self_normalized_ips_value(rows).unwrap(), 0.5);
    }

    #[test]
    fn zero_target_propensity_contributes_zero_weight() {
        let rows = [
            LoggedReward {
                reward: 10.0,
                logging_propensity: 0.5,
                target_propensity: 0.0,
            },
            LoggedReward {
                reward: 1.0,
                logging_propensity: 1.0,
                target_propensity: 1.0,
            },
        ];

        assert_close(ips_value(rows).unwrap(), 0.5);
        assert_close(self_normalized_ips_value(rows).unwrap(), 1.0);
    }

    #[test]
    fn self_normalized_ips_rejects_zero_total_target_weight() {
        let rows = [LoggedReward {
            reward: 1.0,
            logging_propensity: 0.5,
            target_propensity: 0.0,
        }];

        assert_eq!(
            self_normalized_ips_value(rows),
            Err(OpeError::ZeroTotalTargetWeight)
        );
    }

    #[test]
    fn empty_rows_are_rejected() {
        assert_eq!(ips_value([]), Err(OpeError::Empty));
        assert_eq!(self_normalized_ips_value([]), Err(OpeError::Empty));
    }

    #[test]
    fn invalid_logging_propensity_reports_first_bad_row() {
        let rows = [
            LoggedReward {
                reward: 1.0,
                logging_propensity: 0.5,
                target_propensity: 0.5,
            },
            LoggedReward {
                reward: 1.0,
                logging_propensity: 0.0,
                target_propensity: 0.5,
            },
        ];

        assert_eq!(
            ips_value(rows),
            Err(OpeError::InvalidLoggingPropensity {
                index: 1,
                propensity: 0.0,
            })
        );
    }

    #[test]
    fn invalid_target_propensity_is_rejected() {
        let rows = [LoggedReward {
            reward: 1.0,
            logging_propensity: 0.5,
            target_propensity: 1.1,
        }];

        assert!(matches!(
            ips_value(rows),
            Err(OpeError::InvalidTargetPropensity { index: 0, .. })
        ));
    }

    #[test]
    fn invalid_reward_is_rejected() {
        let rows = [LoggedReward {
            reward: f64::NAN,
            logging_propensity: 0.5,
            target_propensity: 0.5,
        }];

        assert!(matches!(
            ips_value(rows),
            Err(OpeError::InvalidReward { index: 0, .. })
        ));
    }

    #[test]
    fn non_finite_weighted_reward_is_rejected() {
        let rows = [LoggedReward {
            reward: f64::MAX,
            logging_propensity: 1e-308,
            target_propensity: 1.0,
        }];

        assert!(matches!(
            ips_value(rows),
            Err(OpeError::InvalidWeightedReward { index: 0, .. })
        ));
    }
}
