//! Domain-neutral selection over caller-provided metric vectors.

use pare::{Direction, ParetoFrontier};

use crate::finite_or_zero;

/// A caller-provided assessment of one arm.
///
/// `metrics` has no built-in meaning. The caller supplies the values and the
/// [`MetricObjective`] list identifies which positions participate in Pareto
/// filtering and scalarization.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CandidateAssessment {
    /// Arm identifier.
    pub arm: String,
    /// Number of observations represented by this assessment.
    ///
    /// This is diagnostic metadata and does not affect selection.
    pub observations: u64,
    /// Caller-defined metric vector.
    pub metrics: Vec<f64>,
}

impl CandidateAssessment {
    /// Construct an assessment from an arm name, observation count, and metrics.
    #[must_use]
    pub fn new(arm: impl Into<String>, observations: u64, metrics: Vec<f64>) -> Self {
        Self {
            arm: arm.into(),
            observations,
            metrics,
        }
    }
}

/// One caller-defined metric objective.
///
/// `metric` indexes [`CandidateAssessment::metrics`]. Higher values are
/// preferred for [`Direction::Maximize`] and lower values for
/// [`Direction::Minimize`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MetricObjective {
    /// Zero-based metric-vector index.
    pub metric: usize,
    /// Whether larger or smaller values are preferred.
    pub direction: Direction,
    /// Non-negative scalarization weight. Zero keeps the Pareto axis but omits
    /// its contribution to the scalarized tie-break.
    pub weight: f64,
}

impl MetricObjective {
    /// Construct a maximizing metric objective.
    #[must_use]
    pub fn maximize(metric: usize, weight: f64) -> Self {
        Self {
            metric,
            direction: Direction::Maximize,
            weight,
        }
    }

    /// Construct a minimizing metric objective.
    #[must_use]
    pub fn minimize(metric: usize, weight: f64) -> Self {
        Self {
            metric,
            direction: Direction::Minimize,
            weight,
        }
    }
}

/// A resolved metric objective value for one candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MetricObjectiveValue {
    /// Metric-vector index that produced this value.
    pub metric: usize,
    /// Raw caller-provided value.
    pub value: f64,
    /// Value oriented so larger is always better.
    pub pareto_value: f64,
    /// Signed scalarization contribution.
    pub scalar_contribution: f64,
}

/// Debug row returned by [`select_candidate_assessments`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CandidateAssessmentDebug {
    /// Arm identifier.
    pub arm: String,
    /// Diagnostic observation count supplied by the caller.
    pub observations: u64,
    /// Caller-defined metric vector.
    pub metrics: Vec<f64>,
    /// Resolved objective values in objective-list order.
    pub objective_values: Vec<MetricObjectiveValue>,
    /// Total scalarized score.
    pub score: f64,
}

/// Result of generic metric-vector selection.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CandidateAssessmentSelection {
    /// Selected arm, or `None` when the input assessment list is empty.
    pub chosen: Option<String>,
    /// Arms on the Pareto frontier in stable input order.
    pub frontier: Vec<String>,
    /// Resolved candidate rows in stable input order.
    pub candidates: Vec<CandidateAssessmentDebug>,
}

/// Select from caller-provided metric vectors.
///
/// This function does not know about rewards, quality labels, costs, latency,
/// or monitoring categories. Non-empty input requires at least one objective;
/// weights must be finite and non-negative. It then applies Pareto filtering
/// followed by weighted scalarization.
pub fn select_candidate_assessments(
    assessments: &[CandidateAssessment],
    objectives: &[MetricObjective],
) -> Result<CandidateAssessmentSelection, logp::Error> {
    validate_assessments(assessments, objectives)?;
    if assessments.is_empty() {
        return Ok(CandidateAssessmentSelection {
            chosen: None,
            frontier: Vec::new(),
            candidates: Vec::new(),
        });
    }

    let candidates: Vec<CandidateAssessmentDebug> = assessments
        .iter()
        .map(|assessment| {
            let objective_values: Vec<MetricObjectiveValue> = objectives
                .iter()
                .map(|objective| {
                    let value = assessment.metrics[objective.metric];
                    let pareto_value = match objective.direction {
                        Direction::Maximize => value,
                        Direction::Minimize => -value,
                    };
                    let scalar_contribution = match objective.direction {
                        Direction::Maximize => objective.weight * value,
                        Direction::Minimize => -(objective.weight * value),
                    };
                    MetricObjectiveValue {
                        metric: objective.metric,
                        value,
                        pareto_value,
                        scalar_contribution,
                    }
                })
                .collect();
            let score = objective_values
                .iter()
                .map(|value| value.scalar_contribution)
                .sum::<f64>();
            CandidateAssessmentDebug {
                arm: assessment.arm.clone(),
                observations: assessment.observations,
                metrics: assessment.metrics.clone(),
                objective_values,
                score,
            }
        })
        .collect();
    if candidates
        .iter()
        .any(|candidate| !candidate.score.is_finite())
    {
        return Err(logp::Error::Domain(
            "candidate objective scalarization overflows",
        ));
    }

    let frontier_indices = frontier_indices(&candidates, objectives.len());
    let frontier: Vec<String> = frontier_indices
        .iter()
        .filter_map(|index| {
            candidates
                .get(*index)
                .map(|candidate| candidate.arm.clone())
        })
        .collect();
    let chosen = frontier_indices
        .iter()
        .filter_map(|index| candidates.get(*index))
        .fold(
            None,
            |best: Option<&CandidateAssessmentDebug>, candidate| {
                let replace = match best {
                    None => true,
                    Some(current) => {
                        candidate.score > current.score
                            || ((candidate.score - current.score).abs() <= 1e-12
                                && candidate.arm < current.arm)
                    }
                };
                if replace {
                    Some(candidate)
                } else {
                    best
                }
            },
        )
        .map(|candidate| candidate.arm.clone());

    Ok(CandidateAssessmentSelection {
        chosen,
        frontier,
        candidates,
    })
}

fn validate_assessments(
    assessments: &[CandidateAssessment],
    objectives: &[MetricObjective],
) -> Result<(), logp::Error> {
    if !assessments.is_empty() && objectives.is_empty() {
        return Err(logp::Error::Domain(
            "candidate selection requires at least one objective",
        ));
    }
    let mut names = std::collections::BTreeSet::new();
    for assessment in assessments {
        if assessment.arm.is_empty() || !names.insert(assessment.arm.clone()) {
            return Err(logp::Error::Domain(
                "candidate assessments require unique non-empty arm names",
            ));
        }
        if assessment.metrics.iter().any(|value| !value.is_finite()) {
            return Err(logp::Error::Domain(
                "candidate assessment metrics must be finite",
            ));
        }
    }
    for objective in objectives {
        if assessments
            .iter()
            .any(|assessment| objective.metric >= assessment.metrics.len())
        {
            return Err(logp::Error::Domain(
                "candidate objective metric index exceeds assessment dimensions",
            ));
        }
        if !objective.weight.is_finite() || objective.weight < 0.0 {
            return Err(logp::Error::Domain(
                "candidate objective weights must be finite and non-negative",
            ));
        }
        if assessments.iter().any(|assessment| {
            let value = assessment.metrics[objective.metric];
            (objective.weight * value).is_infinite()
        }) {
            return Err(logp::Error::Domain(
                "candidate objective scalarization overflows",
            ));
        }
    }
    Ok(())
}

fn frontier_indices(candidates: &[CandidateAssessmentDebug], dimensions: usize) -> Vec<usize> {
    if dimensions == 0 {
        return (0..candidates.len()).collect();
    }
    let mut frontier = ParetoFrontier::new(vec![Direction::Maximize; dimensions]);
    for (index, candidate) in candidates.iter().enumerate() {
        let values = candidate
            .objective_values
            .iter()
            .map(|value| finite_or_zero(value.pareto_value))
            .collect::<Vec<_>>();
        frontier.push(values, index);
    }
    let mut indices: Vec<usize> = frontier.points().iter().map(|point| point.data).collect();
    indices.sort_unstable();
    if indices.is_empty() {
        (0..candidates.len()).collect()
    } else {
        indices
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_assessments_return_no_choice() {
        let result = select_candidate_assessments(&[], &[]).unwrap();
        assert_eq!(result.chosen, None);
        assert!(result.frontier.is_empty());
    }

    #[test]
    fn metric_vectors_drive_pareto_and_scalarization() {
        let assessments = vec![
            CandidateAssessment::new("cheap", 10, vec![0.8, 1.0]),
            CandidateAssessment::new("accurate", 10, vec![0.95, 5.0]),
        ];
        let objectives = [
            MetricObjective::maximize(0, 1.0),
            MetricObjective::minimize(1, 0.02),
        ];
        let result = select_candidate_assessments(&assessments, &objectives).unwrap();
        assert_eq!(result.chosen.as_deref(), Some("accurate"));
        assert_eq!(result.frontier.len(), 2);
    }

    #[test]
    fn duplicate_names_and_nonfinite_values_are_rejected() {
        let duplicate = vec![
            CandidateAssessment::new("a", 1, vec![1.0]),
            CandidateAssessment::new("a", 1, vec![1.0]),
        ];
        assert!(select_candidate_assessments(&duplicate, &[]).is_err());
        let nonfinite = [CandidateAssessment::new("a", 1, vec![f64::NAN])];
        assert!(select_candidate_assessments(&nonfinite, &[]).is_err());
    }

    #[test]
    fn every_candidate_must_supply_each_objective_dimension() {
        let assessments = [
            CandidateAssessment::new("a", 1, vec![1.0, 2.0]),
            CandidateAssessment::new("b", 1, vec![1.0]),
        ];
        let objectives = [MetricObjective::maximize(1, 1.0)];
        assert!(select_candidate_assessments(&assessments, &objectives).is_err());
    }

    #[test]
    fn scalarization_overflow_is_rejected() {
        let assessments = [CandidateAssessment::new("a", 1, vec![f64::MAX, f64::MAX])];
        let objectives = [
            MetricObjective::maximize(0, 1.0),
            MetricObjective::maximize(1, 1.0),
        ];
        assert!(select_candidate_assessments(&assessments, &objectives).is_err());
    }

    #[test]
    fn nonempty_assessments_require_an_objective() {
        let assessments = [CandidateAssessment::new("a", 1, vec![1.0])];

        assert!(select_candidate_assessments(&assessments, &[]).is_err());
    }

    #[test]
    fn negative_objective_weights_are_rejected() {
        let assessments = [CandidateAssessment::new("a", 1, vec![1.0])];
        let objectives = [MetricObjective::maximize(0, -1.0)];

        assert!(select_candidate_assessments(&assessments, &objectives).is_err());
    }
}
