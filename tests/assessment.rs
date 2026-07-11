use muxer::{select_candidate_assessments, CandidateAssessment, MetricObjective};

#[test]
fn caller_defined_metrics_can_trade_utility_against_latency() {
    let assessments = vec![
        CandidateAssessment::new("accurate", 100, vec![0.95, 240.0]),
        CandidateAssessment::new("fast", 100, vec![0.90, 80.0]),
    ];
    let objectives = [
        MetricObjective::maximize(0, 40.0),
        MetricObjective::minimize(1, 0.01),
    ];

    let selection = select_candidate_assessments(&assessments, &objectives).unwrap();

    assert_eq!(selection.chosen.as_deref(), Some("accurate"));
    assert_eq!(selection.frontier, vec!["accurate", "fast"]);
    assert_eq!(selection.candidates.len(), 2);
}

#[test]
fn generic_selector_rejects_a_metric_vector_with_missing_dimensions() {
    let assessments = [
        CandidateAssessment::new("a", 1, vec![1.0]),
        CandidateAssessment::new("b", 1, vec![]),
    ];
    let objectives = [MetricObjective::maximize(0, 1.0)];

    assert!(select_candidate_assessments(&assessments, &objectives).is_err());
}

#[cfg(feature = "serde")]
#[test]
fn generic_assessments_and_selection_round_trip_with_serde() {
    let assessments = vec![CandidateAssessment::new("a", 3, vec![0.8, 12.0])];
    let objectives = [
        MetricObjective::maximize(0, 1.0),
        MetricObjective::minimize(1, 0.01),
    ];
    let selection = select_candidate_assessments(&assessments, &objectives).unwrap();

    let encoded = serde_json::to_string(&selection).unwrap();
    let decoded: muxer::CandidateAssessmentSelection = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded.chosen, selection.chosen);
    assert_eq!(decoded.frontier, selection.frontier);
    assert_eq!(decoded.candidates[0].metrics, assessments[0].metrics);
}
