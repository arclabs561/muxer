use muxer::{
    policy_fill_k_observed_guardrail_first_with_coverage, policy_fill_k_observed_with_coverage,
    CoverageConfig, LatencyGuardrailConfig,
};

fn main() {
    println!("== guardrail semantics demo ==");
    println!("This compares:");
    println!("- soft pipeline: novelty/coverage pre-picks happen BEFORE guardrail");
    println!("- guardrail-first: strict guardrail happens BEFORE novelty/coverage\n");

    // --- Case 1: novelty vs require_measured ---
    {
        let arms = vec!["unseen".to_string(), "seen".to_string()];
        let observed_calls = |arm: &str| -> (u64, u64) {
            match arm {
                "unseen" => (0, 0),
                "seen" => (10, 0), // mean=0ms, measured
                _ => (0, 0),
            }
        };
        let guard = LatencyGuardrailConfig {
            max_mean_ms: Some(1.0),
            require_measured: true,
            allow_fewer: true,
        };

        let soft = policy_fill_k_observed_with_coverage(
            123,
            &arms,
            1,
            true, // novelty enabled
            CoverageConfig::default(),
            guard,
            observed_calls,
            |_eligible, _k| panic!("soft pipeline should fill k via novelty here"),
        );

        let hard = policy_fill_k_observed_guardrail_first_with_coverage(
            123,
            &arms,
            1,
            true, // novelty enabled
            CoverageConfig::default(),
            guard,
            observed_calls,
            |eligible, _k| vec![eligible[0].clone()],
        );

        println!("-- novelty + require_measured --");
        println!("soft (novelty before guardrail): chosen={:?}", soft.chosen);
        println!(
            "hard (guardrail first, strict): chosen={:?}, stopped_early={}",
            hard.chosen, hard.stopped_early
        );
        println!();
    }

    // --- Case 2: coverage vs require_measured ---
    {
        let arms = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let observed_calls = |arm: &str| -> (u64, u64) {
            match arm {
                "a" => (100, 0),
                "b" => (100, 0),
                "c" => (0, 0), // unmeasured
                _ => (0, 0),
            }
        };
        let guard = LatencyGuardrailConfig {
            max_mean_ms: Some(1.0),
            require_measured: true,
            allow_fewer: true,
        };
        let cov = CoverageConfig {
            enabled: true,
            min_fraction: 0.0,
            min_calls_floor: 1,
        };

        let soft = policy_fill_k_observed_with_coverage(
            123,
            &arms,
            1,
            false, // novelty disabled
            cov,
            guard,
            observed_calls,
            |_eligible, _k| panic!("soft pipeline should fill k via coverage here"),
        );

        let hard = policy_fill_k_observed_guardrail_first_with_coverage(
            123,
            &arms,
            1,
            false, // novelty disabled
            cov,
            guard,
            observed_calls,
            |eligible, _k| vec![eligible[0].clone()],
        );

        println!("-- coverage + require_measured --");
        println!("soft (coverage before guardrail): chosen={:?}", soft.chosen);
        println!(
            "hard (guardrail first, strict): chosen={:?}, stopped_early={}",
            hard.chosen, hard.stopped_early
        );
        println!();
    }
}
