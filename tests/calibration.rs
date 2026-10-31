//! Integration tests for CUSUM threshold calibration.
//!
//! These use small `n_trials` / `m` to keep CI fast. They test deterministic helper
//! behavior, not an end-to-end false-alarm guarantee.

use muxer::monitor::{calibrate_threshold_from_max_scores, ThresholdCalibration};

#[cfg(feature = "stochastic")]
use muxer::monitor::{calibrate_cusum_threshold, simulate_cusum_null_max_scores};

// ---------------------------------------------------------------------------
// simulate_cusum_null_max_scores
// ---------------------------------------------------------------------------

#[cfg(feature = "stochastic")]
#[test]
fn null_max_scores_returns_n_trials_values() {
    let p0 = vec![0.85, 0.05, 0.05, 0.05];
    let alts = vec![vec![0.40, 0.10, 0.40, 0.10]];
    let scores = simulate_cusum_null_max_scores(&p0, &alts, 100, 1e-3, 5, 50, 42).unwrap();
    assert_eq!(scores.len(), 50);
}

#[cfg(feature = "stochastic")]
#[test]
fn null_max_scores_are_non_negative() {
    let p0 = vec![0.85, 0.05, 0.05, 0.05];
    let alts = vec![vec![0.40, 0.10, 0.40, 0.10]];
    let scores = simulate_cusum_null_max_scores(&p0, &alts, 50, 1e-3, 5, 100, 7).unwrap();
    for s in &scores {
        assert!(s.is_finite() && *s >= 0.0, "score {s} is invalid");
    }
}

#[cfg(feature = "stochastic")]
#[test]
fn null_max_scores_are_deterministic() {
    let p0 = vec![0.85, 0.05, 0.05, 0.05];
    let alts = vec![vec![0.40, 0.10, 0.40, 0.10]];
    let a = simulate_cusum_null_max_scores(&p0, &alts, 50, 1e-3, 5, 80, 99).unwrap();
    let b = simulate_cusum_null_max_scores(&p0, &alts, 50, 1e-3, 5, 80, 99).unwrap();
    assert_eq!(a, b, "same seed must give same scores");
}

// ---------------------------------------------------------------------------
// calibrate_cusum_threshold
// ---------------------------------------------------------------------------

#[cfg(feature = "stochastic")]
#[test]
fn calibrate_threshold_produces_valid_output() {
    let p0 = vec![0.85, 0.05, 0.05, 0.05];
    let alts = vec![vec![0.40, 0.10, 0.40, 0.10]];
    // Small n_trials for CI speed.
    let cal = calibrate_cusum_threshold(&p0, &alts, 0.10, 200, 300, 1e-3, 10, 42, false)
        .expect("calibration should succeed");
    assert!(cal.threshold > 0.0, "threshold must be positive");
    assert!(
        cal.fa_hat >= 0.0 && cal.fa_hat <= 1.0,
        "fa_hat must be a rate"
    );
    assert!(
        cal.fa_wilson_hi >= cal.fa_hat,
        "Wilson hi must be >= empirical"
    );
    assert_eq!(cal.trials, 300);
}

#[cfg(feature = "stochastic")]
#[test]
fn calibrate_threshold_stricter_alpha_gives_higher_threshold() {
    let p0 = vec![0.85, 0.05, 0.05, 0.05];
    let alts = vec![vec![0.40, 0.10, 0.40, 0.10]];
    let cal_loose =
        calibrate_cusum_threshold(&p0, &alts, 0.20, 200, 300, 1e-3, 10, 42, false).unwrap();
    let cal_strict =
        calibrate_cusum_threshold(&p0, &alts, 0.05, 200, 300, 1e-3, 10, 42, false).unwrap();
    // Stricter alpha → fewer false alarms → higher threshold (or equal at boundary).
    assert!(
        cal_strict.threshold >= cal_loose.threshold - 1e-9,
        "strict α={} threshold={:.3} should be ≥ loose α={} threshold={:.3}",
        0.05,
        cal_strict.threshold,
        0.20,
        cal_loose.threshold
    );
}

// ---------------------------------------------------------------------------
// calibrate_threshold_from_max_scores (deterministic, no stochastic feature)
// ---------------------------------------------------------------------------

#[test]
fn calibrate_from_scores_finds_correct_threshold() {
    // Known scores: 1.0, 2.0, 3.0, 4.0, 5.0.  alpha=0.2 → top 20% = 1 of 5 → threshold=5.
    let mut scores = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let grid = vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5];
    let cal = calibrate_threshold_from_max_scores(&mut scores, &grid, 0.2, 1.96, false);
    // fa at 4.5: only score 5.0 >= 4.5 → fa=1/5=0.2.
    assert!(cal.grid_satisfied);
    assert!((cal.fa_hat - 0.2).abs() < 1e-9, "fa_hat={}", cal.fa_hat);
    assert_eq!(cal.threshold, 4.5);
}

#[test]
fn calibrate_from_scores_empty_grid_returns_unsatisfied() {
    let mut scores = vec![1.0, 2.0];
    let cal: ThresholdCalibration =
        calibrate_threshold_from_max_scores(&mut scores, &[], 0.1, 1.96, false);
    assert!(!cal.grid_satisfied);
    assert_eq!(cal.threshold, 0.0);
}

#[test]
fn calibrate_from_scores_impossible_constraint_not_satisfied() {
    // All scores > every grid point → fa_hat always high → alpha=0.001 won't be satisfied.
    let mut scores: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
    let grid = vec![0.1, 0.2, 0.3];
    let cal = calibrate_threshold_from_max_scores(&mut scores, &grid, 0.001, 1.96, false);
    assert!(!cal.grid_satisfied);
}

// ---------------------------------------------------------------------------
// docs/QUICKSTART.md monitoring configuration
// ---------------------------------------------------------------------------

/// The "Monitoring for regressions" block in `docs/QUICKSTART.md`. Keep the
/// two in sync.
#[cfg(feature = "stochastic")]
fn quickstart_monitoring_config() -> muxer::RouterConfig {
    use muxer::{RouterConfig, TriageSessionConfig};

    let mut triage = TriageSessionConfig::default();
    let alts: Vec<Vec<f64>> = triage.alts.iter().map(|alt| alt.to_vec()).collect();
    // At most 5% false alarms over 500 observations of an unchanged arm.
    let calibration = calibrate_cusum_threshold(
        &triage.p0,
        &alts,
        0.05,
        500,
        2000,
        triage.cusum_alpha,
        triage.min_n,
        7,
        false,
    )
    .expect("calibration");
    triage.threshold = calibration.threshold;
    RouterConfig::default()
        .with_monitoring(400, 80)
        .with_triage_cfg(triage)
}

/// Stationary arms drawn from the configured null `p0` must rarely alarm.
/// The default threshold alarms on most such runs within 500 observations.
#[cfg(feature = "stochastic")]
#[test]
fn quickstart_monitoring_does_not_false_alarm_on_stationary_arms() {
    use muxer::{Outcome, Router, TriageSessionConfig};

    let config = quickstart_monitoring_config();
    let p0 = TriageSessionConfig::default().p0;
    let (trials, horizon) = (400, 500);
    let mut state = 0x5EED_u64;
    let mut alarmed = 0;
    for _ in 0..trials {
        let mut router = Router::new(vec!["a".to_owned()], config.clone()).unwrap();
        for _ in 0..horizon {
            // SplitMix64, independent of the calibration seed.
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            let u = ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64;
            let (ok, junk, hard_junk) = if u < p0[0] {
                (true, false, false)
            } else if u < p0[0] + p0[1] {
                (true, true, false)
            } else if u < p0[0] + p0[1] + p0[2] {
                (true, true, true)
            } else {
                (false, false, false)
            };
            assert!(router.observe("a", Outcome::new(ok, junk, hard_junk, 1, 10)));
        }
        if router.mode().is_triage() {
            alarmed += 1;
        }
    }
    let rate = f64::from(alarmed) / f64::from(trials);
    // alpha plus three binomial standard errors at n = 400.
    assert!(
        rate <= 0.05 + 3.0 * (0.05_f64 * 0.95 / 400.0).sqrt(),
        "rate={rate}"
    );
}
