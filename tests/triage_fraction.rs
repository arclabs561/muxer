//! Triage share regressions: `triage_fraction` must bound the share of picks
//! that go to alarmed arms, including single picks (k = 1), which is the only
//! size `Muxer<QualityProfile>::decide` requests.

use muxer::{Outcome, Router, RouterConfig, TriageSessionConfig};

fn names(xs: &[&str]) -> Vec<String> {
    xs.iter().map(|s| (*s).to_owned()).collect()
}

fn alarmed_router(fraction: f64) -> Router {
    let cfg = RouterConfig {
        triage_fraction: fraction,
        ..RouterConfig::default().with_triage_cfg(TriageSessionConfig {
            min_n: 5,
            threshold: 2.0,
            ..TriageSessionConfig::default()
        })
    };
    let mut r = Router::new(names(&["good", "bad"]), cfg).unwrap();
    for _ in 0..20 {
        assert!(r.observe("good", Outcome::success(1, 50)));
        assert!(r.observe("bad", Outcome::success(1, 50)));
    }
    for _ in 0..30 {
        assert!(r.observe("bad", Outcome::failure(1, 200)));
    }
    assert!(r.mode().is_triage());
    assert_eq!(r.mode().alarmed_arms(), &["bad".to_owned()]);
    r
}

/// RouterConfig::triage_fraction: "0.5 means half the picks go to worst-first
/// investigation, half go to normal MAB selection". With k = 1 (the only size
/// `Muxer<QualityProfile>::decide` uses) every pick goes to the alarmed arm.
#[test]
fn triage_fraction_bounds_single_pick_share() {
    for fraction in [0.5, 0.1] {
        let r = alarmed_router(fraction);
        let to_bad = (0..200u64)
            .filter(|seed| r.select(1, *seed).primary() == Some("bad"))
            .count();
        let share = to_bad as f64 / 200.0;
        assert!(
            share <= fraction + 0.25,
            "triage_fraction={fraction}: {to_bad}/200 k=1 picks went to the alarmed arm"
        );
    }
}

/// Same defect through the correlated runtime: once a quality-profile arm
/// alarms, every issued decision goes to the failing arm. `Muxer` exposes only
/// `&P`, so the alarm cannot be acknowledged without replacing the policy.
#[test]
fn quality_muxer_keeps_normal_traffic_after_alarm() {
    use muxer::{DecisionReason, Muxer, QualityFeedback, QualityProfile};
    let cfg = RouterConfig::default().with_triage_cfg(TriageSessionConfig {
        min_n: 5,
        threshold: 2.0,
        ..TriageSessionConfig::default()
    });
    let arms = names(&["good", "bad"]);
    let mut profile = QualityProfile::new(arms.clone(), cfg).unwrap();
    for _ in 0..20 {
        assert!(profile.seed("good", Outcome::success(1, 50)));
        assert!(profile.seed("bad", Outcome::success(1, 50)));
    }
    for _ in 0..30 {
        assert!(profile.seed("bad", Outcome::failure(1, 200)));
    }
    let mut muxer = Muxer::new(arms, profile).unwrap();
    let ctx: [f64; 0] = [];
    let mut to_bad = 0;
    let mut triage_reason = 0;
    for _ in 0..100 {
        let d = muxer.decide(&ctx).unwrap();
        let outcome = if d.action() == "bad" {
            to_bad += 1;
            Outcome::failure(1, 200)
        } else {
            Outcome::success(1, 50)
        };
        if d.reason() == DecisionReason::Triage {
            triage_reason += 1;
        }
        muxer
            .tell(d.id(), QualityFeedback::execution(outcome))
            .unwrap();
    }
    assert!(
        to_bad <= 75,
        "{to_bad}/100 decisions (triage reason {triage_reason}) went to the failing arm"
    );
}
