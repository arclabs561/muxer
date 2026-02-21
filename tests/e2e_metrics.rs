#![cfg(feature = "stochastic")]

use muxer::{select_mab_explain, MabConfig, Outcome, StickyConfig, StickyMab, Window};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::collections::BTreeMap;

#[derive(Clone, Copy)]
struct ArmTruth {
    ok_p: f64,
    junk_p: f64,
    hard_junk_p: f64,
    cost_units: u64,
    latency_ms: u64,
}

fn sample_outcome(rng: &mut StdRng, t: ArmTruth) -> (Outcome, bool, bool) {
    let ok = rng.random::<f64>() < t.ok_p;

    let is_junk = ok && (rng.random::<f64>() < t.junk_p);
    let is_hard = is_junk && (rng.random::<f64>() < t.hard_junk_p);

    let o = Outcome {
        ok,
        junk: false,
        hard_junk: false,
        cost_units: t.cost_units.saturating_add(rng.random_range(0..=1)),
        elapsed_ms: t.latency_ms.saturating_add(rng.random_range(0..=50)),
        quality_score: None,
    };
    (o, is_junk, is_hard)
}

#[test]
fn sticky_bounds_switch_rate_in_stable_environment() {
    let arms = vec![
        "cheap".to_string(),
        "fast".to_string(),
        "reliable".to_string(),
    ];

    let truth: BTreeMap<&'static str, ArmTruth> = BTreeMap::from([
        (
            "cheap",
            ArmTruth {
                ok_p: 0.90,
                junk_p: 0.02,
                hard_junk_p: 0.05,
                cost_units: 1,
                latency_ms: 520,
            },
        ),
        (
            "fast",
            ArmTruth {
                ok_p: 0.92,
                junk_p: 0.07,
                hard_junk_p: 0.20,
                cost_units: 2,
                latency_ms: 260,
            },
        ),
        (
            "reliable",
            ArmTruth {
                ok_p: 0.97,
                junk_p: 0.01,
                hard_junk_p: 0.05,
                cost_units: 4,
                latency_ms: 680,
            },
        ),
    ]);

    let cfg = MabConfig {
        max_hard_junk_rate: Some(0.10),
        cost_weight: 0.20,
        latency_weight: 0.001,
        junk_weight: 1.0,
        hard_junk_weight: 2.0,
        ..MabConfig::default()
    };

    let mut windows: BTreeMap<String, Window> =
        arms.iter().map(|a| (a.clone(), Window::new(60))).collect();

    let mut sticky = StickyMab::new(StickyConfig {
        min_dwell: 10,
        min_switch_margin: 0.02,
    });

    let mut rng = StdRng::seed_from_u64(7);

    let mut prev: Option<String> = None;
    let mut switches = 0u64;
    for _t in 0..800u64 {
        let summaries: BTreeMap<String, _> = windows
            .iter()
            .map(|(k, w)| (k.clone(), w.summary()))
            .collect();
        let base = select_mab_explain(&arms, &summaries, cfg);
        let chosen = sticky.apply_mab(base).selection.chosen;

        if let Some(p) = &prev {
            if p != &chosen {
                switches += 1;
            }
        }
        prev = Some(chosen.clone());

        let tr = *truth.get(chosen.as_str()).unwrap();
        let (o, late_junk, late_hard) = sample_outcome(&mut rng, tr);
        let w = windows.get_mut(&chosen).unwrap();
        w.push(o);
        w.set_last_junk_level(late_junk, late_hard);
    }

    // We don't want a brittle hard bound; just ensure it's not flapping constantly.
    assert!(switches <= 80, "switches={}", switches);
}

#[test]
fn constraints_hold_in_windowed_summary_when_one_arm_is_bad() {
    let arms = vec!["bad".to_string(), "good".to_string()];

    let cfg = MabConfig {
        max_hard_junk_rate: Some(0.2),
        ..MabConfig::default()
    };

    let mut w_bad = Window::new(50);
    let mut w_good = Window::new(50);

    for _ in 0..50 {
        w_bad.push(Outcome {
            ok: false,
            junk: true,
            hard_junk: true,
            cost_units: 1,
            elapsed_ms: 100,
            quality_score: None,
        });
        w_good.push(Outcome {
            ok: true,
            junk: false,
            hard_junk: false,
            cost_units: 1,
            elapsed_ms: 120,
            quality_score: None,
        });
    }

    let summaries = BTreeMap::from([
        ("bad".to_string(), w_bad.summary()),
        ("good".to_string(), w_good.summary()),
    ]);

    let d = select_mab_explain(&arms, &summaries, cfg);
    assert_eq!(d.selection.chosen, "good");
}
