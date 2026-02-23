//! Medical-triage harness example, offline and deterministic.
//!
//! This models routing among triage/risk backends:
//! - slices are `(setting, acuity, cohort)` cells,
//! - model eligibility depends on data richness by slice,
//! - muxer picks a small model set with coverage + guardrails,
//! - outcomes are fed back into per-slice windows.
//!
//! Run:
//! `cargo run --example medical_triage_harness`

use muxer::{
    coverage_pick_under_sampled, policy_fill_k_observed_with_coverage,
    select_k_without_replacement_by, select_mab_explain, stable_hash64, CoverageConfig,
    LatencyGuardrail, MabConfig, Outcome, Summary, Window,
};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct MedSlice {
    setting: &'static str,
    acuity: &'static str,
    cohort: &'static str,
}

impl MedSlice {
    fn id(self) -> String {
        format!(
            "setting={}.acuity={}.cohort={}",
            self.setting, self.acuity, self.cohort
        )
    }
}

fn cell_key(model: &str, slice: MedSlice) -> String {
    format!("{model}@@{}", slice.id())
}

fn compatible_models(slice: MedSlice) -> Vec<String> {
    let mut out = vec![
        "rules_triage".to_string(),
        "gboost_risk".to_string(),
        "sequence_vitals".to_string(),
        "bayes_fallback".to_string(),
    ];

    // Telehealth often lacks high-frequency vitals needed by sequence model.
    if slice.setting == "telehealth" {
        out.retain(|m| m != "sequence_vitals");
    }
    // Pediatric data path not fully calibrated for boosted model in this toy setup.
    if slice.cohort == "pediatric" && slice.acuity == "high" {
        out.retain(|m| m != "gboost_risk");
    }
    out
}

fn summaries_for_slice(
    arms: &[String],
    slice: MedSlice,
    windows: &BTreeMap<String, Window>,
) -> BTreeMap<String, Summary> {
    arms.iter()
        .map(|b| {
            let k = cell_key(b, slice);
            (
                b.clone(),
                windows.get(&k).map(Window::summary).unwrap_or_default(),
            )
        })
        .collect()
}

fn observed_calls_and_elapsed(
    model: &str,
    slice: MedSlice,
    windows: &BTreeMap<String, Window>,
) -> (u64, u64) {
    let k = cell_key(model, slice);
    let s = windows.get(&k).map(Window::summary).unwrap_or_default();
    (s.calls, s.elapsed_ms_sum)
}

fn simulated_outcome(round: u64, model: &str, slice: MedSlice) -> Outcome {
    // Drift: rules model degrades for high-acuity pediatric ED cases after protocol change.
    let protocol_shift = round >= 66
        && model == "rules_triage"
        && slice.setting == "ed"
        && slice.acuity == "high"
        && slice.cohort == "pediatric";

    let base_quality: f64 = match model {
        "sequence_vitals" => 0.92,
        "gboost_risk" => 0.87,
        "rules_triage" => 0.76,
        "bayes_fallback" => 0.70,
        _ => 0.50,
    };
    let acuity_penalty: f64 = if slice.acuity == "high" { 0.04 } else { 0.0 };
    let shift_penalty: f64 = if protocol_shift { 0.31 } else { 0.0 };
    let quality = (base_quality - acuity_penalty - shift_penalty).max(0.0_f64);

    let hard_pm_base = match model {
        "sequence_vitals" => 9,
        "gboost_risk" => 13,
        "rules_triage" => 22,
        "bayes_fallback" => 18,
        _ => 15,
    };
    let soft_pm_base = match model {
        "sequence_vitals" => 50,
        "gboost_risk" => 66,
        "rules_triage" => 120,
        "bayes_fallback" => 90,
        _ => 80,
    };
    let hard_pm = hard_pm_base + if protocol_shift { 170 } else { 0 };
    let soft_pm = soft_pm_base + if protocol_shift { 240 } else { 0 };

    let h = stable_hash64(round ^ 0x7700_1001, &format!("{model}|{}|h", slice.id())) % 1000;
    let s = stable_hash64(round ^ 0x7700_1002, &format!("{model}|{}|s", slice.id())) % 1000;
    let hard = h < hard_pm;
    let soft = !hard && s < soft_pm;
    let junk = hard || soft;

    let base_ms = match model {
        "rules_triage" => 12,
        "bayes_fallback" => 24,
        "gboost_risk" => 40,
        "sequence_vitals" => 86,
        _ => 20,
    };
    let setting_ms = match slice.setting {
        "ed" => 14,
        "inpatient" => 9,
        "telehealth" => 7,
        _ => 8,
    };
    let jitter = stable_hash64(round ^ 0x7700_1003, &format!("{model}|{}|j", slice.id())) % 22;

    let base_cost = match model {
        "rules_triage" => 1,
        "bayes_fallback" => 2,
        "gboost_risk" => 4,
        "sequence_vitals" => 7,
        _ => 2,
    };

    Outcome {
        ok: !hard,
        junk,
        hard_junk: hard,
        cost_units: base_cost,
        elapsed_ms: base_ms + setting_ms + jitter,
        quality_score: Some(if hard {
            0.0
        } else if soft {
            0.19
        } else {
            quality
        }),
    }
}

fn main() {
    let slices = vec![
        MedSlice {
            setting: "ed",
            acuity: "high",
            cohort: "adult",
        },
        MedSlice {
            setting: "ed",
            acuity: "high",
            cohort: "pediatric",
        },
        MedSlice {
            setting: "inpatient",
            acuity: "medium",
            cohort: "adult",
        },
        MedSlice {
            setting: "telehealth",
            acuity: "low",
            cohort: "adult",
        },
        MedSlice {
            setting: "telehealth",
            acuity: "medium",
            cohort: "pediatric",
        },
    ];

    let ids: Vec<String> = slices.iter().map(|s| s.id()).collect();
    let by_id: BTreeMap<String, MedSlice> = slices.iter().map(|s| (s.id(), *s)).collect();

    let mut cell_windows: BTreeMap<String, Window> = BTreeMap::new();
    let mut model_windows: BTreeMap<String, Window> = BTreeMap::new();
    let mut slice_calls: BTreeMap<String, u64> = BTreeMap::new();

    let slice_coverage = CoverageConfig {
        enabled: true,
        min_fraction: 0.12,
        min_calls_floor: 2,
    };
    let model_coverage = CoverageConfig {
        enabled: true,
        min_fraction: 0.12,
        min_calls_floor: 1,
    };
    let guard = LatencyGuardrail {
        max_mean_ms: Some(280.0),
        require_measured: false,
        allow_fewer: true,
    };
    let cfg = MabConfig {
        exploration_c: 0.80,
        junk_weight: 1.0,
        hard_junk_weight: 2.1,
        quality_weight: 1.1,
        latency_weight: 0.0022,
        cost_weight: 0.12,
        ..MabConfig::default()
    };

    for round in 0u64..102 {
        let under =
            coverage_pick_under_sampled(round ^ 0x7700_2001, &ids, 1, slice_coverage, |sid| {
                slice_calls.get(sid).copied().unwrap_or(0)
            });
        let sid = under
            .first()
            .cloned()
            .unwrap_or_else(|| ids[(round as usize) % ids.len()].clone());
        let slice = *by_id.get(&sid).expect("slice id exists");
        let candidates = compatible_models(slice);
        if candidates.is_empty() {
            continue;
        }

        let seed = stable_hash64(round ^ 0x7700_2002, &sid);
        let fill = policy_fill_k_observed_with_coverage(
            seed,
            &candidates,
            2,
            true,
            model_coverage,
            guard,
            |b| observed_calls_and_elapsed(b, slice, &cell_windows),
            |eligible, need| {
                select_k_without_replacement_by(
                    seed ^ 0x7700_2003,
                    eligible,
                    need,
                    |_s, rem, _k| {
                        let summaries = summaries_for_slice(rem, slice, &cell_windows);
                        let d = select_mab_explain(rem, &summaries, cfg);
                        vec![d.selection.chosen]
                    },
                )
            },
        );

        for model in fill.chosen {
            let o = simulated_outcome(round, &model, slice);
            cell_windows
                .entry(cell_key(&model, slice))
                .or_insert_with(|| Window::new(64))
                .push(o);
            model_windows
                .entry(model)
                .or_insert_with(|| Window::new(240))
                .push(o);
            *slice_calls.entry(sid.clone()).or_insert(0) += 1;
        }
    }

    println!("== medical_triage_harness ==");
    println!("slice coverage:");
    for s in &slices {
        let n = slice_calls.get(&s.id()).copied().unwrap_or(0);
        println!(
            "  setting={:<10} acuity={:<6} cohort={:<10} calls={:>3}",
            s.setting, s.acuity, s.cohort, n
        );
        assert!(n > 0, "each medical slice should receive traffic");
    }
    println!("model aggregate (windowed):");
    for (b, w) in &model_windows {
        let s = w.summary();
        println!(
            "  {:<16} calls={:>3} ok={:.3} junk={:.3} hard={:.3} mean_ms={:.1}",
            b,
            s.calls,
            s.ok_rate(),
            s.junk_rate(),
            s.hard_junk_rate(),
            s.mean_elapsed_ms()
        );
    }
}
