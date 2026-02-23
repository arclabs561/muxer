//! Ad-auction harness example (recsys/ads), offline and deterministic.
//!
//! This demonstrates muxer in a ranking/auction setting:
//! - slices are `(objective, geo, device)` traffic cells,
//! - backend eligibility and behavior vary by slice,
//! - one model drifts post-privacy-shift on EU mobile traffic,
//! - muxer maintains coverage while adapting backend picks.
//!
//! Run:
//! `cargo run --example ad_auction_harness`

use muxer::{
    coverage_pick_under_sampled, policy_fill_k_observed_with_coverage,
    select_k_without_replacement_by, select_mab_explain, stable_hash64, CoverageConfig,
    LatencyGuardrail, MabConfig, Outcome, Summary, Window,
};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct AdSlice {
    objective: &'static str,
    geo: &'static str,
    device: &'static str,
}

impl AdSlice {
    fn id(self) -> String {
        format!(
            "obj={}.geo={}.dev={}",
            self.objective, self.geo, self.device
        )
    }
}

fn cell_key(model: &str, slice: AdSlice) -> String {
    format!("{model}@@{}", slice.id())
}

fn compatible_models(slice: AdSlice) -> Vec<String> {
    let mut v = vec![
        "lr_ctr".to_string(),
        "gbrt_ctr".to_string(),
        "deep_multitask".to_string(),
        "rules_fallback".to_string(),
    ];

    // Assume deep model does not have sufficient calibrated signal for low-latency branding flows.
    if slice.objective == "brand_reach" && slice.device == "mobile" {
        v.retain(|m| m != "deep_multitask");
    }
    v
}

fn summaries_for_slice(
    arms: &[String],
    slice: AdSlice,
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
    slice: AdSlice,
    windows: &BTreeMap<String, Window>,
) -> (u64, u64) {
    let k = cell_key(model, slice);
    let s = windows.get(&k).map(Window::summary).unwrap_or_default();
    (s.calls, s.elapsed_ms_sum)
}

fn simulated_outcome(round: u64, model: &str, slice: AdSlice) -> Outcome {
    let privacy_shift =
        round >= 58 && slice.geo == "eu" && slice.device == "mobile" && model == "deep_multitask";

    let base_quality: f64 = match model {
        "deep_multitask" => 0.90,
        "gbrt_ctr" => 0.84,
        "lr_ctr" => 0.78,
        "rules_fallback" => 0.64,
        _ => 0.50,
    };
    let objective_penalty: f64 = if slice.objective == "conversion" {
        0.0
    } else {
        0.05
    };
    let shift_penalty: f64 = if privacy_shift { 0.33 } else { 0.0 };
    let quality = (base_quality - objective_penalty - shift_penalty).max(0.0_f64);

    let hard_pm_base = match model {
        "deep_multitask" => 10,
        "gbrt_ctr" => 16,
        "lr_ctr" => 22,
        "rules_fallback" => 30,
        _ => 20,
    };
    let soft_pm_base = match model {
        "deep_multitask" => 70,
        "gbrt_ctr" => 85,
        "lr_ctr" => 110,
        "rules_fallback" => 145,
        _ => 100,
    };
    let hard_pm = hard_pm_base + if privacy_shift { 190 } else { 0 };
    let soft_pm = soft_pm_base + if privacy_shift { 250 } else { 0 };

    let h = stable_hash64(round ^ 0xADAD_AA01, &format!("{model}|{}|h", slice.id())) % 1000;
    let s = stable_hash64(round ^ 0xADAD_BB02, &format!("{model}|{}|s", slice.id())) % 1000;
    let hard = h < hard_pm;
    let soft = !hard && s < soft_pm;
    let junk = hard || soft;

    let base_ms = match model {
        "rules_fallback" => 8,
        "lr_ctr" => 20,
        "gbrt_ctr" => 34,
        "deep_multitask" => 70,
        _ => 25,
    };
    let mobile_ms = if slice.device == "mobile" { 14 } else { 6 };
    let jitter = stable_hash64(round ^ 0xADAD_CC03, &format!("{model}|{}|j", slice.id())) % 18;

    let base_cost = match model {
        "rules_fallback" => 1,
        "lr_ctr" => 2,
        "gbrt_ctr" => 3,
        "deep_multitask" => 6,
        _ => 2,
    };

    Outcome {
        ok: !hard,
        junk,
        hard_junk: hard,
        cost_units: base_cost,
        elapsed_ms: base_ms + mobile_ms + jitter,
        quality_score: Some(if hard {
            0.0
        } else if soft {
            0.2
        } else {
            quality
        }),
    }
}

fn main() {
    let slices = vec![
        AdSlice {
            objective: "conversion",
            geo: "us",
            device: "mobile",
        },
        AdSlice {
            objective: "conversion",
            geo: "us",
            device: "desktop",
        },
        AdSlice {
            objective: "conversion",
            geo: "eu",
            device: "mobile",
        },
        AdSlice {
            objective: "brand_reach",
            geo: "us",
            device: "mobile",
        },
        AdSlice {
            objective: "brand_reach",
            geo: "eu",
            device: "desktop",
        },
    ];

    let ids: Vec<String> = slices.iter().map(|s| s.id()).collect();
    let by_id: BTreeMap<String, AdSlice> = slices.iter().map(|s| (s.id(), *s)).collect();

    let mut cell_windows: BTreeMap<String, Window> = BTreeMap::new();
    let mut model_windows: BTreeMap<String, Window> = BTreeMap::new();
    let mut slice_calls: BTreeMap<String, u64> = BTreeMap::new();

    let slice_coverage = CoverageConfig {
        enabled: true,
        min_fraction: 0.11,
        min_calls_floor: 2,
    };
    let model_coverage = CoverageConfig {
        enabled: true,
        min_fraction: 0.12,
        min_calls_floor: 1,
    };
    let guard = LatencyGuardrail {
        max_mean_ms: Some(220.0),
        require_measured: false,
        allow_fewer: true,
    };
    let cfg = MabConfig {
        exploration_c: 0.75,
        junk_weight: 1.0,
        hard_junk_weight: 2.0,
        quality_weight: 1.1,
        latency_weight: 0.003,
        cost_weight: 0.15,
        ..MabConfig::default()
    };

    for round in 0u64..100 {
        let under =
            coverage_pick_under_sampled(round ^ 0xADA0_0001, &ids, 1, slice_coverage, |sid| {
                slice_calls.get(sid).copied().unwrap_or(0)
            });
        let sid = under
            .first()
            .cloned()
            .unwrap_or_else(|| ids[(round as usize) % ids.len()].clone());
        let slice = *by_id.get(&sid).expect("slice id exists");
        let candidates = compatible_models(slice);

        let seed = stable_hash64(round ^ 0xADA0_0002, &sid);
        let fill = policy_fill_k_observed_with_coverage(
            seed,
            &candidates,
            2,
            true,
            model_coverage,
            guard,
            |m| observed_calls_and_elapsed(m, slice, &cell_windows),
            |eligible, need| {
                select_k_without_replacement_by(
                    seed ^ 0xADA0_0003,
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

    println!("== ad_auction_harness ==");
    println!("slice coverage:");
    for s in &slices {
        let n = slice_calls.get(&s.id()).copied().unwrap_or(0);
        println!(
            "  objective={:<11} geo={:<2} device={:<7} calls={:>3}",
            s.objective, s.geo, s.device, n
        );
        assert!(n > 0, "each ad slice should receive traffic");
    }
    println!("model aggregate (windowed):");
    for (m, w) in &model_windows {
        let s = w.summary();
        println!(
            "  {:<16} calls={:>3} ok={:.3} junk={:.3} hard={:.3} mean_ms={:.1}",
            m,
            s.calls,
            s.ok_rate(),
            s.junk_rate(),
            s.hard_junk_rate(),
            s.mean_elapsed_ms()
        );
    }
}
