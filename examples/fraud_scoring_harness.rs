//! Fraud-scoring harness example (payments risk), offline and deterministic.
//!
//! This models a realistic risk-routing loop:
//! - slices are `(channel, region, segment)` traffic cells,
//! - scorer eligibility depends on feature availability by slice,
//! - muxer picks a small scorer set with coverage + guardrails,
//! - outcomes are fed back into per-slice windows.
//!
//! Run:
//! `cargo run --example fraud_scoring_harness`

use muxer::{
    coverage_pick_under_sampled, policy_fill_k_observed_with_coverage,
    select_k_without_replacement_by, select_mab_explain, stable_hash64, CoverageConfig,
    LatencyGuardrail, MabConfig, Outcome, Summary, Window,
};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct FraudSlice {
    channel: &'static str,
    region: &'static str,
    segment: &'static str,
}

impl FraudSlice {
    fn id(self) -> String {
        format!(
            "channel={}.region={}.segment={}",
            self.channel, self.region, self.segment
        )
    }
}

fn cell_key(scorer: &str, slice: FraudSlice) -> String {
    format!("{scorer}@@{}", slice.id())
}

fn compatible_scorers(slice: FraudSlice) -> Vec<String> {
    let mut out = vec![
        "rules_engine".to_string(),
        "gbdt_fraud".to_string(),
        "graph_fraud".to_string(),
        "deep_sequence".to_string(),
    ];

    // ACH often lacks graph features used by graph_fraud.
    if slice.channel == "ach" {
        out.retain(|b| b != "graph_fraud");
    }
    // Assume low volume for APAC SMB makes deep sequence less reliable.
    if slice.region == "apac" && slice.segment == "smb" {
        out.retain(|b| b != "deep_sequence");
    }
    out
}

fn summaries_for_slice(
    arms: &[String],
    slice: FraudSlice,
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
    scorer: &str,
    slice: FraudSlice,
    windows: &BTreeMap<String, Window>,
) -> (u64, u64) {
    let k = cell_key(scorer, slice);
    let s = windows.get(&k).map(Window::summary).unwrap_or_default();
    (s.calls, s.elapsed_ms_sum)
}

fn simulated_outcome(round: u64, scorer: &str, slice: FraudSlice) -> Outcome {
    // Drift scenario: rules degrade for EU card-not-present after a policy/pattern shift.
    let policy_shift =
        round >= 62 && scorer == "rules_engine" && slice.region == "eu" && slice.channel == "cnp";

    let base_quality: f64 = match scorer {
        "deep_sequence" => 0.91,
        "graph_fraud" => 0.88,
        "gbdt_fraud" => 0.83,
        "rules_engine" => 0.72,
        _ => 0.50,
    };
    let segment_penalty: f64 = if slice.segment == "smb" { 0.04 } else { 0.0 };
    let shift_penalty: f64 = if policy_shift { 0.30 } else { 0.0 };
    let quality = (base_quality - segment_penalty - shift_penalty).max(0.0_f64);

    let hard_pm_base = match scorer {
        "deep_sequence" => 10,
        "graph_fraud" => 14,
        "gbdt_fraud" => 18,
        "rules_engine" => 26,
        _ => 20,
    };
    let soft_pm_base = match scorer {
        "deep_sequence" => 55,
        "graph_fraud" => 65,
        "gbdt_fraud" => 90,
        "rules_engine" => 130,
        _ => 90,
    };
    let hard_pm = hard_pm_base + if policy_shift { 180 } else { 0 };
    let soft_pm = soft_pm_base + if policy_shift { 260 } else { 0 };

    let h = stable_hash64(round ^ 0xF00D_1001, &format!("{scorer}|{}|h", slice.id())) % 1000;
    let s = stable_hash64(round ^ 0xF00D_1002, &format!("{scorer}|{}|s", slice.id())) % 1000;
    let hard = h < hard_pm;
    let soft = !hard && s < soft_pm;
    let junk = hard || soft;

    let base_ms = match scorer {
        "rules_engine" => 9,
        "gbdt_fraud" => 26,
        "graph_fraud" => 40,
        "deep_sequence" => 76,
        _ => 20,
    };
    let channel_ms = match slice.channel {
        "cnp" => 16,
        "wallet" => 10,
        "ach" => 6,
        _ => 8,
    };
    let jitter = stable_hash64(round ^ 0xF00D_1003, &format!("{scorer}|{}|j", slice.id())) % 20;

    let base_cost = match scorer {
        "rules_engine" => 1,
        "gbdt_fraud" => 3,
        "graph_fraud" => 4,
        "deep_sequence" => 7,
        _ => 2,
    };

    Outcome {
        ok: !hard,
        junk,
        hard_junk: hard,
        cost_units: base_cost,
        elapsed_ms: base_ms + channel_ms + jitter,
        quality_score: Some(if hard {
            0.0
        } else if soft {
            0.18
        } else {
            quality
        }),
    }
}

fn main() {
    let slices = vec![
        FraudSlice {
            channel: "cnp",
            region: "us",
            segment: "consumer",
        },
        FraudSlice {
            channel: "cnp",
            region: "eu",
            segment: "consumer",
        },
        FraudSlice {
            channel: "wallet",
            region: "us",
            segment: "consumer",
        },
        FraudSlice {
            channel: "wallet",
            region: "eu",
            segment: "smb",
        },
        FraudSlice {
            channel: "ach",
            region: "us",
            segment: "smb",
        },
        FraudSlice {
            channel: "ach",
            region: "apac",
            segment: "smb",
        },
    ];

    let ids: Vec<String> = slices.iter().map(|s| s.id()).collect();
    let by_id: BTreeMap<String, FraudSlice> = slices.iter().map(|s| (s.id(), *s)).collect();

    let mut cell_windows: BTreeMap<String, Window> = BTreeMap::new();
    let mut scorer_windows: BTreeMap<String, Window> = BTreeMap::new();
    let mut slice_calls: BTreeMap<String, u64> = BTreeMap::new();

    let slice_coverage = CoverageConfig {
        enabled: true,
        min_fraction: 0.10,
        min_calls_floor: 2,
    };
    let scorer_coverage = CoverageConfig {
        enabled: true,
        min_fraction: 0.12,
        min_calls_floor: 1,
    };
    let guard = LatencyGuardrail {
        max_mean_ms: Some(260.0),
        require_measured: false,
        allow_fewer: true,
    };
    let cfg = MabConfig {
        exploration_c: 0.75,
        junk_weight: 1.0,
        hard_junk_weight: 2.1,
        quality_weight: 1.1,
        latency_weight: 0.0025,
        cost_weight: 0.14,
        ..MabConfig::default()
    };

    for round in 0u64..108 {
        let under =
            coverage_pick_under_sampled(round ^ 0xF00D_2001, &ids, 1, slice_coverage, |sid| {
                slice_calls.get(sid).copied().unwrap_or(0)
            });
        let sid = under
            .first()
            .cloned()
            .unwrap_or_else(|| ids[(round as usize) % ids.len()].clone());
        let slice = *by_id.get(&sid).expect("slice id exists");
        let candidates = compatible_scorers(slice);
        if candidates.is_empty() {
            continue;
        }

        let seed = stable_hash64(round ^ 0xF00D_2002, &sid);
        let fill = policy_fill_k_observed_with_coverage(
            seed,
            &candidates,
            2,
            true,
            scorer_coverage,
            guard,
            |b| observed_calls_and_elapsed(b, slice, &cell_windows),
            |eligible, need| {
                select_k_without_replacement_by(
                    seed ^ 0xF00D_2003,
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

        for scorer in fill.chosen {
            let o = simulated_outcome(round, &scorer, slice);
            cell_windows
                .entry(cell_key(&scorer, slice))
                .or_insert_with(|| Window::new(64))
                .push(o);
            scorer_windows
                .entry(scorer)
                .or_insert_with(|| Window::new(250))
                .push(o);
            *slice_calls.entry(sid.clone()).or_insert(0) += 1;
        }
    }

    println!("== fraud_scoring_harness ==");
    println!("slice coverage:");
    for s in &slices {
        let n = slice_calls.get(&s.id()).copied().unwrap_or(0);
        println!(
            "  channel={:<6} region={:<4} segment={:<8} calls={:>3}",
            s.channel, s.region, s.segment, n
        );
        assert!(n > 0, "each fraud slice should receive traffic");
    }
    println!("scorer aggregate (windowed):");
    for (b, w) in &scorer_windows {
        let s = w.summary();
        println!(
            "  {:<14} calls={:>3} ok={:.3} junk={:.3} hard={:.3} mean_ms={:.1}",
            b,
            s.calls,
            s.ok_rate(),
            s.junk_rate(),
            s.hard_junk_rate(),
            s.mean_elapsed_ms()
        );
    }
}
