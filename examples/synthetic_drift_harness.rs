//! Synthetic drift harness example, offline and deterministic.
//!
//! This is a controlled world useful for tests, demos, and docs:
//! - matrix slices are synthetic workload cells,
//! - one backend drifts on "hard" cells after a known epoch,
//! - muxer should keep measuring alternatives and adapt choices.
//!
//! Run:
//! `cargo run --example synthetic_drift_harness`

use muxer::{
    coverage_pick_under_sampled, policy_fill_k_observed_with_coverage,
    select_k_without_replacement_by, select_mab_explain, stable_hash64, CoverageConfig,
    LatencyGuardrail, MabConfig, Outcome, Summary, Window,
};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct SynthSlice {
    family: &'static str,
    hardness: &'static str,
    region: &'static str,
}

impl SynthSlice {
    fn id(self) -> String {
        format!("{}.{}.{}", self.family, self.hardness, self.region)
    }

    fn is_hard(self) -> bool {
        self.hardness == "hard"
    }
}

fn cell_key(backend: &str, slice: SynthSlice) -> String {
    format!("{backend}@@{}", slice.id())
}

fn candidate_backends(_slice: SynthSlice) -> Vec<String> {
    vec![
        "tiny_rule".to_string(),
        "balanced_nn".to_string(),
        "large_transformer".to_string(),
        "hard_specialist".to_string(),
    ]
}

fn summaries_for_slice(
    arms: &[String],
    slice: SynthSlice,
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
    backend: &str,
    slice: SynthSlice,
    windows: &BTreeMap<String, Window>,
) -> (u64, u64) {
    let k = cell_key(backend, slice);
    let s = windows.get(&k).map(Window::summary).unwrap_or_default();
    (s.calls, s.elapsed_ms_sum)
}

fn synthetic_outcome(round: u64, backend: &str, slice: SynthSlice) -> Outcome {
    let drift_epoch = 64;
    let shifted = round >= drift_epoch && slice.is_hard() && backend == "large_transformer";

    let base_quality: f64 = match backend {
        "large_transformer" => 0.92,
        "hard_specialist" => {
            if slice.is_hard() {
                0.88
            } else {
                0.72
            }
        }
        "balanced_nn" => 0.82,
        "tiny_rule" => 0.66,
        _ => 0.50,
    };
    let drift_penalty = if shifted { 0.35 } else { 0.0 };
    let quality = (base_quality - drift_penalty).max(0.0_f64);

    let hard_pm_base = match backend {
        "large_transformer" => 12,
        "hard_specialist" => 16,
        "balanced_nn" => 26,
        "tiny_rule" => 40,
        _ => 30,
    };
    let soft_pm_base = match backend {
        "large_transformer" => 50,
        "hard_specialist" => 70,
        "balanced_nn" => 95,
        "tiny_rule" => 140,
        _ => 100,
    };
    let hard_pm = hard_pm_base + if shifted { 220 } else { 0 };
    let soft_pm = soft_pm_base + if shifted { 260 } else { 0 };

    let h = stable_hash64(round ^ 0x5150_AAED, &format!("{backend}|{}|h", slice.id())) % 1000;
    let s = stable_hash64(round ^ 0xD00D_5150, &format!("{backend}|{}|s", slice.id())) % 1000;
    let hard = h < hard_pm;
    let soft = !hard && s < soft_pm;
    let junk = hard || soft;

    let base_latency = match backend {
        "tiny_rule" => 35,
        "balanced_nn" => 85,
        "large_transformer" => 210,
        "hard_specialist" => 180,
        _ => 80,
    };
    let hardness_latency = if slice.is_hard() { 95 } else { 20 };
    let jitter = stable_hash64(round ^ 0x1234_7788, &format!("{backend}|{}|j", slice.id())) % 45;

    let base_cost = match backend {
        "tiny_rule" => 1,
        "balanced_nn" => 3,
        "large_transformer" => 7,
        "hard_specialist" => 5,
        _ => 2,
    };
    let hard_cost = if slice.is_hard() { 2 } else { 0 };

    Outcome {
        ok: !hard,
        junk,
        hard_junk: hard,
        cost_units: base_cost + hard_cost,
        elapsed_ms: base_latency + hardness_latency + jitter,
        quality_score: Some(if hard {
            0.0
        } else if soft {
            0.22
        } else {
            quality
        }),
    }
}

fn main() {
    let slices = vec![
        SynthSlice {
            family: "vision",
            hardness: "easy",
            region: "us",
        },
        SynthSlice {
            family: "vision",
            hardness: "hard",
            region: "us",
        },
        SynthSlice {
            family: "vision",
            hardness: "hard",
            region: "eu",
        },
        SynthSlice {
            family: "qa",
            hardness: "easy",
            region: "us",
        },
        SynthSlice {
            family: "qa",
            hardness: "hard",
            region: "eu",
        },
        SynthSlice {
            family: "ranking",
            hardness: "easy",
            region: "eu",
        },
        SynthSlice {
            family: "ranking",
            hardness: "hard",
            region: "us",
        },
    ];

    let slice_ids: Vec<String> = slices.iter().map(|s| s.id()).collect();
    let by_id: BTreeMap<String, SynthSlice> = slices.iter().map(|s| (s.id(), *s)).collect();

    let mut cell_windows: BTreeMap<String, Window> = BTreeMap::new();
    let mut slice_calls: BTreeMap<String, u64> = BTreeMap::new();
    let mut pre_hard_counts: BTreeMap<String, u64> = BTreeMap::new();
    let mut post_hard_counts: BTreeMap<String, u64> = BTreeMap::new();

    let slice_coverage = CoverageConfig {
        enabled: true,
        min_fraction: 0.08,
        min_calls_floor: 2,
    };
    let backend_coverage = CoverageConfig {
        enabled: true,
        min_fraction: 0.12,
        min_calls_floor: 1,
    };
    let guard = LatencyGuardrail {
        max_mean_ms: Some(1_100.0),
        require_measured: false,
        allow_fewer: true,
    };
    let mab_cfg = MabConfig {
        exploration_c: 0.8,
        junk_weight: 1.0,
        hard_junk_weight: 2.0,
        quality_weight: 1.0,
        latency_weight: 0.0009,
        cost_weight: 0.08,
        ..MabConfig::default()
    };

    let drift_epoch = 64;
    for round in 0u64..120 {
        let under = coverage_pick_under_sampled(
            round ^ 0x5151_5151,
            &slice_ids,
            1,
            slice_coverage,
            |sid| slice_calls.get(sid).copied().unwrap_or(0),
        );
        let sid = under
            .first()
            .cloned()
            .unwrap_or_else(|| slice_ids[(round as usize) % slice_ids.len()].clone());
        let slice = *by_id.get(&sid).expect("slice id exists");

        let candidates = candidate_backends(slice);
        let seed = stable_hash64(round ^ 0xABAB_CDCD, &sid);
        let fill = policy_fill_k_observed_with_coverage(
            seed,
            &candidates,
            2,
            true,
            backend_coverage,
            guard,
            |b| observed_calls_and_elapsed(b, slice, &cell_windows),
            |eligible, need| {
                select_k_without_replacement_by(
                    seed ^ 0xDEAD_BEEF,
                    eligible,
                    need,
                    |_s, rem, _k| {
                        let summaries = summaries_for_slice(rem, slice, &cell_windows);
                        let d = select_mab_explain(rem, &summaries, mab_cfg);
                        vec![d.selection.chosen]
                    },
                )
            },
        );

        for backend in fill.chosen {
            let o = synthetic_outcome(round, &backend, slice);
            cell_windows
                .entry(cell_key(&backend, slice))
                .or_insert_with(|| Window::new(60))
                .push(o);
            *slice_calls.entry(sid.clone()).or_insert(0) += 1;

            if slice.is_hard() {
                if round < drift_epoch {
                    *pre_hard_counts.entry(backend).or_insert(0) += 1;
                } else {
                    *post_hard_counts.entry(backend).or_insert(0) += 1;
                }
            }
        }
    }

    println!("== synthetic_drift_harness ==");
    println!("slice coverage:");
    for s in &slices {
        let n = slice_calls.get(&s.id()).copied().unwrap_or(0);
        println!(
            "  family={:<8} hardness={:<4} region={:<2} calls={:>3}",
            s.family, s.hardness, s.region, n
        );
        assert!(n > 0, "each synthetic slice should be sampled");
    }

    println!("hard-slice backend picks pre/post drift:");
    let mut backends: Vec<String> = vec![
        "tiny_rule".to_string(),
        "balanced_nn".to_string(),
        "large_transformer".to_string(),
        "hard_specialist".to_string(),
    ];
    backends.sort();
    for b in backends {
        let pre = pre_hard_counts.get(&b).copied().unwrap_or(0);
        let post = post_hard_counts.get(&b).copied().unwrap_or(0);
        println!("  {:<18} pre={:>3} post={:>3}", b, pre, post);
    }
}
