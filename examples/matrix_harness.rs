//! Matrix-style harness example (task x dataset x backend), offline and deterministic.
//!
//! This models the pattern used by evaluation harnesses (including `anno`):
//! - pick a matrix slice (task + dataset facet),
//! - pick a small backend subset with muxer policy + guardrails,
//! - observe outcomes and update per-slice windows,
//! - keep slice coverage from collapsing.
//!
//! Run:
//! `cargo run --example matrix_harness`

use muxer::{
    coverage_pick_under_sampled, policy_fill_k_observed_with_coverage,
    select_k_without_replacement_by, select_mab_explain, stable_hash64, CoverageConfig,
    LatencyGuardrail, MabConfig, Outcome, Summary, Window,
};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Slice {
    task: &'static str,
    dataset: &'static str,
    lang: &'static str,
    domain: &'static str,
}

impl Slice {
    fn id(self) -> String {
        format!("{}::{}", self.task, self.dataset)
    }

    fn tag(self) -> String {
        format!("{}.lang={}.dom={}", self.task, self.lang, self.domain)
    }
}

fn cell_key(backend: &str, slice: Slice) -> String {
    format!("{backend}@@{}", slice.tag())
}

fn compatible_backends(slice: Slice) -> Vec<String> {
    let mut out: Vec<String> = match slice.task {
        "ner" => vec!["heuristic", "stacked", "bert_onnx", "gliner_onnx"],
        "re" => vec!["stacked", "tplinker", "gliner_onnx"],
        "coref" => vec!["stacked", "bert_onnx"],
        _ => Vec::new(),
    }
    .into_iter()
    .map(ToString::to_string)
    .collect();

    // Domain-specific compatibility tweaks.
    if slice.domain == "biomedical" {
        out.retain(|b| b != "heuristic");
    }
    out
}

fn summaries_for_slice(
    arms: &[String],
    slice: Slice,
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
    slice: Slice,
    windows: &BTreeMap<String, Window>,
) -> (u64, u64) {
    let k = cell_key(backend, slice);
    let s = windows.get(&k).map(Window::summary).unwrap_or_default();
    (s.calls, s.elapsed_ms_sum)
}

fn simulated_outcome(round: u64, backend: &str, slice: Slice) -> Outcome {
    let drift_cell =
        backend == "bert_onnx" && slice.task == "ner" && slice.domain == "social_media";
    let drifting = drift_cell && round >= 48;

    let hard_pm = if drifting { 140 } else { 25 }; // per-mille
    let soft_pm = if drifting { 260 } else { 90 }; // per-mille

    let h = stable_hash64(round ^ 0xA11C_E5E1, &format!("{backend}|{}|h", slice.id())) % 1000;
    let s = stable_hash64(round ^ 0xBEEF_CAFE, &format!("{backend}|{}|s", slice.id())) % 1000;
    let hard = h < hard_pm;
    let soft = !hard && s < soft_pm;

    let ok = !hard;
    let junk = hard || soft;

    let base_quality: f64 = match backend {
        "gliner_onnx" => 0.88,
        "bert_onnx" => 0.84,
        "stacked" => 0.80,
        "tplinker" => 0.74,
        "heuristic" => 0.62,
        _ => 0.50,
    };
    let domain_penalty: f64 = match slice.domain {
        "social_media" => 0.08,
        "biomedical" => 0.05,
        _ => 0.0,
    };
    let quality = if hard {
        0.0
    } else if soft {
        0.25
    } else {
        (base_quality - domain_penalty).max(0.0_f64)
    };

    let base_ms = match backend {
        "heuristic" => 120,
        "stacked" => 420,
        "tplinker" => 600,
        "bert_onnx" => 880,
        "gliner_onnx" => 950,
        _ => 500,
    };
    let task_ms = match slice.task {
        "ner" => 60,
        "re" => 180,
        "coref" => 240,
        _ => 0,
    };
    let jitter = stable_hash64(round ^ 0x0D15_EA5E, &format!("{backend}|{}|j", slice.id())) % 120;

    let base_cost = match backend {
        "heuristic" => 2,
        "stacked" => 5,
        "tplinker" => 6,
        "bert_onnx" => 8,
        "gliner_onnx" => 9,
        _ => 4,
    };
    let task_cost = if slice.task == "re" { 2 } else { 0 };

    Outcome {
        ok,
        junk,
        hard_junk: hard,
        cost_units: base_cost + task_cost,
        elapsed_ms: base_ms + task_ms + jitter,
        quality_score: Some(quality),
    }
}

fn main() {
    let slices = vec![
        Slice {
            task: "ner",
            dataset: "WikiGold",
            lang: "en",
            domain: "news",
        },
        Slice {
            task: "ner",
            dataset: "Wnut17",
            lang: "en",
            domain: "social_media",
        },
        Slice {
            task: "ner",
            dataset: "GENIA",
            lang: "en",
            domain: "biomedical",
        },
        Slice {
            task: "re",
            dataset: "DocRED",
            lang: "en",
            domain: "wikipedia",
        },
        Slice {
            task: "coref",
            dataset: "GAP",
            lang: "en",
            domain: "news",
        },
    ];

    let slice_ids: Vec<String> = slices.iter().map(|s| s.id()).collect();
    let by_id: BTreeMap<String, Slice> = slices.iter().map(|s| (s.id(), *s)).collect();

    let mut windows: BTreeMap<String, Window> = BTreeMap::new();
    let mut slice_calls: BTreeMap<String, u64> = BTreeMap::new();

    let mab_cfg = MabConfig {
        exploration_c: 0.8,
        junk_weight: 0.9,
        hard_junk_weight: 1.8,
        latency_weight: 0.0008,
        cost_weight: 0.05,
        quality_weight: 0.8,
        ..MabConfig::default()
    };
    let guard = LatencyGuardrail {
        max_mean_ms: Some(2_200.0),
        require_measured: false,
        allow_fewer: true,
    };
    let backend_coverage = CoverageConfig {
        enabled: true,
        min_fraction: 0.10,
        min_calls_floor: 1,
    };
    let slice_coverage = CoverageConfig {
        enabled: true,
        min_fraction: 0.08,
        min_calls_floor: 2,
    };

    for round in 0u64..90 {
        // Choose an under-sampled matrix slice first (task x dataset facet).
        let under = coverage_pick_under_sampled(
            round ^ 0x0005_11CE,
            &slice_ids,
            1,
            slice_coverage,
            |sid| slice_calls.get(sid).copied().unwrap_or(0),
        );
        let slice_id = under
            .first()
            .cloned()
            .unwrap_or_else(|| slice_ids[(round as usize) % slice_ids.len()].clone());
        let slice = *by_id.get(&slice_id).expect("slice id exists");

        let candidates = compatible_backends(slice);
        if candidates.is_empty() {
            continue;
        }

        let seed = stable_hash64(round, &slice.tag());
        let fill = policy_fill_k_observed_with_coverage(
            seed,
            &candidates,
            2,
            true,
            backend_coverage,
            guard,
            |b| observed_calls_and_elapsed(b, slice, &windows),
            |eligible, need| {
                select_k_without_replacement_by(
                    seed ^ 0xABCD_1234,
                    eligible,
                    need,
                    |_s, rem, _k| {
                        let summaries = summaries_for_slice(rem, slice, &windows);
                        let d = select_mab_explain(rem, &summaries, mab_cfg);
                        vec![d.selection.chosen]
                    },
                )
            },
        );

        for backend in fill.chosen {
            assert!(
                candidates.contains(&backend),
                "chosen backend must be compatible"
            );
            let o = simulated_outcome(round, &backend, slice);
            windows
                .entry(cell_key(&backend, slice))
                .or_insert_with(|| Window::new(48))
                .push(o);
            *slice_calls.entry(slice.id()).or_insert(0) += 1;
        }
    }

    println!("== matrix_harness summary ==");
    for s in &slices {
        let n = slice_calls.get(&s.id()).copied().unwrap_or(0);
        println!("{:<28} calls={:>3}", s.tag(), n);
        assert!(n > 0, "each slice should be sampled at least once");
    }
}
