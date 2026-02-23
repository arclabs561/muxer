//! Search-ranking harness example, offline and deterministic.
//!
//! This models a realistic retrieval/ranking routing loop:
//! - slices are `(intent, locale, device)` traffic cells,
//! - ranker eligibility varies by slice capabilities,
//! - muxer picks a small ranker set with coverage + guardrails,
//! - outcomes are fed back into per-slice windows.
//!
//! Run:
//! `cargo run --example search_ranking_harness`

use muxer::{
    coverage_pick_under_sampled, policy_fill_k_observed_with_coverage,
    select_k_without_replacement_by, select_mab_explain, stable_hash64, CoverageConfig,
    LatencyGuardrail, MabConfig, Outcome, Summary, Window,
};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct SearchSlice {
    intent: &'static str,
    locale: &'static str,
    device: &'static str,
}

impl SearchSlice {
    fn id(self) -> String {
        format!(
            "intent={}.locale={}.device={}",
            self.intent, self.locale, self.device
        )
    }
}

fn cell_key(ranker: &str, slice: SearchSlice) -> String {
    format!("{ranker}@@{}", slice.id())
}

fn compatible_rankers(slice: SearchSlice) -> Vec<String> {
    let mut out = vec![
        "bm25_fast".to_string(),
        "dense_dual".to_string(),
        "hybrid_crossenc".to_string(),
        "rules_safe".to_string(),
    ];

    // Assume dense embeddings are not yet tuned for this locale.
    if slice.locale == "de_de" {
        out.retain(|r| r != "dense_dual");
    }
    // Cross-encoder latency budget is too tight for mobile transactional queries.
    if slice.device == "mobile" && slice.intent == "transactional" {
        out.retain(|r| r != "hybrid_crossenc");
    }
    out
}

fn summaries_for_slice(
    arms: &[String],
    slice: SearchSlice,
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
    ranker: &str,
    slice: SearchSlice,
    windows: &BTreeMap<String, Window>,
) -> (u64, u64) {
    let k = cell_key(ranker, slice);
    let s = windows.get(&k).map(Window::summary).unwrap_or_default();
    (s.calls, s.elapsed_ms_sum)
}

fn simulated_outcome(round: u64, ranker: &str, slice: SearchSlice) -> Outcome {
    // Drift: lexical ranker degrades for informational Spanish after index-content shift.
    let corpus_shift = round >= 68
        && ranker == "bm25_fast"
        && slice.intent == "informational"
        && slice.locale == "es_es";

    let base_quality: f64 = match ranker {
        "hybrid_crossenc" => 0.91,
        "dense_dual" => 0.85,
        "bm25_fast" => 0.79,
        "rules_safe" => 0.66,
        _ => 0.50,
    };
    let intent_penalty: f64 = if slice.intent == "transactional" {
        0.03
    } else {
        0.0
    };
    let shift_penalty: f64 = if corpus_shift { 0.32 } else { 0.0 };
    let quality = (base_quality - intent_penalty - shift_penalty).max(0.0_f64);

    let hard_pm_base = match ranker {
        "hybrid_crossenc" => 8,
        "dense_dual" => 13,
        "bm25_fast" => 18,
        "rules_safe" => 22,
        _ => 15,
    };
    let soft_pm_base = match ranker {
        "hybrid_crossenc" => 55,
        "dense_dual" => 72,
        "bm25_fast" => 95,
        "rules_safe" => 130,
        _ => 90,
    };
    let hard_pm = hard_pm_base + if corpus_shift { 140 } else { 0 };
    let soft_pm = soft_pm_base + if corpus_shift { 250 } else { 0 };

    let h = stable_hash64(round ^ 0x5100_1001, &format!("{ranker}|{}|h", slice.id())) % 1000;
    let s = stable_hash64(round ^ 0x5100_1002, &format!("{ranker}|{}|s", slice.id())) % 1000;
    let hard = h < hard_pm;
    let soft = !hard && s < soft_pm;
    let junk = hard || soft;

    let base_ms = match ranker {
        "rules_safe" => 10,
        "bm25_fast" => 22,
        "dense_dual" => 38,
        "hybrid_crossenc" => 96,
        _ => 20,
    };
    let device_ms = if slice.device == "mobile" { 12 } else { 7 };
    let jitter = stable_hash64(round ^ 0x5100_1003, &format!("{ranker}|{}|j", slice.id())) % 20;

    let base_cost = match ranker {
        "rules_safe" => 1,
        "bm25_fast" => 2,
        "dense_dual" => 4,
        "hybrid_crossenc" => 8,
        _ => 2,
    };

    Outcome {
        ok: !hard,
        junk,
        hard_junk: hard,
        cost_units: base_cost,
        elapsed_ms: base_ms + device_ms + jitter,
        quality_score: Some(if hard {
            0.0
        } else if soft {
            0.20
        } else {
            quality
        }),
    }
}

fn main() {
    let slices = vec![
        SearchSlice {
            intent: "navigational",
            locale: "en_us",
            device: "mobile",
        },
        SearchSlice {
            intent: "informational",
            locale: "en_us",
            device: "desktop",
        },
        SearchSlice {
            intent: "informational",
            locale: "es_es",
            device: "mobile",
        },
        SearchSlice {
            intent: "transactional",
            locale: "en_us",
            device: "mobile",
        },
        SearchSlice {
            intent: "transactional",
            locale: "de_de",
            device: "desktop",
        },
    ];

    let ids: Vec<String> = slices.iter().map(|s| s.id()).collect();
    let by_id: BTreeMap<String, SearchSlice> = slices.iter().map(|s| (s.id(), *s)).collect();

    let mut cell_windows: BTreeMap<String, Window> = BTreeMap::new();
    let mut ranker_windows: BTreeMap<String, Window> = BTreeMap::new();
    let mut slice_calls: BTreeMap<String, u64> = BTreeMap::new();

    let slice_coverage = CoverageConfig {
        enabled: true,
        min_fraction: 0.11,
        min_calls_floor: 2,
    };
    let ranker_coverage = CoverageConfig {
        enabled: true,
        min_fraction: 0.12,
        min_calls_floor: 1,
    };
    let guard = LatencyGuardrail {
        max_mean_ms: Some(300.0),
        require_measured: false,
        allow_fewer: true,
    };
    let cfg = MabConfig {
        exploration_c: 0.78,
        junk_weight: 1.0,
        hard_junk_weight: 2.0,
        quality_weight: 1.0,
        latency_weight: 0.0020,
        cost_weight: 0.12,
        ..MabConfig::default()
    };

    for round in 0u64..105 {
        let under =
            coverage_pick_under_sampled(round ^ 0x5100_2001, &ids, 1, slice_coverage, |sid| {
                slice_calls.get(sid).copied().unwrap_or(0)
            });
        let sid = under
            .first()
            .cloned()
            .unwrap_or_else(|| ids[(round as usize) % ids.len()].clone());
        let slice = *by_id.get(&sid).expect("slice id exists");
        let candidates = compatible_rankers(slice);
        if candidates.is_empty() {
            continue;
        }

        let seed = stable_hash64(round ^ 0x5100_2002, &sid);
        let fill = policy_fill_k_observed_with_coverage(
            seed,
            &candidates,
            2,
            true,
            ranker_coverage,
            guard,
            |b| observed_calls_and_elapsed(b, slice, &cell_windows),
            |eligible, need| {
                select_k_without_replacement_by(
                    seed ^ 0x5100_2003,
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

        for ranker in fill.chosen {
            let o = simulated_outcome(round, &ranker, slice);
            cell_windows
                .entry(cell_key(&ranker, slice))
                .or_insert_with(|| Window::new(64))
                .push(o);
            ranker_windows
                .entry(ranker)
                .or_insert_with(|| Window::new(240))
                .push(o);
            *slice_calls.entry(sid.clone()).or_insert(0) += 1;
        }
    }

    println!("== search_ranking_harness ==");
    println!("slice coverage:");
    for s in &slices {
        let n = slice_calls.get(&s.id()).copied().unwrap_or(0);
        println!(
            "  intent={:<13} locale={:<5} device={:<7} calls={:>3}",
            s.intent, s.locale, s.device, n
        );
        assert!(n > 0, "each search slice should receive traffic");
    }
    println!("ranker aggregate (windowed):");
    for (b, w) in &ranker_windows {
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
