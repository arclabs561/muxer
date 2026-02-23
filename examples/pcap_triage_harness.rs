//! PCAP triage harness example (network security), offline and deterministic.
//!
//! This demonstrates a realistic routing loop for packet-analysis backends:
//! - matrix slices are `(dataset, protocol, environment)`,
//! - backend eligibility depends on protocol/data constraints,
//! - muxer picks a small backend set per slice with coverage + guardrails,
//! - outcomes are fed back into per-slice windows.
//!
//! Run:
//! `cargo run --example pcap_triage_harness`

use muxer::{
    coverage_pick_under_sampled, policy_fill_k_observed_with_coverage,
    select_k_without_replacement_by, select_mab_explain, stable_hash64, CoverageConfig,
    LatencyGuardrail, MabConfig, Outcome, Summary, Window,
};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct PcapSlice {
    dataset: &'static str,
    protocol: &'static str,
    env: &'static str,
}

impl PcapSlice {
    fn id(self) -> String {
        format!("{}::{}::{}", self.dataset, self.protocol, self.env)
    }
}

fn cell_key(engine: &str, slice: PcapSlice) -> String {
    format!(
        "{engine}@@dataset={}.proto={}.env={}",
        slice.dataset, slice.protocol, slice.env
    )
}

fn compatible_engines(slice: PcapSlice) -> Vec<String> {
    let mut v: Vec<String> = vec![
        "suricata_sig".to_string(),
        "zeek_scripts".to_string(),
        "flow_mlp".to_string(),
        "transformer_payload".to_string(),
    ];

    // Payload-based transformer needs decrypted L7 payloads.
    if slice.protocol == "tls" {
        v.retain(|b| b != "transformer_payload");
    }
    // Assume this dataset variant lacks stable flow features.
    if slice.dataset == "CTU-13" {
        v.retain(|b| b != "flow_mlp");
    }
    v
}

fn summaries_for_slice(
    arms: &[String],
    slice: PcapSlice,
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
    engine: &str,
    slice: PcapSlice,
    windows: &BTreeMap<String, Window>,
) -> (u64, u64) {
    let k = cell_key(engine, slice);
    let s = windows.get(&k).map(Window::summary).unwrap_or_default();
    (s.calls, s.elapsed_ms_sum)
}

fn simulated_outcome(round: u64, engine: &str, slice: PcapSlice) -> Outcome {
    // One realistic drift: signature engine underperforms on encrypted traffic after a change.
    let tls_evasion_wave = round >= 52 && slice.protocol == "tls" && engine == "suricata_sig";

    let base_quality: f64 = match engine {
        "transformer_payload" => 0.91,
        "flow_mlp" => 0.84,
        "zeek_scripts" => 0.80,
        "suricata_sig" => 0.76,
        _ => 0.50,
    };
    let protocol_penalty: f64 = match slice.protocol {
        "tls" => 0.07,
        "dns" => 0.02,
        "http" => 0.0,
        _ => 0.03,
    };
    let wave_penalty = if tls_evasion_wave { 0.28 } else { 0.0 };

    let hard_pm_base = match engine {
        "suricata_sig" => 30,
        "zeek_scripts" => 24,
        "flow_mlp" => 22,
        "transformer_payload" => 18,
        _ => 30,
    };
    let soft_pm_base = match engine {
        "suricata_sig" => 120,
        "zeek_scripts" => 90,
        "flow_mlp" => 85,
        "transformer_payload" => 60,
        _ => 120,
    };
    let hard_pm = hard_pm_base + if tls_evasion_wave { 170 } else { 0 };
    let soft_pm = soft_pm_base + if tls_evasion_wave { 220 } else { 0 };

    let h = stable_hash64(round ^ 0x701A_CCA7, &format!("{engine}|{}|h", slice.id())) % 1000;
    let s = stable_hash64(round ^ 0x88EA_1122, &format!("{engine}|{}|s", slice.id())) % 1000;
    let hard = h < hard_pm;
    let soft = !hard && s < soft_pm;
    let junk = hard || soft;

    let quality = if hard {
        0.0
    } else if soft {
        0.24
    } else {
        (base_quality - protocol_penalty - wave_penalty).max(0.0_f64)
    };

    let base_latency = match engine {
        "suricata_sig" => 55,
        "zeek_scripts" => 140,
        "flow_mlp" => 220,
        "transformer_payload" => 480,
        _ => 200,
    };
    let protocol_latency = match slice.protocol {
        "http" => 90,
        "dns" => 40,
        "tls" => 130,
        _ => 50,
    };
    let jitter = stable_hash64(round ^ 0xCA11_BA9E, &format!("{engine}|{}|j", slice.id())) % 60;

    let base_cost = match engine {
        "suricata_sig" => 1,
        "zeek_scripts" => 2,
        "flow_mlp" => 4,
        "transformer_payload" => 9,
        _ => 3,
    };

    Outcome {
        ok: !hard,
        junk,
        hard_junk: hard,
        cost_units: base_cost,
        elapsed_ms: base_latency + protocol_latency + jitter,
        quality_score: Some(quality),
    }
}

fn main() {
    let slices = vec![
        PcapSlice {
            dataset: "CICIDS2017",
            protocol: "http",
            env: "enterprise",
        },
        PcapSlice {
            dataset: "CICIDS2017",
            protocol: "dns",
            env: "enterprise",
        },
        PcapSlice {
            dataset: "CICIDS2017",
            protocol: "tls",
            env: "enterprise",
        },
        PcapSlice {
            dataset: "UNSW-NB15",
            protocol: "http",
            env: "cloud",
        },
        PcapSlice {
            dataset: "UNSW-NB15",
            protocol: "tls",
            env: "cloud",
        },
        PcapSlice {
            dataset: "CTU-13",
            protocol: "dns",
            env: "campus",
        },
    ];

    let slice_ids: Vec<String> = slices.iter().map(|s| s.id()).collect();
    let slice_by_id: BTreeMap<String, PcapSlice> = slices.iter().map(|s| (s.id(), *s)).collect();

    let mut cell_windows: BTreeMap<String, Window> = BTreeMap::new();
    let mut engine_windows: BTreeMap<String, Window> = BTreeMap::new();
    let mut slice_calls: BTreeMap<String, u64> = BTreeMap::new();

    let guard = LatencyGuardrail {
        max_mean_ms: Some(1_300.0),
        require_measured: false,
        allow_fewer: true,
    };
    let slice_coverage = CoverageConfig {
        enabled: true,
        min_fraction: 0.10,
        min_calls_floor: 2,
    };
    let engine_coverage = CoverageConfig {
        enabled: true,
        min_fraction: 0.12,
        min_calls_floor: 1,
    };
    let mab_cfg = MabConfig {
        exploration_c: 0.75,
        junk_weight: 1.0,
        hard_junk_weight: 2.2,
        quality_weight: 0.9,
        latency_weight: 0.0007,
        cost_weight: 0.06,
        ..MabConfig::default()
    };

    for round in 0u64..96 {
        let slice_pick = coverage_pick_under_sampled(
            round ^ 0x5A1C_ECC0,
            &slice_ids,
            1,
            slice_coverage,
            |sid| slice_calls.get(sid).copied().unwrap_or(0),
        );
        let sid = slice_pick
            .first()
            .cloned()
            .unwrap_or_else(|| slice_ids[(round as usize) % slice_ids.len()].clone());
        let slice = *slice_by_id.get(&sid).expect("slice id exists");

        let candidates = compatible_engines(slice);
        if candidates.is_empty() {
            continue;
        }

        let seed = stable_hash64(round ^ 0x0BAD_F00D, &sid);
        let fill = policy_fill_k_observed_with_coverage(
            seed,
            &candidates,
            2,
            true,
            engine_coverage,
            guard,
            |e| observed_calls_and_elapsed(e, slice, &cell_windows),
            |eligible, need| {
                select_k_without_replacement_by(
                    seed ^ 0xA1B2_C3D4,
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

        for engine in fill.chosen {
            let o = simulated_outcome(round, &engine, slice);
            cell_windows
                .entry(cell_key(&engine, slice))
                .or_insert_with(|| Window::new(56))
                .push(o);
            engine_windows
                .entry(engine)
                .or_insert_with(|| Window::new(220))
                .push(o);
            *slice_calls.entry(sid.clone()).or_insert(0) += 1;
        }
    }

    println!("== pcap_triage_harness ==");
    println!("slice coverage:");
    for s in &slices {
        let n = slice_calls.get(&s.id()).copied().unwrap_or(0);
        println!(
            "  dataset={:<10} proto={:<4} env={:<10} calls={:>3}",
            s.dataset, s.protocol, s.env, n
        );
        assert!(n > 0, "each pcap slice should receive traffic");
    }

    println!("engine aggregate (windowed):");
    for (engine, w) in &engine_windows {
        let s = w.summary();
        println!(
            "  {:<20} calls={:>3} ok={:.3} junk={:.3} hard={:.3} mean_ms={:.1}",
            engine,
            s.calls,
            s.ok_rate(),
            s.junk_rate(),
            s.hard_junk_rate(),
            s.mean_elapsed_ms()
        );
    }
}
