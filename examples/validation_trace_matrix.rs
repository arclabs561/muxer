//! Replay multi-domain real-data traces through the quality Router and the
//! domain-neutral metric-vector selector.
//!
//! Build the trace first with:
//!
//! ```text
//! uv run scripts/build_validation_traces.py
//! cargo run --example validation_trace_matrix
//! ```
//!
//! The trace contains three fixed classifier policies for each dataset. This
//! example exercises candidate-set selection, monitoring windows, objective
//! weights, hard-junk filtering, caller-owned observation IDs, and delayed
//! quality labels. It reports a per-row oracle only as an offline reference.

use muxer::{
    select_candidate_assessments, CandidateAssessment, MetricObjective, ObservationId, Outcome,
    Router, RouterConfig,
};
use std::collections::BTreeMap;
use std::fs;

#[derive(Debug, Clone)]
struct TraceRow {
    arm: String,
    ok: bool,
    junk: bool,
    hard_junk: bool,
    quality_score: f64,
    cost_units: u64,
    elapsed_ms: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct Stats {
    calls: u64,
    ok: u64,
    hard_junk: u64,
    quality_sum: f64,
    cost_sum: u64,
    elapsed_ms_sum: u64,
}

impl Stats {
    fn record(&mut self, row: &TraceRow) {
        self.calls += 1;
        self.ok += u64::from(row.ok);
        self.hard_junk += u64::from(row.hard_junk);
        self.quality_sum += row.quality_score;
        self.cost_sum = self.cost_sum.saturating_add(row.cost_units);
        self.elapsed_ms_sum = self.elapsed_ms_sum.saturating_add(row.elapsed_ms);
    }

    fn rate(self) -> f64 {
        self.ok as f64 / self.calls.max(1) as f64
    }

    fn assessment(&self, arm: &str) -> CandidateAssessment {
        let calls = self.calls.max(1) as f64;
        CandidateAssessment::new(
            arm,
            self.calls,
            vec![
                self.quality_sum / calls,
                self.cost_sum as f64 / calls,
                self.elapsed_ms_sum as f64 / calls,
            ],
        )
    }
}

fn parse_bool(value: &str) -> Option<bool> {
    match value {
        "0" => Some(false),
        "1" => Some(true),
        _ => None,
    }
}

fn parse_row(line: &str) -> Option<(String, u64, TraceRow)> {
    let fields: Vec<&str> = line.split(',').collect();
    if fields.len() != 11 {
        return None;
    }
    let dataset = fields[0].to_string();
    let row_id = fields[1].parse().ok()?;
    let arm = fields[2].to_string();
    Some((
        dataset,
        row_id,
        TraceRow {
            arm,
            ok: parse_bool(fields[5])?,
            junk: parse_bool(fields[6])?,
            hard_junk: parse_bool(fields[7])?,
            quality_score: fields[8].parse().ok()?,
            cost_units: fields[9].parse().ok()?,
            elapsed_ms: fields[10].parse().ok()?,
        },
    ))
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/traces/classification-traces.csv".to_string());
    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("no trace supplied at {path}; run scripts/build_validation_traces.py");
            return;
        }
        Err(error) => {
            eprintln!("failed to read {path}: {error}");
            std::process::exit(1);
        }
    };

    let mut datasets: BTreeMap<String, BTreeMap<u64, BTreeMap<String, TraceRow>>> = BTreeMap::new();
    let mut malformed = 0_u64;
    for line in text.lines().skip(1) {
        match parse_row(line) {
            Some((dataset, row_id, row)) => {
                datasets
                    .entry(dataset)
                    .or_default()
                    .entry(row_id)
                    .or_default()
                    .insert(row.arm.clone(), row);
            }
            None => malformed += 1,
        }
    }
    if datasets.is_empty() {
        eprintln!("trace contains no valid rows");
        std::process::exit(1);
    }
    if malformed > 0 {
        eprintln!("skipped malformed trace rows: {malformed}");
    }

    println!("dataset       rows  arms  quality_acc  generic_acc  oracle_acc  q_gap    g_gap    hard_junk  delayed");
    let mut next_id = 0_u64;
    for (dataset, rows) in datasets {
        let arms: Vec<String> = rows
            .values()
            .flat_map(|by_arm| by_arm.keys().cloned())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        let mut cfg = RouterConfig::default()
            .window_cap(128)
            .with_monitoring(500, 50);
        cfg.mab.base.max_junk_rate = Some(0.70);
        cfg.mab.base.max_hard_junk_rate = Some(0.25);
        cfg.mab.base = cfg
            .mab
            .base
            .clone()
            .with_quality_weight(1.0)
            .with_hard_junk_weight(2.0)
            .with_junk_weight(0.4)
            .with_cost_weight(0.01)
            .with_latency_weight(0.001);
        let mut router = Router::new(arms.clone(), cfg).expect("valid trace router config");
        let mut selected = Stats::default();
        let mut generic_selected = Stats::default();
        let mut oracle = Stats::default();
        let mut per_arm = BTreeMap::<String, Stats>::new();
        let mut generic_history = BTreeMap::<String, Stats>::new();
        let mut pending = Vec::<(ObservationId, f64)>::new();
        let mut delayed = 0_u64;
        let mut expired = 0_u64;
        let mut missing = 0_u64;

        for (row_id, by_arm) in rows.iter() {
            let best = by_arm.values().max_by_key(|row| row.ok);
            if let Some(best) = best {
                oracle.record(best);
            }

            let decision = router
                .select_from(&arms, 1, *row_id)
                .expect("trace arms are registered and unique");
            let Some(chosen) = decision.primary() else {
                missing += 1;
                continue;
            };
            let Some(trace) = by_arm.get(chosen) else {
                missing += 1;
                continue;
            };
            selected.record(trace);
            per_arm.entry(chosen.to_string()).or_default().record(trace);

            let assessments: Vec<CandidateAssessment> = arms
                .iter()
                .map(|arm| {
                    generic_history
                        .get(arm)
                        .copied()
                        .unwrap_or_default()
                        .assessment(arm)
                })
                .collect();
            let generic = select_candidate_assessments(
                &assessments,
                &[
                    MetricObjective::maximize(0, 1.0),
                    MetricObjective::minimize(1, 0.01),
                    MetricObjective::minimize(2, 0.001),
                ],
            )
            .expect("trace metrics have a stable finite shape");
            if let Some(generic_arm) = generic.chosen.as_deref() {
                if let Some(generic_trace) = by_arm.get(generic_arm) {
                    generic_selected.record(generic_trace);
                }
            }
            // This is a full-information offline replay: all policy traces are
            // available for updating the generic selector, unlike the Router
            // path above, which only observes its chosen arm.
            for arm in &arms {
                if let Some(trace) = by_arm.get(arm) {
                    generic_history
                        .entry(arm.clone())
                        .or_default()
                        .record(trace);
                }
            }

            let id = ObservationId::new(next_id);
            next_id = next_id.saturating_add(1);
            let outcome = Outcome::new(
                trace.ok,
                trace.junk,
                trace.hard_junk,
                trace.cost_units,
                trace.elapsed_ms,
            );
            assert!(router.observe_with_id(id, chosen, outcome));
            pending.push((id, trace.quality_score));

            if pending.len() >= 2 && row_id % 3 == 2 {
                let newest = pending.pop().expect("pending label");
                let oldest = pending.remove(0);
                assert!(router.set_quality_score_for_id(newest.0, newest.1));
                if router.set_quality_score_for_id(oldest.0, oldest.1) {
                    delayed += 2;
                } else {
                    delayed += 1;
                    expired += 1;
                }
            }
        }
        for (id, quality) in pending.into_iter().rev() {
            delayed += 1;
            if !router.set_quality_score_for_id(id, quality) {
                expired += 1;
            }
        }

        println!(
            "{dataset:12} {:>5}  {:>4}  {:>11.4}  {:>11.4}  {:>10.4}  {:>7.4}  {:>7.4}  {:>9}  {:>7}  expired={expired}",
            selected.calls,
            arms.len(),
            selected.rate(),
            generic_selected.rate(),
            oracle.rate(),
            oracle.rate() - selected.rate(),
            oracle.rate() - generic_selected.rate(),
            selected.hard_junk,
            delayed,
        );
        if missing > 0 {
            eprintln!("{dataset}: rows without a selected trace arm: {missing}");
        }
        for arm in &arms {
            let stats = per_arm.get(arm).copied().unwrap_or_default();
            println!(
                "  {arm:12} calls={:>5} selected_acc={:.4} hard_junk={}",
                stats.calls,
                stats.rate(),
                stats.hard_junk,
            );
        }
    }
}
