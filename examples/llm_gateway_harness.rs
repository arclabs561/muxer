//! LLM gateway harness, offline and deterministic.
//!
//! This models the routing loop used by a model gateway:
//! - task classes have different quality thresholds,
//! - providers have different cost/latency/quality profiles,
//! - one provider regresses on extraction traffic after a prompt change,
//! - muxer uses coverage plus multi-objective selection to adapt.
//!
//! Run:
//! `cargo run --release --example llm_gateway_harness`

use muxer::{
    coverage_pick_under_sampled, select_mab_explain, stable_hash64, CoverageConfig, MabConfig,
    Outcome, Summary, Window,
};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy)]
struct Task {
    name: &'static str,
    threshold: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct Totals {
    calls: u64,
    ok: u64,
    junk: u64,
    hard_junk: u64,
    cost: u64,
    elapsed: u64,
    quality_sum: f64,
}

impl Totals {
    fn observe(&mut self, outcome: Outcome) {
        self.calls += 1;
        self.ok += outcome.ok as u64;
        self.junk += outcome.junk as u64;
        self.hard_junk += outcome.hard_junk as u64;
        self.cost += outcome.cost_units;
        self.elapsed += outcome.elapsed_ms;
        self.quality_sum += outcome.quality_score.unwrap_or(0.0);
    }

    fn ok_rate(self) -> f64 {
        ratio(self.ok, self.calls)
    }

    fn junk_rate(self) -> f64 {
        ratio(self.junk, self.calls)
    }

    fn mean_quality(self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            self.quality_sum / self.calls as f64
        }
    }

    fn mean_cost(self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            self.cost as f64 / self.calls as f64
        }
    }

    fn mean_ms(self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            self.elapsed as f64 / self.calls as f64
        }
    }
}

fn ratio(n: u64, d: u64) -> f64 {
    if d == 0 {
        0.0
    } else {
        n as f64 / d as f64
    }
}

fn task_id(task: Task) -> String {
    task.name.to_string()
}

fn cell_key(model: &str, task: Task) -> String {
    format!("{model}@@{}", task.name)
}

fn task_for_round(round: u64) -> Task {
    const TASKS: [Task; 4] = [
        Task {
            name: "classification",
            threshold: 0.72,
        },
        Task {
            name: "extraction",
            threshold: 0.78,
        },
        Task {
            name: "reasoning",
            threshold: 0.84,
        },
        Task {
            name: "safety_review",
            threshold: 0.88,
        },
    ];
    TASKS[(stable_hash64(0x9a7e_1101, &round.to_string()) as usize) % TASKS.len()]
}

fn eligible_models(task: Task) -> Vec<String> {
    let mut models = vec![
        "local-small".to_string(),
        "balanced".to_string(),
        "frontier-large".to_string(),
        "verifier".to_string(),
    ];
    if task.name == "safety_review" {
        models.retain(|m| m != "local-small");
    }
    models
}

fn summaries_for_task(
    models: &[String],
    task: Task,
    windows: &BTreeMap<String, Window>,
) -> BTreeMap<String, Summary> {
    models
        .iter()
        .map(|m| {
            let key = cell_key(m, task);
            (
                m.clone(),
                windows.get(&key).map(Window::summary).unwrap_or_default(),
            )
        })
        .collect()
}

fn model_calls_for_task(model: &str, task: Task, windows: &BTreeMap<String, Window>) -> u64 {
    windows
        .get(&cell_key(model, task))
        .map(|w| w.summary().calls)
        .unwrap_or(0)
}

fn select_model(round: u64, task: Task, windows: &BTreeMap<String, Window>) -> String {
    let models = eligible_models(task);
    let coverage = CoverageConfig {
        enabled: true,
        min_fraction: 0.08,
        min_calls_floor: 2,
    };

    let prepick = coverage_pick_under_sampled(
        stable_hash64(round ^ 0x9a7e_1201, task.name),
        &models,
        1,
        coverage,
        |model| model_calls_for_task(model, task, windows),
    );
    if let Some(model) = prepick.first() {
        return model.clone();
    }

    let cfg = MabConfig::default()
        .with_quality_weight(1.25)
        .with_junk_weight(1.0)
        .with_hard_junk_weight(2.5)
        .with_latency_weight(0.0015)
        .with_cost_weight(0.055);
    let summaries = summaries_for_task(&models, task, windows);
    select_mab_explain(&models, &summaries, cfg)
        .selection
        .chosen
}

fn base_quality(model: &str, task: Task) -> f64 {
    match (model, task.name) {
        ("local-small", "classification") => 0.81,
        ("local-small", "extraction") => 0.74,
        ("local-small", "reasoning") => 0.56,
        ("balanced", "classification") => 0.86,
        ("balanced", "extraction") => 0.84,
        ("balanced", "reasoning") => 0.73,
        ("balanced", "safety_review") => 0.71,
        ("frontier-large", "classification") => 0.88,
        ("frontier-large", "extraction") => 0.89,
        ("frontier-large", "reasoning") => 0.91,
        ("frontier-large", "safety_review") => 0.86,
        ("verifier", "classification") => 0.80,
        ("verifier", "extraction") => 0.82,
        ("verifier", "reasoning") => 0.87,
        ("verifier", "safety_review") => 0.94,
        _ => 0.50,
    }
}

fn simulated_outcome(round: u64, model: &str, task: Task) -> Outcome {
    let prompt_shift = round >= 240 && model == "balanced" && task.name == "extraction";
    let model_task = format!("{model}|{}|{round}", task.name);
    let jitter = (stable_hash64(0x9a7e_1301, &model_task) % 1000) as f64 / 1000.0;
    let quality_noise = (jitter - 0.5) * 0.08;
    let shift_penalty = if prompt_shift { 0.22 } else { 0.0 };
    let quality = (base_quality(model, task) + quality_noise - shift_penalty).clamp(0.0, 1.0);

    let hard_junk = quality < task.threshold - 0.35
        || stable_hash64(0x9a7e_1302, &model_task) % 1000
            < match model {
                "local-small" => 12,
                "balanced" => {
                    if prompt_shift {
                        36
                    } else {
                        10
                    }
                }
                "frontier-large" => 7,
                "verifier" => 9,
                _ => 15,
            };
    let junk = hard_junk || quality < task.threshold;

    let cost_units = match model {
        "local-small" => 1,
        "balanced" => 4,
        "frontier-large" => 18,
        "verifier" => 12,
        _ => 6,
    };
    let base_ms = match model {
        "local-small" => 85,
        "balanced" => 170,
        "frontier-large" => 680,
        "verifier" => 410,
        _ => 200,
    };
    let task_ms = match task.name {
        "classification" => 0,
        "extraction" => 90,
        "reasoning" => 260,
        "safety_review" => 180,
        _ => 0,
    };
    let latency_jitter = stable_hash64(0x9a7e_1303, &model_task) % 80;

    Outcome::with_quality(
        !hard_junk,
        junk,
        hard_junk,
        cost_units,
        base_ms + task_ms + latency_jitter,
        quality,
    )
}

fn print_table(title: &str, rows: &BTreeMap<String, Totals>) {
    println!("\n{title}");
    println!("model           calls  ok     junk   quality  cost/call  ms/call");
    for (model, totals) in rows {
        println!(
            "{:<15} {:>5}  {:>5.3}  {:>5.3}  {:>7.3}  {:>9.2}  {:>7.1}",
            model,
            totals.calls,
            totals.ok_rate(),
            totals.junk_rate(),
            totals.mean_quality(),
            totals.mean_cost(),
            totals.mean_ms()
        );
    }
}

fn main() {
    let mut windows: BTreeMap<String, Window> = BTreeMap::new();
    let mut pre: BTreeMap<String, Totals> = BTreeMap::new();
    let mut post: BTreeMap<String, Totals> = BTreeMap::new();
    let mut extraction_pre: BTreeMap<String, u64> = BTreeMap::new();
    let mut extraction_post: BTreeMap<String, u64> = BTreeMap::new();

    for round in 0..480_u64 {
        let task = task_for_round(round);
        let model = select_model(round, task, &windows);
        let outcome = simulated_outcome(round, &model, task);

        windows
            .entry(cell_key(&model, task))
            .or_insert_with(|| Window::new(48))
            .push(outcome);

        let target = if round < 240 { &mut pre } else { &mut post };
        target.entry(model.clone()).or_default().observe(outcome);

        if task.name == "extraction" {
            let target = if round < 240 {
                &mut extraction_pre
            } else {
                &mut extraction_post
            };
            *target.entry(model).or_insert(0) += 1;
        }

        let task_seen = windows
            .iter()
            .filter(|(key, w)| key.ends_with(&task_id(task)) && !w.is_empty())
            .count();
        assert!(task_seen > 0, "each selected task should retain feedback");
    }

    println!("== llm_gateway_harness ==");
    println!("480 deterministic requests, drift at round 240: balanced degrades on extraction.");
    print_table("pre-drift aggregate", &pre);
    print_table("post-drift aggregate", &post);

    println!("\nextraction traffic share");
    println!("model           pre_calls  post_calls");
    for model in ["local-small", "balanced", "frontier-large", "verifier"] {
        println!(
            "{:<15} {:>9}  {:>10}",
            model,
            extraction_pre.get(model).copied().unwrap_or(0),
            extraction_post.get(model).copied().unwrap_or(0)
        );
    }

    let balanced_pre = extraction_pre.get("balanced").copied().unwrap_or(0);
    let balanced_post = extraction_post.get("balanced").copied().unwrap_or(0);
    let frontier_post = extraction_post.get("frontier-large").copied().unwrap_or(0);

    assert!(
        balanced_post < balanced_pre,
        "balanced should receive less extraction traffic after its drift"
    );
    assert!(
        frontier_post > 0,
        "coverage and quality weighting should keep an expensive fallback in play"
    );
}
