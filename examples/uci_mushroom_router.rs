//! Route a real UCI Mushroom classification trace between three fixed policies.
//!
//! The example keeps the dataset outside the repository. Pass the downloaded
//! `agaricus-lepiota.data` path as the first argument:
//!
//! ```text
//! cargo run --example uci_mushroom_router -- /path/to/agaricus-lepiota.data
//! ```
//!
//! Rows whose index is divisible by five train the policy surrogates. The
//! remaining rows are replayed as an evaluation stream. The router only sees
//! the selected policy's outcome; a per-row policy oracle is computed
//! separately as an offline reference.

use muxer::{ObservationId, Outcome, Router, RouterConfig};
use std::collections::BTreeMap;
use std::fs;

#[derive(Debug, Clone)]
struct Row {
    poisonous: bool,
    features: Vec<u8>,
}

#[derive(Debug, Clone)]
struct NaiveBayes {
    class_counts: [u64; 2],
    counts: Vec<[[u64; 256]; 2]>,
    cardinalities: Vec<u64>,
}

impl NaiveBayes {
    fn fit(rows: &[Row]) -> Option<Self> {
        let feature_count = rows.first()?.features.len();
        let mut class_counts = [0_u64; 2];
        let mut counts = vec![[[0_u64; 256]; 2]; feature_count];
        let mut seen = vec![[false; 256]; feature_count];

        for row in rows {
            let class = usize::from(row.poisonous);
            class_counts[class] += 1;
            for (feature, &value) in row.features.iter().enumerate() {
                let value = usize::from(value);
                counts[feature][class][value] += 1;
                seen[feature][value] = true;
            }
        }

        let cardinalities = seen
            .into_iter()
            .map(|values| values.into_iter().filter(|present| *present).count() as u64)
            .collect();

        Some(Self {
            class_counts,
            counts,
            cardinalities,
        })
    }

    fn predict_poisonous(&self, row: &Row) -> bool {
        let total = self.class_counts.iter().sum::<u64>() as f64;
        let mut scores = [0.0_f64; 2];
        for (class, score) in scores.iter_mut().enumerate() {
            *score = ((self.class_counts[class] + 1) as f64 / (total + 2.0)).ln();
            for (feature, &value) in row.features.iter().enumerate() {
                let classes = self.class_counts[class] as f64;
                let categories = self.cardinalities[feature].max(1) as f64;
                let count = self.counts[feature][class][usize::from(value)] as f64;
                *score += ((count + 1.0) / (classes + categories)).ln();
            }
        }
        scores[1] > scores[0]
    }
}

#[derive(Debug, Clone)]
struct OdorModel {
    counts: [[u64; 2]; 256],
}

impl OdorModel {
    fn fit(rows: &[Row]) -> Option<Self> {
        let mut counts = [[0_u64; 2]; 256];
        for row in rows {
            let odor = usize::from(*row.features.get(4)?);
            counts[odor][usize::from(row.poisonous)] += 1;
        }
        Some(Self { counts })
    }

    fn predict_poisonous(&self, row: &Row) -> bool {
        let odor = usize::from(row.features[4]);
        self.counts[odor][1] >= self.counts[odor][0]
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct Stats {
    calls: u64,
    correct: u64,
    hard_failures: u64,
    cost: u64,
    elapsed_ms: u64,
}

impl Stats {
    fn record(&mut self, row: &Row, predicted_poisonous: bool, cost: u64, elapsed_ms: u64) {
        let correct = predicted_poisonous == row.poisonous;
        self.calls += 1;
        self.correct += u64::from(correct);
        self.hard_failures += u64::from(!correct && !predicted_poisonous && row.poisonous);
        self.cost += cost;
        self.elapsed_ms += elapsed_ms;
    }

    fn accuracy(self) -> f64 {
        self.correct as f64 / self.calls.max(1) as f64
    }
}

fn parse_row(line: &str) -> Option<Row> {
    let fields: Vec<&str> = line.trim().split(',').collect();
    if fields.len() != 23 {
        return None;
    }
    let label = fields[0].as_bytes().first().copied()?;
    if label != b'e' && label != b'p' {
        return None;
    }
    let features = fields[1..]
        .iter()
        .map(|field| field.as_bytes().first().copied())
        .collect::<Option<Vec<_>>>()?;
    Some(Row {
        poisonous: label == b'p',
        features,
    })
}

fn predict(
    arm: &str,
    row: &Row,
    majority_poisonous: bool,
    odor: &OdorModel,
    bayes: &NaiveBayes,
) -> bool {
    match arm {
        "majority" => majority_poisonous,
        "odor" => odor.predict_poisonous(row),
        "naive_bayes" => bayes.predict_poisonous(row),
        _ => majority_poisonous,
    }
}

fn policy_costs(arm: &str) -> (u64, u64) {
    match arm {
        "majority" => (1, 2),
        "odor" => (3, 8),
        "naive_bayes" => (12, 80),
        _ => (1, 2),
    }
}

fn main() {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("no dataset supplied; download UCI Mushroom and pass agaricus-lepiota.data");
        return;
    };

    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) => {
            eprintln!("failed to read {path}: {error}");
            std::process::exit(1);
        }
    };
    let rows: Vec<Row> = text.lines().filter_map(parse_row).collect();
    if rows.len() < 20 {
        eprintln!(
            "expected at least 20 valid UCI Mushroom rows, found {}",
            rows.len()
        );
        std::process::exit(1);
    }

    let train: Vec<Row> = rows
        .iter()
        .enumerate()
        .filter(|(index, _)| index % 5 == 0)
        .map(|(_, row)| row.clone())
        .collect();
    let eval: Vec<&Row> = rows
        .iter()
        .enumerate()
        .filter(|(index, _)| index % 5 != 0)
        .map(|(_, row)| row)
        .collect();
    let poisonous = train.iter().filter(|row| row.poisonous).count() as u64;
    let majority_poisonous = poisonous * 2 >= train.len() as u64;
    let odor = OdorModel::fit(&train).expect("training rows have an odor feature");
    let bayes = NaiveBayes::fit(&train).expect("training rows have features");

    let arms = vec![
        "majority".to_string(),
        "odor".to_string(),
        "naive_bayes".to_string(),
    ];
    let mut cfg = RouterConfig::default().window_cap(128);
    cfg.mab.base = cfg
        .mab
        .base
        .clone()
        .with_quality_weight(1.0)
        .with_hard_junk_weight(2.0)
        .with_junk_weight(0.4)
        .with_cost_weight(0.01)
        .with_latency_weight(0.001);
    let mut router = Router::new(arms.clone(), cfg).expect("valid router config");
    let mut selected = Stats::default();
    let mut oracle = Stats::default();
    let mut selected_per_arm = BTreeMap::<String, Stats>::new();
    let mut offline_per_arm = BTreeMap::<String, Stats>::new();

    for (index, row) in eval.iter().enumerate() {
        for candidate in &arms {
            let predicted = predict(candidate, row, majority_poisonous, &odor, &bayes);
            let (cost, elapsed_ms) = policy_costs(candidate);
            offline_per_arm
                .entry(candidate.clone())
                .or_default()
                .record(row, predicted, cost, elapsed_ms);
        }

        let decision = router.select(1, index as u64);
        let Some(arm) = decision.primary() else {
            continue;
        };
        let predicted = predict(arm, row, majority_poisonous, &odor, &bayes);
        let (cost, elapsed_ms) = policy_costs(arm);
        selected.record(row, predicted, cost, elapsed_ms);
        selected_per_arm
            .entry(arm.to_string())
            .or_default()
            .record(row, predicted, cost, elapsed_ms);

        let mut best_arm = arms[0].as_str();
        let mut best_correct = false;
        for candidate in &arms {
            let candidate_correct =
                predict(candidate, row, majority_poisonous, &odor, &bayes) == row.poisonous;
            if candidate_correct && !best_correct {
                best_arm = candidate;
                best_correct = true;
            }
        }
        let (oracle_cost, oracle_elapsed) = policy_costs(best_arm);
        oracle.calls += 1;
        oracle.correct += u64::from(best_correct);
        oracle.cost += oracle_cost;
        oracle.elapsed_ms += oracle_elapsed;

        let outcome = Outcome::with_quality(
            predicted == row.poisonous,
            predicted != row.poisonous,
            predicted != row.poisonous && !predicted && row.poisonous,
            cost,
            elapsed_ms,
            f64::from(predicted == row.poisonous),
        );
        assert!(router.observe_with_id(ObservationId::new(index as u64), arm, outcome));
    }

    println!(
        "dataset=UCI Mushroom rows={} train={} eval={}",
        rows.len(),
        train.len(),
        eval.len()
    );
    println!(
        "selected accuracy={:.4} hard_failures={} per-row oracle accuracy={:.4} gap={:.4}",
        selected.accuracy(),
        selected.hard_failures,
        oracle.accuracy(),
        oracle.accuracy() - selected.accuracy()
    );
    println!("arm            calls  selected_acc  trace_acc  hard_failures  mean_cost  mean_ms");
    for arm in &arms {
        let stats = selected_per_arm.get(arm).copied().unwrap_or_default();
        let trace = offline_per_arm.get(arm).copied().unwrap_or_default();
        println!(
            "{arm:13} {:>5}  {:>12.4}  {:>9.4}  {:>13}  {:>9.2}  {:>7.2}",
            stats.calls,
            stats.accuracy(),
            trace.accuracy(),
            stats.hard_failures,
            stats.cost as f64 / stats.calls.max(1) as f64,
            stats.elapsed_ms as f64 / stats.calls.max(1) as f64,
        );
    }
}
