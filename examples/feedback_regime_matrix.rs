//! Exercise muxer primitives on four native real-data feedback regimes.
//!
//! Prepare the ignored data directory first:
//!
//! ```text
//! uv run scripts/fetch_feedback_datasets.py
//! uv run scripts/build_feedback_traces.py
//! cargo run --example feedback_regime_matrix
//! ```
//!
//! This is an offline diagnostic, not a deployment benchmark. Each lane keeps
//! its source semantics instead of translating every signal into `Outcome`.

use muxer::monitor::drift_simplex;
use muxer::{
    ips_value, select_candidate_assessments, self_normalized_ips_value, CandidateAssessment,
    DriftMetric, LoggedReward, MetricObjective,
};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Lines};
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn open_lines(path: &Path) -> io::Result<Lines<BufReader<File>>> {
    Ok(BufReader::new(File::open(path)?).lines())
}

fn expect_header(lines: &mut Lines<BufReader<File>>, path: &Path, expected: &str) -> Result<()> {
    let header = lines
        .next()
        .ok_or_else(|| invalid_data(format!("{} is empty", path.display())))??;
    if header != expected {
        return Err(invalid_data(format!(
            "{} has unexpected header: {header}",
            path.display()
        ))
        .into());
    }
    Ok(())
}

fn fields<'a>(line: &'a str, expected: usize, path: &Path, line_no: usize) -> Result<Vec<&'a str>> {
    let fields: Vec<&str> = line.split(',').collect();
    if fields.len() != expected {
        return Err(invalid_data(format!(
            "{} line {line_no} has {} fields, expected {expected}",
            path.display(),
            fields.len()
        ))
        .into());
    }
    Ok(fields)
}

fn parse_value<T>(raw: &str, path: &Path, line_no: usize, name: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: Error + 'static,
{
    raw.parse::<T>().map_err(|error| {
        invalid_data(format!(
            "{} line {line_no} has invalid {name}: {error}",
            path.display()
        ))
        .into()
    })
}

fn evaluate_open_bandit(path: &Path) -> Result<()> {
    let mut lines = open_lines(path)?;
    expect_header(
        &mut lines,
        path,
        "behavior,row_id,timestamp,item_id,position,reward,logging_propensity,target_propensity",
    )?;

    let mut by_behavior = BTreeMap::<String, Vec<LoggedReward>>::new();
    for (line_index, line) in lines.enumerate() {
        let line_no = line_index + 2;
        let line = line?;
        let row = fields(&line, 8, path, line_no)?;
        let reward = parse_value::<f64>(row[5], path, line_no, "reward")?;
        let logging_propensity = parse_value::<f64>(row[6], path, line_no, "logging propensity")?;
        let target_propensity = parse_value::<f64>(row[7], path, line_no, "target propensity")?;
        by_behavior
            .entry(row[0].to_string())
            .or_default()
            .push(LoggedReward {
                reward,
                logging_propensity,
                target_propensity,
            });
    }

    println!("Open Bandit Dataset: uniform target over item_id 0..39");
    println!("logger   rows   click    ips      snips    ess       max_weight");
    for behavior in ["random", "bts"] {
        let rows = by_behavior
            .get(behavior)
            .ok_or_else(|| invalid_data(format!("missing Open Bandit logger {behavior}")))?;
        if rows.is_empty() {
            return Err(invalid_data(format!("Open Bandit logger {behavior} is empty")).into());
        }
        let naive = rows.iter().map(|row| row.reward).sum::<f64>() / rows.len() as f64;
        let ips = ips_value(rows.iter().copied())?;
        let snips = self_normalized_ips_value(rows.iter().copied())?;
        let weights: Vec<f64> = rows
            .iter()
            .map(|row| row.target_propensity / row.logging_propensity)
            .collect();
        let weight_sum = weights.iter().sum::<f64>();
        let weight_sq_sum = weights.iter().map(|weight| weight * weight).sum::<f64>();
        let ess = weight_sum * weight_sum / weight_sq_sum;
        let max_weight = weights.iter().copied().fold(0.0_f64, f64::max);
        if !ips.is_finite() || !(0.0..=1.0).contains(&snips) || !ess.is_finite() || ess <= 0.0 {
            return Err(invalid_data(format!(
                "Open Bandit logger {behavior} produced invalid diagnostics"
            ))
            .into());
        }
        println!(
            "{behavior:7} {:>5}  {naive:>7.4}  {ips:>7.4}  {snips:>7.4}  {ess:>8.1}  {max_weight:>10.2}",
            rows.len()
        );
    }
    println!();
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Objective {
    Maximize,
    Minimize,
}

impl Objective {
    fn parse(raw: &str) -> Option<Self> {
        match raw {
            "maximize" => Some(Self::Maximize),
            "minimize" => Some(Self::Minimize),
            _ => None,
        }
    }

    fn metric(self) -> MetricObjective {
        match self {
            Self::Maximize => MetricObjective::maximize(0, 1.0),
            Self::Minimize => MetricObjective::minimize(0, 1.0),
        }
    }

    fn better(self, candidate: f64, current: f64) -> bool {
        match self {
            Self::Maximize => candidate > current,
            Self::Minimize => candidate < current,
        }
    }

    fn gap(self, value: f64, oracle: f64) -> f64 {
        match self {
            Self::Maximize => oracle - value,
            Self::Minimize => value - oracle,
        }
    }
}

#[derive(Debug, Clone)]
struct AlgorithmRun {
    value: f64,
    ok: bool,
}

#[derive(Debug, Clone, Copy, Default)]
struct RunningMean {
    sum: f64,
    count: u64,
}

impl RunningMean {
    fn push(&mut self, value: f64) {
        self.sum += value;
        self.count += 1;
    }

    fn mean(self) -> f64 {
        self.sum / self.count.max(1) as f64
    }
}

type AslibData = BTreeMap<String, BTreeMap<String, BTreeMap<String, AlgorithmRun>>>;

fn evaluate_aslib(path: &Path) -> Result<()> {
    let mut lines = open_lines(path)?;
    expect_header(
        &mut lines,
        path,
        "scenario,instance_id,algorithm,repetition,value,objective,runstatus",
    )?;

    let mut scenarios = AslibData::new();
    let mut objectives = BTreeMap::<String, Objective>::new();
    for (line_index, line) in lines.enumerate() {
        let line_no = line_index + 2;
        let line = line?;
        let row = fields(&line, 7, path, line_no)?;
        let value = parse_value::<f64>(row[4], path, line_no, "value")?;
        if !value.is_finite() {
            return Err(invalid_data(format!(
                "{} line {line_no} has non-finite value",
                path.display()
            ))
            .into());
        }
        let objective = Objective::parse(row[5]).ok_or_else(|| {
            invalid_data(format!(
                "{} line {line_no} has invalid objective {}",
                path.display(),
                row[5]
            ))
        })?;
        if objectives
            .insert(row[0].to_string(), objective)
            .is_some_and(|existing| existing != objective)
        {
            return Err(invalid_data(format!(
                "{} mixes objectives for scenario {}",
                path.display(),
                row[0]
            ))
            .into());
        }
        let prior = scenarios
            .entry(row[0].to_string())
            .or_default()
            .entry(row[1].to_string())
            .or_default()
            .insert(
                row[2].to_string(),
                AlgorithmRun {
                    value,
                    ok: row[6] == "ok",
                },
            );
        if prior.is_some() {
            return Err(invalid_data(format!(
                "{} line {line_no} duplicates an algorithm run",
                path.display()
            ))
            .into());
        }
    }

    println!("ASlib: history-only full-information selection");
    println!("scenario                    instances  selected     fixed        oracle       gap  selected_ok");
    for (scenario, instances) in scenarios {
        let objective = objectives[&scenario];
        let algorithms: Vec<String> = instances
            .values()
            .flat_map(|runs| runs.keys().cloned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        if algorithms.is_empty() || instances.is_empty() {
            return Err(invalid_data(format!("ASlib scenario {scenario} is empty")).into());
        }

        let mut history = BTreeMap::<String, RunningMean>::new();
        let mut selected_sum = 0.0;
        let mut selected_ok = 0_u64;
        let mut oracle_sum = 0.0;
        for (instance, runs) in &instances {
            if runs.len() != algorithms.len() {
                return Err(invalid_data(format!(
                    "ASlib scenario {scenario} instance {instance} is incomplete"
                ))
                .into());
            }
            let assessments: Vec<CandidateAssessment> = algorithms
                .iter()
                .map(|algorithm| {
                    let aggregate = history.get(algorithm).copied().unwrap_or_default();
                    CandidateAssessment::new(algorithm, aggregate.count, vec![aggregate.mean()])
                })
                .collect();
            let selection = select_candidate_assessments(&assessments, &[objective.metric()])?;
            let selected = selection
                .chosen
                .as_deref()
                .ok_or_else(|| invalid_data("ASlib selector returned no choice"))?;
            let selected_run = &runs[selected];
            selected_sum += selected_run.value;
            selected_ok += u64::from(selected_run.ok);

            let oracle = runs
                .values()
                .fold(None, |best: Option<f64>, run| match best {
                    None => Some(run.value),
                    Some(current) if objective.better(run.value, current) => Some(run.value),
                    Some(current) => Some(current),
                })
                .ok_or_else(|| invalid_data("ASlib instance has no oracle run"))?;
            oracle_sum += oracle;

            for (algorithm, run) in runs {
                history
                    .entry(algorithm.clone())
                    .or_default()
                    .push(run.value);
            }
        }

        let instance_count = instances.len() as f64;
        let selected_mean = selected_sum / instance_count;
        let oracle_mean = oracle_sum / instance_count;
        let fixed_mean = history
            .values()
            .map(|aggregate| aggregate.mean())
            .fold(None, |best: Option<f64>, value| match best {
                None => Some(value),
                Some(current) if objective.better(value, current) => Some(value),
                Some(current) => Some(current),
            })
            .ok_or_else(|| invalid_data("ASlib scenario has no fixed reference"))?;
        println!(
            "{scenario:27} {:>9}  {selected_mean:>10.4}  {fixed_mean:>10.4}  {oracle_mean:>10.4}  {:>8.4}  {selected_ok:>5}/{}",
            instances.len(),
            objective.gap(selected_mean, oracle_mean),
            instances.len()
        );
    }
    println!();
    Ok(())
}

fn quantile_sorted(values: &[u64], numerator: usize, denominator: usize) -> u64 {
    let last = values.len().saturating_sub(1);
    values[last * numerator / denominator]
}

#[derive(Debug, Clone)]
struct FuzzerStats {
    name: String,
    median: u64,
    iqr: u64,
    trials: usize,
}

fn evaluate_fuzzbench(path: &Path) -> Result<()> {
    let mut lines = open_lines(path)?;
    expect_header(
        &mut lines,
        path,
        "experiment,benchmark,fuzzer,trial_id,time,edges_covered",
    )?;

    let mut final_by_trial = BTreeMap::<(String, String, String), (u64, u64)>::new();
    for (line_index, line) in lines.enumerate() {
        let line_no = line_index + 2;
        let line = line?;
        let row = fields(&line, 6, path, line_no)?;
        let time = parse_value::<u64>(row[4], path, line_no, "time")?;
        let edges = parse_value::<u64>(row[5], path, line_no, "edges_covered")?;
        let key = (row[1].to_string(), row[2].to_string(), row[3].to_string());
        let final_value = final_by_trial.entry(key).or_insert((time, edges));
        if time > final_value.0 {
            *final_value = (time, edges);
        }
    }

    let mut benchmarks = BTreeMap::<String, BTreeMap<String, Vec<u64>>>::new();
    for ((benchmark, fuzzer, _trial), (_time, edges)) in final_by_trial {
        benchmarks
            .entry(benchmark)
            .or_default()
            .entry(fuzzer)
            .or_default()
            .push(edges);
    }

    println!("FuzzBench: final edge coverage versus trial spread");
    println!(
        "benchmark                     chosen             median      best       iqr  frontier"
    );
    let mut tradeoffs = 0_usize;
    for (benchmark, by_fuzzer) in &mut benchmarks {
        let mut stats = Vec::<FuzzerStats>::new();
        for (fuzzer, values) in by_fuzzer {
            values.sort_unstable();
            if values.is_empty() {
                continue;
            }
            let median = quantile_sorted(values, 1, 2);
            let q1 = quantile_sorted(values, 1, 4);
            let q3 = quantile_sorted(values, 3, 4);
            stats.push(FuzzerStats {
                name: fuzzer.clone(),
                median,
                iqr: q3.saturating_sub(q1),
                trials: values.len(),
            });
        }
        let best_median = stats
            .iter()
            .map(|stat| stat.median)
            .max()
            .ok_or_else(|| invalid_data(format!("FuzzBench benchmark {benchmark} is empty")))?;
        if best_median == 0 {
            return Err(
                invalid_data(format!("FuzzBench benchmark {benchmark} has zero coverage")).into(),
            );
        }
        let scale = best_median as f64;
        let assessments: Vec<CandidateAssessment> = stats
            .iter()
            .map(|stat| {
                CandidateAssessment::new(
                    &stat.name,
                    stat.trials as u64,
                    vec![stat.median as f64 / scale, stat.iqr as f64 / scale],
                )
            })
            .collect();
        let selection = select_candidate_assessments(
            &assessments,
            &[
                MetricObjective::maximize(0, 1.0),
                MetricObjective::minimize(1, 0.10),
            ],
        )?;
        let chosen = selection
            .chosen
            .as_deref()
            .ok_or_else(|| invalid_data("FuzzBench selector returned no choice"))?;
        let chosen_stats = stats
            .iter()
            .find(|stat| stat.name == chosen)
            .ok_or_else(|| invalid_data("FuzzBench choice has no statistics"))?;
        tradeoffs += usize::from(chosen_stats.median < best_median);
        println!(
            "{benchmark:29} {chosen:18} {:>7}  {best_median:>8}  {:>8}  {:>8}",
            chosen_stats.median,
            chosen_stats.iqr,
            selection.frontier.len()
        );
    }
    println!(
        "coverage/stability scalarization accepted lower median coverage on {tradeoffs}/{} benchmarks\n",
        benchmarks.len()
    );
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct NabPoint {
    value: f64,
    annotated: bool,
}

fn quantile_f64(values: &[f64], numerator: usize, denominator: usize) -> f64 {
    let last = values.len().saturating_sub(1);
    values[last * numerator / denominator]
}

fn bin(value: f64, cuts: [f64; 3]) -> usize {
    if value <= cuts[0] {
        0
    } else if value <= cuts[1] {
        1
    } else if value <= cuts[2] {
        2
    } else {
        3
    }
}

fn distribution(values: impl IntoIterator<Item = f64>, cuts: [f64; 3]) -> Vec<f64> {
    let mut counts = [1.0_f64; 4];
    for value in values {
        counts[bin(value, cuts)] += 1.0;
    }
    let total = counts.iter().sum::<f64>();
    counts.into_iter().map(|count| count / total).collect()
}

fn evaluate_nab(path: &Path) -> Result<()> {
    let mut lines = open_lines(path)?;
    expect_header(
        &mut lines,
        path,
        "series,source_key,row_id,timestamp,value,annotated_window",
    )?;

    let mut series = BTreeMap::<String, Vec<NabPoint>>::new();
    for (line_index, line) in lines.enumerate() {
        let line_no = line_index + 2;
        let line = line?;
        let row = fields(&line, 6, path, line_no)?;
        let value = parse_value::<f64>(row[4], path, line_no, "value")?;
        let annotated = match row[5] {
            "0" => false,
            "1" => true,
            other => {
                return Err(invalid_data(format!(
                    "{} line {line_no} has invalid annotation {other}",
                    path.display()
                ))
                .into())
            }
        };
        if !value.is_finite() {
            return Err(invalid_data(format!(
                "{} line {line_no} has non-finite value",
                path.display()
            ))
            .into());
        }
        series
            .entry(row[0].to_string())
            .or_default()
            .push(NabPoint { value, annotated });
    }

    println!("NAB: marginal four-bin Hellinger distance");
    println!(
        "series             rows   annotated  heldout_normal  annotated_window  window_larger"
    );
    let mut window_larger = 0_usize;
    for (name, points) in &series {
        let split = points.len() / 5;
        let mut baseline: Vec<f64> = points[..split]
            .iter()
            .filter(|point| !point.annotated)
            .map(|point| point.value)
            .collect();
        let heldout: Vec<f64> = points[split..]
            .iter()
            .filter(|point| !point.annotated)
            .map(|point| point.value)
            .collect();
        let annotated: Vec<f64> = points
            .iter()
            .filter(|point| point.annotated)
            .map(|point| point.value)
            .collect();
        if baseline.is_empty() || heldout.is_empty() || annotated.is_empty() {
            return Err(invalid_data(format!("NAB series {name} has an empty partition")).into());
        }
        baseline.sort_by(f64::total_cmp);
        let cuts = [
            quantile_f64(&baseline, 1, 4),
            quantile_f64(&baseline, 1, 2),
            quantile_f64(&baseline, 3, 4),
        ];
        let baseline_dist = distribution(baseline.iter().copied(), cuts);
        let heldout_dist = distribution(heldout.iter().copied(), cuts);
        let annotated_dist = distribution(annotated.iter().copied(), cuts);
        let heldout_drift =
            drift_simplex(&baseline_dist, &heldout_dist, DriftMetric::Hellinger, 1e-12)?;
        let annotated_drift = drift_simplex(
            &baseline_dist,
            &annotated_dist,
            DriftMetric::Hellinger,
            1e-12,
        )?;
        let larger = annotated_drift > heldout_drift;
        window_larger += usize::from(larger);
        println!(
            "{name:18} {:>6}  {:>9}  {heldout_drift:>14.4}  {annotated_drift:>16.4}  {larger:>13}",
            points.len(),
            annotated.len()
        );
    }
    println!(
        "annotated-window drift exceeded held-out normal drift on {window_larger}/{} streams",
        series.len()
    );
    Ok(())
}

fn main() -> Result<()> {
    let root = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/feedback/derived"));
    if !root.is_dir() {
        eprintln!(
            "no derived feedback data at {}; run scripts/fetch_feedback_datasets.py and scripts/build_feedback_traces.py",
            root.display()
        );
        return Ok(());
    }

    println!("== native feedback-regime validation ==\n");
    evaluate_open_bandit(&root.join("open_bandit.csv"))?;
    evaluate_aslib(&root.join("aslib.csv"))?;
    evaluate_fuzzbench(&root.join("fuzzbench.csv"))?;
    evaluate_nab(&root.join("nab.csv"))?;
    Ok(())
}
