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
    println!("logger   rows   support  click    ips      snips    ess       max_weight");
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
        let support = weights.iter().filter(|weight| **weight > 0.0).count();
        let ess = weight_sum * weight_sum / weight_sq_sum;
        let max_weight = weights.iter().copied().fold(0.0_f64, f64::max);
        if !ips.is_finite() || !(0.0..=1.0).contains(&snips) || !ess.is_finite() || ess <= 0.0 {
            return Err(invalid_data(format!(
                "Open Bandit logger {behavior} produced invalid diagnostics"
            ))
            .into());
        }
        println!(
            "{behavior:7} {:>5}  {support:>7}  {naive:>7.4}  {ips:>7.4}  {snips:>7.4}  {ess:>8.1}  {max_weight:>10.2}",
            rows.len(),
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

#[derive(Debug, Clone, Copy)]
struct AslibReplay {
    mean: f64,
    ok: u64,
}

type AslibData = BTreeMap<String, BTreeMap<String, BTreeMap<String, AlgorithmRun>>>;

fn replay_aslib_order(
    scenario: &str,
    ordered: &[(&String, &BTreeMap<String, AlgorithmRun>)],
    algorithms: &[String],
    objective: Objective,
) -> Result<AslibReplay> {
    let mut history = BTreeMap::<String, RunningMean>::new();
    let mut selected_sum = 0.0;
    let mut selected_ok = 0_u64;
    for (instance, runs) in ordered {
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

        for (algorithm, run) in *runs {
            history
                .entry(algorithm.clone())
                .or_default()
                .push(run.value);
        }
    }
    Ok(AslibReplay {
        mean: selected_sum / ordered.len().max(1) as f64,
        ok: selected_ok,
    })
}

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
    println!(
        "scenario                    instances  ascending   descending  fixed        oracle       gap_range      ok(asc/desc)"
    );
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

        let ascending: Vec<_> = instances.iter().collect();
        let mut descending = ascending.clone();
        descending.reverse();
        let asc = replay_aslib_order(&scenario, &ascending, &algorithms, objective)?;
        let desc = replay_aslib_order(&scenario, &descending, &algorithms, objective)?;

        let mut fixed_history = BTreeMap::<String, RunningMean>::new();
        let mut oracle_sum = 0.0;
        for runs in instances.values() {
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
                fixed_history
                    .entry(algorithm.clone())
                    .or_default()
                    .push(run.value);
            }
        }

        let instance_count = instances.len() as f64;
        let oracle_mean = oracle_sum / instance_count;
        let fixed_mean = fixed_history
            .values()
            .map(|aggregate| aggregate.mean())
            .fold(None, |best: Option<f64>, value| match best {
                None => Some(value),
                Some(current) if objective.better(value, current) => Some(value),
                Some(current) => Some(current),
            })
            .ok_or_else(|| invalid_data("ASlib scenario has no fixed reference"))?;
        let asc_gap = objective.gap(asc.mean, oracle_mean);
        let desc_gap = objective.gap(desc.mean, oracle_mean);
        println!(
            "{scenario:27} {:>9}  {:>10.4}  {:>10.4}  {fixed_mean:>10.4}  {oracle_mean:>10.4}  {:>6.3}..{:>6.3}  {:>3}/{:>3}",
            instances.len(),
            asc.mean,
            desc.mean,
            asc_gap.min(desc_gap),
            asc_gap.max(desc_gap),
            asc.ok,
            desc.ok,
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

fn select_fuzzer(
    stats: &[FuzzerStats],
    best_median: u64,
    spread_weight: f64,
) -> Result<(String, usize)> {
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
            MetricObjective::minimize(1, spread_weight),
        ],
    )?;
    let chosen = selection
        .chosen
        .ok_or_else(|| invalid_data("FuzzBench selector returned no choice"))?;
    Ok((chosen, selection.frontier.len()))
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
    let spread_weights = [0.0, 0.05, 0.10, 0.25, 0.50];
    let mut lower_median = [0_usize; 5];
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
        let mut primary = None;
        for (index, spread_weight) in spread_weights.iter().copied().enumerate() {
            let (chosen, frontier) = select_fuzzer(&stats, best_median, spread_weight)?;
            let chosen_stats = stats
                .iter()
                .find(|stat| stat.name == chosen)
                .ok_or_else(|| invalid_data("FuzzBench choice has no statistics"))?;
            lower_median[index] += usize::from(chosen_stats.median < best_median);
            if index == 2 {
                primary = Some((chosen, frontier));
            }
        }
        let (chosen, frontier) =
            primary.ok_or_else(|| invalid_data("FuzzBench primary weight was not evaluated"))?;
        let chosen_stats = stats
            .iter()
            .find(|stat| stat.name == chosen)
            .ok_or_else(|| invalid_data("FuzzBench choice has no statistics"))?;
        println!(
            "{benchmark:29} {chosen:18} {:>7}  {best_median:>8}  {:>8}  {:>8}",
            chosen_stats.median, chosen_stats.iqr, frontier,
        );
    }
    println!("spread weight sensitivity (benchmarks choosing below-best median):");
    for (weight, lower) in spread_weights.into_iter().zip(lower_median) {
        println!("  weight={weight:>4.2}: {lower}/{}", benchmarks.len());
    }
    println!();
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

fn nab_distances(
    name: &str,
    points: &[NabPoint],
    baseline_numerator: usize,
    baseline_denominator: usize,
) -> Result<(f64, f64, usize)> {
    let split = points.len() * baseline_numerator / baseline_denominator;
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
    let heldout_dist = distribution(heldout, cuts);
    let annotated_count = annotated.len();
    let annotated_dist = distribution(annotated, cuts);
    let heldout_drift =
        drift_simplex(&baseline_dist, &heldout_dist, DriftMetric::Hellinger, 1e-12)?;
    let annotated_drift = drift_simplex(
        &baseline_dist,
        &annotated_dist,
        DriftMetric::Hellinger,
        1e-12,
    )?;
    Ok((heldout_drift, annotated_drift, annotated_count))
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
    let baseline_fractions = [(1, 5), (1, 3), (1, 2)];
    let mut window_larger = [0_usize; 3];
    for (name, points) in &series {
        let mut primary = None;
        for (index, (numerator, denominator)) in baseline_fractions.iter().copied().enumerate() {
            let distances = nab_distances(name, points, numerator, denominator)?;
            window_larger[index] += usize::from(distances.1 > distances.0);
            if (numerator, denominator) == (1, 5) {
                primary = Some(distances);
            }
        }
        let (heldout_drift, annotated_drift, annotated_count) =
            primary.ok_or_else(|| invalid_data("NAB primary baseline was not evaluated"))?;
        let larger = annotated_drift > heldout_drift;
        println!(
            "{name:18} {:>6}  {:>9}  {heldout_drift:>14.4}  {annotated_drift:>16.4}  {larger:>13}",
            points.len(),
            annotated_count,
        );
    }
    println!("baseline fraction sensitivity (streams with larger annotated drift):");
    for ((numerator, denominator), larger) in baseline_fractions.into_iter().zip(window_larger) {
        println!(
            "  baseline={numerator}/{denominator}: {larger}/{}",
            series.len()
        );
    }
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
