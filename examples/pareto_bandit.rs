//! Multi-armed bandit with explicit Pareto frontier analysis.
//!
//! Demonstrates two complementary views of the same arm data:
//!
//! 1. `muxer::select_mab` -- deterministic Pareto + scalarization selection.
//! 2. `pare::ParetoFrontier` -- explicit frontier construction with crowding
//!    distances and hypervolume.
//!
//! Five arms simulate different quality/cost/latency tradeoffs.  After feeding
//! 200 outcomes per arm through `Window`s, we compare which arms `select_mab`
//! picks versus which arms survive on the 3-objective Pareto frontier.
//!
//! Run with:
//!   cargo run --example pareto_bandit

use std::collections::BTreeMap;

use muxer::{select_mab, MabConfig, Outcome, Summary, Window};
use pare::{Direction, ParetoFrontier};

// -----------------------------------------------------------------------
// Deterministic xorshift64 RNG (no external dependency).
// -----------------------------------------------------------------------

struct Xorshift64(u64);

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Bernoulli trial: returns true with probability `p`.
    fn bernoulli(&mut self, p: f64) -> bool {
        self.next_f64() < p
    }
}

// -----------------------------------------------------------------------
// Arm profile: deterministic parameters for each simulated arm.
// -----------------------------------------------------------------------

struct ArmProfile {
    name: &'static str,
    ok_prob: f64,
    junk_prob: f64,
    hard_junk_prob: f64,
    cost_units: u64,
    elapsed_ms: u64,
}

const ARMS: [ArmProfile; 5] = [
    ArmProfile {
        name: "fast-cheap",
        ok_prob: 0.88,
        junk_prob: 0.20,
        hard_junk_prob: 0.05,
        cost_units: 2,
        elapsed_ms: 50,
    },
    ArmProfile {
        name: "slow-quality",
        ok_prob: 0.95,
        junk_prob: 0.04,
        hard_junk_prob: 0.01,
        cost_units: 15,
        elapsed_ms: 800,
    },
    ArmProfile {
        name: "balanced",
        ok_prob: 0.91,
        junk_prob: 0.10,
        hard_junk_prob: 0.03,
        cost_units: 7,
        elapsed_ms: 200,
    },
    ArmProfile {
        name: "budget",
        ok_prob: 0.72,
        junk_prob: 0.30,
        hard_junk_prob: 0.12,
        cost_units: 1,
        elapsed_ms: 100,
    },
    ArmProfile {
        name: "premium",
        ok_prob: 0.97,
        junk_prob: 0.02,
        hard_junk_prob: 0.005,
        cost_units: 25,
        elapsed_ms: 400,
    },
];

fn main() {
    let n_outcomes: usize = 200;

    // -----------------------------------------------------------------
    // 1. Simulate outcomes and build Windows.
    // -----------------------------------------------------------------
    let mut windows: BTreeMap<String, Window> = BTreeMap::new();
    let mut rng = Xorshift64::new(0xDEAD_BEEF_CAFE);

    for profile in &ARMS {
        let mut win = Window::new(n_outcomes);
        for _ in 0..n_outcomes {
            let ok = rng.bernoulli(profile.ok_prob);
            let junk = rng.bernoulli(profile.junk_prob);
            let hard_junk = junk && rng.bernoulli(profile.hard_junk_prob / profile.junk_prob);
            win.push(Outcome {
                ok,
                junk,
                hard_junk,
                cost_units: profile.cost_units,
                elapsed_ms: profile.elapsed_ms,
                quality_score: None,
            });
        }
        windows.insert(profile.name.to_string(), win);
    }

    // -----------------------------------------------------------------
    // 2. Compute Summaries and print per-arm stats.
    // -----------------------------------------------------------------
    let summaries: BTreeMap<String, Summary> = windows
        .iter()
        .map(|(k, w)| (k.clone(), w.summary()))
        .collect();

    println!("--- Arm Summaries ({n_outcomes} outcomes each) ---");
    println!(
        "{:<14} {:>6} {:>8} {:>10} {:>10}",
        "arm", "calls", "ok_rate", "junk_rate", "mean_cost"
    );
    for (name, s) in &summaries {
        println!(
            "{:<14} {:>6} {:>8.4} {:>10.4} {:>10.2}",
            name,
            s.calls,
            s.ok_rate(),
            s.junk_rate(),
            s.mean_cost_units(),
        );
    }

    // -----------------------------------------------------------------
    // 3. muxer::select_mab -- built-in selection.
    // -----------------------------------------------------------------
    let arm_names: Vec<String> = ARMS.iter().map(|a| a.name.to_string()).collect();

    let cfg = MabConfig {
        exploration_c: 0.0, // pure exploitation, no UCB bonus
        cost_weight: 0.3,
        junk_weight: 0.5,
        ..MabConfig::default()
    };

    let selection = select_mab(&arm_names, &summaries, cfg);
    println!("\n--- muxer::select_mab ---");
    println!("chosen:   {}", selection.chosen);
    println!("frontier: {:?}", selection.frontier);

    // -----------------------------------------------------------------
    // 4. Explicit pare::ParetoFrontier with 3 objectives.
    //    Maximize ok_rate, Minimize mean_cost_units, Minimize junk_rate.
    // -----------------------------------------------------------------
    let directions = vec![
        Direction::Maximize,
        Direction::Minimize,
        Direction::Minimize,
    ];
    let labels = vec![
        "ok_rate".to_string(),
        "mean_cost".to_string(),
        "junk_rate".to_string(),
    ];

    let mut frontier: ParetoFrontier<String> = ParetoFrontier::new(directions).with_labels(labels);

    // Track all arms (including dominated ones) for the comparison table.
    let mut all_arm_objectives: Vec<(String, Vec<f64>, bool)> = Vec::new();

    for (name, s) in &summaries {
        let values = vec![s.ok_rate(), s.mean_cost_units(), s.junk_rate()];
        let on_frontier = frontier.push(values.clone(), name.clone());
        all_arm_objectives.push((name.clone(), values, on_frontier));
    }

    println!("\n--- pare::ParetoFrontier (3 objectives) ---");
    println!(
        "{:<14} {:>8} {:>10} {:>10}  {}",
        "arm", "ok_rate", "mean_cost", "junk_rate", "status"
    );
    for (name, vals, on_front) in &all_arm_objectives {
        let status = if *on_front { "FRONTIER" } else { "dominated" };
        println!(
            "{:<14} {:>8.4} {:>10.2} {:>10.4}  {}",
            name, vals[0], vals[1], vals[2], status
        );
    }

    // -----------------------------------------------------------------
    // 5. Crowding distances -- diversity measure on the frontier.
    // -----------------------------------------------------------------
    let cd = frontier.crowding_distances();
    println!("\n--- Crowding Distances (frontier points only) ---");
    for (i, pt) in frontier.points().iter().enumerate() {
        let cd_str = if cd[i].is_infinite() {
            "inf (boundary)".to_string()
        } else {
            format!("{:.4}", cd[i])
        };
        println!("  {:<14}  cd = {}", pt.data, cd_str);
    }

    // -----------------------------------------------------------------
    // 6. Hypervolume -- aggregate quality of the frontier.
    //    Reference point: ok_rate=0.0, mean_cost=50.0, junk_rate=1.0.
    //    (worst plausible values in each objective's direction)
    // -----------------------------------------------------------------
    let ref_point = vec![0.0, 50.0, 1.0];
    let hv = frontier.hypervolume(&ref_point);
    println!("\n--- Hypervolume ---");
    println!("ref_point: [ok_rate=0.0, mean_cost=50.0, junk_rate=1.0]");
    println!("hypervolume: {hv:.4}");

    // -----------------------------------------------------------------
    // 7. Scalar scores -- weighted ranking of frontier points.
    //    Weights: ok_rate=2.0, mean_cost=1.0, junk_rate=1.5.
    // -----------------------------------------------------------------
    let weights = [2.0, 1.0, 1.5];
    println!("\n--- Scalar Scores (weights: ok_rate=2.0, cost=1.0, junk=1.5) ---");
    for (i, pt) in frontier.points().iter().enumerate() {
        let score = frontier.scalar_score(i, &weights);
        println!("  {:<14}  score = {:.4}", pt.data, score);
    }

    // -----------------------------------------------------------------
    // 8. Compare: muxer selection vs Pareto analysis.
    // -----------------------------------------------------------------
    let frontier_names: Vec<&str> = frontier.points().iter().map(|p| p.data.as_str()).collect();
    let muxer_on_pareto = frontier_names.contains(&selection.chosen.as_str());

    println!("\n--- Comparison ---");
    println!(
        "muxer chose '{}' -- on explicit Pareto frontier: {}",
        selection.chosen, muxer_on_pareto
    );
    println!("muxer internal frontier: {:?}", selection.frontier);
    println!("pare explicit frontier:  {:?}", frontier_names);
    println!(
        "\nNote: muxer uses 6 internal objectives (ok_rate, junk_rate, hard_junk_rate,\n\
         mean_cost, mean_latency, exploration bonus) while this Pareto analysis uses 3.\n\
         The two frontiers may differ when the extra dimensions matter."
    );
}
