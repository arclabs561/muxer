#[cfg(not(feature = "stochastic"))]
fn main() {
    eprintln!(
        "This example requires: cargo run --example significant_shift_sim --features stochastic"
    );
}

#[cfg(feature = "stochastic")]
mod common;

#[cfg(feature = "stochastic")]
fn main() {
    use muxer::monitor::CusumCatDetector;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    const K: usize = 2;
    const CATS: usize = 4;
    const BEST_ARM: usize = 0;
    const DRIFTED_ARM: usize = 1;
    const CHANGE_AT: u64 = 600;
    const HORIZON: u64 = 4_000;
    const TRIALS: u64 = 120;
    const COVERAGE_INTERVAL: u64 = 5;
    const RESTART_WARMUP: u64 = 60;

    const P0_BEST: [f64; CATS] = [0.92, 0.06, 0.01, 0.01];
    const P0_OTHER: [f64; CATS] = [0.74, 0.18, 0.05, 0.03];
    const P1_OTHER: [f64; CATS] = [0.55, 0.25, 0.12, 0.08];
    const ALT_BEST_DEGRADED: [f64; CATS] = [0.70, 0.18, 0.08, 0.04];

    #[derive(Debug, Clone, Copy)]
    enum Strategy {
        RestartOnAnyCusum,
        SignificantGate,
    }

    impl Strategy {
        fn name(self) -> &'static str {
            match self {
                Strategy::RestartOnAnyCusum => "restart_on_any_cusum",
                Strategy::SignificantGate => "significant_gate",
            }
        }
    }

    #[derive(Debug, Clone, Copy, Default)]
    struct Metrics {
        restarts: u64,
        suppressed: u64,
        alarms: u64,
        pulls_drifted: u64,
        reward: f64,
        regret: f64,
    }

    #[derive(Debug, Clone, Copy, Default)]
    struct Aggregate {
        restarts: f64,
        suppressed: f64,
        alarms: f64,
        pulls_drifted: f64,
        mean_reward: f64,
        regret: f64,
    }

    impl Aggregate {
        fn add(&mut self, m: Metrics) {
            self.restarts += m.restarts as f64;
            self.suppressed += m.suppressed as f64;
            self.alarms += m.alarms as f64;
            self.pulls_drifted += m.pulls_drifted as f64;
            self.mean_reward += m.reward / (HORIZON as f64);
            self.regret += m.regret;
        }

        fn mean(self, trials: u64) -> Self {
            let n = (trials as f64).max(1.0);
            Self {
                restarts: self.restarts / n,
                suppressed: self.suppressed / n,
                alarms: self.alarms / n,
                pulls_drifted: self.pulls_drifted / n,
                mean_reward: self.mean_reward / n,
                regret: self.regret / n,
            }
        }
    }

    fn expected_reward(p: [f64; CATS]) -> f64 {
        p[0] + 0.4 * p[1]
    }

    fn distribution(arm: usize, t: u64) -> [f64; CATS] {
        match (arm, t >= CHANGE_AT) {
            (BEST_ARM, _) => P0_BEST,
            (DRIFTED_ARM, false) => P0_OTHER,
            (DRIFTED_ARM, true) => P1_OTHER,
            _ => unreachable!("simulation has exactly two arms"),
        }
    }

    fn make_detectors() -> [CusumCatDetector; K] {
        core::array::from_fn(|arm| {
            let (p0, p1) = match arm {
                BEST_ARM => (P0_BEST, ALT_BEST_DEGRADED),
                DRIFTED_ARM => (P0_OTHER, P1_OTHER),
                _ => unreachable!("simulation has exactly two arms"),
            };
            CusumCatDetector::new(&p0, &p1, 1e-3, 20, 8.0, 1e-12).expect("cusum detector")
        })
    }

    fn current_best(counts: &[u64; K], reward_sum: &[f64; K]) -> usize {
        let mut best = 0usize;
        let mut best_mean = f64::NEG_INFINITY;
        for arm in 0..K {
            let mean = if counts[arm] == 0 {
                f64::NEG_INFINITY
            } else {
                reward_sum[arm] / (counts[arm] as f64)
            };
            if mean > best_mean {
                best = arm;
                best_mean = mean;
            }
        }
        best
    }

    fn choose_arm(t: u64, restart_until: u64, counts: &[u64; K], reward_sum: &[f64; K]) -> usize {
        if t < restart_until || counts.contains(&0) {
            return (t as usize) % K;
        }
        if t % COVERAGE_INTERVAL == 0 {
            return DRIFTED_ARM;
        }
        current_best(counts, reward_sum)
    }

    fn restart(
        detectors: &mut [CusumCatDetector; K],
        counts: &mut [u64; K],
        reward_sum: &mut [f64; K],
        restart_until: &mut u64,
        t: u64,
    ) {
        *detectors = make_detectors();
        *counts = [0; K];
        *reward_sum = [0.0; K];
        *restart_until = t.saturating_add(RESTART_WARMUP);
    }

    fn run_trial(seed: u64, strategy: Strategy) -> Metrics {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut detectors = make_detectors();
        let mut counts = [0u64; K];
        let mut reward_sum = [0.0f64; K];
        let mut restart_until = 0u64;
        let mut metrics = Metrics::default();

        for t in 0..HORIZON {
            let arm = choose_arm(t, restart_until, &counts, &reward_sum);
            let p = distribution(arm, t);
            let category = common::sample_cat(&mut rng, common::normalize(p));
            let reward = match category {
                0 => 1.0,
                1 => 0.4,
                _ => 0.0,
            };

            counts[arm] = counts[arm].saturating_add(1);
            reward_sum[arm] += reward;
            metrics.reward += reward;
            metrics.pulls_drifted += u64::from(arm == DRIFTED_ARM);

            let best_expected = expected_reward(distribution(BEST_ARM, t));
            metrics.regret += best_expected - expected_reward(p);

            if detectors[arm].update(category).is_some() {
                metrics.alarms = metrics.alarms.saturating_add(1);
                let best = current_best(&counts, &reward_sum);
                let significant = arm == best;

                match strategy {
                    Strategy::RestartOnAnyCusum => {
                        metrics.restarts = metrics.restarts.saturating_add(1);
                        restart(
                            &mut detectors,
                            &mut counts,
                            &mut reward_sum,
                            &mut restart_until,
                            t,
                        );
                    }
                    Strategy::SignificantGate if significant => {
                        metrics.restarts = metrics.restarts.saturating_add(1);
                        restart(
                            &mut detectors,
                            &mut counts,
                            &mut reward_sum,
                            &mut restart_until,
                            t,
                        );
                    }
                    Strategy::SignificantGate => {
                        metrics.suppressed = metrics.suppressed.saturating_add(1);
                        detectors[arm].reset();
                    }
                }
            }
        }

        metrics
    }

    fn eval(strategy: Strategy) -> Aggregate {
        let mut agg = Aggregate::default();
        for trial in 0..TRIALS {
            let seed = 0x5151_4747_u64 ^ trial.wrapping_mul(0x9E37_79B9);
            agg.add(run_trial(seed, strategy));
        }
        agg.mean(TRIALS)
    }

    let restart_any = eval(Strategy::RestartOnAnyCusum);
    let significant = eval(Strategy::SignificantGate);

    eprintln!("significant-shift simulation");
    eprintln!(
        "best arm stays arm{BEST_ARM}; arm{DRIFTED_ARM} degrades at t={CHANGE_AT}; coverage samples arm{DRIFTED_ARM} every {COVERAGE_INTERVAL} steps"
    );
    eprintln!(
        "strategy                 | restarts alarms suppressed pulls_drifted mean_reward regret"
    );
    for (strategy, m) in [
        (Strategy::RestartOnAnyCusum, restart_any),
        (Strategy::SignificantGate, significant),
    ] {
        eprintln!(
            "{:<24} | {:8.2} {:6.2} {:10.2} {:13.1} {:11.4} {:7.1}",
            strategy.name(),
            m.restarts,
            m.alarms,
            m.suppressed,
            m.pulls_drifted,
            m.mean_reward,
            m.regret
        );
    }

    assert!(
        significant.restarts < restart_any.restarts,
        "significance gate should reduce global restarts"
    );
    assert!(
        significant.regret < restart_any.regret,
        "significance gate should preserve route quality by avoiding harmless restarts"
    );
}
