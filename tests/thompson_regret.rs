//! Thompson sampling on stationary Bernoulli arms has logarithmic expected
//! regret (Agrawal & Goyal 2012; Kaufmann et al. 2012), so the per-round
//! regret must shrink as the horizon grows. A policy with linear regret, such
//! as uniform play, keeps a constant per-round regret and fails this test.
#![cfg(feature = "stochastic")]

use muxer::{ThompsonConfig, ThompsonSampling};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[test]
fn thompson_regret_on_seeded_bernoulli_arms_is_sublinear() {
    let arms: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
    let means = [0.5, 0.6, 0.7];
    let best = 0.7;
    let (t_short, t_long) = (1_000usize, 8_000usize);

    let (mut regret_short, mut regret_long) = (0.0, 0.0);
    for seed in 0..10u64 {
        let mut ts = ThompsonSampling::with_seed(ThompsonConfig::default(), seed);
        let mut env = StdRng::seed_from_u64(1_000 + seed);
        let mut regret = 0.0;
        for t in 1..=t_long {
            let arm = ts.select(&arms).expect("non-empty arms").clone();
            let i = arms.iter().position(|a| *a == arm).unwrap();
            // Pseudo-regret: gap of the chosen arm's mean to the best mean.
            regret += best - means[i];
            let reward = if env.random::<f64>() < means[i] {
                1.0
            } else {
                0.0
            };
            ts.update_reward(&arm, reward);
            if t == t_short {
                regret_short += regret;
            }
        }
        regret_long += regret;
    }

    let rate_short = regret_short / (10.0 * t_short as f64);
    let rate_long = regret_long / (10.0 * t_long as f64);
    // Uniform play has per-round regret 0.1 at both horizons.
    assert!(
        rate_long < 0.02,
        "per-round regret at T={t_long}: {rate_long:.4}"
    );
    assert!(
        rate_long < 0.5 * rate_short,
        "per-round regret did not shrink: {rate_short:.4} at T={t_short}, {rate_long:.4} at T={t_long}"
    );
}
