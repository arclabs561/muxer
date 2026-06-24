#[cfg(not(feature = "stochastic"))]
fn main() {
    eprintln!("This example requires: cargo run --example exp3ix_router --features stochastic");
}

#[cfg(feature = "stochastic")]
fn main() {
    use muxer::{Exp3Ix, Exp3IxConfig};
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;

    let arms = vec!["a".to_string(), "b".to_string(), "c".to_string()];

    // Simulated true reward probabilities (adversarial-ish: they drift).
    let mut p = [0.55f64, 0.50f64, 0.45f64];

    let mut ex = Exp3Ix::new(Exp3IxConfig {
        seed: 0,
        horizon: 5_000,
        confidence_delta: None,
        decay: 0.99,
    });

    let mut env = StdRng::seed_from_u64(123);

    // Accumulate the probability Exp3-IX assigns to the *currently-best* arm in a
    // stable late window (a settled segment between rotations), and compare to the
    // uniform baseline of 1/k per round. This is a deterministic check on the
    // policy's distribution, free of single-run reward sampling noise.
    let mut best_arm_prob_sum = 0.0f64;
    let mut uniform_prob_sum = 0.0f64;
    let mut window_rounds = 0u64;

    for t in 0..2_000u64 {
        // Drift the environment slowly.
        if t % 400 == 0 && t > 0 {
            p.rotate_left(1);
        }

        let d = ex.decide(&arms).unwrap();
        let chosen = d.chosen.as_str();
        let probs = d.probs.clone().unwrap_or_default();
        let idx = match chosen {
            "a" => 0,
            "b" => 1,
            _ => 2,
        };

        // Window 1600..2000 is a single stable segment (last rotation is at 1600).
        if (1_600..2_000).contains(&t) {
            let best_idx = p
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            let best_name = arms[best_idx].as_str();
            best_arm_prob_sum += probs.get(best_name).copied().unwrap_or(0.0);
            uniform_prob_sum += 1.0 / arms.len() as f64;
            window_rounds += 1;
        }

        let reward = if env.random::<f64>() < p[idx] {
            1.0
        } else {
            0.0
        };
        ex.update_reward(chosen, reward);

        if t % 200 == 0 {
            eprintln!(
                "t={:4} decision={:?} reward={} true_p={:?} probs={:?}",
                t, d, reward, p, probs
            );
        }
    }

    // Premise: Exp3-IX tracks the rotating best arm, so within a settled window it
    // weights the current best arm above the uniform 1/k it would assign with no
    // learning.
    let mean_best = best_arm_prob_sum / window_rounds as f64;
    let mean_uniform = uniform_prob_sum / window_rounds as f64;
    eprintln!("late window: mean P(best)={mean_best:.4} vs uniform={mean_uniform:.4}");
    assert!(
        mean_best > mean_uniform,
        "expected Exp3-IX to weight the current best arm above uniform in the settled window, got mean P(best)={mean_best:.4} uniform={mean_uniform:.4}"
    );
}
