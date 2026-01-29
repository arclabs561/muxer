#[cfg(not(feature = "contextual"))]
fn main() {
    eprintln!(
        "This example requires: cargo run --example contextual_propensity_logging --features contextual"
    );
}

#[cfg(feature = "contextual")]
fn main() {
    use muxer::{LinUcb, LinUcbConfig};
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;

    // Motivated example: when doing contextual routing, it's often useful to log
    // (chosen_arm, probability distribution) so you can do offline analysis later.
    let arms = vec!["small".to_string(), "big".to_string()];

    let mut pol = LinUcb::new(LinUcbConfig {
        dim: 2,
        alpha: 1.0,
        lambda: 1.0,
        seed: 0,
        decay: 0.99,
    });

    let mut env = StdRng::seed_from_u64(123);
    let temperature = 0.35;

    for t in 0..1_000u64 {
        let prompt_len = env.random_range(0.0..2000.0);
        let difficulty = env.random::<f64>();
        let ctx = [prompt_len / 2000.0, difficulty];

        let d = pol.decide_softmax_ucb(&arms, &ctx, temperature).unwrap();
        let chosen = d.chosen.as_str();
        let probs = d.probs.clone().unwrap_or_default();

        // Example: compute a "propensity" for the chosen arm (approximate).
        let propensity = probs.get(chosen).copied().unwrap_or(0.0);

        // Simulated reward: big does better when difficulty is high.
        let p_success = match chosen {
            "small" => (0.70 - 0.25 * difficulty).clamp(0.0, 1.0),
            "big" => (0.70 - 0.05 * difficulty).clamp(0.0, 1.0),
            _ => 0.5,
        };
        let reward = if env.random::<f64>() < p_success {
            1.0
        } else {
            0.0
        };
        pol.update_reward(chosen, &ctx, reward);

        if t % 200 == 0 {
            eprintln!(
                "t={:4} ctx={:?} decision={:?} propensity={:.3} reward={} probs={:?}",
                t, ctx, d, propensity, reward, probs
            );
        }
    }
}
