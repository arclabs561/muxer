use muxer::{ThompsonConfig, ThompsonSampling};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

fn main() {
    let arms = vec!["a".to_string(), "b".to_string()];

    // Simulated true success probabilities.
    let true_p_a = 0.70;
    let true_p_b = 0.55;

    // Thompson sampling is good when you can supply a scalar reward per call.
    // Here: reward = 1 for success, 0 for failure.
    let mut ts = ThompsonSampling::with_seed(
        ThompsonConfig {
            decay: 0.995, // mild forgetting to track drift
            ..ThompsonConfig::default()
        },
        0,
    );

    // Separate RNG to simulate the environment.
    let mut env = StdRng::seed_from_u64(123);

    for t in 0..1_000u64 {
        let (chosen, probs) = ts.select_softmax_mean_with_probs(&arms, 0.25).unwrap();
        let p = if chosen.as_str() == "a" {
            true_p_a
        } else {
            true_p_b
        };
        let reward = if env.random::<f64>() < p { 1.0 } else { 0.0 };
        ts.update_reward(chosen, reward);

        if t % 100 == 0 {
            eprintln!(
                "t={:4} chosen={} reward={} alloc={:?}",
                t, chosen, reward, probs
            );
        }
    }
}
