use muxer::{Exp3Ix, Exp3IxConfig};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

fn main() {
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

    for t in 0..2_000u64 {
        // Drift the environment slowly.
        if t % 400 == 0 && t > 0 {
            p.rotate_left(1);
        }

        let (chosen, probs) = ex.select_with_probs(&arms).unwrap();
        let idx = match chosen.as_str() {
            "a" => 0,
            "b" => 1,
            _ => 2,
        };
        let reward = if env.random::<f64>() < p[idx] {
            1.0
        } else {
            0.0
        };
        ex.update_reward(chosen, reward);

        if t % 200 == 0 {
            eprintln!(
                "t={:4} chosen={} reward={} probs={:?} true_p={:?}",
                t, chosen, reward, probs, p
            );
        }
    }
}
