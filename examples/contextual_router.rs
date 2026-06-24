#[cfg(not(feature = "contextual"))]
fn main() {
    eprintln!("This example requires: cargo run --example contextual_router --features contextual");
}

#[cfg(feature = "contextual")]
fn main() {
    use muxer::{LinUcb, LinUcbConfig};
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;

    let arms = vec!["small".to_string(), "big".to_string()];

    // Toy "context": two features
    // - x0: prompt length proxy
    // - x1: "difficulty" proxy
    let cfg = LinUcbConfig {
        dim: 2,
        lambda: 1.0,
        alpha: 1.0,
        seed: 0,
        decay: 1.0,
    };
    let mut pol = LinUcb::new(cfg);

    // Simulated environment: big model does better when difficulty is high, but has same reward otherwise.
    let mut env = StdRng::seed_from_u64(123);

    // Track which arm is preferred on high-difficulty contexts in the second
    // half of the run, once the policy has learned the reward structure.
    let mut big_high_diff = 0u64;
    let mut small_high_diff = 0u64;

    for t in 0..2_000u64 {
        let prompt_len = env.random_range(0.0..2000.0);
        let difficulty = env.random::<f64>(); // 0..1
        let ctx = [prompt_len / 2000.0, difficulty];

        let d = pol.decide(&arms, &ctx).unwrap();
        let chosen = d.chosen.as_str();
        let scores = pol.scores(&arms, &ctx);

        if t >= 1_000 && difficulty > 0.8 {
            if chosen == "big" {
                big_high_diff += 1;
            } else {
                small_high_diff += 1;
            }
        }

        // Reward: success probability depends on context and chosen arm.
        let p_success = match chosen {
            "small" => 0.70 - 0.25 * difficulty,
            "big" => 0.70 - 0.05 * difficulty,
            _ => 0.5,
        }
        .clamp(0.0, 1.0);

        let reward = if env.random::<f64>() < p_success {
            1.0
        } else {
            0.0
        };
        pol.update_reward(chosen, &ctx, reward);

        if t % 200 == 0 {
            eprintln!(
                "t={:4} ctx={:?} decision={:?} reward={} scores={:?}",
                t, ctx, d, reward, scores
            );
        }
    }

    // Premise: `big` keeps reward high as difficulty rises (0.70 - 0.05*d) while
    // `small` degrades faster (0.70 - 0.25*d), so on high-difficulty contexts the
    // learned policy should prefer `big`.
    eprintln!("high-difficulty picks (t>=1000): big={big_high_diff} small={small_high_diff}");
    assert!(
        big_high_diff > small_high_diff,
        "expected `big` to be preferred on high-difficulty contexts, got big={big_high_diff} small={small_high_diff}"
    );
}
