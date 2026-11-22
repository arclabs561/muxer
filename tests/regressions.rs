//! Regression tests for documented behavior that previously failed.

#[cfg(feature = "boltzmann")]
mod boltzmann {
    use muxer::{
        BanditPolicy, BoltzmannConfig, BoltzmannPolicy, BoltzmannProfile, FiniteReward, Muxer,
    };

    fn names(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| (*s).to_owned()).collect()
    }

    /// `Decision::probs` must be the distribution `decide` samples from.
    /// Means of 2e7 and 2e7 + 1 give softmax P(b) = e / (1 + e) ~ 0.731; the
    /// two logits are equal once rounded to f32, which samples P(b) = 0.5.
    #[test]
    fn bug_boltzmann_decide_disagrees_with_logged_probs() {
        let arms = names(&["a", "b"]);
        let mut p = BoltzmannPolicy::new(BoltzmannConfig::default());
        p.update_reward("a", 2.0e7);
        p.update_reward("b", 2.0e7 + 1.0);
        let n = 20_000;
        let mut logged = 0.0;
        let mut chose_b = 0usize;
        for _ in 0..n {
            let d = p.decide(&arms).unwrap();
            logged = d.probs.as_ref().unwrap()["b"];
            chose_b += usize::from(d.chosen == "b");
        }
        let freq = chose_b as f64 / n as f64;
        // 0.02 is about six binomial standard deviations at n = 20_000.
        assert!(
            (freq - logged).abs() < 0.02,
            "Decision.probs[b]={logged:.4} but empirical P(b)={freq:.4}"
        );
    }

    /// Finite rewards whose running sum overflows must not leave the profile
    /// unable to decide: the overflowing feedback is rejected and the policy
    /// keeps its previous finite state.
    #[test]
    fn bug_boltzmann_profile_reward_sum_overflow_bricks_decide() {
        let mut m = Muxer::new(
            names(&["a", "b"]),
            BoltzmannProfile::new(BoltzmannConfig::default()),
        )
        .unwrap();
        let big = FiniteReward::new(f64::MAX / 2.0 * 1.5).unwrap();
        let (mut accepted, mut rejected) = (0, 0);
        for _ in 0..200 {
            if accepted + rejected == 2 {
                break;
            }
            let d = m.decide(&()).unwrap();
            if d.action() == "a" {
                match m.tell(d.id(), big) {
                    Ok(_) => accepted += 1,
                    Err(_) => rejected += 1,
                }
            } else {
                m.tell(d.id(), FiniteReward::new(0.0).unwrap()).unwrap();
            }
        }
        assert_eq!(
            (accepted, rejected),
            (1, 1),
            "second reward overflows the sum"
        );
        let mean = m.policy().inner().mean_reward("a");
        assert!(mean.is_finite(), "mean(a)={mean}");
        assert!(
            m.decide(&()).is_ok(),
            "decide failed after finite rewards; mean(a)={mean}"
        );
    }
}

#[cfg(feature = "stochastic")]
mod cusum {
    use muxer::simulate_cusum_null_max_scores;

    /// `min_n` is the minimum observation count before the bank can alarm, so
    /// the simulated null max-score is taken over rounds with `n >= min_n`.
    /// On the same seeded null sequences, raising `min_n` can only exclude
    /// rounds and therefore never raises a trial's maximum.
    #[test]
    fn bug_cusum_null_max_ignores_min_n() {
        let p0 = vec![0.85, 0.05, 0.05, 0.05];
        let alts = vec![vec![0.40, 0.10, 0.40, 0.10]];
        let m = 40;
        let early = simulate_cusum_null_max_scores(&p0, &alts, m, 1e-3, 1, 2_000, 7).unwrap();
        let late = simulate_cusum_null_max_scores(&p0, &alts, m, 1e-3, m, 2_000, 7).unwrap();
        assert!(early.iter().zip(&late).all(|(e, l)| l <= e));
        assert_ne!(
            early, late,
            "min_n=1 and min_n=m produce identical null max-score samples"
        );
    }
}
