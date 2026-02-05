#[cfg(not(feature = "stochastic"))]
fn main() {
    eprintln!("This example requires: cargo run --example coverage_autotune --features stochastic");
}

#[cfg(feature = "stochastic")]
mod common;

#[cfg(feature = "stochastic")]
fn main() {
    use muxer::monitor::CusumCatDetector;
    use muxer::{coverage_pick_under_sampled_idx, CoverageConfig};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    // Coverage autotune toy:
    //
    // - K arms, categorical outcomes in 4 buckets (muxer-style)
    // - One (unknown) arm changes from p0 -> p1 at time nu.
    // - Per-arm CUSUM(p0 vs p1) is maintained; we STOP when any arm alarms.
    // - Sampling policy:
    //     1) enforce per-arm coverage quotas (CoverageConfig)
    //     2) otherwise, focus on the arm with max CUSUM score
    //
    // The "autotune" step connects a wall-delay target W to a minimum coverage fraction:
    //
    //   E_p1[LLR] = KL(p1||p0)  (when CUSUM uses the same smoothed p0/p1)
    //   => expected samples to reach threshold h:  N ≈ h / KL
    //   => if an arm is sampled at rate r per wall step, wall delay is ~ N / r
    //   => choose coverage floor r := N / W.
    //
    // This is conservative: once the policy correctly focuses, the changed arm can get sampled faster
    // than the floor and beat the target.

    const K: usize = 6;
    const CATS: usize = 4;

    #[derive(Debug, Clone, Copy, Default)]
    struct Metrics {
        false_alarm: bool,
        detected_correct: bool, // alarmed after nu on the changed arm
        wrong_arm_after_nu: bool,
        // Wall-clock delay after nu (only for correct detections).
        wall_delay: Option<u64>,
        // Number of post-change samples from the changed arm until stop (only for correct detections).
        changed_arm_post_samples: Option<u64>,
        // Post-change sampling rate of the changed arm: changed_post_samples / (wall_delay + 1).
        //
        // This is often much larger than the *overall* fraction, because a good focus policy
        // concentrates sampling after it gains confidence.
        post_rate_on_changed: Option<f64>,
        // Fraction of pulls spent on the changed arm.
        frac_on_changed: f64,
    }

    #[derive(Debug, Clone, Copy)]
    struct TrialCfg {
        nu: u64,
        horizon: u64,
        // CUSUM params.
        alpha_smooth: f64,
        min_n: u64,
        threshold: f64,
        tol: f64,
        p0: [f64; CATS],
        p1: [f64; CATS],
    }

    fn run_trial(seed: u64, cov: CoverageConfig, cfg: TrialCfg) -> Metrics {
        let mut rng = StdRng::seed_from_u64(seed);
        let changed = (rng.random::<u64>() as usize) % K;

        let mut cusum: Vec<CusumCatDetector> = (0..K)
            .map(|_| {
                CusumCatDetector::new(
                    &cfg.p0,
                    &cfg.p1,
                    cfg.alpha_smooth,
                    cfg.min_n,
                    cfg.threshold,
                    cfg.tol,
                )
                .expect("cusum")
            })
            .collect();

        let mut pulls: [u64; K] = [0; K];
        let mut changed_post_samples = 0u64;
        let mut stopped_at: Option<u64> = None;
        let mut alarm_arm: Option<usize> = None;

        let argmax = |scores: &[f64; K], tol: f64| -> usize {
            let mut best = 0usize;
            let mut best_s = f64::NEG_INFINITY;
            for (i, &s) in scores.iter().enumerate() {
                if s > best_s || ((s - best_s).abs() <= tol && i < best) {
                    best_s = s;
                    best = i;
                }
            }
            best
        };

        for t in 0..cfg.horizon {
            // 1) Coverage picks (maintenance sampling).
            let cov_pick =
                coverage_pick_under_sampled_idx(seed ^ 0xC0DE_D00D ^ t, K, 1, cov, |idx| {
                    pulls[idx]
                });
            let arm = if let Some(&first) = cov_pick.first() {
                first
            } else {
                // 2) Focus on the most suspicious arm.
                let mut scores = [0.0f64; K];
                for k in 0..K {
                    scores[k] = cusum[k].score();
                }
                argmax(&scores, cfg.tol)
            };

            pulls[arm] = pulls[arm].saturating_add(1);
            let p = if arm == changed && t >= cfg.nu {
                cfg.p1
            } else {
                cfg.p0
            };
            let x = common::sample_cat(&mut rng, p);
            if arm == changed && t >= cfg.nu {
                changed_post_samples = changed_post_samples.saturating_add(1);
            }

            if cusum[arm].update(x).is_some() {
                stopped_at = Some(t);
                alarm_arm = Some(arm);
                break;
            }
        }

        let total_pulls: u64 = pulls.iter().sum::<u64>().max(1);
        let frac_on_changed = (pulls[changed] as f64) / (total_pulls as f64);
        let after_nu = stopped_at.is_some() && stopped_at.unwrap_or(0) >= cfg.nu;
        let false_alarm = stopped_at.is_some() && stopped_at.unwrap_or(0) < cfg.nu;
        let detected_correct = after_nu && alarm_arm == Some(changed);
        let wrong_arm_after_nu = after_nu && alarm_arm != Some(changed);
        let post_rate_on_changed = stopped_at.and_then(|t| {
            if detected_correct && t >= cfg.nu {
                let post_pulls = t.saturating_sub(cfg.nu).saturating_add(1);
                let denom = (post_pulls as f64).max(1.0);
                Some((changed_post_samples as f64) / denom)
            } else {
                None
            }
        });

        Metrics {
            false_alarm,
            detected_correct,
            wrong_arm_after_nu,
            wall_delay: stopped_at.and_then(|t| {
                if detected_correct && t >= cfg.nu {
                    Some(t - cfg.nu)
                } else {
                    None
                }
            }),
            changed_arm_post_samples: detected_correct.then_some(changed_post_samples),
            post_rate_on_changed,
            frac_on_changed,
        }
    }

    fn mean_opt(xs: &[Option<u64>]) -> Option<f64> {
        let mut sum = 0.0;
        let mut n = 0.0;
        for x in xs {
            if let Some(v) = *x {
                sum += v as f64;
                n += 1.0;
            }
        }
        if n > 0.0 {
            Some(sum / n)
        } else {
            None
        }
    }

    fn pctl_u64(xs: &[Option<u64>], q: f64) -> Option<u64> {
        let mut ds: Vec<u64> = xs.iter().copied().flatten().collect();
        if ds.is_empty() {
            return None;
        }
        let q = if q.is_finite() {
            q.clamp(0.0, 1.0)
        } else {
            0.5
        };
        ds.sort_unstable();
        let idx = ((ds.len().saturating_sub(1) as f64) * q).round() as usize;
        ds.get(idx).copied()
    }

    fn fmt_mean_p90(xs: &[Option<u64>]) -> String {
        let mean = mean_opt(xs);
        let p90 = pctl_u64(xs, 0.90);
        match (mean, p90) {
            (Some(m), Some(p)) => format!("{m:7.1}/{p:>6}"),
            (Some(m), None) => format!("{m:7.1}/ never"),
            (None, _) => "  never".into(),
        }
    }

    fn mean_f64_opt(xs: &[Option<f64>]) -> Option<f64> {
        let mut sum = 0.0;
        let mut n = 0.0;
        for x in xs {
            if let Some(v) = *x {
                if v.is_finite() {
                    sum += v;
                    n += 1.0;
                }
            }
        }
        if n > 0.0 {
            Some(sum / n)
        } else {
            None
        }
    }

    // Scenario: mild shift (small KL) so coverage matters.
    let p0 = common::normalize([0.90, 0.03, 0.02, 0.05]);
    let p1 = common::normalize([0.80, 0.05, 0.05, 0.10]);

    let alpha_smooth = 1e-3;
    let min_n = 20;
    let threshold = 12.0;
    let tol = 1e-12;

    // KL computed on the smoothed p0/p1 used by CUSUM (so the h/KL prediction is apples-to-apples).
    let denom = 1.0 + alpha_smooth * (CATS as f64);
    let p0s = p0.map(|x| (x + alpha_smooth) / denom);
    let p1s = p1.map(|x| (x + alpha_smooth) / denom);
    let kl = logp::kl_divergence(&p1s, &p0s, tol).unwrap_or(f64::NAN);
    let pred_samp = if kl.is_finite() && kl > 0.0 {
        threshold / kl
    } else {
        f64::INFINITY
    };

    let nu = 20_000u64;
    let horizon = 60_000u64;
    let trials = 250u64;
    let target_walls: [u64; 5] = [1_000, 2_000, 5_000, 10_000, 20_000];

    println!("== coverage_autotune ==");
    println!("K={K} nu={nu} horizon={horizon} trials={trials}");
    println!(
        "CUSUM: alpha={alpha_smooth} min_n={min_n} thr={threshold:.1} | KL(p1||p0)≈{kl:.6} => pred_samples h/KL≈{pred_samp:.1}"
    );
    println!("(note: wall/post columns are mean/p90 over *correct* detections)");
    println!("targetW | cov_frac  pred_wall | fa   ok  wrong | wall(mean/p90)  post(mean/p90)  post_rate  mean_frac");

    for &target_w in &target_walls {
        let cov_cap = 1.0 / (K as f64);
        let cov_raw = if pred_samp.is_finite() && target_w > 0 {
            (pred_samp / (target_w as f64)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let cov_frac = cov_raw.min(cov_cap);
        let pred_wall = if cov_frac > 0.0 {
            pred_samp / cov_frac
        } else {
            f64::INFINITY
        };
        let capped = cov_raw > cov_cap;

        let cov = CoverageConfig {
            enabled: true,
            min_fraction: cov_frac,
            min_calls_floor: min_n,
        };
        let cfg = TrialCfg {
            nu,
            horizon,
            alpha_smooth,
            min_n,
            threshold,
            tol,
            p0,
            p1,
        };

        let mut fa = 0u64;
        let mut ok = 0u64;
        let mut wrong = 0u64;
        let mut walls: Vec<Option<u64>> = Vec::with_capacity(trials as usize);
        let mut posts: Vec<Option<u64>> = Vec::with_capacity(trials as usize);
        let mut post_rates: Vec<Option<f64>> = Vec::with_capacity(trials as usize);
        let mut fracs: Vec<f64> = Vec::with_capacity(trials as usize);

        for i in 0..trials {
            let seed = 0xC0FE_4A70_u64 ^ ((i + 1) * 0x9E37_79B9) ^ target_w;
            let m = run_trial(seed, cov, cfg);
            fa += m.false_alarm as u64;
            ok += m.detected_correct as u64;
            wrong += m.wrong_arm_after_nu as u64;
            walls.push(m.wall_delay);
            posts.push(m.changed_arm_post_samples);
            post_rates.push(m.post_rate_on_changed);
            fracs.push(m.frac_on_changed);
        }

        let fa_rate = (fa as f64) / (trials as f64);
        let ok_rate = (ok as f64) / (trials as f64);
        let wrong_rate = (wrong as f64) / (trials as f64);
        let wall = fmt_mean_p90(&walls);
        let post = fmt_mean_p90(&posts);
        let mean_post_rate = mean_f64_opt(&post_rates).unwrap_or(f64::NAN);
        let mean_frac = fracs.iter().sum::<f64>() / (fracs.len() as f64).max(1.0);

        println!(
            "{:>7} | {:>7.4}{} {:>9.1} | {:>4.3} {:>4.3} {:>5.3} | {}   {}   {:>7.3}   {:>7.3}",
            target_w,
            cov_frac,
            if capped { "*" } else { " " },
            pred_wall,
            fa_rate,
            ok_rate,
            wrong_rate,
            wall,
            post,
            mean_post_rate,
            mean_frac
        );
    }
}
