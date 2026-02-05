#[cfg(not(feature = "stochastic"))]
fn main() {
    eprintln!("This example requires: cargo run --example bqcd_sampling --features stochastic");
}

#[cfg(feature = "stochastic")]
mod common;

#[cfg(feature = "stochastic")]
mod bqcd_world;

#[cfg(feature = "stochastic")]
fn main() {
    use muxer::monitor::{CatKlDetector, CusumCatDetector, DriftMetric};
    use muxer::{coverage_pick_under_sampled_idx, CoverageConfig};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    // BQCD-inspired sampling toy:
    //
    // - K arms, each arm k emits categorical observations in 4 buckets (muxer-style):
    //   [ok_clean, ok_soft_junk, ok_hard_junk, fail]
    // - Initially all arms follow their baseline p0_k.
    // - At time nu, ONE arm changes to p1_k (unknown to the sampler).
    // - We only observe an arm when we sample it (bandit sensing).
    //
    // Detection mechanism (intentionally simple):
    // - Each arm has a known baseline p0_k (simplex over 4 outcome categories).
    // - We maintain two per-arm detectors:
    //   - CatKL:     S_n = n * KL(q_hat || p0_k)   (cumulative drift score; inertia grows with history)
    //   - CUSUM:     reflected LLR against a fixed alternative p_alt (forgetful; quick alarms)
    // - We STOP on the CUSUM alarm (this matches what we learned in detector_inertia/calibration:
    //   CatKL is a useful drift *feature*, but a poor low-latency *alarm* without forgetting/windowing).
    //
    // Sampling policies compared:
    // - round_robin: uniform monitoring (strong baseline, high coverage)
    // - eps_focus: epsilon exploration + "focus" on most suspicious arm
    //   - suspicion score can be CatKL score or CUSUM score (forgetful)
    //   - optional explicit coverage quota (muxer::CoverageConfig) to prevent starvation / lock-on-to-noise
    //
    // This is not a full reproduction of Gopalan et al. (they use GLR/LR structure),
    // but it tests the key structural claim from BQCD: with partial observations,
    // you must balance exploration (identify informative actions) with exploitation (sample them heavily).

    const K: usize = 6;
    const CATS: usize = 4;

    #[derive(Debug, Clone, Copy)]
    enum Suspicion {
        CatKl,
        Cusum,
    }

    impl Suspicion {
        fn name(&self) -> String {
            match *self {
                Suspicion::CatKl => "catkl".into(),
                Suspicion::Cusum => "cusum".into(),
            }
        }
    }

    #[derive(Debug, Clone, Copy)]
    enum Policy {
        RoundRobin,
        EpsFocus {
            eps: f64,
            suspicion: Suspicion,
            coverage: CoverageConfig,
        },
    }

    impl Policy {
        fn name(&self) -> String {
            match *self {
                Policy::RoundRobin => "round_robin".into(),
                Policy::EpsFocus {
                    eps,
                    suspicion,
                    coverage,
                } => {
                    let cov = if coverage.enabled {
                        format!(
                            "cov(frac={:.3},floor={})",
                            coverage.min_fraction, coverage.min_calls_floor
                        )
                    } else {
                        "cov(off)".into()
                    };
                    format!("eps_focus(eps={eps:.3}, {}, {})", suspicion.name(), cov)
                }
            }
        }
    }

    #[derive(Debug, Clone, Copy, Default)]
    struct Metrics {
        false_alarm: bool,
        detected: bool, // detected after nu
        // Wall-clock delay after nu (if detection occurs after change).
        wall_delay: Option<u64>,
        // Number of post-change samples from the *changed arm* until stop.
        changed_arm_post_samples: Option<u64>,
        // Fraction of pulls spent on changed arm (monitoring focus).
        frac_on_changed: f64,
    }

    #[derive(Debug, Clone, Copy)]
    struct DetectorCfg {
        alpha_smooth: f64,
        min_n: u64,
        cusum_threshold: f64,
        cusum_alt_p: [f64; CATS],
        tol: f64,
    }

    #[derive(Debug, Clone, Copy)]
    struct TrialCfg {
        nu: u64,
        horizon: u64,
        det: DetectorCfg,
    }

    fn run_trial_cusum_alarm(seed: u64, policy: Policy, cfg: TrialCfg) -> (usize, Metrics) {
        let DetectorCfg {
            alpha_smooth,
            min_n,
            cusum_threshold,
            cusum_alt_p,
            tol,
        } = cfg.det;
        let nu = cfg.nu;
        let horizon = cfg.horizon;
        let (p0, p1) = bqcd_world::make_world();
        let mut rng = StdRng::seed_from_u64(seed);

        // Change arm picked uniformly at random.
        let changed = (rng.random::<u64>() as usize) % K;

        let mut catkl: Vec<CatKlDetector> = (0..K)
            .map(|k| CatKlDetector::new(&p0[k], alpha_smooth, min_n, f64::INFINITY, tol).unwrap())
            .collect();
        let mut cusum: Vec<CusumCatDetector> = (0..K)
            .map(|k| {
                CusumCatDetector::new(
                    &p0[k],
                    &cusum_alt_p,
                    alpha_smooth,
                    min_n,
                    cusum_threshold,
                    tol,
                )
                .unwrap()
            })
            .collect();

        let mut pulls: [u64; K] = [0; K];
        let mut changed_post_samples = 0u64;

        let mut stopped_at: Option<u64> = None;

        // Helper: choose argmax with stable tie-break (lowest index).
        let argmax = |scores: &[f64; K]| -> usize {
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

        for t in 0..horizon {
            let arm = match policy {
                Policy::RoundRobin => (t as usize) % K,
                Policy::EpsFocus {
                    eps,
                    suspicion,
                    coverage,
                } => {
                    // 1) Coverage stage: if enabled and some arms are under quota, sample them first.
                    let cov_pick = coverage_pick_under_sampled_idx(
                        seed ^ 0xC0DE_D00D ^ t,
                        K,
                        1,
                        coverage,
                        |idx| pulls[idx],
                    );
                    if let Some(&first) = cov_pick.first() {
                        first
                    } else {
                        // 2) Epsilon exploration.
                        let eps = if eps.is_finite() && (0.0..=1.0).contains(&eps) {
                            eps
                        } else {
                            0.0
                        };
                        if rng.random::<f64>() < eps {
                            (rng.random::<u64>() as usize) % K
                        } else {
                            // 3) Exploit: focus on the most suspicious arm using the chosen score.
                            let mut scores = [0.0f64; K];
                            match suspicion {
                                Suspicion::CatKl => {
                                    for k in 0..K {
                                        scores[k] = catkl[k].score().unwrap_or(0.0);
                                    }
                                }
                                Suspicion::Cusum => {
                                    for k in 0..K {
                                        scores[k] = cusum[k].score();
                                    }
                                }
                            }
                            argmax(&scores)
                        }
                    }
                }
            };

            pulls[arm] = pulls[arm].saturating_add(1);

            let p = if arm == changed && t >= nu {
                p1[arm]
            } else {
                p0[arm]
            };
            let x = common::sample_cat(&mut rng, p);

            // Track post-change sampling of the changed arm only.
            if arm == changed && t >= nu {
                changed_post_samples = changed_post_samples.saturating_add(1);
            }

            // Update detectors for the sampled arm.
            let _ = catkl[arm].update(x);
            let alarm = cusum[arm].update(x).is_some();
            if alarm {
                stopped_at = Some(t);
                break;
            }
        }

        let total_pulls: u64 = pulls.iter().sum::<u64>().max(1);
        let frac_on_changed = (pulls[changed] as f64) / (total_pulls as f64);

        let false_alarm = stopped_at.is_some() && stopped_at.unwrap_or(0) < nu;
        let detected = stopped_at.is_some() && stopped_at.unwrap_or(0) >= nu;

        let m = Metrics {
            false_alarm,
            detected,
            // Avoid eager evaluation footgun: `then_some(t - nu)` would underflow for t<nu.
            wall_delay: stopped_at.and_then(|t| if t >= nu { Some(t - nu) } else { None }),
            changed_arm_post_samples: detected.then_some(changed_post_samples),
            frac_on_changed,
        };

        (changed, m)
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

    // Detector config.
    let alpha_smooth = 1e-3;
    let min_n = 30;
    let cusum_threshold = 12.0;
    // Conservative alternative: "something is wrong" pushes mass to hard_junk/fail.
    let cusum_alt_p = common::normalize([0.05, 0.05, 0.45, 0.45]);
    let det = DetectorCfg {
        alpha_smooth,
        min_n,
        cusum_threshold,
        cusum_alt_p,
        tol: 1e-12,
    };

    // Time config.
    let nu = 20_000u64;
    let horizon = 40_000u64;
    let cfg = TrialCfg { nu, horizon, det };

    // Policy sweep.
    let policies = [
        Policy::RoundRobin,
        Policy::EpsFocus {
            eps: 0.02,
            suspicion: Suspicion::CatKl,
            coverage: CoverageConfig::default(),
        },
        Policy::EpsFocus {
            eps: 0.02,
            suspicion: Suspicion::Cusum,
            coverage: CoverageConfig::default(),
        },
        Policy::EpsFocus {
            eps: 0.02,
            suspicion: Suspicion::Cusum,
            coverage: CoverageConfig {
                enabled: true,
                min_fraction: 0.02,
                min_calls_floor: 10,
            },
        },
    ];

    // Print arm severities (Rao) just for intuition.
    let (p0, p1) = bqcd_world::make_world();
    eprintln!("arm severities (Rao distance; larger => more informative shift):");
    for k in 0..K {
        let rao = muxer::monitor::drift_simplex(&p0[k], &p1[k], DriftMetric::Rao, 1e-12)
            .unwrap_or(f64::NAN);
        let kl = logp::kl_divergence(&p1[k], &p0[k], 1e-12).unwrap_or(f64::NAN);
        eprintln!("  arm={k} rao={rao:6.3}  KL(p1||p0)={kl:7.4}");
    }

    let trials = 200u64;
    eprintln!(
        "\nnu={nu} horizon={horizon} alarm=CUSUM(min_n={min_n},thr={cusum_threshold}) alt={:?}",
        cusum_alt_p
    );
    eprintln!("policy                              | fa_rate det_rate  wall(mean/p90)  post(mean/p90)  mean_frac_on_chg");

    for p in policies {
        let mut fa_ok = 0u64;
        let mut det_ok = 0u64;
        let mut walls = Vec::new();
        let mut samps = Vec::new();
        let mut fracs = Vec::new();

        for i in 0..trials {
            let seed = 0xB0C0_u64 ^ ((i + 1) * 0x9E37_79B9) ^ (p.name().len() as u64);
            let (_changed_arm, m) = run_trial_cusum_alarm(seed, p, cfg);
            fa_ok += m.false_alarm as u64;
            det_ok += m.detected as u64;
            walls.push(m.wall_delay);
            samps.push(m.changed_arm_post_samples);
            fracs.push(m.frac_on_changed);
        }

        let fa_rate = (fa_ok as f64) / (trials as f64);
        let det_rate = (det_ok as f64) / (trials as f64);
        let wall = fmt_mean_p90(&walls);
        let post = fmt_mean_p90(&samps);
        let mean_frac = fracs.iter().sum::<f64>() / (fracs.len() as f64).max(1.0);

        eprintln!(
            "{:<35} |  {:5.3}  {:5.3}  {}   {}          {:7.3}",
            p.name(),
            fa_rate,
            det_rate,
            wall,
            post,
            mean_frac
        );
    }
}
