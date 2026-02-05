#[cfg(not(feature = "stochastic"))]
fn main() {
    eprintln!("This example requires: cargo run --example bqcd_calibrated --features stochastic");
}

#[cfg(feature = "stochastic")]
mod common;

#[cfg(feature = "stochastic")]
mod bqcd_world;

#[cfg(feature = "stochastic")]
fn main() {
    use muxer::monitor::{
        calibrate_threshold_from_max_scores, CatKlDetector, CusumCatBank, DriftMetric,
        ThresholdCalibration,
    };
    use muxer::{coverage_pick_under_sampled_idx, CoverageConfig};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::time::Instant;

    // ============================================================
    // BQCD-style calibrated experiment (multi-arm + unknown post-change)
    //
    // Two additions beyond `bqcd_sampling.rs`:
    //
    // (A) Multi-arm false-alarm calibration (BQCD-style):
    //     For a given policy π, pick a threshold h so that under the null:
    //         P_infty^π[ tau < m ] <= alpha
    //     where tau is the first time ANY arm alarms.
    //
    //     This is a *global* (across arms) constraint, not per-arm.
    //
    // (B) Unknown post-change robustification:
    //     Instead of one fixed alternative p1, use a small bank of alternatives {p_alt^j}
    //     and maintain a CUSUM per (arm, alt). Alarm if max_j CUSUM_j crosses threshold.
    //
    // This is a “GLR-lite” / multiple-hypothesis surrogate: cheap, inspectable, and often robust.
    //
    // Practical notes:
    // - Prefer `--release`; this is Monte Carlo heavy.
    // - You can override Monte Carlo counts via env vars: `CAL_TRIALS`, `EVAL_TRIALS`.
    // - Calibration reports a Wilson upper bound `hi` (set `CAL_Z`, default `1.96`).
    //   If you want calibration to *require* `hi <= alpha`, set `CAL_REQUIRE_WILSON=1`
    //   (you’ll usually need much larger `CAL_TRIALS`).
    // ============================================================

    const K: usize = 6;
    const CATS: usize = 4;

    #[derive(Debug, Clone, Copy)]
    enum Suspicion {
        // Cumulative drift score: inertia-prone.
        CatKl,
        // Forgetful suspicion: current max CUSUM score across alt bank.
        CusumMax,
    }

    impl Suspicion {
        fn name(&self) -> &'static str {
            match *self {
                Suspicion::CatKl => "catkl",
                Suspicion::CusumMax => "cusum_max",
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
                    format!("eps_focus(eps={eps:.3},{},{})", suspicion.name(), cov)
                }
            }
        }
    }

    #[derive(Debug, Clone, Copy)]
    enum AltBank {
        SingleConservative,
        Trio,
        Quad,
    }

    impl AltBank {
        fn name(&self) -> &'static str {
            match *self {
                AltBank::SingleConservative => "alt=single_conservative",
                AltBank::Trio => "alt=trio",
                AltBank::Quad => "alt=quad",
            }
        }

        fn alts(&self) -> Vec<[f64; CATS]> {
            match *self {
                AltBank::SingleConservative => vec![common::normalize([0.05, 0.05, 0.45, 0.45])],
                AltBank::Trio => vec![
                    // hard break / fail heavy
                    common::normalize([0.05, 0.05, 0.45, 0.45]),
                    // mostly fail
                    common::normalize([0.10, 0.05, 0.05, 0.80]),
                    // mostly hard junk (ok but hard-junky)
                    common::normalize([0.10, 0.05, 0.75, 0.10]),
                ],
                AltBank::Quad => vec![
                    // hard break / fail heavy
                    common::normalize([0.05, 0.05, 0.45, 0.45]),
                    // mostly fail
                    common::normalize([0.10, 0.05, 0.05, 0.80]),
                    // mostly hard junk (ok but hard-junky)
                    common::normalize([0.10, 0.05, 0.75, 0.10]),
                    // mostly soft junk (ok but unusable-ish)
                    common::normalize([0.10, 0.75, 0.05, 0.10]),
                ],
            }
        }
    }

    #[derive(Debug, Clone, Copy, Default)]
    struct Metrics {
        false_alarm: bool,
        detected: bool,
        wall_delay: Option<u64>,
        changed_arm_post_samples: Option<u64>,
        frac_on_changed: f64,
        // Maximum CUSUM score observed while the arm was eligible to alarm (n >= min_n),
        // maximized over arms and alt-bank for the whole run.
        //
        // This is useful for calibration because the CUSUM recursion is threshold-free:
        // an alarm occurs for threshold h iff `max_alarm_score >= h`.
        max_alarm_score: f64,
    }

    #[derive(Debug, Clone, Copy)]
    struct DetectorCfg {
        alpha_smooth: f64,
        min_n: u64,
        tol: f64,
    }

    #[derive(Debug, Clone, Copy)]
    struct RunCfg {
        // time config
        nu: u64,
        horizon: u64,
        // detector config
        det: DetectorCfg,
        threshold: f64,
        // if true: null run (no change)
        is_null: bool,
    }

    #[derive(Debug, Clone, Copy)]
    struct CalCfg<'a> {
        // BQCD-style: survive to m under null
        m: u64,
        alpha: f64,
        // threshold grid (ascending)
        grid: &'a [f64],
        trials: u64,
        // calibration conservatism: require Wilson upper bound <= alpha
        z: f64,
        require_wilson: bool,
        // detector config (used in the null simulation to generate max scores)
        det: DetectorCfg,
    }

    fn argmax(scores: &[f64; K]) -> usize {
        let mut best = 0usize;
        let mut best_s = f64::NEG_INFINITY;
        for (i, &s) in scores.iter().enumerate() {
            if s > best_s || ((s - best_s).abs() <= 1e-12 && i < best) {
                best_s = s;
                best = i;
            }
        }
        best
    }

    fn run_once(seed: u64, policy: Policy, bank: AltBank, cfg: RunCfg) -> Metrics {
        let nu = cfg.nu;
        let horizon = cfg.horizon;
        let alpha_smooth = cfg.det.alpha_smooth;
        let min_n = cfg.det.min_n;
        let threshold = cfg.threshold;
        let tol = cfg.det.tol;
        let is_null = cfg.is_null;

        let (p0, p1) = bqcd_world::make_world();
        let mut rng = StdRng::seed_from_u64(seed);

        let changed = (rng.random::<u64>() as usize) % K;
        let alts_vec: Vec<Vec<f64>> = bank.alts().into_iter().map(|a| a.to_vec()).collect();

        let mut pulls: [u64; K] = [0; K];
        let mut changed_post_samples = 0u64;
        let mut stopped_at: Option<u64> = None;
        let mut max_alarm_score = 0.0f64;

        let mut catkl: [CatKlDetector; K] = core::array::from_fn(|k| {
            // We keep CatKL as a suspicion score only (never alarms here).
            CatKlDetector::new(&p0[k], alpha_smooth, min_n, f64::INFINITY, tol).unwrap()
        });

        // For each arm, keep a CUSUM bank (one per alt).
        let mut cusum: [CusumCatBank; K] = core::array::from_fn(|k| {
            CusumCatBank::new(&p0[k], &alts_vec, alpha_smooth, min_n, threshold, tol).unwrap()
        });

        // Cache suspicion scores to avoid recomputing them for every arm on every step.
        // (They only change when an arm is pulled.)
        let track_catkl_scores = matches!(
            policy,
            Policy::EpsFocus {
                suspicion: Suspicion::CatKl,
                ..
            }
        );
        let mut catkl_scores: [f64; K] = [0.0; K];
        let mut cusum_scores: [f64; K] = [0.0; K];

        for t in 0..horizon {
            let arm = match policy {
                Policy::RoundRobin => (t as usize) % K,
                Policy::EpsFocus {
                    eps,
                    suspicion,
                    coverage,
                } => {
                    // Coverage stage.
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
                        let eps = if eps.is_finite() && (0.0..=1.0).contains(&eps) {
                            eps
                        } else {
                            0.0
                        };
                        if rng.random::<f64>() < eps {
                            (rng.random::<u64>() as usize) % K
                        } else {
                            match suspicion {
                                Suspicion::CatKl => argmax(&catkl_scores),
                                Suspicion::CusumMax => argmax(&cusum_scores),
                            }
                        }
                    }
                }
            };

            pulls[arm] = pulls[arm].saturating_add(1);

            let p = if is_null {
                p0[arm]
            } else if arm == changed && t >= nu {
                p1[arm]
            } else {
                p0[arm]
            };
            let x = common::sample_cat(&mut rng, p);

            if !is_null && arm == changed && t >= nu {
                changed_post_samples = changed_post_samples.saturating_add(1);
            }

            let _ = catkl[arm].update(x);
            if track_catkl_scores {
                catkl_scores[arm] = catkl[arm].score().unwrap_or(0.0);
            }

            let upd = cusum[arm].update(x);
            cusum_scores[arm] = upd.score_max;
            if upd.n >= min_n && upd.score_max.is_finite() && upd.score_max > max_alarm_score {
                max_alarm_score = upd.score_max;
            }

            if upd.alarmed {
                stopped_at = Some(t);
                break;
            }
        }

        let total_pulls: u64 = pulls.iter().sum::<u64>().max(1);
        let frac_on_changed = (pulls[changed] as f64) / (total_pulls as f64);

        let false_alarm = stopped_at.is_some() && stopped_at.unwrap_or(0) < nu;
        let detected = stopped_at.is_some() && stopped_at.unwrap_or(0) >= nu;

        Metrics {
            false_alarm,
            detected,
            wall_delay: stopped_at.and_then(|t| if t >= nu { Some(t - nu) } else { None }),
            changed_arm_post_samples: detected.then_some(changed_post_samples),
            frac_on_changed,
            max_alarm_score,
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

    fn env_u64(name: &str, default: u64) -> u64 {
        std::env::var(name)
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(default)
    }

    fn env_f64(name: &str, default: f64) -> f64 {
        std::env::var(name)
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(default)
    }

    fn calibrate_threshold_global(
        seed0: u64,
        policy: Policy,
        bank: AltBank,
        cfg: CalCfg<'_>,
    ) -> ThresholdCalibration {
        // Calibrate using a threshold-free summary:
        //
        // Under the null, for each Monte Carlo trial we simulate the policy for `m` wall steps
        // with `threshold=∞` (so we never early-stop), and record:
        //     M = max_{t < m} max_{arm,alt} S_{arm,alt}(t)   (only counting times when n>=min_n).
        //
        // Then for any threshold h:
        //     1{tau < m} == 1{ M >= h }.
        //
        // This avoids re-simulating for every candidate threshold and enforces monotonicity
        // (common random numbers across all grid points).
        let trials = cfg.trials.max(1);
        let mut maxes: Vec<f64> = Vec::with_capacity(trials as usize);
        for i in 0..trials {
            let seed = seed0 ^ ((i + 1) * 0x9E37_79B9);
            let m0 = run_once(
                seed,
                policy,
                bank,
                RunCfg {
                    nu: cfg.m,
                    horizon: cfg.m,
                    det: cfg.det,
                    threshold: f64::INFINITY,
                    is_null: true,
                },
            );
            maxes.push(m0.max_alarm_score);
        }
        calibrate_threshold_from_max_scores(
            &mut maxes,
            cfg.grid,
            cfg.alpha,
            cfg.z,
            cfg.require_wilson,
        )
    }

    // ------------------------------------------------------------
    // Experiment config
    // ------------------------------------------------------------

    let tol = 1e-12;
    let (p0, p1) = bqcd_world::make_world();
    eprintln!("arm severities (Rao + KL(p1||p0)):");
    for k in 0..K {
        let rao = muxer::monitor::drift_simplex(&p0[k], &p1[k], DriftMetric::Rao, tol)
            .unwrap_or(f64::NAN);
        let kl = logp::kl_divergence(&p1[k], &p0[k], tol).unwrap_or(f64::NAN);
        eprintln!("  arm={k} rao={rao:6.3}  KL(p1||p0)={kl:7.4}");
    }

    // BQCD-style parameters.
    let m = 20_000u64;
    let nu = m;
    let horizon = 40_000u64;
    let alpha = 0.01;

    // Detector parameters.
    let alpha_smooth = 1e-3;
    let min_n = 30;
    let cal_z = env_f64("CAL_Z", 1.96);
    let require_wilson = env_u64("CAL_REQUIRE_WILSON", 0) != 0;
    let det_cfg = DetectorCfg {
        alpha_smooth,
        min_n,
        tol,
    };

    // Threshold grid (increasing => fewer false alarms). Wide enough that we can satisfy alpha in most cases.
    let thr_grid: Vec<f64> = (0..240).map(|i| 0.5 + (i as f64) * 0.5).collect(); // 0.5..120.0

    // Policies and alt banks to evaluate.
    let policies = [
        Policy::RoundRobin,
        Policy::EpsFocus {
            eps: 0.02,
            suspicion: Suspicion::CatKl,
            coverage: CoverageConfig::default(),
        },
        Policy::EpsFocus {
            eps: 0.02,
            suspicion: Suspicion::CatKl,
            coverage: CoverageConfig {
                enabled: true,
                min_fraction: 0.02,
                min_calls_floor: 10,
            },
        },
        Policy::EpsFocus {
            eps: 0.02,
            suspicion: Suspicion::CusumMax,
            coverage: CoverageConfig::default(),
        },
        Policy::EpsFocus {
            eps: 0.02,
            suspicion: Suspicion::CusumMax,
            coverage: CoverageConfig {
                enabled: true,
                min_fraction: 0.02,
                min_calls_floor: 10,
            },
        },
    ];

    let banks = [AltBank::SingleConservative, AltBank::Trio, AltBank::Quad];

    // Monte Carlo counts (keep moderate; this is an example, not a benchmark suite).
    let cal_trials = env_u64("CAL_TRIALS", 200).max(1);
    let eval_trials = env_u64("EVAL_TRIALS", 250).max(1);

    eprintln!(
        "\n== calibrated BQCD run: P_infty[tau < m] <= alpha (m={m}, alpha={alpha}) ==\n(cal_trials={cal_trials}, eval_trials={eval_trials}, cal_z={cal_z}, require_wilson={require_wilson})\npolicy | bank | thr fa hi ok | det_rate mean_wall mean_post_samp(chg) mean_frac_on_chg"
    );
    eprintln!("(note: wall/post columns are mean/p90 over detected trials)");

    for bank in banks {
        for policy in policies {
            // Calibrate threshold under null for this (policy,bank).
            let cal_started = Instant::now();
            let cal = calibrate_threshold_global(
                0xCA11_BA7E_u64 ^ (bank as u64) ^ (policy.name().len() as u64),
                policy,
                bank,
                CalCfg {
                    m,
                    alpha,
                    det: det_cfg,
                    grid: &thr_grid,
                    trials: cal_trials,
                    z: cal_z,
                    require_wilson,
                },
            );
            let thr = cal.threshold;
            let fa = cal.fa_hat;
            let fa_hi = cal.fa_wilson_hi;
            let cal_elapsed = cal_started.elapsed();

            // Evaluate under shift at nu=m with the calibrated threshold.
            let eval_started = Instant::now();
            let mut fa_shift = 0u64;
            let mut det_ok = 0u64;
            let mut walls = Vec::new();
            let mut samps = Vec::new();
            let mut fracs = Vec::new();
            for i in 0..eval_trials {
                let seed = 0xE7A1_BA7E_u64
                    ^ ((i + 1) * 0x9E37_79B9)
                    ^ (bank as u64)
                    ^ (policy.name().len() as u64);
                let r = run_once(
                    seed,
                    policy,
                    bank,
                    RunCfg {
                        nu,
                        horizon,
                        det: det_cfg,
                        threshold: thr,
                        is_null: false,
                    },
                );
                fa_shift += r.false_alarm as u64;
                det_ok += r.detected as u64;
                walls.push(r.wall_delay);
                samps.push(r.changed_arm_post_samples);
                fracs.push(r.frac_on_changed);
            }

            let fa_shift_rate = (fa_shift as f64) / (eval_trials as f64);
            let det_rate = (det_ok as f64) / (eval_trials as f64);
            let wall = fmt_mean_p90(&walls);
            let post = fmt_mean_p90(&samps);
            let mean_frac = fracs.iter().sum::<f64>() / (fracs.len() as f64).max(1.0);
            let eval_elapsed = eval_started.elapsed();
            let total_elapsed = cal_elapsed + eval_elapsed;

            eprintln!(
                "{} | {} | thr={:>6.2} fa≈{:>6.4} hi≈{:>6.4} ok={} (shift_fa≈{:>6.4}) | det={:>5.3} wall={} post={} frac={:>6.3}",
                policy.name(),
                bank.name(),
                thr,
                fa,
                fa_hi,
                cal.grid_satisfied,
                fa_shift_rate,
                det_rate,
                wall,
                post,
                mean_frac,
            );
            eprintln!(
                "  (elapsed: cal={:.2?}, eval={:.2?}, total={:.2?})",
                cal_elapsed, eval_elapsed, total_elapsed
            );
        }
    }
}
