#[cfg(not(feature = "stochastic"))]
fn main() {
    eprintln!(
        "This example requires: cargo run --example detector_calibration --features stochastic"
    );
}

#[cfg(feature = "stochastic")]
mod common;

#[cfg(feature = "stochastic")]
fn main() {
    use muxer::monitor::{
        calibrate_threshold_from_max_scores, drift_simplex, CatKlDetector, CusumCatDetector,
        DriftMetric, ThresholdCalibration,
    };
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    // Goal:
    // - Fair-ish comparison of CatKL vs CUSUM requires matching false-alarm behavior.
    // - We calibrate thresholds under the *null* (no change) using a BQCD-style constraint:
    //     P_infty[ tau < m ] <= alpha_fa
    //   i.e. the detector must "survive" at least `m` wall-clock steps with high probability.
    // - Then we compare detection delay under a fixed shift at nu=m (same p0->p1 as detector_inertia).
    //
    // This is deliberately Monte-Carlo, not asymptotic theory: it gives us actionable knobs.

    #[derive(Debug, Clone, Copy, Default)]
    struct SimOut {
        alarmed: bool,
        // Delay in wall-clock steps after the change time (only meaningful when alarmed=true in change sims).
        wall_delay: Option<u64>,
        // Delay in post-change *samples* (observations).
        samp_delay: Option<u64>,
    }

    #[derive(Debug, Clone, Copy)]
    struct DetectorCfg {
        alpha_smooth: f64,
        min_n_catkl: u64,
        thr_catkl: f64,
        min_n_cusum: u64,
        thr_cusum: f64,
        tol: f64,
    }

    #[derive(Debug, Clone, Copy)]
    struct SimCfg {
        horizon: u64,
        t_change: u64,
        interval: u64,
        p0: [f64; 4],
        p1: [f64; 4],
        // If true: null scenario where p1 is ignored and stream stays at p0.
        is_null: bool,
        det: DetectorCfg,
    }

    fn simulate(seed: u64, cfg: SimCfg) -> (SimOut, SimOut) {
        let DetectorCfg {
            alpha_smooth,
            min_n_catkl,
            thr_catkl,
            min_n_cusum,
            thr_cusum,
            tol,
        } = cfg.det;
        let horizon = cfg.horizon;
        let t_change = cfg.t_change;
        let interval = cfg.interval;
        let p0 = cfg.p0;
        let p1 = cfg.p1;
        let is_null = cfg.is_null;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut catkl = CatKlDetector::new(&p0, alpha_smooth, min_n_catkl, thr_catkl, tol).unwrap();
        let mut cusum =
            CusumCatDetector::new(&p0, &p1, alpha_smooth, min_n_cusum, thr_cusum, tol).unwrap();

        let mut out_cat = SimOut::default();
        let mut out_cus = SimOut::default();

        let mut post_samples: u64 = 0;

        // Event-driven simulation: we only "observe" at times t = i * interval.
        // This keeps runtime proportional to number of observations, not wall-clock horizon.
        let interval = interval.max(1);
        let max_obs = ((horizon.saturating_sub(1)) / interval).saturating_add(1);

        for i in 0..max_obs {
            let t = i.saturating_mul(interval);
            let p = if is_null || t < t_change { p0 } else { p1 };
            let x = common::sample_cat(&mut rng, p);
            if t >= t_change {
                post_samples = post_samples.saturating_add(1);
            }

            if !out_cat.alarmed {
                if catkl.update(x).is_some() {
                    out_cat.alarmed = true;
                    if t >= t_change {
                        out_cat.wall_delay = Some(t - t_change);
                        out_cat.samp_delay = Some(post_samples);
                    }
                }
            } else {
                let _ = catkl.update(x);
            }

            if !out_cus.alarmed {
                if cusum.update(x).is_some() {
                    out_cus.alarmed = true;
                    if t >= t_change {
                        out_cus.wall_delay = Some(t - t_change);
                        out_cus.samp_delay = Some(post_samples);
                    }
                }
            } else {
                let _ = cusum.update(x);
            }

            if out_cat.alarmed && out_cus.alarmed {
                break;
            }
        }

        (out_cat, out_cus)
    }

    #[derive(Debug, Clone, Copy)]
    struct Calibrated {
        catkl: ThresholdCalibration,
        cusum: ThresholdCalibration,
    }

    // Calibration approach (threshold-free):
    //
    // Under the null we simulate once per trial with threshold=+∞ and record the per-trial max score:
    //     M = max_{t < m} S(t)
    // (only counting samples after `min_n` for each detector).
    //
    // Then for any threshold h:
    //     1{tau < m} == 1{M >= h}.
    //
    // This avoids re-simulating for every grid point and enforces monotonicity.
    #[derive(Debug, Clone, Copy)]
    struct CalCfg<'a> {
        // BQCD-style false-alarm constraint: must not stop before m (wall-clock).
        m: u64,
        alpha_fa: f64,
        interval: u64,
        p0: [f64; 4],
        // Used as the CUSUM alternative (the stream stays at p0 during null calibration).
        p1: [f64; 4],
        // Detector params (thresholds should be `+∞` for max-score calibration).
        det: DetectorCfg,
        // Search grids (increasing => fewer alarms).
        catkl_grid: &'a [f64],
        cusum_grid: &'a [f64],
        // Trials.
        trials: u64,
        seed0: u64,
        // Calibration conservatism: require Wilson upper bound <= alpha_fa.
        z: f64,
        require_wilson: bool,
    }

    fn null_max_scores(seed: u64, cfg: CalCfg<'_>) -> (f64, f64) {
        let DetectorCfg {
            alpha_smooth,
            min_n_catkl,
            min_n_cusum,
            tol,
            ..
        } = cfg.det;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut catkl =
            CatKlDetector::new(&cfg.p0, alpha_smooth, min_n_catkl, f64::INFINITY, tol).unwrap();
        let mut cusum = CusumCatDetector::new(
            &cfg.p0,
            &cfg.p1,
            alpha_smooth,
            min_n_cusum,
            f64::INFINITY,
            tol,
        )
        .unwrap();

        let interval = cfg.interval.max(1);
        let max_obs = ((cfg.m.saturating_sub(1)) / interval).saturating_add(1);

        let mut max_cat = 0.0f64;
        let mut max_cus = 0.0f64;

        for _i in 0..max_obs {
            let x = common::sample_cat(&mut rng, cfg.p0);

            let _ = catkl.update(x);
            let _ = cusum.update(x);

            if let Some(s) = catkl.score() {
                if s.is_finite() && s > max_cat {
                    max_cat = s;
                }
            }
            if cusum.n() >= min_n_cusum {
                let s = cusum.score();
                if s.is_finite() && s > max_cus {
                    max_cus = s;
                }
            }
        }

        (max_cat, max_cus)
    }

    fn calibrate_thresholds_bqcd(cfg: CalCfg<'_>) -> Calibrated {
        let trials = cfg.trials.max(1);
        let mut max_cat: Vec<f64> = Vec::with_capacity(trials as usize);
        let mut max_cus: Vec<f64> = Vec::with_capacity(trials as usize);
        for i in 0..trials {
            let seed = cfg.seed0 ^ ((i + 1) * 0x9E37_79B9);
            let (mc, mu) = null_max_scores(seed, cfg);
            max_cat.push(mc);
            max_cus.push(mu);
        }

        let cat_cal = calibrate_threshold_from_max_scores(
            &mut max_cat,
            cfg.catkl_grid,
            cfg.alpha_fa,
            cfg.z,
            cfg.require_wilson,
        );
        let cus_cal = calibrate_threshold_from_max_scores(
            &mut max_cus,
            cfg.cusum_grid,
            cfg.alpha_fa,
            cfg.z,
            cfg.require_wilson,
        );

        Calibrated {
            catkl: cat_cal,
            cusum: cus_cal,
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

    fn fmt(x: Option<f64>) -> String {
        match x {
            Some(v) => format!("{v:7.1}"),
            None => "  never".to_string(),
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

    // Same p0/p1 as detector_inertia.
    let p0 = common::normalize([0.90, 0.03, 0.02, 0.05]);
    let p1 = common::normalize([0.20, 0.10, 0.20, 0.50]);
    let tol = 1e-12;
    let rao = drift_simplex(&p0, &p1, DriftMetric::Rao, tol).unwrap_or(f64::NAN);
    let hel = drift_simplex(&p0, &p1, DriftMetric::Hellinger, tol).unwrap_or(f64::NAN);
    eprintln!("shift severity: rao={rao:.4} rad, hellinger={hel:.4}");

    // Detector knobs (min_n fixed); we calibrate thresholds.
    let alpha_smooth = 1e-3;
    let min_n_catkl = 30;
    let min_n_cusum = 20;
    let cal_z = env_f64("CAL_Z", 1.96);
    let require_wilson = env_u64("CAL_REQUIRE_WILSON", 0) != 0;

    // Threshold grids (increasing => fewer alarms).
    let catkl_grid: Vec<f64> = (0..120).map(|i| 2.0 + (i as f64) * 0.5).collect(); // up to 61.5
    let cusum_grid: Vec<f64> = (0..240).map(|i| 0.5 + (i as f64) * 0.5).collect(); // up to 120.0

    // BQCD-style calibration parameters.
    let alpha_fa = 0.01;
    let m = 20_000u64; // false-alarm "must survive" time
    let post_horizon = 20_000u64; // how long after change we watch
    let horizon = m + post_horizon;

    let cal_trials = env_u64("CAL_TRIALS", 250).max(1);
    let eval_trials = env_u64("EVAL_TRIALS", 400).max(1);

    eprintln!(
        "\n== BQCD-style calibration: P_infty[tau < m] <= alpha = {alpha_fa} (m={m}) ==\n(cal_trials={cal_trials}, eval_trials={eval_trials}, cal_z={cal_z}, require_wilson={require_wilson})"
    );
    eprintln!("interval | catkl_thr fa hi | cusum_thr fa hi | delay_catkl_samp delay_cusum_samp");

    for interval in [1u64, 2, 5, 10, 20, 50, 100, 200] {
        let det_inf = DetectorCfg {
            alpha_smooth,
            min_n_catkl,
            thr_catkl: f64::INFINITY,
            min_n_cusum,
            thr_cusum: f64::INFINITY,
            tol,
        };
        let cal = calibrate_thresholds_bqcd(CalCfg {
            m,
            alpha_fa,
            interval,
            p0,
            p1,
            det: det_inf,
            catkl_grid: &catkl_grid,
            cusum_grid: &cusum_grid,
            trials: cal_trials,
            seed0: 0xCA11_8A7E ^ interval,
            z: cal_z,
            require_wilson,
        });

        // Evaluate under shift at nu=m.
        let det = DetectorCfg {
            alpha_smooth,
            min_n_catkl,
            thr_catkl: cal.catkl.threshold,
            min_n_cusum,
            thr_cusum: cal.cusum.threshold,
            tol,
        };
        let mut cat_wall = Vec::new();
        let mut cus_wall = Vec::new();
        let mut cat_samp = Vec::new();
        let mut cus_samp = Vec::new();
        let mut cat_alarm_before_m = 0u64;
        let mut cus_alarm_before_m = 0u64;

        for i in 0..eval_trials {
            let seed = 0x00DE_7E57_u64 ^ ((i + 1) * 0x9E37_79B9) ^ interval;
            let (c, u) = simulate(
                seed,
                SimCfg {
                    horizon,
                    t_change: m,
                    interval,
                    p0,
                    p1,
                    is_null: false,
                    det,
                },
            );
            // A detector that alarms but has no post-change delay is (by our bookkeeping)
            // an early alarm before nu.
            if c.alarmed && c.wall_delay.is_none() {
                cat_alarm_before_m += 1;
            }
            if u.alarmed && u.wall_delay.is_none() {
                cus_alarm_before_m += 1;
            }
            cat_wall.push(c.wall_delay);
            cus_wall.push(u.wall_delay);
            cat_samp.push(c.samp_delay);
            cus_samp.push(u.samp_delay);
        }

        let early_cat = (cat_alarm_before_m as f64) / (eval_trials as f64);
        let early_cus = (cus_alarm_before_m as f64) / (eval_trials as f64);
        if early_cat > 0.0 || early_cus > 0.0 {
            eprintln!(
                "  note: interval={interval} early_alarm_under_shift: catkl={early_cat:.4} cusum={early_cus:.4}"
            );
        }
        if !cal.catkl.grid_satisfied || !cal.cusum.grid_satisfied {
            eprintln!(
                "  note: interval={interval} grid_not_satisfied: catkl={} cusum={}",
                cal.catkl.grid_satisfied, cal.cusum.grid_satisfied
            );
        }

        eprintln!(
            "{:>8} | {:>8.3} {:>7.4} {:>7.4} | {:>8.3} {:>7.4} {:>7.4} | {:>14} {:>14}",
            interval,
            cal.catkl.threshold,
            cal.catkl.fa_hat,
            cal.catkl.fa_wilson_hi,
            cal.cusum.threshold,
            cal.cusum.fa_hat,
            cal.cusum.fa_wilson_hi,
            fmt(mean_opt(&cat_samp)),
            fmt(mean_opt(&cus_samp)),
        );
    }
}
