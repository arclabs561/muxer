#[cfg(not(feature = "stochastic"))]
fn main() {
    eprintln!("This example requires: cargo run --example detector_inertia --features stochastic");
}

#[cfg(feature = "stochastic")]
mod common;

#[cfg(feature = "stochastic")]
fn main() {
    use muxer::monitor::{drift_simplex, CatKlDetector, CusumCatDetector, DriftMetric};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    // Controlled experiment:
    // - One categorical stream with a change at time `t_change`.
    // - The arm is *only observed* when “sampled”, i.e. at a fixed interval.
    // - We compare CatKL (cumulative) vs CUSUM (forgetful) detectors.
    //
    // This isolates two effects:
    // - Sampling *rate* (interval) sets the wall-clock delay.
    // - CatKL “inertia” grows with pre-change history length; CUSUM is much less sensitive.

    #[derive(Debug, Clone, Copy)]
    struct Delays {
        // Delay measured in wall-clock steps.
        wall_catkl: Option<u64>,
        wall_cusum: Option<u64>,
        // Delay measured in number of post-change *samples* (i.e. observations).
        samp_catkl: Option<u64>,
        samp_cusum: Option<u64>,
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
        det: DetectorCfg,
    }

    fn run_once(seed: u64, cfg: SimCfg) -> Delays {
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
        let mut rng = StdRng::seed_from_u64(seed);

        let mut catkl =
            CatKlDetector::new(&p0, alpha_smooth, min_n_catkl, thr_catkl, tol).expect("catkl");
        let mut cusum = CusumCatDetector::new(&p0, &p1, alpha_smooth, min_n_cusum, thr_cusum, tol)
            .expect("cusum");

        let mut wall_catkl: Option<u64> = None;
        let mut wall_cusum: Option<u64> = None;
        let mut post_samples: u64 = 0;
        let mut samp_catkl: Option<u64> = None;
        let mut samp_cusum: Option<u64> = None;

        for t in 0..horizon {
            // Only observe on the sampling schedule.
            if interval == 0 || (t % interval != 0) {
                continue;
            }
            let p = if t < t_change { p0 } else { p1 };
            let x = common::sample_cat(&mut rng, p);
            if t >= t_change {
                post_samples = post_samples.saturating_add(1);
            }

            if wall_catkl.is_none() {
                if catkl.update(x).is_some() && t >= t_change {
                    wall_catkl = Some(t - t_change);
                    samp_catkl = Some(post_samples);
                }
            } else {
                let _ = catkl.update(x);
            }

            if wall_cusum.is_none() {
                if cusum.update(x).is_some() && t >= t_change {
                    wall_cusum = Some(t - t_change);
                    samp_cusum = Some(post_samples);
                }
            } else {
                let _ = cusum.update(x);
            }
        }

        Delays {
            wall_catkl,
            wall_cusum,
            samp_catkl,
            samp_cusum,
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

    // Baseline and shift, expressed on muxer’s 4 outcome categories.
    let p0 = common::normalize([0.90, 0.03, 0.02, 0.05]);
    let p1 = common::normalize([0.20, 0.10, 0.20, 0.50]);

    let tol = 1e-12;
    let rao = drift_simplex(&p0, &p1, DriftMetric::Rao, tol).unwrap_or(f64::NAN);
    let hel = drift_simplex(&p0, &p1, DriftMetric::Hellinger, tol).unwrap_or(f64::NAN);
    eprintln!("shift severity: rao={rao:.4} rad, hellinger={hel:.4}");

    // Detector knobs (not “optimal”; chosen to make failures visible).
    let alpha = 1e-3;
    let min_n_catkl = 30;
    let thr_catkl = 8.0;
    let min_n_cusum = 20;
    let thr_cusum = 4.0;
    let det = DetectorCfg {
        alpha_smooth: alpha,
        min_n_catkl,
        thr_catkl,
        min_n_cusum,
        thr_cusum,
        tol,
    };

    // Sweep 1: sampling interval vs wall-clock delay (fixed pre-change length).
    let horizon = 40_000u64;
    let t_change = 20_000u64;
    let trials = 200u64;

    eprintln!("\n== sweep: sampling interval (fixed pre-history) ==");
    eprintln!("interval | catkl_wall  cusum_wall | catkl_samp  cusum_samp");
    for interval in [1u64, 2, 5, 10, 20, 50, 100, 200] {
        let mut wc = Vec::new();
        let mut wu = Vec::new();
        let mut sc = Vec::new();
        let mut su = Vec::new();
        for i in 0..trials {
            let seed = 0xD371_u64 ^ ((i + 1) * 0x9E37_79B9) ^ interval;
            let d = run_once(
                seed,
                SimCfg {
                    horizon,
                    t_change,
                    interval,
                    p0,
                    p1,
                    det,
                },
            );
            wc.push(d.wall_catkl);
            wu.push(d.wall_cusum);
            sc.push(d.samp_catkl);
            su.push(d.samp_cusum);
        }
        eprintln!(
            "{:>8} | {}  {} | {}  {}",
            interval,
            fmt(mean_opt(&wc)),
            fmt(mean_opt(&wu)),
            fmt(mean_opt(&sc)),
            fmt(mean_opt(&su))
        );
    }

    // Sweep 2: pre-change history length vs *sample* delay, holding sampling interval fixed.
    //
    // This is the “inertia” sweep: CatKL should generally require more post-change samples
    // as pre-change history grows, because the empirical q remains dominated by pre-change counts.
    let interval = 20u64; // ~5% sampling rate
    eprintln!("\n== sweep: pre-change history length (interval={interval}) ==");
    eprintln!("pre_steps | catkl_samp  cusum_samp | catkl_wall  cusum_wall");
    for pre in [200u64, 1_000, 5_000, 20_000, 80_000] {
        let horizon = pre.saturating_mul(2);
        let t_change = pre;
        let mut wc = Vec::new();
        let mut wu = Vec::new();
        let mut sc = Vec::new();
        let mut su = Vec::new();
        for i in 0..trials {
            let seed = 0xB17E_u64 ^ ((i + 1) * 0x9E37_79B9) ^ pre;
            let d = run_once(
                seed,
                SimCfg {
                    horizon,
                    t_change,
                    interval,
                    p0,
                    p1,
                    det,
                },
            );
            wc.push(d.wall_catkl);
            wu.push(d.wall_cusum);
            sc.push(d.samp_catkl);
            su.push(d.samp_cusum);
        }
        eprintln!(
            "{:>8} | {}  {} | {}  {}",
            pre,
            fmt(mean_opt(&sc)),
            fmt(mean_opt(&su)),
            fmt(mean_opt(&wc)),
            fmt(mean_opt(&wu))
        );
    }
}
