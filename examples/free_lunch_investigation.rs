#[cfg(not(feature = "stochastic"))]
fn main() {
    eprintln!(
        "This example requires: cargo run --example free_lunch_investigation --features stochastic"
    );
}

#[cfg(feature = "stochastic")]
fn main() {
    use muxer::monitor::{drift_simplex, CatKlDetector, CusumCatDetector, DriftMetric};
    use muxer::{
        policy_fill_k_observed_with_coverage, select_mab_decide, CoverageConfig,
        LatencyGuardrailConfig, MabConfig, Outcome, ThompsonConfig, ThompsonSampling, Window,
    };
    use pare::{Direction, ParetoFrontier};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::collections::BTreeMap;

    // ============================================================
    // What this example is trying to show (in code, not prose):
    //
    // "Free lunch" for change detection depends on sampling frequency.
    // If the arm that changes is sampled anyway (small gap / high uncertainty),
    // detection is quick "for free". If it's starved, detection delay explodes,
    // unless you enforce explicit coverage/maintenance sampling.
    // ============================================================

    #[derive(Debug, Clone, Copy)]
    struct Cat4 {
        // [ok_clean, ok_soft_junk, ok_hard_junk, fail]
        p: [f64; 4],
    }

    fn normalize_cat(mut p: [f64; 4]) -> [f64; 4] {
        for x in &mut p {
            if !x.is_finite() || *x < 0.0 {
                *x = 0.0;
            }
        }
        let s: f64 = p.iter().sum();
        if s <= 0.0 {
            return [0.25, 0.25, 0.25, 0.25];
        }
        for x in &mut p {
            *x /= s;
        }
        p
    }

    fn categorical_sample(rng: &mut StdRng, p: [f64; 4]) -> usize {
        let r: f64 = rng.random();
        let mut cdf = 0.0;
        for (i, &pi) in p.iter().enumerate() {
            cdf += pi;
            if r < cdf {
                return i;
            }
        }
        3
    }

    fn outcome_from_cat(idx: usize) -> Outcome {
        match idx {
            0 => Outcome {
                ok: true,
                junk: false,
                hard_junk: false,
                cost_units: 0,
                elapsed_ms: 0,
                quality_score: None,
            },
            1 => Outcome {
                ok: true,
                junk: true,
                hard_junk: false,
                cost_units: 0,
                elapsed_ms: 0,
                quality_score: None,
            },
            2 => Outcome {
                ok: true,
                junk: true,
                hard_junk: true,
                cost_units: 0,
                elapsed_ms: 0,
                quality_score: None,
            },
            _ => Outcome {
                ok: false,
                // In muxer-style routing, operational failures are typically treated as “hard junk”.
                junk: true,
                hard_junk: true,
                cost_units: 0,
                elapsed_ms: 0,
                quality_score: None,
            },
        }
    }

    fn reward01(o: Outcome) -> f64 {
        // Scalar “goodness” signal: only ok_clean is 1.
        if o.ok && !o.junk {
            1.0
        } else {
            0.0
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct Scenario {
        horizon: u64,
        change_at: u64,
        // Arm A is stable. Arm B changes at change_at.
        a: Cat4,
        b_before: Cat4,
        b_after: Cat4,
        // Detection config: fixed-baseline CatKL.
        // Thresholding is intentionally simple; the point here is sampling-rate sensitivity.
        det_alpha: f64,
        det_min_n: u64,
        det_threshold: f64,

        // A second detector: categorical CUSUM against a fixed alternative.
        // This is a "forgetful" detector, contrasting with cumulative CatKL inertia.
        cusum_min_n: u64,
        cusum_threshold: f64,
    }

    #[derive(Debug, Clone)]
    enum Policy {
        Thompson { decay: f64 },
        MabDeterministic,
        MabWithCoverage { min_fraction: f64, min_floor: u64 },
    }

    impl Policy {
        fn name(&self) -> String {
            match *self {
                Policy::Thompson { decay } => format!("thompson(decay={decay:.4})"),
                Policy::MabDeterministic => "mab(deterministic)".to_string(),
                Policy::MabWithCoverage {
                    min_fraction,
                    min_floor,
                } => format!("mab+coverage(frac={min_fraction:.3},floor={min_floor})"),
            }
        }
    }

    #[derive(Debug, Clone, Copy, Default)]
    struct TrialMetrics {
        mean_reward: f64,
        // detection delays for arm B (the one that changes); None => never detected.
        det_delay_b_catkl: Option<u64>,
        det_delay_b_cusum: Option<u64>,
        // Detection rates and quantiles (filled in `summarize`; per-trial values are left as defaults).
        det_rate_b_catkl: f64,
        det_rate_b_cusum: f64,
        det_p90_b_catkl: Option<u64>,
        det_p90_b_cusum: Option<u64>,
        // crude estimation error: |p_hat(ok_clean for B before change) - p_true|
        // computed from the policy's internal estimate after warmup.
        est_abs_err_b_before: f64,
        // share of pulls spent on arm B (proxy for “monitoring budget”).
        frac_pulls_b: f64,
    }

    fn run_trial(seed: u64, scenario: Scenario, policy: &Policy, gap_label: &str) -> TrialMetrics {
        let arms = vec!["A".to_string(), "B".to_string()];
        let mut rng = StdRng::seed_from_u64(seed ^ 0xA11C_EED5);

        // Sliding-window stats used by deterministic MAB selection.
        let mut windows: BTreeMap<String, Window> = arms
            .iter()
            // keep a moderately sized window so deterministic MAB can adapt eventually if it samples
            .map(|a| (a.clone(), Window::new(200)))
            .collect();

        // Lifetime counters (used for coverage quotas; windows are intentionally bounded).
        let mut lifetime_calls: BTreeMap<String, (u64, u64)> = arms
            .iter()
            .map(|a| (a.clone(), (0u64, 0u64))) // (calls, elapsed_ms_sum)
            .collect();

        // Fixed-baseline detectors on arm B only.
        let b0 = normalize_cat(scenario.b_before.p);
        let mut det_b = CatKlDetector::new(
            &b0,
            scenario.det_alpha,
            scenario.det_min_n,
            scenario.det_threshold,
            1e-12,
        )
        .expect("detector should be constructible");

        let b1 = normalize_cat(scenario.b_after.p);
        let mut cusum_b = CusumCatDetector::new(
            &b0,
            &b1,
            scenario.det_alpha,
            scenario.cusum_min_n,
            scenario.cusum_threshold,
            1e-12,
        )
        .expect("cusum detector should be constructible");

        // For Thompson policies: one TS instance.
        let mut ts = ThompsonSampling::with_seed(
            ThompsonConfig {
                decay: match *policy {
                    Policy::Thompson { decay } => decay,
                    _ => 1.0,
                },
                ..ThompsonConfig::default()
            },
            seed ^ 0x7A5E_D00D,
        );

        // Deterministic MAB config: success only (junk/fail already impact ok_clean reward).
        // This intentionally creates the “starvation” failure mode unless coverage exists.
        let mab_cfg = MabConfig {
            exploration_c: 0.7,
            cost_weight: 0.0,
            latency_weight: 0.0,
            junk_weight: 0.0,
            hard_junk_weight: 0.0,
            ..MabConfig::default()
        };

        let guard = LatencyGuardrailConfig::default();

        let mut total_reward = 0.0;
        let mut pulls_b = 0u64;
        let mut det_delay_b_catkl: Option<u64> = None;
        let mut det_delay_b_cusum: Option<u64> = None;

        // Warmup: estimate p(ok_clean for B before change) from the policy's internal state.
        // We'll snapshot the estimate just before the change point.
        let mut est_b_before: Option<f64> = None;

        for t in 0..scenario.horizon {
            let chosen = match *policy {
                Policy::Thompson { .. } => ts.select(&arms).expect("non-empty arms").clone(),

                Policy::MabDeterministic => {
                    let summaries: BTreeMap<String, _> = windows
                        .iter()
                        .map(|(k, w)| (k.clone(), w.summary()))
                        .collect();
                    select_mab_decide(&arms, &summaries, mab_cfg).chosen
                }

                Policy::MabWithCoverage {
                    min_fraction,
                    min_floor,
                } => {
                    let coverage = CoverageConfig {
                        enabled: true,
                        min_fraction,
                        min_calls_floor: min_floor,
                    };
                    let fill = policy_fill_k_observed_with_coverage(
                        seed ^ 0xC0DE_4A6E, // deterministic tag
                        &arms,
                        1,
                        false,
                        coverage,
                        guard,
                        |arm| lifetime_calls.get(arm).copied().unwrap_or((0, 0)),
                        |eligible, _k_remaining| {
                            let summaries: BTreeMap<String, _> = windows
                                .iter()
                                .map(|(k, w)| (k.clone(), w.summary()))
                                .collect();
                            // Deterministic selector operates only on the eligible set.
                            vec![select_mab_decide(eligible, &summaries, mab_cfg).chosen]
                        },
                    );
                    fill.chosen
                        .first()
                        .cloned()
                        .unwrap_or_else(|| "A".to_string())
                }
            };

            let cat = match chosen.as_str() {
                "A" => normalize_cat(scenario.a.p),
                "B" => {
                    if t < scenario.change_at {
                        normalize_cat(scenario.b_before.p)
                    } else {
                        normalize_cat(scenario.b_after.p)
                    }
                }
                _ => normalize_cat(scenario.a.p),
            };

            let idx = categorical_sample(&mut rng, cat);
            let o = outcome_from_cat(idx);
            let r01 = reward01(o);

            // Update policy state.
            match *policy {
                Policy::Thompson { .. } => ts.update_reward(&chosen, r01),
                Policy::MabDeterministic | Policy::MabWithCoverage { .. } => {}
            }

            // Update windows + detection if the chosen arm is sampled.
            windows.get_mut(&chosen).expect("arm window exists").push(o);
            if let Some((c, ms)) = lifetime_calls.get_mut(&chosen) {
                *c = c.saturating_add(1);
                *ms = ms.saturating_add(o.elapsed_ms);
            }

            if chosen == "B" {
                pulls_b += 1;
                if det_delay_b_catkl.is_none() && t >= scenario.change_at {
                    if det_b.update(idx).is_some() {
                        det_delay_b_catkl = Some(t - scenario.change_at);
                    }
                } else {
                    let _ = det_b.update(idx);
                }

                if det_delay_b_cusum.is_none() && t >= scenario.change_at {
                    if cusum_b.update(idx).is_some() {
                        det_delay_b_cusum = Some(t - scenario.change_at);
                    }
                } else {
                    let _ = cusum_b.update(idx);
                }
            }

            total_reward += r01;

            if t + 1 == scenario.change_at {
                // Snapshot estimate "just before change".
                let est = match *policy {
                    Policy::Thompson { .. } => ts
                        .stats()
                        .get("B")
                        .map(|s| s.expected_value())
                        .unwrap_or(0.5),
                    _ => {
                        // Use empirical ok_clean rate in window.
                        let s = windows.get("B").map(|w| w.summary()).unwrap_or_default();
                        // ok_clean is "ok and not junk"; our summary doesn't track that directly,
                        // but in this simplified sim we treat junk as a subset of ok, so:
                        // ok_clean ~= ok - junk (clamped).
                        if s.calls == 0 {
                            0.5
                        } else {
                            let ok_clean = s.ok.saturating_sub(s.junk) as f64;
                            (ok_clean / (s.calls as f64)).clamp(0.0, 1.0)
                        }
                    }
                };
                est_b_before = Some(est);
            }
        }

        let mean_reward = total_reward / (scenario.horizon as f64).max(1.0);
        let frac_pulls_b = (pulls_b as f64) / (scenario.horizon as f64).max(1.0);
        let p_true_b0_ok_clean = normalize_cat(scenario.b_before.p)[0];
        let est_abs_err_b_before = (est_b_before.unwrap_or(0.5) - p_true_b0_ok_clean).abs();

        if seed.is_multiple_of(10_000) {
            eprintln!(
                "debug: gap={gap_label} policy={} mean_reward={:.4} det_catkl={:?} det_cusum={:?} frac_B={:.3} est_err_b0={:.3}",
                policy.name(),
                mean_reward,
                det_delay_b_catkl,
                det_delay_b_cusum,
                frac_pulls_b,
                est_abs_err_b_before
            );
        }

        TrialMetrics {
            mean_reward,
            det_delay_b_catkl,
            det_delay_b_cusum,
            det_rate_b_catkl: 0.0,
            det_rate_b_cusum: 0.0,
            det_p90_b_catkl: None,
            det_p90_b_cusum: None,
            est_abs_err_b_before,
            frac_pulls_b,
        }
    }

    fn summarize(trials: &[TrialMetrics]) -> TrialMetrics {
        let n = trials.len().max(1) as f64;
        let mean_reward = trials.iter().map(|t| t.mean_reward).sum::<f64>() / n;
        let est_abs_err_b_before = trials.iter().map(|t| t.est_abs_err_b_before).sum::<f64>() / n;
        let frac_pulls_b = trials.iter().map(|t| t.frac_pulls_b).sum::<f64>() / n;

        fn quantile_sorted(xs: &[u64], q: f64) -> Option<u64> {
            if xs.is_empty() {
                return None;
            }
            let q = if q.is_finite() {
                q.clamp(0.0, 1.0)
            } else {
                0.5
            };
            let idx = ((xs.len().saturating_sub(1) as f64) * q).round() as usize;
            xs.get(idx).copied()
        }

        fn rate_and_p90(
            xs: &[TrialMetrics],
            f: fn(&TrialMetrics) -> Option<u64>,
        ) -> (f64, Option<u64>) {
            if xs.is_empty() {
                return (0.0, None);
            }
            let mut ds: Vec<u64> = xs.iter().filter_map(f).collect();
            let rate = (ds.len() as f64) / (xs.len() as f64);
            ds.sort_unstable();
            let p90 = quantile_sorted(&ds, 0.90);
            (rate, p90)
        }

        fn avg_delay(xs: &[TrialMetrics], f: fn(&TrialMetrics) -> Option<u64>) -> Option<u64> {
            let mut sum = 0.0;
            let mut n = 0.0;
            for t in xs {
                if let Some(d) = f(t) {
                    sum += d as f64;
                    n += 1.0;
                }
            }
            if n > 0.0 {
                Some((sum / n).round() as u64)
            } else {
                None
            }
        }

        let det_delay_b_catkl = avg_delay(trials, |t| t.det_delay_b_catkl);
        let det_delay_b_cusum = avg_delay(trials, |t| t.det_delay_b_cusum);
        let (det_rate_b_catkl, det_p90_b_catkl) = rate_and_p90(trials, |t| t.det_delay_b_catkl);
        let (det_rate_b_cusum, det_p90_b_cusum) = rate_and_p90(trials, |t| t.det_delay_b_cusum);

        TrialMetrics {
            mean_reward,
            det_delay_b_catkl,
            det_delay_b_cusum,
            det_rate_b_catkl,
            det_rate_b_cusum,
            det_p90_b_catkl,
            det_p90_b_cusum,
            est_abs_err_b_before,
            frac_pulls_b,
        }
    }

    fn det_compact(mean: Option<u64>, p90: Option<u64>, rate: f64) -> String {
        if mean.is_none() || !(rate.is_finite() && rate > 0.0) {
            return "never".into();
        }
        let mean = mean.unwrap_or(0);
        let p90 = p90.unwrap_or(mean);
        format!("{mean}/{p90}@{rate:.2}")
    }

    // ------------------------------------------------------------
    // Scenario family: arm B starts slightly worse than A, then breaks.
    // We vary the pre-change "gap" by shifting ok_clean probability.
    // ------------------------------------------------------------

    let horizon = 3_000u64;
    let change_at = 1_500u64;

    let a = Cat4 {
        p: normalize_cat([0.92, 0.02, 0.01, 0.05]),
    };
    let b_after = Cat4 {
        p: normalize_cat([0.10, 0.10, 0.30, 0.50]),
    };

    // Detector settings:
    // - CatKL is cumulative -> shows inertia.
    // - CUSUM is max-with-zero -> forgetful -> shows that "just change the detector"
    //   can turn "never" into "bounded", if you sample the arm at all.
    let det_alpha = 1e-3;
    let det_min_n = 30;
    let det_threshold = 8.0;
    let cusum_min_n = 20;
    let cusum_threshold = 4.0;

    let policies: Vec<Policy> = vec![
        Policy::MabDeterministic,
        Policy::MabWithCoverage {
            min_fraction: 0.02,
            min_floor: 10,
        },
        Policy::MabWithCoverage {
            min_fraction: 0.05,
            min_floor: 10,
        },
        Policy::Thompson { decay: 1.0 },
        Policy::Thompson { decay: 0.995 },
        Policy::Thompson { decay: 0.98 },
    ];

    // Gaps: higher means B is worse (more likely starved) => detection should get harder.
    let gaps: Vec<(String, f64)> = vec![
        ("gap=0.01".to_string(), 0.01),
        ("gap=0.03".to_string(), 0.03),
        ("gap=0.06".to_string(), 0.06),
        ("gap=0.10".to_string(), 0.10),
    ];

    // Trials per configuration.
    let trials_per = 60u64;
    let base_seed = 123u64;

    for (gap_label, gap) in gaps {
        // Move probability mass from ok_clean to fail for B_before to create a gap.
        let b0_ok_clean = (a.p[0] - gap).clamp(0.01, 0.99);
        let b_before = Cat4 {
            p: normalize_cat([b0_ok_clean, 0.02, 0.01, 1.0 - b0_ok_clean - 0.02 - 0.01]),
        };

        let severity = drift_simplex(
            &normalize_cat(b_before.p),
            &normalize_cat(b_after.p),
            DriftMetric::Rao,
            1e-12,
        )
        .unwrap_or(f64::NAN);

        eprintln!("\n=== {} (rao_severity={:.4} rad) ===", gap_label, severity);

        // Pareto frontier over:
        // - maximize mean_reward
        // - minimize detection delay (CUSUM; None treated as +infty in reporting; excluded)
        // - minimize estimation error
        let mut frontier: ParetoFrontier<String> = ParetoFrontier::new(vec![
            Direction::Maximize,
            Direction::Minimize,
            Direction::Minimize,
        ])
        .with_labels(vec!["reward".into(), "det_delay".into(), "est_err".into()]);

        // Collect full table for printing.
        let mut rows: Vec<(TrialMetrics, String)> = Vec::new();

        for p in &policies {
            let scenario = Scenario {
                horizon,
                change_at,
                a,
                b_before,
                b_after,
                det_alpha,
                det_min_n,
                det_threshold,
                cusum_min_n,
                cusum_threshold,
            };

            let mut trials: Vec<TrialMetrics> = Vec::new();
            for i in 0..trials_per {
                let seed = base_seed ^ ((i + 1) * 0x9E37_79B9) ^ gap.to_bits();
                trials.push(run_trial(seed, scenario, p, &gap_label));
            }
            let m = summarize(&trials);

            // For the frontier, treat "never detected" as not comparable (infinite delay).
            if let Some(d) = m.det_delay_b_cusum {
                frontier.push(
                    vec![m.mean_reward, d as f64, m.est_abs_err_b_before],
                    p.name(),
                );
            }

            rows.push((m, p.name()));
        }

        rows.sort_by(|a, b| b.0.mean_reward.total_cmp(&a.0.mean_reward));
        for (m, name) in &rows {
            eprintln!(
                "{:<28} reward={:.4} cusum={:<14} catkl={:<14} frac_B={:.3} est_err_b0={:.3}",
                name,
                m.mean_reward,
                det_compact(m.det_delay_b_cusum, m.det_p90_b_cusum, m.det_rate_b_cusum),
                det_compact(m.det_delay_b_catkl, m.det_p90_b_catkl, m.det_rate_b_catkl),
                m.frac_pulls_b,
                m.est_abs_err_b_before
            );
        }

        let mut frontier_rows: Vec<(String, Vec<f64>)> = frontier
            .points()
            .iter()
            .map(|pt| (pt.data.clone(), pt.values.clone()))
            .collect();
        frontier_rows.sort_by(|a, b| b.1[0].total_cmp(&a.1[0]));

        eprintln!("\nfrontier (reward vs cusum_delay vs est_err):");
        for (name, v) in frontier_rows {
            eprintln!(
                "  {:<28} reward={:.4} cusum_delay={:>6.1} est_err={:.3}",
                name, v[0], v[1], v[2]
            );
        }
    }
}
