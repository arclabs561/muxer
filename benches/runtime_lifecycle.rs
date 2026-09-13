//! Lifecycle cost at the same action counts as the legacy router benchmark.
//!
//! The delayed cases retain real open receipts before measuring a complete new
//! issue/feedback round.  They intentionally measure lookup and correlation
//! cost, not heap use: Criterion has no allocator instrumentation here.

#[cfg(feature = "stochastic")]
mod enabled {
    use criterion::{black_box, BenchmarkId, Criterion};
    use muxer::{
        BernoulliThompson, BoundedReward, Exp3Ix, Exp3IxConfig, Exp3Profile, Muxer, Outcome,
        QualityFeedback, QualityProfile, Router, RouterConfig, RuntimeConfig, ThompsonConfig,
        ThompsonSampling,
    };

    fn actions(n: usize) -> Vec<String> {
        (0..n).map(|index| format!("arm{index}")).collect()
    }

    fn outcome() -> Outcome {
        Outcome::success(3, 80)
    }

    fn runtime_config(pending_capacity: usize) -> RuntimeConfig {
        RuntimeConfig {
            pending_capacity,
            terminal_capacity: 1,
            event_capacity: 2,
            retired_epoch_capacity: 1,
            seed: 17,
        }
    }

    fn quality_runtime(n: usize, pending_capacity: usize, delayed: bool) -> Muxer<QualityProfile> {
        let profile =
            QualityProfile::new(actions(n), RouterConfig::default()).expect("valid router");
        let profile = if delayed {
            profile.with_delayed_score()
        } else {
            profile
        };
        Muxer::with_config(actions(n), profile, runtime_config(pending_capacity))
            .expect("valid runtime")
    }

    fn fill_pending(runtime: &mut Muxer<QualityProfile>, pending: usize) {
        let context: &[f64] = &[];
        for _ in 0..pending {
            black_box(runtime.decide(context).expect("capacity reserved"));
        }
    }

    pub fn bench_scalar_lifecycle(c: &mut Criterion) {
        let mut group = c.benchmark_group("runtime_scalar");
        for &n in &[5_usize, 25] {
            let arms = actions(n);

            group.bench_with_input(
                BenchmarkId::new("legacy-thompson/decide-tell", n),
                &n,
                |b, &_| {
                    let mut policy = ThompsonSampling::with_seed(ThompsonConfig::default(), 17);
                    b.iter(|| {
                        let decision = policy.decide(black_box(&arms)).expect("actions present");
                        policy.update_reward(black_box(&decision.chosen), black_box(1.0));
                        black_box(decision);
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new("runtime-bernoulli/decide-tell", n),
                &n,
                |b, &_| {
                    let mut runtime = Muxer::new(
                        actions(n),
                        BernoulliThompson::with_seed(ThompsonConfig::default(), 17),
                    )
                    .expect("valid runtime");
                    b.iter(|| {
                        let receipt = runtime.decide(black_box(&())).expect("decision");
                        let disposition = runtime
                            .tell(receipt.id(), black_box(true))
                            .expect("feedback");
                        black_box((receipt, disposition));
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new("legacy-exp3ix/decide-tell", n),
                &n,
                |b, &_| {
                    let mut policy = Exp3Ix::new(Exp3IxConfig::default());
                    let mut seed = 17_u64;
                    b.iter(|| {
                        let decision = policy
                            .decide_deterministic_filtered(black_box(&arms), &arms, seed)
                            .expect("actions present");
                        let probability =
                            decision.probs.as_ref().expect("EXP3 propensity")[&decision.chosen];
                        policy.update_reward_with_prob(
                            black_box(&decision.chosen),
                            black_box(1.0),
                            probability,
                        );
                        seed = seed.wrapping_add(1);
                        black_box(decision);
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new("runtime-exp3/decide-tell", n),
                &n,
                |b, &_| {
                    let mut runtime = Muxer::new(
                        actions(n),
                        Exp3Profile::new(actions(n), Exp3IxConfig::default())
                            .expect("valid EXP3 profile"),
                    )
                    .expect("valid runtime");
                    let reward = BoundedReward::new(1.0).expect("bounded reward");
                    b.iter(|| {
                        let receipt = runtime.decide(black_box(&())).expect("decision");
                        let disposition = runtime
                            .tell(receipt.id(), black_box(reward))
                            .expect("feedback");
                        black_box((receipt, disposition));
                    });
                },
            );
        }
        group.finish();
    }

    pub fn bench_quality_lifecycle(c: &mut Criterion) {
        let mut group = c.benchmark_group("runtime_quality");
        let context: &[f64] = &[];
        for &n in &[5_usize, 25] {
            let arms = actions(n);
            group.bench_with_input(
                BenchmarkId::new("legacy-router/select-observe", n),
                &n,
                |b, &_| {
                    let mut router =
                        Router::new(actions(n), RouterConfig::default()).expect("valid router");
                    let mut seed = 17_u64;
                    b.iter(|| {
                        let decision = router.select(1, seed);
                        let action = decision.primary().expect("one action").to_owned();
                        assert!(router.observe(&action, outcome()));
                        seed = seed.wrapping_add(1);
                        black_box(decision);
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new("runtime-immediate/decide-tell", n),
                &n,
                |b, &_| {
                    let mut runtime = quality_runtime(n, 1, false);
                    b.iter(|| {
                        let receipt = runtime.decide(black_box(context)).expect("decision");
                        let disposition = runtime
                            .tell(receipt.id(), QualityFeedback::execution(outcome()))
                            .expect("execution feedback");
                        black_box((receipt, disposition));
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new("runtime-delayed/decide-tell", n),
                &n,
                |b, &_| {
                    let mut runtime = quality_runtime(n, 1, true);
                    b.iter(|| {
                        let receipt = runtime.decide(black_box(context)).expect("decision");
                        let execution = runtime
                            .tell(receipt.id(), QualityFeedback::execution(outcome()))
                            .expect("execution feedback");
                        let score = runtime
                            .tell(receipt.id(), QualityFeedback::score(0.9).expect("score"))
                            .expect("score feedback");
                        black_box((receipt, execution, score));
                    });
                },
            );

            for &pending in &[1_usize, 100, 1_000] {
                group.bench_with_input(
                    BenchmarkId::new(format!("runtime-delayed/pending={pending}/decide-tell"), n),
                    &pending,
                    |b, &pending| {
                        let mut runtime = quality_runtime(n, pending + 1, true);
                        fill_pending(&mut runtime, pending);
                        b.iter(|| {
                            let receipt =
                                runtime.decide(black_box(context)).expect("one free ticket");
                            let execution = runtime
                                .tell(receipt.id(), QualityFeedback::execution(outcome()))
                                .expect("execution feedback");
                            let score = runtime
                                .tell(receipt.id(), QualityFeedback::score(0.9).expect("score"))
                                .expect("score feedback");
                            black_box((receipt, execution, score));
                        });
                    },
                );
            }
            black_box(arms);
        }
        group.finish();
    }
}

#[cfg(feature = "stochastic")]
criterion::criterion_group!(
    benches,
    enabled::bench_scalar_lifecycle,
    enabled::bench_quality_lifecycle
);

#[cfg(feature = "stochastic")]
criterion::criterion_main!(benches);

#[cfg(not(feature = "stochastic"))]
fn main() {}
