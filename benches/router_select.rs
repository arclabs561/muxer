use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use muxer::{Outcome, Router, RouterConfig};
use std::hint::black_box;

fn clean(cost: u64, latency: u64) -> Outcome {
    Outcome::success(cost, latency)
}

fn soft_junk(cost: u64, latency: u64) -> Outcome {
    Outcome::new(true, true, false, cost, latency)
}

fn hard_junk(cost: u64, latency: u64) -> Outcome {
    Outcome::failure(cost, latency)
}

fn synthetic_outcome(arm_idx: usize, t: usize) -> Outcome {
    let cost = 1 + (arm_idx as u64 % 5);
    let latency = 50 + ((arm_idx as u64 * 13 + t as u64 * 7) % 200);
    if (t + arm_idx * 17) % 29 == 0 {
        hard_junk(cost, latency)
    } else if (t + arm_idx * 11) % 13 == 0 {
        soft_junk(cost, latency)
    } else {
        clean(cost, latency)
    }
}

fn make_router(n_arms: usize, monitored: bool) -> Router {
    let arms: Vec<String> = (0..n_arms).map(|i| format!("arm{i}")).collect();
    let cfg = if monitored {
        RouterConfig::default().with_monitoring(500, 50)
    } else {
        RouterConfig::default()
    };
    let mut router = Router::new(arms, cfg).unwrap();

    for t in 0..600 {
        for arm_idx in 0..n_arms {
            let arm = format!("arm{arm_idx}");
            assert!(router.observe(&arm, synthetic_outcome(arm_idx, t)));
        }
    }
    router
}

fn bench_router_select(c: &mut Criterion) {
    let mut group = c.benchmark_group("router_select");
    for &n_arms in &[5usize, 25usize] {
        for &k in &[1usize, 5usize, 10usize] {
            if k > n_arms {
                continue;
            }

            let plain = make_router(n_arms, false);
            group.bench_with_input(
                BenchmarkId::new(format!("plain/k={k}"), n_arms),
                &(n_arms, k),
                |b, &(_n, k)| {
                    b.iter(|| {
                        let d = plain.select(k, black_box(42));
                        black_box(d);
                    });
                },
            );

            let monitored = make_router(n_arms, true);
            group.bench_with_input(
                BenchmarkId::new(format!("monitored/k={k}"), n_arms),
                &(n_arms, k),
                |b, &(_n, k)| {
                    b.iter(|| {
                        let d = monitored.select(k, black_box(42));
                        black_box(d);
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_router_select);
criterion_main!(benches);
