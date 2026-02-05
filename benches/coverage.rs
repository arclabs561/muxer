use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use muxer::{coverage_pick_under_sampled, coverage_pick_under_sampled_idx, CoverageConfig};
use std::hint::black_box;

fn bench_coverage(c: &mut Criterion) {
    let cfg = CoverageConfig {
        enabled: true,
        min_fraction: 0.02,
        min_calls_floor: 10,
    };

    let mut group = c.benchmark_group("coverage_pick");
    for &n_arms in &[6usize, 32usize, 256usize] {
        let arms: Vec<String> = (0..n_arms).map(|i| format!("arm{i}")).collect();

        // A deterministic, slightly-non-uniform call count pattern.
        let pulls: Vec<u64> = (0..n_arms).map(|i| ((i as u64) * 17 + 3) % 101).collect();

        group.bench_with_input(BenchmarkId::new("string", n_arms), &n_arms, |b, &_n| {
            b.iter(|| {
                let picked = coverage_pick_under_sampled(123, black_box(&arms), 1, cfg, |a| {
                    let idx = a
                        .strip_prefix("arm")
                        .and_then(|s| s.parse::<usize>().ok())
                        .unwrap_or(0);
                    pulls[idx]
                });
                black_box(picked);
            })
        });

        group.bench_with_input(BenchmarkId::new("idx", n_arms), &n_arms, |b, &_n| {
            b.iter(|| {
                let picked = coverage_pick_under_sampled_idx(123, n_arms, 1, cfg, |idx| pulls[idx]);
                black_box(picked);
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_coverage);
criterion_main!(benches);
