use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use muxer::monitor::{CatKlDetector, CusumCatBank, CusumCatDetector};
use std::hint::black_box;

fn bench_monitor(c: &mut Criterion) {
    // A deterministic categorical stream (length chosen to dwarf setup costs).
    let n = 4096usize;
    let mut xs: Vec<usize> = Vec::with_capacity(n);
    for i in 0..n {
        // Mostly category 0, with periodic spikes to categories 2/3.
        let x = if i % 97 == 0 {
            3
        } else if i % 53 == 0 {
            2
        } else if i % 17 == 0 {
            1
        } else {
            0
        };
        xs.push(x);
    }

    let p0 = [0.90, 0.03, 0.02, 0.05];
    let alts = [
        [0.05, 0.05, 0.45, 0.45], // fail+hard heavy
        [0.10, 0.05, 0.05, 0.80], // mostly fail
        [0.10, 0.05, 0.75, 0.10], // mostly hard junk
        [0.10, 0.75, 0.05, 0.10], // mostly soft junk
    ];

    let alpha = 1e-3;
    let min_n = 30;
    let tol = 1e-12;
    let threshold = 1e9; // effectively "never alarm" for these microbenches

    let mut group = c.benchmark_group("monitor_update");

    group.bench_function("cusum/single", |b| {
        let base = CusumCatDetector::new(&p0, &alts[0], alpha, min_n, threshold, tol).unwrap();
        b.iter(|| {
            let mut d = base.clone();
            for &x in &xs {
                black_box(d.update(x));
            }
            black_box(d.score());
        })
    });

    group.bench_with_input(BenchmarkId::new("cusum/bank", 4), &4usize, |b, &_k| {
        let base: Vec<CusumCatDetector> = alts
            .iter()
            .map(|p1| CusumCatDetector::new(&p0, p1, alpha, min_n, threshold, tol).unwrap())
            .collect();
        b.iter(|| {
            let mut ds = base.clone();
            let mut smax = 0.0f64;
            for &x in &xs {
                for d in ds.iter_mut() {
                    black_box(d.update(x));
                    smax = smax.max(d.score());
                }
            }
            black_box(smax);
        })
    });

    group.bench_with_input(
        BenchmarkId::new("cusum/bank_struct", 4),
        &4usize,
        |b, &_k| {
            let alts_vec: Vec<Vec<f64>> = alts.iter().map(|p1| p1.to_vec()).collect();
            let base = CusumCatBank::new(&p0, &alts_vec, alpha, min_n, threshold, tol).unwrap();
            b.iter(|| {
                let mut bank = base.clone();
                let mut smax = 0.0f64;
                for &x in &xs {
                    let upd = bank.update(x);
                    smax = smax.max(upd.score_max);
                }
                black_box(smax);
            })
        },
    );

    group.bench_function("catkl/update+score", |b| {
        let base = CatKlDetector::new(&p0, alpha, min_n, f64::INFINITY, tol).unwrap();
        b.iter(|| {
            let mut d = base.clone();
            let mut last = 0.0f64;
            for &x in &xs {
                black_box(d.update(x));
                last = d.score().unwrap_or(0.0);
            }
            black_box(last);
        })
    });

    group.finish();
}

criterion_group!(benches, bench_monitor);
criterion_main!(benches);
