# Shared runtime performance

Measured 2026-09-13 on an Apple M3 Max (16 CPU cores, 128 GiB),
`aarch64-apple-darwin`, Rust/Cargo 1.98.1. These are local measurements,
not throughput guarantees. The runtime adds identity, retained evidence,
transactional updates and finality; direct kernels do not provide those services.

## Quality lifecycle

The benchmark source is [runtime_lifecycle.rs](../benches/runtime_lifecycle.rs).
Reproduce with default features:

```sh
cargo bench --bench runtime_lifecycle -- runtime_quality --sample-size 30 --warm-up-time 0.3 --measurement-time 0.5
```

Ranges below span three isolated repetitions with no concurrent project builds.
Each sample issues and resolves one new decision; the pending workload also
keeps 1,000 real receipts open. Terminal retention is one decision for these
quality workloads. These are timing measurements, not heap measurements.

| Workload | 5 actions | 25 actions |
| --- | ---: | ---: |
| Direct Router selection and observation | 6.14–6.17 µs | 24.05–25.15 µs |
| Runtime immediate execution | 9.13–9.25 µs | 33.97–34.41 µs |
| Runtime execution plus delayed score | 10.82–10.91 µs | 37.58–37.82 µs |
| Runtime delayed score with 1,000 pending | 11.17–11.31 µs | 38.30–39.11 µs |

Immediate lifecycle overhead is approximately 35–51%. Keeping 1,000 pending
receipts adds approximately 1–4% over the delayed path without that backlog.
No retention scaling cliff appeared at the tested sizes; this does not establish
behavior at arbitrary capacities or payload sizes.

## Scalar lifecycle and compatibility baseline

Three 30-sample repetitions of the `runtime_scalar` group measured:

| Kernel versus runtime | 5 actions | 25 actions |
| --- | ---: | ---: |
| Direct Thompson | 0.852–0.857 µs | 4.80–4.81 µs |
| Runtime Bernoulli Thompson | 2.27–2.29 µs | 9.85–9.90 µs |
| Direct EXP3 | 0.418–0.446 µs | 3.09–3.11 µs |
| Runtime EXP3 | 1.87–2.01 µs | 8.71–8.76 µs |

The runtime costs roughly 2–5 times the bare scalar kernel in these small
workloads. Kernel and runtime comparisons intentionally differ in lifecycle
services and retention settings; they are not equivalent-capability systems.

An immutable archive of baseline `307a03a` was also tested and benchmarked.
Short Router selection smoke comparisons across 5/25 actions, batch sizes
1/5/10 where applicable, and monitoring on/off stayed within approximately
7% of baseline. These short runs cannot distinguish small regressions from
host noise. Earlier quality repetitions overlapping CI builds were excluded.

## Interpretation and next optimization gate

The added lifecycle exceeds the provisional 20% overhead review threshold.
The implementation keeps it additive and opt-in: direct kernels remain available
when an application does not need correlated delayed feedback. It makes no
zero-overhead claim. Prepare/apply atomicity currently uses cloned policy state;
optimizing that boundary must preserve rejection atomicity and delayed-evidence
tests, not merely improve a microbenchmark.

A symbolicated, headless pre-freeze sample of the real delayed-quality workload
found leaf samples in `Window::summary` (8.1%), outcome-window cloning (6.6%)
and observation-ID-window cloning (1.6%). These are leaf shares, not inclusive
cost attribution or proof that three changes would recover those percentages.
They identify window preparation and summary work as a useful next investigation.

Entry counts are bounded; arbitrary ticket/context bytes have not been measured.
Serialized restart and its cost are outside this implementation.

## Local evidence handoff

Raw baseline, benchmark and profiling artifacts remain at
`/private/tmp/muxer-baseline.tiPxjJ/`; final isolated quality logs are
`bench-runtime-quality-quiet-{1,2,3}.log`. Final local validation logs remain at
`/tmp/muxer-final-qa.WuNKKd/`. These session-owned artifacts are retained for the
operator to inspect, not portable dependencies or published benchmark data.
