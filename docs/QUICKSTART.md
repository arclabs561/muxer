# muxer — 5-minute quickstart

Route repeated calls between a small set of arms (backends, model versions, data sources)
using sliding-window statistics, Pareto selection, and optional CUSUM monitoring.

## Start here

```bash
cargo run --example getting_started
```

Expected last two lines:

```
Best arm now: "claude-sonnet"  (mode: Normal)
With quality_weight=1.0: best arm is "gpt-4o"
```

This demonstrates the 80% case — 3 arms, explore-first, delayed quality labeling —
and shows exactly why `quality_weight` matters for breaking ties that binary rates can't.

## Two-minute version (Router API)

```rust
use muxer::{Outcome, Router, RouterConfig};

let mut router = Router::new(
    vec!["arm-a".to_string(), "arm-b".to_string()],
    RouterConfig::default(),
).unwrap();

for round in 0..100_u64 {
    let d = router.select(1, round);           // pick an arm
    let arm = d.primary().unwrap().to_string();

    // ... make the call ...

    // Step 1: record outcome immediately.
    router.observe(&arm, Outcome {
        ok: true, junk: false, hard_junk: false,
        cost_units: 5, elapsed_ms: 200, quality_score: None,
    });

    // Step 2 (delayed): update quality after scoring.
    router.set_last_junk_level(&arm, false, false); // or true when junk
    router.set_last_quality_score(&arm, 0.85);      // [0, 1], higher = better
}
```

## Typical output shape

After enough rounds, the arm with the lowest junk rate and highest quality score
dominates. The router explores every arm first (`prechosen` in the decision),
then exploits via UCB.

```
arm-a  calls=50  ok_rate=0.96  junk_rate=0.02  quality=0.91
arm-b  calls=50  ok_rate=0.84  junk_rate=0.12  quality=0.65
→ router selects arm-a
```

## Configuration checklist

| Goal | Config |
|------|--------|
| Tune window size for your throughput | `RouterConfig::default().window_cap(suggested_window_cap(calls_per_arm, change_rate))` |
| Use quality gradient for selection | `MabConfig { quality_weight: 0.5, ..Default::default() }` |
| Detect regressions automatically | `.with_monitoring(400, 80).with_triage()` |
| Faster initial coverage (K > 10) | `router.select(3, seed)` (k=3 per round) |
| Production CUSUM calibration | `cargo run --example router_production --features stochastic` |

## Common failures

- **`missing field quality_score`** when constructing `Outcome`: add `quality_score: None`.
- **`missing field mean_quality_score`** when constructing `Summary`: add `mean_quality_score: None`.
- **Triage never fires**: use `.with_triage()` on `RouterConfig`; default has no CUSUM.
- **All arms always explore first**: expected behavior — each arm is tried once before
  exploitation.  Use `select(k=3)` to batch initial exploration over many arms.
- **CI can't resolve muxer**: the crate is on [crates.io](https://crates.io/crates/muxer).
  For local dev with a checkout, create `.cargo/config.toml` from `.cargo/config.toml.example`.

## Next steps

| What | Where |
|------|-------|
| Production pattern (monitoring, calibration) | `cargo run --example router_production --features stochastic` |
| Matrix harness pattern (task x dataset x backend) | `cargo run --example matrix_harness` |
| PCAP/security harness pattern | `cargo run --example pcap_triage_harness` |
| Synthetic drift harness pattern | `cargo run --example synthetic_drift_harness` |
| Recsys/ad-auction harness pattern | `cargo run --example ad_auction_harness` |
| Theoretical background (BQCD, two clocks) | `examples/EXPERIMENTS.md` |
| Full API reference | `https://docs.rs/muxer` |
| Changelog | `CHANGELOG.md` |
