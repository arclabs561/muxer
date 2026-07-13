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

This demonstrates the 80% case: 3 arms, explore-first, finalized outcomes, and a
quality-versus-cost tradeoff resolved by scalarization.

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

    // Score the completed call, then record one finalized outcome.
    let outcome = Outcome::with_quality(true, false, false, 5, 200, 0.85);
    assert!(router.observe(&arm, outcome));
}
```

When labels arrive after later calls for the same arm, attach a caller-owned
`ObservationId` to each outcome and update that ID directly:

```rust
use muxer::{ObservationId, Outcome, Router, RouterConfig};

let mut router = Router::new(vec!["arm-a".to_string()], RouterConfig::default()).unwrap();
let first = ObservationId::new(1);
let second = ObservationId::new(2);
assert!(router.observe_with_id(first, "arm-a", Outcome::success(1, 20)));
assert!(router.observe_with_id(second, "arm-a", Outcome::success(1, 20)));
assert!(router.set_quality_score_for_id(first, 0.9));
```

IDs are caller-owned and must not be reused while the observation may still be
retained. The existing `set_last_*` methods remain for single-flight callers.

## Metric-vector selection

`Router` is the quality-profile API. For a different feedback schema, pass
caller-defined metric vectors directly:

```rust
use muxer::{select_candidate_assessments, CandidateAssessment, MetricObjective};

let candidates = vec![
    CandidateAssessment::new("accurate", 100, vec![0.95, 240.0]),
    CandidateAssessment::new("fast", 100, vec![0.90, 80.0]),
];
let selection = select_candidate_assessments(
    &candidates,
    &[
        MetricObjective::maximize(0, 40.0),
        MetricObjective::minimize(1, 0.01),
    ],
)
.unwrap();
assert_eq!(selection.chosen.as_deref(), Some("accurate"));
```

Metric positions are caller-defined. The selector validates finite values and
matching dimensions, then returns the frontier and resolved score rows for
inspection or, with `serde`, serialization. It is stateless: the caller owns
aggregation, normalization, history, and context. The `observations` value is
diagnostic metadata and does not affect selection.

## Typical output shape

Selection depends on the configured objectives and current windows. With the
default configuration, unseen arms are pre-picked first, then the Router chooses
from the Pareto frontier using its scalar score and stable tiebreak.

```
arm-a  calls=50  ok_rate=0.96  junk_rate=0.02  quality=0.91
arm-b  calls=50  ok_rate=0.84  junk_rate=0.12  quality=0.65
→ router selects arm-a
```

## Configuration checklist

| Goal | Config |
|------|--------|
| Tune window size for your throughput | `RouterConfig::default().window_cap(suggested_window_cap(calls_per_arm, change_rate))` |
| Use quality gradient for selection | `MabConfig::default().with_quality_weight(0.5)` |
| Detect regressions automatically | `.with_monitoring(400, 80).with_triage()` |
| Faster initial coverage (K > 10) | `router.select(3, seed)` (k=3 per round) |
| Restrict one request to ready/capable arms | `router.select_from(&eligible, k, seed)?` |
| CUSUM threshold-selection demo | `cargo run --example router_production --features stochastic` |

## Common failures

- **`missing field quality_score`** when constructing `Outcome`: add `quality_score: None`.
- **`missing field mean_quality_score`** when constructing `Summary`: add `mean_quality_score: None`.
- **Triage never fires**: use `.with_triage()` on `RouterConfig`; default has no CUSUM.
- **All arms always explore first**: expected behavior — each arm is tried once before
  exploitation. Use `select(3, seed)` to batch initial exploration over many arms.
- **An unavailable arm was selected**: pass the authoritative request-local set to
  `select_from`; Router filters are empirical preferences, not readiness checks.
- **CI can't resolve muxer**: the crate is on [crates.io](https://crates.io/crates/muxer).
  For local dev with a checkout, create `.cargo/config.toml` from `.cargo/config.toml.example`.

## Next steps

| What | Where |
|------|-------|
| Combined monitoring/config demo | `cargo run --example router_production --features stochastic` |
| Harmless drift vs restart policy | `cargo run --example significant_shift_sim --features stochastic` |
| Matrix harness pattern (task x dataset x backend) | `cargo run --example matrix_harness` |
| PCAP/security harness pattern | `cargo run --example pcap_triage_harness` |
| Synthetic drift harness pattern | `cargo run --example synthetic_drift_harness` |
| Recsys/ad-auction harness pattern | `cargo run --example ad_auction_harness` |
| Payments fraud harness pattern | `cargo run --example fraud_scoring_harness` |
| Search ranking harness pattern | `cargo run --example search_ranking_harness` |
| Medical triage harness pattern | `cargo run --example medical_triage_harness` |
| Detection simulations and failure modes | `examples/EXPERIMENTS.md` |
| Full API reference | `https://docs.rs/muxer` |
| Changelog | `CHANGELOG.md` |
