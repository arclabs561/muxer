# muxer

[![crates.io](https://img.shields.io/crates/v/muxer.svg)](https://crates.io/crates/muxer)
[![Documentation](https://docs.rs/muxer/badge.svg)](https://docs.rs/muxer)

Multi-objective bandit routing with drift detection.

Select among K arms from caller-aggregated metric vectors, or use the stateful
quality router with rolling windows and drift detection.

See [examples/EXPERIMENTS.md](examples/EXPERIMENTS.md) for simulations and failure modes.

## Usage

```toml
[dependencies]
muxer = "0.5.3"
```

Deterministic core only (no stochastic bandits):

```toml
[dependencies]
muxer = { version = "0.5", default-features = false }
```

### Feature flags

| Feature | Default | Adds |
| --- | --- | --- |
| `stochastic` | yes | Thompson sampling and EXP3-IX |
| `contextual` | no | `LinUcb` contextual selection |
| `serde` | no | serialization for supported public configs and state |
| `boltzmann` | no | stochastic Gumbel-max softmax selection |

## Quickstart

```rust
use muxer::{Router, RouterConfig, Outcome};

let arms = vec!["backend-a".to_string(), "backend-b".to_string()];
let mut router = Router::new(arms, RouterConfig::default()).unwrap();

loop {
    let d = router.select(1, 0);
    let arm = d.primary().unwrap().to_string();

    let outcome = Outcome::success(5, 120);
    assert!(router.observe(&arm, outcome));
}
```

When availability or capability varies by request, pass the allowed arms
explicitly. Every Router selection stage stays inside this set:

```rust
let eligible = vec!["backend-b".to_string()];
let d = router.select_from(&eligible, 1, 0).unwrap();
assert_eq!(d.primary(), Some("backend-b"));
```

For overlapping calls whose labels arrive later, use caller-owned observation
IDs so labels target the original outcome:

```rust
use muxer::{ObservationId, Outcome};

let id = ObservationId::new(1);
assert!(router.observe_with_id(id, "backend-b", Outcome::success(5, 120)));
assert!(router.set_quality_score_for_id(id, 0.9));
```

These setters correct retained window summaries only. They do not replay
triage detectors. When triage is enabled, pass final `ok`, `junk`, and
`hard_junk` categories to `observe_with_id`; continuous quality scores may
arrive later.

For larger arm counts, pass `k > 1` to batch exploration:

```rust
let cfg = RouterConfig::default().with_coverage(0.02, 1);
let d = router.select(3, 0); // K=30, k=3 -> coverage in ~10 rounds
```

## Examples

### Deterministic multi-objective selection

```rust
use muxer::{select_mab, MabConfig, Summary};
use std::collections::BTreeMap;

let arms = vec!["a".to_string(), "b".to_string()];
let mut summaries = BTreeMap::new();
summaries.insert("a".to_string(), Summary { calls: 10, ok: 9, junk: 0, hard_junk: 0, cost_units: 10, elapsed_ms_sum: 900, mean_quality_score: None });
summaries.insert("b".to_string(), Summary { calls: 10, ok: 9, junk: 2, hard_junk: 0, cost_units: 10, elapsed_ms_sum: 900, mean_quality_score: None });

let sel = select_mab(&arms, &summaries, MabConfig::default());
assert_eq!(sel.chosen, "a"); // lower junk rate wins
```

### Domain-neutral metric selection

When the metrics are not quality labels, pass them directly. Metric positions
are caller-defined; each objective declares whether to maximize or minimize it.

```rust
use muxer::{select_candidate_assessments, CandidateAssessment, MetricObjective};

let candidates = vec![
    CandidateAssessment::new("accurate", 100, vec![0.95, 240.0]),
    CandidateAssessment::new("fast", 100, vec![0.90, 80.0]),
];
let objectives = [
    MetricObjective::maximize(0, 40.0), // metric 0: caller-defined utility
    MetricObjective::minimize(1, 0.01), // metric 1: caller-defined latency
];
let selection = select_candidate_assessments(&candidates, &objectives).unwrap();
assert_eq!(selection.chosen.as_deref(), Some("accurate"));
```

`select_candidate_assessments` is stateless. The caller owns aggregation,
normalization, history, and context; `observations` is diagnostic metadata and
does not affect selection. `Router`, `Outcome`, and `Summary` form the separate
stateful quality-routing profile.

### Detect-then-triage

```rust
use muxer::{TriageSession, TriageSessionConfig, OutcomeIdx};

let arms = vec!["a".to_string(), "b".to_string()];
let mut session = TriageSession::new(&arms, TriageSessionConfig::default()).unwrap();

session.observe("a", OutcomeIdx::OK, &[0.2, 0.3]);
session.observe("b", OutcomeIdx::HARD_JUNK, &[0.8, 0.9]);

let alarmed = session.alarmed_arms();
let bins = session.tracker().active_bins();
let cells = session.top_alarmed_cells(&bins, 3);
```

### Runnable examples

Start here:

```bash
cargo run --example getting_started        # minimal 3-backend routing loop
cargo run --release --example llm_gateway_harness  # model routing under drift
cargo run --example router_quickstart      # eligibility, routing, and CUSUM triage
cargo run --example router_production --features stochastic  # combined config demo
cargo run --example off_policy_evaluation  # IPS over logged rewards and propensities
cargo run --example feedback_regime_matrix # real data; preparation in examples/README.md
```

Algorithm variants: `deterministic_router`, `thompson_router`, `exp3ix_router`, `contextual_router` (requires `contextual` feature), `sticky_mab_router`, `monitored_router`.

Domain harnesses simulate domain-shaped routing with injected drift: LLM
gateways (`llm_gateway_harness`), NLP (`matrix_harness`), network security
(`pcap_triage_harness`), ad ranking, fraud scoring, clinical triage, and search
ranking.

See [examples/README.md](examples/README.md) for runnable examples with captured output, and [examples/EXPERIMENTS.md](examples/EXPERIMENTS.md) for mini-experiments on trade-offs and failure modes.

## Limitations

The generic metric-vector selector does not retain observations or learn a
policy. Callers that need stateful aggregation, contextual models, or online
updates own that state. The built-in `Router` is stateful, but its `Outcome`,
`Summary`, objectives, and triage categories form a quality-oriented profile.

Latency filters, rolling-window comparisons, and detector thresholds are
empirical routing mechanisms. They are not hard safety constraints or
system-wide regret, false-alarm, or detection-delay guarantees. Request-local
eligibility remains a caller decision enforced through `Router::select_from`.

## Development

```bash
cargo test -p muxer
cargo bench -p muxer --bench coverage
```

[Quickstart guide](docs/QUICKSTART.md) | [API docs](https://docs.rs/muxer) | [Changelog](CHANGELOG.md)

## License

Licensed under MIT or Apache-2.0 ([LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE)).
