# muxer

[![crates.io](https://img.shields.io/crates/v/muxer.svg)](https://crates.io/crates/muxer)
[![Documentation](https://docs.rs/muxer/badge.svg)](https://docs.rs/muxer)
[![CI](https://github.com/arclabs561/muxer/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/muxer/actions/workflows/ci.yml)

Multi-objective bandit routing.

Given K arms (model versions, endpoints, backends), the router observes per-call outcomes (ok/junk/hard_junk, quality score, cost, latency) and selects the next arm. Reward distributions may change at unknown times.

Four selection policies: `select_mab` (deterministic multi-objective Pareto), `ThompsonSampling`, `Exp3Ix` (adversarial), and `LinUcb` (contextual, feature-gated). Regression detection via CUSUM with per-arm and per-(arm, context-bin) triage. Coverage enforcement prevents arm starvation. Designed for 2--10 arms.

See the [API docs](https://docs.rs/muxer) and [examples/EXPERIMENTS.md](examples/EXPERIMENTS.md) for derivations and failure modes.

## Usage

```toml
[dependencies]
muxer = "0.4.0"
```

Deterministic core only (no stochastic bandits):

```toml
[dependencies]
muxer = { version = "0.4.0", default-features = false }
```

## Quickstart

```rust
use muxer::{Router, RouterConfig, Outcome};

let arms = vec!["backend-a".to_string(), "backend-b".to_string()];
let mut router = Router::new(arms, RouterConfig::default()).unwrap();

loop {
    let d = router.select(1, 0);
    let arm = d.primary().unwrap().to_string();

    let outcome = Outcome::success(5, 120);
    router.observe(&arm, outcome);

    if router.mode().is_triage() {
        router.acknowledge_change(&arm);
    }
}
```

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
cargo run --example router_quickstart      # full lifecycle with CUSUM triage
cargo run --example router_production --features stochastic  # production config
```

Algorithm variants: `deterministic_router`, `thompson_router`, `exp3ix_router`, `contextual_router` (requires `contextual` feature), `sticky_mab_router`, `monitored_router`.

Domain harnesses simulate realistic routing with injected drift: NLP (`matrix_harness`), network security (`pcap_triage_harness`), ad ranking, fraud scoring, clinical triage, search ranking.

See `examples/` for 25+ examples and [examples/EXPERIMENTS.md](examples/EXPERIMENTS.md) for mini-experiments on trade-offs and failure modes.

## Development

```bash
cargo test -p muxer
cargo bench -p muxer --bench coverage
```

[Quickstart guide](docs/QUICKSTART.md) | [API docs](https://docs.rs/muxer) | [Changelog](CHANGELOG.md)

## References

1. P. Auer, N. Cesa-Bianchi, and P. Fischer. "Finite-time analysis of the multiarmed bandit problem." *Machine Learning*, 47(2-3):235--256, 2002.
2. S. Agrawal and N. Goyal. "Analysis of Thompson Sampling for the Multi-armed Bandit Problem." *COLT*, 2012.
3. P. Auer, N. Cesa-Bianchi, Y. Freund, and R. E. Schapire. "The Nonstochastic Multiarmed Bandit Problem." *SIAM J. Comput.*, 32(1):48--77, 2002.
4. W. Chu, L. Li, L. Reyzin, and R. Schapire. "Contextual Bandits with Linear Payoff Functions." *AISTATS*, 2011.
5. E. S. Page. "Continuous inspection schemes." *Biometrika*, 41(1-2):100--115, 1954.
6. A. Garivier and E. Moulines. "On Upper-Confidence Bound Policies for Switching Bandit Problems." *ALT*, 2011.
7. R. R. Drugan and A. Nowe. "Designing multi-objective multi-armed bandits algorithms: A study." *IJCNN*, 2013.
8. L. Besson, E. Kaufmann, O.-A. Maillard, and J. Seznec. "Efficient Change-Point Detection for Tackling Piecewise-Stationary Bandits." arXiv:1902.01575, 2019.
9. M. Ehrgott and S. Nickel. "On the number of criteria needed to decide Pareto optimality." *Math. Meth. Oper. Res.*, 55:329--345, 2002.
10. T. Banerjee and V. V. Veeravalli. "Data-efficient quickest change detection." arXiv:1211.3729, 2012.
11. V. Hadad, D. A. Hirshberg, R. Zhan, S. Wager, and S. Athey. "Confidence Intervals for Policy Evaluation in Adaptive Experiments." arXiv:1911.02768, 2021.

## License

Licensed under MIT or Apache-2.0 ([LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE)).
