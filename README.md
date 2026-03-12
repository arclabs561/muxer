# muxer

[![crates.io](https://img.shields.io/crates/v/muxer.svg)](https://crates.io/crates/muxer)
[![Documentation](https://docs.rs/muxer/badge.svg)](https://docs.rs/muxer)
[![CI](https://github.com/arclabs561/muxer/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/muxer/actions/workflows/ci.yml)

Deterministic, multi-objective routing for piecewise-stationary multi-armed bandit problems.

## Problem setting

Given `K` arms (model versions, inference endpoints, backends -- any discrete action set selected repeatedly), an agent observes vector-valued outcomes per call and must choose the next arm. Each outcome carries binary quality signals (`ok`, `junk`, `hard_junk`), an optional continuous quality score in `[0, 1]`, a cost proxy, and wall-clock latency. Reward distributions are **piecewise-stationary**: they may change at unknown times, and the agent must detect and adapt.

The setting is a **multi-objective** extension of the stochastic MAB [1]: the agent simultaneously minimizes cost, latency, and degradation rate while maximizing success rate. These objectives cannot in general be collapsed to a single scalar without losing information. The core selection algorithm:

1. Maintains a sliding window of recent `Outcome`s per arm [6].
2. Computes a **Pareto frontier** over the objective vector using standard Pareto dominance [7].
3. Selects deterministically from the frontier via **linear weighted scalarization** with configurable weights and a stable lexicographic tie-break.

Standard single-objective approaches fail in characteristic ways: pure exploitation prevents detection of regressions on non-selected arms; uniform allocation wastes budget on known-degraded arms; threshold-based cooldown misses gradual distribution drift. An effective policy must:

- **explore** arms with high uncertainty or recent distributional changes
- **detect regressions** by maintaining sufficient sampling rate on all arms
- **remain deterministic by default** (identical statistics and configuration produce identical selections)

## Outcome fields

- `ok`: the call produced a usable result
- `junk`: quality fell below the caller's threshold; `hard_junk=true` implies `junk=true`
- `hard_junk`: complete failure (error, timeout, parse failure) -- strict subset of junk, penalized separately
- `quality_score: Option<f64>`: continuous quality signal in `[0, 1]`. When `MabConfig::quality_weight > 0`, incorporated as a separate Pareto dimension.
- `cost_units`: caller-defined cost proxy
- `elapsed_ms`: wall-clock time

Designed for 2--10 arms and moderate window sizes (hundreds to low thousands of observations).

## Capabilities

| Category | Primitive | Description |
|---|---|---|
| **Selection** | `select_mab` / `select_mab_explain` | Deterministic Pareto + scalarization over sliding-window summaries |
| | `ThompsonSampling` | Seedable Thompson sampling [2] for scalar rewards in `[0, 1]`; optional decay |
| | `Exp3Ix` | Seedable EXP3-IX [3] for adversarial / non-stationary rewards |
| | `LinUcb` (feature `contextual`) | Linear contextual bandit [4] with per-request feature vectors |
| | `BanditPolicy` trait (feature `stochastic`) | Common `decide`/`update_reward` interface over `ThompsonSampling` and `Exp3Ix` |
| | `softmax_map` | Score-to-probability conversion for traffic splitting |
| **Monitoring** | `TriageSession` | Per-arm CUSUM [5] detection with per-`(arm, context-bin)` cell investigation |
| | `CusumCatBank` | Bank of CUSUM alternatives at multiple reference levels (GLR-inspired [8] robustification) |
| | `calibrate_cusum_threshold` | Monte Carlo: find threshold `h` such that `P[alarm within m rounds] <= alpha` |
| **Triage** | `worst_first_pick_k` | Post-detection: prioritize the most degraded arms for investigation |
| | `ContextualCoverageTracker` | Lift triage to `(arm, context-bin)` pairs for localized regressions |
| **Coverage** | `CoverageConfig` | Maintenance sampling floor: keep all arms measured above a quota |
| | `ControlConfig` / `pick_control_arms` | Reserve deterministic-random picks as a selection-bias anchor |
| **Guardrails** | `LatencyGuardrailConfig` | Hard pre-filter by mean latency with stop-early semantics |
| **Orchestration** | `Router` | Stateful session: owns per-arm state, full select/observe/triage lifecycle, dynamic arm add/remove |
| | `suggested_window_cap` | SW-UCB-derived [6] window sizing: `sqrt(throughput / change_rate)` |
| | `PipelineOrder` / `PolicyPlan` | Harness glue for composing custom routing pipelines |
| **Utilities** | `Decision` envelope | Unified decision record (`chosen` + `probs` + typed audit `notes`) for logging/replay |
| | `novelty_pick_unseen` / `apply_prior_counts_to_summary` | Novelty helpers and prior smoothing |

## Which policy to use

- **`select_mab`**: Multiple objectives with deterministic selection. `O(sqrt(K T ln T))` pseudo-regret on the scalarized objective under stationarity [1].
- **`ThompsonSampling`**: Single scalar reward in `[0, 1]`. `O(sqrt(K T ln T))` Bayes regret [2]. Seedable; optional decay.
- **`Exp3Ix`**: Adversarial rewards. `O(sqrt(K T ln K))` expected regret [3]. Seedable; optional decay.
- **`LinUcb`** (feature `contextual`): Per-request feature vector. `O(d sqrt(T ln T))` regret under linear realizability [4].
- **`BanditPolicy` trait** (feature `stochastic`): Generic `fn run<P: BanditPolicy>(p: &mut P)` over both stochastic policies.

## Routing lifecycle

1. **Normal** (`select_mab` / `ThompsonSampling` / `Exp3Ix`): select the best arm while exploring.
2. **Regression investigation** (`worst_first_pick_k`): after CUSUM [5] fires, route additional traffic to characterize the change. `TriageSession` automates the handoff.
3. **Control** (`pick_control_arms`): uniform-random baseline to anchor quality estimates and detect selection bias.

`CoverageConfig` bridges 1 and 3: minimum per-arm sampling rate prevents starvation that would delay regression detection.

## Three goals for sampling

Every routing decision serves three competing objectives:

1. **Exploitation** -- minimize cumulative regret.
2. **Estimation** -- reduce uncertainty about each arm's current performance.
3. **Detection** -- minimize delay between a distributional change and the alarm.

**The two clocks.** An arm is only observed when selected:

- **Wall time** `t`: global decision steps.
- **Sample time** `n_k`: observations for arm `k`.

Detection delay in wall time: `delay_wall ~ delay_samples / rate_k`. `CoverageConfig` sets a floor on `rate_k`.

**The non-contextual collapse.** Under fixed allocation, MSE and average CUSUM detection delay are both `O(1/n_k)`. Their sensitivity functions are scalar multiples. The three-way Pareto surface collapses to a 1-D curve parameterized by `n_k`:

```
R_T * D_avg = C * Delta * T / delta^2
```

You cannot independently improve both regret and detection delay without increasing the total sample budget.

**The contextual revival.** With `LinUcb`, average detection delay stays proportional to estimation error. But **worst-case detection delay** -- concentrated on the covariate cell with fewest observations -- is genuinely independent. This is why `ContextualCoverageTracker` and `TriageSession` exist: localized regressions in sparse covariate regions need cell-level triage.

See the [API docs](https://docs.rs/muxer) and [examples/EXPERIMENTS.md](examples/EXPERIMENTS.md) for derivations and failure modes.

## Usage

```toml
[dependencies]
muxer = "0.3.13"
```

Deterministic core only (no stochastic bandits):

```toml
[dependencies]
muxer = { version = "0.3.11", default-features = false }
```

## Quickstart

```rust
use muxer::{Router, RouterConfig, Outcome};

let arms = vec!["backend-a".to_string(), "backend-b".to_string()];
let mut router = Router::new(arms, RouterConfig::default()).unwrap();

loop {
    let d = router.select(1, 0);
    let arm = d.primary().unwrap().to_string();

    let outcome = Outcome { ok: true, junk: false, hard_junk: false,
                            cost_units: 5, elapsed_ms: 120, quality_score: None };
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

#### Getting started

[**getting_started.rs**](examples/getting_started.rs) -- Minimal 3-backend routing loop in 60 lines. Shows the core lifecycle: select an arm, observe an outcome with delayed quality labeling, let the router adapt. Start here.

[**router_quickstart.rs**](examples/router_quickstart.rs) -- Full routing lifecycle: two arms diverge in quality, CUSUM fires a triage alarm, the operator acknowledges and the router recovers. Also demonstrates large-K batch exploration.

[**router_production.rs**](examples/router_production.rs) -- Production configuration. CUSUM threshold calibration, monitoring windows, coverage enforcement, guardrails, control arms, and snapshot/restore for process restarts.

```bash
cargo run --example getting_started
cargo run --example router_quickstart
cargo run --example router_production --features stochastic
```

#### Algorithm variants

[**deterministic_router.rs**](examples/deterministic_router.rs) -- Multi-objective MAB with hard constraints on junk rate and cost/latency weights. Deterministic tie-breaking for reproducible selections.

[**thompson_router.rs**](examples/thompson_router.rs) -- Thompson sampling with decay for non-stationary binary rewards. Two arms with drifting success probabilities.

[**exp3ix_router.rs**](examples/exp3ix_router.rs) -- Exp3-IX for adversarial settings where reward distributions change over time.

[**contextual_router.rs**](examples/contextual_router.rs) -- LinUCB with 2D context features (prompt length, difficulty). Arm performance depends on input characteristics.

[**sticky_mab_router.rs**](examples/sticky_mab_router.rs) -- Dwell-time and margin-based switching. Prevents backend thrashing when arm estimates are noisy.

[**monitored_router.rs**](examples/monitored_router.rs) -- Drift detection via Hellinger distance between baseline and recent observation windows. Maintenance sampling prevents lock-on.

```bash
cargo run --example deterministic_router
cargo run --example thompson_router
cargo run --example exp3ix_router
cargo run --example contextual_router --features contextual
cargo run --example sticky_mab_router
cargo run --example monitored_router --features stochastic
cargo run --example end_to_end_router --features stochastic
```

#### Configuration patterns

[**end_to_end_router.rs**](examples/end_to_end_router.rs) -- Delayed quality labeling: push outcomes with unknown junk status, update later after validation. Combines stickiness and constraints.

[**mab_constraints_tuning.rs**](examples/mab_constraints_tuning.rs) -- Constraints-first design. Hard-filter arms exceeding 10% junk, then tune cost/latency weights among the clean set.

[**coverage_autotune.rs**](examples/coverage_autotune.rs) -- Derive the coverage sampling rate from KL divergence and CUSUM theory to guarantee regression detection within a wall-time budget.

[**contextual_propensity_logging.rs**](examples/contextual_propensity_logging.rs) -- Extract per-arm selection probabilities from LinUCB decisions for offline inverse-propensity-weighted analysis.

[**guardrail_semantics.rs**](examples/guardrail_semantics.rs) -- Soft (novelty before guardrail) vs hard (guardrail-first) policy ordering. Shows how the order of filtering stages affects which arms get trialed.

[**decision_unified.rs**](examples/decision_unified.rs) -- The unified `Decision` type with stickiness. Shows how dwell-time prevents thrashing when the base policy flaps between arms.

[**window_delayed_junk_label.rs**](examples/window_delayed_junk_label.rs) -- Minimal pattern for deferred quality labeling: push an outcome with `junk=false`, then update to `true` after downstream validation.

```bash
cargo run --example mab_constraints_tuning
cargo run --example coverage_autotune
cargo run --example contextual_propensity_logging --features contextual
cargo run --example guardrail_semantics
cargo run --example decision_unified
cargo run --example window_delayed_junk_label
```

#### Detection and calibration

[**detector_calibration.rs**](examples/detector_calibration.rs) -- Monte Carlo threshold calibration: find the CUSUM threshold `h` such that `P[false alarm within m rounds] <= alpha`.

[**detector_inertia.rs**](examples/detector_inertia.rs) -- Cumulative vs forgetful detectors. CatKL accumulates history and loses sensitivity over time; CUSUM forgets and stays responsive. Shows the effect of pre-change baseline length on detection delay.

[**bqcd_sampling.rs**](examples/bqcd_sampling.rs) -- Partial-observation multi-armed detection. K arms are only observed when sampled, so the policy must balance exploration (find the changed arm), exploitation (focus on it), and coverage (don't starve others).

[**bqcd_calibrated.rs**](examples/bqcd_calibrated.rs) -- Global false-alarm calibration across arms with a CUSUM bank of alternative hypotheses.

```bash
cargo run --example detector_calibration
cargo run --example detector_inertia
cargo run --example bqcd_sampling
cargo run --example bqcd_calibrated
```

#### Domain harnesses

Each harness simulates a realistic routing scenario with multi-dimensional slicing (task x dataset x environment), per-slice arm eligibility, and injected drift at a known epoch.

[**matrix_harness.rs**](examples/matrix_harness.rs) -- NLP evaluation: 5 task-dataset pairs (NER, relation extraction, coreference). A BERT-ONNX model drifts on social-media NER at epoch 48.

[**pcap_triage_harness.rs**](examples/pcap_triage_harness.rs) -- Network security: route packet-analysis backends (Suricata, Zeek, ML) per protocol/dataset. TLS evasion causes drift.

[**ad_auction_harness.rs**](examples/ad_auction_harness.rs) -- Ad ranking: route auction models per objective/geo/device. A privacy policy change degrades the deep model on EU mobile traffic.

[**fraud_scoring_harness.rs**](examples/fraud_scoring_harness.rs) -- Payments fraud: route fraud scorers per channel/region/segment. A rules model drifts on EU card-not-present after a fraud pattern shift.

[**medical_triage_harness.rs**](examples/medical_triage_harness.rs) -- Clinical risk scoring: route triage models by setting/acuity/cohort. A protocol change breaks high-acuity pediatric ED scoring.

[**search_ranking_harness.rs**](examples/search_ranking_harness.rs) -- Search ranking: route rankers by intent/locale/device. BM25 regresses on informational Spanish queries after a corpus shift.

[**synthetic_drift_harness.rs**](examples/synthetic_drift_harness.rs) -- Controlled experiment with known drift injection point. Validates that the router detects and adapts correctly.

```bash
cargo run --example matrix_harness
cargo run --example pcap_triage_harness
cargo run --example ad_auction_harness
cargo run --example fraud_scoring_harness
cargo run --example medical_triage_harness
cargo run --example search_ranking_harness
cargo run --example synthetic_drift_harness
```

#### Investigation

[**free_lunch_investigation.rs**](examples/free_lunch_investigation.rs) -- When does change detection require coverage? Compares Thompson sampling, deterministic MAB, and MAB+coverage when one arm drifts. Shows that passive monitoring fails if the changed arm gets starved of observations.

```bash
cargo run --example free_lunch_investigation
```

Mini-experiments (trade-offs and failure modes): [examples/EXPERIMENTS.md](examples/EXPERIMENTS.md).

## Documentation

- [Quickstart (5-minute guide)](docs/QUICKSTART.md)
- [Docs index](docs/README.md)
- [API docs](https://docs.rs/muxer)
- [Changelog](CHANGELOG.md)
- [Mini-experiments](examples/EXPERIMENTS.md)

## Development

```bash
cargo test -p muxer
cargo bench -p muxer --bench coverage
cargo bench -p muxer --bench monitor
cargo fmt -p muxer --check
cargo clippy -p muxer --all-targets -- -D warnings
```

## References

1. P. Auer, N. Cesa-Bianchi, and P. Fischer. "Finite-time analysis of the multiarmed bandit problem." *Machine Learning*, 47(2-3):235--256, 2002.
2. S. Agrawal and N. Goyal. "Analysis of Thompson Sampling for the Multi-armed Bandit Problem." *COLT*, 2012.
3. P. Auer, N. Cesa-Bianchi, Y. Freund, and R. E. Schapire. "The Nonstochastic Multiarmed Bandit Problem." *SIAM J. Comput.*, 32(1):48--77, 2002.
4. W. Chu, L. Li, L. Reyzin, and R. Schapire. "Contextual Bandits with Linear Payoff Functions." *AISTATS*, 2011.
5. E. S. Page. "Continuous inspection schemes." *Biometrika*, 41(1-2):100--115, 1954.
6. A. Garivier and E. Moulines. "On Upper-Confidence Bound Policies for Switching Bandit Problems." *ALT*, 2011.
7. R. R. Drugan and A. Nowe. "Designing multi-objective multi-armed bandits algorithms: A study." *IJCNN*, 2013.
8. L. Besson, E. Kaufmann, O.-A. Maillard, and J. Seznec. "Efficient Change-Point Detection for Tackling Piecewise-Stationary Bandits." arXiv:1902.01575, 2019.

## License

Licensed under MIT or Apache-2.0 ([LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE)).
