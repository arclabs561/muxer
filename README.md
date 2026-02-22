# muxer

Deterministic, multi-objective routing primitives for piecewise-stationary multi-armed bandit problems with small action sets.

## Problem setting

Given `K` arms (model versions, inference endpoints, backends, or any discrete action set selected repeatedly), an agent observes vector-valued outcomes per call and must choose the next arm. Each outcome carries binary quality signals (`ok`, `junk`, `hard_junk`), an optional continuous quality score in `[0, 1]`, a cost proxy, and wall-clock latency. Reward distributions are **piecewise-stationary**: they may change at unknown times, and the agent must detect and adapt to these changes.

The setting is a **multi-objective** extension of the stochastic MAB: the agent simultaneously minimizes cost, latency, and degradation rate while maximizing success rate -- objectives that cannot, in general, be collapsed to a single scalar without losing information. `muxer` addresses this by computing a **Pareto frontier** over the objective vector and selecting via **linear weighted scalarization** with configurable weights and hard constraints.

Standard single-objective approaches fail in characteristic ways: pure exploitation (always-best-arm) prevents detection of regressions on non-selected arms; uniform allocation (round-robin) wastes budget on known-degraded arms; threshold-based cooldown misses gradual distribution drift that never triggers a discrete failure. An effective policy must:

- **explore** arms with high uncertainty or recent distributional changes
- **detect regressions** by maintaining sufficient sampling rate on all arms
- **remain deterministic by default** (identical statistics and configuration produce identical selections), enabling reproducible debugging

## What it is

An `Outcome` has three caller-defined quality fields plus cost and latency:

- `ok`: the call produced a usable result
- `junk`: quality fell below the caller's threshold (low F1, empty extraction, low-confidence score); `hard_junk=true` implies `junk=true`
- `hard_junk`: complete failure (error, timeout, parse failure) -- a strict subset of junk, penalized separately
- `quality_score: Option<f64>`: optional continuous quality signal in `[0, 1]` (higher is better), set via `Window::set_last_quality_score` or `Router::set_last_quality_score`. When `MabConfig::quality_weight > 0`, this is incorporated as a separate Pareto dimension and scalarization term.
- `cost_units`: caller-defined cost proxy (token count, API credits, etc.)
- `elapsed_ms`: wall-clock time

Designed for small arm counts (typically 2--10) and moderate window sizes (hundreds to low thousands of observations). The core selection algorithm:

1. Maintains a sliding window of recent `Outcome`s per arm [1, 7].
2. Computes a **Pareto frontier** over the objective vector (ok rate, junk rate, hard junk rate, mean cost, mean latency, and optionally mean quality score) using standard Pareto dominance [8].
3. Selects deterministically from the frontier via **linear weighted scalarization** with configurable weights and a stable lexicographic tie-break [8].

## Capabilities

| Category | Primitive | Description |
|---|---|---|
| **Selection** | `select_mab` / `select_mab_explain` | Deterministic Pareto + scalarization over sliding-window summaries |
| | `ThompsonSampling` | Seedable Thompson sampling [3] for scalar rewards in `[0, 1]`; optional decay |
| | `Exp3Ix` | Seedable EXP3-IX [4] for adversarial / non-stationary reward settings |
| | `LinUcb` (feature `contextual`) | Linear contextual bandit [5] with per-request feature vectors |
| | `BanditPolicy` trait (feature `stochastic`) | Common `decide`/`update_reward` interface over `ThompsonSampling` and `Exp3Ix` |
| | `softmax_map` | Stable score-to-probability conversion for traffic splitting |
| **Monitoring** | `TriageSession` | Per-arm CUSUM [6] detection with per-`(arm, context-bin)` cell investigation |
| | `CusumCatBank` | Bank of CUSUM alternatives at multiple reference levels (GLR-inspired [9] robustification) |
| | `calibrate_cusum_threshold` | Monte Carlo calibration: given a null model, find the threshold `h` such that `P[alarm within m rounds] <= alpha` |
| **Triage** | `worst_first_pick_k` | Post-detection: prioritize the most degraded arms for investigation |
| | `ContextualCoverageTracker` / `contextual_worst_first_pick_k` | Lift triage to `(arm, context-bin)` pairs for localized regressions |
| **Coverage** | `CoverageConfig` / `coverage_pick_under_sampled` | Maintenance sampling floor: keep all arms measured above a quota |
| | `ControlConfig` / `pick_control_arms` | Reserve deterministic-random picks as a selection-bias anchor |
| **Guardrails** | `LatencyGuardrailConfig` / `apply_latency_guardrail` | Hard pre-filter by mean latency with stop-early semantics |
| | Multi-pick (`select_mab_k_guardrailed_*`) | Select up to `k` unique arms per decision with per-round guardrails |
| **Orchestration** | `Router` | Stateful session: owns per-arm state, handles the full select/observe/triage lifecycle, supports dynamic arm add/remove |
| | `suggested_window_cap` | SW-UCB-derived [7] window sizing: `sqrt(throughput / change_rate)` |
| | `PipelineOrder` / `PolicyPlan` | Harness glue for composing custom routing pipelines |
| **Utilities** | `Decision` envelope | Unified decision record: `chosen` + `probs` + typed audit `notes` for logging/replay |
| | `novelty_pick_unseen` / `apply_prior_counts_to_summary` | Novelty helpers and Bayesian prior smoothing |

## Which policy to use

- **`select_mab`**: Multiple objectives (success, failure, quality, cost, latency) with deterministic selection. Under stationarity with UCB exploration, achieves `O(sqrt(K T ln T))` pseudo-regret on the scalarized objective [2]. Set `MabConfig::quality_weight > 0` to incorporate continuous quality.
- **`ThompsonSampling`**: Single scalar reward in `[0, 1]`, Bayesian exploration. Achieves `O(sqrt(K T ln T))` Bayes regret under independent arms [3]. Seedable; optional geometric decay for non-stationarity.
- **`Exp3Ix`**: Adversarial or rapidly non-stationary rewards. Achieves `O(sqrt(K T ln K))` expected regret against an oblivious adversary [4]. Seedable; optional decay.
- **`LinUcb`** (feature `contextual`): Per-request feature vector with `O(d sqrt(T ln T))` regret under the linear realizability assumption [5].
- **`BanditPolicy` trait** (feature `stochastic`): Generic interface -- `fn run<P: BanditPolicy>(p: &mut P)` works with both `ThompsonSampling` and `Exp3Ix`.

## Routing lifecycle

A deployment typically operates in three modes:

1. **Normal** (`select_mab` / `ThompsonSampling` / `Exp3Ix`): select the best arm while exploring. Runs on every call.
2. **Regression investigation** (`worst_first_pick_k`): after CUSUM [6] fires on an arm, route additional traffic to characterize the distributional change. `TriageSession` automates the detect-to-investigate handoff.
3. **Control** (`pick_control_arms`): reserve a small fraction of calls as a uniform-random baseline to anchor quality estimates and detect selection bias.

`CoverageConfig` bridges modes 1 and 3: it enforces a minimum per-arm sampling rate, preventing starvation that would delay regression detection.

## Three goals for sampling

Every routing decision simultaneously serves three competing objectives:

1. **Exploitation** -- minimize cumulative regret by routing to the best arm.
2. **Estimation** -- reduce uncertainty about each arm's current performance.
3. **Detection** -- minimize the delay between a distributional change and the alarm.

**The two clocks.** An arm is only observed when selected, so two time scales govern behavior:

- **Wall time** `t`: global decision steps (total calls across all arms).
- **Sample time** `n_k`: observations accumulated for arm `k`.

Detection delay in wall time scales as `delay_wall ~ delay_samples / rate_k`, where `rate_k = n_k / t` is the sampling rate of arm `k`. `CoverageConfig` sets a floor on `rate_k`, which is the direct lever for bounding wall-clock detection delay.

**The non-contextual collapse.** Under a fixed (non-adaptive) allocation with `K` arms: estimation error (MSE) and average CUSUM [6] detection delay are both `O(1/n_k)` in the per-arm sample count. Their sensitivity functions are scalar multiples -- both depend solely on "how many observations at this arm." The three-way Pareto surface collapses to a one-dimensional curve parameterized by `n_k`. This yields the product identity:

```
R_T * D_avg = C * Delta * T / delta^2
```

where `R_T` is cumulative regret, `D_avg` is average detection delay, `Delta` is the suboptimality gap, `delta` is the changepoint magnitude, and `C` is a constant depending on the noise variance and significance level. Implication: you cannot independently improve both regret and detection delay without increasing the total sample budget.

**The contextual revival.** With contextual routing (`LinUcb`), the allocation depends on a per-request feature vector. Average detection delay remains proportional to estimation error (both share the same design-measure sensitivity). However, **worst-case detection delay** -- concentrated on the covariate cell with the fewest observations -- is genuinely independent. This is why `ContextualCoverageTracker` and `TriageSession` exist: localized regressions in sparse covariate regions require explicit cell-level coverage and triage, not just arm-level monitoring.

For derivations and concrete failure modes, see the [API docs](https://docs.rs/muxer) and [examples/EXPERIMENTS.md](examples/EXPERIMENTS.md).

## Positioning

`muxer` occupies a specific niche: **deterministic multi-objective selection with integrated changepoint detection**, in Rust, for small `K`. It is not a general-purpose bandit platform (no storage layer, no off-policy evaluation pipeline, no dashboard).

Compared to broader libraries: MABWiser [10] provides a wide policy catalog with parallelized training in Python; Vowpal Wabbit provides industrial-strength contextual bandits with cost-sensitive reductions; River provides streaming ML including bandits as one module among many. `muxer` trades breadth for depth in the multi-objective + monitoring + triage space, with an emphasis on determinism and auditability.

## Quick examples

### Start here -- `getting_started` (3 arms, minimal setup)

```bash
cargo run --example getting_started
```

Covers the most common use case in ~60 lines: create a `Router`, run a select/observe loop, use delayed quality labeling (`set_last_junk_level` + `set_last_quality_score`), and observe how `quality_weight` disambiguates arms that binary rates cannot distinguish.

### Deterministic multi-objective selection (Pareto + scalarization)

```rust
use muxer::{select_mab, MabConfig, Summary};
use std::collections::BTreeMap;

let arms = vec!["a".to_string(), "b".to_string()];
let mut summaries = BTreeMap::new();
// arm "a": 9/10 ok, 0 junk (quality above threshold)
summaries.insert("a".to_string(), Summary { calls: 10, ok: 9, junk: 0, hard_junk: 0, cost_units: 10, elapsed_ms_sum: 900, mean_quality_score: None });
// arm "b": same ok rate, but 2 results fell below quality threshold
summaries.insert("b".to_string(), Summary { calls: 10, ok: 9, junk: 2, hard_junk: 0, cost_units: 10, elapsed_ms_sum: 900, mean_quality_score: None });

let sel = select_mab(&arms, &summaries, MabConfig::default());
assert_eq!(sel.chosen, "a"); // lower junk rate wins when all else is equal
```

### Online routing loop (Window ingestion)

Maintain a `Window` per arm, push `Outcome`s as requests complete, call `select_mab` on each decision.

```bash
cargo run --example deterministic_router
```

Note: this example simulates an environment and requires `--features stochastic` if default features are disabled.

### Monitored selection (baseline vs recent drift + uncertainty-aware rates)

Maintain a baseline and recent window per arm for change monitoring via `MonitoredWindow` + `select_mab_monitored_*`:

```bash
cargo run --example monitored_router --features stochastic
```

### End-to-end router demo (Window + constraints + stickiness + delayed junk)

Combines multiple production patterns: window ingestion, constraints+weights, stickiness, and delayed junk labeling.

```bash
cargo run --example end_to_end_router --features stochastic
```

CI-checked regression test in `tests/e2e_metrics.rs`.

### Window ingestion with delayed quality labeling

In practice, quality is determined after processing: call the arm, receive a response, then score it (compute F1, run a parser, check embedding similarity). Pattern: push the `Outcome` immediately with `junk: false`, then call `set_last_junk_level` once scoring completes.

```bash
cargo run --example window_delayed_junk_label
```

### Constraint + trade-off tuning for `select_mab`

Demonstrates "constraints first, then weights" configuration:

```bash
cargo run --example mab_constraints_tuning
```

### Router -- production pattern (monitoring + triage + calibration + snapshot)

```bash
cargo run --example router_production --features stochastic
```

Demonstrates: CUSUM threshold calibration, full `RouterConfig` with monitoring/triage/coverage/control, regression detection, acknowledgment, and snapshot/restore for persistence.

### Router -- full lifecycle (select / observe / triage / acknowledge)

```bash
cargo run --example router_quickstart
```

Covers: two-arm routing, quality divergence, triage detection, acknowledgment, large-K batch exploration (K=20 in 7 rounds with k=3), and dynamic arm management.

### Multi-pick selection with a latency guardrail

`select_mab_k_guardrailed_explain_full` selects up to `k` unique arms per decision, applying a `LatencyGuardrailConfig` each round. Guardrail semantics (soft pre-filter vs hard constraint):

```bash
cargo run --example guardrail_semantics
```

### Unified decision records

```bash
cargo run --example decision_unified
```

### Detect-then-triage (`TriageSession`)

`TriageSession` combines per-arm CUSUM detection [6] with per-`(arm, context-bin)` cell investigation:

```rust
use muxer::{TriageSession, TriageSessionConfig, OutcomeIdx};

let arms = vec!["a".to_string(), "b".to_string()];
let mut session = TriageSession::new(&arms, TriageSessionConfig::default()).unwrap();

// Feed observations: arm name, outcome category, feature context.
session.observe("a", OutcomeIdx::OK, &[0.2, 0.3]);
session.observe("b", OutcomeIdx::HARD_JUNK, &[0.8, 0.9]);

// Which arms have CUSUM-alarmed?
let alarmed = session.alarmed_arms();

// Top (arm, bin) cells to route extra investigation traffic to.
let bins = session.tracker().active_bins();
let cells = session.top_alarmed_cells(&bins, 3);
```

`OutcomeIdx::from_outcome(ok, junk, hard_junk)` maps an `Outcome` triple to the 4-category index space (`OK` / `SOFT_JUNK` / `HARD_JUNK` / `FAIL`). For non-contextual triage, use `worst_first_pick_k`.

### EXP3-IX (adversarial bandit) with probabilities

```rust
use muxer::{Exp3Ix, Exp3IxConfig};

let arms = vec!["a".to_string(), "b".to_string(), "c".to_string()];
let mut ex = Exp3Ix::new(Exp3IxConfig { seed: 123, decay: 0.98, ..Exp3IxConfig::default() });

let d = ex.decide(&arms).unwrap();
// ... run request with `d.chosen` ...
ex.update_reward(&d.chosen, 0.7); // reward in [0, 1]

let probs = d.probs.unwrap();
let s: f64 = probs.values().sum();
assert!((s - 1.0).abs() < 1e-9);
```

```bash
cargo run --example exp3ix_router
```

Note: requires `--features stochastic` if default features are disabled.

### Thompson "traffic splitting" selector (mean-softmax allocation)

```rust
use muxer::{ThompsonConfig, ThompsonSampling};

let arms = vec!["a".to_string(), "b".to_string()];
let mut ts = ThompsonSampling::with_seed(
    ThompsonConfig {
        decay: 0.99,
        ..ThompsonConfig::default()
    },
    0,
);
let d = ts.decide_softmax_mean(&arms, 0.3).unwrap();
ts.update_reward(&d.chosen, 1.0);

let alloc = d.probs.unwrap();
let s: f64 = alloc.values().sum();
assert!((s - 1.0).abs() < 1e-9);
```

```bash
cargo run --example thompson_router
```

Note: requires `--features stochastic` if default features are disabled.

### Contextual routing (LinUCB)

```bash
cargo run --example contextual_router --features contextual
```

Notes:

- For a probability distribution over arms (traffic splitting or propensity logging), use `LinUcb::probabilities(...)` or `LinUcb::decide_softmax_ucb(...)`.
- Algorithm: LinUCB [5].

Contextual propensity logging:

```bash
cargo run --example contextual_propensity_logging --features contextual
```

### Stickiness / switching-cost control

To reduce arm-switching frequency, wrap deterministic selection with `StickyMab`:

```bash
cargo run --example sticky_mab_router
```

### Mini-experiments (bandits, monitoring, false alarms)

Runnable probes that make trade-offs and failure modes explicit -- see [examples/EXPERIMENTS.md](examples/EXPERIMENTS.md) for guided walkthroughs:

- `cargo run --example guardrail_semantics`
- `cargo run --example coverage_autotune --features stochastic`
- `cargo run --example free_lunch_investigation --features stochastic`
- `cargo run --example detector_inertia --features stochastic`
- `cargo run --example detector_calibration --features stochastic`
- `cargo run --example bqcd_sampling --features stochastic`
- `cargo run --release --example bqcd_calibrated --features stochastic`

Reusable components from these experiments live in `muxer::monitor`:

- `CusumCatBank`: bank of CUSUM alternatives at multiple reference levels (GLR-inspired [9] robustification).
- `calibrate_threshold_from_max_scores`: threshold calibration from null max-score samples (Wilson-conservative mode available).

## Documentation

- [**Quickstart** (5-minute guide)](docs/QUICKSTART.md)
- [Docs index](docs/README.md)
- [API docs (docs.rs)](https://docs.rs/muxer)
- [Changelog](CHANGELOG.md)
- [Mini-experiments / research probes](examples/EXPERIMENTS.md)

## Quickstart (Router)

The `Router` struct owns all per-arm state and handles the full lifecycle in three calls:

```rust
use muxer::{Router, RouterConfig, Outcome};

let arms = vec!["backend-a".to_string(), "backend-b".to_string()];
let mut router = Router::new(arms, RouterConfig::default()).unwrap();

loop {
    let d = router.select(1, 0);             // pick an arm (seed for tie-breaking)
    let arm = d.primary().unwrap().to_string();

    // ... make the call, evaluate quality ...
    let outcome = Outcome { ok: true, junk: false, hard_junk: false,
                            cost_units: 5, elapsed_ms: 120, quality_score: None };
    router.observe(&arm, outcome);           // record result

    // If quality degrades, triage mode fires automatically:
    if router.mode().is_triage() {
        // d.triage_cells contains (arm, context-bin) cells to investigate
        router.acknowledge_change(&arm);     // reset after investigation
    }
}
```

For larger arm counts, pass `k > 1` to batch the initial exploration:

```rust
// K=30 arms, k=3 per round -> initial coverage in ~10 rounds.
let cfg = RouterConfig::default().with_coverage(0.02, 1);
let d = router.select(3, 0);
```

## Usage

```toml
[dependencies]
muxer = "0.3.9"
```

Deterministic core only (no stochastic bandits):

```toml
[dependencies]
muxer = { version = "0.3.9", default-features = false }
```

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
2. P. Auer, N. Cesa-Bianchi, and P. Fischer. UCB1 algorithm and variants. Regret bound: `O(sqrt(K T ln T))` for the tuned variant. See [1].
3. S. Agrawal and N. Goyal. "Analysis of Thompson Sampling for the Multi-armed Bandit Problem." *COLT*, 2012. (Original formulation: W. R. Thompson, 1933.)
4. P. Auer, N. Cesa-Bianchi, Y. Freund, and R. E. Schapire. "The Nonstochastic Multiarmed Bandit Problem." *SIAM J. Comput.*, 32(1):48--77, 2002. (EXP3; the IX variant adds implicit exploration.)
5. W. Chu, L. Li, L. Reyzin, and R. Schapire. "Contextual Bandits with Linear Payoff Functions." *AISTATS*, 2011. (Also: L. Li, W. Chu, J. Langford, R. E. Schapire. "A contextual-bandit approach to personalized news article recommendation." *WWW*, 2010.)
6. E. S. Page. "Continuous inspection schemes." *Biometrika*, 41(1-2):100--115, 1954. (CUSUM. For the piecewise-stationary MAB extension, see: Y. Cao, Z. Wen, B. Kveton, Y. Xie. "Nearly optimal adaptive procedure with change detection for piecewise-stationary bandit." *AISTATS*, 2019.)
7. A. Garivier and E. Moulines. "On Upper-Confidence Bound Policies for Switching Bandit Problems." *ALT*, 2011. (Sliding-window UCB; `muxer` derives `suggested_window_cap` from their Theorem 1.)
8. R. R. Drugan and A. Nowe. "Designing multi-objective multi-armed bandits algorithms: A study." *IJCNN*, 2013. (Pareto UCB1 and scalarized multi-objective MAB.)
9. L. Besson, E. Kaufmann, O.-A. Maillard, and J. Seznec. "Efficient Change-Point Detection for Tackling Piecewise-Stationary Bandits." 2019. arXiv:1902.01575. (GLR-klUCB; `CusumCatBank` is inspired by the multi-alternative GLR approach.)
10. E. Strong, B. Kleynhans, and S. Kadioglu. "MABWiser: Parallelizable Contextual Multi-Armed Bandits." *IJAIT*, 30(4), 2021.

## Citation

```bibtex
@software{muxer,
  author  = {Arc Labs},
  title   = {muxer: Deterministic multi-objective routing for piecewise-stationary bandits},
  url     = {https://github.com/arclabs561/muxer},
  version = {0.3.9},
  year    = {2025}
}
```

## License

See [LICENSE](LICENSE).
