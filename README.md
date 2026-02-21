# muxer

Deterministic, multi-objective routing primitives for "provider selection" problems.

## What problem this solves

You have a small set of arms (model versions, inference endpoints, backends, data sources — anything you choose between repeatedly) and calls where you evaluate quality after the fact. After each call you label the result: did it succeed? was the output good enough? was it completely broken? You define those thresholds; muxer tracks the rates and routes future calls accordingly.

Naive approaches fall short in predictable ways: always-best-arm starves new and recovering arms so regressions go undetected; round-robin wastes calls on arms you know are degraded; cooldown-on-failure misses slow quality drift that never triggers a hard error. You want an **online policy** that:

- **explores** new or recently-changed arms
- **avoids regressions** (routes away from arms with rising failure or quality-degradation rates)
- is **deterministic by default** (same stats/config → same choice), so it's easy to debug

## What it is

An `Outcome` has three caller-defined quality fields plus cost and latency:

- `ok`: the call produced a usable result
- `junk`: quality was below your threshold — low F1, empty extraction, low-confidence score. Also set when `hard_junk=true` (hard failure is a subset of junk, tracked and penalized separately)
- `hard_junk`: the call failed entirely (error, timeout, parse failure) — implies `junk=true`
- `quality_score: Option<f64>`: optional continuous quality signal `[0, 1]` (higher = better). Set after scoring via `Window::set_last_quality_score` or `Router::set_last_quality_score`. When `MabConfig::quality_weight > 0`, this gradient signal influences arm selection alongside the binary rates.
- `cost_units`: caller-defined cost proxy (token count, API credits, examples processed, etc.)
- `elapsed_ms`: wall-clock time

The framework is designed for small arm counts (typically 2–10) and moderate window sizes (hundreds to low thousands of observations). The core selection idea is:

- maintain a small sliding window of recent `Outcome`s per arm
- compute a Pareto frontier over ok rate, junk rate, cost, and latency
- pick deterministically via scalarization + stable tie-break

This crate also includes:

- a **seedable Thompson-sampling** policy (`ThompsonSampling`) for cases where you can provide a scalar reward in `[0, 1]` per call
- a **seedable EXP3-IX** policy (`Exp3Ix`) for more adversarial / fast-shifting reward settings
- (feature `contextual`) a **linear contextual bandit** policy (`LinUcb`) for per-request routing with feature vectors
- **latency guardrails** (`LatencyGuardrailConfig` / `apply_latency_guardrail`) — hard pre-filter by mean latency, with stop-early semantics for multi-pick loops
- **multi-pick selection** (`select_mab_k_guardrailed_explain_full` and variants) — select up to `k` unique arms per decision with a per-round guardrail loop
- **maintenance sampling** (`CoverageConfig` / `coverage_pick_under_sampled`) — ensure all arms stay measured above a quota; see [Three goals for sampling](#three-goals-for-sampling) for why this matters
- **post-detection triage** (`WorstFirstConfig` / `worst_first_pick_k`) — prioritize the most degraded arms for investigation after monitoring fires
- **contextual cell triage** (`ContextualCoverageTracker` / `contextual_worst_first_pick_k`) — lift triage to `(arm, context-bin)` pairs so localised regressions don't average away
- **combined detect + triage sessions** (`TriageSession`) — wires per-arm CUSUM detection and per-cell investigation into one stateful session
- **`softmax_map`** — stable score → probability helper for traffic splitting
- **`Router`** — stateful session that owns all per-arm state and handles the full lifecycle (`select` / `observe` / `acknowledge_change`); supports dynamic arm add/remove and large K
- **threshold calibration** (`calibrate_cusum_threshold`) — Monte Carlo calibration: "what CUSUM threshold gives `P[alarm within m rounds] ≤ α`?"
- **window size guidance** (`suggested_window_cap`) — SW-UCB–derived: `sqrt(throughput / change_rate)`
- **control arms** (`ControlConfig` / `pick_control_arms`) — reserve deterministic-random picks as a selection-bias anchor
- **`BanditPolicy` trait** — common `decide(arms) → Decision` + `update_reward(arm, reward)` interface that both `ThompsonSampling` and `Exp3Ix` implement, enabling generic routing harnesses
- **novelty helpers** (`novelty_pick_unseen`), **prior smoothing** (`apply_prior_counts_to_summary`), and **pipeline glue** (`PipelineOrder` / `PolicyPlan`) for building custom routing harnesses

## Which policy should I use?

- **`select_mab` (Window + Pareto + scalarization)**: when you care about **multiple objectives** at once (success rate, failure rate, quality degradation, cost, latency) and want deterministic selection with hard constraints.  Set `MabConfig::quality_weight > 0` to incorporate the continuous `quality_score` gradient into the objective.
- **`ThompsonSampling`**: when you can provide a **single reward** per call (in `[0, 1]`) and want a classic explore/exploit policy (seedable, optionally decayed).
- **`Exp3Ix`**: when reward is **non-stationary / adversarial-ish** and you still want a probabilistic policy (seedable, optionally decayed).
- **`LinUcb` (feature `contextual`)**: when you have a per-request feature vector (e.g. cheap "difficulty" features, embeddings, metadata) and want a contextual policy.
- **`BanditPolicy` trait** (feature `stochastic`): when you want to write code that works with both `ThompsonSampling` and `Exp3Ix` without committing to one — `fn run<P: BanditPolicy>(p: &mut P)` covers both.

## Routing lifecycle

A typical deployment has three modes:

1. **Normal** (`select_mab` / `ThompsonSampling` / `Exp3Ix`): route to the best arm while exploring. This runs on every call.
2. **Regression investigation** (`worst_first_pick_k`): after monitoring fires on an arm, route extra traffic there to characterize the change. `TriageSession` automates the detect → investigate handoff.
3. **Control** (`pick_random_subset`): reserve a small fraction of calls as a random baseline to anchor quality estimates and detect selection bias.

`CoverageConfig` provides a floor that bridges modes 1 and 3: it ensures no arm is so starved that you'd miss a regression in it.

## Three goals for sampling

Every routing decision involves three objectives that generally compete:

1. **Exploitation** — minimize regret; route to the best arm now.
2. **Estimation** — understand each arm's true rates; keep all arms measured.
3. **Detection** — notice when an arm changes; minimize delay between the change and the alarm.

**The two clocks.** You only observe an arm when you sample it, so there are two notions of time:

- **Wall time** $t$: global decision steps.
- **Sample time** $n_k$: observations from arm $k$.

Detection delay in wall time scales as `delay_wall ≈ delay_samples / rate_k`. `CoverageConfig` sets a minimum sampling-rate floor, which is the direct lever for bounding wall-clock detection delay.

**The non-contextual collapse.** In the non-contextual case with a static allocation, estimation error and average detection delay are both $O(1/n_k)$ in the per-arm sample count. They are structurally proportional — the same lever (how often you sample an arm) drives both. This means there is no free-lunch between estimation and average detection: the Pareto surface collapses to a 1-D curve parameterized by $n_k$.

**The contextual revival.** In the contextual regime (`LinUcb`), routing also depends on a per-request feature vector. Average detection delay remains proportional to estimation (they share the same design-measure sensitivity). But **worst-case detection delay** — which concentrates on the covariate cell with the fewest observations — is genuinely independent. This is why `ContextualCoverageTracker` and `TriageSession` exist: localised regressions in sparse covariate regions need explicit coverage and cell-level triage, not just arm-level monitoring.

For the full treatment and concrete failure modes, see the [API docs](https://docs.rs/muxer) and [examples/EXPERIMENTS.md](examples/EXPERIMENTS.md).

## Unified decision records (recommended for logging/replay)

Most production routers want a single "decision object" shape regardless of policy so logging, auditing, and replay don't depend on per-policy conventions. `muxer` provides a unified `Decision` envelope with:

- `chosen`: the arm name
- `probs`: optional probability distribution (when a policy has one)
- `notes`: typed audit notes (explore-first, constraint gating, numerical fallback, etc.)

Each policy has a `*_decide` / `decide_*` method that returns this.

## Quick examples

### Start here — getting_started (3 arms, no CUSUM, no theory required)

```bash
cargo run --example getting_started
```

This covers the 80% case in ~60 lines: create a `Router`, run a select/observe loop, use delayed quality labeling (`set_last_junk_level` + `set_last_quality_score`), and see why `quality_weight` breaks ties that binary ok/junk rates can't. No feature flags needed.

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

You maintain a `Window` per arm, push `Outcome`s as requests finish, and call `select_mab` on each decision.

```bash
cargo run --example deterministic_router
```

Note: this example simulates an environment and therefore requires `--features stochastic` if you disabled default features.

### Monitored selection (baseline vs recent drift + uncertainty-aware rates)

If you maintain a baseline and recent window per arm for change monitoring, use `MonitoredWindow`
plus `select_mab_monitored_*`:

```bash
cargo run --example monitored_router --features stochastic
```

### End-to-end router demo (Window + constraints + stickiness + delayed junk)

This combines multiple production patterns in one loop: window ingestion, constraints+weights, stickiness reasons, and delayed junk labeling.

```bash
cargo run --example end_to_end_router --features stochastic
```

This same scenario has a CI-checked regression test in `tests/e2e_metrics.rs` and logs whether constraint fallback was used.

### Window ingestion with delayed quality labeling

In most real-world routing, quality is known only after processing: you call the arm, receive a response, then score it (compute F1, run a parser, check embedding similarity) and label it junk if it falls below your threshold. The pattern is: push the `Outcome` immediately with `junk: false`, then call `set_last_junk_level` once scoring completes.

```bash
cargo run --example window_delayed_junk_label
```

### Constraint + trade-off tuning for `select_mab`

Example showing "constraints first, then weights":

```bash
cargo run --example mab_constraints_tuning
```

### Router — production pattern (monitoring + triage + calibration + snapshot)

```bash
cargo run --example router_production --features stochastic
```

Shows: CUSUM threshold calibration, full `RouterConfig` with monitoring/triage/coverage/control, regression detection, acknowledgment, and snapshot/restore for persistence across restarts.

### Router — full lifecycle (select / observe / triage / acknowledge)

```bash
cargo run --example router_quickstart
```

This covers: basic two-arm routing, quality divergence, triage detection, acknowledgment, large-K batch exploration (K=20 in 7 rounds with k=3), and dynamic arm management. No `--features` flag needed.

### Multi-pick selection with a latency guardrail

`select_mab_k_guardrailed_explain_full` selects up to `k` unique arms per decision, applying
a `LatencyGuardrailConfig` each round. When combined with `MonitoredWindow`s, use
`select_mab_k_guardrailed_monitored_explain_full`. `log_mab_k_rounds_typed` converts the
explanation into compact, log-ready round rows.

Guardrail semantics (soft pre-filter vs hard constraint) are shown in:

```bash
cargo run --example guardrail_semantics
```

### Unified decision records

```bash
cargo run --example decision_unified
```

### Detect-then-triage (`TriageSession`)

`TriageSession` wires per-arm CUSUM detection with per-`(arm, context-bin)` cell investigation:

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

`OutcomeIdx::from_outcome(ok, junk, hard_junk)` maps a `muxer::Outcome` triple to the 4-category index space (`OK` / `SOFT_JUNK` / `HARD_JUNK` / `FAIL`). For the non-contextual case, `worst_first_pick_k` provides arm-level triage without feature vectors.

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

Note: this example requires `--features stochastic` if you disabled default features.

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

Note: this example requires `--features stochastic` if you disabled default features.

### Contextual routing (LinUCB)

```bash
cargo run --example contextual_router --features contextual
```

Notes:

- If you want a probability distribution over arms for this context (e.g. for traffic-splitting or logging approximate propensities), use `LinUcb::probabilities(...)` or `LinUcb::decide_softmax_ucb(...)`.
- Algorithm reference: LinUCB (Chu et al., "Contextual bandits with linear payoff functions").

Contextual "propensity logging" example:

```bash
cargo run --example contextual_propensity_logging --features contextual
```

### Stickiness / switching-cost control

If you want to reduce "flapping" between arms, wrap deterministic selection with `StickyMab`:

```bash
cargo run --example sticky_mab_router
```

### Mini-experiments (bandits × monitoring × false alarms)

Runnable probes that make tradeoffs and failure modes explicit — see [examples/EXPERIMENTS.md](examples/EXPERIMENTS.md) for guided walkthroughs of each:

- `cargo run --example guardrail_semantics`
- `cargo run --example coverage_autotune --features stochastic`
- `cargo run --example free_lunch_investigation --features stochastic`
- `cargo run --example detector_inertia --features stochastic`
- `cargo run --example detector_calibration --features stochastic`
- `cargo run --example bqcd_sampling --features stochastic`
- `cargo run --release --example bqcd_calibrated --features stochastic`

Reusable bits extracted from these experiments live in `muxer::monitor`, notably:

- `CusumCatBank`: "GLR-lite" robustification via a small bank of CUSUM alternatives.
- `calibrate_threshold_from_max_scores`: threshold calibration from null max-score samples (supports Wilson-conservative mode).

## Documentation

- [**Quickstart** (5-minute guide)](docs/QUICKSTART.md)
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
// K=30 arms, k=3 per round → initial coverage in ~10 rounds.
let cfg = RouterConfig::default().with_coverage(0.02, 1);
let d = router.select(3, 0);
```

## Usage

```toml
[dependencies]
muxer = "0.3.8"
```

If you only want the deterministic `Window` + `select_mab*` core (no stochastic bandits), disable default features:

```toml
[dependencies]
muxer = { version = "0.3.8", default-features = false }
```

## Development

```bash
# If you are in a larger Cargo workspace, scope to this package:
cargo test -p muxer

# Microbenches (criterion):
cargo bench -p muxer --bench coverage
cargo bench -p muxer --bench monitor

# (Optional) Match CI checks:
cargo fmt -p muxer --check
cargo clippy -p muxer --all-targets -- -D warnings
```
