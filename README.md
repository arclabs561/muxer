# muxer

Deterministic, multi-objective routing primitives for "provider selection" problems.

## What problem this solves

You have a small set of arms (providers/models/backends) and repeated calls that produce outcomes (success/429/junk), plus cost + latency. You want an **online policy** that:

- **explores** new or recently-changed arms
- **avoids regressions** (junk/429 spikes)
- is **deterministic by default** (same stats/config → same choice), so it's easy to debug

## What it is

The core selection idea is:

- maintain a small sliding window of recent outcomes per provider (ok/429/junk, cost, latency)
- compute a Pareto frontier over the objectives
- pick a single provider deterministically via scalarization + stable tie-break

This crate also includes:

- a **seedable Thompson-sampling** policy (`ThompsonSampling`) for cases where you can provide a scalar reward in `[0, 1]` per call
- a **seedable EXP3-IX** policy (`Exp3Ix`) for more adversarial / fast-shifting reward settings
- (feature `contextual`) a **linear contextual bandit** policy (`LinUcb`) for per-request routing with feature vectors
- **latency guardrails** (`LatencyGuardrailConfig` / `apply_latency_guardrail`) — hard pre-filter by mean latency, with stop-early semantics for multi-pick loops
- **multi-pick selection** (`select_mab_k_guardrailed_explain_full` and variants) — select up to `k` unique arms per decision with a per-round guardrail loop
- **maintenance sampling** (`CoverageConfig` / `coverage_pick_under_sampled`) — ensure all arms stay measured above a quota; see [Three goals for sampling](#three-goals-for-sampling) for why this matters
- **post-detection triage** (`WorstFirstConfig` / `worst_first_pick_k`) — prioritize the most degraded arms for investigation after monitoring fires
- **contextual cell triage** (`ContextualCoverageTracker` / `contextual_worst_first_pick_k`) — lift triage to `(arm, context-bin)` pairs so localised regressions don't average away
- **combined detect + triage sessions** (`TriageSession`) — wires per-arm CUSUM detection and per-cell cell investigation into one stateful session
- **`softmax_map`** — stable score → probability helper for traffic splitting
- **novelty helpers** (`novelty_pick_unseen`), **prior smoothing** (`apply_prior_counts_to_summary`), and **pipeline glue** (`PipelineOrder` / `PolicyPlan`) for building custom routing harnesses

## Which policy should I use?

- **`select_mab` (Window + Pareto + scalarization)**: when you care about **multiple objectives** at once (success, 429, junk, cost, latency) and you want deterministic selection with hard constraints.
- **`ThompsonSampling`**: when you can provide a **single reward** per call (in `[0, 1]`) and want a classic explore/exploit policy (seedable, optionally decayed).
- **`Exp3Ix`**: when reward is **non-stationary / adversarial-ish** and you still want a probabilistic policy (seedable, optionally decayed).
- **`LinUcb` (feature `contextual`)**: when you have a per-request feature vector (e.g. cheap "difficulty" features, embeddings, metadata) and want a contextual policy.

## Three goals for sampling

Every routing decision involves three objectives that generally compete:

1. **Exploitation** — minimize regret; route to the best arm now.
2. **Estimation** — understand each arm's true rates; keep all arms measured.
3. **Detection** — notice when an arm changes; minimize delay between the change and the alarm.

**The two clocks.** You only observe an arm when you sample it, so there are two notions of time:

- **Wall time** $t$: global decision steps.
- **Sample time** $n_k$: observations from arm $k$.

Detection delay in wall time scales as $\text{delay}_{wall} \approx \text{delay}_{samples} / \text{rate}_k$. `CoverageConfig` sets a minimum sampling-rate floor, which is the direct lever for bounding wall-clock detection delay.

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

### Deterministic multi-objective selection (Pareto + scalarization)

```rust
use muxer::{select_mab, MabConfig, Summary};
use std::collections::BTreeMap;

let arms = vec!["a".to_string(), "b".to_string()];
let mut summaries = BTreeMap::new();
summaries.insert("a".to_string(), Summary { calls: 10, ok: 9, junk: 0, hard_junk: 0, cost_units: 10, elapsed_ms_sum: 900 });
summaries.insert("b".to_string(), Summary { calls: 10, ok: 9, junk: 2, hard_junk: 0, cost_units: 10, elapsed_ms_sum: 900 });

let sel = select_mab(&arms, &summaries, MabConfig::default());
assert_eq!(sel.chosen, "a"); // lower junk when all else is equal
```

### Realistic "online routing loop" (Window ingestion)

This is closer to production usage: you maintain a `Window` per arm, push `Outcome`s as requests finish, and call `select_mab` each decision.

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

### Window ingestion with delayed junk labeling

If your "junk" classification is only known after downstream parsing/validation, you can update the most recent outcome:

```bash
cargo run --example window_delayed_junk_label
```

### Constraint + trade-off tuning for `select_mab`

Example showing "constraints first, then weights":

```bash
cargo run --example mab_constraints_tuning
```

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

Runnable:

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

Runnable:

```bash
cargo run --example thompson_router
```

Note: this example requires `--features stochastic` if you disabled default features.

### Contextual routing (LinUCB)

Runnable:

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

If you want runnable "research probes" that make tradeoffs/failure modes explicit, see:

- [EXPERIMENTS](examples/EXPERIMENTS.md)
- Examples:
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

- [API docs (docs.rs)](https://docs.rs/muxer)
- [Changelog](CHANGELOG.md)
- [Mini-experiments / research probes](examples/EXPERIMENTS.md)

## Usage

```toml
[dependencies]
muxer = "0.2.0"
```

If you only want the deterministic `Window` + `select_mab*` core (no stochastic bandits), disable default features:

```toml
[dependencies]
muxer = { version = "0.2.0", default-features = false }
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
