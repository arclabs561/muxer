# muxer

Deterministic, multi-objective routing primitives for “provider selection” problems.

## What it is

The core idea is:

- maintain a small sliding window of recent outcomes per provider (ok/429/junk, cost, latency)
- compute a Pareto frontier over the objectives
- pick a single provider deterministically via scalarization + stable tie-break

This crate also includes:

- a **seedable Thompson-sampling** policy (`ThompsonSampling`) for cases where you can provide a scalar reward in `[0, 1]` per call
- a **seedable EXP3-IX** policy (`Exp3Ix`) for more adversarial / fast-shifting reward settings

## Quick examples

### Deterministic multi-objective selection (Pareto + scalarization)

```rust
use muxer::{select_mab, MabConfig, Summary};
use std::collections::BTreeMap;

let arms = vec!["a".to_string(), "b".to_string()];
let mut summaries = BTreeMap::new();
summaries.insert("a".to_string(), Summary { calls: 10, ok: 9, http_429: 0, junk: 0, hard_junk: 0, cost_units: 10, elapsed_ms_sum: 900 });
summaries.insert("b".to_string(), Summary { calls: 10, ok: 9, http_429: 0, junk: 2, hard_junk: 0, cost_units: 10, elapsed_ms_sum: 900 });

let sel = select_mab(&arms, &summaries, MabConfig::default());
assert_eq!(sel.chosen, "a"); // lower junk when all else is equal
```

### EXP3-IX (adversarial bandit) with probabilities

```rust
use muxer::{Exp3Ix, Exp3IxConfig};

let arms = vec!["a".to_string(), "b".to_string(), "c".to_string()];
let mut ex = Exp3Ix::new(Exp3IxConfig { seed: 123, decay: 0.98, ..Exp3IxConfig::default() });

let (chosen, probs) = ex.select_with_probs(&arms).unwrap();
// ... run request with `chosen` ...
ex.update_reward(chosen, 0.7); // reward in [0, 1]

let s: f64 = probs.values().sum();
assert!((s - 1.0).abs() < 1e-9);
```

### Thompson “traffic splitting” selector (mean-softmax allocation)

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
let (chosen, alloc) = ts.select_softmax_mean_with_probs(&arms, 0.3).unwrap();
ts.update_reward(chosen, 1.0);

let s: f64 = alloc.values().sum();
assert!((s - 1.0).abs() < 1e-9);
```

## Usage

```toml
[dependencies]
muxer = "0.1.0"
```

## Development

```bash
# If you are in a larger Cargo workspace, scope to this package:
cargo test -p muxer

# (Optional) Match CI checks:
cargo fmt -p muxer --check
cargo clippy -p muxer --all-targets -- -D warnings
```
