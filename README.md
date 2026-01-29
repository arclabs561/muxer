# muxer

Deterministic, multi-objective routing primitives for “provider selection” problems.

The core idea is:

- maintain a small sliding window of recent outcomes per provider (ok/429/junk, cost, latency)
- compute a Pareto frontier over the objectives
- pick a single provider deterministically via scalarization + stable tie-break

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
