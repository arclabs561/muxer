# Changelog

## [0.1.3]

### Added
- `context_bin` / `ContextBinConfig`: deterministic feature-vector → bin-ID quantisation (levels × stable_hash64).
- `ContextualCell`: typed `(arm, context_bin)` pair for per-cell triage.
- `contextual_worst_first_pick_one` / `contextual_worst_first_pick_k`: per-(arm, context-bin) investigation helpers for the contextual regime.
  Closes the per-cell monitoring gap noted in `monitor.rs`: arm-level detection followed by cell-level triage without any per-cell sliding-window overhead.
- `CellStats`: per-(arm, bin) call/hard-junk/soft-junk counters with `hard_junk_rate()` / `soft_junk_rate()`.
- `ContextualCoverageTracker`: self-contained accumulator for per-cell stats; exposes `record()`, `active_bins()`, `pick_one()`, and `pick_k()` — callers no longer need to roll their own `HashMap`.
- `scenarios.rs` integration tests for the detect→triage loop: `contextual_tracker_surfaces_localised_regression` and `contextual_tracker_pick_k_targets_degraded_bins`.

## [0.1.2]
- Objective manifold documentation + tests.
- Contextual routing: `LinUcb` persistence helpers (`snapshot`/`restore`) and `theta_vectors()` for sensitivity analysis.
- Bump dependency to `pare 0.1.1`.

## [0.1.1]
- EXP3-IX + guardrail/novelty fill helpers and monitoring primitives.
- Drop the experimental HTTP-429 rate dimension.

## [0.1.0]
- Initial release.

