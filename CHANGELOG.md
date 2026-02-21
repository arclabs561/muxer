# Changelog

## [0.3.8]

### Fixed

- **`log_top_candidates_mab` score**: now includes `quality_weight × mean_quality_score`
  in the ranking score, so audit logs reflect the same objective as selection.

### Added

- **`LogTopCandidate::mean_quality_score: Option<f64>`** — quality score now included
  in the top-candidates log rows.
- **`DecisionNote::Diagnostics::mean_quality_score`** — the per-arm diagnostics note
  now carries the quality score for the chosen arm.
- **README link to `docs/QUICKSTART.md`** at the top of the Documentation section.
- **`BanditPolicy` re-exported** from anno's `muxer_harness.rs`.

## [0.3.7]

### Added

- **`CandidateDebug::mean_quality_score: Option<f64>`** — audit field populated from
  `Summary::mean_quality_score` in both `select_mab_explain` and monitored selection.
- **`Router::mean_quality_score(arm)`** — convenience accessor for the selection window's
  mean quality score.
- **`Router::window_len(arm)`** — number of observations currently in the selection window.
- **`apply_prior_counts_to_summary` quality blending** — if the prior has
  `mean_quality_score` set, it is now inherited (or calls-weighted merged with the
  observed quality) instead of being silently dropped.
- **`docs/QUICKSTART.md`** — 5-minute guide: start-here command, two-minute Router
  snippet, expected output, configuration checklist, common failure modes.

### Tests added

- `prior_inherits_quality_score_when_out_has_none`
- `prior_blends_quality_score_weighted_by_calls`
- `prior_quality_none_does_not_overwrite`

## [0.3.6]

### Changed

- **`getting_started` example rewritten**: now actually calls `set_last_junk_level`
  and `set_last_quality_score` (both were commented-out placeholders before), and
  adds a step 4 demonstrating that `quality_weight=1.0` correctly picks the arm
  with `quality_score=0.92` over the arm with `quality_score=0.78`.
- **README "Quick examples"**: `getting_started` is now the first recommended run.
  Fixed undefined `seed` in the Router quickstart code block.  Added `mean_quality_score: None`
  to inline `Summary` examples.  Added `BanditPolicy` to "What it is" and "Which policy."
- **`lib.rs` module overview**: add `BanditPolicy` and `quality_weight` to the
  selection policies description.

### Fixed

- **`wilson_half_width_decreases_with_more_trials` proptest**: the old test held
  success COUNT fixed while increasing n — this can widen the Wilson interval by
  pushing the rate toward 0.5 (maximum variance).  Replaced with
  `wilson_half_width_decreases_with_proportional_scaling` which scales both
  successes and trials by the same integer multiplier (exact same rate), where
  monotone decrease is guaranteed.

### Added

- **`quality_weight_is_monotone` proptest**: for any arm A with `quality_score ∈ [0.6, 1]`
  and arm B with `quality_score ∈ [0, 0.5]` — all else equal, `quality_weight > 0`
  always selects A.  Runs 100 random cases.

## [0.3.5]

### Added

- **`Summary::mean_quality_score: Option<f64>`** — pre-computed mean quality
  score from `Window::summary()`.  `None` when no outcomes have `quality_score`
  set.  Backward-compatible: defaults to `None` in serde and struct `Default`.
- **`MabConfig::quality_weight: f64`** (default `0.0`) — when non-zero, adds
  `quality_weight × mean_quality_score` to `objective_success` in `select_mab`
  and `select_mab_monitored_*`.  Completes the gradient-signal loop opened in 0.3.4.
- **`MonitoredWindow::set_last_quality_score`** — delayed quality labeling
  propagated to both baseline and recent windows.
- **`Router::set_last_quality_score`** — same, via the Router front door.
- 4 new tests in `tests/outcome_invariants.rs`: `mean_quality_score` None/average,
  `set_last_quality_score` clamping, and `quality_weight` selection influence.

## [0.3.2]

### Fixed

- `Router::select_mab_round` no longer clones `MonitoredWindow` per MAB round.
  Previously O(K × window_cap × sizeof(Outcome)) bytes were cloned per `select`
  call; now the full map is passed by reference with zero allocation.

### Added

- **`RouterSnapshot`** — serializable snapshot of `Router` state (windows,
  monitored windows, config, total observations).  CUSUM scores are intentionally
  excluded and reset on restore to avoid stale alarms across process restarts.
- **`Router::snapshot()`** — capture current state.
- **`Router::from_snapshot(snap)`** — restore from a snapshot.
- **`MonitoredWindow`** now derives `serde::Serialize/Deserialize` (feature `serde`).
- **`examples/router_production.rs`** — full production pattern: CUSUM threshold
  calibration, `RouterConfig` builder with monitoring+triage+coverage+control,
  routing loop with simulated regression, triage detection, acknowledgment, and
  snapshot/restore.
- 4 new snapshot tests in `tests/router_props.rs`.

## [0.3.1]

### Added

- `examples/router_quickstart.rs`: runnable full-lifecycle demo (K=20 in 7 rounds with k=3).
- `tests/router_props.rs`: 11 property/integration tests for `Router`.
- `tests/calibration.rs`: 8 tests for `calibrate_cusum_threshold` and related utilities.

### Fixed

- Clippy: `PartialEq` for `ControlConfig` is now derived; `RangeInclusive::contains`
  in `outcome_invariants.rs`; `too_many_arguments` suppressed on `calibrate_cusum_threshold`.

## [0.3.0]

### Added

- **`Router`** — stateful routing session that owns all per-arm state and
  exposes a three-method interface: `select(k, seed)` / `observe(arm, outcome)` /
  `acknowledge_change(arm)`.  Handles the full normal → triage → acknowledge
  lifecycle.  Supports `add_arm` / `remove_arm` for dynamic arm management.
  Efficient for large K: with `k > 1`, K=30 arms reach initial coverage in ~10
  rounds; `CoverageConfig` prevents starvation.
- **`RouterConfig`** — single builder-friendly config that collapses `MabConfig`,
  `DriftConfig`, `CoverageConfig`, `LatencyGuardrailConfig`, `TriageSessionConfig`,
  and `ControlConfig` into one struct with `with_*` builder methods.
- **`RouterMode`** — `Normal | Triage { alarmed_arms }` enum returned by
  `Router::mode()` and embedded in `RouterDecision`.
- **`RouterDecision`** — output of `Router::select`: chosen arms, current mode,
  pre-picks, control picks, eligible arms, and triage cells.
- **`MonitoredWindow::acknowledge_change`** — promotes the recent window into the
  baseline and clears it; completes the post-detection protocol.
- **`MonitoredWindow::promote_recent_to_baseline`** — soft merge (no clear).
- **`MonitoredWindow::baseline_len` / `recent_len`** — window size accessors.
- **`calibrate_cusum_threshold`** (feature `stochastic`) — convenience Monte Carlo
  calibration: simulate null max-scores, build a grid, call
  `calibrate_threshold_from_max_scores`.  Answers "what threshold gives
  `P[alarm within m rounds] ≤ α`?"
- **`simulate_cusum_null_max_scores`** (feature `stochastic`) — underlying Monte
  Carlo simulator; useful when you want to reuse the null samples across grid sweeps.
- **`ThresholdCalibration`** and **`calibrate_threshold_from_max_scores`** promoted
  to top-level re-exports (were monitor-internal).
- **`ControlConfig`** / **`pick_control_arms`** / **`split_control_budget`** —
  reserve a deterministic-random fraction of picks as a selection-bias anchor.
- **`suggested_window_cap(throughput, change_rate)`** — SW-UCB–derived window
  size guidance: returns `sqrt(throughput / change_rate)` clamped to `[10, 10_000]`.
- **`suggested_window_cap_for_k(k, total_throughput, change_rate)`** — per-arm
  variant for large-K deployments.

## [0.2.0]

### Added
- `TriageSession`: combines per-arm `CusumCatBank` detection with
  `ContextualCoverageTracker` cell triage in a single stateful session.
  `observe(arm, outcome_idx, context)` feeds both; `alarmed_arms()` surfaces
  regressions; `top_alarmed_cells(bins, k)` returns investigation targets.
- `TriageSessionConfig`: defaults for CUSUM p0/alts/threshold and bin/scoring config.
- `ArmTriageState`: snapshot of `(n, score_max, alarmed)` for one arm.
- `OutcomeIdx`: typed constants for the 4-category outcome space (`OK`, `SOFT_JUNK`,
  `HARD_JUNK`, `FAIL`) with `from_outcome(ok, junk, hard_junk)` helper.

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

