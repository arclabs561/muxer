# Design: 1.0 API Checklist

status: proposal
date: 2026-07-09
grounded-in:
- `docs/design/config-api-hardening.md`
- `docs/design/post-round-3-roadmap.md`
- commit `20e6fb0`

## Purpose

Run this checklist before any release candidate that claims 1.0 stability.
The goal is to make breaking Rust API choices explicit while the crate is still
pre-1.0.

## Construction Contracts

- Decide whether config structs remain literal-friendly or become
  `#[non_exhaustive]`.
- Include at least these config types in the decision: `RouterConfig`,
  `MabConfig`, `MonitoredMabConfig`, `CoverageConfig`, `ControlConfig`,
  `LatencyGuardrailConfig`, `ThompsonConfig`, `Exp3IxConfig`, `LinUcbConfig`,
  `BoltzmannConfig`, `StickyConfig`, `TriageSessionConfig`,
  `WorstFirstConfig`, `ContextBinConfig`, `DriftConfig`, and
  `UncertaintyConfig`.
- If a public config struct stays exhaustively constructible, treat adding a
  public field after 1.0 as breaking.
- If the preferred style is builder-first, make sure every field with common
  user demand has a builder or setter before adding `#[non_exhaustive]`.

## Enum Exhaustiveness

- Keep operational enums that may grow as `#[non_exhaustive]`, including
  policy, mode, decision-note, monitor-metric, and error enums.
- For any exhaustive enum, write down why callers should be allowed to match
  all variants without a wildcard.

## Data Records

- Confirm which records are stable data contracts rather than debug snapshots:
  `Outcome`, `Summary`, `LoggedReward`, `RouterDecision`, `RouterSnapshot`,
  `Selection`, `CandidateDebug`, `ObjectiveValue`, and the monitor decision
  records.
- Decide whether debug records should stay public-field structs, move behind
  accessors, or remain unstable until a later major release.
- Verify serde shape for every public record when `features = ["serde"]` is
  enabled.

## Routing And Observation Surface

- Confirm `Router::select(k, seed)` seed semantics are final: deterministic MAB
  ordering is stable; seed affects coverage, control picks, and triage.
- Confirm `Router::select_from(eligible, k, seed)` semantics are final: it
  validates registered unique arms, preserves Router registration order, and
  constrains every selection and fallback stage to that request-local set.
- Confirm `Router::observe_with_id` and targeted delayed-label APIs are
  sufficient for wrappers such as `muxer-tower`. IDs are caller-owned; durable
  storage, expiry, and cross-process correlation remain adapter concerns.
- Do not add router-level OPE propensities unless the deterministic-policy
  semantics are designed.

## OPE Surface

- Keep the first 1.0 OPE contract limited to `LoggedReward`, `ips_value`, and
  `self_normalized_ips_value` unless a separate design covers reward models,
  doubly robust estimators, adaptive weighting, and confidence intervals.
- Decide whether OPE stays in `muxer` or moves to a companion crate before any
  larger evaluator surface is added.

## Feature Flags

- Check every public re-export under `stochastic`, `contextual`, `boltzmann`,
  and `serde`.
- Verify `cargo check --no-default-features --all-targets` and all individual
  feature combinations used in CI.
- Do not make a default feature required for the core deterministic routing API.

## Release Gate

- README, `docs/QUICKSTART.md`, examples, rustdoc, and `CHANGELOG.md` agree on
  the public surface.
- `cargo semver-checks` runs clean against the previous published release, or
  every breaking item is listed in the changelog.
- MSRV is stated in `Cargo.toml` and checked in CI.
- A fresh clone can run the getting-started example and at least one monitoring
  example from the documented commands.
