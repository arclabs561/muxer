# Design: Config API Hardening

status: proposal
date: 2026-07-09
grounded-in:
- `.claude/reports/scrutinize-2026-03-15.md`
- `.claude/reports/qa-2026-07-09.md`
- `docs/design/post-round-3-roadmap.md`

## Problem

`RouterConfig`, `MabConfig`, and `MonitoredMabConfig` are public config
structs in a pre-1.0 crate. Adding `#[non_exhaustive]` now would preserve more
future compatibility, but it would also make struct-literal construction harder
for current users.

## Chosen Direction

Do not add `#[non_exhaustive]` to config structs as part of unrelated bugfix or
feature work. Keep current construction ergonomics for now, and make config
exhaustiveness a named item in the 1.0 API checklist.

This matches the current evidence: the Round 3 patch needed validation and docs,
not a construction-contract change. The decision can still be revisited before
1.0 if config fields continue to churn.

The checklist now lives at `docs/design/api-1.0-checklist.md`.

## Non-Goals

- Do not make config fields private in this pass.
- Do not add builders just to compensate for `#[non_exhaustive]`.
- Do not change `Outcome` construction semantics here.
- Do not treat this as a 1.0 readiness decision.

## Why Not Add `#[non_exhaustive]` Now?

It would be cheap mechanically, but it changes downstream construction behavior.
The crate's examples and public docs still teach direct defaults plus builder
methods, and there is no concrete field-addition pressure that requires forcing
all users away from literals today.

## Decision Gates

- Revisit before any 1.0 release candidate.
- Revisit if config fields are added or renamed in two consecutive feature
  releases.
- Revisit if a downstream crate reports breakage from config field churn.
- Favor `#[non_exhaustive]` if the public API moves toward builders as the
  primary construction path.
