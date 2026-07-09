# Design: Off-Policy Evaluation Primitives

status: proposal
date: 2026-07-09
grounded-in:
- `.claude/reports/useful-2026-03-15.md`
- `.claude/reports/enrich-2026-03-15.md`
- `docs/design/bandit-theory-grounding.md`
- `docs/design/post-round-3-roadmap.md`

## Problem

Users who log bandit routing decisions need a way to ask "what would a
different policy have achieved?" without re-running production traffic. Naive
means over bandit logs are biased by adaptive data collection, so any public
offline-analysis surface must make propensities explicit.

## Chosen Direction

Add minimal scalar OPE primitives to `muxer`, not a storage or replay system.
The core surface should operate over records containing an observed reward,
the logged action propensity, and optionally the candidate policy propensity.
Start with IPS and self-normalized IPS for scalar rewards. Multi-objective
reward construction remains caller-defined.

This keeps the feature in the crate's primitive layer: no async runtime, no
database format, no HTTP concepts, and no full experiment platform.

## Non-Goals

- No event store or log ingestion framework.
- No dashboard or statistical reporting UI.
- No automatic conversion from `Outcome` to reward. Callers own their scalar
  utility function.
- No doubly-robust estimator in the first pass unless a concrete regression
  fixture needs it.
- No claim of valid confidence intervals until the interval method is designed
  separately.

## Options Considered

### Put OPE in `muxer`

Best when the API is generic and small: `LoggedReward`, `ips_value`, and
`self_normalized_ips` style helpers. This matches the crate's primitive
identity and avoids a companion crate with one small module.

### Put OPE in `muxer-eval`

Better if the feature grows into replay engines, bootstrap intervals, policy
comparison reports, or file formats. This is the escape hatch if the first-pass
API starts wanting storage or plotting.

### Example-only OPE

Good for teaching but weak as an API. It would not give downstream users a
stable place to build against, and it would likely be copied incorrectly.

## Proposed API Shape

The first implementation should be roughly:

```rust
pub struct LoggedReward {
    pub reward: f64,
    pub logging_propensity: f64,
    pub target_propensity: f64,
}

pub fn ips_value(rows: impl IntoIterator<Item = LoggedReward>) -> Result<f64, OpeError>;
pub fn self_normalized_ips_value(
    rows: impl IntoIterator<Item = LoggedReward>,
) -> Result<f64, OpeError>;
```

Names are placeholders. The design requirement is that propensities are visible
at the call site and validated as finite values in `[0, 1]`.

## Gates

- Add a fixture where naive averaging gives the wrong answer and IPS corrects
  it.
- Reject zero, negative, NaN, and infinite logging propensities.
- Document that confidence intervals are not part of the first pass.
- Do not add OPE methods to `Router` until `RouterDecision` has a deliberate
  propensity story.

## Open Questions

- Should zero target propensity rows be accepted as zero contribution or
  filtered by the caller?
- Should the API consume borrowed rows to avoid allocation, or prefer value
  types for simplicity?
- Should error reporting return the first invalid row index?
- How should stochastic policy decisions expose propensities consistently with
  deterministic `Router` decisions?

## Review Trigger

Move OPE to a companion crate if the implementation wants file formats,
simulation environments, bootstrap intervals, dashboards, or policy replay.
