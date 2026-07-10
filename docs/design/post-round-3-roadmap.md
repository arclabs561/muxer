# Post-Round-3 Roadmap

status: proposal
date: 2026-07-09
grounded-in:
- `.claude/reports/scrutinize-2026-03-15.md`
- `.claude/reports/useful-2026-03-15.md`
- `.claude/reports/enrich-2026-03-15.md`
- `.claude/reports/qa-2026-07-09.md`
- `docs/design/bandit-theory-grounding.md`
- commit `691e675`

## Current State

The Round 3 maintenance patch is built and pushed. It fixes monitored-router
acknowledgement semantics, validates monitoring capacities, caches monitoring
scores during multi-pick selection, aligns the declared Rust version with
`pare`, and adds a router selection benchmark.

The remaining work is structural. It should not be folded into the bugfix
patch.

## Settled Constraints

- `muxer` stays a primitive crate. HTTP, gRPC, dashboard, and storage concerns
  belong outside the core crate.
- Significant-shift-aware CUSUM is grounded but not a task yet. It needs source
  paper review and a real consumer before implementation.
- Off-policy analysis must account for adaptive data collection. Naive
  statistics on bandit logs are not valid enough to expose as a feature.
- API hardening before 1.0 is cheaper than after 1.0, but construction
  ergonomics still matter for this crate.

## Phase 1: Land The Maintenance Patch

Consumer: current users of monitored routing.

Status: in progress until CI is green on `main`.

Gate:
- `muxer` CI passes on commit `691e675` or a follow-up fix-forward commit.
- `pare` CI passes on the changelog commit.

Reversibility: reversible. The changes are mostly bug fixes, docs, and internal
selection-score reuse.

## Phase 2: Decide Config Struct API Hardening

Consumer: downstream users constructing `RouterConfig`, `MabConfig`, and
`MonitoredMabConfig`.

Fork:

1. Add `#[non_exhaustive]` to config structs before 1.0.
   - Better future compatibility.
   - Worse struct-literal ergonomics.
   - Requires users to construct with defaults/builders.
2. Keep config structs literal-friendly until a 1.0 API hardening pass.
   - Better current ergonomics.
   - Higher future breakage cost if fields need to change.

Recommendation: defer the attribute for now, but create a 1.0 API checklist
before any release that claims stability. The checklist now exists at
`docs/design/api-1.0-checklist.md`.

Gate:
- Do not add new public config fields until this fork is decided.
- If a second downstream crate starts constructing configs with literals, favor
  literal-friendly compatibility.
- If config field churn continues, favor `#[non_exhaustive]`.

Reversibility: partially reversible before 1.0, one-way after 1.0.

## Phase 3: Design Off-Policy Evaluation Primitives

Consumer: users who log routing decisions and want to compare candidate
policies offline.

Status: minimal primitive surface implemented.

Minimum useful scope:
- Define a log row type or example schema that carries action, observed
  outcome, and propensity.
- Add IPS estimate helpers for scalar rewards.
- Document that adaptive logs require debiasing and that naive sample means are
  misleading.

Open questions:
- Should this live in `muxer` or a `muxer-eval` companion crate?
- What is the canonical reward scalar for a multi-objective `Outcome`?
- Can `RouterDecision` expose enough propensity information for deterministic
  and monitored selection, or is OPE initially limited to stochastic policies?

Gate:
- Write a focused design doc before code.
- Add at least one replay-style fixture that would fail for naive means.

Reversibility: partially reversible. Estimator APIs are now public surface, but
the first pass is isolated from router state and storage formats.

## Phase 4: Revisit Significant-Shift-Aware Monitoring

Consumer: users seeing too many harmless drift alarms.

This is not ready for implementation. The current design note explicitly warns
that the surviving idea needs primary-source review and consumer evidence.

Gate:
- Read the significant-shift paper path at the source. Done for Suk and
  Kpotufe (2022), arXiv:2112.13838: the mechanism is safe-arm elimination with
  scheduled replays and episode restarts, not a scalar CUSUM-threshold change.
- Produce a simulation where current CUSUM over-triggers but a best-arm-aware
  trigger preserves route quality with fewer restarts. Done in
  `examples/significant_shift_sim.rs`: restarting on any CUSUM alarm averages
  11.50 restarts and 285.0 regret in the fixed-seed run, while the
  best-arm-aware gate averages 0.14 restarts and 224.1 regret.
- Decide whether the feature belongs in the monitoring guard, triage trigger, or
  a higher-level policy layer.

Reversibility: partially reversible. Incorrect monitoring semantics can be hard
to unwind once users tune thresholds around them.

## Phase 5: Prototype `muxer-tower` Outside The Core Crate

Consumer: Rust service teams routing HTTP/gRPC requests across backends.

Scope:
- Separate crate or example workspace.
- `tower::Layer` integration only.
- No storage, dashboard, or async runtime assumptions in `muxer` itself.

Gate:
- `RouterDecision` and outcome-reporting ergonomics are stable enough that the
  tower layer does not need workaround APIs.
- At least one runnable service example proves the integration shape.

Reversibility: reversible if kept outside the core crate.

## Do Not Start Until Decided

- Do not implement OPE until the propensity/log-row surface is designed.
- Do not implement significant-shift-aware CUSUM until source-paper review and a
  false-restart simulation exist.
- Do not add HTTP/tower dependencies to `muxer`.
- Do not change config struct exhaustiveness as part of an unrelated patch.
