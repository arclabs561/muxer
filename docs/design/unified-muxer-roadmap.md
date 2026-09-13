# Unified muxer technical roadmap

status: additive core delivered locally; serialized restart and release remain

date: 2026-09-13

scope: unified adaptive allocation lifecycle

grounded-in: [architecture](unified-muxer-architecture.md), [protocol](unified-muxer-protocol.md), [evidence](unified-muxer-evidence.md)

## Position and governing constraints

### Implementation outcome

The additive runtime, scalar/contextual/external/quality profiles, typed event
ledger, retained policy epochs, per-item batch correlation and OPE projection
are implemented. Event retention limits apply to the whole decision, not each
batch item. See [migration](../UNIFIED_MUXER.md) and
[performance evidence](../UNIFIED_MUXER_PERFORMANCE.md).

Local validation passed the canonical six feature configurations for tests and
Clippy, default/all/Boltzmann doctests, both strict rustdoc configurations,
all-feature build, formatting, canonical examples and all six new unified
examples. New integration suites cover lifecycle, events, epochs, checkpoint
handoff, profile kernel comparisons, delayed quality and synthetic OPE truth.
This is local validation, not remote CI or a semver release check.

Remaining gates and deliberate boundaries:

- Checkpoints preserve complete state through a consuming in-memory handoff.
  Serialized process restart, model-reference resolution and versioned disk
  compatibility remain implementation work; phase 4 is not fully complete.
- Quality reuses the existing Router kernels under the shared lifecycle.
  Further reducer extraction needs parity and performance evidence; it is not
  required to remove the legacy public Router API.
- Item-local missing-channel resolution is implemented; a separate item-local
  cancellation operation remains a protocol target.
- Runtime lifecycle overhead exceeds the provisional 20% review threshold.
  It remains opt-in, with the measured tradeoff recorded rather than hidden.
- No legacy deprecation, package publication, commit, push or remote release
  gate was performed. The release owner must run those gates before shipping.

The phase descriptions below retain the original acceptance targets; this
outcome section is the authoritative distinction between delivered work and
remaining targets.

The live baseline is `307a03a`. Generic assessments, scalar/contextual policies,
retained delayed-label corrections, scalar OPE and routing/monitoring harnesses
already exist. They are reuse targets, not new work items. A common issued
decision and feedback lifecycle is the missing connection verified in the
evidence map.

This proposes new sequencing for API consolidation. It does not assert that
historical delivery claims in the [older roadmap](post-round-3-roadmap.md)
remain verified. Preserve [config-hardening](config-api-hardening.md),
[1.0 checklist](api-1.0-checklist.md), [OPE](ope-primitives.md), and
[Tower boundary](muxer-tower.md) constraints. In particular, do not change old
config exhaustiveness incidentally, expand OPE into storage, or add Tower
dependencies to the core.

The older Tower proposal says not to change the core *for that adapter* before
consumer evidence. This program is independently motivated by the unified-API
request and existing contextual/delayed consumers. It does not use Tower as
justification for new runtime dependencies.

## Dependency order

```text
0 baseline and behavior corpus
             |
1 protocol and ergonomic proof
             |
2 runtime plus scalar/contextual adapters
             |
3 typed feedback and quality composition
             |
4 external refresh and complete checkpoints
             |
5 evaluation projection and migration release gate
```

Phase numbers express dependencies, not time estimates. Each phase ends in
a reviewable change with its consumer, test evidence and compatibility notes.
One implementation owner owns each phase; a reviewer checks the gate before
the next dependent phase. Research-dependent algorithm additions are parked.

## Phase 0 — Capture the behavior baseline

Consumer: existing callers whose behavior must survive migration. Reversible.

- Run the canonical CI commands locally at the chosen implementation baseline.
  Record toolchain/features and failures without claiming a historical run.
- Use existing `router_props`, `assessment`, contextual-propensity and policy
  tests to inventory contracts; add only missing regression cases.
- Capture deterministic traces of eligible candidates, selections, stage reasons,
  observations, corrections, acknowledgements and restore behavior.
- Survey actual downstream imports and struct literals in explicitly selected
  consumer repositories before planning a public rename. No inferred absence
  of consumers from this checkout alone.

Gate: a reusable baseline corpus and a current test result; each intentional
future difference has a named expected behavior. Preserve current snapshot
triage reset as a legacy contract, not a defect to silently repair.

## Phase 1 — Prove the public contract before generalizing

Consumer: users of the simple, delayed, contextual and externally scored loops.
Reversible prototype; no stable public promise.

- Prototype `interaction` records and `Muxer<P>` with private fields,
  validated constructors, associated context/feedback/ticket types.
- Compile four real consumers: Bernoulli outcome, delayed quality score,
  external metric assessment, and propensity-aware contextual reward.
- Exercise a custom policy from an external test crate; ensure no access to
  private internals is required.
- Resolve the decision records below using the researched recommendations.
  Record any deviation and evidence before implementation builds on it.

Gate: common Bernoulli use is at most five interaction statements without
explicit generics; invalid model/policy combinations fail clearly; external
scoring needs no training/runtime dependency; request eligibility remains
authoritative. Examples call real production kernels, not mock-only interfaces.

## Phase 2 — Correlated runtime and policy adapters

Consumer: overlapping scalar/contextual requests. Partially reversible.

- Implement issuance, receipts, bounded pending/terminal caches, cancellation,
  epoch checks and validate/prepare/apply update ordering.
- Wrap existing Thompson, Boltzmann, EXP3 and LinUCB. Use EXP3's explicit
  probability update; retain immutable LinUCB context. Expose `Unavailable`
  for posterior-max propensities.
- Give the new Boltzmann adapter an owned trial RNG and one authoritative
  categorical distribution for sampling/logging. Document the change from
  the existing drawset-based low-level sampler; test probability consistency.
- Keep catalogue changes separate from request eligibility. EXP3's full
  ordered universe must remain stable across subset requests.
- Preserve low-level APIs and the existing `Decision`; use a new receipt name
  rather than repurposing its current public/serde contract.

Gate: D1/D2/D3 can receive rewards in D3/D1/D2 order; each update uses its own
features/probability. Duplicate, wrong-engine, wrong-action and expired events
cannot alter state. Capacity errors do not evict pending data or consume a
decision. Feedbackless assessment decisions never accumulate pending tickets.
Capture parameter/decay semantics, without claiming delay-free bounds.

## Phase 3 — Quality feedback and shared composition

Consumer: current Router users and multi-channel quality workloads.
Partially reversible.

- Implement final/provisional/missing channel states and declared subscribers.
- Initially adapt Router execution feedback, then extract reusable bounded
  bookkeeping/reducers and concrete quality stages into the shared runtime.
- Preserve control -> triage -> policy behavior, including novelty, coverage
  and guardrail ordering options. Retain current fallback semantics inside
  hard eligibility.
- Implement explicit ordered-distinct batch receipts for quality; keep scalar
  evaluation unavailable for them.
- Keep legacy correction setters working as documented. New finalized channels
  reject revision rather than silently changing only some consumers.
- Keep direct historical `Router::observe*` distinct from receipt-correlated
  feedback; migrate live callers to decision IDs and seed historical state
  explicitly without fabricating behavior probabilities.

Gate: immediate-feedback quality traces match the baseline except documented
issuance metadata. Delayed scalar-score traces also match retained quality-row
membership/counts; exercise score-before-execution and score-after-row-eviction.
Scores never replay categorical samples.
A rejected final-category correction changes no subscriber. Custom scalar
and quality profiles share the runtime; a permanent facade over two independent
lifecycle engines does not pass. Cut duplicated orchestration after parity.

## Phase 4 — External model refresh and restart continuity

Consumer: out-of-band training pipelines and restarted routing applications.
Partially reversible.

- Add explicit model/representation/config revisions and compatibility checks.
  Start with drain-before-incompatible-replace, then bounded retained epochs
  for uninterrupted refresh.
- Demonstrate external assessments, frozen models, same-schema refresh and
  offline embeddings with a small online contextual head.
- Provide optional complete checkpoint capability for supported built-ins,
  including pending tickets, RNG continuation and monitoring state.
  Keep legacy snapshots and statistical warm starts distinctly named.
- Record model references; do not add Python, ONNX, network or training
  dependencies merely to prove BYO.

Gate: a pending decision from representation v1 never updates v2 implicitly;
equal dimensions do not waive revision checks. Checkpoint/restore plus the
same accepted-event stream produces the same next decisions on the same build.
Missing external model references fail explicitly. Epoch capacity is tested.

## Phase 5 — Evaluation projection, ergonomics and release

Consumer: offline analysts and downstream adopters. Publication is one-way
for that release; legacy removal requires a separate compatibility decision.

- Project eligible single-execution/final-reward receipts into existing
  `LoggedReward`; account for unsupported/missing rows with reason counts.
  Demonstrate one synthetic full-information truth fixture where IPS agrees
  and naive averaging is biased. Do not simulate full information from real
  bandit logs.
- Port representative examples to the shared API. Keep standalone
  `CandidateAssessment` and statistical functions as useful low-level entrypoints.
- Compare `benches/router_select.rs` baseline at 5/25 arms and k=1/5/10;
  additionally measure issuance, feedback and retained-ticket memory.
  Provisional review threshold: median hot-path overhead over 20% in two
  same-machine runs requires profiling and a documented choice, not automatic
  rejection. This is a proposed budget, not a measured result.
- Set retention defaults from measured ticket sizes and expected concurrency;
  document entry limits versus arbitrary payload bytes.
- Update README, quickstart, doc status and migration guide together. Before
  deprecation, survey downstream consumers and run their representative builds.
  Rerun the 1.0 checklist; no automatic version/date promise.

Gate: canonical CI matrix and new contract tests pass, new examples are actually
executed, migration differences are explicit, and semver changes are reviewed.
Verify remote CI only when an authorized delivery step actually publishes work.

## Contract test matrix

| Boundary | Required assertion |
| --- | --- |
| Eligibility | Every control/triage/policy/fallback pick belongs to request set |
| Probability | Exact rows normalize; sampled action has positive probability; unknown never becomes one |
| Mixture | Action reachable from two branches logs the sum of branch contributions |
| Delays | Original context and propensity survive intervening decisions and updates |
| Deduplication | Same payload has no second effect; conflicting payload leaves all state unchanged |
| Partial feedback | Cost/latency arrival does not fabricate quality or repeat execution count |
| Quality join | Score changes retained execution rows only; no row resurrection or extra execution |
| Completion | Stateless decisions do not retain tickets; quality score expectation survives execution arrival |
| Finality | Rejected correction leaves learner, detector and window unchanged |
| Missingness | Expiry/censoring emits no zero reward; evaluation reports missing cohort |
| Replacement | Old epoch cannot update new representation; pending work is drained or retained explicitly |
| Batches | Unique picks, bounded length, declared probability unit; scalar OPE rejects |
| Batch feedback | Each selected position has its own execution/score channels; item 2 can finish before item 1 without overwriting or closing it |
| Restore | Full continuation reproduces next choice; warm start cannot accept old pending handles |
| Ergonomics | Built-in and BYO consumers compile without ceremonial trait plumbing |

## Accepted implementation decisions

Accepted on 2026-09-13 with the user's authorization to implement the full
researched roadmap. This table is the decision record for this additive API;
it does not create a separate ADR ledger or authorize legacy API removal.

| Accepted decision | Governs | Options considered and decision |
| --- | --- | --- |
| Unified lifecycle ownership | `src/interaction*`, `src/runtime*`, `src/policy.rs` | Permanent facade / universal component engine / typed runtime: choose typed runtime for one lifecycle with algorithm-specific evidence |
| Feedback finality and retention | `src/runtime*`, `src/router.rs`, reducers | Mutable final events with replay / immutable final channels: choose immutable final channels and bounded dedup; preserve legacy setters separately |
| Probability and execution unit | `src/decision.rs`, `src/ope.rs`, adapters | Optional scalar everywhere / typed single and batch records: choose typed records with unavailable probability explicit |

Implementation starts with a captured baseline and compiled consumer proof.
Lifecycle ownership, immutable finality and explicit probability units above
govern the dependent work. Each phase still requires its stated validation
gate; acceptance of the design is not evidence of implementation completion.

## Parked work and review trigger

IDS, generic information geometry, automatic scientific inference, decision-aware
drift restarts, online final-label correction replay, durable stores and a full
Tower balancer are outside this delivery sequence. Existing primitives and
examples can support future evidence; their presence does not authorize new
guarantees or a broader framework.

Review after phase 1, after the first quality migration, or when a real consumer
needs an unsupported feedback/probability unit. Rewrite the proposal if those
checks contradict its assumptions; do not add optional interfaces to preserve
an abstraction the examples have disproved.
