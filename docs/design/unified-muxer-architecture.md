# Unified muxer architecture

status: additive core implemented and locally validated; restart/release gates remain

date: 2026-09-13

baseline: `307a03ac32f9096ac71fbad3ce566bc9e7b1abe9`

scope: adaptive allocation API, policy composition, feedback ownership

## Decision and reading order

Build one adaptive decision runtime, `Muxer<P>`, with useful built-in profiles
and a small typed policy extension contract. Keep standalone statistical
primitives usable. Unify the interaction lifecycle and its invariants; let
algorithms retain the distinct state and evidence they require.

The user authorized implementation of this researched recommendation on
2026-09-13. This design records the accepted direction, not a claim that every
gate has passed or a stable API promise.
Read this document first, then the [protocol](unified-muxer-protocol.md),
[delivery roadmap](unified-muxer-roadmap.md), and
[source evidence](unified-muxer-evidence.md). The protocol specifies the
otherwise ambiguous behavior behind the examples.

## Problem and present architecture

The user's requirements are cumulative: cohesive API, sound mathematical
grounding, externally trained components, and an easy, enjoyable common case.
The existing crate has useful parts of this system already:

| Boundary | Existing mechanism | Consequence for this design |
| --- | --- | --- |
| External estimates | `CandidateAssessment` and Pareto/scalar selection in `assessment.rs` | Reuse caller-produced metrics; no mandatory inference framework |
| Online scalar policies | `BanditPolicy`, Thompson, EXP3-IX, Boltzmann | Preserve kernels; replace arm-only correlation at the runtime boundary |
| Contextual learning | `LinUcb` receives features on both decide and update | Retain decision-time features in a typed ticket |
| Quality routing | `Router` owns windows, control, triage and selection stages | Initially adapt; eventually share runtime and reducers |
| Late labels | `ObservationId` and retained-window corrections | Preserve capability while making correction effects explicit |
| Evaluation | `LoggedReward`, IPS and self-normalized IPS | Derive validated inputs; retain scalar evaluator scope |

Selection and feedback have different ownership today. `Router::select` is
read-only; policies may advance random state. Router feedback identifies an
observation, not an issued decision. Triage consumes initial observations but
does not replay corrected labels. These are live contracts, not incidental
details a rename can resolve. See the evidence map for source locations.

## Options considered

1. **Facade over everything indefinitely.** Smallest first patch, but leaves
   different correlation and feedback semantics behind a uniform spelling.
   Use only as a migration stage with a deletion gate.
2. **Universal `Muxer<E, P, I, A, R>`.** Mirrors the conversation's conceptual
   diagram, but requires every algorithm to expose estimator, preference,
   information and regime objects, even EXP3 or an externally supplied action
   distribution. Reject as the mandatory public contract.
3. **Small runtime plus typed policies and concrete compositions. Recommended.**
   One lifecycle enforces identity, eligibility and feedback rules. Policies
   supply decision-specific learning tickets. Introduce reusable capability
   traits only where two real implementations need the same operation.

## User experience

These illustrate the implemented API; complete executable consumers live in
`examples/unified_*.rs`. See the [migration guide](../UNIFIED_MUXER.md) for
delivered semantics and the [roadmap](unified-muxer-roadmap.md) for remaining gates.

```rust,ignore
let mut mux = Muxer::bernoulli(["small", "large"])?;
let d = mux.decide(&())?;
let accepted = execute(d.action());
mux.tell(d.id(), accepted)?;
```

The Bernoulli profile takes a boolean. A fractional bounded reward profile
must be named and documented separately: existing fractional Beta updates
must not be sold as an exact Bernoulli posterior for arbitrary scores.

```rust,ignore
let mut mux = Muxer::quality(models)
    .config(router_config)
    .with_delayed_score()
    .build()?;
let d = mux.decide_from(&ready_models, &context)?;
mux.tell(d.id(), QualityFeedback::execution(outcome))?;
mux.tell(d.id(), QualityFeedback::score(0.91)?)?;
```

`decide` uses the registered catalogue; `decide_from` restricts every stage.
Quality defaults preserve today's deterministic policy and explicit metric
units. There is no surprise switch to IDS or stochastic exploration.
Detailed receipts expose eligibility, stage reasons, probability availability,
feedback status and revisions. Diagnostics explain actual computations;
they do not invent uncertainty or an exploration-value score.

## Ownership and component boundaries

```text
Application: features / external inference / readiness / execution / storage
                    |                         ^
          context or assessments              | immutable receipt
                    v                         |
Muxer<P>: validate request -> issue decision -> retain learning ticket
                    |                         |
                    v                         v
Policy P: estimator + allocation       correlated feedback validation
                    ^                         |
                    +----- prepared update ---+
                              |
                         optional monitors
```

The runtime owns the registered action catalogue, session-local decision
identity, bounded pending records, revision checks and delivery receipts.
The policy owns learned state, decision-time sufficient data, and update
semantics. The application owns execution, retries, persistence, training,
network inference and authoritative availability. A single mutable owner
serializes runtime operations; adapters choose locks or message passing.

Proposed modules are `interaction` (public records), `runtime` (lifecycle),
and `profiles` (recipes). Existing policy and monitoring modules retain their
names and algorithms. Start these as modules in this crate, not new crates.
Leave `tower`, HTTP and storage in companions, consistent with
[the Tower design](muxer-tower.md).

Use associated context, canonical feedback, ticket and prepared-effect types on
the policy contract. A context can be `()`, a feature slice or caller-defined
assessments. No requirement to serialize, clone or train an external model.
Built-in combinations should fail at construction/type checking when their
capabilities disagree; invalid per-request dimensions still return errors.

Provide adapters for a stateless scoring function and an externally supplied
categorical distribution. These supply the no-op prepared effects and feedback
normalization machinery, so plugging in a scorer does not require implementing
the entire advanced policy contract. The phase 1 external test crate proves
this escape hatch using actual kernels and types.

| Policy/composition | Required evidence | Ticket retains |
| --- | --- | --- |
| Pareto over external assessments | Finite metric vectors and explicit objectives | Provenance; no synthetic learning state |
| Bernoulli Thompson | Beta sufficient statistics, binary outcomes | Selected action and policy epoch |
| EXP3-IX | Bounded reward and actual sampling probability | Selected probability, relevant parameters and epoch |
| LinUCB | Fixed-schema decision features and scalar reward | Exact decision features and representation revision |
| Softmax over supplied scores | Finite scores and temperature | Actual categorical distribution |
| Future IDS | Expected regret and decision-relevant information gain | Model-specific evidence; optional later capability |

Research supports distinct capability requirements, not universal
interchangeability. In particular, posterior samples, standard errors, UCB
bonuses and mutual information are different objects.

## Generic state and quality migration

Common runtime does not require a universal `Vec<f64>` posterior. Extract
bounded observation bookkeeping and typed reducers from quality windows only
after scalar and quality profiles exercise them. Preserve `Outcome` as a
convenient typed schema and `Summary` as its view. An external-assessment
profile may own no estimator at all.

The first consolidation shares Router's validated observation and identified
score reducers between legacy methods and the quality profile. Prepared quality
updates are opaque single-use deltas, not whole-Router replacements. Batch
tickets share immutable context; pending-to-terminal conversion moves grouped
per-item evidence through one common path. This preserves the legacy public
surface without maintaining separate observation mutation implementations.

Monitoring consumes finalized channel values. Allocation consumes monitoring
evidence; a detector does not implicitly reset all learners. Preserve the
existing control, triage, novelty, coverage and guardrail ordering first as a
concrete quality composition. Only then expose the independently useful
composition operations. Hard eligibility remains outside every fallback.

A model replacement carries a revision and update-compatibility declaration.
Same-schema frozen predictors may be replaced without retraining inside
muxer. A new embedding schema requires a new contextual learner epoch unless
an explicit migration is supplied. The protocol handles pending old decisions.

Feedbackless external assessments terminate at issue; delayed quality explicitly
opts into retaining its score channel. Those modes share issuance without
forcing stateless callers to manufacture feedback.

## Non-goals

- Training frameworks, representation learning and remote inference runtimes:
  applications supply their results.
- Bellman planning, queue management and global resource-budget enforcement:
  allocation can consume their outputs, but does not own those systems.
- Durable event storage and a full evaluation platform: preserve the
  [OPE boundary](ope-primitives.md).
- Universal regret, false-alarm or safety guarantees for composed profiles:
  current empirical semantics remain explicit.
- Immediate removal of public primitives or a wholesale rewrite: compatibility
  wrappers retire only after the roadmap's migration gates.

## Tradeoffs and reversal gates

Pending tickets add memory and mutable issuance to the high-level API.
Typed composition is less permissive than a bag of optional interfaces but
gives meaningful compatibility errors. Receipt validation adds work to the
hot path; retention and diagnostics must be configurable.

Reject this abstraction if the first four executable consumers require
policy-name switches in the runtime, fabricated evidence, or separate feedback
lifecycles. Revise it if the simple example exceeds five interaction statements
or needs explicit generic parameters. Measure overhead before choosing a
default retention capacity; do not make an unmeasured latency promise.

The roadmap resolves implementation in dependency order. API publication and
legacy retirement remain explicit review gates; the proposed contracts are
concrete enough to prototype without pretending those gates have passed.
