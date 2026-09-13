# Unified muxer evidence and resolved questions

status: research-backed proposal input

date: 2026-09-13

baseline: `307a03ac32f9096ac71fbad3ce566bc9e7b1abe9`

## Scope and source boundary

Question: what shared runtime contract makes muxer cohesive and approachable
without erasing algorithm-specific evidence, delayed-feedback semantics or
the existing transport-neutral boundary?

Criteria: correctness, fit to current consumers, reversibility, interface and
maintenance cost, then performance. Research stops when each load-bearing
protocol choice has either primary evidence or an explicit engineering decision
with a falsification test. This is not a survey of every bandit algorithm.

Intent source: ChatGPT conversation titled “Review muxer architecture,” selected
branch ending with the plug-and-play discussion. Local raw source:
`../../chatgpt-export-6aa661aa-updated.raw.json`; parsed source:
`../../chatgpt-export-6aa661aa-updated.md`; capture manifest:
`../../chatgpt-export-6aa661aa-updated.manifest.json`.
Those are local capture artifacts, not required published documentation assets.
Raw JSON retains alternate branches and tool activity; analysis uses the selected
branch. First selected conversational role is user, last is assistant. The
raw capture is 5,917,062 bytes and the parsed Markdown is 118,456 bytes. The
manifest reports 353 nodes, 352 messages, 282 selected records and 70 alternate
records. No recognized assets were listed; enumeration is not exhaustive.
Export citations alone are not verification of their paper claims.

## Local orientation and verified constraints

Single Rust library; `src/lib.rs` contains substantial quality/state primitives
alongside module exports. `src/router.rs` is orchestration; `monitor.rs` and
`triage.rs` own monitoring mechanisms. `examples/` holds application and
simulation consumers; `tests/` holds integration/property checks; `benches/`
holds Criterion benchmarks. `scripts/` builds optional offline datasets.
Manifest version is 0.5.3, MSRV 1.75. No release or remote CI status is inferred.

| Source at baseline | Observation | Design implication |
| --- | --- | --- |
| `src/assessment.rs:14`, `:124`; `tests/assessment.rs` | Finite caller metric vectors, indexed objectives, stateless selection | External inference already has a viable boundary |
| `src/policy.rs:1`, `:34`; `src/boltzmann.rs:127` | Common trait covers Thompson, EXP3 and Boltzmann, intentionally excludes LinUCB | Extend lifecycle without pretending context is arm-only |
| `src/decision.rs:1`, `:203` | Runtime/debug envelope explicitly not a replay/evaluation record | Add a receipt rather than silently upgrading `Decision`'s guarantee |
| `src/router.rs:544`, `:580`, `:606` | Pure seeded select; authoritative eligible set; ordered stages | Preserve eligibility and distinguish issuance from preview |
| `src/router.rs:621-762`, `:1049` | Control, triage, summary snapshot, novelty/coverage/guardrail/MAB fill | Preserve concrete composition before extracting public stage traits |
| `src/router.rs:780`, `:810`; `src/lib.rs:461` | ID dedup is window-retention scoped | No durable exactly-once claim |
| `src/router.rs:875`; `tests/router_props.rs:476` | Corrected windows do not replay triage | Immutable final channels or explicit rebuild, never silent partial correction |
| `src/router.rs:937`, `:1150`; `tests/router_props.rs:578` | Acknowledge resets selected monitoring state; restore recreates triage | Reset, warm start and full continuation need different names |
| `src/exp3ix.rs:120`, `:304`, `:385`, `:528` | Arm order changes reset state; explicit probability update already exists | Keep universe separate from eligibility; capture propensity in ticket |
| `src/contextual.rs:261`, `:437` | Learning consumes context again | Retain original features, not newly computed embeddings |
| `src/thompson.rs:381`, `:412` | Posterior-max has no exact logged probability; accepts fractional updates | Preserve unavailable probability and label reward model honestly |
| `src/boltzmann.rs:110`, `:127` | Sampling uses f32 logits and an external sampler; diagnostic softmax is separately computed in f64 | New adapter uses one distribution and owned RNG; no unchanged-wrapper checkpoint promise |
| `src/ope.rs:10`; `examples/contextual_propensity_logging.rs:14` | Scalar OPE and explicit warnings about incomplete logging | Receipt projection, not a new evaluator platform |

Read order for implementation: `policy.rs` and `decision.rs`, `router.rs`
selection/observe, `lib.rs` windows, `exp3ix.rs` filtered decisions/updates,
`contextual.rs`, then `router_props.rs`. The generic and stateful boundaries
matter more than reading modules by size.

Existing CI in `.github/workflows/ci.yml` runs formatting, lint/tests across
default, no-default, all-feature and selected individual-feature builds,
doctests, rustdoc and assertion-bearing examples. Its toolchain is stable;
the declared MSRV is not separately exercised by that workflow. This pass
inspected gates but did not execute the Rust suite. `git log` in the last
30 days showed manifest/changelog activity, not a newly implemented unified
runtime. The memory listing returned no text; no historical memory claims
are used to close work.

## Primary research ledger

**R1 — Separate observations, rewards and delayed delivery.** Joulani, György
and Szepesvári, [Online Learning under Delayed Feedback](https://proceedings.mlr.press/v28/joulani13.pdf),
§2 and §3, model feedback identified with its originating round, potentially
arriving out of order. Their reductions depend on assumptions and base learners.
High confidence: correlation is structural. Engineering inference: retain
typed decision-time tickets; do not claim that an event API preserves every
immediate-feedback bound.

**R2 — IDS is not a universal uncertainty interface.** Russo and Van Roy,
[Learning to Optimize via Information-Directed Sampling](https://arxiv.org/html/1403.5556),
§3, §4.2 and §6, define regret and information about the optimal action, then
develop model-specific computations. High confidence: a point estimate plus
an arbitrary uncertainty scalar is insufficient. Decision: defer an IDS
capability until a concrete information model and comparison fixture exist.
Their Beta-Bernoulli example explicitly uses binary rewards; fractional score
updates need separate interpretation.

**R3 — Evaluation needs the behavior policy, not a score.** Dudík, Langford
and Li, [Doubly Robust Policy Evaluation and Learning](https://icml.cc/2011/papers/554_icmlpaper.pdf),
§2–3, distinguish reward models from logging probabilities in contextual
bandit data. High confidence: accurate event records support estimation but
do not remove its assumptions. Decision: preserve scalar IPS/SNIPS and require
a checked projection, with no automatic confidence intervals or missing-label
correction.

**R4 — Practical prediction/learning separation.** Vowpal Wabbit's
[contextual-bandit tutorial](https://vowpalwabbit.org/docs/vowpal_wabbit/python/latest/tutorials/python_Simulating_a_news_personalization_scenario_using_Contextual_Bandits.html),
“Getting a decision,” samples a returned distribution and retains the selected
probability for learning. Its
[Python tests](https://github.com/VowpalWabbit/vowpal_wabbit/blob/master/python/tests/test_pyvw.py)
exercise CB action/cost/probability labels, CCB/slates label structures and
saved-model prediction continuity. High confidence in these observed contracts;
no claim that muxer should adopt VW's text format or reduction architecture.

**R5 — Batch feedback needs its own unit.** VW's
[Slates tutorial](https://vowpalwabbit.org/docs/vowpal_wabbit/python/latest/tutorials/python_slates.html)
distinguishes per-slot decisions from a whole-slate reward. High confidence
that a scalar action probability is not a universal batch record. Decision:
preserve distinct ordered-batch shape and refuse scalar OPE projection. The
conditional-product rule follows elementary probability, not an independence
assumption about the draws.

**R6 — A lightweight API is credible, but not enough for this lifecycle.**
River's [policy implementation](https://github.com/online-ml/river/blob/main/river/bandit/base.py)
exposes pull/update and pluggable reward statistics/scaling. This supports
the ergonomics target, while muxer's contextual and delayed consumers require
stronger correlation. Source inspection only; no interoperability or
performance comparison is claimed.

## Resolved ambiguities

| Question | Recommendation and credible alternative | What would reverse it |
| --- | --- | --- |
| Must every algorithm share a belief representation? | No: typed policy ticket; alternative universal belief vector loses structure | Two independent policies genuinely share a representation and update law |
| Does BYO require a predictor trait? | No: caller assessments/context first; alternative runtime-owned predictor adds execution concerns | Two real consumers need identical in-process inference hooks |
| Should final labels be mutable? | Immutable finalized channels; alternative checkpoint replay is more costly | A named online correction consumer supplies replay/storage constraints |
| How are delays handled? | Arrival-order updates with captured selection data; alternative reorder buffers stall on missing outcomes | A consumer needs event-time equivalence and defines a watermark/expiry model |
| Where is idempotency owned? | Bounded runtime protection, durable adapter protection; alternative event store expands crate scope | A separately approved storage boundary changes the existing non-goal |
| Should the new API immediately replace old types? | Additive modules and parity-tested migration; alternative rewrite increases simultaneous changes | Compatibility becomes demonstrably more complex than a documented breaking cut |
| Is a point predictor useless for exploration? | It can support explicit random exploration; it does not automatically support IDS | A calibrated richer interface is available for the selected policy |

These resolve design direction. Performance defaults, compiler ergonomics and
behavioral parity still need executable evidence; the roadmap supplies those
gates instead of treating further literature reading as a substitute.
