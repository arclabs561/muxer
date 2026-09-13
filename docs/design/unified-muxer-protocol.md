# Unified muxer interaction protocol

status: accepted contract; additive core implemented with boundaries below

date: 2026-09-13

governing-design: [architecture](unified-muxer-architecture.md)

This document specifies the accepted runtime contract. Names below are
interface notation; use the [migration guide](../UNIFIED_MUXER.md) and runnable
examples for exact Rust types. All profiles support a consuming in-memory
checkpoint handoff. The optional serialized checkpoint is concrete to
`QualityProfile`; it does not provide external-model loading or durable
single-writer coordination.
Requirements apply to the new API; existing low-level APIs keep their contracts.

## 1. Requests, identities and receipts

Keep a canonical registered action catalogue with stable order. A request
selects from a nonempty unique subset; reject unknown actions and duplicates
before changing policy state. Never pass the shrinking eligible list as the
EXP3 arm universe: that currently resets its learning state.

The initial public operation issues one action. The quality migration also
supports an explicitly named ordered distinct batch, preserving current `k`
behavior. It does not reinterpret a batch as independent categorical draws.

Every selected item has an `ExecutionKey = (DecisionId, selected_position)`.
Position is zero for a single selection. Batch feedback and channel finality
are keyed by execution key, not decision ID alone. `tell(id, value)` is single-
selection convenience; batches use `tell_item(id, position, value)` and reject
ambiguous position-free calls. A batch becomes terminal only after every item's
expected channels are final/missing or the item was explicitly cancelled.
`cancel_item(id, position)` rejects items with accepted provisional/final values,
preserves missing-only evidence, and leaves siblings unchanged. Repeated item
cancellation succeeds without repeating cleanup. `item_status` retains each
item's result after batch completion; overall `Completed` means every item
closed, not that each executed. Whole-decision cancellation requires no accepted
value feedback. Expiry changes only the statuses of still-open items.

```rust,ignore
struct Request<'a, C: ?Sized> {
    eligible: &'a [ActionId],
    context: &'a C,
}

struct DecisionReceipt {
    id: DecisionId,             // runtime epoch + monotone sequence
    catalogue_revision: Revision,
    eligible: Vec<ActionId>,    // canonical order; immutable
    selected: SelectionRecord, // One or OrderedDistinctBatch
    revisions: Revisions,      // policy, model, representation, preferences
    decision_sequence: u64,
    diagnostics: Diagnostics,
}
```

Receipts are immutable with private fields and accessors. Typed newtypes
validate probabilities, bounded rewards and identifiers at ingress. Engine
instance checks prevent accidentally submitting another runtime's handle.
Local IDs are not globally unique durable IDs: cross-process adapters supply
and persist a namespace. Never infer uniqueness from a fixed RNG seed.

The runtime retains a policy-owned ticket, separate from the public receipt.
LinUCB tickets copy or retain immutable features; a caller's later mutation of
the original feature buffer cannot change the update. Receipts retain a
representation/model revision and optional application context reference.
An opaque context reference is not sufficient for OPE unless the application
can resolve the decision-time context.

`decide` advances issuance/RNG state but not reward state. It is not the old
pure `Router::select`. A separately named preview may inspect scores but must
not masquerade as an issued decision. Concurrent outstanding decisions are
allowed unless a profile explicitly declares a one-outstanding-decision limit.
Pending counts are diagnostic, not automatically hallucinated reward samples.

## 2. Policy extension and atomic updates

The minimum extension surface has these conceptual operations:

```rust,ignore
trait InteractionPolicy {
    type Context: ?Sized;
    type Feedback;
    type CanonicalFeedback: Eq;
    type Ticket;
    type PreparedIssue;
    type PreparedUpdate;

    // Randomness belongs to a trial stream committed only on successful issue.
    fn prepare_decision(&self, request: Request<'_, Self::Context>, rng: &mut TrialRng)
        -> Result<PolicyDecision<Self::Ticket, Self::PreparedIssue>, PolicyError>;
    fn commit_issue(&mut self, issue: Self::PreparedIssue);

    fn normalize(&self, feedback: Self::Feedback)
        -> Result<Self::CanonicalFeedback, PolicyError>;

    // All domain errors occur before changes to learned state.
    fn prepare(&self, ticket: &Self::Ticket, feedback: &Self::CanonicalFeedback)
        -> Result<Self::PreparedUpdate, PolicyError>;

    // Infallible after successful preparation; no I/O.
    fn apply(&mut self, update: Self::PreparedUpdate);
}
```

The runtime validates eligibility, capacity and identity before calling the
policy. It validates the returned choice/distribution before committing the
prepared issue, trial RNG position and pending record together. Preparation
is read-only with respect to observable policy state; custom interior mutation
must not violate this contract. Initial adapters may prepare against a clone;
the benchmark gate decides whether to extract finer-grained kernel operations.
Built-ins prepare feedback for every subscribed reducer before any apply operation.
An invalid detector input cannot leave the learner updated but the detector
untouched. Panics and external side effects are outside this atomicity contract;
custom policies must obey it and conformance tests exercise it.

A composite policy's `PreparedUpdate` is an owned `PreparedEffects` containing
the window, learner and monitor changes plus their declared no-op reasons.
Its `prepare_all` validates them together; `apply_all` applies them in stable
window, learner, monitor order under the runtime's exclusive borrow. The runtime
then commits channel finality and dedup metadata before returning success.
`NotSubscribed` and `OutsideHorizon` are explicit planned no-ops; an invalid
subscribed value is an error for the whole event. No reducer is allowed to
discover a recoverable validation error during apply. Reserve required storage
before commit; this is in-memory error atomicity, not crash transactionality.

Canonical feedback equality is part of the extension contract. Built-ins reject
nonfinite numeric inputs and normalize equivalent representations (including
signed zero) before producing Eq-capable validated values. Custom policies
provide the same deterministic normalization and equality, without a mandatory
serialization dependency. Store canonical values for retained event IDs;
do not claim collision-free deduplication from an arbitrary hash alone.

Do not expose five mandatory traits because five boxes appeared in a diagram.
Concrete compositions can implement this contract; extract `Scores`, posterior
sampling or information-gain capabilities when tested compositions need them.

## 3. Execution, multiple attempts and feedback

Issuing a decision does not prove execution. The first accepted value-bearing
feedback for the selected action confirms execution in the simple API; a
`Missing` disposition alone does not. `cancel(id)` closes
an unexecuted decision without synthesizing a failure reward. Cancellation
after accepted feedback is invalid.

Profiles with an explicit execution channel confirm execution through that
channel (or `confirm_execution`), not through a score that arrived first.

Advanced adapters report execution explicitly. A retry or fallback to another
action requires another decision with an optional parent reference. Reporting
that other action against the original decision is an error: the original
propensity did not select it. Multiple real executions need distinct execution
records; first-version scalar OPE accepts only one execution per decision.

Feedback uses a runtime-owned envelope with decision ID, selected position, executed action,
event ID, channel, source revision, revision number, observed time and
`Disposition<Payload> = Provisional(Payload) | Final(Payload) | Missing(Reason)`.
The runtime checks executed action against that position in the receipt, independent of the
opaque policy payload. Simple `tell` fills it from the selection; callers are
responsible for telling the truth about execution. Advanced `confirm_execution`
records the actual action explicitly. The runtime assigns a monotone received
sequence to each new accepted event. Wall-clock timestamps are metadata; replay order
is received sequence. A channel names a measurement meaning, including units,
labeler/source and schema, not merely a display label like `quality`.

`tell(id, bool)` and `tell(id, BoundedReward)` construct the single final
reward channel internally. Repeating the same value is idempotent while the
receipt is retained; a different second value is a conflict. Multi-channel
profiles define equally explicit convenience keys and completion rules.

| Incoming event | State transition / result |
| --- | --- |
| First provisional event | Retain canonical value; no learning effects |
| First final event for an open channel | Validate all subscribers, apply once, acknowledge |
| Same event ID and normalized payload | `Duplicate`, no mutation |
| Same event ID, different payload | `ConflictingEvent`, no mutation |
| Different event ID repeats an already finalized channel | Identical value: duplicate; different value: conflict |
| Higher revision of a provisional value | Replace provisional value; no learner/detector update yet |
| Finalization | Commit one sample; same revision/value can finalize a provisional value, or a higher revision can replace and finalize atomically |
| Correction of a final value | `FinalizedChannel`; no silent replay or partial correction |
| Missing/censored final outcome | Close as missing with reason; never turn absence into zero |
| Unknown/expired decision, wrong epoch/action, invalid value | Typed error; no mutation |

Revisions are ordered within an execution-key/channel/source identity. Lower revisions return
`StaleRevision`; equal revision with changed canonical value conflicts. Finality
can advance at the same revision only with an unchanged value and a new event ID.
Changing labeler/source revision requires a separately declared channel; it
does not overwrite another source's final sample. A final or missing channel
cannot reopen. Event-ID equality compares the canonical envelope except the
runtime-assigned received sequence; a reused ID with different metadata conflicts.

Provisional observations are optional advanced functionality, not required
for simple feedback. A channel's final value is immutable in the first unified
runtime. Applications can delay finalization or produce a separately named
corrected-label dataset for offline rebuilds. Existing Router setters continue
their documented window-only corrections through the legacy API; that behavior
is not secretly mapped onto a supposedly universal event stream.

Quality has separate execution and scalar-score channels. Execution feedback
contains final categorical `ok/junk/hard_junk` values and cost/latency; a later
score can update the score reducer without replaying those categories.
Category detectors subscribe only to final categories, and a score detector
only to final scores. A profile declares subscribers and its completion rule.

For quality routing, each item's execution inserts exactly one row keyed by execution key
into the same bounded selection/monitoring populations as today's `Outcome`.
Final score joins that row in every window that still retains it, preserving
its original position and execution count. It never adds a second execution
or a separately clocked selection-score window. If score arrives first, retain
it in the pending ticket; execution commits the joined row and each subscribed
sample once. If all rows have since expired, finalize with an explicit
`OutsideHorizon` effect for those windows; do not resurrect rows. An optional
score detector declares its own horizon and consumes at most one final score.
The generic runtime does not dictate one window population for every reducer.

Legacy `Router::observe*` continues accepting registered-arm observations
without issuance and retains caller-owned `ObservationId` behavior. It is not
aliased to `tell`. Migrate live callers by keeping the returned decision ID;
map historical warm-start data through an explicit profile seeding operation
before issuance, with no invented propensity or execution receipt. Compatibility
parity tests distinguish legacy direct observations from new issued traffic.

## 4. Retention and ordering

Retain open tickets and terminal deduplication records under explicit separate
limits. Full open capacity returns `PendingCapacity`; never evict a pending
decision merely to admit another. Caller-driven `expire` closes a record with
a reason, not a reward. Closed records eventually leave the bounded cache;
feedback after eviction is rejected as unknown/expired, not reapplied.

Each policy decision declares `FeedbackExpectation::None` or a finite list of
channels to resolve. `None` terminates at issue and retains no learning ticket;
external-assessment users can keep selecting without `tell` or `cancel`.
Its receipt does not assert execution and cannot enter scalar OPE by itself.
For learning profiles, terminal means every selected item's expected channel is final or
explicitly missing. Terminal storage keeps the receipt, accepted event IDs and
canonical payloads/channel outcomes; the learning ticket is released.
Scalar profiles expect one reward. Quality defaults to execution only;
`.with_delayed_score()` declares execution plus score, so execution alone does
not prematurely release the ticket. Missing scores require caller-driven
closure/expiry, whose deadline belongs to the application. Undeclared channels
are rejected; arbitrary extension cannot keep a decision open forever.

Also cap retained event records/revisions per decision; exceeding that cap
returns `FeedbackCapacity` before mutation. A stream of provisional revisions
must not bypass pending-record limits and grow memory indefinitely. Expected
channel sets are finite and validated at issuance.

The built-in runtime promises bounded entry counts, not bounded bytes for
arbitrary custom ticket types. Profiles document per-ticket storage; applications
bound feature/payload sizes. Size and retention defaults are chosen from the
roadmap benchmark, exposed through builders and visible in diagnostics.
The core does not promise durable exactly-once delivery. Storage adapters own
durable event deduplication and crash recovery.

All online updates use arrival order. Standalone channel-window clocks advance
on accepted final samples; the quality join above is anchored to execution rows.
Dependency-buffered values take effect when their required execution arrives.
They do not reorder old observations by wall clock or pretend a late label was
available at decision time. This may differ
from a desired event-time analysis; that analysis belongs in an explicit rebuild.

EXP3 uses the ticket's actual chosen probability and policy epoch. Decay applies
on accepted reward updates, matching the kernel's update clock. Capturing a
probability fixes attribution, but does not prove an immediate-feedback regret
bound for delayed execution. Parameter/canonical-arm changes create a new
epoch; old tickets cannot update it by accident. Thompson/LinUCB likewise
declare their supported arrival-order behavior and forgetting clocks.

## 5. Revisions and replacement

Separate representation, predictive model, allocation/preference configuration
and learned-state epochs. A change to one is not automatically a reset of all.

| Change | Default handling |
| --- | --- |
| Frozen scorer version, same output schema | New model revision; existing tickets remain interpretable |
| Same representation with a compatible model update | Preserve learner only when its adapter declares compatibility |
| New embedding coordinates, even same dimension | New representation and contextual learner epoch |
| Reward transform, EXP3 parameters or arm-universe reset | New learning epoch |
| Preference weights only | New preference revision; retain raw measurements |
| Request-local readiness changes | No catalogue/learning reset |

Retain an old epoch until its pending decisions close, subject to an explicit
epoch-capacity limit; replacement fails at capacity unless the caller chooses
to expire those decisions. This supports continuous replacement without
training an old ticket against the new model. A compatible migration is a
separate tested operation, never inferred from equal vector length.
The first implementation may require drain-before-replace; it must report that
restriction rather than claim seamless refresh until retained epochs exist.

## 6. Probabilities, batches and evaluation

Represent probability availability as a typed distinction:

```rust,ignore
enum Propensity {
    Exact(Probability),
    Unavailable(PropensityReason),
}
```

The number is under the full behavior policy given decision-time information,
before the random draw. A known PRNG seed does not turn randomized exploration
into an OPE point mass. Conversely deterministic selection has chosen probability
one and zero support elsewhere. Initialization rules belong to that policy too.

The new Boltzmann adapter samples a single authoritative categorical distribution
using the runtime-owned RNG and records that same distribution. The current
implementation uses `drawset::gumbel_max_sample` without an owned RNG and computes
diagnostic probabilities separately; it cannot be wrapped unchanged and promised
exact RNG continuation. Preserve its low-level API; make this new adapter's
sampling/precision change explicit and test it before enabling full checkpoints.

Posterior-max Thompson can report `Unavailable`; its sampled scores are not
action probabilities. Mean-softmax Thompson is a different policy and must
remain named as such. Scores never enter a probability field.

Composition records the final distribution actually sampled after eligibility
and policy stages. If a random mixture is used, log its marginal action
probability, summing contributions from every branch capable of selecting that
action. Do not log just the probability conditional on the selected branch.

For an ordered batch, the product of per-draw probabilities conditional on the
previous selections is the ordered joint probability. It is not a marginal
inclusion probability or an unordered-set probability. The first quality batch
adapter reports probability unavailable; a future categorical batch adapter
must retain the conditional distributions. Scalar OPE rejects batch receipts.

An OPE projection requires one confirmed execution, a final scalar reward,
exact logged propensity, resolvable decision-time context and eligible set,
and target-policy probability for the same action. The caller must establish
target support across the evaluation domain; observed rows alone cannot prove
support for unobserved actions. Missing labels require a separately justified
observation model. Report exclusions and reasons; dropping missing outcomes
does not make an unbiased cohort. Keep IPS/SNIPS unchanged as low-level helpers.

## 7. Persistence and deterministic replay

Do not label today's `RouterSnapshot` a complete continuation checkpoint: its
restore rebuilds triage. Preserve that legacy behavior.

A complete runtime checkpoint includes catalogue and revisions, sequence state,
pending tickets, terminal deduplication records, every supported learner and
monitor, and RNG continuation or documented deterministic draw state.
Checkpoint support is an optional capability; an external model may supply
an immutable reference rather than serialize itself. Incomplete restorations
are named statistical warm starts and invalidate pending handles.

The quality checkpoint uses a versioned schema, crate version and caller build
key. Capture borrows a quiesced runtime; the application owns persistence and
single-writer transfer. Restore validates before reserving the stored engine ID.
Local namespace allocation is monotone, so restore rejects an ID previously
allocated in that process; it does not prevent a copied checkpoint from being
resumed by two different processes. Use the consuming in-memory checkpoint for
same-process continuation.

Replaying accepted events in recorded received order against a compatible
checkpoint must reproduce state and subsequent decisions. This is a same-build
conformance gate, not an unqualified promise of bitwise stability across Rust,
dependency, algorithm or schema versions.
