# Shared decision lifecycle

The unreleased `Muxer<P>` API adds one owner for issuance, eligibility,
decision-time learning evidence and delayed feedback. Existing `Router`,
`BanditPolicy`, `Decision`, snapshot and standalone statistical APIs remain
available. This is an additive migration, not a deprecation announcement.

## Choose the evidence, then the policy

| Evidence supplied by the application | Profile | Feedback |
| --- | --- | --- |
| Binary success/failure | `BernoulliThompson` | `bool` |
| Bounded fractional score | `FractionalThompson` | `BoundedReward` |
| Bounded adversarial reward | `Exp3Profile` | `BoundedReward` |
| Fixed-schema feature vector | `ContextualProfile` | `BoundedReward` |
| Finite scalar reward | `BoltzmannProfile` | `FiniteReward` |
| Quality, cost and latency | `QualityProfile` | `QualityFeedback` |
| Scores, metric assessments or masses from external inference | `ExternalScores`, `ExternalAssessments`, `ExternalDistribution` | None |

Thompson and EXP3 require `stochastic`; LinUCB requires `contextual`;
Boltzmann requires `boltzmann`. Quality and external adapters work without
default features. Fractional Beta updates are pseudo-count updates, not an
exact Bernoulli posterior for arbitrary real scores.

## Migration rules

- Replace arm-only delayed updates with `decide` followed by `tell(id, value)`.
  Keep the immutable receipt until the application finishes execution and
  feedback delivery. The runtime retains its own decision-time ticket.
- Use `decide_from` for request-local readiness. Subset order is canonicalized
  to registered order; unknown and repeated actions fail before issuance.
- An issued decision does not mean an action executed. Cancellation and expiry
  close outstanding work without inventing a zero reward.
- For an ordered batch, use `cancel_item(id, position)` to cancel an open item
  that has accepted no provisional or final value. Its siblings remain live;
  missing-only evidence is retained. Repeated cancellation is idempotent.
  `item_status` distinguishes open, completed, cancelled and expired items,
  including after the batch closes. A batch whose items all resolved or were
  individually cancelled has overall `TerminalStatus::Completed`; this is not
  an assertion that every item executed. Whole-decision expiry preserves the
  statuses of already completed or cancelled siblings.
- `QualityProfile::with_delayed_score()` explicitly retains the score channel.
  Default quality expects execution only. Scores join the original retained
  execution row; they never add a second execution or replay categorical drift
  observations. Legacy Router correction setters keep their existing semantics.
  Immediate mode preserves an embedded `Outcome::with_quality` score; delayed
  mode rejects an embedded score so execution and score cannot silently disagree.
- Retention limits count entries, not arbitrary user payload bytes. Pending
  capacity errors do not evict open work. Terminal retention bounds duplicate
  detection and evaluation access; export application records before eviction.
- Keep historical `Router::observe*` input separate from issued interactions.
  Historical observations do not acquire fabricated behavior probabilities.

The application still owns execution, retries, scheduling, locks, feature
generation and external model storage. No network or training dependency is
needed to adapt an externally trained scorer.

The quality profile and legacy Router share the same observation and score
mutation routines. Quality feedback prepares an opaque, single-use update;
it does not clone the Router or its score-tracking maps. Direct trait users must
apply a prepared update to the same unchanged policy state. Runtime calls
enforce that sequence internally. Terminal promotion moves each item's retained
evidence through one shared conversion, rather than copying parallel records.

## Revisions, advanced feedback and continuation

`replace_policy_with_revisions(new_policy, model_revision, representation_revision)`
starts a new policy epoch. Old receipts continue updating their original policy,
including normalization and cleanup. Equal feature dimensions do not establish
representation compatibility: provide a fresh contextual head when replacing
embeddings. `replace_policy` also creates a new epoch, but leaves the explicit
model/representation labels unchanged. The catalogue is immutable per runtime.

Prior policies stay retained while either pending or terminal receipts reference
them, so duplicate normalization uses the original policy too. Defaults allow
1,024 pending decisions, 1,024 terminal decisions, 32 accepted events per decision
and four retired epochs. `EpochCapacity` fails before replacement. Application
code can explicitly `forget_terminal(id)` after exporting a completed record;
this relinquishes duplicate detection and can release its old epoch. Event limits
are shared across batch items, and admission reserves enough slots for each
declared channel's first resolution. Additional provisional revisions can still
exhaust the event limit, requiring expiry rather than silently dropping history.

`FeedbackEvent` adds an event ID, execution position, actual action, channel,
producer revision, value revision and optional Unix-millisecond observation
time. `submit` normalizes before comparing retained envelopes. Provisionals do
not train. Same-ID changes conflict; lower revisions are stale; finalized and
missing channels never reopen. A source/schema change needs a separately named
declared channel. Runtime `received_sequence` is the replay ordering key, not
the caller's timestamp. Receipts expose immutable per-item `DecisionReason`
diagnostics separately from probability availability.

`into_checkpoint()` consumes the runtime and `Muxer::restore(checkpoint)` resumes
its complete in-memory state, including RNG, pending tickets, retired policies,
event ledgers and triage. It supports move-only external models without forking
decision IDs. Automatic external model loading and durable exactly-once delivery
remain application responsibilities. Legacy statistical snapshots are still
warm starts, not complete runtime checkpoints.

As a restart prerequisite, `TrialRng::state()` captures a versioned SplitMix64
position, serializable with `serde`; `TrialRng::from_state` rejects unsupported
versions. This is non-cryptographic state, not an engine identity generator.
Both Thompson profiles now own such a stream, initialized by their `with_seed`
argument and committed only with accepted issuance. Their unreleased seeded
choice sequences therefore change from the initial `StdRng`-backed adapter;
legacy `ThompsonSampling` seeded behavior is unchanged. Restoring distribution
choices additionally requires the same compatible build and policy state,
not just matching RNG bits. RNG state alone is not a complete profile checkpoint.

With `serde`, `Router::checkpoint(build_key)` and
`Router::from_checkpoint(checkpoint, expected_build_key)` now preserve the
underlying Router's complete windows, monitoring history, live triage detectors,
sticky alarms and coverage cells. The opaque `RouterCheckpoint` is distinct
from `RouterSnapshot`, which still deliberately resets triage on warm start.

With `serde` and `stochastic`, `Muxer<BernoulliThompson>::bernoulli_checkpoint`
captures `BernoulliMuxerCheckpoint`; `Muxer::from_bernoulli_checkpoint` restores
it. This includes configuration, posterior, profile RNG, the original legacy
kernel seed, pending feedback and retained epochs. It shares the runtime graph
validator with quality; it does not enable serialization for other profiles.
The same build compatibility and external writer-fencing requirements below
apply to both concrete checkpoint types.
If a caller uses the low-level policy trait to inject an advanced or
differently seeded Thompson kernel, Bernoulli capture rejects that unsupported
kernel RNG state explicitly. Ordinary `Muxer` issuance leaves the kernel-owned
RNG untouched; its separate profile RNG advances and is captured in full.

For the complete quality lifecycle, `Muxer<QualityProfile>::quality_checkpoint`
captures an opaque, serde-enabled `QualityMuxerCheckpoint` and
`Muxer::from_quality_checkpoint(checkpoint, expected_build_key)` restores it.
This includes pending execution/score joins, retired policy epochs, immutable
receipts, per-item cancellation/finality, retained event history, terminal
eviction order and runtime RNG. Profiles other than quality and Bernoulli use the consuming
in-memory handoff; this is not a generic custom-policy serialization contract.

Capture borrows the runtime and leaves it usable if validation or later encoding
fails. The application must quiesce issuance/feedback at the capture boundary,
persist the checkpoint, and transfer single-writer ownership before restoring.
Restore reserves the saved engine namespace locally and rejects IDs already
allocated in that process, even if the previous runtime was dropped. Prefer
the move-only in-memory handoff for same-process continuation. Reservations
prevent local ID collisions, not copied-file forks across processes.

The checkpoint envelope checks its schema version, crate version and a nonempty
application-supplied build key. Use a key identifying compatible application,
dependency, feature and target builds. It is not authentication, integrity
protection, or single-writer fencing. The application owns those boundaries,
storage and input-size limits. Use a serializer that preserves floating-point
bits; the process-boundary JSON test enables `serde_json/float_roundtrip` and
uses finite JSON-compatible configuration. Formats that cannot represent a
configured infinity cannot be used for that quality state. Bernoulli encodes
configuration and posterior floating-point values as integer bit patterns;
its strict checkpoint path rejects nonpositive or nonfinite posterior values
rather than silently dropping them like the legacy warm-start restore.

Complete quality checkpoint decoding rejects misaligned IDs and noncanonical outcomes
instead of invoking legacy repair behavior. Restore validates configuration,
arm membership, retained identities and detector/coverage/count consistency.
It preserves independently seeded warm-start window histories rather than
requiring primary and monitoring windows to share an identical suffix.

## Probability and evaluation

Receipts distinguish an exact selected-action probability from `Unavailable`.
Posterior-max Thompson has no exact propensity here. Contextual softmax is a
different allocation rule from deterministic LinUCB. The new Boltzmann adapter
uses the runtime's transactional random stream and one categorical distribution;
the legacy drawset-based sampler is unchanged.

`project_logged_reward` reads retained final feedback, checks execution and
single-action probability, then calls the application's target-policy lookup
with the original receipt. The lookup must resolve the original context and
eligible actions, not substitute a new embedding or current availability set.
It produces the existing `LoggedReward` type for `ips_value` or
`self_normalized_ips_value`.

`EvaluationCohort` counts exclusions explicitly. An absent label is not zero;
filtering it out can bias an estimate. An exact observed-action propensity does
not establish target-policy support over unobserved actions. Ordered batches
are excluded from scalar OPE, and no confidence interval or missingness
correction is implied.

## Examples and executable contracts

See the [performance measurements](UNIFIED_MUXER_PERFORMANCE.md) for the
measured lifecycle cost and the next optimization gate.

```bash
cargo run --example unified_bernoulli
cargo run --example unified_quality --no-default-features
cargo run --example unified_external --no-default-features
cargo run --example unified_contextual --no-default-features --features contextual
cargo run --example unified_evaluation --no-default-features --features contextual
cargo run --example unified_refresh --no-default-features --features contextual
cargo test --test evaluation_projection --no-default-features
```

The evaluation test uses a synthetic environment with known full-information
truth: action `a` always succeeds, action `b` fails, the logger favors `a`, and
the target is uniform. IPS recovers the target's value while the naive logged
reward mean reflects the logger's selection bias. This fixture is not a claim
that real bandit logs contain unobserved outcomes.

The [architecture](design/unified-muxer-architecture.md),
[protocol](design/unified-muxer-protocol.md) and
[roadmap](design/unified-muxer-roadmap.md) record the accepted design and gates.
