//! Correlated decision issuance and delayed-feedback runtime.

use crate::interaction::{
    Channel, DecisionId, EngineId, EventDisposition, ExecutionKey, FeedbackExpectation,
    ProbabilityAvailability,
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

#[path = "runtime/events.rs"]
mod events;
pub use events::{EventOutcome, FeedbackEvent};

static NEXT_ENGINE: AtomicU64 = AtomicU64::new(1);

/// A deterministic random stream whose proposed position is committed only on issue.
#[derive(Debug, Clone)]
pub struct TrialRng {
    state: u64,
}
impl TrialRng {
    /// Construct a deterministic trial stream.
    #[must_use]
    pub const fn seeded(seed: u64) -> Self {
        Self { state: seed }
    }
    /// Draw the next uniformly distributed bits.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    /// Draw a unit interval variate in `[0, 1)`.
    pub fn unit_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64))
    }
    /// Draw an integer below `upper`.
    pub fn below(&mut self, upper: usize) -> Option<usize> {
        let upper = u64::try_from(upper).ok()?;
        if upper == 0 {
            return None;
        }
        let threshold = u64::MAX - u64::MAX % upper;
        loop {
            let draw = self.next_u64();
            if draw < threshold {
                return Some((draw % upper) as usize);
            }
        }
    }
}

/// Borrowed request passed to a policy after runtime validation.
pub struct PolicyRequest<'a, C: ?Sized> {
    eligible: &'a [String],
    context: &'a C,
}
impl<'a, C: ?Sized> PolicyRequest<'a, C> {
    pub(crate) fn new(eligible: &'a [String], context: &'a C) -> Self {
        Self { eligible, context }
    }
    /// Authoritative, canonical-order eligible actions.
    #[must_use]
    pub fn eligible(&self) -> &'a [String] {
        self.eligible
    }
    /// Decision-time context.
    #[must_use]
    pub fn context(&self) -> &'a C {
        self.context
    }
}

/// Policy output prepared without mutating observable policy state.
pub struct PolicyDecision<T, I> {
    /// Selected action. It must appear exactly once in request eligibility.
    pub selection: String,
    /// Propensity for the selected action, when exact.
    pub probability: ProbabilityAvailability,
    /// Learning ticket retained until final feedback; `None` is feedbackless.
    pub ticket: Option<T>,
    /// Opaque issuance effect, applied only after runtime validation.
    pub issue: I,
    /// Finite feedback contract for this selected item.
    pub expectation: FeedbackExpectation,
}

/// One prepared item in an ordered-distinct batch.
pub struct BatchSelection<T> {
    /// Selected action, unique within the batch.
    pub selection: String,
    /// Selected action propensity, when exact for this batch unit.
    pub probability: ProbabilityAvailability,
    /// Learning ticket for this item, or none for feedbackless profiles.
    pub ticket: Option<T>,
    /// Expected finite feedback channels.
    pub expectation: FeedbackExpectation,
}

/// Policy output for an explicitly ordered distinct batch.
pub struct PolicyBatchDecision<T, I> {
    /// Ordered distinct selected items.
    pub selections: Vec<BatchSelection<T>>,
    /// Opaque issuance effect committed only after runtime validation.
    pub issue: I,
}

/// Errors returned by policy implementations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyError(String);
impl PolicyError {
    /// Construct an explanatory policy error.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
    /// Error text.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.0
    }
}
impl fmt::Display for PolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for PolicyError {}

/// Minimal extension point for algorithms with distinct learning evidence.
pub trait InteractionPolicy {
    /// Borrowed decision context.
    type Context: ?Sized;
    /// Application feedback accepted by the profile.
    type Feedback;
    /// Validated canonical feedback. Equality drives deduplication.
    type CanonicalFeedback: Eq + Clone;
    /// Immutable decision-time learning evidence.
    type Ticket;
    /// Prepared mutation of issuance state.
    type PreparedIssue;
    /// Prepared mutation of learning state.
    type PreparedUpdate;
    /// Prepare a choice without observable mutation.
    fn prepare_decision(
        &self,
        request: PolicyRequest<'_, Self::Context>,
        rng: &mut TrialRng,
    ) -> Result<PolicyDecision<Self::Ticket, Self::PreparedIssue>, PolicyError>;
    /// Commit a validated issue.
    fn commit_issue(&mut self, issue: Self::PreparedIssue);
    /// Describe the actual selection mechanism without inferring a probability.
    fn decision_reason(&self, _ticket: Option<&Self::Ticket>) -> crate::DecisionReason {
        crate::DecisionReason::Unspecified
    }
    /// Validate and canonicalize application feedback.
    fn normalize(&self, feedback: Self::Feedback) -> Result<Self::CanonicalFeedback, PolicyError>;
    /// Name the feedback channel; scalar profiles use `reward` by default.
    fn feedback_channel(&self, _feedback: &Self::CanonicalFeedback) -> Channel {
        Channel::reward()
    }
    /// Prepare a complete update without mutation.
    fn prepare(
        &self,
        ticket: &Self::Ticket,
        feedback: &Self::CanonicalFeedback,
    ) -> Result<Self::PreparedUpdate, PolicyError>;
    /// Report the semantic effect of a prepared update.
    fn update_disposition(&self, _update: &Self::PreparedUpdate) -> EventDisposition {
        EventDisposition::Accepted
    }
    /// Apply a previously validated update without fallible work or I/O.
    fn apply(&mut self, update: Self::PreparedUpdate);
    /// Discard policy-owned state for an expired, still-open learning ticket.
    fn expire_ticket(&mut self, _ticket: &Self::Ticket) {}
    /// Discard policy-owned state for an explicitly missing channel.
    fn missing_channel(&mut self, _ticket: &Self::Ticket, _channel: &Channel) {}
    /// Release policy-owned state when a ticket resolves or is cancelled.
    fn finish_ticket(&mut self, _ticket: &Self::Ticket) {}
}

/// Optional capability for policies that can issue an ordered distinct batch.
pub trait BatchInteractionPolicy: InteractionPolicy {
    /// Prepared mutation of batch issuance state.
    type PreparedBatchIssue;
    /// Prepare exactly `count` ordered distinct selections without mutation.
    fn prepare_batch(
        &self,
        request: PolicyRequest<'_, Self::Context>,
        count: usize,
        rng: &mut TrialRng,
    ) -> Result<PolicyBatchDecision<Self::Ticket, Self::PreparedBatchIssue>, PolicyError>;
    /// Commit validated batch issuance state.
    fn commit_batch(&mut self, issue: Self::PreparedBatchIssue);
}

/// Bounded retention configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeConfig {
    /// Maximum open decisions.
    pub pending_capacity: usize,
    /// Recently closed decision records.
    pub terminal_capacity: usize,
    /// Accepted events per retained decision.
    pub event_capacity: usize,
    /// Maximum prior policy epochs retained for delayed feedback.
    pub retired_epoch_capacity: usize,
    /// Seed for the trial RNG.
    pub seed: u64,
}
impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            pending_capacity: 1024,
            terminal_capacity: 1024,
            event_capacity: 32,
            retired_epoch_capacity: 4,
            seed: 0,
        }
    }
}

/// Immutable issued decision receipt.
#[derive(Debug, Clone)]
pub struct DecisionReceipt {
    id: DecisionId,
    eligible: Vec<String>,
    selected: Vec<SelectedItem>,
    batch: bool,
    sequence: u64,
    policy_revision: u64,
    model_revision: u64,
    representation_revision: u64,
    config_revision: u64,
    catalogue_revision: u64,
}
#[derive(Debug, Clone)]
struct SelectedItem {
    action: String,
    probability: ProbabilityAvailability,
    reason: crate::DecisionReason,
}
impl DecisionReceipt {
    /// Correlation identity.
    #[must_use]
    pub const fn id(&self) -> DecisionId {
        self.id
    }
    /// Selected action.
    #[must_use]
    pub fn action(&self) -> &str {
        &self.selected[0].action
    }
    /// Ordered selected actions.
    #[must_use]
    pub fn selected(&self) -> impl ExactSizeIterator<Item = &str> {
        self.selected.iter().map(|item| item.action.as_str())
    }
    /// Canonical eligible actions.
    #[must_use]
    pub fn eligible(&self) -> &[String] {
        &self.eligible
    }
    /// Selected action propensity.
    #[must_use]
    pub fn probability(&self) -> ProbabilityAvailability {
        self.selected[0].probability
    }
    /// Mechanism used for the primary selected action.
    #[must_use]
    pub fn reason(&self) -> crate::DecisionReason {
        self.selected[0].reason
    }
    /// Mechanism used for an independently selected batch position.
    #[must_use]
    pub fn item_reason(&self, position: usize) -> Option<crate::DecisionReason> {
        self.selected.get(position).map(|item| item.reason)
    }
    /// Exact or unavailable probability for a selected position.
    #[must_use]
    pub fn item_probability(&self, position: usize) -> Option<ProbabilityAvailability> {
        self.selected.get(position).map(|item| item.probability)
    }
    /// Issuance sequence.
    #[must_use]
    pub const fn sequence(&self) -> u64 {
        self.sequence
    }
    /// Monotone revision of the policy that issued this decision.
    #[must_use]
    pub const fn policy_revision(&self) -> u64 {
        self.policy_revision
    }
    /// Caller-declared model revision at issuance.
    #[must_use]
    pub const fn model_revision(&self) -> u64 {
        self.model_revision
    }
    /// Caller-declared representation revision at issuance.
    #[must_use]
    pub const fn representation_revision(&self) -> u64 {
        self.representation_revision
    }
    /// Runtime configuration revision at issuance.
    #[must_use]
    pub const fn config_revision(&self) -> u64 {
        self.config_revision
    }
    /// Registered catalogue revision at issuance.
    #[must_use]
    pub const fn catalogue_revision(&self) -> u64 {
        self.catalogue_revision
    }
    /// Engine namespace.
    #[must_use]
    pub const fn engine(&self) -> EngineId {
        self.id.engine()
    }
    /// Key for the primary selected item. Use `ExecutionKey::new(id, position)` for a batch item.
    #[must_use]
    pub const fn execution_key(&self) -> ExecutionKey {
        ExecutionKey::new(self.id, 0)
    }
    /// Whether this receipt represents an ordered batch, including an explicit singleton batch.
    #[must_use]
    pub const fn is_batch(&self) -> bool {
        self.batch
    }
    /// Number of independently correlated selected items.
    #[must_use]
    pub fn selected_len(&self) -> usize {
        self.selected.len()
    }
}

struct PendingItem<P: InteractionPolicy> {
    ticket: Option<P::Ticket>,
    expectation: FeedbackExpectation,
    events: Vec<(Channel, P::CanonicalFeedback)>,
    missing: Vec<Channel>,
    missing_reasons: BTreeMap<Channel, String>,
    status: ItemStatus,
}
struct Pending<P: InteractionPolicy> {
    receipt: DecisionReceipt,
    items: Vec<PendingItem<P>>,
    ledger: events::EventLedger<P::CanonicalFeedback>,
}
struct Terminal<P: InteractionPolicy> {
    receipt: DecisionReceipt,
    values: Vec<Vec<(Channel, P::CanonicalFeedback)>>,
    missing_reasons: Vec<BTreeMap<Channel, String>>,
    status: TerminalStatus,
    item_statuses: Vec<ItemStatus>,
    ledger: events::EventLedger<P::CanonicalFeedback>,
}

/// Lifecycle state of one selected item in a receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemStatus {
    /// Awaiting its declared channels.
    Open,
    /// All declared channels resolved final or missing.
    Completed,
    /// Explicitly cancelled before accepting any channel value.
    Cancelled,
    /// Closed by whole-decision expiry before resolution.
    Expired,
}

/// Why a retained interaction record is terminal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalStatus {
    /// Every item resolved its channels or was individually cancelled.
    Completed,
    /// Closed before execution.
    Cancelled,
    /// Closed by caller-driven expiry.
    Expired,
}

/// Errors returned by the lifecycle runtime. Every error leaves policy learning state unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RuntimeError {
    /// No registered actions were supplied.
    EmptyCatalogue,
    /// An action appeared more than once.
    DuplicateAction,
    /// A request named an action absent from the catalogue.
    UnknownAction,
    /// A decision request had no eligible actions.
    EmptyEligibility,
    /// Open-ticket retention is full.
    PendingCapacity,
    /// Terminal retention configuration is invalid.
    TerminalCapacity,
    /// Per-decision feedback retention is full.
    FeedbackCapacity,
    /// The handle belongs to another muxer engine.
    WrongEngine,
    /// The receipt is not retained by this engine.
    UnknownDecision,
    /// The profile does not expect the supplied feedback channel.
    NoFeedbackExpected,
    /// The selected position does not exist in this receipt.
    WrongPosition,
    /// An event claimed an action other than the selected action.
    WrongAction,
    /// An event identifier was reused with different canonical content.
    ConflictingEvent,
    /// An event revision is older than the retained stream revision.
    StaleRevision,
    /// A source revision cannot overwrite a different source stream.
    SourceRevisionMismatch,
    /// A final channel already has a distinct value.
    AlreadyFinalized,
    /// The decision has been cancelled.
    Cancelled,
    /// The decision has expired.
    Expired,
    /// A replacement requires resolving existing tickets first.
    PendingDecisions,
    /// Retained old policy epochs have reached their configured bound.
    EpochCapacity,
    /// Batch size must be nonzero and no larger than the eligible set.
    InvalidBatchSize,
    /// A policy rejected preparation or normalization.
    Policy(PolicyError),
    /// A policy prepared an invalid selection/ticket pairing.
    InvalidPolicyChoice,
}
impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "muxer runtime error: {self:?}")
    }
}
impl std::error::Error for RuntimeError {}
impl From<PolicyError> for RuntimeError {
    fn from(value: PolicyError) -> Self {
        Self::Policy(value)
    }
}

/// Stateful owner of issuance, correlation, and bounded feedback retention.
pub struct Muxer<P: InteractionPolicy> {
    actions: Vec<String>,
    policy: P,
    retired: BTreeMap<u64, P>,
    engine: EngineId,
    next_sequence: u64,
    next_received_sequence: u64,
    policy_revision: u64,
    model_revision: u64,
    representation_revision: u64,
    config_revision: u64,
    catalogue_revision: u64,
    rng: TrialRng,
    config: RuntimeConfig,
    pending: BTreeMap<DecisionId, Pending<P>>,
    terminal: BTreeMap<DecisionId, Terminal<P>>,
    terminal_order: VecDeque<DecisionId>,
}
impl<P: InteractionPolicy> Muxer<P> {
    /// Construct a muxer with default retention.
    pub fn new(actions: Vec<String>, policy: P) -> Result<Self, RuntimeError> {
        Self::with_config(actions, policy, RuntimeConfig::default())
    }
    /// Construct a muxer with explicit bounded retention.
    pub fn with_config(
        actions: Vec<String>,
        policy: P,
        config: RuntimeConfig,
    ) -> Result<Self, RuntimeError> {
        validate_actions(&actions)?;
        if config.pending_capacity == 0
            || config.terminal_capacity == 0
            || config.event_capacity == 0
        {
            return Err(RuntimeError::FeedbackCapacity);
        }
        let raw = NEXT_ENGINE.fetch_add(1, Ordering::Relaxed);
        let engine = EngineId(raw);
        Ok(Self {
            actions,
            policy,
            retired: BTreeMap::new(),
            engine,
            next_sequence: 0,
            next_received_sequence: 0,
            policy_revision: 0,
            model_revision: 0,
            representation_revision: 0,
            config_revision: 0,
            catalogue_revision: 0,
            rng: TrialRng::seeded(config.seed),
            config,
            pending: BTreeMap::new(),
            terminal: BTreeMap::new(),
            terminal_order: VecDeque::new(),
        })
    }
    /// Borrow the registered action catalogue.
    #[must_use]
    pub fn actions(&self) -> &[String] {
        &self.actions
    }
    /// Borrow the policy.
    #[must_use]
    pub fn policy(&self) -> &P {
        &self.policy
    }
    /// Replace the policy while retaining its prior epoch for delayed receipts.
    pub fn replace_policy(&mut self, policy: P) -> Result<(), RuntimeError> {
        self.prune_retired_epochs();
        let retain_current = self
            .pending
            .values()
            .any(|record| record.receipt.policy_revision() == self.policy_revision)
            || self
                .terminal
                .values()
                .any(|record| record.receipt.policy_revision() == self.policy_revision);
        if retain_current && self.retired.len() >= self.config.retired_epoch_capacity {
            return Err(RuntimeError::EpochCapacity);
        }
        let next_revision = self
            .policy_revision
            .checked_add(1)
            .ok_or(RuntimeError::EpochCapacity)?;
        let next_config = self
            .config_revision
            .checked_add(1)
            .ok_or(RuntimeError::EpochCapacity)?;
        let prior = std::mem::replace(&mut self.policy, policy);
        if retain_current {
            self.retired.insert(self.policy_revision, prior);
        }
        self.policy_revision = next_revision;
        self.config_revision = next_config;
        Ok(())
    }
    /// Replace policy and atomically advance model/representation receipt metadata.
    pub fn replace_policy_with_revisions(
        &mut self,
        policy: P,
        model: u64,
        representation: u64,
    ) -> Result<(), RuntimeError> {
        self.replace_policy(policy)?;
        self.model_revision = model;
        self.representation_revision = representation;
        Ok(())
    }
    /// Lookup the policy that issued a retained receipt revision.
    pub(crate) fn policy_at_revision(&self, revision: u64) -> Option<&P> {
        if revision == self.policy_revision {
            Some(&self.policy)
        } else {
            self.retired.get(&revision)
        }
    }
    /// Mutably look up the policy that issued a retained receipt revision.
    pub(crate) fn policy_at_revision_mut(&mut self, revision: u64) -> Option<&mut P> {
        if revision == self.policy_revision {
            Some(&mut self.policy)
        } else {
            self.retired.get_mut(&revision)
        }
    }
    fn prune_retired_epochs(&mut self) {
        let mut live: BTreeSet<u64> = self
            .pending
            .values()
            .map(|pending| pending.receipt.policy_revision())
            .collect();
        live.extend(
            self.terminal
                .values()
                .map(|terminal| terminal.receipt.policy_revision()),
        );
        self.retired.retain(|revision, _| live.contains(revision));
    }
    /// Engine namespace for receipts.
    #[must_use]
    pub const fn engine_id(&self) -> EngineId {
        self.engine
    }
    /// Current number of retained open tickets.
    #[must_use]
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }
    /// Number of prior policy instances retained for pending or terminal evidence.
    #[must_use]
    pub fn retired_epoch_count(&self) -> usize {
        self.retired.len()
    }
    /// Issue from the registered catalogue.
    pub fn decide(&mut self, context: &P::Context) -> Result<DecisionReceipt, RuntimeError> {
        let actions = self.actions.clone();
        self.decide_from(&actions, context)
    }
    /// Issue from a validated subset; registered catalogue order becomes canonical order.
    pub fn decide_from(
        &mut self,
        eligible: &[String],
        context: &P::Context,
    ) -> Result<DecisionReceipt, RuntimeError> {
        let eligible = self.canonical_eligible(eligible)?;
        let mut trial = self.rng.clone();
        let prepared = self
            .policy
            .prepare_decision(PolicyRequest::new(&eligible, context), &mut trial)?;
        if !eligible.iter().any(|action| action == &prepared.selection) {
            return Err(RuntimeError::InvalidPolicyChoice);
        }
        let needs_ticket = !matches!(prepared.expectation, FeedbackExpectation::None);
        if needs_ticket != prepared.ticket.is_some() {
            return Err(RuntimeError::InvalidPolicyChoice);
        }
        validate_expectation(&prepared.expectation)?;
        if prepared.expectation.channels().len() > self.config.event_capacity {
            return Err(RuntimeError::FeedbackCapacity);
        }
        if matches!(prepared.probability, ProbabilityAvailability::Exact(value) if value.get() == 0.0)
        {
            return Err(RuntimeError::InvalidPolicyChoice);
        }
        if needs_ticket && self.pending.len() >= self.config.pending_capacity {
            return Err(RuntimeError::PendingCapacity);
        }
        let id = DecisionId::new(
            self.engine,
            self.next_sequence
                .checked_add(1)
                .ok_or(RuntimeError::PendingCapacity)?,
        );
        let reason = self.policy.decision_reason(prepared.ticket.as_ref());
        let receipt = DecisionReceipt {
            id,
            eligible,
            selected: vec![SelectedItem {
                action: prepared.selection,
                probability: prepared.probability,
                reason,
            }],
            batch: false,
            sequence: id.sequence(),
            policy_revision: self.policy_revision,
            model_revision: self.model_revision,
            representation_revision: self.representation_revision,
            config_revision: self.config_revision,
            catalogue_revision: self.catalogue_revision,
        };
        self.policy.commit_issue(prepared.issue);
        self.rng = trial;
        self.next_sequence = id.sequence();
        if let Some(ticket) = prepared.ticket {
            self.pending.insert(
                id,
                Pending {
                    receipt: receipt.clone(),
                    items: vec![PendingItem {
                        ticket: Some(ticket),
                        expectation: prepared.expectation,
                        events: Vec::new(),
                        missing: Vec::new(),
                        missing_reasons: BTreeMap::new(),
                        status: ItemStatus::Open,
                    }],
                    ledger: events::EventLedger::default(),
                },
            );
        } else {
            self.insert_terminal(
                id,
                Terminal {
                    receipt: receipt.clone(),
                    values: vec![Vec::new()],
                    missing_reasons: vec![BTreeMap::new()],
                    status: TerminalStatus::Completed,
                    item_statuses: vec![ItemStatus::Completed],
                    ledger: events::EventLedger::default(),
                },
            );
        }
        Ok(receipt)
    }
    /// Issue an explicitly ordered distinct batch from the registered catalogue.
    pub fn decide_batch(
        &mut self,
        count: usize,
        context: &P::Context,
    ) -> Result<DecisionReceipt, RuntimeError>
    where
        P: BatchInteractionPolicy,
    {
        let actions = self.actions.clone();
        self.decide_batch_from(&actions, count, context)
    }
    /// Issue an explicitly ordered distinct batch from an authoritative subset.
    pub fn decide_batch_from(
        &mut self,
        eligible: &[String],
        count: usize,
        context: &P::Context,
    ) -> Result<DecisionReceipt, RuntimeError>
    where
        P: BatchInteractionPolicy,
    {
        let eligible = self.canonical_eligible(eligible)?;
        if count == 0 || count > eligible.len() {
            return Err(RuntimeError::InvalidBatchSize);
        }
        let mut trial = self.rng.clone();
        let prepared =
            self.policy
                .prepare_batch(PolicyRequest::new(&eligible, context), count, &mut trial)?;
        if prepared.selections.len() != count {
            return Err(RuntimeError::InvalidPolicyChoice);
        }
        let mut seen = BTreeSet::new();
        let mut items = Vec::with_capacity(count);
        let mut selected = Vec::with_capacity(count);
        for selection in prepared.selections {
            if !eligible.iter().any(|action| action == &selection.selection)
                || !seen.insert(selection.selection.clone())
            {
                return Err(RuntimeError::InvalidPolicyChoice);
            }
            if !matches!(selection.expectation, FeedbackExpectation::None)
                != selection.ticket.is_some()
            {
                return Err(RuntimeError::InvalidPolicyChoice);
            }
            validate_expectation(&selection.expectation)?;
            if selection.expectation.channels().len() > self.config.event_capacity {
                return Err(RuntimeError::FeedbackCapacity);
            }
            if matches!(selection.probability, ProbabilityAvailability::Exact(value) if value.get() == 0.0)
            {
                return Err(RuntimeError::InvalidPolicyChoice);
            }
            let reason = self.policy.decision_reason(selection.ticket.as_ref());
            selected.push(SelectedItem {
                action: selection.selection,
                probability: selection.probability,
                reason,
            });
            items.push(selection.ticket.map(|ticket| PendingItem {
                ticket: Some(ticket),
                expectation: selection.expectation,
                events: Vec::new(),
                missing: Vec::new(),
                missing_reasons: BTreeMap::new(),
                status: ItemStatus::Open,
            }));
        }
        let needs_ticket = items.iter().any(Option::is_some);
        if needs_ticket && self.pending.len() >= self.config.pending_capacity {
            return Err(RuntimeError::PendingCapacity);
        }
        let required_channels: usize = items
            .iter()
            .flatten()
            .map(|item| item.expectation.channels().len())
            .sum();
        if required_channels > self.config.event_capacity {
            return Err(RuntimeError::FeedbackCapacity);
        }
        let id = DecisionId::new(
            self.engine,
            self.next_sequence
                .checked_add(1)
                .ok_or(RuntimeError::PendingCapacity)?,
        );
        let receipt = DecisionReceipt {
            id,
            eligible,
            selected,
            batch: true,
            sequence: id.sequence(),
            policy_revision: self.policy_revision,
            model_revision: self.model_revision,
            representation_revision: self.representation_revision,
            config_revision: self.config_revision,
            catalogue_revision: self.catalogue_revision,
        };
        self.policy.commit_batch(prepared.issue);
        self.rng = trial;
        self.next_sequence = id.sequence();
        if needs_ticket {
            let items = items
                .into_iter()
                .map(|item| {
                    item.unwrap_or(PendingItem {
                        ticket: None,
                        expectation: FeedbackExpectation::None,
                        events: Vec::new(),
                        missing: Vec::new(),
                        missing_reasons: BTreeMap::new(),
                        status: ItemStatus::Completed,
                    })
                })
                .collect();
            self.pending.insert(
                id,
                Pending {
                    receipt: receipt.clone(),
                    items,
                    ledger: events::EventLedger::default(),
                },
            );
        } else {
            self.insert_terminal(
                id,
                Terminal {
                    receipt: receipt.clone(),
                    values: vec![Vec::new(); count],
                    missing_reasons: vec![BTreeMap::new(); count],
                    status: TerminalStatus::Completed,
                    item_statuses: vec![ItemStatus::Completed; count],
                    ledger: events::EventLedger::default(),
                },
            );
        }
        Ok(receipt)
    }
    /// Submit final feedback for a one-item decision.
    pub fn tell(
        &mut self,
        id: DecisionId,
        feedback: P::Feedback,
    ) -> Result<EventDisposition, RuntimeError> {
        if self.receipt(id).is_some_and(DecisionReceipt::is_batch) {
            return Err(RuntimeError::WrongPosition);
        }
        self.tell_item(id, 0, feedback)
    }
    /// Submit feedback for an explicitly selected position.
    pub fn tell_item(
        &mut self,
        id: DecisionId,
        position: usize,
        feedback: P::Feedback,
    ) -> Result<EventDisposition, RuntimeError> {
        if id.engine() != self.engine {
            return Err(RuntimeError::WrongEngine);
        }
        if let Some(error) = self.closed_error(id) {
            return Err(error);
        }
        let revision = self
            .receipt(id)
            .ok_or(RuntimeError::UnknownDecision)?
            .policy_revision();
        let origin = self
            .policy_at_revision(revision)
            .ok_or(RuntimeError::PendingDecisions)?;
        let canonical = origin.normalize(feedback)?;
        self.accept_canonical(id, position, canonical)
    }
    /// Apply already-normalized feedback through the issuing policy epoch.
    pub(crate) fn accept_canonical(
        &mut self,
        id: DecisionId,
        position: usize,
        canonical: P::CanonicalFeedback,
    ) -> Result<EventDisposition, RuntimeError> {
        if id.engine() != self.engine {
            return Err(RuntimeError::WrongEngine);
        }
        let revision = self
            .receipt(id)
            .ok_or(RuntimeError::UnknownDecision)?
            .policy_revision();
        let origin = self
            .policy_at_revision(revision)
            .ok_or(RuntimeError::PendingDecisions)?;
        let channel = origin.feedback_channel(&canonical);
        if let Some(done) = self.terminal.get(&id) {
            if done.item_statuses.get(position) == Some(&ItemStatus::Cancelled) {
                return Err(RuntimeError::Cancelled);
            }
            let Some(values) = done.values.get(position) else {
                return Err(RuntimeError::WrongPosition);
            };
            return if values
                .iter()
                .any(|(seen, value)| seen == &channel && value == &canonical)
            {
                Ok(EventDisposition::Duplicate)
            } else {
                Err(RuntimeError::AlreadyFinalized)
            };
        }
        let Some(pending) = self.pending.get(&id) else {
            return Err(RuntimeError::UnknownDecision);
        };
        let Some(item) = pending.items.get(position) else {
            return Err(RuntimeError::WrongPosition);
        };
        if item.status == ItemStatus::Cancelled {
            return Err(RuntimeError::Cancelled);
        }
        if !item.expectation.channels().contains(&channel) {
            return Err(RuntimeError::NoFeedbackExpected);
        }
        if item.missing.iter().any(|missing| missing == &channel) {
            return Err(RuntimeError::AlreadyFinalized);
        }
        if let Some((_, value)) = item.events.iter().find(|(seen, _)| seen == &channel) {
            return if value == &canonical {
                Ok(EventDisposition::Duplicate)
            } else {
                Err(RuntimeError::AlreadyFinalized)
            };
        }
        if self.retained_event_slots(id) >= self.config.event_capacity {
            return Err(RuntimeError::FeedbackCapacity);
        }
        let update = self
            .policy_at_revision(revision)
            .ok_or(RuntimeError::PendingDecisions)?
            .prepare(
                item.ticket.as_ref().ok_or(RuntimeError::AlreadyFinalized)?,
                &canonical,
            )?;
        // No operation below is fallible: update plus terminal promotion is atomic in memory.
        let disposition = self
            .policy_at_revision(revision)
            .ok_or(RuntimeError::PendingDecisions)?
            .update_disposition(&update);
        self.policy_at_revision_mut(revision)
            .ok_or(RuntimeError::PendingDecisions)?
            .apply(update);
        let (complete, finished_ticket) = {
            let pending = self.pending.get_mut(&id).expect("pending checked above");
            let item = &mut pending.items[position];
            item.events.push((channel, canonical));
            let finished = item.expectation.channels().iter().all(|expected| {
                item.events.iter().any(|(seen, _)| seen == expected)
                    || item.missing.contains(expected)
            });
            if finished {
                item.status = ItemStatus::Completed;
            }
            let ticket = if finished { item.ticket.take() } else { None };
            (
                pending.items.iter().all(|item| item.ticket.is_none()),
                ticket,
            )
        };
        if let Some(ticket) = finished_ticket.as_ref() {
            self.policy_at_revision_mut(revision)
                .ok_or(RuntimeError::PendingDecisions)?
                .finish_ticket(ticket);
        }
        if complete {
            let pending = self.pending.remove(&id).expect("pending checked above");
            self.insert_terminal(
                id,
                Terminal {
                    receipt: pending.receipt,
                    values: pending
                        .items
                        .iter()
                        .map(|item| item.events.clone())
                        .collect(),
                    missing_reasons: pending
                        .items
                        .iter()
                        .map(|item| item.missing_reasons.clone())
                        .collect(),
                    status: TerminalStatus::Completed,
                    item_statuses: pending.items.iter().map(|item| item.status).collect(),
                    ledger: pending.ledger,
                },
            );
        }
        Ok(disposition)
    }
    /// Mark one expected channel as explicitly missing without manufacturing a reward.
    pub fn tell_missing(
        &mut self,
        id: DecisionId,
        position: usize,
        channel: Channel,
        reason: impl Into<String>,
    ) -> Result<EventDisposition, RuntimeError> {
        let reason = reason.into();
        if id.engine() != self.engine {
            return Err(RuntimeError::WrongEngine);
        }
        if let Some(error) = self.closed_error(id) {
            return Err(error);
        }
        if let Some(terminal) = self.terminal.get(&id) {
            if terminal.item_statuses.get(position) == Some(&ItemStatus::Cancelled) {
                return Err(RuntimeError::Cancelled);
            }
            let reasons = terminal
                .missing_reasons
                .get(position)
                .ok_or(RuntimeError::WrongPosition)?;
            return match reasons.get(&channel) {
                Some(previous) if previous == &reason => Ok(EventDisposition::Duplicate),
                Some(_) => Err(RuntimeError::ConflictingEvent),
                None => Err(RuntimeError::AlreadyFinalized),
            };
        }
        {
            let pending = self.pending.get(&id).ok_or(RuntimeError::UnknownDecision)?;
            let item = pending
                .items
                .get(position)
                .ok_or(RuntimeError::WrongPosition)?;
            if item.status == ItemStatus::Cancelled {
                return Err(RuntimeError::Cancelled);
            }
            if !item.expectation.channels().contains(&channel) {
                return Err(RuntimeError::NoFeedbackExpected);
            }
            if item.events.iter().any(|(seen, _)| seen == &channel) {
                return Err(RuntimeError::AlreadyFinalized);
            }
            if let Some(old) = item.missing_reasons.get(&channel) {
                return if old == &reason {
                    Ok(EventDisposition::Duplicate)
                } else {
                    Err(RuntimeError::ConflictingEvent)
                };
            }
        }
        if self.retained_event_slots(id) >= self.config.event_capacity {
            return Err(RuntimeError::FeedbackCapacity);
        }
        let pending = self.pending.remove(&id).expect("pending checked above");
        let item = pending.items.get(position).expect("position checked above");
        let revision = pending.receipt.policy_revision();
        if let Some(ticket) = item.ticket.as_ref() {
            self.policy_at_revision_mut(revision)
                .ok_or(RuntimeError::PendingDecisions)?
                .missing_channel(ticket, &channel);
        }
        let mut pending = pending;
        let (complete, finished_ticket) = {
            let item = &mut pending.items[position];
            item.missing.push(channel.clone());
            item.missing_reasons.insert(channel, reason);
            let finished = item.expectation.channels().iter().all(|expected| {
                item.events.iter().any(|(seen, _)| seen == expected)
                    || item.missing.contains(expected)
            });
            if finished {
                item.status = ItemStatus::Completed;
            }
            let ticket = if finished { item.ticket.take() } else { None };
            (
                pending.items.iter().all(|item| item.ticket.is_none()),
                ticket,
            )
        };
        if let Some(ticket) = finished_ticket.as_ref() {
            self.policy_at_revision_mut(revision)
                .ok_or(RuntimeError::PendingDecisions)?
                .finish_ticket(ticket);
        }
        if complete {
            self.insert_terminal(
                id,
                Terminal {
                    receipt: pending.receipt,
                    values: pending
                        .items
                        .iter()
                        .map(|item| item.events.clone())
                        .collect(),
                    missing_reasons: pending
                        .items
                        .iter()
                        .map(|item| item.missing_reasons.clone())
                        .collect(),
                    status: TerminalStatus::Completed,
                    item_statuses: pending.items.iter().map(|item| item.status).collect(),
                    ledger: pending.ledger,
                },
            );
        } else {
            self.pending.insert(id, pending);
        }
        Ok(EventDisposition::Accepted)
    }
    /// Submit an advanced receipt-correlated feedback envelope.
    pub fn submit(
        &mut self,
        event: FeedbackEvent<P::Feedback>,
    ) -> Result<EventOutcome, RuntimeError> {
        self.submit_recorded(event)
    }
    /// Close an unexecuted decision without fabricating feedback.
    pub fn cancel(&mut self, id: DecisionId) -> Result<(), RuntimeError> {
        self.close(id, RuntimeError::Cancelled, false)
    }
    /// Cancel one unobserved selected item without fabricating feedback.
    ///
    /// Provisional or final values prevent cancellation. Missing-only evidence
    /// is retained, and sibling items remain unchanged. Repeated cancellation
    /// of the same retained item succeeds without repeating policy cleanup.
    /// When all items close this way or resolve their channels, the decision is
    /// `TerminalStatus::Completed`; inspect `item_status` for each item's result.
    pub fn cancel_item(&mut self, id: DecisionId, position: usize) -> Result<(), RuntimeError> {
        if id.engine() != self.engine {
            return Err(RuntimeError::WrongEngine);
        }
        if let Some(terminal) = self.terminal.get(&id) {
            return match terminal.item_statuses.get(position) {
                Some(ItemStatus::Cancelled) => Ok(()),
                Some(_) => Err(RuntimeError::AlreadyFinalized),
                None => Err(RuntimeError::WrongPosition),
            };
        }
        {
            let pending = self.pending.get(&id).ok_or(RuntimeError::UnknownDecision)?;
            let item = pending
                .items
                .get(position)
                .ok_or(RuntimeError::WrongPosition)?;
            if item.status == ItemStatus::Cancelled {
                return Ok(());
            }
            if item.status != ItemStatus::Open
                || !item.events.is_empty()
                || pending.ledger.has_value_at(position)
            {
                return Err(RuntimeError::AlreadyFinalized);
            }
            if self
                .policy_at_revision(pending.receipt.policy_revision())
                .is_none()
            {
                return Err(RuntimeError::PendingDecisions);
            }
        }
        let mut pending = self.pending.remove(&id).expect("pending checked above");
        let revision = pending.receipt.policy_revision();
        let ticket = {
            let item = &mut pending.items[position];
            item.status = ItemStatus::Cancelled;
            item.ticket.take()
        };
        if let Some(ticket) = ticket.as_ref() {
            self.policy_at_revision_mut(revision)
                .ok_or(RuntimeError::PendingDecisions)?
                .finish_ticket(ticket);
        }
        let complete = pending
            .items
            .iter()
            .all(|item| item.status != ItemStatus::Open);
        if complete {
            self.insert_terminal(
                id,
                Terminal {
                    receipt: pending.receipt,
                    values: pending
                        .items
                        .iter()
                        .map(|item| item.events.clone())
                        .collect(),
                    missing_reasons: pending
                        .items
                        .iter()
                        .map(|item| item.missing_reasons.clone())
                        .collect(),
                    status: TerminalStatus::Completed,
                    item_statuses: pending.items.iter().map(|item| item.status).collect(),
                    ledger: pending.ledger,
                },
            );
        } else {
            self.pending.insert(id, pending);
        }
        Ok(())
    }
    /// Expire an open decision without synthesizing feedback.
    pub fn expire(&mut self, id: DecisionId) -> Result<(), RuntimeError> {
        self.close(id, RuntimeError::UnknownDecision, true)
    }
    fn close(
        &mut self,
        id: DecisionId,
        terminal_error: RuntimeError,
        expire: bool,
    ) -> Result<(), RuntimeError> {
        if id.engine() != self.engine {
            return Err(RuntimeError::WrongEngine);
        }
        let Some(pending) = self.pending.get(&id) else {
            return if self.terminal.contains_key(&id) {
                Err(terminal_error)
            } else {
                Err(RuntimeError::UnknownDecision)
            };
        };
        if !expire
            && (pending.items.iter().any(|item| !item.events.is_empty())
                || pending.ledger.has_values())
        {
            return Err(RuntimeError::AlreadyFinalized);
        }
        let pending = self.pending.remove(&id).expect("pending checked above");
        let revision = pending.receipt.policy_revision();
        for item in &pending.items {
            if let Some(ticket) = item.ticket.as_ref() {
                if expire {
                    self.policy_at_revision_mut(revision)
                        .ok_or(RuntimeError::PendingDecisions)?
                        .expire_ticket(ticket);
                } else {
                    self.policy_at_revision_mut(revision)
                        .ok_or(RuntimeError::PendingDecisions)?
                        .finish_ticket(ticket);
                }
            }
        }
        let status = if expire {
            TerminalStatus::Expired
        } else {
            TerminalStatus::Cancelled
        };
        self.insert_terminal(
            id,
            Terminal {
                receipt: pending.receipt,
                values: pending
                    .items
                    .iter()
                    .map(|item| item.events.clone())
                    .collect(),
                missing_reasons: pending
                    .items
                    .iter()
                    .map(|item| item.missing_reasons.clone())
                    .collect(),
                status,
                item_statuses: pending
                    .items
                    .iter()
                    .map(|item| {
                        if expire && item.status == ItemStatus::Open {
                            ItemStatus::Expired
                        } else if !expire && item.status == ItemStatus::Open {
                            ItemStatus::Cancelled
                        } else {
                            item.status
                        }
                    })
                    .collect(),
                ledger: pending.ledger,
            },
        );
        Ok(())
    }
    fn insert_terminal(&mut self, id: DecisionId, terminal: Terminal<P>) {
        self.terminal.insert(id, terminal);
        self.terminal_order.push_back(id);
        while self.terminal_order.len() > self.config.terminal_capacity {
            if let Some(old) = self.terminal_order.pop_front() {
                self.terminal.remove(&old);
            }
        }
        if !self.retired.is_empty() {
            self.prune_retired_epochs();
        }
    }
    fn canonical_eligible(&self, eligible: &[String]) -> Result<Vec<String>, RuntimeError> {
        if eligible.is_empty() {
            return Err(RuntimeError::EmptyEligibility);
        }
        let mut requested = BTreeSet::new();
        for action in eligible {
            if !requested.insert(action) {
                return Err(RuntimeError::DuplicateAction);
            }
            if !self.actions.iter().any(|known| known == action) {
                return Err(RuntimeError::UnknownAction);
            }
        }
        Ok(self
            .actions
            .iter()
            .filter(|action| requested.contains(*action))
            .cloned()
            .collect())
    }
    /// Return a receipt while its record is retained.
    #[must_use]
    pub fn receipt(&self, id: DecisionId) -> Option<&DecisionReceipt> {
        if id.engine() != self.engine {
            return None;
        }
        self.pending
            .get(&id)
            .map(|record| &record.receipt)
            .or_else(|| self.terminal.get(&id).map(|record| &record.receipt))
    }
    /// Return retained final canonical feedback for a selected item/channel.
    #[must_use]
    pub fn final_feedback(
        &self,
        id: DecisionId,
        position: usize,
        channel: &Channel,
    ) -> Option<&P::CanonicalFeedback> {
        if id.engine() != self.engine {
            return None;
        }
        self.pending
            .get(&id)
            .and_then(|record| record.items.get(position))
            .and_then(|item| {
                item.events
                    .iter()
                    .find(|(seen, _)| seen == channel)
                    .map(|(_, value)| value)
            })
            .or_else(|| {
                self.terminal
                    .get(&id)
                    .and_then(|record| record.values.get(position))
                    .and_then(|values| {
                        values
                            .iter()
                            .find(|(seen, _)| seen == channel)
                            .map(|(_, value)| value)
                    })
            })
    }
    /// Retained reason for an explicitly missing channel.
    #[must_use]
    pub fn missing_reason(
        &self,
        id: DecisionId,
        position: usize,
        channel: &Channel,
    ) -> Option<&str> {
        self.pending
            .get(&id)
            .and_then(|pending| pending.items.get(position))
            .and_then(|item| item.missing_reasons.get(channel))
            .or_else(|| {
                self.terminal
                    .get(&id)
                    .and_then(|terminal| terminal.missing_reasons.get(position))
                    .and_then(|reasons| reasons.get(channel))
            })
            .map(String::as_str)
    }
    /// Status of a retained terminal receipt, or none for open/unknown receipts.
    #[must_use]
    pub fn terminal_status(&self, id: DecisionId) -> Option<TerminalStatus> {
        self.terminal.get(&id).map(|record| record.status)
    }
    /// Status of one selected item while its record is retained.
    #[must_use]
    pub fn item_status(&self, id: DecisionId, position: usize) -> Option<ItemStatus> {
        self.pending
            .get(&id)
            .and_then(|pending| pending.items.get(position))
            .map(|item| item.status)
            .or_else(|| {
                self.terminal
                    .get(&id)
                    .and_then(|terminal| terminal.item_statuses.get(position).copied())
            })
    }

    /// Explicitly release a terminal receipt and its duplicate-detection evidence.
    ///
    /// Open decisions cannot be forgotten. Subsequent feedback for a forgotten
    /// receipt is unknown, and evaluation must use an application-owned export.
    /// This can release an old epoch when its final terminal record is forgotten.
    pub fn forget_terminal(&mut self, id: DecisionId) -> Result<(), RuntimeError> {
        if id.engine() != self.engine {
            return Err(RuntimeError::WrongEngine);
        }
        if self.pending.contains_key(&id) {
            return Err(RuntimeError::PendingDecisions);
        }
        self.terminal
            .remove(&id)
            .ok_or(RuntimeError::UnknownDecision)?;
        self.terminal_order.retain(|known| *known != id);
        self.prune_retired_epochs();
        Ok(())
    }
    /// Closed-state error for a retained cancelled or expired record.
    pub(crate) fn closed_error(&self, id: DecisionId) -> Option<RuntimeError> {
        match self.terminal.get(&id)?.status {
            TerminalStatus::Completed => None,
            TerminalStatus::Cancelled => Some(RuntimeError::Cancelled),
            TerminalStatus::Expired => Some(RuntimeError::Expired),
        }
    }
}
fn validate_actions(actions: &[String]) -> Result<(), RuntimeError> {
    if actions.is_empty() {
        return Err(RuntimeError::EmptyCatalogue);
    }
    let mut seen = BTreeSet::new();
    for action in actions {
        if action.is_empty() || !seen.insert(action) {
            return Err(RuntimeError::DuplicateAction);
        }
    }
    Ok(())
}
fn validate_expectation(expectation: &FeedbackExpectation) -> Result<(), RuntimeError> {
    let channels = expectation.channels();
    if matches!(expectation, FeedbackExpectation::FinalValues(_)) && channels.is_empty() {
        return Err(RuntimeError::InvalidPolicyChoice);
    }
    let mut seen = BTreeSet::new();
    if channels
        .iter()
        .any(|channel| !seen.insert(channel.as_str()))
    {
        return Err(RuntimeError::InvalidPolicyChoice);
    }
    Ok(())
}

/// A complete in-memory handoff containing the policy, RNG, receipts and ledgers.
///
/// This is not a serialized disk checkpoint. It supports move-only external
/// models and prevents namespace forks by consuming the original runtime.
pub struct MuxerCheckpoint<P: InteractionPolicy> {
    runtime: Muxer<P>,
}
impl<P: InteractionPolicy> Muxer<P> {
    /// Consume this runtime into a complete in-memory checkpoint.
    ///
    /// Consumption prevents two live engines from accepting the same retained
    /// receipt namespace after a local checkpoint.
    #[must_use]
    pub fn into_checkpoint(self) -> MuxerCheckpoint<P> {
        MuxerCheckpoint { runtime: self }
    }
    /// Restore a complete checkpoint with its retained identity and open tickets.
    #[must_use]
    pub fn restore(checkpoint: MuxerCheckpoint<P>) -> Self {
        checkpoint.runtime
    }
}
