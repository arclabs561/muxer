//! Shared checkpoint core with concrete built-in profile formats.

use super::{
    events::{EventLedger, EventLedgerCheckpoint},
    DecisionReceipt, InteractionPolicy, ItemStatus, Muxer, Pending, PendingItem, RuntimeConfig,
    SelectedItem, Terminal, TerminalItem, TerminalStatus, TrialRng, TrialRngState,
};
use crate::interaction::{DecisionReason, Disposition};
use crate::profiles::quality::{
    CanonicalQualityFeedback, QualityProfileCheckpoint, QualityTicketCheckpoint,
};
#[cfg(feature = "stochastic")]
use crate::profiles::scalar::{BernoulliThompsonCheckpoint, ScalarTicketCheckpoint};
use crate::{
    Channel, DecisionId, EngineId, FeedbackExpectation, PolicyError, ProbabilityAvailability,
    QualityProfile,
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::atomic::Ordering;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeCheckpoint<State, TicketWire, Canonical> {
    version: u32,
    engine: EngineId,
    actions: Vec<String>,
    policy: State,
    retired: Vec<(u64, State)>,
    next_sequence: u64,
    next_received_sequence: u64,
    policy_revision: u64,
    model_revision: u64,
    representation_revision: u64,
    config_revision: u64,
    catalogue_revision: u64,
    rng: TrialRngState,
    config: RuntimeConfig,
    pending: Vec<PendingCheckpoint<TicketWire, Canonical>>,
    terminal: Vec<TerminalCheckpoint<Canonical>>,
    terminal_order: Vec<DecisionId>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PendingCheckpoint<TicketWire, Canonical> {
    receipt: ReceiptCheckpoint,
    items: Vec<PendingItemCheckpoint<TicketWire, Canonical>>,
    ledger: EventLedgerCheckpoint<Canonical>,
}
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PendingItemCheckpoint<TicketWire, Canonical> {
    ticket: Option<TicketWire>,
    expectation: FeedbackExpectation,
    values: Vec<(Channel, Canonical)>,
    missing: Vec<Channel>,
    missing_reasons: BTreeMap<Channel, String>,
    status: ItemStatus,
}
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TerminalCheckpoint<Canonical> {
    receipt: ReceiptCheckpoint,
    items: Vec<TerminalItemCheckpoint<Canonical>>,
    status: TerminalStatus,
    ledger: EventLedgerCheckpoint<Canonical>,
}
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TerminalItemCheckpoint<Canonical> {
    values: Vec<(Channel, Canonical)>,
    missing_reasons: BTreeMap<Channel, String>,
    status: ItemStatus,
}

/// A build-bound, complete serialized snapshot of a [`Muxer<QualityProfile>`].
///
/// The transparent wrapper preserves the version-1 quality wire document.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct QualityMuxerCheckpoint(
    RuntimeCheckpoint<QualityProfileCheckpoint, QualityTicketCheckpoint, CanonicalQualityFeedback>,
);

/// A build-bound, complete serialized snapshot of a [`Muxer<crate::BernoulliThompson>`].
#[cfg(feature = "stochastic")]
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct BernoulliMuxerCheckpoint(
    RuntimeCheckpoint<BernoulliThompsonCheckpoint, ScalarTicketCheckpoint, bool>,
);

/// Private checked wire form. Keeping this separate prevents untrusted serde
/// input from constructing a public receipt whose convenience accessors assume
/// at least one selection.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ReceiptCheckpoint {
    id: DecisionId,
    eligible: Vec<String>,
    selected: Vec<SelectedCheckpoint>,
    batch: bool,
    sequence: u64,
    policy_revision: u64,
    model_revision: u64,
    representation_revision: u64,
    config_revision: u64,
    catalogue_revision: u64,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SelectedCheckpoint {
    action: String,
    probability: ProbabilityAvailability,
    reason: DecisionReason,
}

impl From<&DecisionReceipt> for ReceiptCheckpoint {
    fn from(receipt: &DecisionReceipt) -> Self {
        Self {
            id: receipt.id,
            eligible: receipt.eligible.clone(),
            selected: receipt
                .selected
                .iter()
                .map(|item| SelectedCheckpoint {
                    action: item.action.clone(),
                    probability: item.probability,
                    reason: item.reason,
                })
                .collect(),
            batch: receipt.batch,
            sequence: receipt.sequence,
            policy_revision: receipt.policy_revision,
            model_revision: receipt.model_revision,
            representation_revision: receipt.representation_revision,
            config_revision: receipt.config_revision,
            catalogue_revision: receipt.catalogue_revision,
        }
    }
}

/// Profile-specific persistence and semantic checks. This remains private so
/// checkpoint portability is only exposed by concrete built-in wrappers.
trait CheckpointProfile: InteractionPolicy + Sized {
    const LABEL: &'static str;
    type State: Clone;
    type TicketWire: Clone;

    fn checkpoint_state(&self, build_key: &str) -> Result<Self::State, PolicyError>;
    fn from_checkpoint_state(state: Self::State, build_key: &str) -> Result<Self, PolicyError>;
    fn encode_ticket(ticket: &Self::Ticket) -> Self::TicketWire;
    fn decode_ticket(ticket: Self::TicketWire) -> Result<Self::Ticket, PolicyError>;
    fn checkpoint_expectation(&self) -> FeedbackExpectation;
    fn validate_receipt_semantics(receipt: &ReceiptCheckpoint) -> Result<(), PolicyError>;
    fn validate_value_channel(&self, channel: &Channel, value: &Self::CanonicalFeedback) -> bool;
    fn validate_open_tickets(&self, tickets: &[OpenTicket<'_, Self>]) -> Result<(), PolicyError>;
}

struct OpenTicket<'a, P: CheckpointProfile> {
    ticket: &'a P::Ticket,
    action: &'a str,
    reason: DecisionReason,
    values: &'a [(Channel, P::CanonicalFeedback)],
    missing: &'a [Channel],
}

type ProfileWire<P> = RuntimeCheckpoint<
    <P as CheckpointProfile>::State,
    <P as CheckpointProfile>::TicketWire,
    <P as InteractionPolicy>::CanonicalFeedback,
>;

impl CheckpointProfile for QualityProfile {
    const LABEL: &'static str = "quality";
    type State = QualityProfileCheckpoint;
    type TicketWire = QualityTicketCheckpoint;
    fn checkpoint_state(&self, build_key: &str) -> Result<Self::State, PolicyError> {
        self.checkpoint_state(build_key)
    }
    fn from_checkpoint_state(state: Self::State, build_key: &str) -> Result<Self, PolicyError> {
        Self::from_checkpoint_state(state, build_key)
    }
    fn encode_ticket(ticket: &Self::Ticket) -> Self::TicketWire {
        QualityTicketCheckpoint::from(ticket)
    }
    fn decode_ticket(ticket: Self::TicketWire) -> Result<Self::Ticket, PolicyError> {
        ticket.into_ticket()
    }
    fn checkpoint_expectation(&self) -> FeedbackExpectation {
        self.checkpoint_expectation()
    }
    fn validate_receipt_semantics(receipt: &ReceiptCheckpoint) -> Result<(), PolicyError> {
        if receipt.selected.iter().any(|item| {
            !matches!(item.probability, ProbabilityAvailability::Unavailable)
                || !matches!(
                    item.reason,
                    DecisionReason::Control
                        | DecisionReason::Triage
                        | DecisionReason::NoveltyOrCoverage
                        | DecisionReason::Policy
                )
        }) {
            return Err(PolicyError::new(
                "quality checkpoint receipt selections are invalid",
            ));
        }
        Ok(())
    }
    fn validate_value_channel(&self, channel: &Channel, value: &Self::CanonicalFeedback) -> bool {
        self.feedback_channel(value) == *channel
    }
    fn validate_open_tickets(&self, tickets: &[OpenTicket<'_, Self>]) -> Result<(), PolicyError> {
        let states: Vec<_> = tickets
            .iter()
            .map(|item| crate::profiles::quality::QualityTicketState {
                ticket: item.ticket,
                action: item.action,
                reason: item.reason,
                values: item.values,
                missing: item.missing,
            })
            .collect();
        self.validate_checkpoint_tickets(&states)
    }
}

#[cfg(feature = "stochastic")]
impl CheckpointProfile for crate::BernoulliThompson {
    const LABEL: &'static str = "bernoulli";
    type State = BernoulliThompsonCheckpoint;
    type TicketWire = ScalarTicketCheckpoint;
    fn checkpoint_state(&self, build_key: &str) -> Result<Self::State, PolicyError> {
        self.checkpoint_state(build_key)
    }
    fn from_checkpoint_state(state: Self::State, build_key: &str) -> Result<Self, PolicyError> {
        Self::from_checkpoint_state(state, build_key)
    }
    fn encode_ticket(ticket: &Self::Ticket) -> Self::TicketWire {
        ScalarTicketCheckpoint::from(ticket)
    }
    fn decode_ticket(ticket: Self::TicketWire) -> Result<Self::Ticket, PolicyError> {
        ticket.into_ticket()
    }
    fn checkpoint_expectation(&self) -> FeedbackExpectation {
        self.checkpoint_expectation()
    }
    fn validate_receipt_semantics(receipt: &ReceiptCheckpoint) -> Result<(), PolicyError> {
        if receipt.batch
            || receipt.selected.iter().any(|item| {
                !matches!(item.probability, ProbabilityAvailability::Unavailable)
                    || !matches!(
                        item.reason,
                        DecisionReason::ExploreFirst | DecisionReason::PosteriorSample
                    )
            })
        {
            return Err(PolicyError::new(
                "bernoulli checkpoint receipt selections are invalid",
            ));
        }
        Ok(())
    }
    fn validate_value_channel(&self, channel: &Channel, _value: &bool) -> bool {
        *channel == Channel::reward()
    }
    fn validate_open_tickets(&self, tickets: &[OpenTicket<'_, Self>]) -> Result<(), PolicyError> {
        let states: Vec<_> = tickets
            .iter()
            .map(|item| crate::profiles::scalar::ScalarTicketState {
                ticket: item.ticket,
                action: item.action,
                reason: item.reason,
                values: item.values,
                missing: item.missing,
            })
            .collect();
        self.validate_checkpoint_tickets(&states)
    }
}

fn capture_core<P>(runtime: &Muxer<P>, build_key: &str) -> Result<ProfileWire<P>, PolicyError>
where
    P: CheckpointProfile,
    P::CanonicalFeedback: Clone,
{
    validate_open_ticket_groups_generic::<P>(
        &runtime.policy,
        &runtime.retired,
        runtime.policy_revision,
        &runtime.pending,
    )?;
    let policy = P::checkpoint_state(&runtime.policy, build_key)?;
    let retired = runtime
        .retired
        .iter()
        .map(|(revision, profile)| Ok((*revision, P::checkpoint_state(profile, build_key)?)))
        .collect::<Result<Vec<_>, PolicyError>>()?;
    Ok(RuntimeCheckpoint {
        version: 1,
        engine: runtime.engine,
        actions: runtime.actions.clone(),
        policy,
        retired,
        next_sequence: runtime.next_sequence,
        next_received_sequence: runtime.next_received_sequence,
        policy_revision: runtime.policy_revision,
        model_revision: runtime.model_revision,
        representation_revision: runtime.representation_revision,
        config_revision: runtime.config_revision,
        catalogue_revision: runtime.catalogue_revision,
        rng: runtime.rng.state(),
        config: runtime.config,
        pending: runtime
            .pending
            .values()
            .map(|record| PendingCheckpoint {
                receipt: ReceiptCheckpoint::from(&record.receipt),
                items: record
                    .items
                    .iter()
                    .map(|item| PendingItemCheckpoint {
                        ticket: item.ticket.as_ref().map(P::encode_ticket),
                        expectation: item.expectation.clone(),
                        values: item.events.clone(),
                        missing: item.missing.clone(),
                        missing_reasons: item.missing_reasons.clone(),
                        status: item.status,
                    })
                    .collect(),
                ledger: EventLedgerCheckpoint::from(&record.ledger),
            })
            .collect(),
        terminal: runtime
            .terminal
            .values()
            .map(|record| TerminalCheckpoint {
                receipt: ReceiptCheckpoint::from(&record.receipt),
                items: record
                    .items
                    .iter()
                    .map(|item| TerminalItemCheckpoint {
                        values: item.values.clone(),
                        missing_reasons: item.missing_reasons.clone(),
                        status: item.status,
                    })
                    .collect(),
                status: record.status,
                ledger: EventLedgerCheckpoint::from(&record.ledger),
            })
            .collect(),
        terminal_order: runtime.terminal_order.iter().copied().collect(),
    })
}

impl Muxer<QualityProfile> {
    /// Capture a complete, build-bound quality runtime checkpoint.
    ///
    /// `build_key` is a non-empty exact compatibility marker for the caller's
    /// application build, dependency features, and target representation. It
    /// is not authentication and does not provide a writer fence. This method
    /// borrows the live muxer, so callers must quiesce issuance and feedback
    /// before capturing and keep a single external writer while copying it.
    /// A validation error leaves the live muxer unchanged.
    ///
    /// For deterministic continuation, choose a serializer that preserves
    /// floating-point bits. JSON round-trips ordinary finite values but cannot
    /// represent configured `NaN` or infinity values; serde compatibility alone
    /// is not a replay guarantee across serializer choices.
    pub fn quality_checkpoint(
        &self,
        build_key: &str,
    ) -> Result<QualityMuxerCheckpoint, PolicyError> {
        Ok(QualityMuxerCheckpoint(capture_core(self, build_key)?))
    }

    /// Restore a complete quality checkpoint after external single-writer fencing.
    ///
    /// `build_key` must be the same non-empty compatibility marker passed to
    /// [`Self::quality_checkpoint`]; it is not a credential. Restore validates
    /// the complete receipt, evidence, epoch, and ledger graph before reserving
    /// its process-local engine namespace. A namespace ever allocated locally
    /// is rejected, so same-process handoff should use the consuming
    /// [`Muxer::into_checkpoint`] / [`Muxer::restore`] path instead.
    /// Across serialized copies or processes, the application must supply
    /// external single-writer fencing before calling this method.
    pub fn from_quality_checkpoint(
        checkpoint: QualityMuxerCheckpoint,
        build_key: &str,
    ) -> Result<Self, PolicyError> {
        restore_core::<QualityProfile>(checkpoint.0, build_key)
    }
}

#[cfg(feature = "stochastic")]
impl Muxer<crate::BernoulliThompson> {
    /// Capture complete Bernoulli runtime state without consuming the runtime.
    ///
    /// The non-empty `build_key` binds exact application/build compatibility,
    /// not authenticity. Quiesce issuance and feedback while capturing, and
    /// provide external single-writer fencing before restoring a saved copy.
    pub fn bernoulli_checkpoint(
        &self,
        build_key: &str,
    ) -> Result<BernoulliMuxerCheckpoint, PolicyError> {
        Ok(BernoulliMuxerCheckpoint(capture_core(self, build_key)?))
    }

    /// Restore a build-bound Bernoulli checkpoint after external writer fencing.
    ///
    /// All state is validated before reserving its process-local engine ID.
    /// Previously allocated local IDs cannot be restored; same-process handoff
    /// should use [`Muxer::into_checkpoint`] and [`Muxer::restore`] instead.
    pub fn from_bernoulli_checkpoint(
        checkpoint: BernoulliMuxerCheckpoint,
        build_key: &str,
    ) -> Result<Self, PolicyError> {
        restore_core::<crate::BernoulliThompson>(checkpoint.0, build_key)
    }
}

fn restore_core<P: CheckpointProfile>(
    checkpoint: RuntimeCheckpoint<P::State, P::TicketWire, P::CanonicalFeedback>,
    build_key: &str,
) -> Result<Muxer<P>, PolicyError> {
    validate_header::<P>(&checkpoint)?;
    let policy = P::from_checkpoint_state(checkpoint.policy, build_key)?;
    let mut retired = BTreeMap::new();
    for (revision, state) in checkpoint.retired {
        if revision >= checkpoint.policy_revision
            || retired
                .insert(revision, P::from_checkpoint_state(state, build_key)?)
                .is_some()
        {
            return Err(PolicyError::new(format!(
                "invalid {} checkpoint policy epochs",
                P::LABEL
            )));
        }
    }
    if retired.len() > checkpoint.config.retired_epoch_capacity {
        return Err(checkpoint_error::<P>("retired policy capacity mismatch"));
    }
    let rng = TrialRng::from_state(checkpoint.rng)?;
    let mut pending = BTreeMap::new();
    for saved in checkpoint.pending {
        let receipt = validate_receipt::<P>(
            saved.receipt,
            &checkpoint.actions,
            checkpoint.engine,
            checkpoint.policy_revision,
        )?;
        if saved.items.len() != receipt.selected.len() {
            return Err(checkpoint_error::<P>("pending item count mismatch"));
        }
        let profile = profile_for_revision(
            &policy,
            &retired,
            checkpoint.policy_revision,
            receipt.policy_revision,
        )?;
        if receipt
            .selected
            .len()
            .checked_mul(profile.checkpoint_expectation().channels().len())
            .map_or(true, |count| count > checkpoint.config.event_capacity)
        {
            return Err(checkpoint_error::<P>("pending event capacity mismatch"));
        }
        let mut items = Vec::with_capacity(saved.items.len());
        for item in saved.items {
            validate_pending_item(&item, profile)?;
            let ticket = match item.ticket {
                Some(ticket) => Some(P::decode_ticket(ticket)?),
                None => None,
            };
            items.push(PendingItem {
                ticket,
                expectation: item.expectation,
                events: item.values,
                missing: item.missing,
                missing_reasons: item.missing_reasons,
                status: item.status,
            });
        }
        let ledger = saved.ledger.into_ledger().map_err(PolicyError::new)?;
        validate_ledger::<P, _>(
            &ledger,
            &receipt,
            &items,
            profile,
            checkpoint.next_received_sequence,
            checkpoint.config.event_capacity,
        )?;
        if pending
            .insert(
                receipt.id(),
                Pending {
                    receipt,
                    items,
                    ledger,
                },
            )
            .is_some()
        {
            return Err(PolicyError::new("duplicate pending receipt"));
        }
    }
    if pending.values().any(|record| {
        !record
            .items
            .iter()
            .any(|item| item.status == ItemStatus::Open)
    }) {
        return Err(checkpoint_error::<P>(
            "retains an all-closed pending record",
        ));
    }
    let mut terminal = BTreeMap::new();
    for saved in checkpoint.terminal {
        let receipt = validate_receipt::<P>(
            saved.receipt,
            &checkpoint.actions,
            checkpoint.engine,
            checkpoint.policy_revision,
        )?;
        if saved.items.len() != receipt.selected.len() {
            return Err(checkpoint_error::<P>("terminal item count mismatch"));
        }
        let profile = profile_for_revision(
            &policy,
            &retired,
            checkpoint.policy_revision,
            receipt.policy_revision,
        )?;
        if receipt
            .selected
            .len()
            .checked_mul(profile.checkpoint_expectation().channels().len())
            .map_or(true, |count| count > checkpoint.config.event_capacity)
        {
            return Err(checkpoint_error::<P>("terminal event capacity mismatch"));
        }
        let ledger = saved.ledger.into_ledger().map_err(PolicyError::new)?;
        let items: Vec<_> = saved
            .items
            .into_iter()
            .map(|item| TerminalItem {
                values: item.values,
                missing_reasons: item.missing_reasons,
                status: item.status,
            })
            .collect();
        validate_terminal_items(&items, saved.status, profile)?;
        validate_ledger::<P, _>(
            &ledger,
            &receipt,
            &items,
            profile,
            checkpoint.next_received_sequence,
            checkpoint.config.event_capacity,
        )?;
        if terminal
            .insert(
                receipt.id(),
                Terminal {
                    receipt,
                    items,
                    status: saved.status,
                    ledger,
                },
            )
            .is_some()
        {
            return Err(PolicyError::new("duplicate terminal receipt"));
        }
    }
    if pending.len() > checkpoint.config.pending_capacity
        || terminal.len() > checkpoint.config.terminal_capacity
        || checkpoint.terminal_order.len() != terminal.len()
        || pending.keys().any(|id| terminal.contains_key(id))
    {
        return Err(checkpoint_error::<P>("retention capacity mismatch"));
    }
    let terminal_order: VecDeque<_> = checkpoint.terminal_order.into_iter().collect();
    if terminal_order.iter().any(|id| !terminal.contains_key(id))
        || terminal_order
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != terminal_order.len()
    {
        return Err(checkpoint_error::<P>("terminal order mismatch"));
    }
    let max_sequence = pending
        .keys()
        .chain(terminal.keys())
        .map(|id| id.sequence())
        .max()
        .unwrap_or(0);
    if checkpoint.next_sequence < max_sequence {
        return Err(checkpoint_error::<P>("sequence regressed"));
    }
    let referenced_epochs: BTreeSet<_> = pending
        .values()
        .map(|record| record.receipt.policy_revision)
        .chain(
            terminal
                .values()
                .map(|record| record.receipt.policy_revision),
        )
        .collect();
    if retired
        .keys()
        .any(|revision| !referenced_epochs.contains(revision))
    {
        return Err(checkpoint_error::<P>("retains an orphan policy epoch"));
    }
    validate_global_ledgers::<P>(&pending, &terminal, checkpoint.next_received_sequence)?;
    validate_open_ticket_groups_generic(&policy, &retired, checkpoint.policy_revision, &pending)?;
    reserve_engine::<P>(checkpoint.engine)?;
    Ok(Muxer {
        actions: checkpoint.actions,
        policy,
        retired,
        engine: checkpoint.engine,
        next_sequence: checkpoint.next_sequence,
        next_received_sequence: checkpoint.next_received_sequence,
        policy_revision: checkpoint.policy_revision,
        model_revision: checkpoint.model_revision,
        representation_revision: checkpoint.representation_revision,
        config_revision: checkpoint.config_revision,
        catalogue_revision: checkpoint.catalogue_revision,
        rng,
        config: checkpoint.config,
        pending,
        terminal,
        terminal_order,
    })
}

fn validate_header<P: CheckpointProfile>(
    checkpoint: &RuntimeCheckpoint<P::State, P::TicketWire, P::CanonicalFeedback>,
) -> Result<(), PolicyError> {
    if checkpoint.version != 1
        || checkpoint.engine.get() == 0
        || checkpoint.engine.get() == u64::MAX
    {
        return Err(if P::LABEL == "quality" {
            PolicyError::new("unsupported quality muxer checkpoint identity")
        } else {
            checkpoint_error::<P>("unsupported identity")
        });
    }
    if checkpoint.actions.is_empty()
        || checkpoint.actions.iter().any(String::is_empty)
        || checkpoint.actions.iter().collect::<BTreeSet<_>>().len() != checkpoint.actions.len()
        || checkpoint.config.pending_capacity == 0
        || checkpoint.config.terminal_capacity == 0
        || checkpoint.config.event_capacity == 0
    {
        return Err(if P::LABEL == "quality" {
            PolicyError::new("invalid quality muxer checkpoint configuration")
        } else {
            checkpoint_error::<P>("configuration is invalid")
        });
    }
    if checkpoint.config_revision != checkpoint.policy_revision
        || checkpoint.catalogue_revision != 0
    {
        return Err(checkpoint_error::<P>("runtime revisions are invalid"));
    }
    Ok(())
}

fn checkpoint_error<P: CheckpointProfile>(suffix: &str) -> PolicyError {
    PolicyError::new(format!("{} checkpoint {suffix}", P::LABEL))
}

fn validate_receipt<P: CheckpointProfile>(
    saved: ReceiptCheckpoint,
    actions: &[String],
    engine: EngineId,
    current_revision: u64,
) -> Result<DecisionReceipt, PolicyError> {
    if saved.id.engine() != engine
        || saved.id.sequence() == 0
        || saved.sequence != saved.id.sequence()
        || saved.selected.is_empty()
        || saved.eligible.is_empty()
        || saved.policy_revision > current_revision
    {
        return Err(checkpoint_error::<P>("receipt is invalid"));
    }
    if saved.config_revision != saved.policy_revision || saved.catalogue_revision != 0 {
        return Err(checkpoint_error::<P>("receipt revisions are invalid"));
    }
    let eligible: BTreeSet<_> = saved.eligible.iter().collect();
    if eligible.len() != saved.eligible.len()
        || saved
            .eligible
            .iter()
            .any(|action| !actions.contains(action))
        || actions
            .iter()
            .filter(|action| eligible.contains(action))
            .ne(saved.eligible.iter())
    {
        return Err(checkpoint_error::<P>("receipt eligibility is invalid"));
    }
    let selected: BTreeSet<_> = saved.selected.iter().map(|item| &item.action).collect();
    if selected.len() != saved.selected.len()
        || saved
            .selected
            .iter()
            .any(|item| !eligible.contains(&item.action))
        || (!saved.batch && saved.selected.len() != 1)
    {
        return Err(checkpoint_error::<P>("receipt selections are invalid"));
    }
    P::validate_receipt_semantics(&saved)?;
    Ok(DecisionReceipt {
        id: saved.id,
        eligible: saved.eligible,
        selected: saved
            .selected
            .into_iter()
            .map(|item| SelectedItem {
                action: item.action,
                probability: item.probability,
                reason: item.reason,
            })
            .collect(),
        batch: saved.batch,
        sequence: saved.sequence,
        policy_revision: saved.policy_revision,
        model_revision: saved.model_revision,
        representation_revision: saved.representation_revision,
        config_revision: saved.config_revision,
        catalogue_revision: saved.catalogue_revision,
    })
}

fn profile_for_revision<'a, P: CheckpointProfile>(
    policy: &'a P,
    retired: &'a BTreeMap<u64, P>,
    current_revision: u64,
    revision: u64,
) -> Result<&'a P, PolicyError> {
    if revision == current_revision {
        Ok(policy)
    } else {
        retired
            .get(&revision)
            .ok_or_else(|| checkpoint_error::<P>("receipt has no policy epoch"))
    }
}

fn validate_pending_item<P: CheckpointProfile>(
    item: &PendingItemCheckpoint<P::TicketWire, P::CanonicalFeedback>,
    profile: &P,
) -> Result<(), PolicyError> {
    let expected = P::checkpoint_expectation(profile);
    if item.expectation != expected || item.status == ItemStatus::Expired {
        return Err(checkpoint_error::<P>(
            "item expectation or status is invalid",
        ));
    }
    validate_evidence::<P>(
        &item.values,
        &item.missing,
        &item.missing_reasons,
        &expected,
        profile,
    )?;
    match item.status {
        ItemStatus::Open
            if item.ticket.is_some()
                && item.values.len() + item.missing.len() < expected.channels().len() =>
        {
            Ok(())
        }
        ItemStatus::Completed
            if item.ticket.is_none()
                && item.values.len() + item.missing.len() == expected.channels().len() =>
        {
            Ok(())
        }
        ItemStatus::Cancelled
            if item.ticket.is_none()
                && item.values.is_empty()
                && item.missing.len() < expected.channels().len() =>
        {
            Ok(())
        }
        _ => Err(checkpoint_error::<P>("pending item resolution is invalid")),
    }
}

fn validate_terminal_items<P: CheckpointProfile>(
    items: &[TerminalItem<P::CanonicalFeedback>],
    status: TerminalStatus,
    profile: &P,
) -> Result<(), PolicyError> {
    let expected = P::checkpoint_expectation(profile);
    let mut saw_cancelled = false;
    let mut saw_expired = false;
    for item in items {
        if item.status == ItemStatus::Open {
            return Err(checkpoint_error::<P>("terminal item remains open"));
        }
        let missing: Vec<_> = item.missing_reasons.keys().cloned().collect();
        validate_evidence::<P>(
            &item.values,
            &missing,
            &item.missing_reasons,
            &expected,
            profile,
        )?;
        match (status, item.status) {
            (TerminalStatus::Completed, ItemStatus::Completed | ItemStatus::Cancelled)
            | (TerminalStatus::Cancelled, ItemStatus::Completed | ItemStatus::Cancelled)
            | (
                TerminalStatus::Expired,
                ItemStatus::Completed | ItemStatus::Cancelled | ItemStatus::Expired,
            ) => {}
            _ => return Err(checkpoint_error::<P>("terminal status is inconsistent")),
        }
        if item.status == ItemStatus::Completed
            && item.values.len() + missing.len() != expected.channels().len()
        {
            return Err(checkpoint_error::<P>("completed item is unresolved"));
        }
        if item.status == ItemStatus::Cancelled && !item.values.is_empty() {
            return Err(checkpoint_error::<P>("cancelled item has final feedback"));
        }
        let resolved = item.values.len() + missing.len() == expected.channels().len();
        if matches!(item.status, ItemStatus::Cancelled | ItemStatus::Expired) && resolved {
            return Err(checkpoint_error::<P>(
                "closed item is already fully resolved",
            ));
        }
        saw_cancelled |= item.status == ItemStatus::Cancelled;
        saw_expired |= item.status == ItemStatus::Expired;
    }
    if (status == TerminalStatus::Cancelled
        && (!saw_cancelled || items.iter().any(|item| !item.values.is_empty())))
        || (status == TerminalStatus::Expired && !saw_expired)
    {
        return Err(checkpoint_error::<P>("terminal close state is invalid"));
    }
    Ok(())
}

fn validate_evidence<P: CheckpointProfile>(
    values: &[(Channel, P::CanonicalFeedback)],
    missing: &[Channel],
    missing_reasons: &BTreeMap<Channel, String>,
    expected: &FeedbackExpectation,
    profile: &P,
) -> Result<(), PolicyError> {
    let channels = expected.channels();
    let mut seen = BTreeSet::new();
    if values.iter().any(|(channel, feedback)| {
        !channels.contains(channel)
            || !P::validate_value_channel(profile, channel, feedback)
            || !seen.insert(channel)
    }) || missing
        .iter()
        .any(|channel| !channels.contains(channel) || !seen.insert(channel))
        || missing_reasons.len() != missing.len()
        || missing
            .iter()
            .any(|channel| !missing_reasons.contains_key(channel))
    {
        return Err(checkpoint_error::<P>("item evidence is invalid"));
    }
    Ok(())
}

trait CheckpointItem<P: CheckpointProfile> {
    fn values(&self) -> &[(Channel, P::CanonicalFeedback)];
    fn missing(&self) -> Vec<Channel>;
    fn missing_reason(&self, channel: &Channel) -> Option<&str>;
    fn status(&self) -> ItemStatus;
}

impl<P: CheckpointProfile> CheckpointItem<P> for PendingItem<P> {
    fn values(&self) -> &[(Channel, P::CanonicalFeedback)] {
        &self.events
    }
    fn missing(&self) -> Vec<Channel> {
        self.missing.clone()
    }
    fn missing_reason(&self, channel: &Channel) -> Option<&str> {
        self.missing_reasons.get(channel).map(String::as_str)
    }
    fn status(&self) -> ItemStatus {
        self.status
    }
}

impl<P: CheckpointProfile> CheckpointItem<P> for TerminalItem<P::CanonicalFeedback> {
    fn values(&self) -> &[(Channel, P::CanonicalFeedback)] {
        &self.values
    }
    fn missing(&self) -> Vec<Channel> {
        self.missing_reasons.keys().cloned().collect()
    }
    fn missing_reason(&self, channel: &Channel) -> Option<&str> {
        self.missing_reasons.get(channel).map(String::as_str)
    }
    fn status(&self) -> ItemStatus {
        self.status
    }
}

fn validate_ledger<P: CheckpointProfile, I: CheckpointItem<P>>(
    ledger: &EventLedger<P::CanonicalFeedback>,
    receipt: &DecisionReceipt,
    items: &[I],
    profile: &P,
    next_received_sequence: u64,
    event_capacity: usize,
) -> Result<(), PolicyError> {
    let mut max_received = 0;
    for entry in ledger.entries() {
        let position = entry.execution.position();
        let item = items
            .get(position)
            .ok_or_else(|| checkpoint_error::<P>("event position is invalid"))?;
        if entry.execution.decision() != receipt.id
            || entry.action != receipt.selected[position].action
            || !P::checkpoint_expectation(profile)
                .channels()
                .contains(entry.channel)
        {
            return Err(checkpoint_error::<P>("event disagrees with receipt"));
        }
        max_received = max_received.max(entry.received_sequence);
        match entry.disposition {
            Disposition::Provisional(value)
                if item.status() != ItemStatus::Cancelled
                    && P::validate_value_channel(profile, entry.channel, value) => {}
            Disposition::Final(value)
                if item
                    .values()
                    .iter()
                    .any(|(channel, stored)| channel == entry.channel && stored == value) => {}
            Disposition::Missing(reason)
                if item.missing_reason(entry.channel) == Some(reason.as_str()) => {}
            _ => return Err(checkpoint_error::<P>("event evidence disagrees with item")),
        }
    }
    let uncovered = items
        .iter()
        .enumerate()
        .map(|(position, item)| {
            item.values()
                .iter()
                .filter(|(channel, _)| !ledger.has_final(position, channel))
                .count()
                + item
                    .missing()
                    .iter()
                    .filter(|channel| !ledger.has_final(position, channel))
                    .count()
        })
        .sum::<usize>();
    if max_received > next_received_sequence
        || ledger.len().saturating_add(uncovered) > event_capacity
    {
        return Err(checkpoint_error::<P>("event sequence is invalid"));
    }
    Ok(())
}

fn validate_global_ledgers<P: CheckpointProfile>(
    pending: &BTreeMap<DecisionId, Pending<P>>,
    terminal: &BTreeMap<DecisionId, Terminal<P>>,
    next_received_sequence: u64,
) -> Result<(), PolicyError> {
    let mut event_ids = BTreeSet::new();
    let mut received_sequences = BTreeSet::new();
    for ledger in pending
        .values()
        .map(|record| &record.ledger)
        .chain(terminal.values().map(|record| &record.ledger))
    {
        for entry in ledger.entries() {
            if !event_ids.insert(entry.event_id.clone()) {
                return Err(checkpoint_error::<P>("repeats retained event ID"));
            }
            if !received_sequences.insert(entry.received_sequence) {
                return Err(checkpoint_error::<P>(
                    "repeats retained event arrival sequence",
                ));
            }
            if entry.received_sequence > next_received_sequence {
                return Err(checkpoint_error::<P>("event sequence is invalid"));
            }
        }
    }
    Ok(())
}

fn validate_open_ticket_groups_generic<P: CheckpointProfile>(
    policy: &P,
    retired: &BTreeMap<u64, P>,
    current_revision: u64,
    pending: &BTreeMap<DecisionId, Pending<P>>,
) -> Result<(), PolicyError> {
    let mut groups: BTreeMap<u64, Vec<OpenTicket<'_, P>>> = BTreeMap::new();
    for record in pending.values() {
        if record.receipt.policy_revision != current_revision
            && !retired.contains_key(&record.receipt.policy_revision)
        {
            return Err(checkpoint_error::<P>("receipt has no policy epoch"));
        }
        for (position, item) in record.items.iter().enumerate() {
            if let Some(ticket) = item.ticket.as_ref() {
                groups
                    .entry(record.receipt.policy_revision)
                    .or_default()
                    .push(OpenTicket {
                        ticket,
                        action: record.receipt.selected[position].action.as_str(),
                        reason: record.receipt.selected[position].reason,
                        values: &item.events,
                        missing: &item.missing,
                    });
            }
        }
    }
    policy.validate_open_tickets(groups.get(&current_revision).map_or(&[], Vec::as_slice))?;
    for (revision, profile) in retired {
        profile.validate_open_tickets(groups.get(revision).map_or(&[], Vec::as_slice))?;
    }
    Ok(())
}

fn reserve_engine<P: CheckpointProfile>(engine: EngineId) -> Result<(), PolicyError> {
    loop {
        let current = super::NEXT_ENGINE.load(Ordering::SeqCst);
        if current > engine.get() {
            return Err(checkpoint_error::<P>(
                "engine namespace was already allocated",
            ));
        }
        let next = engine
            .get()
            .checked_add(1)
            .ok_or_else(|| checkpoint_error::<P>("engine namespace exhausted"))?;
        if super::NEXT_ENGINE
            .compare_exchange(current, next, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            return Ok(());
        }
    }
}
