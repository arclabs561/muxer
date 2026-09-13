//! Concrete serialized checkpoint for the built-in quality profile.

use super::{
    events::{EventLedger, EventLedgerCheckpoint},
    DecisionReceipt, InteractionPolicy, ItemStatus, Muxer, Pending, PendingItem, RuntimeConfig,
    SelectedItem, Terminal, TerminalItem, TerminalStatus, TrialRng, TrialRngState,
};
use crate::interaction::{DecisionReason, Disposition};
use crate::profiles::quality::{
    CanonicalQualityFeedback, QualityProfileCheckpoint, QualityTicketCheckpoint,
};
use crate::{
    Channel, DecisionId, EngineId, FeedbackExpectation, PolicyError, ProbabilityAvailability,
    QualityProfile,
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::atomic::Ordering;

/// A build-bound, complete serialized snapshot of a [`Muxer<QualityProfile>`].
///
/// Capturing this value does not stop another writer. Quiesce the application
/// before capture and ensure only one owner restores a copied checkpoint.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualityMuxerCheckpoint {
    version: u32,
    engine: EngineId,
    actions: Vec<String>,
    policy: QualityProfileCheckpoint,
    retired: Vec<(u64, QualityProfileCheckpoint)>,
    next_sequence: u64,
    next_received_sequence: u64,
    policy_revision: u64,
    model_revision: u64,
    representation_revision: u64,
    config_revision: u64,
    catalogue_revision: u64,
    rng: TrialRngState,
    config: RuntimeConfig,
    pending: Vec<QualityPendingCheckpoint>,
    terminal: Vec<QualityTerminalCheckpoint>,
    terminal_order: Vec<DecisionId>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct QualityPendingCheckpoint {
    receipt: ReceiptCheckpoint,
    items: Vec<QualityPendingItemCheckpoint>,
    ledger: EventLedgerCheckpoint<CanonicalQualityFeedback>,
}
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct QualityPendingItemCheckpoint {
    ticket: Option<QualityTicketCheckpoint>,
    expectation: FeedbackExpectation,
    values: Vec<(Channel, CanonicalQualityFeedback)>,
    missing: Vec<Channel>,
    missing_reasons: BTreeMap<Channel, String>,
    status: ItemStatus,
}
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct QualityTerminalCheckpoint {
    receipt: ReceiptCheckpoint,
    items: Vec<QualityTerminalItemCheckpoint>,
    status: TerminalStatus,
    ledger: EventLedgerCheckpoint<CanonicalQualityFeedback>,
}
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct QualityTerminalItemCheckpoint {
    values: Vec<(Channel, CanonicalQualityFeedback)>,
    missing_reasons: BTreeMap<Channel, String>,
    status: ItemStatus,
}

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
        validate_open_ticket_groups(
            &self.policy,
            &self.retired,
            self.policy_revision,
            &self.pending,
        )?;
        let policy = self.policy.checkpoint_state(build_key)?;
        let mut retired = Vec::with_capacity(self.retired.len());
        for (revision, profile) in &self.retired {
            retired.push((*revision, profile.checkpoint_state(build_key)?));
        }
        Ok(QualityMuxerCheckpoint {
            version: 1,
            engine: self.engine,
            actions: self.actions.clone(),
            policy,
            retired,
            next_sequence: self.next_sequence,
            next_received_sequence: self.next_received_sequence,
            policy_revision: self.policy_revision,
            model_revision: self.model_revision,
            representation_revision: self.representation_revision,
            config_revision: self.config_revision,
            catalogue_revision: self.catalogue_revision,
            rng: self.rng.state(),
            config: self.config,
            pending: self
                .pending
                .values()
                .map(|record| QualityPendingCheckpoint {
                    receipt: ReceiptCheckpoint::from(&record.receipt),
                    items: record
                        .items
                        .iter()
                        .map(|item| QualityPendingItemCheckpoint {
                            ticket: item.ticket.as_ref().map(QualityTicketCheckpoint::from),
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
            terminal: self
                .terminal
                .values()
                .map(|record| QualityTerminalCheckpoint {
                    receipt: ReceiptCheckpoint::from(&record.receipt),
                    items: record
                        .items
                        .iter()
                        .map(|item| QualityTerminalItemCheckpoint {
                            values: item.values.clone(),
                            missing_reasons: item.missing_reasons.clone(),
                            status: item.status,
                        })
                        .collect(),
                    status: record.status,
                    ledger: EventLedgerCheckpoint::from(&record.ledger),
                })
                .collect(),
            terminal_order: self.terminal_order.iter().copied().collect(),
        })
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
        validate_header(&checkpoint)?;
        let policy = QualityProfile::from_checkpoint_state(checkpoint.policy, build_key)?;
        let mut retired = BTreeMap::new();
        for (revision, state) in checkpoint.retired {
            if revision >= checkpoint.policy_revision
                || retired
                    .insert(
                        revision,
                        QualityProfile::from_checkpoint_state(state, build_key)?,
                    )
                    .is_some()
            {
                return Err(PolicyError::new("invalid quality checkpoint policy epochs"));
            }
        }
        if retired.len() > checkpoint.config.retired_epoch_capacity {
            return Err(PolicyError::new(
                "quality checkpoint retired policy capacity mismatch",
            ));
        }
        let rng = TrialRng::from_state(checkpoint.rng)?;
        let mut pending = BTreeMap::new();
        for saved in checkpoint.pending {
            let receipt = validate_receipt(
                saved.receipt,
                &checkpoint.actions,
                checkpoint.engine,
                checkpoint.policy_revision,
            )?;
            if saved.items.len() != receipt.selected.len() {
                return Err(PolicyError::new(
                    "quality checkpoint pending item count mismatch",
                ));
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
                return Err(PolicyError::new(
                    "quality checkpoint pending event capacity mismatch",
                ));
            }
            let mut items = Vec::with_capacity(saved.items.len());
            for item in saved.items {
                validate_pending_item(&item, profile)?;
                let ticket = match item.ticket {
                    Some(ticket) => Some(ticket.into_ticket()?),
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
            validate_ledger(
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
            return Err(PolicyError::new(
                "quality checkpoint retains an all-closed pending record",
            ));
        }
        let mut terminal = BTreeMap::new();
        for saved in checkpoint.terminal {
            let receipt = validate_receipt(
                saved.receipt,
                &checkpoint.actions,
                checkpoint.engine,
                checkpoint.policy_revision,
            )?;
            if saved.items.len() != receipt.selected.len() {
                return Err(PolicyError::new(
                    "quality checkpoint terminal item count mismatch",
                ));
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
                return Err(PolicyError::new(
                    "quality checkpoint terminal event capacity mismatch",
                ));
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
            validate_ledger(
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
            return Err(PolicyError::new(
                "quality checkpoint retention capacity mismatch",
            ));
        }
        let terminal_order: VecDeque<_> = checkpoint.terminal_order.into_iter().collect();
        if terminal_order.iter().any(|id| !terminal.contains_key(id))
            || terminal_order
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != terminal_order.len()
        {
            return Err(PolicyError::new(
                "quality checkpoint terminal order mismatch",
            ));
        }
        let max_sequence = pending
            .keys()
            .chain(terminal.keys())
            .map(|id| id.sequence())
            .max()
            .unwrap_or(0);
        if checkpoint.next_sequence < max_sequence {
            return Err(PolicyError::new("quality checkpoint sequence regressed"));
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
            return Err(PolicyError::new(
                "quality checkpoint retains an orphan policy epoch",
            ));
        }
        validate_global_ledgers(&pending, &terminal, checkpoint.next_received_sequence)?;
        validate_open_ticket_groups(&policy, &retired, checkpoint.policy_revision, &pending)?;
        reserve_engine(checkpoint.engine)?;
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
}

fn validate_header(checkpoint: &QualityMuxerCheckpoint) -> Result<(), PolicyError> {
    if checkpoint.version != 1
        || checkpoint.engine.get() == 0
        || checkpoint.engine.get() == u64::MAX
    {
        return Err(PolicyError::new(
            "unsupported quality muxer checkpoint identity",
        ));
    }
    if checkpoint.actions.is_empty()
        || checkpoint.actions.iter().any(String::is_empty)
        || checkpoint.actions.iter().collect::<BTreeSet<_>>().len() != checkpoint.actions.len()
        || checkpoint.config.pending_capacity == 0
        || checkpoint.config.terminal_capacity == 0
        || checkpoint.config.event_capacity == 0
    {
        return Err(PolicyError::new(
            "invalid quality muxer checkpoint configuration",
        ));
    }
    if checkpoint.config_revision != checkpoint.policy_revision
        || checkpoint.catalogue_revision != 0
    {
        return Err(PolicyError::new(
            "quality checkpoint runtime revisions are invalid",
        ));
    }
    Ok(())
}

fn validate_receipt(
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
        return Err(PolicyError::new("quality checkpoint receipt is invalid"));
    }
    if saved.config_revision != saved.policy_revision || saved.catalogue_revision != 0 {
        return Err(PolicyError::new(
            "quality checkpoint receipt revisions are invalid",
        ));
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
        return Err(PolicyError::new(
            "quality checkpoint receipt eligibility is invalid",
        ));
    }
    let selected: BTreeSet<_> = saved.selected.iter().map(|item| &item.action).collect();
    if selected.len() != saved.selected.len()
        || saved.selected.iter().any(|item| {
            !eligible.contains(&item.action)
                || !matches!(item.probability, ProbabilityAvailability::Unavailable)
                || !matches!(
                    item.reason,
                    DecisionReason::Control
                        | DecisionReason::Triage
                        | DecisionReason::NoveltyOrCoverage
                        | DecisionReason::Policy
                )
        })
        || (!saved.batch && saved.selected.len() != 1)
    {
        return Err(PolicyError::new(
            "quality checkpoint receipt selections are invalid",
        ));
    }
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

fn profile_for_revision<'a>(
    policy: &'a QualityProfile,
    retired: &'a BTreeMap<u64, QualityProfile>,
    current_revision: u64,
    revision: u64,
) -> Result<&'a QualityProfile, PolicyError> {
    if revision == current_revision {
        Ok(policy)
    } else {
        retired
            .get(&revision)
            .ok_or_else(|| PolicyError::new("quality checkpoint receipt has no policy epoch"))
    }
}

fn validate_pending_item(
    item: &QualityPendingItemCheckpoint,
    profile: &QualityProfile,
) -> Result<(), PolicyError> {
    let expected = profile.checkpoint_expectation();
    if item.expectation != expected || item.status == ItemStatus::Expired {
        return Err(PolicyError::new(
            "quality checkpoint item expectation or status is invalid",
        ));
    }
    validate_evidence(
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
        _ => Err(PolicyError::new(
            "quality checkpoint pending item resolution is invalid",
        )),
    }
}

fn validate_terminal_items(
    items: &[TerminalItem<CanonicalQualityFeedback>],
    status: TerminalStatus,
    profile: &QualityProfile,
) -> Result<(), PolicyError> {
    let expected = profile.checkpoint_expectation();
    let mut saw_cancelled = false;
    let mut saw_expired = false;
    for item in items {
        if item.status == ItemStatus::Open {
            return Err(PolicyError::new(
                "quality checkpoint terminal item remains open",
            ));
        }
        let missing: Vec<_> = item.missing_reasons.keys().cloned().collect();
        validate_evidence(
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
            _ => {
                return Err(PolicyError::new(
                    "quality checkpoint terminal status is inconsistent",
                ))
            }
        }
        if item.status == ItemStatus::Completed
            && item.values.len() + missing.len() != expected.channels().len()
        {
            return Err(PolicyError::new(
                "quality checkpoint completed item is unresolved",
            ));
        }
        if item.status == ItemStatus::Cancelled && !item.values.is_empty() {
            return Err(PolicyError::new(
                "quality checkpoint cancelled item has final feedback",
            ));
        }
        let resolved = item.values.len() + missing.len() == expected.channels().len();
        if matches!(item.status, ItemStatus::Cancelled | ItemStatus::Expired) && resolved {
            return Err(PolicyError::new(
                "quality checkpoint closed item is already fully resolved",
            ));
        }
        saw_cancelled |= item.status == ItemStatus::Cancelled;
        saw_expired |= item.status == ItemStatus::Expired;
    }
    if (status == TerminalStatus::Cancelled
        && (!saw_cancelled || items.iter().any(|item| !item.values.is_empty())))
        || (status == TerminalStatus::Expired && !saw_expired)
    {
        return Err(PolicyError::new(
            "quality checkpoint terminal close state is invalid",
        ));
    }
    Ok(())
}

fn validate_evidence(
    values: &[(Channel, CanonicalQualityFeedback)],
    missing: &[Channel],
    missing_reasons: &BTreeMap<Channel, String>,
    expected: &FeedbackExpectation,
    profile: &QualityProfile,
) -> Result<(), PolicyError> {
    let channels = expected.channels();
    let mut seen = BTreeSet::new();
    if values.iter().any(|(channel, feedback)| {
        !channels.contains(channel)
            || profile.feedback_channel(feedback) != *channel
            || !seen.insert(channel)
    }) || missing
        .iter()
        .any(|channel| !channels.contains(channel) || !seen.insert(channel))
        || missing_reasons.len() != missing.len()
        || missing
            .iter()
            .any(|channel| !missing_reasons.contains_key(channel))
    {
        return Err(PolicyError::new(
            "quality checkpoint item evidence is invalid",
        ));
    }
    Ok(())
}

trait CheckpointItem {
    fn values(&self) -> &[(Channel, CanonicalQualityFeedback)];
    fn missing(&self) -> Vec<Channel>;
    fn missing_reason(&self, channel: &Channel) -> Option<&str>;
    fn status(&self) -> ItemStatus;
}

impl CheckpointItem for PendingItem<QualityProfile> {
    fn values(&self) -> &[(Channel, CanonicalQualityFeedback)] {
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

impl CheckpointItem for TerminalItem<CanonicalQualityFeedback> {
    fn values(&self) -> &[(Channel, CanonicalQualityFeedback)] {
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

fn validate_ledger<I: CheckpointItem>(
    ledger: &EventLedger<CanonicalQualityFeedback>,
    receipt: &DecisionReceipt,
    items: &[I],
    profile: &QualityProfile,
    next_received_sequence: u64,
    event_capacity: usize,
) -> Result<(), PolicyError> {
    let mut max_received = 0;
    for entry in ledger.entries() {
        let position = entry.execution.position();
        let item = items
            .get(position)
            .ok_or_else(|| PolicyError::new("quality checkpoint event position is invalid"))?;
        if entry.execution.decision() != receipt.id
            || entry.action != receipt.selected[position].action
            || !profile
                .checkpoint_expectation()
                .channels()
                .contains(entry.channel)
        {
            return Err(PolicyError::new(
                "quality checkpoint event disagrees with receipt",
            ));
        }
        max_received = max_received.max(entry.received_sequence);
        match entry.disposition {
            Disposition::Provisional(value)
                if item.status() != ItemStatus::Cancelled
                    && profile.feedback_channel(value) == *entry.channel => {}
            Disposition::Final(value)
                if item
                    .values()
                    .iter()
                    .any(|(channel, stored)| channel == entry.channel && stored == value) => {}
            Disposition::Missing(reason)
                if item.missing_reason(entry.channel) == Some(reason.as_str()) => {}
            _ => {
                return Err(PolicyError::new(
                    "quality checkpoint event evidence disagrees with item",
                ))
            }
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
        return Err(PolicyError::new(
            "quality checkpoint event sequence is invalid",
        ));
    }
    Ok(())
}

fn validate_global_ledgers(
    pending: &BTreeMap<DecisionId, Pending<QualityProfile>>,
    terminal: &BTreeMap<DecisionId, Terminal<QualityProfile>>,
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
                return Err(PolicyError::new(
                    "quality checkpoint repeats retained event ID",
                ));
            }
            if !received_sequences.insert(entry.received_sequence) {
                return Err(PolicyError::new(
                    "quality checkpoint repeats retained event arrival sequence",
                ));
            }
            if entry.received_sequence > next_received_sequence {
                return Err(PolicyError::new(
                    "quality checkpoint event sequence is invalid",
                ));
            }
        }
    }
    Ok(())
}

fn validate_open_ticket_groups(
    policy: &QualityProfile,
    retired: &BTreeMap<u64, QualityProfile>,
    current_revision: u64,
    pending: &BTreeMap<DecisionId, Pending<QualityProfile>>,
) -> Result<(), PolicyError> {
    let mut groups: BTreeMap<u64, Vec<crate::profiles::quality::QualityTicketState<'_>>> =
        BTreeMap::new();
    for record in pending.values() {
        profile_for_revision(
            policy,
            retired,
            current_revision,
            record.receipt.policy_revision,
        )?;
        for (position, item) in record.items.iter().enumerate() {
            if let Some(ticket) = item.ticket.as_ref() {
                groups
                    .entry(record.receipt.policy_revision)
                    .or_default()
                    .push(crate::profiles::quality::QualityTicketState {
                        ticket,
                        action: record.receipt.selected[position].action.as_str(),
                        reason: record.receipt.selected[position].reason,
                        values: &item.events,
                        missing: &item.missing,
                    });
            }
        }
    }
    policy.validate_checkpoint_tickets(groups.get(&current_revision).map_or(&[], Vec::as_slice))?;
    for (revision, profile) in retired {
        profile.validate_checkpoint_tickets(groups.get(revision).map_or(&[], Vec::as_slice))?;
    }
    Ok(())
}

fn reserve_engine(engine: EngineId) -> Result<(), PolicyError> {
    loop {
        let current = super::NEXT_ENGINE.load(Ordering::SeqCst);
        if current > engine.get() {
            return Err(PolicyError::new(
                "quality checkpoint engine namespace was already allocated",
            ));
        }
        let next = engine
            .get()
            .checked_add(1)
            .ok_or_else(|| PolicyError::new("quality checkpoint engine namespace exhausted"))?;
        if super::NEXT_ENGINE
            .compare_exchange(current, next, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            return Ok(());
        }
    }
}
