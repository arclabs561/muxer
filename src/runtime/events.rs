//! Advanced feedback envelopes for receipt-correlated runtime updates.

use super::{InteractionPolicy, ItemStatus, Muxer, RuntimeError};
use crate::interaction::{Channel, Disposition, EventId, ExecutionKey};
use std::collections::BTreeMap;

/// An application event correlated with one selected execution.
///
/// `action` is checked by [`crate::Muxer`] against the action at `execution`.
/// `source_revision` identifies the producer/schema epoch, while `revision`
/// orders observations from that source for this execution and channel. The
/// runtime assigns its own arrival sequence after accepting the event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeedbackEvent<F> {
    execution: ExecutionKey,
    action: String,
    event_id: EventId,
    channel: Channel,
    source_revision: u64,
    revision: u64,
    observed_at: Option<u64>,
    disposition: Disposition<F>,
}

impl<F> FeedbackEvent<F> {
    /// Construct a feedback event.
    #[must_use]
    pub fn new(
        execution: ExecutionKey,
        action: impl Into<String>,
        event_id: EventId,
        channel: Channel,
        source_revision: u64,
        revision: u64,
        disposition: Disposition<F>,
    ) -> Self {
        Self {
            execution,
            action: action.into(),
            event_id,
            channel,
            source_revision,
            revision,
            observed_at: None,
            disposition,
        }
    }

    /// Selected execution this event reports.
    #[must_use]
    pub const fn execution(&self) -> ExecutionKey {
        self.execution
    }

    /// Action actually executed by the application.
    #[must_use]
    pub fn action(&self) -> &str {
        &self.action
    }

    /// Application-owned idempotency key.
    #[must_use]
    pub fn event_id(&self) -> &EventId {
        &self.event_id
    }

    /// Measurement meaning and source/schema name.
    #[must_use]
    pub fn channel(&self) -> &Channel {
        &self.channel
    }

    /// Producer or schema epoch for this measurement stream.
    #[must_use]
    pub const fn source_revision(&self) -> u64 {
        self.source_revision
    }

    /// Monotone revision within the measurement stream.
    #[must_use]
    pub const fn revision(&self) -> u64 {
        self.revision
    }

    /// Attach application observation time as Unix milliseconds.
    ///
    /// This is metadata, not a freshness check or the replay ordering key.
    #[must_use]
    pub fn with_observed_at(mut self, observed_at: u64) -> Self {
        self.observed_at = Some(observed_at);
        self
    }

    /// Optional application observation time in Unix milliseconds.
    #[must_use]
    pub const fn observed_at(&self) -> Option<u64> {
        self.observed_at
    }

    /// Whether the payload is provisional, final, or explicitly missing.
    #[must_use]
    pub fn disposition(&self) -> &Disposition<F> {
        &self.disposition
    }

    /// Consume this envelope and return its payload disposition.
    #[must_use]
    pub fn into_disposition(self) -> Disposition<F> {
        self.disposition
    }
}

/// Result of submitting an advanced feedback event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventOutcome {
    /// A final value was retained and applied to its subscriber.
    Accepted,
    /// An equivalent retained event was submitted again.
    Duplicate,
    /// A provisional value was retained without applying learning effects.
    Provisional,
    /// A channel was closed explicitly without fabricating a value.
    Missing,
    /// The value finalized, but its subscriber no longer retains the original row.
    OutsideHorizon,
}

#[derive(PartialEq, Eq)]
struct CanonicalEnvelope<C> {
    execution: ExecutionKey,
    action: String,
    channel: Channel,
    source_revision: u64,
    revision: u64,
    observed_at: Option<u64>,
    disposition: Disposition<C>,
}

struct RecordedEvent<C> {
    envelope: CanonicalEnvelope<C>,
    received_sequence: u64,
}

/// Bounded by the containing runtime's per-decision event capacity.
pub(super) struct EventLedger<C> {
    events: BTreeMap<EventId, RecordedEvent<C>>,
    latest: BTreeMap<(usize, Channel), EventId>,
}

impl<C> Default for EventLedger<C> {
    fn default() -> Self {
        Self {
            events: BTreeMap::new(),
            latest: BTreeMap::new(),
        }
    }
}

impl<C> EventLedger<C> {
    pub(super) fn has_values(&self) -> bool {
        self.events
            .values()
            .any(|event| !matches!(event.envelope.disposition, Disposition::Missing(_)))
    }
    pub(super) fn has_value_at(&self, position: usize) -> bool {
        self.events.values().any(|event| {
            event.envelope.execution.position() == position
                && !matches!(event.envelope.disposition, Disposition::Missing(_))
        })
    }
    pub(super) fn len(&self) -> usize {
        self.events.len()
    }
    pub(super) fn has_final(&self, position: usize, channel: &Channel) -> bool {
        self.latest
            .get(&(position, channel.clone()))
            .and_then(|id| self.events.get(id))
            .is_some_and(|event| !matches!(event.envelope.disposition, Disposition::Provisional(_)))
    }
}

impl<P: InteractionPolicy> Muxer<P> {
    fn retained_ledger(&self, id: crate::DecisionId) -> Option<&EventLedger<P::CanonicalFeedback>> {
        self.pending
            .get(&id)
            .map(|record| &record.ledger)
            .or_else(|| self.terminal.get(&id).map(|record| &record.ledger))
    }

    pub(super) fn retained_event_slots(&self, id: crate::DecisionId) -> usize {
        let Some(pending) = self.pending.get(&id) else {
            return 0;
        };
        pending.ledger.len()
            + pending
                .items
                .iter()
                .enumerate()
                .map(|(index, item)| {
                    item.events
                        .iter()
                        .filter(|(channel, _)| !pending.ledger.has_final(index, channel))
                        .count()
                        + item
                            .missing
                            .iter()
                            .filter(|channel| !pending.ledger.has_final(index, channel))
                            .count()
                })
                .sum::<usize>()
    }

    /// Runtime arrival order for an accepted advanced event while it is retained.
    #[must_use]
    pub fn received_sequence(&self, id: crate::DecisionId, event: &EventId) -> Option<u64> {
        self.retained_ledger(id)?
            .events
            .get(event)
            .map(|event| event.received_sequence)
    }

    /// Latest unfinalized value for an explicitly submitted channel.
    #[must_use]
    pub fn provisional_feedback(
        &self,
        id: crate::DecisionId,
        position: usize,
        channel: &Channel,
    ) -> Option<&P::CanonicalFeedback> {
        // Terminal cancellation/expiry never exposes a live provisional value.
        let pending = self.pending.get(&id)?;
        if pending
            .items
            .get(position)?
            .missing
            .iter()
            .any(|missing| missing == channel)
        {
            return None;
        }
        let ledger = &pending.ledger;
        let event = ledger
            .events
            .get(ledger.latest.get(&(position, channel.clone()))?)?;
        if self.final_feedback(id, position, channel).is_some() {
            return None;
        }
        match &event.envelope.disposition {
            Disposition::Provisional(value) => Some(value),
            _ => None,
        }
    }

    pub(super) fn submit_recorded(
        &mut self,
        event: FeedbackEvent<P::Feedback>,
    ) -> Result<EventOutcome, RuntimeError> {
        let FeedbackEvent {
            execution,
            action,
            event_id,
            channel,
            source_revision,
            revision,
            observed_at,
            disposition,
        } = event;
        let id = execution.decision();
        let position = execution.position();
        if id.engine() != self.engine {
            return Err(RuntimeError::WrongEngine);
        }
        let receipt = self.receipt(id).ok_or(RuntimeError::UnknownDecision)?;
        let selected = receipt
            .selected()
            .nth(position)
            .ok_or(RuntimeError::WrongPosition)?;
        if selected != action {
            return Err(RuntimeError::WrongAction);
        }
        let policy = self
            .policy_at_revision(receipt.policy_revision())
            .ok_or(RuntimeError::UnknownDecision)?;
        let canonical = match disposition {
            Disposition::Provisional(value) => {
                let value = policy.normalize(value)?;
                if policy.feedback_channel(&value) != channel {
                    return Err(RuntimeError::NoFeedbackExpected);
                }
                Disposition::Provisional(value)
            }
            Disposition::Final(value) => {
                let value = policy.normalize(value)?;
                if policy.feedback_channel(&value) != channel {
                    return Err(RuntimeError::NoFeedbackExpected);
                }
                Disposition::Final(value)
            }
            Disposition::Missing(reason) => Disposition::Missing(reason),
        };
        let envelope = CanonicalEnvelope {
            execution,
            action,
            channel: channel.clone(),
            source_revision,
            revision,
            observed_at,
            disposition: canonical,
        };

        // IDs are unique across every retained decision, not just one item.
        // A bounded scan avoids a second deduplication store with a different lifetime.
        for ledger in self
            .pending
            .values()
            .map(|record| &record.ledger)
            .chain(self.terminal.values().map(|record| &record.ledger))
        {
            if let Some(previous) = ledger.events.get(&event_id) {
                return if previous.envelope == envelope {
                    Ok(EventOutcome::Duplicate)
                } else {
                    Err(RuntimeError::ConflictingEvent)
                };
            }
        }
        if let Some(error) = self.closed_error(id) {
            return Err(error);
        }
        if self.item_status(id, position) == Some(ItemStatus::Cancelled) {
            return Err(RuntimeError::Cancelled);
        }
        let ledger = self
            .retained_ledger(id)
            .ok_or(RuntimeError::UnknownDecision)?;
        if let Some(previous) = ledger
            .latest
            .get(&(position, channel.clone()))
            .and_then(|key| ledger.events.get(key))
        {
            let previous = &previous.envelope;
            if previous.source_revision != source_revision {
                return Err(RuntimeError::SourceRevisionMismatch);
            }
            if revision < previous.revision {
                return Err(RuntimeError::StaleRevision);
            }
            match (&previous.disposition, &envelope.disposition) {
                (Disposition::Final(old), Disposition::Final(new)) if old == new => {
                    return Ok(EventOutcome::Duplicate)
                }
                (Disposition::Missing(old), Disposition::Missing(new)) if old == new => {
                    return Ok(EventOutcome::Duplicate)
                }
                (Disposition::Final(_) | Disposition::Missing(_), _) => {
                    return Err(RuntimeError::AlreadyFinalized)
                }
                (Disposition::Provisional(old), Disposition::Provisional(new))
                    if revision == previous.revision && old == new =>
                {
                    return Ok(EventOutcome::Duplicate)
                }
                (
                    Disposition::Provisional(old),
                    Disposition::Provisional(new) | Disposition::Final(new),
                ) if revision == previous.revision && old != new => {
                    return Err(RuntimeError::ConflictingEvent)
                }
                (Disposition::Provisional(_), Disposition::Missing(_))
                    if revision == previous.revision =>
                {
                    return Err(RuntimeError::ConflictingEvent)
                }
                _ => {}
            }
        }

        // Convenience tell() values also close channels and cannot be reopened
        // by switching to the advanced API after a final value was accepted.
        if let Some(final_value) = self.final_feedback(id, position, &channel) {
            return match &envelope.disposition {
                Disposition::Final(value) if value == final_value => Ok(EventOutcome::Duplicate),
                _ => Err(RuntimeError::AlreadyFinalized),
            };
        }
        if let Some(reason) = self.missing_reason(id, position, &channel) {
            return match &envelope.disposition {
                Disposition::Missing(incoming) if incoming == reason => Ok(EventOutcome::Duplicate),
                _ => Err(RuntimeError::AlreadyFinalized),
            };
        }
        let pending = self
            .pending
            .get(&id)
            .ok_or(RuntimeError::AlreadyFinalized)?;
        let item = pending
            .items
            .get(position)
            .ok_or(RuntimeError::WrongPosition)?;
        if !item.expectation.channels().contains(&channel) {
            return Err(RuntimeError::NoFeedbackExpected);
        }
        if item.missing.iter().any(|missing| missing == &channel) {
            return Err(RuntimeError::AlreadyFinalized);
        }
        if self.retained_event_slots(id) >= self.config.event_capacity {
            return Err(RuntimeError::FeedbackCapacity);
        }
        let received_sequence = self
            .next_received_sequence
            .checked_add(1)
            .ok_or(RuntimeError::FeedbackCapacity)?;
        let outcome = match &envelope.disposition {
            Disposition::Provisional(_) => EventOutcome::Provisional,
            Disposition::Final(value) => {
                match self.accept_canonical(id, position, value.clone())? {
                    crate::EventDisposition::Accepted => EventOutcome::Accepted,
                    crate::EventDisposition::Duplicate => return Ok(EventOutcome::Duplicate),
                    crate::EventDisposition::OutsideHorizon => EventOutcome::OutsideHorizon,
                }
            }
            Disposition::Missing(reason) => {
                self.tell_missing(id, position, channel.clone(), reason.clone())?;
                EventOutcome::Missing
            }
        };
        // Every recoverable validation and subscriber preparation completed
        // above. Promotion to terminal moves this same ledger, preserving IDs.
        let ledger = if let Some(pending) = self.pending.get_mut(&id) {
            &mut pending.ledger
        } else {
            &mut self
                .terminal
                .get_mut(&id)
                .expect("accepted current record is retained")
                .ledger
        };
        ledger.latest.insert((position, channel), event_id.clone());
        ledger.events.insert(
            event_id,
            RecordedEvent {
                envelope,
                received_sequence,
            },
        );
        self.next_received_sequence = received_sequence;
        Ok(outcome)
    }
}
