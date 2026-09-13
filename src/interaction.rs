//! Immutable interaction records shared by every high-level muxer profile.

use std::fmt;

/// Identity of one in-memory muxer instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EngineId(pub(crate) u64);

impl EngineId {
    /// The numeric engine namespace.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Identity of an issued decision.  It is meaningful only to its originating engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DecisionId {
    engine: EngineId,
    sequence: u64,
}

impl DecisionId {
    pub(crate) const fn new(engine: EngineId, sequence: u64) -> Self {
        Self { engine, sequence }
    }
    /// Originating engine namespace.
    #[must_use]
    pub const fn engine(self) -> EngineId {
        self.engine
    }
    /// Monotonic sequence within the originating engine.
    #[must_use]
    pub const fn sequence(self) -> u64 {
        self.sequence
    }
}

/// Identity of one selected item in a receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExecutionKey {
    decision: DecisionId,
    position: usize,
}

impl ExecutionKey {
    /// Construct an execution key.
    #[must_use]
    pub const fn new(decision: DecisionId, position: usize) -> Self {
        Self { decision, position }
    }
    /// Parent decision.
    #[must_use]
    pub const fn decision(self) -> DecisionId {
        self.decision
    }
    /// Ordered selected position.
    #[must_use]
    pub const fn position(self) -> usize {
        self.position
    }
}

/// A validated finite probability in the closed unit interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Probability(f64);

impl Probability {
    /// Validate a probability.
    pub fn new(value: f64) -> Result<Self, InteractionError> {
        if value.is_finite() && (0.0..=1.0).contains(&value) {
            Ok(Self(if value == 0.0 { 0.0 } else { value }))
        } else {
            Err(InteractionError::InvalidProbability)
        }
    }
    /// Numeric value.
    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }
}

/// Whether a policy can state an action propensity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProbabilityAvailability {
    /// Exact categorical probability.
    Exact(Probability),
    /// No probability is available.
    Unavailable,
}

/// The actual selection mechanism, independent of propensity availability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DecisionReason {
    /// A policy's initial untried-action rule.
    ExploreFirst,
    /// Maximum of sampled posterior scores.
    PosteriorSample,
    /// A draw from the stated categorical allocation distribution.
    CategoricalSample,
    /// Deterministic ranking or argmax.
    Deterministic,
    /// Quality router's reserved comparison allocation.
    Control,
    /// Quality router's investigation allocation.
    Triage,
    /// Quality router's novelty or coverage preselection.
    NoveltyOrCoverage,
    /// Quality router's remaining multi-objective policy selection.
    Policy,
    /// A custom policy did not supply mechanism diagnostics.
    Unspecified,
}

/// A finite channel name, including its units/source semantics.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Channel(String);

impl Channel {
    /// Construct a nonempty channel name.
    pub fn new(name: impl Into<String>) -> Result<Self, InteractionError> {
        let name = name.into();
        if name.trim().is_empty() {
            Err(InteractionError::InvalidChannel)
        } else {
            Ok(Self(name))
        }
    }
    /// Borrow the channel name.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
    /// Conventional scalar reward channel.
    #[must_use]
    pub fn reward() -> Self {
        Self("reward".to_owned())
    }
}

/// Expected feedback for a selected item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeedbackExpectation {
    /// No feedback is retained.
    None,
    /// One final canonical value is required.
    FinalValue {
        /// Measurement channel.
        channel: Channel,
    },
    /// Several independently final channels are required.
    FinalValues(Vec<Channel>),
}

impl FeedbackExpectation {
    /// Borrow declared channel names.
    #[must_use]
    pub fn channels(&self) -> &[Channel] {
        match self {
            Self::None => &[],
            Self::FinalValue { channel } => std::slice::from_ref(channel),
            Self::FinalValues(channels) => channels,
        }
    }
}

/// Finality of an advanced feedback event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Disposition<T> {
    /// Retained but not learned from.
    Provisional(T),
    /// Accepted for learning.
    Final(T),
    /// Explicitly absent/censored.
    Missing(String),
}

/// An application-supplied event identifier.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EventId(String);

impl EventId {
    /// Construct a nonempty event identifier.
    pub fn new(value: impl Into<String>) -> Result<Self, InteractionError> {
        let value = value.into();
        if value.is_empty() {
            Err(InteractionError::InvalidEventId)
        } else {
            Ok(Self(value))
        }
    }
    /// Borrow the identifier.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Result of accepting an event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventDisposition {
    /// The event changed retained state.
    Accepted,
    /// An equivalent event was already applied.
    Duplicate,
    /// A planned event has no learning subscriber.
    OutsideHorizon,
}

/// Errors from immutable interaction ingress validation.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum InteractionError {
    /// A probability was nonfinite or outside the unit interval.
    InvalidProbability,
    /// A channel name was empty.
    InvalidChannel,
    /// An event identifier was empty.
    InvalidEventId,
}

impl fmt::Display for InteractionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid interaction value: {self:?}")
    }
}
impl std::error::Error for InteractionError {}
