//! Private portable state for non-Thompson scalar profiles.

use super::*;

#[cfg(all(feature = "serde", feature = "stochastic"))]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Exp3ProfileCheckpoint {
    schema: u32,
    kind: String,
    crate_version: String,
    build_key: String,
    universe: Vec<String>,
    inner: crate::exp3ix::Exp3IxCheckpoint,
}

#[cfg(all(feature = "serde", feature = "boltzmann"))]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BoltzmannProfileCheckpoint {
    schema: u32,
    kind: String,
    crate_version: String,
    build_key: String,
    inner: crate::boltzmann::BoltzmannCheckpoint,
}

#[cfg(all(feature = "serde", any(feature = "stochastic", feature = "boltzmann")))]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ScalarProfileTicketCheckpoint {
    action: String,
    reason: DecisionReason,
    propensity: Option<u64>,
}

#[cfg(all(feature = "serde", any(feature = "stochastic", feature = "boltzmann")))]
pub(crate) struct ScalarProfileTicketState<'a, T, C> {
    pub(crate) ticket: &'a T,
    pub(crate) action: &'a str,
    pub(crate) reason: DecisionReason,
    pub(crate) probability: ProbabilityAvailability,
    pub(crate) values: &'a [(Channel, C)],
    pub(crate) missing: &'a [Channel],
}

#[cfg(all(feature = "serde", any(feature = "stochastic", feature = "boltzmann")))]
impl ScalarProfileTicketCheckpoint {
    #[cfg(feature = "stochastic")]
    pub(crate) fn exp3(ticket: &Exp3Ticket) -> Self {
        Self {
            action: ticket.action.clone(),
            reason: ticket.reason,
            propensity: Some(ticket.propensity.to_bits()),
        }
    }
    #[cfg(feature = "boltzmann")]
    pub(crate) fn boltzmann(ticket: &ScalarTicket) -> Self {
        Self {
            action: ticket.action.clone(),
            reason: ticket.reason,
            propensity: None,
        }
    }
    #[cfg(feature = "stochastic")]
    pub(crate) fn into_exp3(self) -> Result<Exp3Ticket, PolicyError> {
        let propensity = f64::from_bits(
            self.propensity
                .ok_or_else(|| PolicyError::new("EXP3 checkpoint ticket propensity missing"))?,
        );
        if self.action.is_empty()
            || !propensity.is_finite()
            || !(0.0 < propensity && propensity <= 1.0)
            || !matches!(
                self.reason,
                DecisionReason::ExploreFirst | DecisionReason::CategoricalSample
            )
        {
            return Err(PolicyError::new("EXP3 checkpoint ticket is invalid"));
        }
        Ok(Exp3Ticket {
            action: self.action,
            propensity,
            reason: self.reason,
        })
    }
    #[cfg(feature = "boltzmann")]
    pub(crate) fn into_boltzmann(self) -> Result<ScalarTicket, PolicyError> {
        if self.action.is_empty()
            || self.propensity.is_some()
            || self.reason != DecisionReason::CategoricalSample
        {
            return Err(PolicyError::new("Boltzmann checkpoint ticket is invalid"));
        }
        Ok(ScalarTicket {
            action: self.action,
            reason: self.reason,
        })
    }
}

#[cfg(all(feature = "serde", feature = "stochastic"))]
impl Exp3Profile {
    pub(crate) fn checkpoint_state(
        &self,
        build_key: &str,
    ) -> Result<Exp3ProfileCheckpoint, PolicyError> {
        if build_key.is_empty() {
            return Err(PolicyError::new("EXP3 checkpoint build key is empty"));
        }
        // A new profile has no kernel coordinates until its first decision.
        // Materialize that deterministic initial state in the snapshot only;
        // any non-empty live state must already match the immutable universe.
        let mut inner = self.inner.clone();
        let state = inner.snapshot();
        if state.arms.is_empty()
            && state.uses.is_empty()
            && state.cum_loss_hat.is_empty()
            && state.probs.is_empty()
        {
            inner.probabilities(&self.universe);
        } else if state.arms != self.universe {
            return Err(PolicyError::new(
                "EXP3 checkpoint universe disagrees with kernel",
            ));
        }
        Ok(Exp3ProfileCheckpoint {
            schema: 1,
            kind: "exp3-profile".into(),
            crate_version: env!("CARGO_PKG_VERSION").into(),
            build_key: build_key.into(),
            universe: self.universe.clone(),
            inner: inner.checkpoint_state()?,
        })
    }
    pub(crate) fn from_checkpoint_state(
        state: Exp3ProfileCheckpoint,
        build_key: &str,
    ) -> Result<Self, PolicyError> {
        if state.schema != 1
            || state.kind != "exp3-profile"
            || state.crate_version != env!("CARGO_PKG_VERSION")
            || build_key.is_empty()
            || state.build_key != build_key
        {
            return Err(PolicyError::new("EXP3 checkpoint compatibility mismatch"));
        }
        validate_universe(&state.universe)?;
        let inner = Exp3Ix::from_checkpoint_state(state.inner)?;
        if inner.snapshot().arms != state.universe {
            return Err(PolicyError::new(
                "EXP3 checkpoint universe disagrees with kernel",
            ));
        }
        Ok(Self {
            inner,
            universe: state.universe,
        })
    }
    pub(crate) fn checkpoint_expectation(&self) -> FeedbackExpectation {
        FeedbackExpectation::FinalValue {
            channel: Channel::reward(),
        }
    }
    pub(crate) fn validate_checkpoint_tickets(
        &self,
        tickets: &[ScalarProfileTicketState<'_, Exp3Ticket, BoundedReward>],
    ) -> Result<(), PolicyError> {
        for state in tickets {
            let Some(receipt) = (match state.probability {
                ProbabilityAvailability::Exact(p) => Some(p.get()),
                _ => None,
            }) else {
                return Err(PolicyError::new(
                    "EXP3 checkpoint receipt propensity is invalid",
                ));
            };
            if state.ticket.action != state.action
                || state.ticket.reason != state.reason
                || state.ticket.propensity.to_bits() != receipt.to_bits()
                || receipt <= 0.0
                || !matches!(
                    state.reason,
                    DecisionReason::ExploreFirst | DecisionReason::CategoricalSample
                )
                || !self.universe.iter().any(|a| a == state.action)
                || state.values.iter().any(|(c, _)| c != &Channel::reward())
                || state.missing.iter().any(|c| c != &Channel::reward())
            {
                return Err(PolicyError::new("EXP3 checkpoint ticket is invalid"));
            }
        }
        Ok(())
    }
}

#[cfg(all(feature = "serde", feature = "boltzmann"))]
impl BoltzmannProfile {
    pub(crate) fn checkpoint_state(
        &self,
        build_key: &str,
    ) -> Result<BoltzmannProfileCheckpoint, PolicyError> {
        if build_key.is_empty() {
            return Err(PolicyError::new("Boltzmann checkpoint build key is empty"));
        }
        Ok(BoltzmannProfileCheckpoint {
            schema: 1,
            kind: "boltzmann-profile".into(),
            crate_version: env!("CARGO_PKG_VERSION").into(),
            build_key: build_key.into(),
            inner: self.inner.checkpoint_state()?,
        })
    }
    pub(crate) fn from_checkpoint_state(
        state: BoltzmannProfileCheckpoint,
        build_key: &str,
    ) -> Result<Self, PolicyError> {
        if state.schema != 1
            || state.kind != "boltzmann-profile"
            || state.crate_version != env!("CARGO_PKG_VERSION")
            || build_key.is_empty()
            || state.build_key != build_key
        {
            return Err(PolicyError::new(
                "Boltzmann checkpoint compatibility mismatch",
            ));
        }
        Ok(Self {
            inner: crate::BoltzmannPolicy::from_checkpoint_state(state.inner)?,
        })
    }
    pub(crate) fn checkpoint_expectation(&self) -> FeedbackExpectation {
        FeedbackExpectation::FinalValue {
            channel: Channel::reward(),
        }
    }
    pub(crate) fn validate_checkpoint_tickets(
        &self,
        tickets: &[ScalarProfileTicketState<'_, ScalarTicket, FiniteReward>],
    ) -> Result<(), PolicyError> {
        for state in tickets {
            if state.ticket.action != state.action
                || state.ticket.reason != state.reason
                || state.reason != DecisionReason::CategoricalSample
                || !matches!(state.probability, ProbabilityAvailability::Exact(_))
                || state.values.iter().any(|(c, _)| c != &Channel::reward())
                || state.missing.iter().any(|c| c != &Channel::reward())
            {
                return Err(PolicyError::new("Boltzmann checkpoint ticket is invalid"));
            }
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "serde", feature = "stochastic"))]
mod tests {
    use super::*;

    #[cfg(all(feature = "serde", feature = "stochastic"))]
    #[test]
    fn exp3_initial_profile_checkpoint_materializes_immutable_universe() {
        let profile = Exp3Profile::new(
            vec!["a".into(), "b".into()],
            Exp3IxConfig {
                decay: 0.75,
                ..Exp3IxConfig::default()
            },
        )
        .unwrap();

        let checkpoint = profile.checkpoint_state("test-build").unwrap();
        let restored = Exp3Profile::from_checkpoint_state(checkpoint, "test-build").unwrap();
        assert_eq!(restored.universe(), ["a", "b"]);
        assert_eq!(restored.inner().snapshot().arms, ["a", "b"]);
    }
}
