//! Private checkpoint state for caller-owned external profiles.

use super::*;

const EXTERNAL_PROFILE_CHECKPOINT_SCHEMA: u32 = 1;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ExternalCheckpointEnvelope {
    schema: u32,
    kind: String,
    crate_version: String,
    build_key: String,
}

impl ExternalCheckpointEnvelope {
    fn capture(kind: &str, build_key: &str) -> Result<Self, PolicyError> {
        if build_key.is_empty() {
            return Err(PolicyError::new("external checkpoint build key is empty"));
        }
        Ok(Self {
            schema: EXTERNAL_PROFILE_CHECKPOINT_SCHEMA,
            kind: kind.to_owned(),
            crate_version: env!("CARGO_PKG_VERSION").to_owned(),
            build_key: build_key.to_owned(),
        })
    }

    fn validate(self, kind: &str, build_key: &str) -> Result<(), PolicyError> {
        if self.schema != EXTERNAL_PROFILE_CHECKPOINT_SCHEMA {
            return Err(PolicyError::new("unsupported external checkpoint schema"));
        }
        if self.kind != kind {
            return Err(PolicyError::new("external checkpoint kind mismatch"));
        }
        if self.crate_version != env!("CARGO_PKG_VERSION") {
            return Err(PolicyError::new(
                "external checkpoint crate version mismatch",
            ));
        }
        if build_key.is_empty() || self.build_key != build_key {
            return Err(PolicyError::new("external checkpoint build key mismatch"));
        }
        Ok(())
    }
}

/// Private serialized state for a closure-backed external profile.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExternalReferenceCheckpoint {
    envelope: ExternalCheckpointEnvelope,
    reference: String,
}

impl ExternalReferenceCheckpoint {
    fn capture(
        reference: Option<&ModelReference>,
        kind: &str,
        build_key: &str,
    ) -> Result<Self, PolicyError> {
        let reference = reference
            .ok_or_else(|| PolicyError::new("external checkpoint requires a model reference"))?;
        Ok(Self {
            envelope: ExternalCheckpointEnvelope::capture(kind, build_key)?,
            reference: reference.as_str().to_owned(),
        })
    }

    fn into_reference(self, kind: &str, build_key: &str) -> Result<ModelReference, PolicyError> {
        self.envelope.validate(kind, build_key)?;
        ModelReference::new(self.reference)
            .map_err(|_| PolicyError::new("external checkpoint model reference is invalid"))
    }
}

/// Private serialized state for the self-contained assessment profile.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExternalAssessmentsCheckpoint {
    envelope: ExternalCheckpointEnvelope,
    objectives: Vec<ExternalObjectiveCheckpoint>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ExternalObjectiveCheckpoint {
    metric: usize,
    direction: pare::Direction,
    weight_bits: u64,
}

impl ExternalObjectiveCheckpoint {
    fn capture(objective: MetricObjective) -> Result<Self, PolicyError> {
        if !objective.weight.is_finite() || objective.weight < 0.0 {
            return Err(PolicyError::new(
                "external assessments checkpoint objective is invalid",
            ));
        }
        Ok(Self {
            metric: objective.metric,
            direction: objective.direction,
            weight_bits: objective.weight.to_bits(),
        })
    }

    fn into_objective(self) -> Result<MetricObjective, PolicyError> {
        let weight = f64::from_bits(self.weight_bits);
        if !weight.is_finite() || weight < 0.0 {
            return Err(PolicyError::new(
                "external assessments checkpoint objective is invalid",
            ));
        }
        Ok(MetricObjective {
            metric: self.metric,
            direction: self.direction,
            weight,
        })
    }
}

impl<C: ?Sized, F> ExternalScores<C, F> {
    pub(crate) fn checkpoint_state(
        &self,
        build_key: &str,
    ) -> Result<ExternalReferenceCheckpoint, PolicyError> {
        ExternalReferenceCheckpoint::capture(self.reference.as_ref(), "external-scores", build_key)
    }

    pub(crate) fn from_checkpoint_state_with<R>(
        state: ExternalReferenceCheckpoint,
        build_key: &str,
        mut resolver: R,
    ) -> Result<Self, PolicyError>
    where
        R: FnMut(&ModelReference) -> Result<F, PolicyError>,
    {
        let reference = state.into_reference("external-scores", build_key)?;
        let score = resolver(&reference)?;
        Ok(Self {
            score,
            reference: Some(reference),
            marker: PhantomData,
        })
    }
}

impl<C: ?Sized, F> ExternalDistribution<C, F> {
    pub(crate) fn checkpoint_state(
        &self,
        build_key: &str,
    ) -> Result<ExternalReferenceCheckpoint, PolicyError> {
        ExternalReferenceCheckpoint::capture(
            self.reference.as_ref(),
            "external-distribution",
            build_key,
        )
    }

    pub(crate) fn from_checkpoint_state_with<R>(
        state: ExternalReferenceCheckpoint,
        build_key: &str,
        mut resolver: R,
    ) -> Result<Self, PolicyError>
    where
        R: FnMut(&ModelReference) -> Result<F, PolicyError>,
    {
        let reference = state.into_reference("external-distribution", build_key)?;
        let distribution = resolver(&reference)?;
        Ok(Self {
            distribution,
            reference: Some(reference),
            marker: PhantomData,
        })
    }
}

impl ExternalAssessments {
    pub(crate) fn checkpoint_state(
        &self,
        build_key: &str,
    ) -> Result<ExternalAssessmentsCheckpoint, PolicyError> {
        let objectives = self
            .objectives
            .iter()
            .copied()
            .map(ExternalObjectiveCheckpoint::capture)
            .collect::<Result<_, _>>()?;
        Ok(ExternalAssessmentsCheckpoint {
            envelope: ExternalCheckpointEnvelope::capture("external-assessments", build_key)?,
            objectives,
        })
    }

    pub(crate) fn from_checkpoint_state(
        state: ExternalAssessmentsCheckpoint,
        build_key: &str,
    ) -> Result<Self, PolicyError> {
        state.envelope.validate("external-assessments", build_key)?;
        let objectives = state
            .objectives
            .into_iter()
            .map(ExternalObjectiveCheckpoint::into_objective)
            .collect::<Result<_, _>>()?;
        Ok(Self { objectives })
    }
}
