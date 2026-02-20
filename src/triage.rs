//! Detect-then-triage session: arm-level CUSUM detection + per-cell investigation.
//!
//! ## Motivation
//!
//! `CusumCatBank` (in `monitor`) detects *per-arm* distributional shifts.  Once an
//! arm alarms, you need to characterise *where in the covariate space* the regression
//! lives.  `ContextualCoverageTracker` (in `worst_first`) accumulates per-(arm, bin)
//! badness statistics and prioritises cells for investigation.
//!
//! `TriageSession` wires the two together:
//!
//! ```text
//!          observe(arm, outcome_idx, context)
//!                       │
//!              ┌────────┴────────┐
//!              ▼                 ▼
//!    per-arm CusumCatBank    ContextualCoverageTracker
//!    (detects shift)         (records per-cell stats)
//!              │
//!     alarmed_arms() ─────► top_cells(arms, k)
//!                                    │
//!                         (arm, bin) pairs to route
//!                         extra traffic to
//! ```
//!
//! ## Outcome encoding
//!
//! `TriageSession` uses 4 categorical outcome categories consistent with
//! `muxer`'s `Outcome` struct:
//!
//! | idx | meaning |
//! |-----|---------|
//! | 0   | ok (clean success) |
//! | 1   | soft junk (degraded success) |
//! | 2   | hard junk (operational failure) |
//! | 3   | total failure (ok=false) |
//!
//! Callers use [`OutcomeIdx`] helpers to map `Outcome` values to indices.
//!
//! ## Example
//!
//! ```rust
//! use muxer::{TriageSession, TriageSessionConfig, OutcomeIdx, ContextBinConfig, WorstFirstConfig};
//!
//! let arms = vec!["a".to_string(), "b".to_string()];
//! let cfg = TriageSessionConfig::default();
//! let mut session = TriageSession::new(&arms, cfg).unwrap();
//!
//! // Record 50 clean observations on arm "a".
//! for _ in 0..50 {
//!     session.observe("a", OutcomeIdx::OK, &[0.2, 0.3]);
//! }
//! // Inject 20 hard-junk observations on arm "b" in the high-covariate region.
//! for _ in 0..20 {
//!     session.observe("b", OutcomeIdx::HARD_JUNK, &[0.8, 0.9]);
//! }
//!
//! // The cell tracker has stats; pick the worst cell.
//! let bins = session.tracker().active_bins();
//! let arms_slice: Vec<String> = arms.clone();
//! let picks = session.top_cells(&arms_slice, &bins, 1);
//! assert!(!picks.is_empty());
//! assert_eq!(picks[0].0.arm, "b");
//! ```

use crate::monitor::CusumCatBank;
use crate::{ContextBinConfig, ContextualCell, ContextualCoverageTracker, WorstFirstConfig};
use std::collections::BTreeMap;

/// Categorical outcome index constants for the 4-category muxer outcome space.
pub struct OutcomeIdx;

impl OutcomeIdx {
    /// Clean success (ok=true, junk=false, hard_junk=false).
    pub const OK: usize = 0;
    /// Degraded success (ok=true, junk=true, hard_junk=false).
    pub const SOFT_JUNK: usize = 1;
    /// Operational failure (ok=true, junk=true, hard_junk=true).
    pub const HARD_JUNK: usize = 2;
    /// Total failure (ok=false).
    pub const FAIL: usize = 3;

    /// Map a `(ok, junk, hard_junk)` triple to a category index.
    pub fn from_outcome(ok: bool, junk: bool, hard_junk: bool) -> usize {
        if !ok {
            Self::FAIL
        } else if hard_junk {
            Self::HARD_JUNK
        } else if junk {
            Self::SOFT_JUNK
        } else {
            Self::OK
        }
    }
}

/// Configuration for a [`TriageSession`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TriageSessionConfig {
    /// CUSUM null distribution `p0` over the 4-category outcome space.
    ///
    /// Default: `[0.85, 0.05, 0.05, 0.05]` (85% clean, 5% each degraded category).
    pub p0: [f64; 4],

    /// CUSUM alternative distributions.  Each entry is an alternative hypothesis
    /// (e.g. "hard-junk spike", "total-failure spike") as a 4-category distribution.
    ///
    /// Default: two alternatives — hard-junk-heavy and fail-heavy.
    pub alts: Vec<[f64; 4]>,

    /// Dirichlet smoothing for CUSUM (added to p0 and each alt before log-ratio).
    pub cusum_alpha: f64,

    /// Minimum observations before a detector can alarm.
    pub min_n: u64,

    /// CUSUM alarm threshold (lower = more sensitive).
    pub threshold: f64,

    /// Simplex tolerance for validation.
    pub tol: f64,

    /// Context-bin configuration for the coverage tracker.
    pub bin_cfg: ContextBinConfig,

    /// Worst-first scoring weights for cell prioritisation.
    pub wf_cfg: WorstFirstConfig,

    /// Seed for deterministic tie-breaking in `top_cells`.
    pub seed: u64,
}

impl Default for TriageSessionConfig {
    fn default() -> Self {
        Self {
            p0: [0.85, 0.05, 0.05, 0.05],
            alts: vec![
                [0.40, 0.10, 0.40, 0.10], // hard-junk spike
                [0.40, 0.10, 0.10, 0.40], // total-failure spike
            ],
            cusum_alpha: 1e-3,
            min_n: 20,
            threshold: 5.0,
            tol: 1e-6,
            bin_cfg: ContextBinConfig::default(),
            wf_cfg: WorstFirstConfig {
                exploration_c: 1.0,
                hard_weight: 3.0,
                soft_weight: 1.0,
            },
            seed: 0xCA_FE_BA_BE,
        }
    }
}

/// Per-arm detection state returned by [`TriageSession::arm_state`].
#[derive(Debug, Clone, Copy)]
pub struct ArmTriageState {
    /// Number of observations processed by this arm's CUSUM bank.
    pub n: u64,
    /// Maximum CUSUM score across all alternatives.
    pub score_max: f64,
    /// Whether the arm is currently in "alarmed" state.
    pub alarmed: bool,
}

/// Combined detect-then-triage session.
///
/// See module-level docs for the design and example.
#[derive(Debug, Clone)]
pub struct TriageSession {
    banks: BTreeMap<String, (CusumCatBank, bool)>, // (bank, alarmed)
    tracker: ContextualCoverageTracker,
    bin_cfg: ContextBinConfig,
    wf_cfg: WorstFirstConfig,
    seed: u64,
}

impl TriageSession {
    /// Create a new session for the given `arms` with config `cfg`.
    ///
    /// Returns an error if any CUSUM bank cannot be initialised (e.g. invalid simplex).
    pub fn new(arms: &[String], cfg: TriageSessionConfig) -> Result<Self, logp::Error> {
        let p0: Vec<f64> = cfg.p0.to_vec();
        let alts: Vec<Vec<f64>> = cfg.alts.iter().map(|a| a.to_vec()).collect();

        let mut banks = BTreeMap::new();
        for arm in arms {
            let bank = CusumCatBank::new(
                &p0,
                &alts,
                cfg.cusum_alpha,
                cfg.min_n,
                cfg.threshold,
                cfg.tol,
            )?;
            banks.insert(arm.clone(), (bank, false));
        }

        Ok(Self {
            banks,
            tracker: ContextualCoverageTracker::new(),
            bin_cfg: cfg.bin_cfg,
            wf_cfg: cfg.wf_cfg,
            seed: cfg.seed,
        })
    }

    /// Record one outcome for `(arm, outcome_idx)` with its feature `context`.
    ///
    /// - Updates the arm's CUSUM bank (may trigger an alarm).
    /// - Records the outcome in the coverage tracker under the bin derived from `context`.
    ///
    /// Returns the CUSUM bank update for the arm (or `None` if the arm is unknown).
    pub fn observe(
        &mut self,
        arm: &str,
        outcome_idx: usize,
        context: &[f64],
    ) -> Option<crate::monitor::CusumCatBankUpdate> {
        let bin = crate::context_bin(context, self.bin_cfg);

        // Update CUSUM bank.
        let update = if let Some((bank, alarmed)) = self.banks.get_mut(arm) {
            let upd = bank.update(outcome_idx);
            if upd.alarmed {
                *alarmed = true;
            }
            Some(upd)
        } else {
            None
        };

        // Update cell tracker with outcome classification.
        let hard_junk = outcome_idx == OutcomeIdx::HARD_JUNK;
        let soft_junk = outcome_idx == OutcomeIdx::SOFT_JUNK;
        self.tracker.record(arm, bin, hard_junk, soft_junk);

        update
    }

    /// List arms whose CUSUM bank has alarmed at least once since the last reset.
    pub fn alarmed_arms(&self) -> Vec<String> {
        self.banks
            .iter()
            .filter(|(_, (_, alarmed))| *alarmed)
            .map(|(arm, _)| arm.clone())
            .collect()
    }

    /// Whether any arm has alarmed.
    pub fn any_alarmed(&self) -> bool {
        self.banks.values().any(|(_, alarmed)| *alarmed)
    }

    /// Current detection state for a specific arm.
    pub fn arm_state(&self, arm: &str) -> Option<ArmTriageState> {
        self.banks.get(arm).map(|(bank, alarmed)| ArmTriageState {
            n: bank.n(),
            score_max: bank.score_max(),
            alarmed: *alarmed,
        })
    }

    /// Pick up to `k` (arm, context-bin) cells to route extra investigation traffic to.
    ///
    /// Uses the same badness + exploration scoring as [`ContextualCoverageTracker::pick_k`].
    pub fn top_cells(
        &self,
        arms: &[String],
        active_bins: &[u64],
        k: usize,
    ) -> Vec<(ContextualCell, bool)> {
        self.tracker
            .pick_k(self.seed, arms, active_bins, k, self.wf_cfg)
    }

    /// Pick up to `k` cells restricted to arms that have alarmed.
    ///
    /// Equivalent to `top_cells` filtered to `alarmed_arms()`.
    pub fn top_alarmed_cells(&self, active_bins: &[u64], k: usize) -> Vec<(ContextualCell, bool)> {
        let arms = self.alarmed_arms();
        if arms.is_empty() {
            return Vec::new();
        }
        self.tracker
            .pick_k(self.seed, &arms, active_bins, k, self.wf_cfg)
    }

    /// Reset the CUSUM bank (and alarm state) for a specific arm.
    ///
    /// Call this after a regression has been acknowledged and the arm is expected
    /// to return to baseline behaviour.  Does not clear the cell tracker — historical
    /// badness data is preserved for future triage.
    pub fn reset_arm(&mut self, arm: &str) {
        if let Some((bank, alarmed)) = self.banks.get_mut(arm) {
            bank.reset();
            *alarmed = false;
        }
    }

    /// Reset all arms' CUSUM banks and alarm states.
    pub fn reset_all(&mut self) {
        for (bank, alarmed) in self.banks.values_mut() {
            bank.reset();
            *alarmed = false;
        }
    }

    /// Read-only access to the coverage tracker.
    pub fn tracker(&self) -> &ContextualCoverageTracker {
        &self.tracker
    }

    /// Mutable access to the coverage tracker (e.g. to manually inject observations).
    pub fn tracker_mut(&mut self) -> &mut ContextualCoverageTracker {
        &mut self.tracker
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_arms() -> Vec<String> {
        vec!["arm_a".to_string(), "arm_b".to_string()]
    }

    #[test]
    fn new_session_no_alarms() {
        let session = TriageSession::new(&two_arms(), TriageSessionConfig::default()).unwrap();
        assert!(session.alarmed_arms().is_empty());
        assert!(!session.any_alarmed());
    }

    #[test]
    fn observe_returns_none_for_unknown_arm() {
        let mut session = TriageSession::new(&two_arms(), TriageSessionConfig::default()).unwrap();
        let upd = session.observe("unknown", OutcomeIdx::OK, &[0.5]);
        assert!(upd.is_none());
    }

    #[test]
    fn outcome_idx_mapping() {
        assert_eq!(OutcomeIdx::from_outcome(true, false, false), OutcomeIdx::OK);
        assert_eq!(
            OutcomeIdx::from_outcome(true, true, false),
            OutcomeIdx::SOFT_JUNK
        );
        assert_eq!(
            OutcomeIdx::from_outcome(true, true, true),
            OutcomeIdx::HARD_JUNK
        );
        assert_eq!(
            OutcomeIdx::from_outcome(false, false, false),
            OutcomeIdx::FAIL
        );
    }

    #[test]
    fn hard_junk_flood_triggers_alarm() {
        let cfg = TriageSessionConfig {
            min_n: 10,
            threshold: 3.0,
            ..TriageSessionConfig::default()
        };
        let mut session = TriageSession::new(&two_arms(), cfg).unwrap();

        // arm_a stays clean.
        for _ in 0..30 {
            session.observe("arm_a", OutcomeIdx::OK, &[0.2, 0.3]);
        }
        // arm_b floods with hard junk.
        for _ in 0..30 {
            session.observe("arm_b", OutcomeIdx::HARD_JUNK, &[0.7, 0.8]);
        }

        let alarmed = session.alarmed_arms();
        assert!(alarmed.contains(&"arm_b".to_string()), "arm_b should alarm");
        assert!(
            !alarmed.contains(&"arm_a".to_string()),
            "arm_a should not alarm"
        );
    }

    #[test]
    fn top_alarmed_cells_targets_bad_arm() {
        let cfg = TriageSessionConfig {
            min_n: 10,
            threshold: 3.0,
            ..TriageSessionConfig::default()
        };
        let mut session = TriageSession::new(&two_arms(), cfg).unwrap();

        for _ in 0..30 {
            session.observe("arm_a", OutcomeIdx::OK, &[0.2, 0.3]);
        }
        for _ in 0..30 {
            session.observe("arm_b", OutcomeIdx::HARD_JUNK, &[0.7, 0.8]);
        }

        let bins = session.tracker().active_bins();
        let picks = session.top_alarmed_cells(&bins, 2);
        assert!(!picks.is_empty(), "should have triage picks");
        assert_eq!(
            picks[0].0.arm, "arm_b",
            "top cell should be the alarmed arm"
        );
    }

    #[test]
    fn reset_arm_clears_alarm() {
        let cfg = TriageSessionConfig {
            min_n: 5,
            threshold: 2.0,
            ..TriageSessionConfig::default()
        };
        let mut session = TriageSession::new(&two_arms(), cfg).unwrap();
        for _ in 0..20 {
            session.observe("arm_b", OutcomeIdx::HARD_JUNK, &[0.5]);
        }
        assert!(session.any_alarmed());

        session.reset_arm("arm_b");
        assert!(!session.any_alarmed(), "alarm should clear after reset");

        let state = session.arm_state("arm_b").unwrap();
        assert_eq!(state.n, 0, "CUSUM n should reset to 0");
        assert_eq!(state.score_max, 0.0, "CUSUM score should reset to 0");
    }

    #[test]
    fn top_cells_empty_when_no_bins() {
        let session = TriageSession::new(&two_arms(), TriageSessionConfig::default()).unwrap();
        let arms = two_arms();
        let picks = session.top_cells(&arms, &[], 5);
        assert!(picks.is_empty());
    }

    #[test]
    fn cell_tracker_populated_after_observe() {
        let mut session = TriageSession::new(&two_arms(), TriageSessionConfig::default()).unwrap();
        session.observe("arm_a", OutcomeIdx::OK, &[0.1, 0.2]);
        assert_eq!(session.tracker().total_calls(), 1);
    }
}
