//! Stickiness / switching-cost wrappers.
//!
//! Production routers often need to avoid “flapping” between arms due to:
//! - cache warmup costs
//! - connection pooling / cold-start penalties
//! - rate-limit recovery dynamics
//! - general operational stability requirements
//!
//! This module provides a small stateful wrapper that can be used on top of selection
//! functions/policies by comparing a scalar score margin before switching.

use std::collections::BTreeMap;

use crate::{
    CandidateDebug, Decision, DecisionNote, DecisionPolicy, MabConfig, MabSelectionDecision,
    Selection,
};

/// Stickiness configuration.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StickyConfig {
    /// Minimum number of consecutive decisions to stay on an arm before allowing a switch.
    ///
    /// - `0` disables dwell control.
    pub min_dwell: u64,
    /// Minimum required score advantage to switch away from the previous arm.
    ///
    /// - `0.0` disables margin control.
    pub min_switch_margin: f64,
}

impl Default for StickyConfig {
    fn default() -> Self {
        Self {
            min_dwell: 0,
            min_switch_margin: 0.0,
        }
    }
}

/// Why a decision was made.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DecisionReason {
    /// The base policy was in “explore-first” mode (some arm had zero observations).
    ExploreFirst { chosen: String },
    /// The base policy chose `chosen` without stickiness overriding it.
    BaseChoice { chosen: String },
    /// Stickiness kept the previous arm due to `min_dwell`.
    KeptPreviousDwell {
        previous: String,
        candidate: String,
        dwell: u64,
        min_dwell: u64,
    },
    /// Stickiness kept the previous arm because the candidate advantage was too small.
    KeptPreviousMargin {
        previous: String,
        candidate: String,
        previous_score: f64,
        candidate_score: f64,
        margin: f64,
        min_margin: f64,
    },
    /// Stickiness allowed switching to the candidate.
    Switched {
        previous: String,
        candidate: String,
        previous_score: f64,
        candidate_score: f64,
        margin: f64,
    },
}

/// A selection plus decision reasons (for auditing/debugging).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExplainedSelection {
    pub selection: Selection,
    pub reasons: Vec<DecisionReason>,
}

fn f64_or0(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

/// Compute the scalar score used for selection tie-breaking (higher is better),
/// derived from `CandidateDebug` and `MabConfig`.
pub fn mab_scalar_score(c: &CandidateDebug, cfg: MabConfig) -> f64 {
    let cost_w = f64_or0(cfg.cost_weight);
    let lat_w = f64_or0(cfg.latency_weight);
    let junk_w = f64_or0(cfg.junk_weight);
    let hard_w = f64_or0(cfg.hard_junk_weight);

    f64_or0(c.objective_success)
        - cost_w * f64_or0(c.mean_cost_units)
        - lat_w * f64_or0(c.mean_elapsed_ms)
        - junk_w * f64_or0(c.junk_rate)
        - hard_w * f64_or0(c.hard_junk_rate)
}

/// Stateful stickiness wrapper for deterministic `select_mab`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StickyMab {
    pub cfg: StickyConfig,
    previous: Option<String>,
    dwell: u64,
}

impl StickyMab {
    pub fn new(cfg: StickyConfig) -> Self {
        Self {
            cfg,
            previous: None,
            dwell: 0,
        }
    }

    /// Current previous choice, if any.
    pub fn previous(&self) -> Option<&str> {
        self.previous.as_deref()
    }

    /// Number of consecutive decisions on the current `previous` arm.
    pub fn dwell(&self) -> u64 {
        self.dwell
    }

    /// Reset stickiness state (for tests or epoch resets).
    pub fn reset(&mut self) {
        self.previous = None;
        self.dwell = 0;
    }

    fn scores_by_arm(sel: &Selection) -> BTreeMap<&str, f64> {
        let mut out = BTreeMap::new();
        for c in &sel.candidates {
            out.insert(c.name.as_str(), mab_scalar_score(c, sel.config));
        }
        out
    }

    /// Apply stickiness to a base `Selection`.
    pub fn apply(&mut self, mut sel: Selection) -> ExplainedSelection {
        let mut reasons: Vec<DecisionReason> = Vec::new();

        // Back-compat heuristic for explore-first (prefer `apply_mab` when possible).
        let explore_first = sel.candidates.len() == 1 && sel.candidates[0].calls == 0;
        if explore_first {
            let chosen = sel.chosen.clone();
            self.previous = Some(chosen.clone());
            self.dwell = 1;
            reasons.push(DecisionReason::ExploreFirst { chosen });
            return ExplainedSelection {
                selection: sel,
                reasons,
            };
        }

        let candidate = sel.chosen.clone();
        let Some(prev) = self.previous.clone() else {
            self.previous = Some(candidate.clone());
            self.dwell = 1;
            reasons.push(DecisionReason::BaseChoice { chosen: candidate });
            return ExplainedSelection {
                selection: sel,
                reasons,
            };
        };

        // If previous is not even among the considered candidates, we must follow base choice.
        let scores = Self::scores_by_arm(&sel);
        let Some(prev_score) = scores.get(prev.as_str()).copied() else {
            self.previous = Some(candidate.clone());
            self.dwell = 1;
            reasons.push(DecisionReason::BaseChoice { chosen: candidate });
            return ExplainedSelection {
                selection: sel,
                reasons,
            };
        };

        // If base choice is the previous arm, just bump dwell.
        if candidate == prev {
            self.dwell = self.dwell.saturating_add(1);
            reasons.push(DecisionReason::BaseChoice { chosen: candidate });
            return ExplainedSelection {
                selection: sel,
                reasons,
            };
        }

        // Dwell gate.
        if self.cfg.min_dwell > 0 && self.dwell < self.cfg.min_dwell {
            reasons.push(DecisionReason::KeptPreviousDwell {
                previous: prev.clone(),
                candidate: candidate.clone(),
                dwell: self.dwell,
                min_dwell: self.cfg.min_dwell,
            });
            sel.chosen = prev.clone();
            self.dwell = self.dwell.saturating_add(1);
            return ExplainedSelection {
                selection: sel,
                reasons,
            };
        }

        // Margin gate.
        let cand_score = scores
            .get(candidate.as_str())
            .copied()
            .unwrap_or(f64::NEG_INFINITY);
        let margin = cand_score - prev_score;
        let min_margin = f64_or0(self.cfg.min_switch_margin);
        if min_margin > 0.0 && !(margin.is_finite() && margin >= min_margin) {
            reasons.push(DecisionReason::KeptPreviousMargin {
                previous: prev.clone(),
                candidate: candidate.clone(),
                previous_score: prev_score,
                candidate_score: cand_score,
                margin,
                min_margin,
            });
            sel.chosen = prev.clone();
            self.dwell = self.dwell.saturating_add(1);
            return ExplainedSelection {
                selection: sel,
                reasons,
            };
        }

        // Allow switch.
        reasons.push(DecisionReason::Switched {
            previous: prev.clone(),
            candidate: candidate.clone(),
            previous_score: prev_score,
            candidate_score: cand_score,
            margin,
        });
        self.previous = Some(candidate.clone());
        self.dwell = 1;
        ExplainedSelection {
            selection: sel,
            reasons,
        }
    }

    /// Apply stickiness using `select_mab_explain` output (recommended).
    ///
    /// This avoids heuristics for detecting explore-first and also provides callers access
    /// to constraint-fallback metadata via `decision.constraints_fallback_used`.
    pub fn apply_mab(&mut self, decision: MabSelectionDecision) -> ExplainedSelection {
        let mut sel = decision.selection;
        let mut reasons: Vec<DecisionReason> = Vec::new();

        if decision.explore_first {
            let chosen = sel.chosen.clone();
            self.previous = Some(chosen.clone());
            self.dwell = 1;
            reasons.push(DecisionReason::ExploreFirst { chosen });
            return ExplainedSelection {
                selection: sel,
                reasons,
            };
        }

        // Reuse the legacy logic for all other cases.
        let candidate = sel.chosen.clone();
        let Some(prev) = self.previous.clone() else {
            self.previous = Some(candidate.clone());
            self.dwell = 1;
            reasons.push(DecisionReason::BaseChoice { chosen: candidate });
            return ExplainedSelection {
                selection: sel,
                reasons,
            };
        };

        let scores = Self::scores_by_arm(&sel);
        let Some(prev_score) = scores.get(prev.as_str()).copied() else {
            self.previous = Some(candidate.clone());
            self.dwell = 1;
            reasons.push(DecisionReason::BaseChoice { chosen: candidate });
            return ExplainedSelection {
                selection: sel,
                reasons,
            };
        };

        if candidate == prev {
            self.dwell = self.dwell.saturating_add(1);
            reasons.push(DecisionReason::BaseChoice { chosen: candidate });
            return ExplainedSelection {
                selection: sel,
                reasons,
            };
        }

        if self.cfg.min_dwell > 0 && self.dwell < self.cfg.min_dwell {
            reasons.push(DecisionReason::KeptPreviousDwell {
                previous: prev.clone(),
                candidate: candidate.clone(),
                dwell: self.dwell,
                min_dwell: self.cfg.min_dwell,
            });
            sel.chosen = prev.clone();
            self.dwell = self.dwell.saturating_add(1);
            return ExplainedSelection {
                selection: sel,
                reasons,
            };
        }

        let cand_score = scores
            .get(candidate.as_str())
            .copied()
            .unwrap_or(f64::NEG_INFINITY);
        let margin = cand_score - prev_score;
        let min_margin = f64_or0(self.cfg.min_switch_margin);
        if min_margin > 0.0 && !(margin.is_finite() && margin >= min_margin) {
            reasons.push(DecisionReason::KeptPreviousMargin {
                previous: prev.clone(),
                candidate: candidate.clone(),
                previous_score: prev_score,
                candidate_score: cand_score,
                margin,
                min_margin,
            });
            sel.chosen = prev.clone();
            self.dwell = self.dwell.saturating_add(1);
            return ExplainedSelection {
                selection: sel,
                reasons,
            };
        }

        reasons.push(DecisionReason::Switched {
            previous: prev.clone(),
            candidate: candidate.clone(),
            previous_score: prev_score,
            candidate_score: cand_score,
            margin,
        });
        self.previous = Some(candidate.clone());
        self.dwell = 1;
        ExplainedSelection {
            selection: sel,
            reasons,
        }
    }

    /// Apply stickiness and return a unified `Decision` (recommended for logging/replay).
    ///
    /// This is the “end of the line” decision object: it includes constraint gating metadata
    /// and stickiness actions (kept previous / switched), so downstream systems can ingest a
    /// single format regardless of selection policy and wrappers.
    pub fn apply_mab_decide(&mut self, decision: MabSelectionDecision) -> Decision {
        let constraints = DecisionNote::Constraints {
            eligible_arms: decision.eligible_arms.clone(),
            fallback_used: decision.constraints_fallback_used,
        };

        // Explore-first bypasses stickiness gates (it seeds stickiness state).
        if decision.explore_first {
            return Decision {
                policy: DecisionPolicy::Mab,
                chosen: decision.selection.chosen.clone(),
                probs: None,
                notes: vec![constraints, DecisionNote::ExploreFirst],
            };
        }

        let explained = self.apply_mab(decision);
        let mut notes = vec![constraints, DecisionNote::DeterministicChoice];

        for r in &explained.reasons {
            match r {
                DecisionReason::KeptPreviousDwell {
                    previous,
                    candidate,
                    dwell,
                    min_dwell,
                } => notes.push(DecisionNote::StickyKeptPreviousDwell {
                    previous: previous.clone(),
                    candidate: candidate.clone(),
                    dwell: *dwell,
                    min_dwell: *min_dwell,
                }),
                DecisionReason::KeptPreviousMargin {
                    previous,
                    candidate,
                    previous_score,
                    candidate_score,
                    margin,
                    min_margin,
                } => notes.push(DecisionNote::StickyKeptPreviousMargin {
                    previous: previous.clone(),
                    candidate: candidate.clone(),
                    previous_score: *previous_score,
                    candidate_score: *candidate_score,
                    margin: *margin,
                    min_margin: *min_margin,
                }),
                DecisionReason::Switched {
                    previous,
                    candidate,
                    previous_score,
                    candidate_score,
                    margin,
                } => notes.push(DecisionNote::StickySwitched {
                    previous: previous.clone(),
                    candidate: candidate.clone(),
                    previous_score: *previous_score,
                    candidate_score: *candidate_score,
                    margin: *margin,
                }),
                // Don't copy-through these (too noisy; encoded by chosen + other notes).
                DecisionReason::ExploreFirst { .. } | DecisionReason::BaseChoice { .. } => {}
            }
        }

        Decision {
            policy: DecisionPolicy::Mab,
            chosen: explained.selection.chosen,
            probs: None,
            notes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CandidateDebug, MabConfig};

    fn mk_sel(previous: &str, candidate: &str) -> Selection {
        let cfg = MabConfig::default();
        Selection {
            chosen: candidate.to_string(),
            frontier: vec!["a".to_string(), "b".to_string()],
            candidates: vec![
                CandidateDebug {
                    name: "a".to_string(),
                    calls: 10,
                    ok: 0,
                    junk: 0,
                    hard_junk: 0,
                    ok_rate: 0.0,
                    junk_rate: 0.0,
                    hard_junk_rate: 0.0,
                    soft_junk_rate: 0.0,
                    mean_cost_units: 0.0,
                    mean_elapsed_ms: 0.0,
                    ucb: 0.0,
                    objective_success: if previous == "a" { 1.0 } else { 2.0 },
                },
                CandidateDebug {
                    name: "b".to_string(),
                    calls: 10,
                    ok: 0,
                    junk: 0,
                    hard_junk: 0,
                    ok_rate: 0.0,
                    junk_rate: 0.0,
                    hard_junk_rate: 0.0,
                    soft_junk_rate: 0.0,
                    mean_cost_units: 0.0,
                    mean_elapsed_ms: 0.0,
                    ucb: 0.0,
                    objective_success: if previous == "a" { 2.0 } else { 1.0 },
                },
            ],
            config: cfg,
        }
    }

    #[test]
    fn sticky_never_returns_arm_not_in_candidates() {
        let mut sticky = StickyMab::new(StickyConfig {
            min_dwell: 100,
            min_switch_margin: 100.0,
        });

        // Seed previous as "a".
        let _ = sticky.apply(mk_sel("a", "a"));

        // Now provide a selection that does not include "a" at all.
        let cfg = MabConfig::default();
        let sel = Selection {
            chosen: "x".to_string(),
            frontier: vec!["x".to_string()],
            candidates: vec![CandidateDebug {
                name: "x".to_string(),
                calls: 10,
                ok: 0,
                junk: 0,
                hard_junk: 0,
                ok_rate: 0.0,
                junk_rate: 0.0,
                hard_junk_rate: 0.0,
                soft_junk_rate: 0.0,
                mean_cost_units: 0.0,
                mean_elapsed_ms: 0.0,
                ucb: 0.0,
                objective_success: 0.0,
            }],
            config: cfg,
        };
        let out = sticky.apply(sel);
        assert_eq!(out.selection.chosen, "x");
    }
}
